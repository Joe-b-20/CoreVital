# ============================================================================
# CoreVital - Report Builder
#
# Purpose: Build Report objects from InstrumentationResults
# Inputs: InstrumentationResults, Config
# Outputs: Complete Report object
# Dependencies: reporting.schema, instrumentation, summaries, utils
# Usage: builder = ReportBuilder(config); report = builder.build(results, prompt)
#
# Changelog:
#   2026-01-13: Initial report builder for Phase-0
#   2026-01-14: Fixed model revision extraction - use revision from ModelBundle if available
#                Improved attention summary error handling with better logging
#                Added null checks for attention tensor processing
#   2026-01-15: Added quantization info extraction from config to report model.quantization field
#   2026-01-15: Added Seq2Seq report building - compute encoder_hidden_states summaries,
#                include encoder_attention and cross_attention in layer summaries
#   2026-01-21: Phase-0.5 hardening - removed encoder_attention by index from decoder layers,
#                build proper encoder_layers from encoder pass; added extensions to all layers;
#                standardized logging to DEBUG level
#   2026-02-04: Phase-0.75 - integrated PerformanceMonitor for child operation timing
#                within report_build; per-step timing tracked in _build_timeline
#   2026-02-07: Pre-phase-1 cleanup:
#                - Removed encoder_hidden_states (deprecated) from report assembly
#                - Removed encoder_attention=None from layer summaries
#                - Use ModelBundle.dtype_str for quantized models
#                - Schema version 0.1.0 → 0.2.0
#   2026-02-07: Phase-1a — schema v0.3.0:
#                - Wire new logit metrics (top_k_margin, voter_agreement, perplexity, surprisal)
#                - Wire enhanced attention aggregation (entropy_max, concentration_min, counts)
#                - Wire NaN/Inf detection (TensorAnomalies) per layer
#                - Report now includes prompt_analysis (None until Phase-1b) and
#                  health_flags (None until Phase-1c)
#   2026-02-10: Phase-1b — Prompt telemetry wiring:
#                - _build_prompt_analysis() builds PromptAnalysis from PromptForwardData
#                - Sparse attention extraction + basin scores per layer
#                - Layer transformations (cosine similarity between consecutive layers)
#                - Prompt surprisal (CrossEntropyLoss on prompt logits, CausalLM only)
#   2026-02-10: Phase-1c — Health flags + transient buffer:
#                - _build_health_flags() aggregates all signals into HealthFlags
#                - Transient buffer: last-layer hidden states from last 5 generated steps
#                - Repetition loop detection (cosine similarity > 0.9995 on buffer)
#                  Threshold raised from 0.99 to 0.9995 to handle float16 anisotropy
#                - Mid-layer anomaly detection: per-step dynamic 8× L2 baseline + NaN/Inf
#                  Attention collapse excluded (structural, not runtime anomaly)
#                  Per-step baselines (not global) to handle step-0 prompt processing scale
#                - Aggregate NaN/Inf, attention collapse, high entropy (>4.0) step count
#                - Buffer lifecycle: allocate → consume → kill (never serialized)
#   2026-02-10: Phase-1c fixes (CI/Codex review):
#                - MyPy fix: renamed loop variable to avoid StepData/TimelineStep type conflict
#                - Cross-attention NaN/Inf: pass cross_attentions[layer_idx] to
#                  detect_tensor_anomalies() so Seq2Seq anomalies are fully checked
# ============================================================================

import uuid
from typing import List, Optional

import torch

from CoreVital.config import Config
from CoreVital.instrumentation.collector import InstrumentationResults
from CoreVital.instrumentation.summaries import (
    compute_attention_summary,
    compute_basin_scores,
    compute_hidden_summary,
    compute_layer_transformations,
    compute_logits_summary,
    compute_prompt_surprisal,
    detect_mid_layer_anomaly,
    detect_repetition_loop,
    detect_tensor_anomalies,
    extract_sparse_attention,
)
from CoreVital.logging_utils import get_logger
from CoreVital.reporting.schema import (
    AttentionConfig,
    AttentionSummary,
    GeneratedInfo,
    GenerationConfig,
    HealthFlags,
    HiddenConfig,
    HiddenSummary,
    LayerSummary,
    LogitsConfig,
    LogitsSummary,
    ModelInfo,
    PromptAnalysis,
    PromptAttentionLayer,
    PromptInfo,
    QuantizationInfo,
    Report,
    RunConfig,
    SinkConfig,
    SketchConfig,
    SparseAttentionHead,
    SummariesConfig,
    Summary,
    TensorAnomalies,
    TimelineStep,
    TokenInfo,
    Warning,
)
from CoreVital.utils.time import get_utc_timestamp

logger = get_logger(__name__)


class ReportBuilder:
    """
    Build structured Report objects from instrumentation results.
    """

    def __init__(self, config: Config):
        """
        Initialize builder with configuration.

        Args:
            config: Configuration object
        """
        self.config = config

    def build(self, results: InstrumentationResults, prompt: str) -> Report:
        """
        Build a complete Report from instrumentation results.

        Args:
            results: InstrumentationResults from collector
            prompt: Original prompt text

        Returns:
            Complete Report object
        """
        from contextlib import nullcontext

        logger.debug("Building report...")

        # Performance monitor for nested tracking
        monitor = results.monitor

        def _op(name: str):
            """Helper to wrap operations in monitor.operation() if enabled."""
            return monitor.operation(name) if monitor else nullcontext()

        # Generate trace ID
        trace_id = str(uuid.uuid4())
        created_at = get_utc_timestamp()

        # Build model info (tracked as child of report_build - corevital logic)
        with _op("_build_model_info"):
            model_info = self._build_model_info(results)

        # Build run config
        run_config = self._build_run_config(trace_id)

        # Build prompt info
        prompt_info = PromptInfo(
            text=prompt,
            token_ids=results.prompt_token_ids,
            num_tokens=len(results.prompt_token_ids),
        )

        # Build generated info
        generated_info = GeneratedInfo(
            output_text=results.generated_text,
            token_ids=results.generated_token_ids,
            num_tokens=len(results.generated_token_ids),
        )

        # Build timeline (tracked as child of report_build - corevital logic)
        with _op("_build_timeline"):
            timeline = self._build_timeline(results)

        # Build encoder layers (for Seq2Seq models)
        # encoder_layers represents the encoder pass computed ONCE
        # (Tracked as child of report_build)
        with _op("_build_encoder_layers"):
            encoder_layers = None
            if results.encoder_hidden_states is not None or results.encoder_attentions is not None:
                try:
                    encoder_layers = self._build_encoder_layers(
                        results.encoder_hidden_states,
                        results.encoder_attentions,
                        results.model_bundle.num_layers,
                    )
                    if encoder_layers:
                        logger.debug(f"Computed {len(encoder_layers)} encoder layer summaries")
                except Exception as e:
                    logger.warning(f"Failed to compute encoder layers: {e}")

        # Build summary (tracked as child of report_build)
        with _op("build Summary"):
            summary = Summary(
                prompt_tokens=len(results.prompt_token_ids),
                generated_tokens=len(results.generated_token_ids),
                total_steps=len(results.prompt_token_ids) + len(results.generated_token_ids),
                elapsed_ms=results.elapsed_ms,
            )

        # Convert warnings (tracked as child of report_build)
        with _op("convert warnings"):
            warnings = [Warning(code=w["code"], message=w["message"]) for w in results.warnings]

        # Build prompt analysis (Phase-1b)
        with _op("_build_prompt_analysis"):
            prompt_analysis = self._build_prompt_analysis(results)

        # Build health flags (Phase-1c)
        # Transient buffer lifecycle: allocate → consume → kill, all inside this method
        with _op("_build_health_flags"):
            health_flags = self._build_health_flags(results, timeline)

        # Assemble final Report (tracked as child of report_build)
        with _op("assemble Report"):
            report = Report(
                schema_version="0.3.0",
                trace_id=trace_id,
                created_at_utc=created_at,
                model=model_info,
                run_config=run_config,
                prompt=prompt_info,
                generated=generated_info,
                timeline=timeline,
                summary=summary,
                warnings=warnings,
                encoder_layers=encoder_layers,
                prompt_analysis=prompt_analysis,
                health_flags=health_flags,
            )

        # Note: Performance extensions are injected by CLI into report.extensions before sink.write()

        logger.debug(f"Report built: {len(timeline)} timeline steps")
        return report

    def _build_model_info(self, results: InstrumentationResults) -> ModelInfo:
        """Build ModelInfo from results."""
        bundle = results.model_bundle

        # Use explicit dtype_str if set (e.g. "quantized_unknown"), otherwise derive from torch dtype
        dtype_str = bundle.dtype_str if bundle.dtype_str is not None else str(bundle.dtype).replace("torch.", "")

        # Use revision from bundle if available, otherwise fall back to config
        revision = bundle.revision if bundle.revision else self.config.model.revision

        # Build quantization info from config
        quantization_enabled = self.config.model.load_in_4bit or self.config.model.load_in_8bit
        quantization_method = None
        if quantization_enabled:
            if self.config.model.load_in_4bit:
                quantization_method = "4-bit"
            elif self.config.model.load_in_8bit:
                quantization_method = "8-bit"

        quantization_info = QuantizationInfo(
            enabled=quantization_enabled,
            method=quantization_method,
        )

        return ModelInfo(
            hf_id=self.config.model.hf_id,
            revision=revision,
            architecture=bundle.architecture,
            num_layers=bundle.num_layers,
            hidden_size=bundle.hidden_size,
            num_attention_heads=bundle.num_attention_heads,
            tokenizer_hf_id=self.config.model.hf_id,
            dtype=dtype_str,
            device=str(bundle.device),
            quantization=quantization_info,
        )

    def _build_run_config(self, trace_id: str) -> RunConfig:
        """Build RunConfig from configuration."""
        return RunConfig(
            seed=self.config.generation.seed,
            device_requested=self.config.device.requested,
            max_new_tokens=self.config.generation.max_new_tokens,
            generation=GenerationConfig(
                do_sample=self.config.generation.do_sample,
                temperature=self.config.generation.temperature,
                top_k=self.config.generation.top_k,
                top_p=self.config.generation.top_p,
            ),
            summaries=SummariesConfig(
                hidden=HiddenConfig(
                    enabled=self.config.summaries.hidden.enabled,
                    stats=self.config.summaries.hidden.stats,
                    sketch=SketchConfig(
                        method=self.config.summaries.hidden.sketch.method,
                        dim=self.config.summaries.hidden.sketch.dim,
                        seed=self.config.summaries.hidden.sketch.seed,
                    ),
                ),
                attention=AttentionConfig(
                    enabled=self.config.summaries.attention.enabled,
                    stats=self.config.summaries.attention.stats,
                ),
                logits=LogitsConfig(
                    enabled=self.config.summaries.logits.enabled,
                    stats=self.config.summaries.logits.stats,
                    topk=self.config.summaries.logits.topk,
                ),
            ),
            sink=SinkConfig(
                type=self.config.sink.type,
                target=f"{self.config.sink.output_dir}/trace_{trace_id[:8]}.json",
            ),
        )

    def _build_timeline(self, results: InstrumentationResults) -> List[TimelineStep]:
        """Build timeline from instrumentation results.

        Tracks per-step timing for performance monitoring.
        """
        import time

        timeline = []
        step_times_ms: List[float] = []

        for step_data in results.timeline:
            step_start = time.perf_counter()

            # Token info
            token_info = TokenInfo(
                token_id=step_data.token_id,
                token_text=step_data.token_text,
                is_prompt_token=step_data.is_prompt_token,
            )

            # Logits summary
            logits_summary = LogitsSummary()
            if step_data.logits is not None:
                try:
                    logits_dict = compute_logits_summary(
                        step_data.logits,
                        results.model_bundle.tokenizer,
                        self.config.summaries.logits,
                        actual_token_id=step_data.token_id,
                    )
                    logits_summary = LogitsSummary(**logits_dict)
                except Exception as e:
                    logger.warning(f"Failed to compute logits summary for step {step_data.step_index}: {e}")

            # Layer summaries
            layers = []
            if step_data.hidden_states is not None:
                layers = self._build_layer_summaries(
                    step_data.hidden_states,
                    step_data.attentions,
                    results.model_bundle.num_layers,
                    cross_attentions=step_data.cross_attentions,
                )

            timeline_step = TimelineStep(
                step_index=step_data.step_index,
                token=token_info,
                logits_summary=logits_summary,
                layers=layers,
                extensions={},
            )

            timeline.append(timeline_step)
            step_times_ms.append((time.perf_counter() - step_start) * 1000)

        # Store per_step stats in the monitor's current operation (_build_timeline itself)
        monitor = results.monitor
        if monitor and monitor.stack and step_times_ms:
            # The current operation on stack IS _build_timeline
            current_op = monitor.stack[-1]
            if current_op.operation_name == "_build_timeline":
                current_op.metadata["per_step"] = {
                    "count": len(step_times_ms),
                    "min_ms": min(step_times_ms),
                    "max_ms": max(step_times_ms),
                    "avg_ms": sum(step_times_ms) / len(step_times_ms),
                }

        return timeline

    def _build_layer_summaries(
        self,
        hidden_states: Optional[List[torch.Tensor]],
        attentions: Optional[List[torch.Tensor]],
        num_layers: int,
        cross_attentions: Optional[List[torch.Tensor]] = None,
    ) -> List[LayerSummary]:
        """Build layer summaries from hidden states and attentions."""
        layers = []

        # Handle different output formats from transformers
        # hidden_states can be tuple of tensors, one per layer
        if hidden_states is not None:
            for layer_idx in range(min(len(hidden_states), num_layers)):
                try:
                    hidden_tensor = hidden_states[layer_idx]

                    # Compute hidden summary
                    hidden_dict = compute_hidden_summary(
                        hidden_tensor,
                        self.config.summaries.hidden,
                    )
                    hidden_summary = HiddenSummary(**hidden_dict)

                    # Compute decoder self-attention summary
                    attention_summary = AttentionSummary()
                    if attentions is not None and layer_idx < len(attentions):
                        try:
                            attn_tensor = attentions[layer_idx]
                            if attn_tensor is not None:
                                attn_dict = compute_attention_summary(
                                    attn_tensor,
                                    self.config.summaries.attention,
                                )
                                if attn_dict:  # Only update if we got a non-empty dict
                                    attention_summary = AttentionSummary(**attn_dict)
                                else:
                                    logger.debug(f"Empty attention summary dict for layer {layer_idx}")
                            else:
                                logger.debug(f"Attention tensor is None for layer {layer_idx}")
                        except Exception as e:
                            logger.warning(f"Failed to compute attention summary for layer {layer_idx}: {e}")
                            import traceback

                            logger.debug(traceback.format_exc())

                    # Compute cross-attention summary (decoder attending to encoder)
                    cross_attention_summary = None
                    if cross_attentions is not None and layer_idx < len(cross_attentions):
                        try:
                            cross_attn_tensor = cross_attentions[layer_idx]
                            if cross_attn_tensor is not None:
                                cross_attn_dict = compute_attention_summary(
                                    cross_attn_tensor,
                                    self.config.summaries.attention,
                                )
                                if cross_attn_dict:
                                    cross_attention_summary = AttentionSummary(**cross_attn_dict)
                        except Exception as e:
                            logger.debug(f"Failed to compute cross-attention summary for layer {layer_idx}: {e}")

                    # Compute NaN/Inf anomalies for this layer (hidden + self-attn + cross-attn)
                    anomalies = None
                    try:
                        attn_tensor_for_check = None
                        if attentions is not None and layer_idx < len(attentions):
                            attn_tensor_for_check = attentions[layer_idx]
                        cross_attn_for_check = None
                        if cross_attentions is not None and layer_idx < len(cross_attentions):
                            cross_attn_for_check = cross_attentions[layer_idx]
                        anomaly_dict = detect_tensor_anomalies(
                            hidden_tensor, attn_tensor_for_check, cross_attn_for_check
                        )
                        if anomaly_dict:
                            anomalies = TensorAnomalies(**anomaly_dict)
                    except Exception as e:
                        logger.debug(f"Failed to detect anomalies for layer {layer_idx}: {e}")

                    layer_summary = LayerSummary(
                        layer_index=layer_idx,
                        hidden_summary=hidden_summary,
                        attention_summary=attention_summary,
                        cross_attention=cross_attention_summary,
                        anomalies=anomalies,
                        extensions={},
                    )
                    layers.append(layer_summary)

                except Exception as e:
                    logger.warning(f"Failed to process layer {layer_idx}: {e}")

        return layers

    def _build_encoder_layers(
        self,
        encoder_hidden_states: Optional[List[torch.Tensor]],
        encoder_attentions: Optional[List[torch.Tensor]],
        num_layers: int,
    ) -> Optional[List[LayerSummary]]:
        """
        Build encoder layer summaries from encoder hidden states and attentions.

        Args:
            encoder_hidden_states: List of encoder hidden state tensors (one per layer)
            encoder_attentions: List of encoder attention tensors (one per layer)
            num_layers: Number of layers

        Returns:
            List of LayerSummary objects for encoder layers, or None if no encoder data
        """
        if encoder_hidden_states is None and encoder_attentions is None:
            return None

        encoder_layers = []
        max_layers = num_layers

        # Determine number of layers from available data
        if encoder_hidden_states is not None and len(encoder_hidden_states) > 0:
            max_layers = min(max_layers, len(encoder_hidden_states))
        if encoder_attentions is not None and len(encoder_attentions) > 0:
            max_layers = min(max_layers, len(encoder_attentions))

        # If we have no valid layers, return None
        if max_layers == 0:
            logger.debug("No encoder layers to process (empty encoder_hidden_states and encoder_attentions)")
            return None

        for layer_idx in range(max_layers):
            try:
                # Compute hidden summary
                hidden_summary = HiddenSummary()
                if encoder_hidden_states is not None and layer_idx < len(encoder_hidden_states):
                    try:
                        hidden_tensor = encoder_hidden_states[layer_idx]
                        if hidden_tensor is not None:
                            hidden_dict = compute_hidden_summary(
                                hidden_tensor,
                                self.config.summaries.hidden,
                            )
                            hidden_summary = HiddenSummary(**hidden_dict)
                    except Exception as e:
                        logger.debug(f"Failed to compute encoder hidden summary for layer {layer_idx}: {e}")

                # Compute encoder attention summary
                encoder_attention_summary = None
                if encoder_attentions is not None and layer_idx < len(encoder_attentions):
                    try:
                        enc_attn_tensor = encoder_attentions[layer_idx]
                        if enc_attn_tensor is not None:
                            # Handle tuple structure if present
                            if isinstance(enc_attn_tensor, (tuple, list)) and len(enc_attn_tensor) > 0:
                                enc_attn_tensor = enc_attn_tensor[0]
                            enc_attn_dict = compute_attention_summary(
                                enc_attn_tensor,
                                self.config.summaries.attention,
                            )
                            if enc_attn_dict:
                                encoder_attention_summary = AttentionSummary(**enc_attn_dict)
                    except Exception as e:
                        logger.debug(f"Failed to compute encoder attention summary for layer {layer_idx}: {e}")

                layer_summary = LayerSummary(
                    layer_index=layer_idx,
                    hidden_summary=hidden_summary,
                    attention_summary=encoder_attention_summary if encoder_attention_summary else AttentionSummary(),
                    cross_attention=None,  # Not applicable for encoder layers
                    extensions={},
                )
                encoder_layers.append(layer_summary)

            except Exception as e:
                logger.warning(f"Failed to process encoder layer {layer_idx}: {e}")

        return encoder_layers if encoder_layers else None

    def _build_prompt_analysis(self, results: InstrumentationResults) -> Optional[PromptAnalysis]:
        """Build PromptAnalysis from prompt forward pass data (Phase-1b).

        For CausalLM: uses data from model(input_ids) forward pass.
        For Seq2Seq: uses reused encoder outputs.
        Returns None if prompt telemetry is disabled or no data available.
        """
        pf = results.prompt_forward
        if pf is None:
            logger.debug("No prompt forward data — prompt_analysis will be null")
            return None

        sparse_threshold = self.config.prompt_telemetry.sparse_threshold

        # 1. Build per-layer sparse attention profiles + basin scores
        layers: list[PromptAttentionLayer] = []
        if pf.attentions is not None:
            for layer_idx, attn_tensor in enumerate(pf.attentions):
                try:
                    # Handle tuple wrapping (some models wrap layer attentions in tuples)
                    if isinstance(attn_tensor, (tuple, list)) and len(attn_tensor) > 0:
                        attn_tensor = attn_tensor[0]
                    if attn_tensor is None or not isinstance(attn_tensor, torch.Tensor):
                        layers.append(PromptAttentionLayer())
                        continue

                    # Sparse extraction (vectorized torch.where per head)
                    sparse_heads_dicts = extract_sparse_attention(attn_tensor, threshold=sparse_threshold)
                    sparse_heads = [SparseAttentionHead(**h) for h in sparse_heads_dicts]

                    # Basin scores (middle/boundary ratio per head)
                    basin = compute_basin_scores(attn_tensor)

                    layers.append(PromptAttentionLayer(heads=sparse_heads, basin_scores=basin))
                except Exception as e:
                    logger.warning(f"Failed to build prompt attention for layer {layer_idx}: {e}")
                    layers.append(PromptAttentionLayer())

        # 2. Layer transformations (cosine similarity between consecutive layers)
        layer_transformations: list[float] = []
        if pf.hidden_states is not None and len(pf.hidden_states) >= 2:
            try:
                layer_transformations = compute_layer_transformations(pf.hidden_states)
            except Exception as e:
                logger.warning(f"Failed to compute layer transformations: {e}")

        # 3. Prompt surprisal (CausalLM only — encoder logits not applicable)
        prompt_surprisals: list[float] = []
        if pf.logits is not None and pf.prompt_token_ids is not None:
            try:
                prompt_surprisals = compute_prompt_surprisal(pf.logits, pf.prompt_token_ids)
            except Exception as e:
                logger.warning(f"Failed to compute prompt surprisal: {e}")

        return PromptAnalysis(
            layers=layers,
            layer_transformations=layer_transformations,
            prompt_surprisals=prompt_surprisals,
        )

    def _build_health_flags(
        self,
        results: InstrumentationResults,
        timeline: List[TimelineStep],
    ) -> HealthFlags:
        """Build aggregated health flags from timeline data (Phase-1c).

        Implements the transient buffer lifecycle:
        1. ALLOCATE: Build buffer of last-layer hidden state vectors from last 5 generated steps
        2. CONSUME: Run repetition loop detection (cosine similarity on buffer)
        3. KILL: Explicitly delete buffer and free GPU memory

        Other health flags are aggregated from the already-built timeline (LayerSummary objects).
        The buffer never touches the schema — only the boolean result is stored.
        """
        # --- Transient buffer: allocate ---
        # Extract last-layer hidden state vector (last token) from generated steps
        generated_steps = [s for s in results.timeline if not s.is_prompt_token]
        buffer_steps = generated_steps[-5:]  # FIFO capacity 5

        hidden_state_buffer: list[torch.Tensor] = []
        for step in buffer_steps:
            if step.hidden_states is not None and len(step.hidden_states) > 0:
                try:
                    last_layer = step.hidden_states[-1]  # (batch, seq_len, hidden_dim)
                    if isinstance(last_layer, torch.Tensor) and last_layer.dim() >= 2:
                        # Last token of batch 0
                        vec = last_layer[0, -1, :].detach().cpu()  # (hidden_dim,)
                        hidden_state_buffer.append(vec)
                    elif isinstance(last_layer, torch.Tensor):
                        hidden_state_buffer.append(last_layer.detach().cpu().flatten())
                except Exception as e:
                    logger.debug(f"Failed to extract hidden state for buffer: {e}")

        logger.debug(f"Transient buffer: {len(hidden_state_buffer)} vectors from last {len(buffer_steps)} steps")

        # --- Consume: repetition loop detection ---
        repetition_loop_detected = False
        try:
            repetition_loop_detected = detect_repetition_loop(hidden_state_buffer)
        except Exception as e:
            logger.warning(f"Failed repetition loop detection: {e}")

        # --- Kill: explicit buffer teardown ---
        del hidden_state_buffer
        del buffer_steps
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- Aggregate from built timeline ---
        nan_detected = False
        inf_detected = False
        attention_collapse_detected = False
        high_entropy_steps = 0

        for tl_step in timeline:
            # High entropy check (per-step logits entropy)
            if tl_step.logits_summary and tl_step.logits_summary.entropy is not None:
                if tl_step.logits_summary.entropy > 4.0:
                    high_entropy_steps += 1

            # Per-layer checks
            for layer in tl_step.layers:
                # NaN/Inf from TensorAnomalies
                if layer.anomalies is not None:
                    if layer.anomalies.has_nan:
                        nan_detected = True
                    if layer.anomalies.has_inf:
                        inf_detected = True

                # Attention collapse
                if layer.attention_summary is not None:
                    if layer.attention_summary.collapsed_head_count > 0:
                        attention_collapse_detected = True

        # --- Mid-layer anomaly detection ---
        mid_layer_anomaly_detected = False
        num_layers = results.model_bundle.num_layers
        if timeline and num_layers >= 3:
            try:
                # Pass list of step-layers (each is a List[LayerSummary])
                timeline_layers = [tl_step.layers for tl_step in timeline]
                mid_layer_anomaly_detected = detect_mid_layer_anomaly(timeline_layers, num_layers)
            except Exception as e:
                logger.warning(f"Failed mid-layer anomaly detection: {e}")

        flags = HealthFlags(
            nan_detected=nan_detected,
            inf_detected=inf_detected,
            attention_collapse_detected=attention_collapse_detected,
            high_entropy_steps=high_entropy_steps,
            repetition_loop_detected=repetition_loop_detected,
            mid_layer_anomaly_detected=mid_layer_anomaly_detected,
        )

        logger.debug(
            f"Health flags: nan={nan_detected}, inf={inf_detected}, "
            f"collapse={attention_collapse_detected}, high_entropy={high_entropy_steps}, "
            f"repetition={repetition_loop_detected}, mid_layer_anomaly={mid_layer_anomaly_detected}"
        )
        return flags


# ============================================================================
# Test Harness
# ============================================================================


def _test_report_builder():
    """Test harness for report builder."""
    print("Testing ReportBuilder...")

    from CoreVital.config import Config
    from CoreVital.instrumentation.collector import InstrumentationCollector

    config = Config()
    config.model.hf_id = "gpt2"
    config.device.requested = "cpu"
    config.generation.max_new_tokens = 3

    collector = InstrumentationCollector(config)
    results = collector.run("Hi")

    builder = ReportBuilder(config)
    report = builder.build(results, "Hi")

    print("✓ Report built:")
    print(f"  Trace ID: {report.trace_id}")
    print(f"  Model: {report.model.hf_id}")
    print(f"  Timeline steps: {len(report.timeline)}")
    print(f"  Warnings: {len(report.warnings)}")

    assert report.schema_version == "0.3.0"
    assert len(report.timeline) > 0

    print("✓ All report builder tests passed!\n")


if __name__ == "__main__":
    _test_report_builder()
