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
from typing import Any, List, Optional, Tuple

import torch

from CoreVital.compound_signals import CompoundSignal, detect_compound_signals
from CoreVital.config import Config, load_model_profile
from CoreVital.early_warning import compute_early_warning
from CoreVital.fingerprint import compute_fingerprint_vector, compute_prompt_hash
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
from CoreVital.narrative import build_narrative
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
    RAGContext,
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
from CoreVital.risk import compute_layer_blame, compute_risk_score
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

        # Resolve model profile (per-model thresholds); used in timeline and health flags
        profile = self.config.model_profile or load_model_profile(results.model_bundle.architecture)

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
        # Returns (timeline, layers_by_step) for health flags; summary mode omits layers from storage
        with _op("_build_timeline"):
            timeline, timeline_layers_for_flags = self._build_timeline(results, profile)

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
                        profile=profile,
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
        # When capture_mode is summary/on_risk, use timeline_layers_for_flags (not stored in report)
        with _op("_build_health_flags"):
            health_flags = self._build_health_flags(
                results,
                timeline,
                timeline_layers_override=timeline_layers_for_flags,
                profile=profile,
            )

        # Assemble final Report (tracked as child of report_build)
        with _op("assemble Report"):
            report = Report(
                schema_version="0.4.0",
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

            # RAG context (Foundation F3): store in extensions when provided via CLI/API
            rag_dict = getattr(self.config, "rag_context", None)
            if rag_dict is not None:
                try:
                    report.extensions["rag"] = RAGContext(**rag_dict).model_dump()
                except Exception as e:
                    logger.warning(f"Invalid RAG context, skipping: {e}")

            # Phase-2: compound signals (Issue 6) — after timeline, before risk
            compound_signals: List[CompoundSignal] = []
            try:
                basin_scores: Optional[List[List[float]]] = None
                if prompt_analysis and prompt_analysis.layers:
                    basin_scores = [lyr.basin_scores for lyr in prompt_analysis.layers]
                compound_signals = detect_compound_signals(
                    timeline,
                    layers_by_step=timeline_layers_for_flags,
                    basin_scores=basin_scores,
                )
                report.extensions["compound_signals"] = [
                    {
                        "name": s.name,
                        "description": s.description,
                        "severity": s.severity,
                        "evidence": s.evidence,
                    }
                    for s in compound_signals
                ]
            except Exception as e:
                logger.warning(f"Compound signal detection failed, skipping: {e}")

            # Phase-2: risk score and layer blame (always computed when health_flags exist)
            risk_score = 0.0
            if health_flags is not None:
                try:
                    # New composite score from timeline + compound signals; fallback to legacy if timeline is None
                    risk_score, risk_factors = compute_risk_score(
                        health_flags,
                        summary,
                        timeline=timeline,
                        layers_by_step=timeline_layers_for_flags,
                        compound_signals=compound_signals if compound_signals else None,
                    )
                    blamed_layers = compute_layer_blame(timeline_layers_for_flags)
                    report.extensions["risk"] = {
                        "risk_score": risk_score,
                        "risk_factors": risk_factors,
                        "blamed_layers": blamed_layers,
                    }
                except Exception as e:
                    logger.warning(f"Risk computation failed, skipping: {e}")

            # F2.3 on_risk: when risk or any health flag triggers, attach full layer data to report
            capture_mode = getattr(self.config.capture, "capture_mode", "full")
            risk_threshold = getattr(self.config.capture, "risk_threshold", 0.7)
            any_health_flag_set = health_flags is not None and (
                health_flags.nan_detected
                or health_flags.inf_detected
                or health_flags.attention_collapse_detected
                or health_flags.repetition_loop_detected
                or health_flags.mid_layer_anomaly_detected
                or (health_flags.high_entropy_steps > 0)
            )
            if (
                capture_mode == "on_risk"
                and (risk_score >= risk_threshold or any_health_flag_set)
                and len(timeline_layers_for_flags) == len(report.timeline)
            ):
                # Attach full layers to timeline (already computed in timeline_layers_for_flags)
                report.timeline = [
                    step.model_copy(update={"layers": timeline_layers_for_flags[i]})
                    for i, step in enumerate(report.timeline)
                ]
                # Rebuild prompt_analysis with sparse heads included
                full_prompt_analysis = self._build_prompt_analysis(results, store_sparse_heads_override=True)
                if full_prompt_analysis is not None:
                    report.prompt_analysis = full_prompt_analysis
                logger.debug("on_risk triggered: full timeline layers and prompt_analysis attached")

            # Phase-3: fingerprint (run-summary vector + prompt hash) for every report
            try:
                hf = health_flags if health_flags is not None else HealthFlags()
                vec = compute_fingerprint_vector(timeline, summary, hf, risk_score)
                prompt_hash = compute_prompt_hash(prompt, model_info.hf_id)
                report.extensions["fingerprint"] = {"vector": vec, "prompt_hash": prompt_hash}
            except Exception as e:
                logger.warning(f"Fingerprint computation failed, skipping: {e}")

            # Phase-4: early warning (failure_risk, warning_signals) from timeline + health_flags
            try:
                hf_ew = health_flags if health_flags is not None else HealthFlags()
                failure_risk, warning_signals = compute_early_warning(timeline, hf_ew)
                report.extensions["early_warning"] = {
                    "failure_risk": failure_risk,
                    "warning_signals": warning_signals,
                }
            except Exception as e:
                logger.warning(f"Early warning computation failed, skipping: {e}")

            # Phase-2.3: data-specific narrative (Issue 8)
            try:
                hf_n = health_flags if health_flags is not None else HealthFlags()
                risk_ext = report.extensions.get("risk") or {}
                ew_ext = report.extensions.get("early_warning") or {}
                summary_text = build_narrative(
                    hf_n,
                    risk_ext.get("risk_score", 0.0),
                    risk_ext.get("risk_factors", []),
                    risk_ext.get("blamed_layers", []),
                    ew_ext.get("warning_signals", []),
                    timeline,
                    compound_signals=compound_signals if compound_signals else None,
                    summary=summary,
                )
                report.extensions["narrative"] = {"summary": summary_text}
            except Exception as e:
                logger.warning(f"Narrative build failed, skipping: {e}")

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

        # Derive quantization from actual model state, not config flags.
        # bundle.dtype_str is only set by hf_loader when quantization was
        # actually applied; it stays None on the CPU-fallback path.
        quantization_enabled = bundle.dtype_str is not None
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

    def _build_timeline(
        self,
        results: InstrumentationResults,
        profile: Optional[Any] = None,
    ) -> Tuple[List[TimelineStep], List[List[LayerSummary]]]:
        """Build timeline from instrumentation results.

        Tracks per-step timing for performance monitoring.
        When capture_mode is summary or on_risk, layer summaries are computed for health
        flags but not stored in the report (layers=[]); the second return value holds
        them for _build_health_flags.

        Supports both pre-computed StepSummary objects (Phase 1.3+) and legacy StepData.
        """
        import time

        from CoreVital.instrumentation.step_processor import StepSummary

        capture_mode = getattr(self.config.capture, "capture_mode", "full")
        store_layers = capture_mode == "full"

        timeline: List[TimelineStep] = []
        layers_by_step: List[List[LayerSummary]] = []
        step_times_ms: List[float] = []

        for step_data in results.timeline:
            step_start = time.perf_counter()

            token_info = TokenInfo(
                token_id=step_data.token_id,
                token_text=step_data.token_text,
                is_prompt_token=step_data.is_prompt_token,
            )

            if isinstance(step_data, StepSummary):
                # Pre-computed summaries from step_processor
                logits_summary = LogitsSummary()
                if step_data.logits_summary:
                    try:
                        logits_summary = LogitsSummary(**step_data.logits_summary)
                    except Exception as e:
                        logger.warning(f"Failed to build logits summary for step {step_data.step_index}: {e}")

                layers = self._build_layers_from_step_summary(step_data)
            else:
                # Legacy path: raw tensors (StepData)
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

                layers = []
                if step_data.hidden_states is not None:
                    layers = self._build_layer_summaries(
                        step_data.hidden_states,
                        step_data.attentions,
                        results.model_bundle.num_layers,
                        cross_attentions=step_data.cross_attentions,
                        profile=profile,
                    )

            layers_by_step.append(layers)

            timeline_step = TimelineStep(
                step_index=step_data.step_index,
                token=token_info,
                logits_summary=logits_summary,
                layers=layers if store_layers else [],
                extensions={},
            )

            timeline.append(timeline_step)
            step_times_ms.append((time.perf_counter() - step_start) * 1000)

        monitor = results.monitor
        if monitor and monitor.stack and step_times_ms:
            current_op = monitor.stack[-1]
            if current_op.operation_name == "_build_timeline":
                current_op.metadata["per_step"] = {
                    "count": len(step_times_ms),
                    "min_ms": min(step_times_ms),
                    "max_ms": max(step_times_ms),
                    "avg_ms": sum(step_times_ms) / len(step_times_ms),
                }

        return timeline, layers_by_step

    def _build_layers_from_step_summary(self, step_data: Any) -> List[LayerSummary]:
        """Build LayerSummary objects from pre-computed StepSummary.layer_summaries."""
        layers: List[LayerSummary] = []
        for layer_idx, ls in enumerate(step_data.layer_summaries):
            hidden_summary = HiddenSummary(**ls.hidden_summary) if ls.hidden_summary else HiddenSummary()
            attention_summary = (
                AttentionSummary(**ls.attention_summary) if ls.attention_summary else AttentionSummary()
            )
            cross_attention = None
            if ls.cross_attention_summary:
                cross_attention = AttentionSummary(**ls.cross_attention_summary)
            anomalies = TensorAnomalies(**ls.anomalies) if ls.anomalies else None
            layers.append(
                LayerSummary(
                    layer_index=layer_idx,
                    hidden_summary=hidden_summary,
                    attention_summary=attention_summary,
                    cross_attention=cross_attention,
                    anomalies=anomalies,
                    extensions={},
                )
            )
        return layers

    def _build_layer_summaries(
        self,
        hidden_states: Optional[List[torch.Tensor]],
        attentions: Optional[List[torch.Tensor]],
        num_layers: int,
        cross_attentions: Optional[List[torch.Tensor]] = None,
        profile: Optional[Any] = None,
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
                                    profile=profile,
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
                                    profile=profile,
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
        profile: Optional[Any] = None,
    ) -> Optional[List[LayerSummary]]:
        """
        Build encoder layer summaries from encoder hidden states and attentions.

        Args:
            encoder_hidden_states: List of encoder hidden state tensors (one per layer)
            encoder_attentions: List of encoder attention tensors (one per layer)
            num_layers: Number of layers
            profile: Optional model profile for attention thresholds

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
                                profile=profile,
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

    def _build_prompt_analysis(
        self,
        results: InstrumentationResults,
        store_sparse_heads_override: Optional[bool] = None,
    ) -> Optional[PromptAnalysis]:
        """Build PromptAnalysis from prompt forward pass data (Phase-1b).

        For CausalLM: uses data from model(input_ids) forward pass.
        For Seq2Seq: uses reused encoder outputs.
        Returns None if prompt telemetry is disabled or no data available.
        When capture_mode is summary or on_risk, sparse heads are omitted (heads=[]) to reduce payload.
        When store_sparse_heads_override is True (e.g. on_risk triggered), sparse heads are included.
        """
        pf = results.prompt_forward
        if pf is None:
            logger.debug("No prompt forward data — prompt_analysis will be null")
            return None

        capture_mode = getattr(self.config.capture, "capture_mode", "full")
        store_sparse_heads = (
            store_sparse_heads_override if store_sparse_heads_override is not None else (capture_mode == "full")
        )

        sparse_threshold = self.config.prompt_telemetry.sparse_threshold
        sparse_max_per_head = getattr(self.config.prompt_telemetry, "sparse_max_per_head", None)

        # 1. Build per-layer sparse attention profiles + basin scores (omit heads when summary/on_risk)
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

                    # Sparse extraction only when full capture (otherwise omit to reduce payload)
                    sparse_heads: list = []
                    if store_sparse_heads:
                        sparse_heads_dicts = extract_sparse_attention(
                            attn_tensor,
                            threshold=sparse_threshold,
                            max_per_head=sparse_max_per_head,
                        )
                        sparse_heads = [SparseAttentionHead(**h) for h in sparse_heads_dicts]

                    # Basin scores (middle/boundary ratio per head) — always computed
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
        # Uses manual CrossEntropyLoss; outputs.loss_per_token does not exist in HF CausalLMOutput.
        prompt_surprisals: list[float] = []
        if pf.logits is not None and pf.prompt_token_ids is not None:
            try:
                prompt_surprisals = compute_prompt_surprisal(pf.logits, pf.prompt_token_ids)
            except Exception as e:
                logger.warning(f"Failed to compute prompt surprisal: {e}")
            # Discard raw logits after use — not persisted to report JSON; frees memory.
            pf.logits = None

        return PromptAnalysis(
            layers=layers,
            layer_transformations=layer_transformations,
            prompt_surprisals=prompt_surprisals,
        )

    def _build_health_flags(
        self,
        results: InstrumentationResults,
        timeline: List[TimelineStep],
        timeline_layers_override: Optional[List[List[LayerSummary]]] = None,
        profile: Optional[Any] = None,
    ) -> HealthFlags:
        """Build aggregated health flags from timeline data (Phase-1c).

        Implements the transient buffer lifecycle:
        1. ALLOCATE: Build buffer of last-layer hidden state vectors from last 5 generated steps
        2. CONSUME: Run repetition loop detection (cosine similarity on buffer)
        3. KILL: Explicitly delete buffer and free GPU memory

        Other health flags are aggregated from layer data. When capture_mode is summary/on_risk,
        timeline[].layers is empty; pass timeline_layers_override (computed but not stored).
        """
        # --- Transient buffer: allocate ---
        # Extract last-layer hidden state vector (last token) from generated steps.
        # StepSummary carries a pre-extracted _last_layer_hidden_vec;
        # legacy StepData falls back to extracting from raw hidden_states.
        from CoreVital.instrumentation.step_processor import StepSummary as _StepSummary

        generated_steps = [s for s in results.timeline if not s.is_prompt_token]
        buffer_steps = generated_steps[-5:]  # FIFO capacity 5

        hidden_state_buffer: list[torch.Tensor] = []
        for step in buffer_steps:
            if isinstance(step, _StepSummary):
                if step._last_layer_hidden_vec is not None:
                    hidden_state_buffer.append(step._last_layer_hidden_vec)
            elif hasattr(step, "hidden_states") and step.hidden_states is not None and len(step.hidden_states) > 0:
                try:
                    last_layer = step.hidden_states[-1]
                    if isinstance(last_layer, torch.Tensor) and last_layer.dim() >= 2:
                        vec = last_layer[0, -1, :].detach().cpu()
                        hidden_state_buffer.append(vec)
                    elif isinstance(last_layer, torch.Tensor):
                        hidden_state_buffer.append(last_layer.detach().cpu().flatten())
                except Exception as e:
                    logger.debug(f"Failed to extract hidden state for buffer: {e}")

        logger.debug(f"Transient buffer: {len(hidden_state_buffer)} vectors from last {len(buffer_steps)} steps")

        # --- Consume: repetition loop detection ---
        repetition_loop_detected = False
        try:
            repetition_loop_detected = detect_repetition_loop(hidden_state_buffer, profile=profile)
        except Exception as e:
            logger.warning(f"Failed repetition loop detection: {e}")

        # --- Kill: explicit buffer teardown ---
        del hidden_state_buffer
        del buffer_steps

        # --- Aggregate from layer data (use override when capture_mode is summary/on_risk) ---
        layers_to_aggregate = (
            timeline_layers_override
            if timeline_layers_override is not None
            else [tl_step.layers for tl_step in timeline]
        )

        nan_detected = False
        inf_detected = False
        attention_collapse_detected = False
        high_entropy_steps = 0

        high_entropy_threshold = 4.0
        if profile is not None and hasattr(profile, "high_entropy_threshold_bits"):
            high_entropy_threshold = float(profile.high_entropy_threshold_bits)
        for tl_step in timeline:
            # High entropy check (per-step logits entropy)
            # Threshold from profile or 4.0 bits (model-agnostic; 2^4 = 16 equally likely tokens).
            if tl_step.logits_summary and tl_step.logits_summary.entropy is not None:
                if tl_step.logits_summary.entropy > high_entropy_threshold:
                    high_entropy_steps += 1

        for step_layers in layers_to_aggregate:
            for layer in step_layers:
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
        if layers_to_aggregate and num_layers >= 3:
            try:
                mid_layer_anomaly_detected = detect_mid_layer_anomaly(layers_to_aggregate, num_layers, profile=profile)
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

    assert report.schema_version == "0.4.0"
    assert len(report.timeline) > 0

    print("✓ All report builder tests passed!\n")


if __name__ == "__main__":
    _test_report_builder()
