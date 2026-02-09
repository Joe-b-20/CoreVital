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
# ============================================================================

import uuid
from typing import List, Optional

import torch

from CoreVital.config import Config
from CoreVital.instrumentation.collector import InstrumentationResults
from CoreVital.instrumentation.summaries import (
    compute_attention_summary,
    compute_hidden_summary,
    compute_logits_summary,
)
from CoreVital.logging_utils import get_logger
from CoreVital.reporting.schema import (
    AttentionConfig,
    AttentionSummary,
    GeneratedInfo,
    GenerationConfig,
    HiddenConfig,
    HiddenSummary,
    LayerSummary,
    LogitsConfig,
    LogitsSummary,
    ModelInfo,
    PromptInfo,
    QuantizationInfo,
    Report,
    RunConfig,
    SinkConfig,
    SketchConfig,
    SummariesConfig,
    Summary,
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

        # Assemble final Report (tracked as child of report_build)
        with _op("assemble Report"):
            report = Report(
                schema_version="0.2.0",
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

                    layer_summary = LayerSummary(
                        layer_index=layer_idx,
                        hidden_summary=hidden_summary,
                        attention_summary=attention_summary,
                        cross_attention=cross_attention_summary,
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

    assert report.schema_version == "0.2.0"
    assert len(report.timeline) > 0

    print("✓ All report builder tests passed!\n")


if __name__ == "__main__":
    _test_report_builder()
