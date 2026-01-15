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
# ============================================================================

import uuid
from typing import List, Optional
import torch

from CoreVital.config import Config
from CoreVital.instrumentation.collector import InstrumentationResults, StepData
from CoreVital.instrumentation.summaries import (
    compute_hidden_summary,
    compute_attention_summary,
    compute_logits_summary,
    compute_encoder_hidden_states_summaries,
)
from CoreVital.reporting.schema import (
    GeneratedInfo,
    Report,
    ModelInfo,
    QuantizationInfo,
    RunConfig,
    GenerationConfig,
    SketchConfig,
    HiddenConfig,
    AttentionConfig,
    LogitsConfig,
    SummariesConfig,
    SinkConfig,
    PromptInfo,

    TimelineStep,
    TokenInfo,
    LogitsSummary,
    LayerSummary,
    HiddenSummary,
    AttentionSummary,
    Summary,
    Warning,
    TopKItem,
)
from CoreVital.utils.time import get_utc_timestamp
from CoreVital.logging_utils import get_logger


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
        logger.info("Building report...")
        
        # Generate trace ID
        trace_id = str(uuid.uuid4())
        created_at = get_utc_timestamp()
        
        # Build model info
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
        
        # Build timeline
        timeline = self._build_timeline(results)
        
        # Build encoder hidden states summaries (for Seq2Seq models)
        encoder_hidden_states_summaries = None
        if results.encoder_hidden_states is not None:
            try:
                encoder_summaries_dicts = compute_encoder_hidden_states_summaries(
                    results.encoder_hidden_states,
                    self.config.summaries.hidden,
                )
                encoder_hidden_states_summaries = [
                    HiddenSummary(**summary_dict) if summary_dict else HiddenSummary()
                    for summary_dict in encoder_summaries_dicts
                ]
                logger.info(f"Computed {len(encoder_hidden_states_summaries)} encoder hidden state summaries")
            except Exception as e:
                logger.warning(f"Failed to compute encoder hidden states summaries: {e}")
        
        # Build summary
        summary = Summary(
            prompt_tokens=len(results.prompt_token_ids),
            generated_tokens=len(results.generated_token_ids),
            total_steps=len(results.prompt_token_ids) + len(results.generated_token_ids),
            elapsed_ms=results.elapsed_ms,
        )
        
        # Convert warnings
        warnings = [
            Warning(code=w["code"], message=w["message"])
            for w in results.warnings
        ]
        
        report = Report(
            schema_version="0.1.0",
            trace_id=trace_id,
            created_at_utc=created_at,
            model=model_info,
            run_config=run_config,
            prompt=prompt_info,
            generated=generated_info,
            timeline=timeline,
            summary=summary,
            warnings=warnings,
            encoder_hidden_states=encoder_hidden_states_summaries,
        )
        
        logger.info(f"Report built: {len(timeline)} timeline steps")
        return report
    
    def _build_model_info(self, results: InstrumentationResults) -> ModelInfo:
        """Build ModelInfo from results."""
        bundle = results.model_bundle
        
        dtype_str = str(bundle.dtype).replace("torch.", "")
        
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
        """Build timeline from instrumentation results."""
        timeline = []
        
        for step_data in results.timeline:
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
                    encoder_attentions=results.encoder_attentions,
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
        
        return timeline
    
    def _build_layer_summaries(
        self,
        hidden_states: Optional[List[torch.Tensor]],
        attentions: Optional[List[torch.Tensor]],
        num_layers: int,
        encoder_attentions: Optional[List[torch.Tensor]] = None,
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
                    
                    # Compute encoder attention summary (encoder self-attention for corresponding encoder layer)
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
                        encoder_attention=encoder_attention_summary,
                        cross_attention=cross_attention_summary,
                    )
                    layers.append(layer_summary)
                    
                except Exception as e:
                    logger.warning(f"Failed to process layer {layer_idx}: {e}")
        
        return layers


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
    
    assert report.schema_version == "0.1.0"
    assert len(report.timeline) > 0
    
    print("✓ All report builder tests passed!\n")


if __name__ == "__main__":
    _test_report_builder()