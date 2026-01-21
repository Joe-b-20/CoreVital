# ============================================================================
# CoreVital - Report Schema
#
# Purpose: Pydantic models for JSON report structure
# Inputs: None (schema definitions)
# Outputs: Type-safe report models
# Dependencies: pydantic
# Usage: report = Report(schema_version="0.1.0", ...)
#
# Changelog:
#   2026-01-13: Initial schema for Phase-0
#   2026-01-15: Added Seq2Seq support - encoder_attention and cross_attention to LayerSummary,
#                encoder_hidden_states to Report
#   2026-01-21: Phase-0.5 hardening - added extensions field to Report, TimelineStep, and
#                LayerSummary; added encoder_layers field to Report for proper Seq2Seq support
# ============================================================================

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class QuantizationInfo(BaseModel):
    """Quantization information."""
    enabled: bool = False
    method: Optional[str] = None


class ModelInfo(BaseModel):
    """Model metadata."""
    hf_id: str
    revision: Optional[str] = None
    architecture: str
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    tokenizer_hf_id: str
    dtype: str
    device: str
    quantization: QuantizationInfo = Field(default_factory=QuantizationInfo)


class GenerationConfig(BaseModel):
    """Generation configuration."""
    do_sample: bool
    temperature: float
    top_k: int
    top_p: float


class SketchConfig(BaseModel):
    """Sketch configuration."""
    method: str
    dim: int
    seed: int


class HiddenConfig(BaseModel):
    """Hidden state summary configuration."""
    enabled: bool
    stats: List[str]
    sketch: SketchConfig


class AttentionConfig(BaseModel):
    """Attention summary configuration."""
    enabled: bool
    stats: List[str]


class LogitsConfig(BaseModel):
    """Logits summary configuration."""
    enabled: bool
    stats: List[str]
    topk: int


class SummariesConfig(BaseModel):
    """Summaries configuration."""
    hidden: HiddenConfig
    attention: AttentionConfig
    logits: LogitsConfig


class SinkConfig(BaseModel):
    """Sink configuration."""
    type: str
    target: str


class RunConfig(BaseModel):
    """Runtime configuration snapshot."""
    seed: int
    device_requested: str
    max_new_tokens: int
    generation: GenerationConfig
    summaries: SummariesConfig
    sink: SinkConfig


class PromptInfo(BaseModel):
    """Prompt information."""
    text: str
    token_ids: List[int]
    num_tokens: int
   
    

class GeneratedInfo(BaseModel):
    """Generated text information."""
    output_text: str
    token_ids: List[int]
    num_tokens: int
    


class TopKItem(BaseModel):
    """Top-k token probability item."""
    token_id: int
    token_text: str
    prob: float


class LogitsSummary(BaseModel):
    """Logits summary for a step."""
    entropy: Optional[float] = None
    top1_top2_margin: Optional[float] = None
    topk: List[TopKItem] = Field(default_factory=list)


class HiddenSummary(BaseModel):
    """Hidden state summary for a layer."""
    mean: Optional[float] = None
    std: Optional[float] = None
    l2_norm_mean: Optional[float] = None
    max_abs: Optional[float] = None
    sketch: List[float] = Field(default_factory=list)


class AttentionSummary(BaseModel):
    """Attention summary for a layer."""
    entropy_mean: Optional[float] = None
    entropy_min: Optional[float] = None
    concentration_max: Optional[float] = None


class LayerSummary(BaseModel):
    """Per-layer summary."""
    layer_index: int
    hidden_summary: HiddenSummary = Field(default_factory=HiddenSummary)
    attention_summary: AttentionSummary = Field(default_factory=AttentionSummary)
    encoder_attention: Optional[AttentionSummary] = None
    cross_attention: Optional[AttentionSummary] = None
    extensions: Dict[str, Any] = Field(default_factory=dict)


class TokenInfo(BaseModel):
    """Token information."""
    token_id: int
    token_text: str
    is_prompt_token: bool


class TimelineStep(BaseModel):
    """Single step in the timeline."""
    step_index: int
    token: TokenInfo
    logits_summary: LogitsSummary = Field(default_factory=LogitsSummary)
    layers: List[LayerSummary] = Field(default_factory=list)
    extensions: Dict[str, Any] = Field(default_factory=dict)


class Summary(BaseModel):
    """Overall run summary."""
    prompt_tokens: int
    generated_tokens: int
    total_steps: int
    elapsed_ms: int


class Warning(BaseModel):
    """Warning message."""
    code: str
    message: str


class Report(BaseModel):
    """Complete monitoring report."""
    # Tell Pydantic to use the V2 configuration style
    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "schema_version": "0.1.0",
                "trace_id": "7b1f8c7e-6d4f-4b4a-a7c5-3d9d6a3b5e6a",
                "created_at_utc": "2026-01-11T15:22:08Z",
            }
        }
    )

    schema_version: str = "0.1.0"
    trace_id: str
    created_at_utc: str
    model: ModelInfo
    run_config: RunConfig
    prompt: PromptInfo
    generated: GeneratedInfo
    timeline: List[TimelineStep] = Field(default_factory=list)
    summary: Summary
    warnings: List[Warning] = Field(default_factory=list)
    encoder_hidden_states: Optional[List[HiddenSummary]] = None
    encoder_layers: Optional[List[LayerSummary]] = None
    extensions: Dict[str, Any] = Field(default_factory=dict)