# ============================================================================
# CoreVital - Report Schema
#
# Purpose: Pydantic models for JSON report structure
# Inputs: None (schema definitions)
# Outputs: Type-safe report models
# Dependencies: pydantic
# Usage: report = Report(schema_version="0.4.0", ...)
#
# Changelog:
#   2026-01-13: Initial schema for Phase-0
#   2026-01-15: Added Seq2Seq support - encoder_attention and cross_attention to LayerSummary,
#                encoder_hidden_states to Report
#   2026-01-21: Phase-0.5 hardening - added extensions field to Report, TimelineStep, and
#                LayerSummary; added encoder_layers field to Report for proper Seq2Seq support
#   2026-01-23: Added comprehensive docstrings clarifying field usage, especially why
#                encoder_attention is always null (deprecated) and how attention fields are used
#   2026-02-07: Pre-phase-1 schema cleanup (0.1.0 → 0.2.0):
#                - Removed deprecated encoder_attention from LayerSummary
#                - Removed deprecated encoder_hidden_states from Report
#                - Changed ModelInfo.dtype to Optional[str] to support "quantized_unknown"
#   2026-02-07: Phase-1a schema (0.2.0 → 0.3.0):
#                - Added TensorAnomalies, HealthFlags, SparseAttentionHead,
#                  PromptAttentionLayer, PromptAnalysis
#                - Extended LogitsSummary with top_k_margin, voter_agreement,
#                  perplexity, surprisal
#                - Extended AttentionSummary with entropy_max, concentration_min,
#                  collapsed_head_count, focused_head_count
#                - Extended LayerSummary with anomalies: Optional[TensorAnomalies]
#                - Extended Report with prompt_analysis, health_flags
#   2026-02-10: Phase-1b — No schema changes (PromptAnalysis, SparseAttentionHead,
#                PromptAttentionLayer already defined in 1a). Now populated by
#                collector prompt forward pass and report builder.
# ============================================================================

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


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
    dtype: Optional[str] = None  # May be "quantized_unknown" when detection fails
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


class TensorAnomalies(BaseModel):
    """NaN/Inf detection flags for a layer's tensors."""

    has_nan: bool = False
    has_inf: bool = False


class LogitsSummary(BaseModel):
    """Logits summary for a step."""

    entropy: Optional[float] = None
    top1_top2_margin: Optional[float] = None
    topk: List[TopKItem] = Field(default_factory=list)
    # Phase-1a additions
    top_k_margin: Optional[float] = None
    voter_agreement: Optional[float] = None  # Deprecated — prefer topk_mass
    topk_mass: Optional[float] = None  # Primary name (sum of top-K probs)
    perplexity: Optional[float] = None
    surprisal: Optional[float] = None


class HiddenSummary(BaseModel):
    """Hidden state summary for a layer."""

    mean: Optional[float] = None
    std: Optional[float] = None
    l2_norm_mean: Optional[float] = None
    max_abs: Optional[float] = None
    sketch: List[float] = Field(default_factory=list)
    clipped: bool = False  # True if values were clamped before stats (numerical stability)


class AttentionSummary(BaseModel):
    """Attention summary for a layer."""

    entropy_mean: Optional[float] = None
    entropy_mean_normalized: Optional[float] = None  # Entropy / log(K), in [0, 1]
    entropy_min: Optional[float] = None
    concentration_max: Optional[float] = None
    # Phase-1a additions
    entropy_max: Optional[float] = None
    concentration_min: Optional[float] = None
    collapsed_head_count: int = 0
    collapsed_head_rate: Optional[float] = None  # collapsed_head_count / num_heads, in [0,1]
    focused_head_count: int = 0
    # Per-head max attention weight (Voita et al. 2019: specialist heads ~80% max weight)
    max_weight_per_head: Optional[List[float]] = None


class LayerSummary(BaseModel):
    """
    Per-layer summary for decoder layers (in timeline) or encoder layers (in encoder_layers).

    This class represents summaries for a single transformer layer. The same structure is used
    for both decoder layers (in the timeline) and encoder layers (in encoder_layers for Seq2Seq models).

    Field Usage:
    ------------
    - hidden_summary: Summary statistics for the layer's hidden states

    - attention_summary: Summary statistics for SELF-ATTENTION.
      * For decoder layers (in timeline): This is decoder self-attention
      * For encoder layers (in encoder_layers): This is encoder self-attention
      * This field ALWAYS contains the self-attention summary, regardless of model type

    - cross_attention: Only used in decoder layers (in timeline) for Seq2Seq models.
      Contains summary statistics for decoder-to-encoder cross-attention (decoder attending
      to encoder outputs). Always null for CausalLM models and for encoder layers.

    - extensions: For future metric expansion. Custom key-value pairs.

    Examples:
    --------
    For CausalLM models:
    - timeline[].layers[]: decoder layers with attention_summary (decoder self-attention)
    - encoder_layers: null

    For Seq2Seq models:
    - timeline[].layers[]: decoder layers with attention_summary (decoder self-attention)
      and cross_attention (decoder-to-encoder attention)
    - encoder_layers[]: encoder layers with attention_summary (encoder self-attention)
    """

    layer_index: int = Field(description="Zero-based index of the layer within its context (decoder or encoder)")
    hidden_summary: HiddenSummary = Field(
        default_factory=HiddenSummary,
        description="Summary statistics for the layer's hidden states (mean, std, norms, etc.)",
    )
    attention_summary: AttentionSummary = Field(
        default_factory=AttentionSummary,
        description="Summary statistics for SELF-ATTENTION. For decoder layers (in timeline), "
        "this is decoder self-attention. For encoder layers (in encoder_layers), "
        "this is encoder self-attention.",
    )
    cross_attention: Optional[AttentionSummary] = Field(
        default=None,
        description="Summary statistics for decoder-to-encoder cross-attention. "
        "Only used in decoder layers (in timeline) for Seq2Seq models. "
        "Always null for CausalLM models and for encoder layers.",
    )
    anomalies: Optional[TensorAnomalies] = Field(
        default=None,
        description="NaN/Inf detection flags for this layer's hidden states and attentions. Phase-1a addition.",
    )
    extensions: Dict[str, Any] = Field(
        default_factory=dict, description="For future metric expansion. Custom key-value pairs."
    )


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


class SparseAttentionHead(BaseModel):
    """Sparse attention storage using Structure of Arrays (SoA).

    Stores only significant attention connections (weight > threshold).
    All three arrays have the same length = number of stored connections.
    """

    query_indices: List[int] = Field(default_factory=list)
    key_indices: List[int] = Field(default_factory=list)
    weights: List[float] = Field(default_factory=list)


class PromptAttentionLayer(BaseModel):
    """One layer's sparse attention heads + basin scores."""

    heads: List[SparseAttentionHead] = Field(default_factory=list)
    basin_scores: List[float] = Field(default_factory=list)


class PromptAnalysis(BaseModel):
    """Prompt telemetry from the extra forward pass (Phase-1b).

    Contains sparse attention profiles, layer transformations,
    and per-token surprisal for prompt tokens.
    """

    layers: List[PromptAttentionLayer] = Field(default_factory=list)
    layer_transformations: List[float] = Field(default_factory=list)
    prompt_surprisals: List[float] = Field(default_factory=list)


class HealthFlags(BaseModel):
    """Aggregated health flags from post-processing (Phase-1c)."""

    nan_detected: bool = False
    inf_detected: bool = False
    attention_collapse_detected: bool = False
    attention_collapse_severity: Optional[float] = None  # Variable severity from detect_attention_collapse [0,1]
    high_entropy_steps: int = 0
    repetition_loop_detected: bool = False
    mid_layer_anomaly_detected: bool = False


class RAGContext(BaseModel):
    """Optional RAG (retrieval-augmented generation) context metadata (Foundation F3).

    When the run is part of a RAG pipeline, store context length and retrieval info
    so runs can be correlated with context size, source docs, etc. No retrieval
    instrumentation; caller provides this via --rag-context or API.
    """

    context_token_count: Optional[int] = Field(default=None, description="Number of tokens in the retrieved context.")
    retrieved_doc_ids: Optional[List[str]] = Field(
        default=None, description="IDs of retrieved documents (e.g. chunk or doc IDs)."
    )
    retrieved_doc_titles: Optional[List[str]] = Field(
        default=None, description="Human-readable titles for retrieved documents."
    )
    retrieval_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Extra retrieval info (e.g. k, scores, source).",
    )


class Report(BaseModel):
    """
    Complete monitoring report.

    This is the top-level report structure containing all instrumentation data from a model run.
    The structure varies slightly between CausalLM and Seq2Seq models.

    Seq2Seq-Specific Fields:
    -------------------------
    - encoder_layers: List of LayerSummary objects, one per encoder layer.
      Each LayerSummary contains hidden_summary and attention_summary (encoder self-attention).
      Always null for CausalLM models.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "schema_version": "0.4.0",
                "trace_id": "7b1f8c7e-6d4f-4b4a-a7c5-3d9d6a3b5e6a",
                "created_at_utc": "2026-01-11T15:22:08Z",
            }
        }
    )

    schema_version: str = "0.4.0"
    trace_id: str
    created_at_utc: str
    model: ModelInfo
    run_config: RunConfig
    prompt: PromptInfo
    generated: GeneratedInfo
    timeline: List[TimelineStep] = Field(
        default_factory=list, description="Timeline of generation steps. Each step contains decoder layer summaries."
    )
    summary: Summary
    warnings: List[Warning] = Field(default_factory=list)
    encoder_layers: Optional[List[LayerSummary]] = Field(
        default=None,
        description="List of LayerSummary objects, one per encoder layer (Seq2Seq only). "
        "Each LayerSummary contains hidden_summary and attention_summary (encoder self-attention). "
        "Always null for CausalLM models.",
    )
    prompt_analysis: Optional[PromptAnalysis] = Field(
        default=None,
        description="Prompt telemetry from extra forward pass (Phase-1b). "
        "Contains sparse attention profiles, layer transformations, prompt surprisals. "
        "Null when --no-prompt-telemetry is set or not yet implemented.",
    )
    health_flags: Optional[HealthFlags] = Field(
        default=None,
        description="Aggregated health flags from post-processing (Phase-1c). Null when not yet implemented.",
    )
    extensions: Dict[str, Any] = Field(
        default_factory=dict,
        description="For future metric expansion. Known keys: 'rag' (RAGContext, F3), 'performance', 'risk' (Phase-2).",
    )
