# CoreVital Trace Schema Reference

Schema version: **0.4.0** · Pydantic source: `src/CoreVital/reporting/schema.py`

This document describes every field in a CoreVital trace JSON file, its type, valid range, and how to interpret its values. For annotated real-world examples, see [`sample_trace_annotated.json`](sample_trace_annotated.json).

---

## Top-level fields

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | `str` | Semantic version of the report format (currently `"0.4.0"`). |
| `trace_id` | `str` | UUID v4 unique to this run. |
| `created_at_utc` | `str` | ISO-8601 UTC timestamp when the report was finalized. |
| `model` | `ModelInfo` | Model metadata (see below). |
| `run_config` | `RunConfig` | Configuration snapshot for reproducibility (see below). |
| `prompt` | `PromptInfo` | Input prompt text and tokenization. |
| `generated` | `GeneratedInfo` | Model output text and tokenization. |
| `timeline` | `List[TimelineStep]` | Per-step generation metrics (see below). |
| `summary` | `Summary` | Aggregate run statistics. |
| `warnings` | `List[Warning]` | Warning messages generated during capture. |
| `encoder_layers` | `List[LayerSummary] \| null` | Encoder layer summaries (Seq2Seq only; `null` for CausalLM). |
| `prompt_analysis` | `PromptAnalysis \| null` | Prompt telemetry from extra forward pass. |
| `health_flags` | `HealthFlags \| null` | Aggregated boolean health indicators. |
| `extensions` | `Dict[str, Any]` | Phase-2+ analytics: risk, compound_signals, early_warning, fingerprint, narrative, performance. |

---

## ModelInfo

| Field | Type | Description |
|-------|------|-------------|
| `hf_id` | `str` | HuggingFace model identifier (e.g., `"meta-llama/Llama-3.2-3B-Instruct"`). |
| `revision` | `str \| null` | Git commit SHA of the model weights on HuggingFace Hub. |
| `architecture` | `str` | Model class name (e.g., `LlamaForCausalLM`, `T5ForConditionalGeneration`). |
| `num_layers` | `int` | Number of transformer layers. Determines the size of `layers[]` in each timeline step. |
| `hidden_size` | `int` | Dimensionality of hidden-state vectors. |
| `num_attention_heads` | `int` | Attention heads per layer. Collapsed/focused counts are out of this total. |
| `tokenizer_hf_id` | `str` | HuggingFace ID of the tokenizer used. |
| `dtype` | `str \| null` | Tensor data type (e.g., `"float16"`, `"bfloat16"`, `"quantized_unknown"`). |
| `device` | `str` | Compute device (`"cuda"`, `"cpu"`, etc.). |
| `quantization.enabled` | `bool` | Whether quantization was applied. |
| `quantization.method` | `str \| null` | Quantization method (e.g., `"4-bit"`, `"8-bit"`). |

---

## RunConfig

| Field | Type | Description |
|-------|------|-------------|
| `seed` | `int` | Random seed for reproducibility. |
| `device_requested` | `str` | Device requested by the user. |
| `max_new_tokens` | `int` | Maximum tokens to generate. |
| `generation.do_sample` | `bool` | Whether to use stochastic sampling. |
| `generation.temperature` | `float` | Sampling temperature. < 1.0 sharpens distribution; > 1.0 flattens it. |
| `generation.top_k` | `int` | Top-K sampling cutoff. |
| `generation.top_p` | `float` | Nucleus sampling probability cutoff. |
| `summaries.hidden.enabled` | `bool` | Whether hidden-state summaries are computed. |
| `summaries.hidden.stats` | `List[str]` | Which hidden-state stats to compute (e.g., `["mean", "std", "l2_norm_mean", "max_abs"]`). |
| `summaries.hidden.sketch` | `SketchConfig` | Random-projection sketch config (`method`, `dim`, `seed`). |
| `summaries.attention.enabled` | `bool` | Whether attention summaries are computed. |
| `summaries.attention.stats` | `List[str]` | Which attention stats to compute. |
| `summaries.logits.enabled` | `bool` | Whether logits summaries are computed. |
| `summaries.logits.stats` | `List[str]` | Which logits stats to compute. |
| `summaries.logits.topk` | `int` | Number of top tokens to record per step. |
| `sink.type` | `str` | Output sink type (`"local_file"`, `"sqlite"`, etc.). |
| `sink.target` | `str` | Output path or connection string. |

---

## PromptInfo / GeneratedInfo

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | Raw text of the prompt or generated output. |
| `token_ids` | `List[int]` | Integer token IDs from the tokenizer. |
| `num_tokens` | `int` | Length of `token_ids`. |

For `GeneratedInfo`, `output_text` is used instead of `text`.

---

## Summary

| Field | Type | Description |
|-------|------|-------------|
| `prompt_tokens` | `int` | Number of tokens in the prompt. |
| `generated_tokens` | `int` | Number of tokens generated. |
| `total_steps` | `int` | `prompt_tokens + generated_tokens`. |
| `elapsed_ms` | `int` | Wall-clock time for instrumented generation (excluding model load). |

---

## TimelineStep

Each entry in `timeline[]` represents one generation step.

| Field | Type | Description |
|-------|------|-------------|
| `step_index` | `int` | Absolute position in the sequence (starts at `prompt_tokens` for the first generated token). |
| `token` | `TokenInfo` | Token produced at this step (`token_id`, `token_text`, `is_prompt_token`). |
| `logits_summary` | `LogitsSummary` | Metrics from the logit distribution (see below). |
| `layers` | `List[LayerSummary]` | Per-layer decoder summaries (see below). One entry per decoder layer. |
| `extensions` | `Dict[str, Any]` | Reserved for future per-step extensions. |

---

## Logits metrics (LogitsSummary)

Computed from the full next-token probability distribution at each generation step, **before** sampling.

| Field | Type | Range | Low values mean | High values mean |
|-------|------|-------|-----------------|------------------|
| `entropy` | `float` | \[0, log₂(vocab\_size)\] | Very confident — one dominant token. < 2 bits = high confidence. | High uncertainty — many plausible tokens. > 4 bits = confused / OOD input. |
| `perplexity` | `float` | \[1, vocab\_size\] | Model choosing among very few options (≈ 2^entropy). | Model choosing among many options. 200+ = highly uncertain. |
| `surprisal` | `float` | \[0, ∞) | Chosen token was highly expected (prob near 1). | Chosen token was unexpected. Spikes indicate surprising choices. |
| `top_k_margin` | `float` | \[0, 1\] | Near-tie between top-1 and top-2 tokens — fragile decision. | Clear winner — model is confident in its top choice. |
| `topk_mass` | `float` | \[0, 1\] | Probability spread across many tokens beyond the top-K. | All probability concentrated in top-K tokens. 1.0 = no tail mass. |
| `voter_agreement` | `float` | \[0, 1\] | *(Deprecated alias for `topk_mass`.)* | |
| `top1_top2_margin` | `float` | \[0, 1\] | *(Deprecated alias for `top_k_margin`.)* | |

### TopKItem (in `topk_probs` / `topk`)

| Field | Type | Description |
|-------|------|-------------|
| `token_id` | `int` | Token ID in the vocabulary. |
| `token_text` | `str` | Decoded text for this token. |
| `prob` | `float` | Probability after softmax. Range: \[0, 1\]. |

`topk` is a deprecated alias for `topk_probs`; both are written for backward compatibility.

---

## Hidden-state metrics (HiddenSummary)

Summary statistics of the hidden-state tensor after each layer's forward pass.

| Field | Type | Range | Low values mean | High values mean |
|-------|------|-------|-----------------|------------------|
| `mean` | `float` | (−∞, ∞) | Typical range near 0 due to LayerNorm. Large deviations may indicate bias. | |
| `std` | `float` | \[0, ∞) | Very uniform activations. | High activation variance. Grows with depth in some architectures (e.g., T5 mid-layers). |
| `l2_norm_mean` | `float` | \[0, ∞) | Small activation magnitudes (typical for early layers). | Large activations. Used for explosion detection — z-score > 2.5 = outlier. |
| `max_abs` | `float` | \[0, ∞) | No extreme values. | Extreme activations present. Very high values may indicate numerical instability. |
| `sketch` | `List[float]` | — | Random-projection sketch vector. Reserved for future clustering; may be empty. | |
| `clipped` | `bool` | — | `false` = no clamping needed. | `true` = values were clamped to ±1e6 for numerical stability before computing stats. |

### Clipping diagnostics (when `clipped` is true)

| Field | Type | Description |
|-------|------|-------------|
| `clip_fraction` | `float` | Fraction of tensor elements that exceeded the ±1e6 threshold. Tiny = benign. |
| `clip_max_before` | `float` | Maximum absolute value before clamping. Extreme values = severe instability. |

---

## Attention metrics (AttentionSummary)

Aggregated across all attention heads at a given layer, for the current query position(s).

| Field | Type | Range | Low values mean | High values mean |
|-------|------|-------|-----------------|------------------|
| `entropy_mean` | `float` | \[0, ∞) nats | Attention is peaked — heads attend to few keys. | Attention is diffuse — heads spread weight across many keys. |
| `entropy_mean_normalized` | `float` | \[0, 1\] | 0.0 = perfectly peaked attention. < 0.03 = collapsed head. | 1.0 = uniform attention (all keys weighted equally). |
| `entropy_min` | `float` | \[0, ∞) nats | At least one head has very peaked attention (potentially a specialist). | Even the most peaked head has spread attention. |
| `entropy_max` | `float` | \[0, ∞) nats | All heads are relatively peaked. | At least one head has very diffuse attention. |
| `concentration_max` | `float` | \[0, 1\] | No head strongly focuses on a single key. | 1.0 = one head puts all attention on one token. > 0.9 = focused head. |
| `concentration_min` | `float` | \[0, 1\] | At least one head is very spread out (diffuse). | All heads have some concentration. |
| `collapsed_head_count` | `int` | \[0, num\_heads\] | 0 = no collapsed heads. Healthy. | Many heads collapsed (normalized entropy < 0.03). May indicate underuse or failure. |
| `collapsed_head_rate` | `float` | \[0, 1\] | 0.0 = no collapse. | 1.0 = all heads collapsed. Expected for single-token Seq2Seq decoder self-attention at step 0. |
| `focused_head_count` | `int` | \[0, num\_heads\] | No specialist heads (concentration > 0.9). | Many specialist heads strongly attending to specific tokens. |
| `max_weight_per_head` | `List[float] \| null` | \[0, 1\] each | — | Per-head maximum attention weight (Voita et al. 2019). ~0.8+ indicates specialist heads. |

---

## LayerSummary

One per decoder layer (in `timeline[].layers[]`) or encoder layer (in `encoder_layers[]`).

| Field | Type | Description |
|-------|------|-------------|
| `layer_index` | `int` | Zero-based index within decoder or encoder stack. |
| `hidden_summary` | `HiddenSummary` | Hidden-state statistics (see above). |
| `attention_summary` | `AttentionSummary` | Self-attention statistics (see above). |
| `cross_attention` | `AttentionSummary \| null` | Decoder-to-encoder cross-attention (**Seq2Seq decoder layers only**). Always `null` for CausalLM and for encoder layers. |
| `anomalies` | `TensorAnomalies \| null` | NaN/Inf detection flags. |
| `extensions` | `Dict[str, Any]` | Reserved for future per-layer extensions. |

### TensorAnomalies

| Field | Type | Description |
|-------|------|-------------|
| `has_nan` | `bool` | `true` if NaN detected in this layer. Catastrophic. |
| `has_inf` | `bool` | `true` if Inf detected in this layer. Catastrophic. |

---

## PromptAnalysis

Telemetry from an extra forward pass over the prompt before generation begins.

| Field | Type | Description |
|-------|------|-------------|
| `layers` | `List[PromptAttentionLayer]` | One per model layer. Each contains `heads` and `basin_scores`. |
| `layer_transformations` | `List[float]` | Cosine distance between consecutive layer outputs. Length = `num_layers - 1`. High values = layer significantly transforms its input. |
| `prompt_surprisals` | `List[float]` | Per-token −log₂(prob) for each prompt token given its prefix (CausalLM only). Empty for Seq2Seq (encoder is not autoregressive). |

### PromptAttentionLayer

| Field | Type | Description |
|-------|------|-------------|
| `heads` | `List[SparseAttentionHead]` | Sparse attention in Structure-of-Arrays format. |
| `basin_scores` | `List[float]` | Per-head ratio: (avg attention on middle ⅓ of keys) / (avg attention on boundary ⅓s). |

### Basin scores

| Value | Interpretation |
|-------|----------------|
| < 0.3 | Head largely ignores middle tokens — potential "lost in the middle" (Liu et al. 2025). |
| ≈ 0.5 | Balanced attention across positions. |
| > 1.5 | Head focuses more on middle than boundaries. |

### SparseAttentionHead

| Field | Type | Description |
|-------|------|-------------|
| `query_indices` | `List[int]` | Query positions with significant attention connections. |
| `key_indices` | `List[int]` | Key positions those queries attend to. |
| `weights` | `List[float]` | Attention weights for each (query, key) pair. All three arrays have the same length. |

---

## HealthFlags

Aggregated boolean health indicators computed during post-processing.

| Field | Type | Risk floor | Description |
|-------|------|------------|-------------|
| `nan_detected` | `bool` | 1.0 | Any layer at any step had NaN values. Catastrophic — stop and debug. |
| `inf_detected` | `bool` | 1.0 | Any layer at any step had Inf values. Catastrophic — stop and debug. |
| `attention_collapse_detected` | `bool` | 0.15 | Any layer has collapsed heads. Common in healthy runs; mild concern. |
| `attention_collapse_severity` | `float \| null` | — | \[0, 1\]. Severity from `detect_attention_collapse`. Only present when collapse is detected. |
| `high_entropy_steps` | `int` | — | Count of steps with entropy above the model-specific threshold (default 4.0 bits). A few is normal; many suggests confusion. |
| `repetition_loop_detected` | `bool` | 0.9 | Last-layer hidden states showed cosine similarity > 0.9995 for 3+ consecutive steps. Strong degeneration signal. |
| `mid_layer_anomaly_detected` | `bool` | 0.7 | NaN/Inf or L2 norm explosion in middle-third layers. Suggests factual processing failure. |

---

## Extensions

### `extensions.risk`

Composite risk assessment combining health flags, continuous metrics, and compound signals.

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `risk_score` | `float` | \[0, 1\] | < 0.3 low, 0.3–0.7 moderate, > 0.7 high. |
| `risk_factors` | `List[str]` | — | Contributing factors (e.g., `"elevated_entropy"`, `"compound:context_loss"`). |
| `blamed_layers` | `List[dict]` | — | Layers identified as problematic. Each: `{layer, reasons, severity}`. |
| `blamed_layers_flat` | `List[int]` | — | Backward-compatible flat list of blamed layer indices. |
| `attention_collapse_detail` | `dict` | — | Detailed collapse metrics (see below). |

#### Risk score composition

**Boolean flags (hard floor via `max()`):**

| Signal | Risk floor |
|--------|-----------|
| NaN/Inf | 1.0 |
| Repetition loop | 0.9 |
| Mid-layer anomaly | 0.7 |
| Attention collapse | 0.15 |

**Continuous metrics (additive, capped at 1.0):**

| Signal | Max contribution | Formula |
|--------|-----------------|---------|
| Elevated entropy | 0.3 | `min(1, mean_entropy / 8) × 0.3` |
| Entropy rising trend | 0.2 | Additive when late ⅓ entropy > early ⅓ × 1.3 |
| Low confidence margin | 0.2 | `max(0, 1 − mean_margin × 5) × 0.2` |
| Low top-K mass | 0.15 | `max(0, 1 − mean_mass) × 0.15` |
| Elevated surprisal | 0.1 | `min(0.1, mean_surprisal / 10)` |

**Final:** `min(1.0, max(boolean_floor, boolean_floor + continuous_sum + compound_severity_sum))`

#### Attention collapse detail

| Field | Type | Description |
|-------|------|-------------|
| `mean_collapse_rate` | `float` | \[0, 1\]. Average collapsed-head fraction across all layers and steps. |
| `trend_detected` | `bool` | Collapse rate increasing over time. |
| `trend_peak_deviation` | `float` | Peak deviation from baseline collapse rate. |
| `trend_layers` | `List[int]` | Layers with increasing collapse trend. |
| `catastrophic` | `bool` | Collapse severe enough to warrant stopping generation. |
| `calibration_anomaly` | `bool` | Collapse pattern deviates from calibration baseline. |
| `calibration_anomaly_layers` | `List[int]` | Layers with calibration anomalies. |
| `per_layer_mean_collapse_rate` | `List[float]` | Per-layer mean collapse rate across all steps. Length = `num_layers`. |

#### Layer blame

| Field | Type | Description |
|-------|------|-------------|
| `layer` | `int` | Layer index. |
| `reasons` | `List[str]` | Human-readable reasons (e.g., `"Attention collapse in 86% of steps"`, `"L2 norm outlier (z=4.0)"`). |
| `severity` | `float` | \[0, 1\]. NaN/Inf = 1.0, collapse rate > 50% = 0.4, L2 z-score > 2.5 = 0.5, L2 instability CV > 0.5 = 0.3. |

---

### `extensions.compound_signals`

Multi-metric failure patterns that individual metrics might miss.

| Signal | Trigger condition | Typical severity |
|--------|-------------------|------------------|
| `context_loss` | High entropy + low basin scores (< 0.3) | 0.6 |
| `confident_confusion` | High entropy + high top-K margin | 0.5 |
| `degenerating_generation` | Positive entropy slope + declining margin | 0.5 |
| `attention_bottleneck` | High collapsed head counts + elevated entropy | 0.4 |
| `confident_repetition_risk` | Low entropy + very high top-K mass | 0.3 |

Each entry: `{name, description, severity, evidence}`.

---

### `extensions.early_warning`

Predictive signals detecting degradation *before* health-flag thresholds are tripped.

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `failure_risk` | `float` | \[0, 1\] | Estimated probability that the next N tokens will fail. 0.0 = healthy. |
| `warning_signals` | `List[str]` | — | Active warning signal names. |

| Warning signal | What it detects |
|----------------|-----------------|
| `entropy_accelerating` | Entropy rate-of-change is itself increasing (accelerating confusion). |
| `margin_collapsed` | Top-K margin near zero in recent window. |
| `margin_declining` | Margin dropped to < 30% of its early level. |
| `surprisal_volatile` | Coefficient of variation > 1.5 in recent window. |
| `entropy_margin_divergence` | Entropy above threshold while margin stays high ("confident but confused"). |

Health flags (`repetition_loop`, `mid_layer_anomaly`, `attention_collapse`) are passed through as hard ceilings on `failure_risk`.

---

### `extensions.fingerprint`

25-element numeric vector summarizing a run's behavior (version 2).

| Slots | Content | Typical range |
|-------|---------|---------------|
| 0–4 | Entropy: mean, std, min, max, trend slope | mean 0–8 bits; slope near 0 if stable |
| 5–7 | Margin: mean, std, slope | mean 0–1; slope near 0 if stable |
| 8–10 | Surprisal: mean, std, slope | mean 0–15+; slope near 0 if stable |
| 11–12 | Top-K mass: mean, std | mean 0–1 |
| 13 | Risk score | 0–1 |
| 14 | High-entropy fraction (steps above threshold / total) | 0–1 |
| 15–19 | Boolean flags: nan, inf, repetition, mid-layer anomaly, attention collapse | 0 or 1 |
| 20 | Entropy–margin Pearson correlation | −1 to 1; negative = healthy |
| 21 | Entropy coefficient of variation | 0–∞; > 1 = highly variable |
| 22–23 | First-quarter and last-quarter entropy means | 0–8+ bits |
| 24 | Reserved | 0.0 |

Additional fields: `prompt_hash` (SHA-256 of prompt text), `version` (currently `2`).

---

### `extensions.narrative`

| Field | Type | Description |
|-------|------|-------------|
| `summary` | `str` | Human-readable 2–6 sentence summary referencing actual metric values, step indices, token text, and actionable recommendations. |

---

### `extensions.performance`

Timing breakdown of the CoreVital instrumentation pipeline.

| Field | Type | Description |
|-------|------|-------------|
| `total_wall_time_ms` | `float` | Total wall-clock time including model load, warmup, generation, and report build. |
| `parent_operations` | `List[{name, ms, pct}]` | Top-level timing phases (config_load, model_load, tokenize, prompt_forward_pass, model_inference, report_build). |
| `unaccounted_time` | `{ms, pct}` | Time not attributed to any tracked operation. |
| `original_model_load_ms` | `float` | Time to load the model from disk/hub. |
| `warmup_ms` | `float` | Warmup pass time (not counted in overhead). |
| `baseline_ms` | `float` | Un-instrumented generation time (no hooks). Used for overhead calculation. |
| `instrumented_inference_ms` | `float` | Generation time with CoreVital hooks active. |
| `inference_overhead_ms` | `float` | `instrumented_inference_ms - baseline_ms`. |
| `inference_overhead_pct` | `float` | `(overhead_ms / baseline_ms) × 100`. |
| `corevital_overhead_ms` | `float` | Total CoreVital overhead including report build. |
| `corevital_overhead_pct` | `float` | Total overhead as percentage of baseline. |
| `detailed_breakdown` | `dict` | Nested breakdown with per-step timing and child operations. |

---

### `extensions.calibration` (when a calibration profile is configured)

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `divergence_score` | `float` | \[0, 1\] | How far this trace deviates from a known-healthy baseline. Mean |z-score| / 6, capped at 1. |
| `anomalies` | `List[str]` | — | Human-readable anomaly messages (e.g., `"Step 3 entropy z=4.2"`). |
| `baseline_model_id` | `str` | — | Model used to build the calibration baseline. |
| `baseline_num_runs` | `int` | — | Number of runs in the baseline profile. |

---

## Threshold defaults

| Signal | Default threshold | Config / profile key |
|--------|-------------------|----------------------|
| High entropy | 4.0 bits (model-dependent) | `high_entropy_threshold_bits` |
| Repetition cosine | 0.9995 | `repetition_cosine_threshold` |
| L2 explosion (mid-layer) | 8× median baseline | `l2_explosion_multiplier` |
| Collapsed head (normalized) | 0.03 | `NORMALIZED_COLLAPSED_THRESHOLD` (constant) |
| Focused head concentration | 0.9 | `focused_head_concentration_threshold` |
| Basin anomaly | 0.3 | Used in `get_basin_anomalies()` |

Per-model overrides are in `configs/model_profiles/`. See [Model compatibility](model-compatibility.md) and [Risk calibration](risk-calibration.md) for details.

---

## CausalLM vs Seq2Seq differences

| Aspect | CausalLM | Seq2Seq |
|--------|----------|---------|
| `encoder_layers` | `null` | `List[LayerSummary]` — one per encoder layer |
| `cross_attention` in decoder layers | `null` (absent) | `AttentionSummary` — decoder-to-encoder attention |
| `prompt_analysis.prompt_surprisals` | Populated (one per prompt token minus first) | Empty (encoder is not autoregressive) |
| Decoder self-attention at step 0 | Attends to all prompt tokens (KV cache) | Attends to single start token (all heads trivially collapsed) |
| `timeline[0].step_index` | = `prompt_tokens` | = `prompt_tokens` |
