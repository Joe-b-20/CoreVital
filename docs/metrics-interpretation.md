# CoreVital Metrics Interpretation Guide

This guide explains each metric CoreVital computes, how to interpret thresholds, and the research behind them. For dashboard examples of healthy vs unhealthy runs, see [Visual Examples](visual-examples.md).

## Overview

CoreVital produces **per-step** metrics (entropy, perplexity, surprisal, top-K margin, top-K mass) and **aggregate** signals (health flags, risk score, compound signals, early warning, prompt surprisals, basin scores, attention collapse). This document covers definitions, typical ranges, and when to act.

---

## Per-step metrics (timeline)

### Entropy (bits)

**Definition:** Shannon entropy of the model's next-token probability distribution at each generation step.  
**Formula:** `H = -Σ p_i log₂(p_i)` (Shannon entropy in bits).  
**Research:** Standard information theory; see e.g. Shannon (1948).

| Range    | Interpretation |
|----------|-----------------|
| < 2      | Very confident (one dominant token). |
| 2–4      | Normal (several plausible options). |
| > 4      | High uncertainty; possible confusion or out-of-distribution input. |

**Entropy mode:** By default CoreVital computes entropy over the full vocabulary distribution (`entropy_mode: "full"`). For large-vocabulary models where full softmax is expensive, set `entropy_mode: "topk_approx"` in the logits config to approximate entropy from the top-K logits with a tail-mass correction term.

**In the dashboard:** Entropy is plotted per step; a red line marks the "high uncertainty" threshold (default 4.0 bits; configurable per model via `high_entropy_threshold_bits`). Missing values appear as gaps (not zero).

### Perplexity

**Definition:** 2^entropy; effective number of equiprobable tokens the model is choosing among.  
**Use:** Same information as entropy in a different scale; often used in NLP literature.  
**Invariant:** `perplexity == 2^entropy` (validated by metric consistency checks when `DEBUG` logging is enabled).

### Surprisal (bits)

**Definition:** -log₂(probability of actual token); how surprising the actually generated token was.  
**Use:** Measures "cost" of the chosen token; spikes indicate unexpected choices.

### Top-K margin

**Definition:** Difference between the probability of the top token and the second-most likely token.  
**Use:** Small margin means the model was close to choosing another token; useful for confidence and calibration.  
*Deprecated:* `top1_top2_margin` is the same quantity; prefer `top_k_margin` in config and schema.

### Top-K mass (formerly voter_agreement)

**Definition:** Sum of probabilities of the top-K tokens (default K=10). Previously exposed as `voter_agreement` (deprecated; still written for backward compatibility).  
**Use:** High value means most probability mass is on a small set of candidates; low value means spread across many tokens. Used as a component in the risk score.  
**Config key:** `topk_mass` in `logits.stats`.

### Top-K probs

**Definition:** List of individual probabilities for the top-K tokens. Previously exposed as `topk` (deprecated alias; still written for backward compatibility).  
**Config key:** `topk_probs` in `logits.stats`.

---

## Attention metrics

### Basin score (prompt telemetry)

**Definition:** Per-head ratio: (average attention on **middle** third of keys) / (average attention on **boundary** thirds).  
**Research:** "Attention Basin" (arXiv:2508.05128, Aug 2025): middle tokens can be under-attended ("Lost in the Middle").

| Basin score | Interpretation |
|-------------|----------------|
| < 0.3     | Head largely ignores middle tokens (potential "lost in the middle"). |
| ~0.5     | Balanced attention across positions (middle third vs two boundary thirds). |
| > 1.5     | Head focuses more on middle than boundaries. |

**In the dashboard:** Sparse Attention tab shows a layers×heads basin heatmap and per-layer bar chart.

### Concentration min

**Definition:** The minimum value, across all attention heads and query positions, of the maximum attention weight assigned to any single key. Low concentration_min means at least one head at one position has very diffuse (spread-out) attention, which may indicate an underperforming head.

### Entropy mean (raw)

**Definition:** Mean Shannon entropy of attention distributions across all heads and query positions in a layer. Measured in nats (natural log).

### Entropy mean normalized

**Definition:** `entropy_mean / log(K)` where K is the source sequence length. Normalized to the [0, 1] range so that 1.0 = uniform attention and 0.0 = perfectly peaked attention. This normalization makes collapse detection consistent across different sequence lengths.  
**Config key:** `entropy_mean_normalized` in `attention.stats`.

### Attention collapse

**Definition:** A head is "collapsed" when its normalized entropy is below 0.03 (i.e., attention is almost entirely peaked on a single token). A head is "focused" when concentration (max weight) exceeds a threshold (default 0.9).  
**Research:** Voita et al. (2019), "Analyzing Multi-Head Self-Attention": specialist heads often have high concentration; collapse can indicate underuse or failure.

**Previous behavior:** Collapse was detected using raw entropy < 0.1 nats. This was sequence-length dependent. The current implementation uses normalized entropy (< 0.03) for consistent detection across all sequence lengths.

**Health flag:** `attention_collapse_detected` is true when any layer has at least one collapsed head (`collapsed_head_count` > 0).

**Note:** The `collapsed_head_entropy_threshold` field in model profiles (raw nats) is deprecated for collapse detection. Collapse now uses the normalized threshold constant `NORMALIZED_COLLAPSED_THRESHOLD = 0.03`.

### Attention normalization

When attention weights are slightly drifted from summing to 1.0 (due to floating point or custom attention implementations), CoreVital re-normalizes by division (`attention / sum`) rather than applying softmax. This preserves the relative ratios of attention weights. Previous behavior used `F.softmax()` which exponentially reweights already-post-softmax values.

---

## Health flags and risk

### Repetition loop

**Definition:** Last-layer hidden-state vectors become nearly identical over 3+ consecutive steps (cosine similarity > 0.9995 by default).  
**Enhancement:** When `token_id_buffer` is provided to `detect_repetition_loop`, hidden-state similarity is cross-checked with n-gram repetition (bigram/trigram) for higher confidence detection. The `consecutive_required` parameter controls how many consecutive similar steps are needed (default: 3).  
**Research:** Anisotropy of transformer representations; true repetition yields similarity ~1.0, so 0.9995 separates repetitive from normal variance.

**Health flag:** `repetition_loop_detected`. **Action:** Shorten generation, change prompt, or use `should_intervene()` to resample.

### Mid-layer anomaly

**Definition:** NaN/Inf in middle-third layers, or L2 norm "explosion" detected using the **median** of early-layer L2 norms as baseline (not mean, which is sensitive to outlier layers).  
**Interpretation:** Empirically, mid-layer anomalies correlate with higher failure rates; this is a heuristic and model-dependent.

**Health flag:** `mid_layer_anomaly_detected`. **Action:** Check inputs and numerical stability; consider different model or precision.

### NaN/Inf detected

**Health flags:** `nan_detected`, `inf_detected`.  
**Action:** Critical; stop and debug (inputs, precision, or code).

### High entropy steps

**Definition:** Count of steps where entropy exceeds the threshold (default 4.0 bits; configurable per model via `high_entropy_threshold_bits` in model profiles).  
**Health flag:** `high_entropy_steps`. A few is normal; many suggests confusion.

### Hidden state clipping diagnostics

When hidden state values are clamped for numerical stability, the summary now reports additional diagnostic fields:
- `clip_fraction`: fraction of tensor elements that exceeded the clipping threshold (±1e6)
- `clip_max_before`: the maximum absolute value before clamping was applied

These help distinguish benign clipping (tiny fraction affected) from severe instability (large fraction or extreme values).

---

## Risk score (0–1)

### Composite scoring

The risk score aggregates **boolean health flags**, **continuous timeline metrics**, and **compound signals** into a single 0–1 value.

**Boolean flags** act as hard ceilings via `max()`:

| Signal | Risk ceiling | Notes |
|--------|-------------|-------|
| NaN/Inf | 1.0 | Catastrophic; always returns 1.0 immediately. |
| Repetition loop | 0.9 | Strong indicator of degenerate output. |
| Mid-layer anomaly | 0.7 | Suggests factual processing failure. |
| Attention collapse | 0.15 | Common in healthy runs; mild. |

**Continuous metrics** are additive (each contributes a component capped individually, total capped at 1.0):

| Signal | Max contribution | Formula |
|--------|-----------------|---------|
| Elevated entropy | up to 0.3 | `mean_entropy / 8.0`, capped at 0.3 |
| Entropy rising trend | up to 0.15 | Additive when late-half entropy > early-half |
| Low confidence margin | up to 0.2 | `max(0, 0.3 - mean_margin) / 0.3 * 0.2` |
| Low top-K mass | up to 0.15 | `max(0, 0.5 - mean_agreement) / 0.5 * 0.15` |
| Elevated surprisal | up to 0.1 | `(mean_surprisal - 3) / 7 * 0.1` |

**Compound signals** (from `compound_signals.py`) add their `severity` to the score and `compound:<name>` to risk_factors.

**Legacy fallback:** When timeline data is not available, `compute_risk_score_legacy` uses boolean flags only.

**Typical interpretation:** < 0.3 low risk; 0.3–0.7 moderate; > 0.7 high risk.

---

## Layer blame (enriched)

Layer blame identifies which specific layers are responsible for anomalies. Each blamed layer now includes structured evidence:

```json
{
  "layer": 5,
  "reasons": ["l2_norm_outlier", "attention_collapse_rate"],
  "severity": 0.5
}
```

**Blame conditions:**

| Condition | Severity | Description |
|-----------|----------|-------------|
| NaN/Inf | 1.0 | Layer contains NaN or Inf values |
| Attention collapse rate > 50% | 0.4 | More than half of generation steps show collapsed heads in this layer |
| L2 norm z-score outlier (z > 2.5) | 0.5 | Layer's mean L2 norm is a statistical outlier vs cross-layer baseline |
| L2 norm instability (CV > 0.5) | 0.3 | Layer's L2 norm varies widely across generation steps |

**Backward compatibility:** `blamed_layers_flat` (List[int]) is provided alongside the enriched `blamed_layers` (List[dict]) for consumers expecting the old format.

---

## Compound signals

Compound signals detect multi-metric failure patterns that individual metrics might miss. Each signal includes a name, human-readable description, severity (0–1), and metric evidence.

| Signal | Trigger condition | Severity |
|--------|-------------------|----------|
| `context_loss` | High entropy + low basin scores (< 0.3) | 0.6 |
| `confident_confusion` | High entropy + high top-K margin (model is "confident but wrong") | 0.5 |
| `degenerating_generation` | Entropy slope positive + margin declining over time | 0.5 |
| `attention_bottleneck` | High collapsed head counts + elevated entropy | 0.4 |
| `confident_repetition_risk` | Low entropy + very high top-K mass (model locked onto one token pattern) | 0.3 |

Compound signals are reported in `extensions.compound_signals` and contribute to the risk score.

---

## Early warning

Early warning detects degradation patterns *before* they trip boolean health-flag thresholds. Unlike the risk score (which aggregates what already happened), early warning predicts whether the next N tokens are likely to fail.

### Warning signals

| Signal | What it detects |
|--------|-----------------|
| `entropy_accelerating` | Entropy rate-of-change is itself increasing (accelerating confusion) |
| `margin_collapsed` | Top-K margin near zero in recent window |
| `margin_declining` | Margin dropped to < 30% of its early level |
| `surprisal_volatile` | Coefficient of variation > 1.5 in recent window (erratic choices) |
| `entropy_margin_divergence` | Entropy above threshold while margin stays high ("confident but confused") |

**Note:** The old signal names `entropy_rising` and `high_entropy` from earlier versions are removed. Consumers checking for those specific strings should update to the new signal names above.

**Health flag passthrough:** Boolean health flags (`repetition_loop`, `mid_layer_anomaly`, `attention_collapse`) are passed through as hard ceilings on failure_risk.

**Empty timeline behavior:** Returns `(0.0, [])` (previously returned `(0.3, [])`).

**Threshold configuration:** The `entropy_margin_divergence` check uses the model profile's `high_entropy_threshold_bits` (default 4.0), not a hardcoded constant.

---

## Fingerprint vector (v2)

The fingerprint is a compact 25-element numeric vector summarizing a run's behavior for clustering and pattern detection. Version 2 replaces the original 9-element vector.

| Slots | Content |
|-------|---------|
| 0–4 | Entropy profile: mean, std, min, max, trend slope |
| 5–7 | Margin profile: mean, std, slope |
| 8–10 | Surprisal profile: mean, std, slope |
| 11–12 | Agreement (top-K mass) profile: mean, std |
| 13 | Risk score |
| 14 | High-entropy fraction (steps above threshold / total) |
| 15–19 | Boolean flags (nan, inf, repetition, mid-layer anomaly, attention collapse) |
| 20 | Entropy–margin correlation (Pearson r; negative = healthy) |
| 21 | Entropy coefficient of variation |
| 22–23 | First-quarter and last-quarter entropy means (temporal progression) |
| 24 | Reserved slot |

**Version field:** `extensions.fingerprint.version` reports the fingerprint version (currently 2). Use `is_legacy_fingerprint()` to detect old 9-element vectors.

---

## Calibration profiles

When a calibration profile is configured (via `calibration_profile` in config or `--calibration` CLI flag), CoreVital computes a **divergence score** alongside the heuristic risk score. The divergence score measures how far a trace deviates from a known-healthy baseline using z-scores.

**Report extension:** `extensions.calibration` contains:
- `divergence_score`: 0–1 value (mean |z-score| / 6, capped)
- `anomalies`: list of human-readable anomaly messages (e.g., "Step 3 entropy z=4.2")
- `baseline_model_id`: model used to build the baseline
- `baseline_num_runs`: number of runs in the baseline

See [Risk and Threshold Calibration](risk-calibration.md) for the full workflow.

---

## Metric consistency validation

When logging is at `DEBUG` level, CoreVital automatically validates information-theoretic invariants after building each report:

- `perplexity == 2^entropy` (1% relative tolerance)
- `top_k_margin ≤ topk_mass`
- High concentration implies low entropy per layer
- Non-negative entropy and surprisal
- `perplexity ≥ 1`

Warnings are stored in `extensions.metric_consistency` if any violations are found. These checks never fail the report build — they are advisory only.

---

## Threshold reference (defaults)

| Signal                 | Default threshold | Config / profile key                          |
|------------------------|-------------------|-----------------------------------------------|
| High entropy           | 4.0 bits (model-dependent) | `high_entropy_threshold_bits`       |
| Repetition cosine      | 0.9995            | `repetition_cosine_threshold`                 |
| L2 explosion (mid-layer) | 8x median baseline | `l2_explosion_multiplier`                  |
| Collapsed head (normalized) | 0.03         | `NORMALIZED_COLLAPSED_THRESHOLD` (constant)   |
| Focused head concentration | 0.9           | `focused_head_concentration_threshold`        |
| Basin anomaly          | 0.3               | Used in `get_basin_anomalies()` and dashboard |

**Per-model overrides:** Different model families have calibrated thresholds. For example, GPT-2 uses `high_entropy_threshold_bits: 5.0` (higher tolerance for creative generation) while LLaMA uses `3.5` (catches subtle issues in instruction-tuned models). See [Model compatibility – Per-model threshold profiles](model-compatibility.md#per-model-threshold-profiles).

**Available model profiles:** `gpt2`, `llama`, `mistral`, `mixtral`, `qwen2`, `phi3`, `default`.

---

## Narrative

The narrative is a 2–6 sentence human-readable summary of each run. It references actual metric values, step indices, and token text — not generic descriptions.

**Content includes:**
- Risk level with score and primary contributing factor
- Peak entropy value with step index and the actual token at that step
- Entropy trend (rising/stable) when ≥ 6 steps are available
- Up to 2 compound signals with descriptions
- Blamed layers with reasons (supports both flat and enriched formats)
- Actionable recommendations (e.g., "lower temperature" for repetition, "check precision" for mid-layer anomaly)

---

## Example scenarios

1. **Repetition loop:** Entropy drops; last-layer cos-sim > 0.9995 for 3+ steps; `repetition_loop_detected: true`. → Shorten generation or resample.
2. **Attention collapse:** Many heads with normalized entropy < 0.03; `attention_collapse_detected: true`. → Check model and data; some models have many collapsed heads by design.
3. **High entropy (confusion):** Many steps with entropy > threshold; `high_entropy_steps` large. → Check prompt and domain; consider prompt engineering.
4. **Lost in the middle:** Low basin scores (< 0.3) in middle layers. → Aligns with "Attention Basin" findings; consider context length or model choice.
5. **Context loss (compound):** High entropy combined with low basin scores triggers `context_loss` compound signal. Model is losing track of middle-context tokens.
6. **Degenerating generation (compound):** Entropy slope is positive while margin is declining. Generation quality is deteriorating over time.
7. **Calibration anomaly:** Divergence score from calibration profile is high (> 0.5) with specific anomaly messages pointing to steps/metrics. → Compare against baseline runs to identify what changed.

---

## References

- Shannon, C. E. (1948). A Mathematical Theory of Communication.
- Voita et al. (2019). Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned.
- Liu et al. (2025). Attention Basin (arXiv:2508.05128) — middle vs boundary attention.
- [Visual Examples](visual-examples.md) — good vs bad runs in the dashboard.
- [Model compatibility](model-compatibility.md) — architectures, sparse attention, thresholds.
- [Risk and Threshold Calibration](risk-calibration.md) — ECE, Platt scaling, benchmark workflow.
