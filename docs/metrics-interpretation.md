# CoreVital Metrics Interpretation Guide

This guide explains each metric CoreVital computes, how to interpret thresholds, and the research behind them. For dashboard examples of healthy vs unhealthy runs, see [Visual Examples](visual-examples.md).

## Overview

CoreVital produces **per-step** metrics (entropy, perplexity, surprisal, top-K margin, voter agreement) and **aggregate** signals (health flags, risk score, prompt surprisals, basin scores, attention collapse). This document covers definitions, typical ranges, and when to act.

---

## Per-step metrics (timeline)

### Entropy (bits)

**Definition:** Shannon entropy of the model’s next-token probability distribution at each generation step.  
**Formula:** `H = -Σ p_i log₂(p_i)` (Shannon entropy in bits).  
**Research:** Standard information theory; see e.g. Shannon (1948).

| Range    | Interpretation |
|----------|-----------------|
| < 2      | Very confident (one dominant token). |
| 2–4      | Normal (several plausible options). |
| > 4      | High uncertainty; possible confusion or out-of-distribution input. |

**In the dashboard:** Entropy is plotted per step; a red line at 4.0 marks the “high uncertainty” threshold. Missing values appear as gaps (not zero).

### Perplexity

**Definition:** 2^entropy; effective number of equiprobable tokens the model is choosing among.  
**Use:** Same information as entropy in a different scale; often used in NLP literature.

### Surprisal (bits)

**Definition:** -log₂(probability of actual token); how surprising the actually generated token was.  
**Use:** Measures “cost” of the chosen token; spikes indicate unexpected choices.

### Top-K margin

**Definition:** Difference between the probability of the top token and the second-most likely token.  
**Use:** Small margin means the model was close to choosing another token; useful for confidence and calibration.

### Voter agreement

**Definition:** Sum of probabilities of the top-K tokens (default K=10).  
**Use:** High value means most probability mass is on a small set of candidates; low value means spread across many tokens.

---

## Attention metrics

### Basin score (prompt telemetry)

**Definition:** Per-head ratio: (average attention on **middle** third of keys) / (average attention on **boundary** thirds).  
**Research:** “Attention Basin” (arXiv:2508.05128, Aug 2025): middle tokens can be under-attended (“Lost in the Middle”).

| Basin score | Interpretation |
|-------------|----------------|
| < 0.3     | Head largely ignores middle tokens (potential “lost in the middle”). |
| ~0.5     | Balanced attention across positions (middle third vs two boundary thirds). |
| > 1.5     | Head focuses more on middle than boundaries. |

**In the dashboard:** Sparse Attention tab shows a layers×heads basin heatmap and per-layer bar chart.

### Attention collapse

**Definition:** A head is “collapsed” when its entropy over keys is below a threshold (default 0.1); “focused” when concentration (max weight) exceeds a threshold (default 0.9).  
**Research:** Voita et al. (2019), “Analyzing Multi-Head Self-Attention”: specialist heads often have high concentration; collapse can indicate underuse or failure.

**Health flag:** `attention_collapse_detected` is true when any layer has at least one collapsed head (`collapsed_head_count` > 0). Per-head collapse uses `collapsed_head_entropy_threshold` from the model profile (default 0.1).

---

## Health flags and risk

### Repetition loop

**Definition:** Last-layer hidden-state vectors become nearly identical over 3+ consecutive steps (cosine similarity > 0.9995 by default).  
**Research:** Anisotropy of transformer representations; true repetition yields similarity ~1.0, so 0.9995 separates repetitive from normal variance.

**Health flag:** `repetition_loop_detected`. **Action:** Shorten generation, change prompt, or use `should_intervene()` to resample.

### Mid-layer anomaly

**Definition:** NaN/Inf in middle-third layers, or L2 norm “explosion” (e.g. 8× the early-layer baseline for that step).  
**Research:** Middle layers are associated with “truth” and semantic processing; anomalies there are more concerning than in early (syntactic) or late (token choice) layers.

**Health flag:** `mid_layer_anomaly_detected`. **Action:** Check inputs and numerical stability; consider different model or precision.

### NaN/Inf detected

**Health flags:** `nan_detected`, `inf_detected`.  
**Action:** Critical; stop and debug (inputs, precision, or code).

### High entropy steps

**Definition:** Count of steps where entropy > 4.0 (threshold configurable).  
**Health flag:** `high_entropy_steps`. A few is normal; many suggests confusion.

### Risk score (0–1)

**Definition:** Aggregate score combining health flags and summary statistics; higher means higher likelihood of poor or anomalous output.  
**Typical use:** < 0.3 low risk; 0.3–0.7 moderate; > 0.7 high risk. Use with `on_risk` capture or `should_intervene()`.

---

## Threshold reference (defaults)

| Signal                 | Default threshold | Config / profile key                          |
|------------------------|-------------------|-----------------------------------------------|
| High entropy           | 4.0 bits          | `high_entropy_threshold_bits`                 |
| Repetition cosine      | 0.9995            | `repetition_cosine_threshold`                 |
| L2 explosion (mid-layer) | 8x baseline     | `l2_explosion_multiplier`                     |
| Collapsed head entropy | 0.1               | `collapsed_head_entropy_threshold`            |
| Focused head concentration | 0.9           | `focused_head_concentration_threshold`        |
| Basin anomaly          | 0.3               | Used in `get_basin_anomalies()` and dashboard |

Per-model overrides: see [Model compatibility – Per-model threshold profiles](model-compatibility.md#per-model-threshold-profiles).

---

## Example scenarios

1. **Repetition loop:** Entropy drops; last-layer cos-sim > 0.9995 for 3+ steps; `repetition_loop_detected: true`. → Shorten generation or resample.
2. **Attention collapse:** Many heads with entropy < 0.1; `attention_collapse_detected: true`. → Check model and data; some models have many collapsed heads by design.
3. **High entropy (confusion):** Many steps with entropy > 4; `high_entropy_steps` large. → Check prompt and domain; consider prompt engineering.
4. **Lost in the middle:** Low basin scores (< 0.3) in middle layers. → Aligns with “Attention Basin” findings; consider context length or model choice.

---

## References

- Shannon, C. E. (1948). A Mathematical Theory of Communication.
- Voita et al. (2019). Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned.
- Liu et al. (2025). Attention Basin (arXiv:2508.05128) — middle vs boundary attention.
- [Visual Examples](visual-examples.md) — good vs bad runs in the dashboard.
- [Model compatibility](model-compatibility.md) — architectures, sparse attention, thresholds.
