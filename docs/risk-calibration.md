# Risk and Threshold Calibration

CoreVital's risk score and health-flag thresholds are heuristic defaults tuned for typical LLM inference. This document explains how they work and how to validate (calibrate) them against labeled data.

## Current behavior

- **Risk score** (`risk.py`): Aggregates health flags, continuous metrics, and compound signals into a single 0-1 score. Used for `should_intervene()` and `--capture on_risk`.
- **Thresholds** are overridable per model family via [per-model profiles](model-compatibility.md#per-model-threshold-profiles) in `configs/model_profiles/` (e.g. `gpt2.yaml`, `llama.yaml`, `mistral.yaml`, `mixtral.yaml`, `qwen2.yaml`, `phi3.yaml`).
- **Calibration profiles** (`calibration.py`): Data-driven baselines from known-healthy runs — divergence scoring replaces static thresholds when a profile is present.

### Composite risk scoring (Phase 2)

The risk score combines three layers of evidence:

**1. Boolean health flags (hard ceilings)**

Boolean flags act as a floor via `max()` — the score is at least as high as the worst boolean flag:

| Signal | Default risk | Notes |
|--------|--------------|--------|
| NaN/Inf | 1.0 | Catastrophic; always flag. Score is exactly 1.0. |
| Repetition loop | 0.9 | Strong indicator of degenerate output. |
| Mid-layer anomaly | 0.7 | Suggests factual processing failure. |
| Attention collapse | 0.15 | Common in healthy runs; mild. |

**2. Continuous timeline metrics (additive)**

When timeline data is available, continuous metrics add components on top of the boolean floor:

| Signal | Max contribution | How it's computed |
|--------|-----------------|-------------------|
| Elevated entropy | 0.3 | `min(1, mean_ent / 8) * 0.3`, threshold > 0.05 |
| Entropy rising trend | 0.2 | `min(0.2, (last⅓ - first⅓) / first⅓ * 0.1)`, requires last⅓ > first⅓ × 1.3 |
| Low confidence margin | 0.2 | `max(0, 1 - mean_margin * 5) * 0.2`, threshold > 0.05 |
| Low top-K mass | 0.15 | `max(0, 1 - mean_mass) * 0.15`, threshold > 0.03 |
| Elevated surprisal | 0.1 | `min(0.1, mean_surprisal / 10)`, threshold > 0.02 |

Each component contributes to `risk_factors` (e.g., `elevated_entropy`, `entropy_rising`, `low_confidence_margin`, `low_topk_mass`, `elevated_surprisal`).

**3. Compound signals (severity-based)**

Detected multi-metric patterns from `compound_signals.py` add their severity to the score:

| Signal | Typical severity | Trigger |
|--------|-----------------|---------|
| `context_loss` | 0.6 | High entropy + low basin scores |
| `confident_confusion` | 0.5 | High entropy + high margin |
| `degenerating_generation` | 0.5 | Rising entropy + declining margin |
| `attention_bottleneck` | 0.4 | High collapse + elevated entropy |
| `confident_repetition_risk` | 0.3 | Low entropy + very high mass |

Each adds `compound:<name>` to risk_factors.

**Final score:** `min(1.0, max(boolean_floor, boolean_floor + continuous_sum + compound_sum))`.

**Legacy fallback:** When no timeline is available, `compute_risk_score_legacy` uses boolean flags only with the same ceiling values. This preserves backward compatibility for consumers that don't pass timeline data.

### Enriched layer blame

Layer blame now returns structured evidence for each blamed layer:

```python
from CoreVital.risk import compute_layer_blame, compute_layer_blame_flat

blamed = compute_layer_blame(layers_by_step)
# [{"layer": 5, "reasons": ["l2_norm_outlier"], "severity": 0.5}, ...]

blamed_flat = compute_layer_blame_flat(layers_by_step)
# [5, ...]  (backward-compatible)
```

Blame conditions: NaN/Inf (severity 1.0), attention collapse rate > 50% across steps (0.4), L2 norm z-score outlier z > 2.5 (0.5), L2 norm instability CV > 0.5 (0.3).

### Early warning

Early warning signals detect degradation patterns *before* they become hard failures. Signals: `entropy_accelerating`, `margin_collapsed`, `margin_declining`, `surprisal_volatile`, `entropy_margin_divergence`. See [Metrics Interpretation Guide](metrics-interpretation.md#early-warning) for details.

## Benchmark Calibration with ECE

`calibration_risk.py` provides the tools to validate and improve the risk score against ground-truth data.

### Key concepts

- **Expected Calibration Error (ECE)**: Measures whether "risk 0.8" actually corresponds to ~80% failure rate. Lower is better; 0.0 = perfectly calibrated. Target: **ECE < 0.10**.
- **Platt scaling**: Fits a logistic curve `calibrated = sigmoid(a * raw_score + b)` that maps heuristic scores to calibrated probabilities.

### API

```python
from CoreVital.calibration_risk import (
    compute_ece,
    fit_platt_scaling,
    apply_platt_scaling,
    evaluate_calibration,
    RiskCalibrationResult,
)

# End-to-end: fit + evaluate
result = evaluate_calibration(raw_scores, labels)
print(f"ECE raw: {result.ece_raw:.3f}")
print(f"ECE calibrated: {result.ece_calibrated:.3f}")

# Production: apply fitted params
calibrated_prob = apply_platt_scaling(raw_score, result.a, result.b)
```

## Data Collection Workflow

Calibrating the risk score requires labeled `(risk_score, quality_label)` pairs. Here is the concrete workflow:

### Step 1: Choose benchmark datasets

Pick datasets where ground-truth output quality is known or can be assessed:

| Dataset | What it tests | Label strategy |
|---------|---------------|----------------|
| [TruthfulQA](https://github.com/sylinrl/TruthfulQA) | Hallucination / factual accuracy | Mark truthful answers as label=0 (good), untruthful as label=1 (failure) |
| [HellaSwag](https://rowanzellers.com/hellaswag/) | Commonsense reasoning | Correct continuation = 0, incorrect = 1 |
| [HaluEval](https://github.com/RUCAIBox/HaluEval) | Hallucination detection | Non-hallucinated = 0, hallucinated = 1 |
| Custom prompt suite | Domain-specific quality | Human-labeled good/bad outputs |

### Step 2: Run CoreVital traces

For each prompt in the benchmark, run CoreVital instrumentation and collect the risk score:

```bash
# Option A: Use the CLI
corevital run --model <model_id> --prompt "<prompt>" --out trace.json

# Option B: Use the CLI with a calibration profile
corevital run --model <model_id> --prompt "<prompt>" --calibration calibration/gpt2.json

# Option C: Use the Python API
from CoreVital import CoreVitalMonitor, Config
monitor = CoreVitalMonitor(Config())
result = monitor.run(model, tokenizer, prompt)
risk_score = result.report.extensions["risk"]["risk_score"]
```

### Step 3: Collect labeled pairs

Build parallel lists of `(risk_score, label)`:

```python
raw_scores = []  # from Step 2
labels = []      # from Step 1: 1 = failure, 0 = success

for prompt, ground_truth in benchmark:
    result = monitor.run(model, tokenizer, prompt)
    raw_scores.append(result.report.extensions["risk"]["risk_score"])
    labels.append(1 if is_failure(result, ground_truth) else 0)
```

**Minimum sample size**: 200+ labeled pairs recommended. At least 30 in each class (failure/success).

### Step 4: Fit calibration and evaluate

```python
from CoreVital.calibration_risk import evaluate_calibration

result = evaluate_calibration(raw_scores, labels)
print(f"Samples: {result.n_samples}, Failure rate: {result.label_rate:.1%}")
print(f"ECE (raw):        {result.ece_raw:.4f}")
print(f"ECE (calibrated): {result.ece_calibrated:.4f}")
print(f"Platt params:     a={result.a:.4f}, b={result.b:.4f}")
```

**Interpret**:
- ECE < 0.05: excellent calibration
- ECE < 0.10: good — safe for SLA thresholds
- ECE < 0.20: acceptable — use with caution
- ECE > 0.20: risk score needs weight tuning before production use

### Step 5: Deploy calibrated scoring (optional)

Save the Platt parameters and apply in production:

```python
from CoreVital.calibration_risk import apply_platt_scaling, RiskCalibrationResult

# Load saved calibration (store result.to_dict() in your config)
params = load_calibration_params()  # {"platt_a": ..., "platt_b": ...}
cal = RiskCalibrationResult.from_dict(params)

# In production
raw_risk = result.report.extensions["risk"]["risk_score"]
calibrated_risk = apply_platt_scaling(raw_risk, cal.a, cal.b)
```

### Combining with calibration profiles

Calibration profiles (`calibration.py`) provide *divergence scores* — how far a trace deviates from a known-healthy baseline. These divergence scores can also be calibrated:

```python
from CoreVital.calibration import CalibrationProfile, compute_divergence_score
from CoreVital.calibration_risk import evaluate_calibration

profile = CalibrationProfile.load("calibration/gpt2.json")

# Collect divergence scores + labels
div_scores = []
labels = []
for prompt, ground_truth in benchmark:
    trace = run_and_get_trace(prompt)
    div, _ = compute_divergence_score(trace, profile)
    div_scores.append(div)
    labels.append(1 if is_failure(trace, ground_truth) else 0)

result = evaluate_calibration(div_scores, labels)
```

### Building a calibration profile

Use the CLI to build a calibration profile from known-healthy prompts:

```bash
# Build profile from a prompt file (one prompt per line)
corevital calibrate --model gpt2 --prompts prompts.txt --out calibration/gpt2.json

# Use additional options
corevital calibrate \
  --model gpt2 \
  --prompts prompts.txt \
  --out calibration/gpt2.json \
  --device cuda \
  --max_new_tokens 50 \
  --config configs/custom.yaml
```

On weak-CPU hosts (e.g. RunPod), add `--report-on-gpu` to `corevital run` so report computation runs on GPU; same for long `corevital calibrate` runs if CPU becomes the bottleneck.

The resulting JSON file contains empirical distributions for entropy, margin, surprisal (per-step) and L2 norm, attention entropy (per-layer). Pass it to `corevital run --calibration <path>` or set `calibration_profile` in config to enable divergence scoring.

## Per-model threshold profiles

Different model families have different threshold needs:

| Model | `high_entropy_threshold_bits` | `l2_explosion_multiplier` | Notes |
|-------|------------------------------|--------------------------|-------|
| GPT-2 | 5.0 | 5 | Higher entropy tolerance for creative generation |
| LLaMA | 3.5 | 10 | Tighter entropy threshold; higher L2 tolerance |
| Mistral | 4.0 (default) | 8 (default) | Stub — not yet calibrated |
| Mixtral | 4.5 | 8 (default) | MoE routing adds entropy variance |
| Qwen2 | 4.5 | 8 (default) | Large vocab (~152k) raises baseline entropy |
| Phi-3 | 3.8 | 7.0 | Small model; slightly tighter |
| Default | 4.0 | 8 | Fallback for unrecognized architectures |

GPT-2 and LLaMA also have `typical_entropy_range` and `typical_l2_norm_range` ([p10, p90]) from calibration runs. Other profiles are stubs awaiting calibration data. Contributors can run `corevital calibrate` to produce empirical thresholds.

## Experiment validation (benchmark)

A validation experiment was run on GSM8K (math) and HumanEval (code) with four models (Llama-3.1-8B, Qwen-2.5-7B, Mistral-7B, Mixtral-8x7B) using pass@k sampling (10 runs per prompt). 14,540 total traces; 11,403 non-format-failure runs. The pipeline is implemented in **`experiment/scripts/calibrate_risk.py`**; outputs live under **`experiment/calibration/`**. See the [Validation Report](validation-report.md) for full methodology, [experiment/README.md](../experiment/README.md) for accessible summary, and `experiment/analysis/` for detailed findings.

### Key findings

**Ablation (HistGradientBoosting, grouped 5-fold CV):**
- **T6 (full CoreVital, 104 features) AUROC range:** 0.60-0.90 across 8 model/dataset cells
- **Best:** Qwen/HumanEval (0.90)
- **Worst:** Qwen/GSM8K (0.60)
- **Key nuance:** Biggest gains vary by model/task. Early-window features (T4) drive HumanEval (Qwen: 0.73→0.85). Mistral/GSM8K peaks at T5 (0.71); T6 drops to 0.67.

**Pooled logistic regression (step 5):**
- **AUROC:** 0.744, **ECE:** 0.164 (6 features: entropy_mean, margin_mean, topk_mass_mean, prompt_surprisal_mean, basin_score_mean, entropy_slope)
- Weaker than ablation but simpler. **Not a universal replacement:** step 6 per-model evaluation shows mixed results—underperforms built-in risk_score in most individual cells.

**Models are overconfident, but CoreVital still discriminates:**
- Models assign high confidence (margin) to many incorrect outputs
- CoreVital signals catch confident-but-wrong runs (Qwen/HumanEval: compound_density_per_100t AUROC 0.92; Mistral/GSM8K: hidden_max_abs_last_layer_mean AUROC 0.90)

**Three-way outcome signatures:**
- Correct, incorrect, and format-failure have **distinct, architecture-dependent signals**
- Mistral: hidden_max_abs distinguishes incorrect (PP 0.90, gap 0.17)
- Mixtral: focused_head for incorrect (PP 0.90, gap 0.14)
- Qwen: concentration_min for format-failure (PP 0.77, gap 0.23)

**Temperature-robust:**
- Signals stable across temp 0.7 vs 0.8 (mean predictive power shift 0.028)
- Validates that signals measure model internals, not sampling artifacts

### Step-by-step validation results

**Step 1 (ECE + Platt):** The built-in **risk_score** was evaluated per (model, dataset) with grouped held-out CV. **Raw ECE 0.24–0.70** (poor calibration); after Platt scaling, ECE drops sharply (&lt; 0.02). **AUROC 0.48-0.62** (modest discrimination)—Platt improves probability calibration but does not turn the heuristic into a strong discriminator.

**Built-in risk_score (status):**
- **Saturates at 1.0** for Mistral (96%) and Mixtral (94%)
- **Poor calibration:** ECE 0.24-0.70 before Platt
- **Weak discrimination:** AUROC 0.48-0.62 (near chance in some cells)
- **Implication:** Heuristic is a research placeholder, **not production-calibrated**. Treat as indicative only.

**Step 5 (data-driven failure model):** Pooled logistic regression on 11,403 runs achieves **grouped-CV AUROC 0.744, ECE 0.164**. Features and coefficients in `experiment/calibration/step5_proposed_risk_model.json`. This is a **pooled improvement path**, not a universal replacement.

**Step 6 (per-model transfer):** The pooled model (step 5) **underperforms** built-in risk_score in most individual (model, dataset) cells. Delta vs risk_score: negative in 5/8 cells, positive in 3/8. **Implication:** Need **per-model calibration**, not pooled formula.

**Built-in failure_risk:** Evaluated in step 1. **Discrete** (2-5 unique values), **AUROC near chance** in many cells. Did **not** behave like a reliable calibrated predictor. Trend detectors remain **theoretical / exploratory**.

### Recommendations for production

Based on validation findings:

1. **Do NOT use built-in risk_score or failure_risk as production-calibrated predictors** — they are heuristic placeholders
2. **Implement per-model calibration** — Fit separate models per (model, dataset) or per architecture
3. **Add confident-but-wrong detector** — Use signals from confidence calibration (compound_density, l2_norm_slope, entropy_range)
4. **Add per-outcome models** — Separate predictors for format-failure vs task-incorrectness (distinct signatures)
5. **Leverage early-window features** — Qwen/HumanEval shows huge T4 gains; implement partial-generation risk estimation (offline)
6. **Run your own calibration** — Use the data collection workflow above with your domain/model/task

For detailed analysis, ablation curves, confidence calibration, and outcome profiling, see [Validation Report](validation-report.md).

## Current status

- **Risk score** (`risk.py`): Composite heuristic with continuous metrics and compound signals. Implemented (Phase 2). **Not production-calibrated**; see experiment validation above.
- **Enriched layer blame** (`risk.py`): Structured blame with reasons and severity. Implemented (Phase 2).
- **Compound signals** (`compound_signals.py`): Five multi-metric patterns. Implemented (Phase 2).
- **Early warning** (`early_warning.py`): Five trend detectors. Implemented (Phase 2). **Exploratory**; not validated as production quality predictors.
- **Narratives** (`narrative.py`): Data-specific text with step indices, token text, recommendations. Implemented (Phase 2).
- **Calibration profiles** (`calibration.py`): Data-driven baselines with divergence scoring. Implemented (Phase 4).
- **ECE + Platt scaling** (`calibration_risk.py`): Functions implemented (Phase 5). **Benchmark run complete** via `experiment/scripts/calibrate_risk.py`; see `experiment/calibration/` and [Validation Report](validation-report.md).
- **Experiment calibration** (`experiment/scripts/calibrate_risk.py`): Steps 1–6 (ECE/Platt, failure_risk eval, profiles, per-cell LR, consensus, step 5 pooled model, step 6 per-model eval). Artifacts in `experiment/calibration/`.
- **Model profiles** (`configs/model_profiles/`): GPT-2 and LLaMA calibrated; Mistral, Mixtral, Qwen2, Phi-3 stubs. (Phase 4).
- **Metric consistency validation** (`reporting/validation.py`): Information-theoretic invariant checks. Implemented (Phase 5).

The built-in **risk_score** is heuristic and indicative only unless you calibrate it (e.g. via the data collection workflow above). The built-in **failure_risk** and trend detectors were evaluated in the experiment and are **not** reliable calibrated predictors; treat them as **exploratory debugging aids**, not as production-calibrated alerting signals. For calibrated failure probabilities, use a learned/calibrated model: the step 5 model (see experiment artifacts) or run the data collection workflow and fit your own calibration. Tune per model via `configs/model_profiles/` for heuristic thresholds only.
