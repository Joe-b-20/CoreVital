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

## Current status

- **Risk score** (`risk.py`): Composite heuristic with continuous metrics and compound signals. Implemented (Phase 2).
- **Enriched layer blame** (`risk.py`): Structured blame with reasons and severity. Implemented (Phase 2).
- **Compound signals** (`compound_signals.py`): Five multi-metric patterns. Implemented (Phase 2).
- **Early warning** (`early_warning.py`): Five trend detectors. Implemented (Phase 2).
- **Narratives** (`narrative.py`): Data-specific text with step indices, token text, recommendations. Implemented (Phase 2).
- **Calibration profiles** (`calibration.py`): Data-driven baselines with divergence scoring. Implemented (Phase 4).
- **ECE + Platt scaling** (`calibration_risk.py`): Functions implemented (Phase 5). Benchmark runs pending.
- **Model profiles** (`configs/model_profiles/`): GPT-2 and LLaMA calibrated; Mistral, Mixtral, Qwen2, Phi-3 stubs. (Phase 4).
- **Metric consistency validation** (`reporting/validation.py`): Information-theoretic invariant checks. Implemented (Phase 5).
- **Benchmark validation**: Not yet run. Contributions welcome — see data collection workflow above.

Until benchmark results are available, treat risk scores as indicative: use for relative comparison and alerting, and tune per model via `configs/model_profiles/` as needed.
