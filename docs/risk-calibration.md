# Risk and Threshold Calibration

CoreVital's risk score and health-flag thresholds are heuristic defaults tuned for typical LLM inference. This document explains how they work and how to validate (calibrate) them against labeled data.

## Current behavior

- **Risk score** (`risk.py`): Aggregates health flags, continuous metrics, and compound signals into a single 0-1 score. Used for `should_intervene()` and `--capture on_risk`.
- **Thresholds** are overridable per model family via [per-model profiles](model-compatibility.md#per-model-threshold-profiles) in `configs/model_profiles/` (e.g. `gpt2.yaml`, `llama.yaml`).
- **Calibration profiles** (`calibration.py`): Data-driven baselines from known-healthy runs — divergence scoring replaces static thresholds when a profile is present.

### Default risk contributions (heuristic)

| Signal | Default risk | Notes |
|--------|--------------|--------|
| NaN/Inf | 1.0 | Catastrophic; always flag. |
| Repetition loop | 0.9 | Strong indicator of degenerate output. |
| Mid-layer anomaly | 0.7 | Suggests factual processing failure. |
| Attention collapse | 0.15 | Common in healthy runs. |
| Elevated entropy | up to 0.3 | Additive, based on mean entropy / 8.0. |
| Low confidence margin | up to 0.2 | Additive, based on top-k margin. |
| Compound signals | severity-dependent | From `compound_signals.py`. |

These weights have not yet been validated on a large labeled dataset.

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

# Option B: Use the Python API
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

Calibration profiles (`calibration.py`, Issue 33) provide *divergence scores* — how far a trace deviates from a known-healthy baseline. These divergence scores can also be calibrated:

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

## Current status

- **Risk score** (`risk.py`): Composite heuristic with continuous metrics and compound signals. Implemented (Phase 2).
- **Calibration profiles** (`calibration.py`): Data-driven baselines with divergence scoring. Implemented (Phase 4).
- **ECE + Platt scaling** (`calibration_risk.py`): Functions implemented (Phase 5). Benchmark runs pending.
- **Benchmark validation**: Not yet run. Contributions welcome — see data collection workflow above.

Until benchmark results are available, treat risk scores as indicative: use for relative comparison and alerting, and tune per model via `configs/model_profiles/` as needed.
