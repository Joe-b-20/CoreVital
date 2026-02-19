# Risk and Threshold Calibration

CoreVital's risk score and health-flag thresholds are heuristic defaults tuned for typical LLM inference. This document explains how they work and how calibration (validation against labeled data) is planned.

## Current behavior

- **Risk score** (`risk.py`): Aggregates health flags and layer anomalies into a single 0-1 score. Used for `should_intervene()` and `--capture on_risk`.
- **Thresholds** are overridable per model family via [per-model profiles](model-compatibility.md#per-model-threshold-profiles) in `configs/model_profiles/` (e.g. `gpt2.yaml`, `llama.yaml`).

### Default risk contributions (heuristic)

| Signal | Default risk | Notes |
|--------|--------------|--------|
| NaN/Inf | 1.0 | Catastrophic; always flag. |
| Repetition loop | 0.9 | Strong indicator of degenerate output. |
| Mid-layer anomaly | 0.7 | Suggests factual processing failure. |
| Attention collapse | 0.3 | Common in healthy runs. |
| High entropy steps | Profile-dependent | `high_entropy_threshold_bits` in model profile (default 4.0). |

These values have not been validated on a large labeled dataset of good vs. bad generations.

## Calibration (planned)

**Calibration** means aligning reported risk with actual failure rates:

- **ECE (Expected Calibration Error)**: Whether "risk 0.8" corresponds to ~80% actual failure rate.
- **Benchmark validation**: Evaluate on labeled benchmarks (e.g. TruthfulQA, HaluEval) to tune thresholds and risk weights.

Planned work: evaluate risk score and flags on labeled good/bad generations; optionally fit a calibration curve; document ECE and benchmark results.

Until then, treat risk and thresholds as indicative: use for relative comparison and alerting, and tune per model via `configs/model_profiles/` as needed.
