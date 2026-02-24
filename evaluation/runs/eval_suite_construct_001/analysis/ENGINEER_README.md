# CoreVital Metric Evaluation

This document summarizes findings from the automated CoreVital metric evaluation run in:

- `evaluation/runs/eval_suite_construct_001`

Goal of this evaluation:

- Evaluate **CoreVital metrics themselves** (correctness, validity, usefulness)
- Not primarily benchmark model output quality
- Use output quality only as a **secondary operational usefulness target**

## Scope

Run configuration (from `evaluation/runs/eval_suite_construct_001`):

- Models (3):
  - `meta-llama/Llama-3.2-1B-Instruct`
  - `Qwen/Qwen2-0.5B-Instruct`
  - `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Seeds: `42,123,999`
- Cases: 10 (includes gradable prompts + probe prompts)
- Total runs: `90`
- Success rate: `100%` (`90/90`)

Primary outputs:

- `evaluation/runs/eval_suite_construct_001/analysis/summary.json`
- `evaluation/runs/eval_suite_construct_001/analysis/graded_runs.csv`
- `evaluation/runs/eval_suite_construct_001/analysis/report.md`

## Key Findings

## 1) Metric implementation correctness looks strong

The analyzer recomputed selected metrics from raw report fields and matched CoreVital outputs:

- `high_entropy_steps` recomputation match rate: `1.0`
- `risk_score` recomputation match rate: `1.0`

Interpretation:

- The metrics are being reported consistently with the current formulas.
- This validates implementation consistency, not construct validity.

## 2) `attention_collapse_detected` is not discriminative in this evaluation

Observed prevalence:

- Overall: `1.0`
- Baseline-tag runs (`low_entropy_expected`): `1.0`
- Probe-tag runs: `1.0`
- All 3 models: `1.0`

Interpretation:

- The metric is likely capturing a common architecture/runtime pattern rather than a useful runtime pathology indicator.
- In current form, it behaves like a near-constant signal.
- It creates a practical risk floor (around `0.3`) in `risk_score`.

Engineering implication:

- `attention_collapse_detected` should not be treated as a standalone alert in this configuration.
- Its contribution to `risk_score` should be re-weighted, gated, or baseline-normalized.

## 3) `high_entropy_steps` shows real signal and practical usefulness

Construct-validity signals by tag:

- `high_entropy` tag: high-entropy run rate = `1.0`
- `probe` tag: high-entropy run rate = `0.889`
- `factual` tag: high-entropy run rate = `0.0`
- `low_entropy_expected` tag: high-entropy run rate = `0.222`

Operational usefulness (secondary target, bad outputs):

- `high_entropy_steps > 0` vs bad output:
  - Precision: `0.9375`
  - Recall: `0.3333`
  - F1: `0.4918`

Interpretation:

- Strong precision: when it triggers, it often indicates something useful.
- Limited recall: it misses many undesirable outputs.
- This is expected for a single metric and still makes it a useful diagnostic signal.

## 4) `repetition_loop_detected` likely under-triggers on current repetition probes

Observed:

- `repetition` tag high entropy often increases (`0.778`), but `repetition_loop_detected` stays `0.0`
- Output repetition heuristic found positives:
  - confusion shows `FN = 3`, `TP = 0`

Interpretation:

- Current hidden-state repetition logic and/or threshold likely too strict for these models/prompts.
- Or the probe induces repeated output patterns that do not match the hidden-state pattern CoreVital expects.

Engineering implication:

- This metric needs targeted validation and likely retuning.

## 5) `risk_score` is a coarse health heuristic, not a calibrated quality/correctness score

Threshold behavior vs bad outputs:

- `risk >= 0.3`: high recall, weak precision (floor effect)
- `risk >= 0.5` / `0.7`: higher precision but poor recall (`0.2667`)

Interpretation:

- `risk_score` is implemented correctly but is not calibrated for downstream quality/correctness prediction.
- Current weighting is likely dominated by the always-on attention-collapse term.

Engineering implication:

- Treat `risk_score` as a coarse triage/health signal.
- Do not position it as a correctness score.

## 6) Seed stability is reasonable (not chaotic)

Across same `model + case` groups (3 seeds):

- Avg stddev `risk_score`: `~0.0369`
- Avg stddev `entropy_max`: `~0.3455`
- Avg stddev `high_entropy_steps`: `~0.3434`

Interpretation:

- Metrics are not unstable across seeds.
- There is still meaningful variation (expected with sampling).
- This supports usefulness for monitoring/trending, not deterministic pass/fail.

## What Is Validated vs Not Yet Validated

Validated (first-pass):

- Metric computation consistency for:
  - `high_entropy_steps`
  - `risk_score`
- Practical usefulness of `high_entropy_steps` as a high-precision warning signal
- Lack of discriminative utility for `attention_collapse_detected` in current setup

Not yet validated / needs more work:

- Construct validity of `repetition_loop_detected`
- Architecture-specific calibration of entropy/collapse thresholds
- Generalization across broader prompt/task distributions
- Production calibration and alert thresholds

## Recommendations (Prioritized)

## Priority 1: De-risk `risk_score` by reducing collapse domination

Actions:

- Add an offline/analyzer variant:
  - `risk_score_no_collapse`
  - `risk_score_reweighted_collapse`
- Compare threshold metrics against current `risk_score` before changing core code

Why:

- Fastest way to quantify how much `attention_collapse` is hurting usefulness
- No CoreVital runtime changes needed to test the hypothesis

## Priority 2: Rework `attention_collapse_detected` for discriminative use

Options:

- Baseline-normalize per model family (or per prompt/model baseline)
- Increase thresholds in model profiles
- Change aggregation logic (e.g., require sustained or multi-layer abnormality, not any collapse)
- Treat collapse as informational telemetry, not a direct risk factor

Why:

- Current prevalence (`1.0`) makes it non-actionable

## Priority 3: Retune / stress-test `repetition_loop_detected`

Actions:

- Add stronger repetition probes (longer outputs, lower-temp repetition prompts)
- Compare hidden-state detector vs output n-gram heuristic across seeds/models
- Sweep repetition threshold (`repetition_cosine_threshold`) offline first if possible

Why:

- Current eval suggests under-detection (`FN` on repetition probes)

## Priority 4: Expand construct-validity probe suite (metric-focused)

Add/expand probes for:

- Repetition
- High entropy / ambiguity
- Low-entropy deterministic responses
- Formatting constraints

Then report expected-trigger rates by tag/category (already supported by analyzer).

Why:

- This directly tests whether formulas measure what they claim

## Priority 5: Separate product messaging: health score vs correctness

Recommendations for docs/product positioning:

- State clearly that CoreVital metrics are **internal-health/inference-behavior signals**
- Treat output correctness as an external downstream validation target
- Avoid presenting `risk_score` as a semantic correctness probability

Why:

- Current evaluation strongly supports this distinction

## Suggested Next Experiments

1. Offline alternative risk scoring in analyzer

- Compute and compare:
  - `risk_no_collapse`
  - `risk_half_weight_collapse`
  - `risk_entropy_plus_repetition_only`

2. Repetition detector validation sweep

- Run only repetition probes across 3 seeds and 3 models
- Increase `max_new_tokens`
- Compare detector recall vs output repetition heuristic

3. Threshold calibration by model family

- Use `model_profiles` to tune:
  - entropy threshold
  - collapse thresholds
  - repetition cosine threshold

4. Add a “metric validity dashboard” report

- Tag-based expected-trigger tables
- Per-model prevalence
- Seed-stability plots

## TL;DR

- CoreVital metric implementation appears correct for tested formulas.
- `high_entropy_steps` is the strongest validated metric so far (useful, high precision).
- `attention_collapse_detected` is currently non-discriminative in this setup and likely over-weighted in `risk_score`.
- `repetition_loop_detected` appears under-sensitive on current probes and needs targeted retuning/validation.
- `risk_score` is a coarse internal-health heuristic, not a correctness score, and needs calibration for practical alerting.
