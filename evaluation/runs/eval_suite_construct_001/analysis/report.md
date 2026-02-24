# CoreVital Eval Suite Report

- Run dir: `/Users/amyb/Documents/github/joe-b-20/corevital/CoreVital/runs/eval_suite_construct_001`
- Successful runs: `90`
- Gradable runs: `72`

## Model Summary

| Model | Runs | Gradable | Task Accuracy | Avg Risk | High-Entropy Run Rate |
|---|---:|---:|---:|---:|---:|
| `meta-llama/Llama-3.2-1B-Instruct` | 30 | 24 | 0.583 | 0.308 | 0.300 |
| `Qwen/Qwen2-0.5B-Instruct` | 30 | 24 | 0.208 | 0.309 | 0.367 |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 30 | 24 | 0.333 | 0.632 | 0.400 |

## CoreVital Metric Internal Consistency

- `high_entropy_steps` recomputation match rate: `1.000`
- `risk_score` recomputation match rate: `1.000`

## CoreVital Signal Prevalence (All Runs)

- `attention_collapse_detected` rate: `1.000`
- `repetition_loop_detected` rate: `0.000`
- `high_entropy_steps > 0` run rate: `0.356`

## Metric Usefulness on Stability Baselines

- Baseline runs (cases tagged `low_entropy_expected`): `18`
- `high_entropy_steps > 0` on baseline runs (lower is better): `0.222`
- Avg risk on baseline runs (lower is better): `0.439`

## Metric Construct Validity Checks

### Tag Signal Expectations

| Tag | Runs | High-Entropy Run Rate | Avg Risk | Attention Collapse Rate | Repetition Rate |
|---|---:|---:|---:|---:|---:|
| `boolean` | 9 | 0.556 | 0.446 | 1.000 | 0.000 |
| `factual` | 27 | 0.000 | 0.433 | 1.000 | 0.000 |
| `formatting` | 9 | 0.333 | 0.354 | 1.000 | 0.000 |
| `gradable` | 72 | 0.222 | 0.418 | 1.000 | 0.000 |
| `high_entropy` | 9 | 1.000 | 0.425 | 1.000 | 0.000 |
| `low_entropy_expected` | 18 | 0.222 | 0.439 | 1.000 | 0.000 |
| `math` | 18 | 0.222 | 0.439 | 1.000 | 0.000 |
| `probe` | 18 | 0.889 | 0.412 | 1.000 | 0.000 |
| `repetition` | 9 | 0.778 | 0.399 | 1.000 | 0.000 |
| `translation` | 9 | 0.444 | 0.363 | 1.000 | 0.000 |

### Seed Stability (Per Metric Across Same Model+Case)

- Groups with 2+ seeds: `30` (total groups: `30`)
- Avg stddev `risk_score`: `0.037`
- Avg stddev `entropy_max`: `0.346`
- Avg stddev `high_entropy_steps`: `0.343`

### Attention Collapse Diagnostics

- Overall collapse rate: `1.000`
- Baseline-tag collapse rate (`low_entropy_expected`): `1.000`
- Probe-tag collapse rate (`probe`): `1.000`
- Models with collapse rate >= 0.9: `Qwen/Qwen2-0.5B-Instruct, TinyLlama/TinyLlama-1.1B-Chat-v1.0, meta-llama/Llama-3.2-1B-Instruct`

## Risk Thresholds vs Bad Output (Auto-Graded Only)

| Threshold | Precision | Recall | F1 | TP | FP | TN | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 0.625 | 1.000 | 0.769 | 45 | 27 | 0 | 0 |
| 0.5 | 0.600 | 0.267 | 0.369 | 12 | 8 | 19 | 33 |
| 0.7 | 0.600 | 0.267 | 0.369 | 12 | 8 | 19 | 33 |

## Risk Thresholds vs Quality-Risk Proxy (Output-Only)

| Threshold | Precision | Recall | F1 | TP | FP | TN | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 0.744 | 1.000 | 0.854 | 67 | 23 | 0 | 0 |
| 0.5 | 0.833 | 0.299 | 0.440 | 20 | 4 | 19 | 47 |
| 0.7 | 0.833 | 0.299 | 0.440 | 20 | 4 | 19 | 47 |

## Signal Checks

- `repetition_loop_detected` vs output repetition heuristic: precision=n/a, recall=0.000, f1=n/a
- `high_entropy_steps > 0` vs bad output (auto-graded prompts): precision=0.938, recall=0.333, f1=0.492
- `high_entropy_steps > 0` vs quality-risk proxy (output-only): precision=0.562, recall=0.269, f1=0.364

## Caveats

- bad_output_label is only computed for auto-gradable prompts
- quality_risk_proxy is output-only and includes bad output, format violations, and repetition
- repetition validation uses output text repetition heuristic, not hidden-state ground truth
- attention_collapse_detected is reported but not treated as standalone correctness target

## Outputs

- `graded_runs.csv`: `/Users/amyb/Documents/github/joe-b-20/corevital/CoreVital/runs/eval_suite_construct_001/analysis/graded_runs.csv`
- `summary.json`: `/Users/amyb/Documents/github/joe-b-20/corevital/CoreVital/runs/eval_suite_construct_001/analysis/summary.json`