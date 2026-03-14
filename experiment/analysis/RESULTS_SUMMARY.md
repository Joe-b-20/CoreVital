# CoreVital Validation Experiment — Results Summary

## Experiment Design

- **Design:** Pass@k (k=10) under sampling (temp 0.7 + 0.8)
- **Datasets:** GSM8K (200 prompts) + HumanEval (164 prompts)
- **Models:** Llama-3.1-8B, Qwen-2.5-7B, Mistral-7B, Mixtral-8x7B (8-bit)
- **Total traces:** 14,540
- **After format-failure exclusion:** 11,403

## Per-Model Accuracy

| Model | Dataset | Prompts | Runs | Accuracy | Format Fail % |
|-------|---------|---------|------|----------|---------------|
| llama | gsm8k | 200 | 2000 | 65.7% | 17.9% |
| llama | humaneval | 164 | 1640 | 28.0% | 0.0% |
| mistral7b | gsm8k | 200 | 2000 | 16.0% | 72.2% |
| mistral7b | humaneval | 164 | 1640 | 6.3% | 0.0% |
| mixtral | gsm8k | 201 | 2001 | 26.8% | 62.1% |
| mixtral | humaneval | 162 | 1620 | 4.1% | 0.0% |
| qwen | gsm8k | 200 | 2000 | 23.2% | 4.5% |
| qwen | humaneval | 164 | 1640 | 9.9% | 0.0% |

## Section Summaries

Each section below has its own directory with CSV tables, figures, and a summary.json.

### Focus 1: Metric Correlation

See `focus_01_metric_correlation/summary.json` for machine-readable findings.

_Per-metric correlation with output quality (direction-aware)_

### Focus 2: MoE vs Dense

See `focus_02_moe_vs_dense/summary.json` for machine-readable findings.

_MoE (Mixtral) vs Dense architectural comparison_

### Focus 3: Self-Consistency

See `focus_03_self_consistency/summary.json` for machine-readable findings.

_Within-prompt signal divergence and self-consistency_

### Focus 4: Layer Analysis

See `focus_04_layer_analysis/summary.json` for machine-readable findings.

_Per-layer signal association with correctness_

### Focus 5: Difficulty Profiling

See `focus_05_difficulty/summary.json` for machine-readable findings.

_Prompt-level signal correlation with empirical difficulty_

### Focus 6: Cross-Model Alignment

See `focus_06_cross_model/summary.json` for machine-readable findings.

_Cross-model behavioral alignment_

### Ranking Evaluation

See `ranking/summary.json` for machine-readable findings.

_Best-of-k selection using run-varying CoreVital signals_

### Signal Ablation

See `ablation/summary.json` for machine-readable findings.

_Incremental signal ablation using grouped prompt-level CV_

### Format Failure Analysis

See `format_failure/summary.json` for machine-readable findings.

_Format failure analysis — can CoreVital signals predict output format failures?_

### Risk Calibration

See `risk_calibration/summary.json` for machine-readable findings.

_Risk score calibration and saturation analysis_

### Outcome Signal Profiling

See `outcome_profiling/summary.json` for machine-readable findings.

_Three-way outcome profiling: do correct, incorrect, and format-failure runs have distinct signal signatures, or is failure a single spectrum?_

### Signal Redundancy Mapping

See `signal_redundancy/summary.json` for machine-readable findings.

_Signal redundancy mapping. Clusters correlated signals (|r| > 0.80) into families and identifies the best representative per family._

### Confidence-Correctness Calibration

See `confidence_calibration/summary.json` for machine-readable findings.

_Confidence-correctness calibration. Tests whether model confidence (margin_mean) reliably predicts accuracy, and identifies CoreVital signals that catch confident-but-wrong runs._

### Temperature Effects

See `temperature_effects/summary.json` for machine-readable findings.

_Temperature effect decomposition. Tests whether CoreVital signal distributions and predictive power change between temperatures._

### Difficulty-Stratified Analysis

See `difficulty_stratified/summary.json` for machine-readable findings.

_Difficulty-stratified signal analysis. Tests whether CoreVital signals are equally predictive on easy, medium, and hard questions._

## Artifact Index

- `key_findings.json` — All section findings in one file (AI-oriented)
- `global_manifest.json` — Complete artifact listing
- `focus_01_metric_correlation/` — Focus 1: Metric Correlation
- `focus_02_moe_vs_dense/` — Focus 2: MoE vs Dense
- `focus_03_self_consistency/` — Focus 3: Self-Consistency
- `focus_04_layer_analysis/` — Focus 4: Layer Analysis
- `focus_05_difficulty/` — Focus 5: Difficulty Profiling
- `focus_06_cross_model/` — Focus 6: Cross-Model Alignment
- `ranking/` — Ranking Evaluation
- `ablation/` — Signal Ablation
- `format_failure/` — Format Failure Analysis
- `risk_calibration/` — Risk Calibration
- `outcome_profiling/` — Outcome Signal Profiling
- `signal_redundancy/` — Signal Redundancy Mapping
- `confidence_calibration/` — Confidence-Correctness Calibration
- `temperature_effects/` — Temperature Effects
- `difficulty_stratified/` — Difficulty-Stratified Analysis
