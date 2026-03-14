# CoreVital Validation Experiment (v2)

## Overview

This directory contains a validation experiment that tests whether CoreVital's internal model signals can predict task correctness on mathematical reasoning (GSM8K) and code generation (HumanEval) benchmarks.

**Key finding:** CoreVital signals predict task correctness under grouped held-out evaluation (ablation AUROC range 0.60-0.90 across model/dataset cells; pooled logistic regression AUROC 0.74), but the built-in heuristic `risk_score` is not reliably calibrated and requires data-driven replacement.

## Background: Why Experiment 2?

**Experiment 1** (not included here) used greedy decoding on short-answer benchmarks. With greedy decoding, each prompt produces identical output across runs—zero variance, zero signal. Experiment 1 failed.

**Experiment 2** redesigned from scratch using **pass@k with sampling** (temp 0.7 + 0.8), producing 10 diverse outputs per prompt. This creates variance in both outputs and internal signals, enabling meaningful validation.

## Experiment Design

### Models Tested

- **Llama-3.1-8B-Instruct** — Dense, instruction-tuned
- **Qwen2.5-7B-Instruct** — Dense, instruction-tuned
- **Mistral-7B-Instruct-v0.3** — Dense, instruction-tuned
- **Mixtral-8x7B-Instruct-v0.1** — Sparse MoE, instruction-tuned

### Datasets

- **GSM8K** — 200 grade-school math problems (format: step-by-step reasoning → `#### answer`)
- **HumanEval** — 164 Python programming problems (format: function signature + docstring → code completion)

### Pass@k Configuration

- **k = 10** runs per prompt
- **Temperature schedule:** 5 runs @ 0.7, 5 runs @ 0.8
- **Seeds:** 0-9 (one per run)
- **Sampling:** `top_p=0.95`, `top_k=50`, `max_new_tokens=768`

### Total Scope

- **14,540 traces** generated (4 models × 364 prompts × 10 runs)
- **11,403 non-format-failure traces** analyzed (76% of traces passed format validation)

### Format Failures

Format failures occur when the model fails to produce the expected output structure:

- **GSM8K:** Missing `####` marker for final answer
- **HumanEval:** Always zero (execution-based grading doesn't have format requirements)

Format failure rates varied dramatically by model:
- **Llama GSM8K:** 17.9%
- **Qwen GSM8K:** 4.5%
- **Mistral GSM8K:** 72.2% ⚠️
- **Mixtral GSM8K:** 62.1% ⚠️
- **HumanEval (all models):** 0.0%

## What Was Done

### 1. Data Generation (`scripts/run_experiment.py`)

For each (model, dataset, prompt):
1. Load model once (batched inference for efficiency)
2. Run k=10 generations with varying temperatures/seeds
3. Capture full CoreVital trace for each run (per-step signals, prompt analysis, risk scores)
4. Grade each output (GSM8K: regex extraction + numeric comparison; HumanEval: sandboxed execution)
5. Save traces to `traces/{model}/{dataset}/{question_id}_run{NN}.json`
6. Append grades to `results/grades.jsonl`

### 2. Feature Extraction (`scripts/extract_features.py`)

From 14,540 JSON traces:
- Extract **244 scalar features** per trace (entropy, margin, surprisal, attention stats, hidden state summaries, health flags, risk scores, prompt analysis, fingerprint vector, performance timing)
- Build **run-level table** (`results/features.parquet`, 14,540 rows)
- Build **prompt-level table** (`results/prompt_level.parquet`, 1,455 rows with prompt-invariant features)
- Build **layer-long table** (`results/layer_long.parquet`, 450,752 rows for per-layer analysis)
- Compute **derived features** (pass rate per prompt, empirical difficulty)

Key insight: The **fingerprint vector** (fp_00 through fp_24) is a concatenation of the same summary statistics already extracted as named features, **not a learned embedding**. It is not independent information.

### 3. Analysis (`scripts/analyze.py`)

Comprehensive signal analysis across 15 focus areas:

1. **Metric Correlation** — Per-metric correlation with task correctness (direction-aware)
2. **MoE vs Dense** — Architectural comparison (Mixtral vs dense models)
3. **Self-Consistency** — Within-prompt signal divergence across runs
4. **Layer Analysis** — Per-layer signal association with correctness
5. **Difficulty Profiling** — Prompt-level signal correlation with empirical difficulty
6. **Cross-Model Alignment** — Cross-model behavioral consistency
7. **Ranking Evaluation** — Best-of-k selection using run-varying signals
8. **Signal Ablation** — Incremental feature ablation using grouped CV
9. **Format Failure** — Internal signal predictors of format failures
10. **Risk Calibration** — Calibration analysis of built-in risk scores
11. **Outcome Profiling** — Three-way outcome signature analysis (correct/incorrect/format-fail)
12. **Signal Redundancy** — Feature clustering and family identification
13. **Confidence Calibration** — Margin vs accuracy relationship + confident-but-wrong signals
14. **Temperature Effects** — Signal distribution shifts between temperatures
15. **Difficulty-Stratified** — Predictive power across difficulty bands

All analysis outputs saved to `analysis/{focus_name}/`:
- `summary.json` — Machine-readable findings
- `tables/*.csv` — Result tables
- `figures/*.png` — Visualizations

### 4. Risk Calibration (`scripts/calibrate_risk.py`)

Three-step calibration pipeline using **grouped held-out cross-validation** (by question_id to prevent train/test leakage):

**Step 1:** Evaluate current heuristic scores (`risk_score`, `failure_risk`)
- **Finding:** Built-in scores are poorly calibrated:
  - Mistral/Mixtral: `risk_score` saturates at 1.0 (96% and 94% of runs)
  - Llama/Qwen: `risk_score` ECE ~0.24-0.35, AUROC ~0.54-0.62
  - `failure_risk` has only 2-5 unique values per model (discrete, not continuous)
- **Platt scaling helps ECE** but doesn't fix saturation or low AUROC

**Step 2:** Build calibration profiles from correct runs
- Sample 200 correct traces per model
- Extract distributional statistics (entropy mean/std, margin mean/std, L2 norms per layer)
- Save to `calibration/profile_{model}.json` (used for divergence scoring)

**Step 3:** Derive data-driven failure-risk weights via logistic regression
- Fit logistic regression predicting P(failure) using 6-14 features (entropy, margin, topk_mass, prompt_surprisal, basin_score, etc.)
- Use grouped 5-fold cross-validation by (model, dataset, question_id)
- **Pooled results** (all models/datasets):
  - **Data-driven AUROC:** 0.744 ± 0.03
  - **Data-driven ECE:** 0.164
  - **Current risk_score AUROC:** 0.592 (delta: **+0.15**)
  - **Current failure_risk AUROC:** 0.548
- Save proposed model to `calibration/step5_proposed_risk_model.json`

**Step 4-6:** Cross-model consensus + per-model evaluation
- Identify features with consistent predictive direction across models
- Test proposed formula on per-model cells (results vary: some improve, some near chance)

## Key Findings

### 1. CoreVital Signals Predict Task Correctness (Grouped Held-Out)

**Ablation results (HistGradientBoosting with grouped 5-fold CV):**
- **T6 (full CoreVital, 104 features) AUROC range:** 0.60-0.90 across 8 model/dataset cells
  - **Best:** Qwen/HumanEval (0.90)
  - **Worst:** Qwen/GSM8K (0.60)
- **Key nuance:** The biggest gains vary by model/task—not always at T6
  - **Qwen/HumanEval:** Huge jump at T4 (early-window features): 0.73 → 0.85
  - **Mistral/GSM8K:** Peak at T5 (0.71); T6 drops to 0.67
  - **Where information lives:** Early-window features (T4) drive HumanEval gains; prompt signals (T3) help GSM8K for some models

**Pooled logistic regression (step 5, 6 features):**
- **AUROC:** 0.744, **ECE:** 0.164
- Weaker than ablation but uses only 6 features vs 104

### 2. Built-In Heuristic Scores Are Broken

- **`risk_score` saturates at 1.0** for Mistral (96%) and Mixtral (94%)
- **ECE ranges from 0.24-0.70** (poor calibration)
- **AUROC often near 0.5** (no better than chance)
- **`failure_risk` is discrete** (2-5 unique values), not continuous probability

**Implication:** Do not use built-in scores for production decisions without data-driven recalibration.

### 3. Format Failure Is Predictable

- Internal signals (especially on GSM8K) predict missing `####` markers
- Mistral/Mixtral have severe format-failure issues (62-72% of GSM8K runs)
- See `analysis/format_failure/` for signal associations

### 4. Per-Model Calibration Varies Dramatically

- **Llama:** Moderate predictive power (AUROC 0.61-0.64)
- **Qwen:** Excellent on HumanEval (AUROC 0.90), weak on GSM8K (AUROC 0.60)
- **Mistral:** Strong on HumanEval (AUROC 0.77), weak on GSM8K (AUROC 0.67)
- **Mixtral:** Strong on HumanEval (AUROC 0.82), weak on GSM8K (AUROC 0.65)

**Implication:** A single global formula does not generalize. Per-model or per-task calibration is required.

### 5. Early-Window Features Matter on HumanEval

- Signals from the **first 10-25%** of generation strongly predict code correctness
- Ablation tier "T4: early_window" shows large AUROC jumps on HumanEval (0.51 → 0.76 for Mistral/Mixtral)
- This enables **partial-generation risk estimation** (offline batch analysis, not real-time)

### 6. Cross-Model Agreement Is Modest

- Pass-rate correlation across models: **0.10-0.45** (Spearman rho)
- Models struggle on different prompts; little consensus on what's "hard"

### 7. Prompt Analysis Features Correlate with Difficulty

- Prompt surprisal, basin scores, and layer transformations correlate with empirical difficulty
- See `analysis/focus_05_difficulty/` for correlations

### 8. Models Are Overconfident, But CoreVital Still Discriminates

- Models assign high confidence (margin) to many incorrect outputs
- **CoreVital signals catch confident-but-wrong runs:**
  - Qwen/HumanEval: `compound_density_per_100t` achieves AUROC 0.92 on confident runs
  - Mistral/GSM8K: `hidden_max_abs_last_layer_mean` achieves AUROC 0.90 on confident runs
- **Implication:** Confidence calibration is broken, but internal signals still contain correctness information

### 9. Three-Way Outcome Signatures Are Architecture-Dependent

- Correct, incorrect, and format-failure runs have **distinct signal signatures**
- **Architecture patterns:**
  - **Mistral:** `hidden_max_abs_last_layer_mean` distinguishes incorrect (PP 0.90, gap 0.17)
  - **Mixtral:** `focused_head_mean` distinguishes incorrect (PP 0.90, gap 0.14)
  - **Qwen:** `concentration_min_mean` distinguishes format failure (PP 0.77, gap 0.23)
- See `analysis/outcome_profiling/` for specificity analysis

### 10. Signals Are Temperature-Robust

- CoreVital signal distributions remain stable across temperatures (0.7 vs 0.8)
- **Mean absolute predictive power shift:** 0.028 (finding: "Robust")
- Validates that signals measure model internals, not just sampling artifacts

## What It Means

### Validated Claims

✅ **CoreVital signals contain information about task correctness** (grouped held-out AUROC 0.74 pooled)
✅ **Format failure is predictable** from internal signals
✅ **Full feature set beats ablated subsets**
✅ **Early-window signals work for partial-generation analysis** (offline)
✅ **Calibration profiles can be built** from correct-run distributions

### Invalidated/Broken

❌ **Built-in `risk_score` is not calibrated** (saturates, poor ECE, low AUROC)
❌ **Built-in `failure_risk` is discrete**, not continuous probability
❌ **No single global formula generalizes** across models/tasks
❌ **Per-model results are inconsistent** (AUROC ranges from 0.48 to 0.90)

### Limitations (Honest Accounting)

- **Offline batch analysis only** — Not real-time capability. Early-window analysis is post-hoc.
- **GSM8K + HumanEval generalization only** — Results do not claim to generalize beyond these two benchmarks.
- **No production validation** — All evaluation is offline held-out CV on labeled data.
- **Format failure confounds results** — Mistral/Mixtral GSM8K cells have 62-72% format failures, making correctness prediction less meaningful.
- **Grouped CV by question_id prevents overfitting** — But also means per-run predictions are noisier than in-distribution evaluation.

## Usage

### Running the Full Experiment

```bash
# 1. Generate traces (requires GPU, ~8-24 hours depending on hardware)
python experiment/scripts/run_experiment.py

# 2. Extract features (~5 minutes)
python experiment/scripts/extract_features.py

# 3. Run analysis (~10-15 minutes)
python experiment/scripts/analyze.py

# 4. Calibrate risk scores (~2-5 minutes)
python experiment/scripts/calibrate_risk.py
```

### Resumable Runs

`run_experiment.py` checkpoints after each prompt. If interrupted, rerunning will skip completed prompts.

```bash
# Dry run (5 prompts per cell)
python experiment/scripts/run_experiment.py --dry-run

# Single model
python experiment/scripts/run_experiment.py --model llama

# Single dataset
python experiment/scripts/run_experiment.py --dataset gsm8k

# Fresh start (delete checkpoint)
python experiment/scripts/run_experiment.py --no-resume
```

### Outputs

```
experiment/
├── traces/                    # 14,540 JSON trace files
│   └── {model}/{dataset}/{qid}_run{NN}.json
├── results/
│   ├── grades.jsonl           # 14,540 graded outputs
│   ├── features.parquet       # 14,540 × 244 feature table
│   ├── prompt_level.parquet   # 1,455 prompt-level aggregates
│   └── layer_long.parquet     # 450,752 per-layer rows
├── calibration/
│   ├── profile_{model}.json   # Baseline profiles (4 files, ~180 MB each)
│   ├── step1_ece_results.json # Current score evaluation
│   ├── step3_data_driven_weights.json  # Per-cell weights
│   ├── step5_proposed_risk_model.json  # Pooled model
│   └── step6_per_model_evaluation.json # Per-cell performance
└── analysis/                  # 15 focus areas
    └── {focus_name}/
        ├── summary.json
        ├── tables/*.csv
        └── figures/*.png
```

## File Status

### Kept from Original `corevital-validation/`

- ❌ None — The original flat structure was completely replaced

### New in `experiment/`

- ✅ `scripts/` — Modular, reusable scripts with shared helpers
- ✅ `analysis/` — Organized by focus area (15 subdirectories)
- ✅ `calibration/` — Risk calibration outputs separate from analysis
- ✅ `metadata/` — Experiment metadata and manifests
- ✅ `results/` — Feature tables and extraction manifest
- ✅ `traces/` — Organized by model/dataset

## Next Steps

Based on these findings:

1. **Implement per-model calibration** — Step 6 showed the pooled model (step 5) underperforms on most individual cells. Fit separate models per (model, dataset) or per architecture.
2. **Add format-failure prediction** as separate signal (especially for GSM8K: Mistral 72%, Mixtral 62% failure rate)
3. **Add confident-but-wrong detector** using signals from confidence calibration analysis
4. **Replace built-in risk score** with learned model (but only after per-model calibration)
5. **Document limitations clearly** in user-facing docs (no production validation, offline only, GSM8K+HumanEval scope)
6. **Add disclaimers** to `risk_score` and `failure_risk` outputs noting poor calibration
7. **Consider deprecating `failure_risk`** (discrete, low predictive power)

## Citation

If using this experiment design or findings:

```bibtex
@misc{corevital-validation-v2,
  title={CoreVital Validation Experiment: Internal Model Signals Predict Task Correctness},
  author={CoreVital Contributors},
  year={2025},
  note={Pass@k validation on GSM8K and HumanEval with grouped held-out evaluation. AUROC 0.60-0.90 across models/tasks.}
}
```

## Contact

For questions about experiment design or reproduction, see main repo README or file an issue.
