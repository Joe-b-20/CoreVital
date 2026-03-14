# Validation Experiment — Quick Reference

**Location:** `experiment/` directory

**Purpose:** Validate that CoreVital's internal model signals predict task correctness on labeled benchmarks (GSM8K math, HumanEval code).

## Overview

- **14,540 traces** from 4 models (Llama, Qwen, Mistral, Mixtral) × 2 datasets × ~200 prompts × 10 runs (pass@k)
- **11,403 non-format-failure runs** analyzed for correctness prediction
- **15 analysis focus areas** covering correlation, ablation, calibration, confidence, outcomes, temperature
- **Full documentation:** [Validation Report](validation-report.md) (technical), [experiment/README.md](../experiment/README.md) (accessible)

## Why Experiment 2?

**Experiment 1** used greedy decoding on short-answer benchmarks → identical output per prompt → zero variance, zero signal → **failed**.

**Experiment 2** uses **pass@k with sampling** (temp 0.7 + 0.8, k=10 per prompt) → diverse outputs and signals → **succeeded**.

## Key Findings (Honest Summary)

### ✅ What Works

| Finding | Evidence |
|---------|----------|
| **Signals predict correctness** | Ablation AUROC 0.60-0.90 (HistGradientBoosting, grouped 5-fold CV) |
| **Early-window matters for code** | Qwen/HumanEval: 0.73 → 0.85 AUROC at T4 (early-window features) |
| **Confident-but-wrong detectable** | Models are overconfident, but CoreVital catches confident-wrong runs (AUROC up to 0.92) |
| **Distinct outcome signatures** | Correct/incorrect/format-failure have architecture-dependent signal patterns |
| **Temperature-robust** | Signals stable across temp 0.7 vs 0.8 (mean shift 0.028) |
| **Full features usually best** | T6 (104 features) is best in 6/8 cells (but not always: Mistral/GSM8K peaks at T5) |

### ❌ What's Broken

| Problem | Evidence |
|---------|----------|
| **risk_score saturates** | Mistral: 96% at 1.0, Mixtral: 94% at 1.0 |
| **Poor calibration** | ECE 0.24-0.70 before Platt scaling |
| **Weak discrimination** | AUROC 0.48-0.62 (near chance in some cells) |
| **failure_risk is discrete** | 2-5 unique values, AUROC near chance |
| **Pooled model doesn't transfer** | Step 5 model underperforms in 5/8 individual cells |
| **Format failure confounds** | Mistral 72%, Mixtral 62% on GSM8K (missing `####` marker) |

### 📊 Ablation Details (T1-T6)

| Tier | Features | Description | AUROC Range |
|------|----------|-------------|-------------|
| T1 | 1 | entropy_only | 0.49-0.69 |
| T2 | 3 | + confidence_baseline (margin, topk_mass) | 0.50-0.73 |
| T3 | 12 | + prompt_signals | 0.51-0.73 |
| T4 | 30 | + early_window (first 10-50% of generation) | 0.54-0.85 |
| T5 | 44 | + health_signals | 0.56-0.85 |
| T6 | 104 | full_corevital | **0.60-0.90** |

**Key nuance:** Biggest jumps vary by task:
- **HumanEval (code):** T4 (early-window) provides huge gain
- **GSM8K (math):** Mixed; some models benefit from T3 (prompt) or T5 (health)
- **T6 not always best:** Mistral/GSM8K peaks at T5 (0.71), drops to 0.67 at T6

## What It Means for You

### If you're deploying to production:

1. **DO NOT** use `risk_score` or `failure_risk` as production-calibrated predictors without calibration
2. **DO NOT** set SLAs or circuit breakers on raw scores
3. **DO** use CoreVital metrics (entropy, margin, surprisal, attention stats) as **inputs** to your own learned detector
4. **DO** run your own calibration with labeled data from your domain (see [Risk Calibration](risk-calibration.md))

### If you're using CoreVital for research/debugging:

1. **DO** use built-in scores as indicative signals for exploration
2. **DO** trust that internal signals contain information (validated on GSM8K/HumanEval)
3. **DO** expect model- and task-dependent behavior (what works for Qwen/HumanEval ≠ Mistral/GSM8K)

## Recommendations Based on Findings

1. **Implement per-model calibration** — Pooled models don't transfer well
2. **Add confident-but-wrong detector** — Use `compound_density_per_100t`, `l2_norm_slope`, `entropy_range`
3. **Add per-outcome models** — Format-failure vs task-incorrectness have distinct signatures
4. **Leverage early-window features** — For code tasks, first 10-25% of generation is highly predictive
5. **Run your own calibration** — GSM8K/HumanEval results don't generalize to all domains

## Where to Find Outputs

```
experiment/
├── traces/                    # 14,540 JSON trace files
├── results/
│   ├── grades.jsonl           # 14,540 labeled runs
│   ├── features.parquet       # 244 features × 14,540 rows
│   └── prompt_level.parquet   # 1,455 prompt-level aggregates
├── calibration/
│   ├── step1_ece_results.json           # Built-in score evaluation
│   ├── step5_proposed_risk_model.json   # Pooled failure model (AUROC 0.744)
│   └── step6_per_model_evaluation.json  # Per-cell transfer results
└── analysis/                  # 15 focus areas
    ├── ablation/summary.json
    ├── confidence_calibration/summary.json
    ├── outcome_profiling/summary.json
    ├── temperature_effects/summary.json
    └── ... (11 more)
```

## Quick Access

- **Accessible summary:** [experiment/README.md](../experiment/README.md)
- **Technical report:** [docs/validation-report.md](validation-report.md)
- **Risk calibration workflow:** [docs/risk-calibration.md](risk-calibration.md)
- **Metrics interpretation:** [docs/metrics-interpretation.md](metrics-interpretation.md)
- **Validation experiment flow diagram:** [docs/mermaid/validation-experiment-flow.mmd](mermaid/validation-experiment-flow.mmd)

## Citation

```bibtex
@misc{corevital-validation-v2,
  title={CoreVital Validation Experiment: Internal Model Signals Predict Task Correctness},
  author={Joe Bachir},
  year={2026},
  note={Pass@k validation on GSM8K and HumanEval with grouped held-out evaluation. AUROC 0.60-0.90 across models/tasks.}
}
```
