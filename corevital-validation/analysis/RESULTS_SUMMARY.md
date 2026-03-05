# CoreVital Validation Experiment — Results Summary

Generated automatically by analyze.py

## H1: Risk Score Separation

**0/5 cells** passed (p < α with d ≥ 0.3)

| model   | dataset    |   n_correct |   n_incorrect |   mean_correct |   mean_incorrect |    t_stat |    p_value |   cohens_d | significant   |      alpha |
|:--------|:-----------|------------:|--------------:|---------------:|-----------------:|----------:|-----------:|-----------:|:--------------|-----------:|
| llama   | gsm8k      |         201 |            50 |       0.245055 |         0.279804 |  1.18768  | 0.238567   |  0.182582  | False         | 0.00166667 |
| llama   | mmlu       |         628 |           371 |       0.344688 |         0.375783 |  2.64411  | 0.00836114 |  0.174844  | False         | 0.00166667 |
| qwen3b  | gsm8k      |         360 |            79 |       0.413244 |         0.421886 |  3.17205  | 0.00200074 |  0.449235  | False         | 0.00166667 |
| qwen3b  | mmlu       |         627 |           373 |       0.450965 |         0.460947 |  0.748358 | 0.454466   |  0.0487188 | False         | 0.00166667 |
| qwen3b  | truthfulqa |         194 |           306 |       0.44367  |         0.435449 | -0.821137 | 0.411961   | -0.068257  | False         | 0.00166667 |

## H2: Per-Signal Discrimination

**6/60 tests** significant after correction

### Top signals by AUROC:

| model   | dataset    | signal                  |    auroc |   ci_low |   ci_high |   perm_p | significant   |
|:--------|:-----------|:------------------------|---------:|---------:|----------:|---------:|:--------------|
| qwen3b  | gsm8k      | collapsed_rate_mean     | 0.639838 | 0.575206 |  0.703129 |    0     | True          |
| qwen3b  | truthfulqa | collapsed_rate_mean     | 0.622187 | 0.569823 |  0.673003 |    0     | True          |
| llama   | gsm8k      | failure_risk            | 0.584776 | 0.509752 |  0.660599 |    0.015 | False         |
| qwen3b  | truthfulqa | n_compound_signals      | 0.567617 | 0.54227  |  0.593328 |    0     | True          |
| qwen3b  | mmlu       | l2_norm_last_layer_mean | 0.556542 | 0.518782 |  0.595137 |    0     | True          |
| llama   | mmlu       | entropy_slope           | 0.5489   | 0.512146 |  0.586567 |    0.004 | False         |
| llama   | mmlu       | perplexity_mean         | 0.548204 | 0.511274 |  0.587254 |    0.006 | True          |
| llama   | gsm8k      | entropy_slope           | 0.547861 | 0.46303  |  0.634253 |    0.144 | False         |
| llama   | mmlu       | l2_norm_last_layer_mean | 0.543607 | 0.50701  |  0.580679 |    0.01  | True          |
| qwen3b  | mmlu       | collapsed_rate_mean     | 0.535197 | 0.500318 |  0.570958 |    0.027 | False         |
| llama   | gsm8k      | l2_norm_last_layer_mean | 0.528856 | 0.436423 |  0.619787 |    0.275 | False         |
| qwen3b  | truthfulqa | failure_risk            | 0.522994 | 0.479804 |  0.567884 |    0.169 | False         |
| llama   | mmlu       | high_entropy_frac       | 0.521224 | 0.484355 |  0.559288 |    0.147 | False         |
| llama   | mmlu       | topk_mass_mean          | 0.518327 | 0.480838 |  0.55641  |    0.176 | False         |
| llama   | mmlu       | entropy_mean            | 0.515799 | 0.479355 |  0.554406 |    0.221 | False         |
| qwen3b  | mmlu       | failure_risk            | 0.514243 | 0.485479 |  0.54363  |    0.158 | False         |
| llama   | mmlu       | failure_risk            | 0.504861 | 0.476589 |  0.532517 |    0.336 | False         |
| llama   | mmlu       | margin_mean             | 0.502605 | 0.46705  |  0.539655 |    0.454 | False         |
| qwen3b  | mmlu       | entropy_slope           | 0.502102 | 0.464797 |  0.536529 |    0.446 | False         |
| qwen3b  | gsm8k      | n_compound_signals      | 0.5      | 0.5      |  0.5      |    1     | False         |

## H2b: Incremental Value Over Confidence Baseline

**4/5 cells** show Δ ≥ 0.03 AUROC lift

| model   | dataset    |   baseline_auroc_lr |   full_auroc_lr |   full_auroc_gb |    delta_lr |   delta_gb |   n_samples |   n_baseline_features |   n_full_features |
|:--------|:-----------|--------------------:|----------------:|----------------:|------------:|-----------:|------------:|----------------------:|------------------:|
| llama   | gsm8k      |            0.608024 |        0.644256 |        0.624037 |  0.0362317  |  0.0160122 |         251 |                     3 |                30 |
| llama   | mmlu       |            0.560714 |        0.631304 |        0.621388 |  0.07059    |  0.0606748 |         999 |                     3 |                30 |
| qwen3b  | gsm8k      |            0.749271 |        0.798553 |        0.759456 |  0.0492824  |  0.0101852 |         439 |                     3 |                30 |
| qwen3b  | mmlu       |            0.508531 |        0.612235 |        0.580349 |  0.103704   |  0.0718176 |        1000 |                     3 |                30 |
| qwen3b  | truthfulqa |            0.740481 |        0.730575 |        0.756847 | -0.00990583 |  0.0163664 |         500 |                     3 |                30 |

## H3: Early Warning Prediction (GSM8K)

| model   | signal               |    auroc |   ci_low |   ci_high |   perm_p |   null_95th |
|:--------|:---------------------|---------:|---------:|----------:|---------:|------------:|
| llama   | failure_risk         | 0.584776 | 0.509752 |  0.660599 |    0.015 |    0.563284 |
| llama   | early_entropy_mean   | 0.372139 | 0.289549 |  0.458778 |    0.998 |    0.576552 |
| llama   | early_surprisal_mean | 0.390647 | 0.303154 |  0.480808 |    0.992 |    0.576627 |
| llama   | early_margin_mean    | 0.601194 | 0.513018 |  0.690669 |    0.02  |    0.575438 |
| llama   | early_entropy_slope  | 0.439005 | 0.34656  |  0.536452 |    0.905 |    0.573239 |
| qwen3b  | failure_risk         | 0.487412 | 0.434936 |  0.539916 |    0.705 |    0.543601 |
| qwen3b  | early_entropy_mean   | 0.31417  | 0.245681 |  0.382719 |    1     |    0.559265 |
| qwen3b  | early_surprisal_mean | 0.357278 | 0.284753 |  0.423668 |    1     |    0.560415 |
| qwen3b  | early_margin_mean    | 0.636322 | 0.56972  |  0.708389 |    0     |    0.562052 |
| qwen3b  | early_entropy_slope  | 0.446519 | 0.376313 |  0.52003  |    0.927 |    0.555135 |

---

*See analysis/ directory for all visualizations.*
