#!/usr/bin/env python3
"""
CoreVital Validation Experiment - Statistical Analysis

Reads features.parquet and runs all hypothesis tests (H1, H2, H2b, H3),
generates visualizations, and produces a results summary.

Usage:
    python3 analyze.py
    python3 analyze.py --features ~/experiment/results/features.parquet
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for RunPod
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=FutureWarning)

EXPERIMENT_DIR = Path.home() / "experiment"
RESULTS_DIR = EXPERIMENT_DIR / "results"
ANALYSIS_DIR = EXPERIMENT_DIR / "analysis"

# Pre-registered signal categories
PRIMARY_SIGNALS = [
    "entropy_mean", "surprisal_mean", "margin_mean", "topk_mass_mean",
    "risk_score", "failure_risk", "entropy_slope",
]
SECONDARY_SIGNALS = [
    "collapsed_rate_mean", "l2_norm_last_layer_mean", "prompt_surprisal_mean",
    "basin_score_min", "high_entropy_frac", "n_compound_signals", "perplexity_mean",
]
# Baseline features (the "simple confidence" comparator)
BASELINE_FEATURES = ["entropy_mean", "surprisal_mean", "margin_mean"]

# All features for the full model
FULL_FEATURES = PRIMARY_SIGNALS + SECONDARY_SIGNALS + [
    "entropy_std", "surprisal_std", "surprisal_volatility", "margin_slope",
    "l2_norm_slope", "l2_norm_cross_layer_max", "focused_head_mean",
    "nan_detected", "repetition_detected", "mid_layer_anomaly",
    "attention_collapse_detected", "n_warning_signals",
    "max_compound_severity", "basin_score_mean", "layer_transform_mean",
    "prompt_surprisal_mean",
]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def bootstrap_auroc_ci(y_true, y_score, n_boot=2000, ci=0.95, seed=42):
    """Bootstrap confidence interval for AUROC."""
    rng = np.random.RandomState(seed)
    aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    alpha = (1 - ci) / 2
    return np.percentile(aucs, 100 * alpha), np.percentile(aucs, 100 * (1 - alpha))


def permutation_test_auroc(y_true, y_score, n_perm=1000, seed=42):
    """Permutation test: is the real AUROC significantly above chance?"""
    rng = np.random.RandomState(seed)
    real_auc = roc_auc_score(y_true, y_score)
    null_aucs = []
    for _ in range(n_perm):
        shuffled = rng.permutation(y_score)
        if len(np.unique(y_true)) < 2:
            continue
        null_aucs.append(roc_auc_score(y_true, shuffled))
    null_aucs = np.array(null_aucs)
    p_value = np.mean(null_aucs >= real_auc)
    null_95 = np.percentile(null_aucs, 95)
    return real_auc, p_value, null_95


def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-12:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def cross_val_auroc(X, y, model_cls, n_splits=5, seed=42):
    """Stratified k-fold cross-validated AUROC."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        model = model_cls()
        model.fit(X_train_s, y_train)
        probs = model.predict_proba(X_test_s)[:, 1]
        if len(np.unique(y_test)) < 2:
            continue
        aucs.append(roc_auc_score(y_test, probs))
    return aucs


# ---------------------------------------------------------------------------
# H1: Risk Score Separation
# ---------------------------------------------------------------------------

def test_h1(df: pd.DataFrame) -> pd.DataFrame:
    """Test H1: mean risk_score differs between correct and incorrect outputs."""
    print("\n" + "=" * 60)
    print("H1: RISK SCORE SEPARATION")
    print("=" * 60)

    results = []
    models = sorted(df["model"].unique())
    datasets = sorted(df["dataset"].unique())
    n_tests = len(models) * len(datasets)
    alpha = 0.01 / n_tests  # Bonferroni

    for model in models:
        for dataset in datasets:
            cell = df[(df["model"] == model) & (df["dataset"] == dataset)].dropna(subset=["correct", "risk_score"])
            correct = cell[cell["correct"] == True]["risk_score"].values
            incorrect = cell[cell["correct"] == False]["risk_score"].values

            if len(correct) < 10 or len(incorrect) < 10:
                print(f"  {model}/{dataset}: SKIP (too few samples: {len(correct)} correct, {len(incorrect)} incorrect)")
                continue

            t_stat, p_val = ttest_ind(incorrect, correct, equal_var=False)
            d = cohens_d(incorrect, correct)
            significant = p_val < alpha

            result = {
                "model": model, "dataset": dataset,
                "n_correct": len(correct), "n_incorrect": len(incorrect),
                "mean_correct": np.mean(correct), "mean_incorrect": np.mean(incorrect),
                "t_stat": t_stat, "p_value": p_val,
                "cohens_d": d, "significant": significant,
                "alpha": alpha,
            }
            results.append(result)

            icon = "✓" if significant and d >= 0.3 else "~" if significant else "✗"
            print(f"  {icon} {model}/{dataset}: d={d:.3f}, p={p_val:.2e}, "
                  f"mean_risk: correct={np.mean(correct):.3f}, incorrect={np.mean(incorrect):.3f}")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# H2: Per-Signal Discrimination
# ---------------------------------------------------------------------------

def test_h2(df: pd.DataFrame) -> pd.DataFrame:
    """Test H2: individual signal AUROC for classifying correct vs incorrect."""
    print("\n" + "=" * 60)
    print("H2: PER-SIGNAL DISCRIMINATION (AUROC)")
    print("=" * 60)

    all_signals = PRIMARY_SIGNALS + SECONDARY_SIGNALS
    # Signals where lower = incorrect (need to negate for AUROC)
    negate_signals = {"margin_mean", "topk_mass_mean"}

    results = []
    models = sorted(df["model"].unique())
    datasets = sorted(df["dataset"].unique())

    for model in models:
        for dataset in datasets:
            cell = df[(df["model"] == model) & (df["dataset"] == dataset)].dropna(subset=["correct"])
            y = cell["correct"].astype(int).values

            if len(np.unique(y)) < 2 or len(y) < 50:
                continue

            for signal in all_signals:
                if signal not in cell.columns:
                    continue
                vals = cell[signal].values
                mask = ~np.isnan(vals.astype(float))
                if mask.sum() < 50:
                    continue

                y_masked = y[mask]
                scores = vals[mask].astype(float)

                if len(np.unique(y_masked)) < 2:
                    continue

                # Negate if lower values mean incorrect
                if signal in negate_signals:
                    scores = -scores

                real_auc, perm_p, null_95 = permutation_test_auroc(y_masked, scores)
                ci_lo, ci_hi = bootstrap_auroc_ci(y_masked, scores)

                is_primary = signal in PRIMARY_SIGNALS
                results.append({
                    "model": model, "dataset": dataset, "signal": signal,
                    "auroc": real_auc, "ci_low": ci_lo, "ci_high": ci_hi,
                    "perm_p": perm_p, "null_95th": null_95,
                    "n_samples": mask.sum(),
                    "is_primary": is_primary,
                })

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print("  No results (not enough data).")
        return results_df

    # FDR correction
    primary_mask = results_df["is_primary"]
    # Primary signals: Bonferroni
    if primary_mask.sum() > 0:
        _, p_corrected, _, _ = multipletests(
            results_df.loc[primary_mask, "perm_p"], alpha=0.01, method="bonferroni"
        )
        results_df.loc[primary_mask, "p_corrected"] = p_corrected
        results_df.loc[primary_mask, "correction"] = "bonferroni"

    # Secondary signals: BH-FDR
    sec_mask = ~primary_mask
    if sec_mask.sum() > 0:
        rejected, p_corrected, _, _ = multipletests(
            results_df.loc[sec_mask, "perm_p"], alpha=0.05, method="fdr_bh"
        )
        results_df.loc[sec_mask, "p_corrected"] = p_corrected
        results_df.loc[sec_mask, "correction"] = "fdr_bh"

    results_df["significant"] = results_df["p_corrected"] < 0.05

    # Print top signals per model
    for model in models:
        model_res = results_df[results_df["model"] == model].sort_values("auroc", ascending=False)
        print(f"\n  {model} — Top 10 signals by AUROC:")
        for _, row in model_res.head(10).iterrows():
            icon = "✓" if row["significant"] else " "
            print(f"    {icon} {row['signal']:30s} {row['dataset']:12s} "
                  f"AUROC={row['auroc']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}] "
                  f"p={row['perm_p']:.4f}")

    return results_df


# ---------------------------------------------------------------------------
# H2b: Incremental Value Over Confidence Baseline
# ---------------------------------------------------------------------------

def test_h2b(df: pd.DataFrame) -> pd.DataFrame:
    """Test H2b: full CoreVital features add AUROC over simple confidence baseline."""
    print("\n" + "=" * 60)
    print("H2b: INCREMENTAL VALUE OVER CONFIDENCE BASELINE")
    print("=" * 60)

    results = []
    models = sorted(df["model"].unique())
    datasets = sorted(df["dataset"].unique())

    def make_lr():
        return LogisticRegression(penalty="l2", class_weight="balanced", max_iter=1000, C=1.0)

    def make_gb():
        return HistGradientBoostingClassifier(max_iter=100, max_depth=4, class_weight="balanced")

    for model in models:
        for dataset in datasets:
            cell = df[(df["model"] == model) & (df["dataset"] == dataset)].dropna(subset=["correct"])
            y = cell["correct"].astype(int).values

            if len(np.unique(y)) < 2 or len(y) < 100:
                print(f"  {model}/{dataset}: SKIP (n={len(y)}, need ≥100)")
                continue

            # Baseline features
            baseline_cols = [c for c in BASELINE_FEATURES if c in cell.columns]
            X_base = cell[baseline_cols].fillna(0).values

            # Full features
            full_cols = [c for c in FULL_FEATURES if c in cell.columns]
            X_full = cell[full_cols].fillna(0).values

            # Cross-validated AUROC for each
            try:
                base_aucs_lr = cross_val_auroc(X_base, y, make_lr)
                full_aucs_lr = cross_val_auroc(X_full, y, make_lr)
                full_aucs_gb = cross_val_auroc(X_full, y, make_gb)

                delta_lr = np.mean(full_aucs_lr) - np.mean(base_aucs_lr)
                delta_gb = np.mean(full_aucs_gb) - np.mean(base_aucs_lr)

                result = {
                    "model": model, "dataset": dataset,
                    "baseline_auroc_lr": np.mean(base_aucs_lr),
                    "full_auroc_lr": np.mean(full_aucs_lr),
                    "full_auroc_gb": np.mean(full_aucs_gb),
                    "delta_lr": delta_lr,
                    "delta_gb": delta_gb,
                    "n_samples": len(y),
                    "n_baseline_features": len(baseline_cols),
                    "n_full_features": len(full_cols),
                }
                results.append(result)

                icon = "✓" if delta_lr >= 0.03 else "~" if delta_lr > 0 else "✗"
                print(f"  {icon} {model}/{dataset}: "
                      f"baseline={np.mean(base_aucs_lr):.3f}, "
                      f"full_LR={np.mean(full_aucs_lr):.3f} (Δ={delta_lr:+.3f}), "
                      f"full_GB={np.mean(full_aucs_gb):.3f} (Δ={delta_gb:+.3f})")

            except Exception as e:
                print(f"  {model}/{dataset}: ERROR — {e}")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# H3: Early Warning Prediction (GSM8K only)
# ---------------------------------------------------------------------------

def test_h3(df: pd.DataFrame) -> pd.DataFrame:
    """Test H3: early-warning features at 50% generation predict final correctness."""
    print("\n" + "=" * 60)
    print("H3: EARLY WARNING PREDICTION (GSM8K ONLY)")
    print("=" * 60)

    gsm = df[df["dataset"] == "gsm8k"].dropna(subset=["correct"])
    results = []
    models = sorted(gsm["model"].unique())

    early_features = ["early_entropy_mean", "early_surprisal_mean", "early_margin_mean", "early_entropy_slope"]

    for model in models:
        cell = gsm[gsm["model"] == model]
        y = cell["correct"].astype(int).values

        if len(np.unique(y)) < 2 or len(y) < 50:
            print(f"  {model}: SKIP (n={len(y)})")
            continue

        # Test failure_risk (the main early warning score)
        if "failure_risk" in cell.columns:
            fr = cell["failure_risk"].fillna(0).values
            if np.std(fr) > 1e-9:
                auc, perm_p, null_95 = permutation_test_auroc(y, fr)
                ci_lo, ci_hi = bootstrap_auroc_ci(y, fr)
                results.append({
                    "model": model, "signal": "failure_risk",
                    "auroc": auc, "ci_low": ci_lo, "ci_high": ci_hi,
                    "perm_p": perm_p, "null_95th": null_95,
                })
                icon = "✓" if auc > 0.55 and perm_p < 0.05 else "✗"
                print(f"  {icon} {model}/failure_risk: AUROC={auc:.3f} [{ci_lo:.3f}, {ci_hi:.3f}], p={perm_p:.4f}")

        # Test individual early features
        for signal in early_features:
            if signal not in cell.columns:
                continue
            vals = cell[signal].fillna(0).values
            if np.std(vals) < 1e-9:
                continue
            auc, perm_p, null_95 = permutation_test_auroc(y, vals)
            ci_lo, ci_hi = bootstrap_auroc_ci(y, vals)
            results.append({
                "model": model, "signal": signal,
                "auroc": auc, "ci_low": ci_lo, "ci_high": ci_hi,
                "perm_p": perm_p, "null_95th": null_95,
            })
            icon = "✓" if auc > 0.55 else " "
            print(f"  {icon} {model}/{signal}: AUROC={auc:.3f}")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_violin_risk(df: pd.DataFrame, out_dir: Path):
    """Violin plots: risk_score for correct vs incorrect, per model × dataset."""
    print("\n  Generating violin plots...")
    plot_df = df.dropna(subset=["correct", "risk_score"]).copy()
    plot_df["Outcome"] = plot_df["correct"].map({True: "Correct", False: "Incorrect"})

    datasets = sorted(plot_df["dataset"].unique())
    for dataset in datasets:
        subset = plot_df[plot_df["dataset"] == dataset]
        if len(subset) < 20:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.violinplot(data=subset, x="model", y="risk_score", hue="Outcome",
                       split=True, inner="quart", palette={"Correct": "#4CAF50", "Incorrect": "#F44336"},
                       ax=ax)
        ax.set_title(f"Risk Score Distribution: Correct vs Incorrect — {dataset.upper()}", fontsize=13)
        ax.set_ylabel("Risk Score")
        ax.set_xlabel("Model")
        fig.tight_layout()
        fig.savefig(out_dir / f"violin_risk_{dataset}.png", dpi=150)
        plt.close(fig)


def plot_auroc_heatmap(h2_results: pd.DataFrame, out_dir: Path):
    """Heatmap: models × signals, cell = AUROC value."""
    print("  Generating AUROC heatmap...")
    if len(h2_results) == 0:
        return

    # Pivot: rows = signals, columns = model/dataset
    h2_results["cell"] = h2_results["model"] + "/" + h2_results["dataset"]
    pivot = h2_results.pivot_table(index="signal", columns="cell", values="auroc")

    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns) * 1.5), max(6, len(pivot) * 0.4)))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0.45, vmax=0.75,
                center=0.5, ax=ax, linewidths=0.5)
    ax.set_title("Signal AUROC: Correct vs Incorrect Classification", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "auroc_heatmap.png", dpi=150)
    plt.close(fig)


def plot_roc_curves(df: pd.DataFrame, h2_results: pd.DataFrame, out_dir: Path):
    """ROC curves for top 3 signals per model."""
    print("  Generating ROC curves...")
    models = sorted(df["model"].unique())
    negate_signals = {"margin_mean", "topk_mass_mean"}

    for model in models:
        model_res = h2_results[h2_results["model"] == model].sort_values("auroc", ascending=False)
        top3 = model_res.head(3)
        if len(top3) == 0:
            continue

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Chance")

        for _, row in top3.iterrows():
            signal = row["signal"]
            dataset = row["dataset"]
            cell = df[(df["model"] == model) & (df["dataset"] == dataset)].dropna(subset=["correct", signal])
            y = cell["correct"].astype(int).values
            scores = cell[signal].astype(float).values
            if signal in negate_signals:
                scores = -scores

            if len(np.unique(y)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y, scores)
            ax.plot(fpr, tpr, label=f"{signal} ({dataset}) AUROC={row['auroc']:.3f}")

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curves — {model}", fontsize=13)
        ax.legend(loc="lower right", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / f"roc_{model}.png", dpi=150)
        plt.close(fig)


def plot_signal_trajectories(df: pd.DataFrame, out_dir: Path):
    """Mean entropy/surprisal/margin over generation steps: correct vs incorrect (GSM8K)."""
    print("  Generating signal trajectory plots...")
    # This requires per-step data which we don't store in features.parquet.
    # Instead we plot the aggregate features as a proxy.
    # A future version could read raw traces for this visualization.
    gsm = df[df["dataset"] == "gsm8k"].dropna(subset=["correct"])
    if len(gsm) < 50:
        return

    signals_to_plot = ["entropy_mean", "surprisal_mean", "margin_mean",
                       "entropy_slope", "risk_score", "failure_risk"]
    plot_data = gsm[["model", "correct"] + [s for s in signals_to_plot if s in gsm.columns]]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, signal in enumerate(signals_to_plot):
        if signal not in plot_data.columns or i >= len(axes):
            continue
        ax = axes[i]
        subset = plot_data.dropna(subset=[signal])
        sns.boxplot(data=subset, x="model", y=signal, hue="correct",
                    palette={True: "#4CAF50", False: "#F44336"}, ax=ax)
        ax.set_title(signal, fontsize=11)
        ax.legend([], [], frameon=False)  # Remove redundant legends

    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, ["Correct", "Incorrect"], loc="upper right", fontsize=10)
    fig.suptitle("GSM8K: Signal Distributions by Correctness", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "gsm8k_signal_distributions.png", dpi=150)
    plt.close(fig)


def plot_correlation_matrix(df: pd.DataFrame, out_dir: Path):
    """Correlation matrix of all numeric features."""
    print("  Generating correlation matrix...")
    numeric_cols = [c for c in PRIMARY_SIGNALS + SECONDARY_SIGNALS + ["risk_score", "failure_risk"]
                    if c in df.columns]
    corr_df = df[numeric_cols].dropna(axis=1, how="all")
    if corr_df.shape[1] < 3:
        return

    corr = corr_df.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                vmin=-1, vmax=1, ax=ax, square=True, linewidths=0.5)
    ax.set_title("Feature Correlation Matrix", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "correlation_matrix.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Results summary
# ---------------------------------------------------------------------------

def write_summary(h1_results, h2_results, h2b_results, h3_results, out_dir: Path):
    """Write a human-readable results summary."""
    path = out_dir / "RESULTS_SUMMARY.md"
    with open(path, "w") as f:
        f.write("# CoreVital Validation Experiment — Results Summary\n\n")
        f.write(f"Generated automatically by analyze.py\n\n")

        # H1
        f.write("## H1: Risk Score Separation\n\n")
        if len(h1_results) > 0:
            passed = h1_results[(h1_results["significant"]) & (h1_results["cohens_d"] >= 0.3)]
            f.write(f"**{len(passed)}/{len(h1_results)} cells** passed (p < α with d ≥ 0.3)\n\n")
            f.write(h1_results.to_markdown(index=False))
        else:
            f.write("No results.\n")

        # H2
        f.write("\n\n## H2: Per-Signal Discrimination\n\n")
        if len(h2_results) > 0:
            sig = h2_results[h2_results["significant"]]
            f.write(f"**{len(sig)}/{len(h2_results)} tests** significant after correction\n\n")
            f.write("### Top signals by AUROC:\n\n")
            top = h2_results.sort_values("auroc", ascending=False).head(20)
            f.write(top[["model", "dataset", "signal", "auroc", "ci_low", "ci_high", "perm_p", "significant"]].to_markdown(index=False))
        else:
            f.write("No results.\n")

        # H2b
        f.write("\n\n## H2b: Incremental Value Over Confidence Baseline\n\n")
        if len(h2b_results) > 0:
            passed = h2b_results[h2b_results["delta_lr"] >= 0.03]
            f.write(f"**{len(passed)}/{len(h2b_results)} cells** show Δ ≥ 0.03 AUROC lift\n\n")
            f.write(h2b_results.to_markdown(index=False))
        else:
            f.write("No results.\n")

        # H3
        f.write("\n\n## H3: Early Warning Prediction (GSM8K)\n\n")
        if len(h3_results) > 0:
            f.write(h3_results.to_markdown(index=False))
        else:
            f.write("No results.\n")

        f.write("\n\n---\n\n*See analysis/ directory for all visualizations.*\n")

    print(f"\n  Summary written to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CoreVital Validation Experiment Analysis")
    parser.add_argument("--features", type=Path, default=RESULTS_DIR / "features.parquet")
    args = parser.parse_args()

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading features from {args.features}...")
    df = pd.read_parquet(args.features)
    print(f"  {len(df)} rows, {len(df.columns)} columns")
    print(f"  Models: {sorted(df['model'].unique())}")
    print(f"  Datasets: {sorted(df['dataset'].unique())}")

    # Filter out format failures for main analysis
    df_clean = df[df["format_failure"] != True].copy()
    print(f"  After removing format failures: {len(df_clean)} rows")

    # Run all tests
    h1_results = test_h1(df_clean)
    h2_results = test_h2(df_clean)
    h2b_results = test_h2b(df_clean)
    h3_results = test_h3(df_clean)

    # Save raw results
    if len(h1_results) > 0:
        h1_results.to_csv(ANALYSIS_DIR / "h1_results.csv", index=False)
    if len(h2_results) > 0:
        h2_results.to_csv(ANALYSIS_DIR / "h2_results.csv", index=False)
    if len(h2b_results) > 0:
        h2b_results.to_csv(ANALYSIS_DIR / "h2b_results.csv", index=False)
    if len(h3_results) > 0:
        h3_results.to_csv(ANALYSIS_DIR / "h3_results.csv", index=False)

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_violin_risk(df_clean, ANALYSIS_DIR)
    plot_auroc_heatmap(h2_results, ANALYSIS_DIR)
    plot_roc_curves(df_clean, h2_results, ANALYSIS_DIR)
    plot_signal_trajectories(df_clean, ANALYSIS_DIR)
    plot_correlation_matrix(df_clean, ANALYSIS_DIR)

    # Write summary
    write_summary(h1_results, h2_results, h2b_results, h3_results, ANALYSIS_DIR)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  Results:        {ANALYSIS_DIR}")
    print(f"  Summary:        {ANALYSIS_DIR / 'RESULTS_SUMMARY.md'}")
    print(f"  Visualizations: {ANALYSIS_DIR}/*.png")


if __name__ == "__main__":
    main()