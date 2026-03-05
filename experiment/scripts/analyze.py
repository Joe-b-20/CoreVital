#!/usr/bin/env python3
"""
CoreVital Validation Experiment — Analysis

6 Focus Areas:
  1. Per-metric correlation with output quality
  2. MoE vs Dense architectural comparison
  3. Self-consistency mapping (within-prompt signal divergence)
  4. Layer-level signal analysis
  5. Dataset difficulty profiling (prompt signals → difficulty)
  6. Cross-model behavioral alignment

Plus:
  - Ranking evaluation: can CoreVital pick the best run from k?
  - Signal ablation: entropy-only vs baseline vs full CoreVital

Usage:
    python3 analyze.py
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pointbiserialr, spearmanr, ttest_rel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

EXPERIMENT_DIR = Path.home() / "experiment"
RESULTS_DIR = EXPERIMENT_DIR / "results"
ANALYSIS_DIR = EXPERIMENT_DIR / "analysis"

# Signal groups
CORE_SIGNALS = [
    "entropy_mean", "entropy_std", "entropy_slope", "entropy_max",
    "surprisal_mean", "surprisal_std", "surprisal_volatility",
    "margin_mean", "margin_slope",
    "topk_mass_mean", "perplexity_mean",
]
HIDDEN_SIGNALS = ["l2_norm_last_layer_mean", "l2_norm_slope", "l2_norm_cross_layer_max"]
ATTN_SIGNALS = ["collapsed_rate_mean", "focused_head_mean"]
HEALTH_SIGNALS = [
    "risk_score", "failure_risk", "high_entropy_frac",
    "n_compound_signals", "max_compound_severity",
    "nan_detected", "repetition_detected", "mid_layer_anomaly", "attention_collapse_detected",
]
PROMPT_SIGNALS = [
    "prompt_surprisal_mean", "prompt_surprisal_max", "prompt_surprisal_std",
    "basin_score_min", "basin_score_mean", "basin_score_std",
    "layer_transform_mean", "layer_transform_std", "layer_transform_max",
    "n_sparse_heads",
]
COMPOUND_SIGNALS = [
    "cs_context_loss", "cs_confident_confusion", "cs_degenerating_generation",
    "cs_attention_bottleneck", "cs_confident_repetition_risk",
]

ALL_SIGNALS = CORE_SIGNALS + HIDDEN_SIGNALS + ATTN_SIGNALS + HEALTH_SIGNALS + PROMPT_SIGNALS + COMPOUND_SIGNALS
BASELINE_FEATURES = ["entropy_mean", "surprisal_mean", "margin_mean"]


def load_data(path):
    df = pd.read_parquet(path)
    df = df[df["format_failure"] != True].copy()
    return df


# ===================================================================
# Focus 1: Per-Metric Correlation with Output Quality
# ===================================================================

def focus1_metric_correlation(df, out_dir):
    print("\n" + "=" * 60)
    print("FOCUS 1: PER-METRIC CORRELATION WITH OUTPUT")
    print("=" * 60)

    results = []
    for model in sorted(df["model"].unique()):
        for dataset in sorted(df["dataset"].unique()):
            cell = df[(df["model"] == model) & (df["dataset"] == dataset)].dropna(subset=["correct"])
            y = cell["correct"].astype(int).values
            if len(np.unique(y)) < 2 or len(y) < 50:
                continue

            for sig in ALL_SIGNALS:
                if sig not in cell.columns: continue
                vals = cell[sig].values.astype(float)
                mask = ~np.isnan(vals)
                if mask.sum() < 50: continue

                r, p = pointbiserialr(y[mask], vals[mask])
                try:
                    auc = roc_auc_score(y[mask], vals[mask])
                except: auc = 0.5

                results.append({
                    "model": model, "dataset": dataset, "signal": sig,
                    "correlation": r, "p_value": p, "auroc": auc,
                    "n": mask.sum(),
                    "signal_group": (
                        "core" if sig in CORE_SIGNALS else
                        "hidden" if sig in HIDDEN_SIGNALS else
                        "attention" if sig in ATTN_SIGNALS else
                        "health" if sig in HEALTH_SIGNALS else
                        "prompt" if sig in PROMPT_SIGNALS else
                        "compound"
                    ),
                })

    res_df = pd.DataFrame(results)
    if len(res_df) == 0: return res_df

    res_df.to_csv(out_dir / "focus1_correlations.csv", index=False)

    # Print top signals
    for model in sorted(df["model"].unique()):
        top = res_df[res_df["model"] == model].sort_values("auroc", ascending=False).head(15)
        print(f"\n  {model} — Top 15 by AUROC:")
        for _, r in top.iterrows():
            print(f"    {r['signal']:35s} {r['dataset']:10s} r={r['correlation']:+.3f} AUROC={r['auroc']:.3f} p={r['p_value']:.4f}")

    # Heatmap
    for dataset in df["dataset"].unique():
        sub = res_df[res_df["dataset"] == dataset]
        if len(sub) == 0: continue
        pivot = sub.pivot_table(index="signal", columns="model", values="correlation")
        if pivot.empty: continue
        fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.3)))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdBu_r", center=0, ax=ax, vmin=-0.3, vmax=0.3)
        ax.set_title(f"Signal-Correctness Correlation — {dataset}")
        fig.tight_layout()
        fig.savefig(out_dir / f"focus1_correlation_{dataset}.png", dpi=150)
        plt.close(fig)

    return res_df


# ===================================================================
# Focus 2: MoE vs Dense Comparison
# ===================================================================

def focus2_moe_vs_dense(df, out_dir):
    print("\n" + "=" * 60)
    print("FOCUS 2: MoE VS DENSE ARCHITECTURAL COMPARISON")
    print("=" * 60)

    if "mixtral" not in df["model"].unique():
        print("  Mixtral not in data — skipping.")
        return

    dense_models = [m for m in df["model"].unique() if m != "mixtral"]

    # Compare signal distributions
    signals_to_compare = ["entropy_mean", "surprisal_mean", "margin_mean",
                          "collapsed_rate_mean", "focused_head_mean",
                          "l2_norm_last_layer_mean", "risk_score"]

    for dataset in sorted(df["dataset"].unique()):
        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        axes = axes.flatten()

        for i, sig in enumerate(signals_to_compare):
            if i >= len(axes) or sig not in df.columns: continue
            ax = axes[i]
            sub = df[df["dataset"] == dataset][[sig, "model", "correct"]].dropna()
            if len(sub) < 10: continue
            sns.boxplot(data=sub, x="model", y=sig, hue="correct",
                        palette={True: "#4CAF50", False: "#F44336"}, ax=ax)
            ax.set_title(sig, fontsize=10)
            ax.legend([], frameon=False)

        if len(axes) > len(signals_to_compare):
            for j in range(len(signals_to_compare), len(axes)):
                axes[j].set_visible(False)

        fig.suptitle(f"MoE (Mixtral) vs Dense — {dataset}", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(out_dir / f"focus2_moe_vs_dense_{dataset}.png", dpi=150)
        plt.close(fig)

    # Per-layer attention entropy comparison
    layer_cols = [c for c in df.columns if c.startswith("layer_") and c.endswith("_attn_entropy")]
    if layer_cols:
        for dataset in sorted(df["dataset"].unique()):
            fig, ax = plt.subplots(figsize=(14, 6))
            for model in df["model"].unique():
                sub = df[(df["model"] == model) & (df["dataset"] == dataset)]
                means = [sub[c].mean() for c in sorted(layer_cols)]
                label = f"{model} ({'MoE' if model == 'mixtral' else 'dense'})"
                ax.plot(range(len(means)), means, label=label, linewidth=2)
            ax.set_xlabel("Layer Index")
            ax.set_ylabel("Mean Attention Entropy")
            ax.set_title(f"Per-Layer Attention Entropy — {dataset}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / f"focus2_layer_entropy_{dataset}.png", dpi=150)
            plt.close(fig)


# ===================================================================
# Focus 3: Self-Consistency Mapping
# ===================================================================

def focus3_self_consistency(df, out_dir):
    print("\n" + "=" * 60)
    print("FOCUS 3: SELF-CONSISTENCY (WITHIN-PROMPT DIVERGENCE)")
    print("=" * 60)

    results = []
    for model in sorted(df["model"].unique()):
        for dataset in sorted(df["dataset"].unique()):
            cell = df[(df["model"] == model) & (df["dataset"] == dataset)]
            prompts = cell.groupby("question_id")

            for qid, group in prompts:
                if len(group) < 5: continue
                correct_runs = group[group["correct"] == True]
                incorrect_runs = group[group["correct"] == False]

                if len(correct_runs) < 1 or len(incorrect_runs) < 1:
                    continue  # Need both pass and fail for comparison

                for sig in ["entropy_mean", "surprisal_mean", "margin_mean", "risk_score"]:
                    if sig not in group.columns: continue
                    c_vals = correct_runs[sig].dropna()
                    i_vals = incorrect_runs[sig].dropna()
                    if len(c_vals) < 1 or len(i_vals) < 1: continue

                    results.append({
                        "model": model, "dataset": dataset, "question_id": qid,
                        "signal": sig,
                        "correct_mean": c_vals.mean(),
                        "incorrect_mean": i_vals.mean(),
                        "delta": i_vals.mean() - c_vals.mean(),
                        "pass_rate": group["correct"].mean(),
                        "n_runs": len(group),
                    })

    res_df = pd.DataFrame(results)
    if len(res_df) == 0:
        print("  No prompts with both pass and fail runs.")
        return res_df

    res_df.to_csv(out_dir / "focus3_within_prompt.csv", index=False)

    # Summary: mean delta per signal
    for model in sorted(df["model"].unique()):
        print(f"\n  {model} — mean within-prompt delta (incorrect - correct):")
        sub = res_df[res_df["model"] == model]
        for sig in ["entropy_mean", "surprisal_mean", "margin_mean", "risk_score"]:
            sig_data = sub[sub["signal"] == sig]
            if len(sig_data) > 0:
                d = sig_data["delta"]
                print(f"    {sig:25s} mean_delta={d.mean():+.4f} (n={len(sig_data)} prompts, "
                      f"{(d > 0).sum()}/{len(d)} positive)")

    # Scatter: pass_rate vs mean entropy delta
    for dataset in sorted(df["dataset"].unique()):
        sub = res_df[(res_df["dataset"] == dataset) & (res_df["signal"] == "entropy_mean")]
        if len(sub) < 10: continue
        fig, ax = plt.subplots(figsize=(8, 6))
        for model in sorted(sub["model"].unique()):
            m_data = sub[sub["model"] == model]
            ax.scatter(m_data["pass_rate"], m_data["delta"], alpha=0.5, s=20, label=model)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Pass Rate (k=10)")
        ax.set_ylabel("Entropy Delta (incorrect - correct)")
        ax.set_title(f"Within-Prompt Entropy Divergence — {dataset}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"focus3_entropy_divergence_{dataset}.png", dpi=150)
        plt.close(fig)

    return res_df


# ===================================================================
# Focus 4: Layer-Level Signal Analysis
# ===================================================================

def focus4_layer_analysis(df, out_dir):
    print("\n" + "=" * 60)
    print("FOCUS 4: LAYER-LEVEL SIGNAL ANALYSIS")
    print("=" * 60)

    layer_cols = sorted([c for c in df.columns if c.startswith("layer_") and "_attn_entropy" in c])
    l2_cols = sorted([c for c in df.columns if c.startswith("layer_") and "_l2_norm" in c])

    if not layer_cols:
        print("  No per-layer data found.")
        return

    for model in sorted(df["model"].unique()):
        for dataset in sorted(df["dataset"].unique()):
            cell = df[(df["model"] == model) & (df["dataset"] == dataset)].dropna(subset=["correct"])
            if len(cell) < 50: continue

            correct = cell[cell["correct"] == True]
            incorrect = cell[cell["correct"] == False]
            if len(incorrect) < 10: continue

            # Per-layer correlation with correctness
            layer_corrs = []
            for col in layer_cols:
                vals = cell[col].dropna()
                if len(vals) < 50: continue
                mask = cell[col].notna()
                r, p = pointbiserialr(cell.loc[mask, "correct"].astype(int), cell.loc[mask, col])
                layer_idx = int(col.split("_")[1])
                layer_corrs.append({"layer": layer_idx, "correlation": r, "p_value": p, "metric": "attn_entropy"})

            for col in l2_cols:
                vals = cell[col].dropna()
                if len(vals) < 50: continue
                mask = cell[col].notna()
                r, p = pointbiserialr(cell.loc[mask, "correct"].astype(int), cell.loc[mask, col])
                layer_idx = int(col.split("_")[1])
                layer_corrs.append({"layer": layer_idx, "correlation": r, "p_value": p, "metric": "l2_norm"})

            if not layer_corrs: continue
            lc_df = pd.DataFrame(layer_corrs)

            fig, ax = plt.subplots(figsize=(12, 5))
            for metric in lc_df["metric"].unique():
                sub = lc_df[lc_df["metric"] == metric].sort_values("layer")
                ax.plot(sub["layer"], sub["correlation"], label=metric, marker="o", markersize=3)
            ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
            ax.set_xlabel("Layer Index")
            ax.set_ylabel("Correlation with Correctness")
            ax.set_title(f"Per-Layer Correlation — {model}/{dataset}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / f"focus4_layers_{model}_{dataset}.png", dpi=150)
            plt.close(fig)

            print(f"  {model}/{dataset}: saved per-layer correlation plot")


# ===================================================================
# Focus 5: Dataset Difficulty Profiling
# ===================================================================

def focus5_difficulty(df, out_dir):
    print("\n" + "=" * 60)
    print("FOCUS 5: DATASET DIFFICULTY PROFILING")
    print("=" * 60)

    if "question_difficulty" not in df.columns:
        print("  No difficulty data.")
        return

    # Correlate prompt-level signals with difficulty
    prompt_level = df.groupby(["dataset", "question_id"]).agg({
        "question_difficulty": "first",
        "prompt_surprisal_mean": "first",
        "basin_score_min": "first",
        "basin_score_mean": "first",
        "layer_transform_mean": "first",
        "n_sparse_heads": "first",
    }).dropna()

    if len(prompt_level) < 20:
        print("  Not enough data.")
        return

    print("\n  Prompt signal → Difficulty correlations:")
    for sig in ["prompt_surprisal_mean", "basin_score_min", "basin_score_mean",
                "layer_transform_mean", "n_sparse_heads"]:
        if sig in prompt_level.columns:
            vals = prompt_level[[sig, "question_difficulty"]].dropna()
            if len(vals) < 20: continue
            r, p = spearmanr(vals[sig], vals["question_difficulty"])
            print(f"    {sig:30s} ρ={r:+.3f} p={p:.4f}")

    # Scatter plot
    if "prompt_surprisal_mean" in prompt_level.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(prompt_level["prompt_surprisal_mean"], prompt_level["question_difficulty"],
                   alpha=0.4, s=20)
        ax.set_xlabel("Prompt Surprisal (mean)")
        ax.set_ylabel("Question Difficulty (1 - pass rate)")
        ax.set_title("Prompt Surprisal vs Empirical Difficulty")
        fig.tight_layout()
        fig.savefig(out_dir / f"focus5_difficulty_vs_surprisal.png", dpi=150)
        plt.close(fig)


# ===================================================================
# Focus 6: Cross-Model Behavioral Alignment
# ===================================================================

def focus6_cross_model(df, out_dir):
    print("\n" + "=" * 60)
    print("FOCUS 6: CROSS-MODEL BEHAVIORAL ALIGNMENT")
    print("=" * 60)

    models = sorted(df["model"].unique())
    if len(models) < 2:
        print("  Need ≥2 models.")
        return

    # For each question: do models agree on difficulty?
    model_difficulty = df.groupby(["model", "dataset", "question_id"])["correct"].mean().reset_index()
    model_difficulty.columns = ["model", "dataset", "question_id", "pass_rate"]

    for dataset in sorted(df["dataset"].unique()):
        sub = model_difficulty[model_difficulty["dataset"] == dataset]
        pivot = sub.pivot_table(index="question_id", columns="model", values="pass_rate")
        if pivot.shape[1] < 2: continue

        corr = pivot.corr()
        print(f"\n  {dataset} — pass-rate correlation between models:")
        print(corr.to_string())

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, annot=True, fmt=".3f", cmap="YlOrRd", vmin=0, vmax=1, ax=ax)
        ax.set_title(f"Cross-Model Difficulty Agreement — {dataset}")
        fig.tight_layout()
        fig.savefig(out_dir / f"focus6_model_agreement_{dataset}.png", dpi=150)
        plt.close(fig)

    # Fingerprint comparison
    fp_cols = [c for c in df.columns if c.startswith("fp_")]
    if len(fp_cols) >= 5:
        print("\n  Model fingerprint centroids (mean across all runs):")
        for model in models:
            fp = df[df["model"] == model][fp_cols].mean()
            print(f"    {model}: [{', '.join(f'{v:.3f}' for v in fp.values[:5])}...]")


# ===================================================================
# Ranking Evaluation: Can CoreVital pick the best run?
# ===================================================================

def ranking_evaluation(df, out_dir):
    print("\n" + "=" * 60)
    print("RANKING EVALUATION: Best-of-k Selection")
    print("=" * 60)

    ranking_signals = ["entropy_mean", "surprisal_mean", "margin_mean", "risk_score", "failure_risk"]
    # For margin: higher = more confident = pick highest
    # For entropy/surprisal/risk: lower = more confident = pick lowest
    pick_highest = {"margin_mean", "topk_mass_mean"}

    results = []
    for model in sorted(df["model"].unique()):
        for dataset in sorted(df["dataset"].unique()):
            cell = df[(df["model"] == model) & (df["dataset"] == dataset)]
            prompts = cell.groupby("question_id")

            for signal in ranking_signals:
                if signal not in cell.columns: continue

                random_correct = 0
                signal_correct = 0
                n_prompts = 0

                for qid, group in prompts:
                    group = group.dropna(subset=[signal, "correct"])
                    if len(group) < 3: continue

                    # Random selection: mean correctness
                    random_correct += group["correct"].mean()

                    # Signal-based selection: pick run with best signal
                    if signal in pick_highest:
                        best_idx = group[signal].idxmax()
                    else:
                        best_idx = group[signal].idxmin()

                    signal_correct += int(group.loc[best_idx, "correct"])
                    n_prompts += 1

                if n_prompts < 10: continue
                results.append({
                    "model": model, "dataset": dataset, "signal": signal,
                    "random_acc": random_correct / n_prompts,
                    "signal_acc": signal_correct / n_prompts,
                    "lift": (signal_correct / n_prompts) - (random_correct / n_prompts),
                    "n_prompts": n_prompts,
                })

    res_df = pd.DataFrame(results)
    if len(res_df) == 0: return res_df

    res_df.to_csv(out_dir / "ranking_results.csv", index=False)

    for model in sorted(df["model"].unique()):
        print(f"\n  {model}:")
        sub = res_df[res_df["model"] == model].sort_values("lift", ascending=False)
        for _, r in sub.iterrows():
            icon = "✓" if r["lift"] > 0.02 else " "
            print(f"    {icon} {r['signal']:20s} {r['dataset']:10s} "
                  f"random={r['random_acc']:.1%} signal={r['signal_acc']:.1%} lift={r['lift']:+.1%}")

    return res_df


# ===================================================================
# Signal Ablation
# ===================================================================

def signal_ablation(df, out_dir):
    print("\n" + "=" * 60)
    print("SIGNAL ABLATION: entropy-only → baseline → full")
    print("=" * 60)

    def cv_auroc(X, y, n_splits=5):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        aucs = []
        for tr, te in skf.split(X, y):
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X[tr])
            Xte = scaler.transform(X[te])
            m = LogisticRegression(penalty="l2", class_weight="balanced", max_iter=1000)
            m.fit(Xtr, y[tr])
            aucs.append(roc_auc_score(y[te], m.predict_proba(Xte)[:, 1]))
        return np.mean(aucs)

    for model in sorted(df["model"].unique()):
        for dataset in sorted(df["dataset"].unique()):
            cell = df[(df["model"] == model) & (df["dataset"] == dataset)].dropna(subset=["correct"])
            y = cell["correct"].astype(int).values
            if len(np.unique(y)) < 2 or len(y) < 100: continue

            # Tier 1: entropy only
            t1_cols = [c for c in ["entropy_mean"] if c in cell.columns]
            # Tier 2: simple confidence (baseline)
            t2_cols = [c for c in BASELINE_FEATURES if c in cell.columns]
            # Tier 3: + prompt signals
            t3_cols = t2_cols + [c for c in PROMPT_SIGNALS if c in cell.columns]
            # Tier 4: full CoreVital
            t4_cols = [c for c in ALL_SIGNALS if c in cell.columns]

            tiers = [
                ("entropy_only", t1_cols),
                ("confidence_baseline", t2_cols),
                ("+ prompt_signals", t3_cols),
                ("full_corevital", t4_cols),
            ]

            print(f"\n  {model}/{dataset} (n={len(y)}):")
            for name, cols in tiers:
                if not cols: continue
                X = cell[cols].fillna(0).values
                try:
                    auc = cv_auroc(X, y)
                    print(f"    {name:25s} ({len(cols):2d} features) AUROC={auc:.3f}")
                except Exception as e:
                    print(f"    {name:25s} ERROR: {e}")


# ===================================================================
# Summary Report
# ===================================================================

def write_summary(focus1_df, ranking_df, out_dir):
    path = out_dir / "RESULTS_SUMMARY.md"
    with open(path, "w") as f:
        f.write("# CoreVital Validation Experiment — Results Summary\n\n")
        f.write("**Design:** Pass@k (k=10) under sampling (temp 0.7 + 0.8)\n")
        f.write("**Datasets:** GSM8K (200) + HumanEval (164)\n")
        f.write("**Models:** Llama-3.1-8B, Qwen-2.5-7B, Mistral-7B, Mixtral-8x7B (8-bit)\n\n")

        f.write("## Focus 1: Per-Metric Correlations\n\n")
        if focus1_df is not None and len(focus1_df) > 0:
            top = focus1_df.sort_values("auroc", ascending=False).head(20)
            f.write(top[["model", "dataset", "signal", "correlation", "auroc", "signal_group"]].to_markdown(index=False))
        f.write("\n\n")

        f.write("## Ranking Evaluation\n\n")
        if ranking_df is not None and len(ranking_df) > 0:
            f.write(ranking_df.to_markdown(index=False))
        f.write("\n\n")

        f.write("## See Also\n\n")
        f.write("- `focus1_correlations.csv` — all signal correlations\n")
        f.write("- `focus3_within_prompt.csv` — within-prompt divergence\n")
        f.write("- `ranking_results.csv` — best-of-k selection results\n")
        f.write("- `*.png` — all visualizations\n")

    print(f"\n  Summary: {path}")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=RESULTS_DIR / "features.parquet")
    args = parser.parse_args()

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.features}...")
    df = load_data(args.features)
    print(f"  {len(df)} rows, {df['model'].nunique()} models, {df['dataset'].nunique()} datasets")

    focus1_df = focus1_metric_correlation(df, ANALYSIS_DIR)
    focus2_moe_vs_dense(df, ANALYSIS_DIR)
    focus3_df = focus3_self_consistency(df, ANALYSIS_DIR)
    focus4_layer_analysis(df, ANALYSIS_DIR)
    focus5_difficulty(df, ANALYSIS_DIR)
    focus6_cross_model(df, ANALYSIS_DIR)
    ranking_df = ranking_evaluation(df, ANALYSIS_DIR)
    signal_ablation(df, ANALYSIS_DIR)

    write_summary(focus1_df, ranking_df, ANALYSIS_DIR)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print(f"  Results: {ANALYSIS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()