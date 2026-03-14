#!/usr/bin/env python3
"""
CoreVital Validation Experiment — Analysis (v2)
=================================================
Evidence generator for the CoreVital validation experiment.
Reads features.parquet and produces structured artifacts per focus area.

Focus Areas:
  1. Per-metric correlation with output quality (direction-aware)
  2. MoE vs Dense architectural comparison (with effect sizes)
  3. Self-consistency / within-prompt signal divergence
  4. Layer-level signal analysis
  5. Dataset difficulty profiling
  6. Cross-model behavioral alignment
  + Ranking evaluation (best-of-k selection)
  + Signal ablation (entropy → baseline → full)
  + Format failure analysis (new)
  + Risk calibration analysis (new)

Outputs per section:
  analysis/{section}/tables/*.csv
  analysis/{section}/figures/*.png
  analysis/{section}/summary.json

Global outputs:
  analysis/key_findings.json
  analysis/RESULTS_SUMMARY.md
  analysis/global_manifest.json

Design principles:
  - Tables first, plots second (every figure backed by saved data)
  - Direction-aware AUROC with FDR correction
  - Dark-theme figures throughout
  - Dual human/AI output layers
  - Format failures analyzed, not discarded
  - Temperature available as a control axis

Usage:
    python3 analyze.py [--features PATH]
"""

import argparse
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import (
    mannwhitneyu, pointbiserialr, spearmanr,
)

# Selective warning suppression — only suppress known noisy warnings,
# NOT convergence or rank warnings which indicate real problems
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*is_categorical_dtype.*")
warnings.filterwarnings("ignore", message=".*ConstantInputWarning.*")

try:
    from helpers import (
        setup_logging, apply_dark_theme, ensure_dir, save_table,
        save_json, save_focus_outputs, save_figure, build_manifest,
        ACCENT_COLORS, CORRECT_COLOR, INCORRECT_COLOR,
    )
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from helpers import (
        setup_logging, apply_dark_theme, ensure_dir, save_table,
        save_json, save_focus_outputs, save_figure, build_manifest,
        ACCENT_COLORS, CORRECT_COLOR, INCORRECT_COLOR,
    )

log = setup_logging("analyze")

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = EXPERIMENT_DIR.parent  # For relative paths in saved JSON (public-friendly)
RESULTS_DIR = EXPERIMENT_DIR / "results"
ANALYSIS_DIR = EXPERIMENT_DIR / "analysis"

# Baseline feature sets for ablation
BASELINE_FEATURES = ["entropy_mean", "surprisal_mean", "margin_mean"]

# Signals known to be architecture-sensitive (need z-scoring for cross-model)
ARCH_SENSITIVE = [
    "l2_norm_last_layer_mean", "l2_norm_cross_layer_max",
    "focused_head_mean", "collapsed_rate_mean",
    "hidden_std_last_layer_mean", "hidden_max_abs_last_layer_mean",
]

PROMPT_INVARIANT_SIGNALS = [
    "prompt_surprisal_mean",
    "prompt_surprisal_max",
    "prompt_surprisal_std",
    "prompt_surprisal_p90",
    "prompt_token_count",
    "basin_score_min",
    "basin_score_mean",
    "basin_score_std",
    "layer_transform_mean",
    "layer_transform_std",
    "layer_transform_max",
    "n_sparse_heads",
]

PERFORMANCE_SIGNALS = [
    "total_run_ms",
    "inference_ms",
    "report_build_ms",
    "tokenize_ms",
    "overhead_ms",
    "overhead_pct",
    "known_non_inference_ms",
    "unattributed_ms",
    "inference_pct",
]


# Data loading, feature scope, and grouped CV helpers

def load_data(path: Path, label: str = "features") -> pd.DataFrame:
    """Load a parquet dataset and log its high-level shape."""
    df = pd.read_parquet(path)
    log.info(
        f"Loaded {label}: {len(df)} rows, {df['model'].nunique()} models, "
        f"{df['dataset'].nunique()} datasets"
    )
    return df


def get_prompt_signals(columns: List[str]) -> List[str]:
    """Return prompt-level signals that are invariant within a prompt group."""
    return [signal for signal in PROMPT_INVARIANT_SIGNALS if signal in columns]


def get_performance_signals(columns: List[str]) -> List[str]:
    """Return operational timing metrics that should stay out of core claims."""
    return [signal for signal in PERFORMANCE_SIGNALS if signal in columns]


def build_group_labels(df: pd.DataFrame, columns: Tuple[str, ...]) -> pd.Series:
    """Build stable group labels for grouped validation splits."""
    return df.loc[:, list(columns)].astype(str).agg("::".join, axis=1)


def iter_group_splits(
    df: pd.DataFrame,
    y: np.ndarray,
    group_cols: Tuple[str, ...] = ("question_id",),
    desired_splits: int = 5,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], int, str]:
    """Create grouped CV splits, preferring stratified groups when available."""
    from sklearn.model_selection import GroupKFold

    groups = build_group_labels(df, group_cols).to_numpy()
    n_groups = int(pd.Series(groups).nunique())
    n_splits = min(desired_splits, n_groups)
    if n_splits < 3:
        return [], n_groups, "insufficient_groups"

    split_kind = "GroupKFold"
    try:
        from sklearn.model_selection import StratifiedGroupKFold

        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(splitter.split(df, y, groups))
        split_kind = "StratifiedGroupKFold"
    except Exception:
        splitter = GroupKFold(n_splits=n_splits)
        splits = list(splitter.split(df, y, groups))

    return splits, n_groups, split_kind


def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive additional features from existing parquet columns.

    This lets the current experiment benefit from enrichment
    without re-extracting from raw traces.
    """
    df = df.copy()

    # Saturation flags
    if "risk_score" in df.columns:
        df["risk_score_is_zero"] = (df["risk_score"] == 0.0).astype(int)
        df["risk_score_is_one"] = (df["risk_score"] == 1.0).astype(int)
    if "failure_risk" in df.columns:
        df["failure_risk_is_zero"] = (df["failure_risk"] == 0.0).astype(int)
        df["failure_risk_is_one"] = (df["failure_risk"] == 1.0).astype(int)

    # Length-normalized densities
    gen = df["generated_tokens"].clip(lower=1)
    if "n_warning_signals" in df.columns:
        df["warning_density_per_100t"] = df["n_warning_signals"] / gen * 100
    if "n_compound_signals" in df.columns:
        df["compound_density_per_100t"] = df["n_compound_signals"] / gen * 100

    # Layer shape summaries (from wide layer columns)
    for metric_name, col_pattern in [
        ("layer_attn_entropy", "_attn_entropy"),
        ("layer_l2_norm", "_l2_norm"),
    ]:
        layer_cols = sorted([
            c for c in df.columns
            if c.startswith("layer_") and c.endswith(col_pattern)
        ])
        if len(layer_cols) >= 4:
            layer_vals = df[layer_cols]
            df[f"{metric_name}_cross_range"] = layer_vals.max(axis=1) - layer_vals.min(axis=1)
            df[f"{metric_name}_peak_layer"] = layer_vals.idxmax(axis=1).str.extract(r"(\d+)").astype(float)
            n = len(layer_cols)
            third = max(1, n // 3)
            early_cols = layer_cols[:third]
            late_cols = layer_cols[-third:]
            df[f"{metric_name}_early_mean"] = df[early_cols].mean(axis=1)
            df[f"{metric_name}_late_mean"] = df[late_cols].mean(axis=1)
            df[f"{metric_name}_early_late_delta"] = (
                df[f"{metric_name}_late_mean"] - df[f"{metric_name}_early_mean"]
            )

    # Model-relative z-scores for architecture-sensitive features
    for col in ARCH_SENSITIVE:
        if col in df.columns:
            df[f"{col}_zscore"] = df.groupby("model")[col].transform(
                lambda x: (x - x.mean()) / x.std(ddof=1)
                if x.std(ddof=1) > 1e-9 else 0.0
            )

    if "entropy_max" in df.columns and "entropy_mean" in df.columns:
        df["entropy_range_approx"] = df["entropy_max"] - df["entropy_mean"]

    log.info(f"Enriched to {len(df.columns)} columns")
    return df


def get_analysis_signals(
    df: pd.DataFrame,
    *,
    include_prompt: bool = False,
    include_performance: bool = False,
) -> List[str]:
    """
    Dynamically discover numeric columns suitable for analysis.

    Primary analyses exclude prompt-invariant features (handled at prompt level)
    and operational timing metrics (handled as secondary diagnostics).
    """
    exclude_prefixes = ("layer_", "fp_")
    exclude_exact = {
        "run_idx", "seed", "correct", "format_failure", "temperature",
        "pass_rate", "question_difficulty", "empirical_difficulty_pooled",
        "generated_tokens", "total_steps", "grade_found",
        "trace_has_prompt_analysis", "trace_has_fingerprint",
        "trace_has_performance", "trace_has_extensions",
        "trace_has_health_flags", "trace_has_early_warning", "trace_has_risk",
        "arch_num_layers", "arch_num_heads", "arch_hidden_size", "arch_is_moe",
        "fingerprint_dim_total", "fingerprint_dim_kept", "fingerprint_truncated",
    }
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    signals = [
        c for c in numeric_cols
        if c not in exclude_exact
        and not any(c.startswith(p) for p in exclude_prefixes)
    ]

    prompt_signals = set(get_prompt_signals(list(signals)))
    performance_signals = set(get_performance_signals(list(signals)))

    filtered = []
    for signal in signals:
        if not include_prompt and signal in prompt_signals:
            continue
        if not include_performance and signal in performance_signals:
            continue
        filtered.append(signal)

    return sorted(filtered)


def correctness_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to rows suitable for correctness-conditioned analysis."""
    mask = (
        df["correct"].notna()
        & (df["format_failure"].fillna(False) != True)
    )
    return df[mask].copy()


# ?????? Focus 1: Per-Metric Correlation ???????????????????????????????????????????????????????????????????????????

def focus1_metric_correlation(df: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    """
    Direction-aware metric correlation with output quality.
    
    For each signal × model × dataset cell:
      - Point-biserial correlation with correctness
      - Raw AUROC and direction-aware predictive power
      - FDR-adjusted p-values
    
    Sorts by predictive_power (= max(auc, 1-auc)) so inverse signals
    like entropy are properly surfaced.
    """
    log.info("Focus 1: Per-metric correlation with output quality")
    out_dir = ensure_dir(base_dir / "focus_01_metric_correlation")
    fig_dir = ensure_dir(out_dir / "figures")

    cdf = correctness_subset(df)
    signals = get_analysis_signals(cdf)
    log.info(f"  Testing {len(signals)} signals")

    from sklearn.metrics import roc_auc_score

    results = []
    for model in sorted(cdf["model"].unique()):
        for dataset in sorted(cdf["dataset"].unique()):
            cell = cdf[(cdf["model"] == model) & (cdf["dataset"] == dataset)]
            y = cell["correct"].astype(int).values
            if len(np.unique(y)) < 2 or len(y) < 50:
                continue
            class_balance = y.mean()

            for sig in signals:
                if sig not in cell.columns:
                    continue
                vals = cell[sig].values.astype(float)
                mask = ~np.isnan(vals)
                if mask.sum() < 50:
                    continue
                # Skip constant signals (all same value → no information)
                if np.std(vals[mask]) < 1e-12:
                    continue
                missing_rate = 1.0 - (mask.sum() / len(vals))

                r, p = pointbiserialr(y[mask], vals[mask])
                try:
                    raw_auc = roc_auc_score(y[mask], vals[mask])
                except Exception:
                    raw_auc = 0.5

                pred_power = max(raw_auc, 1.0 - raw_auc)
                direction = "higher_better" if raw_auc >= 0.5 else "lower_better"
                auroc_oriented = raw_auc if raw_auc >= 0.5 else 1.0 - raw_auc

                results.append({
                    "model": model, "dataset": dataset, "signal": sig,
                    "correlation": r, "p_value": p,
                    "raw_auroc": raw_auc,
                    "predictive_power": pred_power,
                    "auroc_oriented": auroc_oriented,
                    "direction": direction,
                    "n": int(mask.sum()),
                    "missing_rate": missing_rate,
                    "class_balance": class_balance,
                })

    res_df = pd.DataFrame(results)
    if len(res_df) == 0:
        log.warning("  No correlation results")
        save_json({"finding": "No valid correlation results", "n_signals": 0}, out_dir / "summary.json")
        return res_df

    # ── FDR correction (Benjamini-Hochberg) ──
    # Drop any rows with NaN p-values (from degenerate inputs) before FDR
    res_df = res_df[res_df["p_value"].notna() & np.isfinite(res_df["p_value"])].reset_index(drop=True)

    if len(res_df) > 0:
        p_vals = res_df["p_value"].values
        n_tests = len(p_vals)
        sorted_idx = np.argsort(p_vals)
        ranks = np.empty(n_tests, dtype=int)
        ranks[sorted_idx] = np.arange(1, n_tests + 1)
        fdr = np.clip(p_vals * n_tests / ranks, 0, 1)
        # Enforce monotonicity
        fdr_sorted = fdr[sorted_idx]
        for i in range(n_tests - 2, -1, -1):
            fdr_sorted[i] = min(fdr_sorted[i], fdr_sorted[i + 1] if i + 1 < n_tests else 1.0)
        fdr[sorted_idx] = fdr_sorted
        res_df["p_value_fdr"] = fdr

    # ── Save tables ──
    save_focus_outputs(out_dir, "focus1", {
        "results": res_df,
    }, summary={}, logger=log)

    # ── Console: top signals per model ──
    for model in sorted(cdf["model"].unique()):
        sub = res_df[res_df["model"] == model].sort_values("predictive_power", ascending=False)
        print(f"\n  {model} — Top 20 by predictive power:")
        for _, r in sub.head(20).iterrows():
            dir_arrow = "↑" if r["direction"] == "higher_better" else "↓"
            fdr_flag = "*" if r["p_value_fdr"] < 0.05 else " "
            print(f"   {fdr_flag}{dir_arrow} {r['signal']:40s} {r['dataset']:10s} "
                  f"PP={r['predictive_power']:.3f} r={r['correlation']:+.3f} "
                  f"p_fdr={r['p_value_fdr']:.4f}")

    # ── Heatmaps ──
    for dataset in sorted(cdf["dataset"].unique()):
        sub = res_df[res_df["dataset"] == dataset]
        if len(sub) == 0:
            continue

        # Predictive power heatmap
        pivot = sub.pivot_table(
            index="signal", columns="model", values="predictive_power"
        )
        # Keep top 25 signals by mean predictive power
        top_signals = pivot.mean(axis=1).sort_values(ascending=False).head(25).index
        pivot = pivot.loc[pivot.index.isin(top_signals)].sort_values(
            by=pivot.columns[0], ascending=False
        )

        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.35)))
        sns.heatmap(
            pivot, annot=True, fmt=".3f", cmap="magma",
            vmin=0.5, vmax=0.75, ax=ax, linewidths=0.5,
        )
        ax.set_title(f"Predictive Power (|AUROC - 0.5| + 0.5) — {dataset}")
        save_figure(fig, fig_dir / f"focus1_pred_power_{dataset}.png")

        # Signed correlation heatmap
        pivot_corr = sub.pivot_table(
            index="signal", columns="model", values="correlation"
        )
        pivot_corr = pivot_corr.loc[pivot_corr.index.isin(top_signals)]
        fig, ax = plt.subplots(figsize=(10, max(6, len(pivot_corr) * 0.35)))
        sns.heatmap(
            pivot_corr, annot=True, fmt=".3f", cmap="RdBu_r",
            center=0, vmin=-0.3, vmax=0.3, ax=ax, linewidths=0.5,
        )
        ax.set_title(f"Signed Correlation with Correctness — {dataset}")
        save_figure(fig, fig_dir / f"focus1_correlation_{dataset}.png")

        # Save heatmap source tables
        save_table(pivot.reset_index(), out_dir / "tables" / f"focus1_heatmap_pred_power_{dataset}.csv")
        save_table(pivot_corr.reset_index(), out_dir / "tables" / f"focus1_heatmap_correlation_{dataset}.csv")

    # ── Section summary ──
    top_global = res_df.sort_values("predictive_power", ascending=False).head(10)
    summary = {
        "description": "Per-metric correlation with output quality (direction-aware)",
        "n_signals_tested": len(signals),
        "n_results": len(res_df),
        "n_significant_fdr05": int((res_df["p_value_fdr"] < 0.05).sum()),
        "top_10_global": top_global[
            ["model", "dataset", "signal", "predictive_power", "direction", "correlation", "p_value_fdr"]
        ].to_dict("records"),
        "top_5_per_model": {
            m: res_df[res_df["model"] == m].sort_values("predictive_power", ascending=False)
            .head(5)[["dataset", "signal", "predictive_power", "direction"]]
            .to_dict("records")
            for m in sorted(cdf["model"].unique())
        },
    }
    save_json(summary, out_dir / "summary.json")

    return res_df


# ── Focus 2: MoE vs Dense ───────────────────────────────────

def focus2_moe_vs_dense(df: pd.DataFrame, base_dir: Path) -> Optional[pd.DataFrame]:
    """
    MoE vs Dense architectural comparison with numeric evidence.
    
    Compares Mixtral (MoE) against dense models using:
      - Raw signal distributions
      - Model-relative z-scored distributions
      - Mann-Whitney U tests for distributional differences
      - Effect sizes (rank-biserial correlation)
      - Correctness-stratified comparisons
    """
    log.info("Focus 2: MoE vs Dense architectural comparison")
    out_dir = ensure_dir(base_dir / "focus_02_moe_vs_dense")
    fig_dir = ensure_dir(out_dir / "figures")

    if "mixtral" not in df["model"].unique():
        log.warning("  Mixtral not in data — skipping")
        save_json({"finding": "Mixtral not present in dataset"}, out_dir / "summary.json")
        return None

    cdf = correctness_subset(df)
    dense_models = [m for m in sorted(cdf["model"].unique()) if m != "mixtral"]

    # Signals to compare: both raw and z-scored where available
    raw_signals = [
        "entropy_mean", "surprisal_mean", "margin_mean",
        "collapsed_rate_mean", "focused_head_mean",
        "l2_norm_last_layer_mean", "risk_score",
        "high_entropy_frac", "n_compound_signals",
    ]
    zscore_signals = [f"{s}_zscore" for s in ARCH_SENSITIVE if f"{s}_zscore" in cdf.columns]

    # ── Comparison table (raw + z-scored) ──
    comparison_rows = []
    for dataset in sorted(cdf["dataset"].unique()):
        ds = cdf[cdf["dataset"] == dataset]
        mixtral = ds[ds["model"] == "mixtral"]
        dense_all = ds[ds["model"].isin(dense_models)]

        for sig in raw_signals + zscore_signals:
            if sig not in ds.columns:
                continue
            mx = mixtral[sig].dropna()
            dn = dense_all[sig].dropna()
            if len(mx) < 20 or len(dn) < 20:
                continue

            # Mann-Whitney U
            try:
                u_stat, u_p = mannwhitneyu(mx, dn, alternative="two-sided")
                # Rank-biserial correlation as effect size
                n1, n2 = len(mx), len(dn)
                rank_biserial = 1 - (2 * u_stat) / (n1 * n2)
            except Exception:
                u_stat, u_p, rank_biserial = None, None, None

            comparison_rows.append({
                "dataset": dataset, "signal": sig,
                "is_zscore": sig.endswith("_zscore"),
                "mixtral_mean": mx.mean(), "mixtral_std": mx.std(),
                "mixtral_median": mx.median(), "mixtral_n": len(mx),
                "dense_mean": dn.mean(), "dense_std": dn.std(),
                "dense_median": dn.median(), "dense_n": len(dn),
                "delta_mean": mx.mean() - dn.mean(),
                "mann_whitney_u": u_stat, "mann_whitney_p": u_p,
                "rank_biserial": rank_biserial,
            })

    comp_df = pd.DataFrame(comparison_rows)

    # ── Correctness-stratified comparison ──
    strat_rows = []
    for dataset in sorted(cdf["dataset"].unique()):
        for correct_val in [True, False]:
            ds = cdf[(cdf["dataset"] == dataset) & (cdf["correct"] == correct_val)]
            mx = ds[ds["model"] == "mixtral"]
            dn = ds[ds["model"].isin(dense_models)]
            for sig in raw_signals[:5]:  # Keep this focused
                if sig not in ds.columns:
                    continue
                mx_vals = mx[sig].dropna()
                dn_vals = dn[sig].dropna()
                if len(mx_vals) < 10 or len(dn_vals) < 10:
                    continue
                strat_rows.append({
                    "dataset": dataset, "correct": correct_val, "signal": sig,
                    "mixtral_mean": mx_vals.mean(), "dense_mean": dn_vals.mean(),
                    "delta": mx_vals.mean() - dn_vals.mean(),
                })

    strat_df = pd.DataFrame(strat_rows) if strat_rows else pd.DataFrame()

    # ── Save tables ──
    tables = {"comparison": comp_df}
    if not strat_df.empty:
        tables["correctness_stratified"] = strat_df

    save_focus_outputs(out_dir, "focus2", tables, summary={}, logger=log)

    # ── Figures: boxplots for key raw signals ──
    for dataset in sorted(cdf["dataset"].unique()):
        ds = cdf[cdf["dataset"] == dataset]
        plot_sigs = [s for s in raw_signals[:6] if s in ds.columns]
        n_sigs = len(plot_sigs)
        if n_sigs == 0:
            continue

        ncols = min(3, n_sigs)
        nrows = (n_sigs + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
        axes = np.atleast_2d(axes).flatten()

        for i, sig in enumerate(plot_sigs):
            sub = ds[[sig, "model", "correct"]].dropna()
            if len(sub) < 10:
                continue
            sns.boxplot(
                data=sub, x="model", y=sig, hue="correct",
                palette={True: CORRECT_COLOR, False: INCORRECT_COLOR},
                ax=axes[i], fliersize=2,
            )
            axes[i].set_title(sig, fontsize=10)
            axes[i].legend([], frameon=False)

        for j in range(n_sigs, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f"MoE (Mixtral) vs Dense — {dataset}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        save_figure(fig, fig_dir / f"focus2_boxplots_{dataset}.png")

    # ── Layer entropy profile comparison ──
    layer_cols = sorted([c for c in df.columns if c.startswith("layer_") and c.endswith("_attn_entropy")])
    if layer_cols:
        for dataset in sorted(cdf["dataset"].unique()):
            fig, ax = plt.subplots(figsize=(14, 6))
            profile_rows = []
            for model in sorted(cdf["model"].unique()):
                sub = cdf[(cdf["model"] == model) & (cdf["dataset"] == dataset)]
                means = [sub[c].mean() for c in layer_cols]
                label = f"{model} ({'MoE' if model == 'mixtral' else 'dense'})"
                ax.plot(range(len(means)), means, label=label, linewidth=2)
                for li, m in enumerate(means):
                    profile_rows.append({"model": model, "dataset": dataset,
                                         "layer": li, "attn_entropy_mean": m})

            ax.set_xlabel("Layer Index")
            ax.set_ylabel("Mean Attention Entropy")
            ax.set_title(f"Per-Layer Attention Entropy — {dataset}")
            ax.legend()
            fig.tight_layout()
            save_figure(fig, fig_dir / f"focus2_layer_entropy_{dataset}.png")

            # Save profile table
            save_table(pd.DataFrame(profile_rows),
                       out_dir / "tables" / f"focus2_layer_profile_{dataset}.csv")

    # ── Section summary ──
    top_diffs = comp_df.sort_values("rank_biserial", key=abs, ascending=False).head(5) if len(comp_df) > 0 else pd.DataFrame()
    summary = {
        "description": "MoE (Mixtral) vs Dense architectural comparison",
        "n_comparisons": len(comp_df),
        "n_significant_p05": int((comp_df["mann_whitney_p"] < 0.05).sum()) if len(comp_df) > 0 else 0,
        "largest_effects": top_diffs[
            ["dataset", "signal", "delta_mean", "rank_biserial", "mann_whitney_p"]
        ].to_dict("records") if len(top_diffs) > 0 else [],
    }
    save_json(summary, out_dir / "summary.json")

    # ── Console ──
    if len(comp_df) > 0:
        print(f"\n  Top MoE vs Dense effects:")
        for _, r in comp_df.sort_values("rank_biserial", key=abs, ascending=False).head(10).iterrows():
            sig = "*" if r["mann_whitney_p"] and r["mann_whitney_p"] < 0.05 else " "
            print(f"   {sig} {r['signal']:40s} {r['dataset']:10s} "
                  f"Δ={r['delta_mean']:+.4f} r_rb={r['rank_biserial']:+.3f} "
                  f"p={r['mann_whitney_p']:.4f}")

    return comp_df


# ── Focus 3: Self-Consistency ────────────────────────────────

def focus3_self_consistency(df: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    """
    Within-prompt signal divergence analysis.
    
    For each prompt with both correct and incorrect runs:
      - Correct vs incorrect mean gap for all signals
      - Within-prompt std across all runs
      - Temperature sensitivity check
    """
    log.info("Focus 3: Self-consistency / within-prompt divergence")
    out_dir = ensure_dir(base_dir / "focus_03_self_consistency")
    fig_dir = ensure_dir(out_dir / "figures")

    cdf = correctness_subset(df)
    signals = get_analysis_signals(cdf)

    # ── Per-prompt correct vs incorrect divergence ──
    divergence_rows = []
    dispersion_rows = []

    for model in sorted(cdf["model"].unique()):
        for dataset in sorted(cdf["dataset"].unique()):
            cell = cdf[(cdf["model"] == model) & (cdf["dataset"] == dataset)]
            for qid, group in cell.groupby("question_id"):
                if len(group) < 3:
                    continue
                correct = group[group["correct"] == True]
                incorrect = group[group["correct"] == False]

                # Within-prompt dispersion (all runs)
                for sig in signals:
                    if sig not in group.columns:
                        continue
                    vals = group[sig].dropna()
                    if len(vals) < 3:
                        continue
                    dispersion_rows.append({
                        "model": model, "dataset": dataset, "question_id": qid,
                        "signal": sig, "n_runs": len(vals),
                        "within_std": vals.std(),
                        "within_iqr": vals.quantile(0.75) - vals.quantile(0.25),
                        "within_range": vals.max() - vals.min(),
                        "pass_rate": group["correct"].mean(),
                    })

                # Correct vs incorrect gap
                if len(correct) < 1 or len(incorrect) < 1:
                    continue
                for sig in signals:
                    if sig not in group.columns:
                        continue
                    c_vals = correct[sig].dropna()
                    i_vals = incorrect[sig].dropna()
                    if len(c_vals) < 1 or len(i_vals) < 1:
                        continue
                    divergence_rows.append({
                        "model": model, "dataset": dataset, "question_id": qid,
                        "signal": sig,
                        "correct_mean": c_vals.mean(),
                        "incorrect_mean": i_vals.mean(),
                        "delta": i_vals.mean() - c_vals.mean(),
                        "pass_rate": group["correct"].mean(),
                        "n_correct": len(c_vals),
                        "n_incorrect": len(i_vals),
                    })

    div_df = pd.DataFrame(divergence_rows) if divergence_rows else pd.DataFrame()
    disp_df = pd.DataFrame(dispersion_rows) if dispersion_rows else pd.DataFrame()

    # ── Aggregate: mean delta per signal per model ──
    if not div_df.empty:
        agg_delta = div_df.groupby(["model", "dataset", "signal"]).agg(
            mean_delta=("delta", "mean"),
            median_delta=("delta", "median"),
            frac_positive=("delta", lambda x: (x > 0).mean()),
            n_prompts=("delta", "count"),
        ).reset_index()
    else:
        agg_delta = pd.DataFrame()

    # ── Temperature sensitivity ──
    temp_rows = []
    if "temperature" in cdf.columns and cdf["temperature"].nunique() > 1:
        for model in sorted(cdf["model"].unique()):
            for dataset in sorted(cdf["dataset"].unique()):
                cell = cdf[(cdf["model"] == model) & (cdf["dataset"] == dataset)]
                for temp in sorted(cell["temperature"].dropna().unique()):
                    t_cell = cell[cell["temperature"] == temp]
                    for sig in ["entropy_mean", "surprisal_mean", "margin_mean", "risk_score"]:
                        if sig not in t_cell.columns:
                            continue
                        y = t_cell["correct"].astype(int).values
                        vals = t_cell[sig].values.astype(float)
                        mask = ~np.isnan(vals) & ~np.isnan(y.astype(float))
                        if mask.sum() < 20 or len(np.unique(y[mask])) < 2:
                            continue
                        if np.std(vals[mask]) < 1e-12:
                            continue
                        r, p = pointbiserialr(y[mask], vals[mask])
                        temp_rows.append({
                            "model": model, "dataset": dataset,
                            "temperature": temp, "signal": sig,
                            "correlation": r, "p_value": p, "n": int(mask.sum()),
                        })

    temp_df = pd.DataFrame(temp_rows) if temp_rows else pd.DataFrame()

    # ── Save tables ──
    tables = {}
    if not div_df.empty:
        tables["divergence"] = div_df
    if not disp_df.empty:
        tables["dispersion"] = disp_df
    if not agg_delta.empty:
        tables["delta_summary"] = agg_delta
    if not temp_df.empty:
        tables["temperature_sensitivity"] = temp_df

    save_focus_outputs(out_dir, "focus3", tables, summary={}, logger=log)

    # ── Console ──
    if not agg_delta.empty:
        for model in sorted(cdf["model"].unique()):
            sub = agg_delta[agg_delta["model"] == model]
            top = sub.sort_values("mean_delta", key=abs, ascending=False).head(10)
            print(f"\n  {model} — top within-prompt deltas (incorrect - correct):")
            for _, r in top.iterrows():
                print(f"    {r['signal']:40s} {r['dataset']:10s} "
                      f"Δ={r['mean_delta']:+.4f} ({r['frac_positive']:.0%} positive, "
                      f"n={r['n_prompts']})")

    # ── Figures: entropy divergence scatter ──
    if not div_df.empty:
        for dataset in sorted(cdf["dataset"].unique()):
            for sig in ["entropy_mean", "margin_mean"]:
                sub = div_df[(div_df["dataset"] == dataset) & (div_df["signal"] == sig)]
                if len(sub) < 10:
                    continue
                fig, ax = plt.subplots(figsize=(8, 6))
                for model in sorted(sub["model"].unique()):
                    m_data = sub[sub["model"] == model]
                    ax.scatter(m_data["pass_rate"], m_data["delta"],
                               alpha=0.5, s=20, label=model)
                ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
                ax.set_xlabel("Pass Rate (k=10)")
                ax.set_ylabel(f"{sig} Delta (incorrect - correct)")
                ax.set_title(f"Within-Prompt Divergence — {sig} — {dataset}")
                ax.legend()
                fig.tight_layout()
                save_figure(fig, fig_dir / f"focus3_{sig}_divergence_{dataset}.png")

    # ── Section summary ──
    summary = {
        "description": "Within-prompt signal divergence and self-consistency",
        "n_divergence_pairs": len(div_df),
        "n_dispersion_records": len(disp_df),
        "temperature_sensitivity_available": not temp_df.empty,
        "top_divergent_signals": (
            agg_delta.sort_values("mean_delta", key=abs, ascending=False)
            .head(10)[["model", "dataset", "signal", "mean_delta", "frac_positive", "n_prompts"]]
            .to_dict("records")
        ) if not agg_delta.empty else [],
    }
    save_json(summary, out_dir / "summary.json")

    return div_df


# ── Focus 4: Layer-Level Signal Analysis ─────────────────────

def focus4_layer_analysis(df: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    """
    Per-layer correlation with correctness.
    
    Outputs a long-form evidence table with per-layer associations,
    correct/incorrect mean differences, and normalized layer positions.
    """
    log.info("Focus 4: Layer-level signal analysis")
    out_dir = ensure_dir(base_dir / "focus_04_layer_analysis")
    fig_dir = ensure_dir(out_dir / "figures")

    cdf = correctness_subset(df)
    # Match only layer_NN_attn_entropy / layer_NN_l2_norm (not enriched shape columns)
    import re
    layer_attn_cols = sorted([c for c in df.columns if re.match(r"layer_\d+_attn_entropy$", c)])
    layer_l2_cols = sorted([c for c in df.columns if re.match(r"layer_\d+_l2_norm$", c)])

    if not layer_attn_cols:
        log.warning("  No per-layer data found")
        save_json({"finding": "No per-layer data available"}, out_dir / "summary.json")
        return pd.DataFrame()

    all_layer_rows = []

    for model in sorted(cdf["model"].unique()):
        for dataset in sorted(cdf["dataset"].unique()):
            cell = cdf[(cdf["model"] == model) & (cdf["dataset"] == dataset)].dropna(subset=["correct"])
            if len(cell) < 50:
                continue
            correct = cell[cell["correct"] == True]
            incorrect = cell[cell["correct"] == False]
            if len(incorrect) < 10:
                continue

            n_layers = len(layer_attn_cols)

            for cols_set, metric_name in [
                (layer_attn_cols, "attn_entropy"),
                (layer_l2_cols, "l2_norm"),
            ]:
                for col in cols_set:
                    vals = cell[col].dropna()
                    if len(vals) < 50:
                        continue
                    mask = cell[col].notna()
                    y = cell.loc[mask, "correct"].astype(int)
                    x = cell.loc[mask, col]
                    if x.std() < 1e-12:
                        continue

                    r, p = pointbiserialr(y, x)
                    layer_idx = int(col.split("_")[1])

                    c_mean = correct[col].mean() if col in correct.columns else None
                    i_mean = incorrect[col].mean() if col in incorrect.columns else None

                    all_layer_rows.append({
                        "model": model, "dataset": dataset,
                        "metric": metric_name,
                        "layer_idx": layer_idx,
                        "layer_pos_norm": layer_idx / max(1, n_layers - 1),
                        "correlation": r, "p_value": p,
                        "mean_correct": c_mean,
                        "mean_incorrect": i_mean,
                        "delta": (i_mean - c_mean) if c_mean is not None and i_mean is not None else None,
                        "n": int(mask.sum()),
                    })

    layer_df = pd.DataFrame(all_layer_rows)

    # ── Save tables ──
    save_focus_outputs(out_dir, "focus4", {"layer_results": layer_df}, summary={}, logger=log)

    if layer_df.empty:
        save_json({"finding": "Insufficient data for layer analysis"}, out_dir / "summary.json")
        return layer_df

    # ── Figures ──
    for model in sorted(cdf["model"].unique()):
        for dataset in sorted(cdf["dataset"].unique()):
            sub = layer_df[(layer_df["model"] == model) & (layer_df["dataset"] == dataset)]
            if len(sub) == 0:
                continue

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            for ax, metric in zip(axes, ["attn_entropy", "l2_norm"]):
                m_data = sub[sub["metric"] == metric].sort_values("layer_idx")
                if len(m_data) == 0:
                    ax.set_visible(False)
                    continue
                ax.plot(m_data["layer_idx"], m_data["correlation"],
                        marker="o", markersize=3, linewidth=1.5)
                ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
                ax.set_xlabel("Layer Index")
                ax.set_ylabel("Correlation with Correctness")
                ax.set_title(f"{metric}")

            fig.suptitle(f"Per-Layer Correlation — {model}/{dataset}", fontsize=13)
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            save_figure(fig, fig_dir / f"focus4_layers_{model}_{dataset}.png")

            # Correct vs incorrect mean profiles
            for metric in ["attn_entropy", "l2_norm"]:
                m_data = sub[sub["metric"] == metric].sort_values("layer_idx")
                if len(m_data) == 0 or m_data["mean_correct"].isna().all():
                    continue
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(m_data["layer_idx"], m_data["mean_correct"],
                        label="Correct", color=CORRECT_COLOR, linewidth=2)
                ax.plot(m_data["layer_idx"], m_data["mean_incorrect"],
                        label="Incorrect", color=INCORRECT_COLOR, linewidth=2)
                ax.fill_between(
                    m_data["layer_idx"],
                    m_data["mean_correct"], m_data["mean_incorrect"],
                    alpha=0.15, color="white",
                )
                ax.set_xlabel("Layer Index")
                ax.set_ylabel(f"Mean {metric}")
                ax.set_title(f"{metric} by Correctness — {model}/{dataset}")
                ax.legend()
                fig.tight_layout()
                save_figure(fig, fig_dir / f"focus4_{metric}_profile_{model}_{dataset}.png")

    # ── Summary: which layers matter most? ──
    peak_layers = []
    for (model, dataset, metric), grp in layer_df.groupby(["model", "dataset", "metric"]):
        if len(grp) == 0:
            continue
        best = grp.loc[grp["correlation"].abs().idxmax()]
        peak_layers.append({
            "model": model, "dataset": dataset, "metric": metric,
            "peak_layer": int(best["layer_idx"]),
            "peak_correlation": best["correlation"],
            "peak_p": best["p_value"],
        })

    summary = {
        "description": "Per-layer signal association with correctness",
        "n_layer_tests": len(layer_df),
        "peak_layers": peak_layers,
    }
    save_json(summary, out_dir / "summary.json")
    print(f"\n  Layer analysis: {len(layer_df)} layer-metric tests across {cdf['model'].nunique()} models")

    return layer_df


# ── Focus 5: Dataset Difficulty Profiling ────────────────────

def focus5_difficulty(
    df: pd.DataFrame,
    prompt_df: Optional[pd.DataFrame],
    base_dir: Path,
) -> pd.DataFrame:
    """
    Correlate prompt-level signals with empirical difficulty.

    Uses the dedicated prompt-level table when available so prompt telemetry is
    analyzed once per prompt group instead of being repeated across runs.
    """
    log.info("Focus 5: Dataset difficulty profiling")
    out_dir = ensure_dir(base_dir / "focus_05_difficulty")
    fig_dir = ensure_dir(out_dir / "figures")

    if "question_difficulty" not in df.columns and "empirical_difficulty_pooled" not in df.columns:
        log.warning("  No difficulty data")
        save_json({"finding": "No difficulty data available"}, out_dir / "summary.json")
        return pd.DataFrame()

    diff_col = "empirical_difficulty_pooled" if "empirical_difficulty_pooled" in df.columns else "question_difficulty"

    prompt_signals = get_prompt_signals(list(df.columns))
    prompt_source = None
    if prompt_df is not None and not prompt_df.empty:
        prompt_source = prompt_df.copy()
        prompt_source = prompt_source.dropna(subset=[diff_col]) if diff_col in prompt_source.columns else prompt_source
    else:
        prompt_agg = {"pass_rate": "first"}
        for signal in prompt_signals:
            if signal in df.columns:
                prompt_agg[signal] = "first"
        if diff_col in df.columns:
            prompt_agg[diff_col] = "first"
        prompt_source = (
            df.groupby(["model", "dataset", "question_id"]).agg(prompt_agg).reset_index()
            if prompt_agg
            else pd.DataFrame()
        )

    if prompt_source is None or prompt_source.empty:
        log.warning("  No prompt-level source data available")
        save_json({"finding": "No prompt-level source data available"}, out_dir / "summary.json")
        return pd.DataFrame()

    prompt_signals = [signal for signal in prompt_signals if signal in prompt_source.columns]

    invariance_rows = []
    for signal in prompt_signals:
        for (model, dataset, qid), grp in df.groupby(["model", "dataset", "question_id"]):
            vals = grp[signal].dropna()
            if len(vals) < 2:
                continue
            invariance_rows.append({
                "model": model,
                "dataset": dataset,
                "question_id": qid,
                "signal": signal,
                "n_runs": len(vals),
                "std_across_runs": vals.std(),
                "range_across_runs": vals.max() - vals.min(),
            })

    inv_df = pd.DataFrame(invariance_rows) if invariance_rows else pd.DataFrame()
    if not inv_df.empty:
        varying = inv_df[inv_df["range_across_runs"] > 1e-6]
        if len(varying) > 0:
            n_varying = varying["signal"].nunique()
            log.warning(f"  {n_varying} prompt signals vary across runs - see invariance table")

    corr_rows = []
    for model in sorted(prompt_source["model"].unique()):
        for dataset in sorted(prompt_source["dataset"].unique()):
            cell = prompt_source[(prompt_source["model"] == model) & (prompt_source["dataset"] == dataset)]
            if len(cell) < 20:
                continue
            for signal in prompt_signals:
                vals = cell[[signal, diff_col]].dropna()
                if len(vals) < 20:
                    continue
                if vals[signal].std() < 1e-12 or vals[diff_col].std() < 1e-12:
                    continue
                rho, p = spearmanr(vals[signal], vals[diff_col])
                corr_rows.append({
                    "model": model,
                    "dataset": dataset,
                    "signal": signal,
                    "spearman_rho": rho,
                    "p_value": p,
                    "n_prompts": len(vals),
                })

    corr_df = pd.DataFrame(corr_rows) if corr_rows else pd.DataFrame()

    pooled_rows = []
    for dataset in sorted(prompt_source["dataset"].unique()):
        cell = prompt_source[prompt_source["dataset"] == dataset]
        pooled = cell.groupby("question_id").agg(
            {signal: "mean" for signal in prompt_signals} | {diff_col: "first"}
        )
        if len(pooled) < 20:
            continue
        for signal in prompt_signals:
            vals = pooled[[signal, diff_col]].dropna()
            if len(vals) < 20:
                continue
            if vals[signal].std() < 1e-12 or vals[diff_col].std() < 1e-12:
                continue
            rho, p = spearmanr(vals[signal], vals[diff_col])
            pooled_rows.append({
                "dataset": dataset,
                "signal": signal,
                "spearman_rho": rho,
                "p_value": p,
                "n_prompts": len(vals),
            })

    pooled_df = pd.DataFrame(pooled_rows) if pooled_rows else pd.DataFrame()

    hard_easy_rows = []
    for dataset in sorted(prompt_source["dataset"].unique()):
        cell = prompt_source[prompt_source["dataset"] == dataset]
        prompt_diff = cell.groupby("question_id")[diff_col].first().sort_values(ascending=False)
        top_hard = prompt_diff.head(10)
        top_easy = prompt_diff.tail(10)
        for qid, diff in top_hard.items():
            hard_easy_rows.append({"dataset": dataset, "question_id": qid, "difficulty": diff, "category": "hardest"})
        for qid, diff in top_easy.items():
            hard_easy_rows.append({"dataset": dataset, "question_id": qid, "difficulty": diff, "category": "easiest"})

    hard_easy_df = pd.DataFrame(hard_easy_rows) if hard_easy_rows else pd.DataFrame()

    tables = {}
    if not corr_df.empty:
        tables["difficulty_correlations"] = corr_df
    if not pooled_df.empty:
        tables["difficulty_correlations_pooled"] = pooled_df
    if not inv_df.empty:
        tables["invariance_check"] = inv_df
    if not hard_easy_df.empty:
        tables["hardest_easiest"] = hard_easy_df

    save_focus_outputs(out_dir, "focus5", tables, summary={}, logger=log)

    if not corr_df.empty:
        for dataset in sorted(prompt_source["dataset"].unique()):
            sub = corr_df[corr_df["dataset"] == dataset]
            if len(sub) < 3:
                continue
            fig, ax = plt.subplots(figsize=(10, 6))
            pivot = sub.pivot_table(index="signal", columns="model", values="spearman_rho")
            if not pivot.empty:
                sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdBu_r", center=0, ax=ax, linewidths=0.5)
                ax.set_title(f"Prompt Signal -> Difficulty Correlation - {dataset}")
                fig.tight_layout()
                save_figure(fig, fig_dir / f"focus5_difficulty_corr_{dataset}.png")
                save_table(pivot.reset_index(), out_dir / "tables" / f"focus5_heatmap_{dataset}.csv")
            else:
                plt.close(fig)

    if not corr_df.empty:
        print("\n  Prompt signal -> Difficulty correlations:")
        for _, row in corr_df.sort_values("spearman_rho", key=abs, ascending=False).head(10).iterrows():
            print(
                f"    {row['signal']:30s} {row['model']:10s} {row['dataset']:10s} "
                f"rho={row['spearman_rho']:+.3f} p={row['p_value']:.4f}"
            )

    summary = {
        "description": "Prompt-level signal correlation with empirical difficulty",
        "difficulty_definition": "1 - mean(correct) pooled across all models and temperatures",
        "n_per_model_correlations": len(corr_df),
        "n_pooled_correlations": len(pooled_df),
        "invariance_issues": int((inv_df["range_across_runs"] > 1e-6).sum()) if not inv_df.empty else 0,
        "top_correlations": corr_df.sort_values("spearman_rho", key=abs, ascending=False).head(5).to_dict("records") if not corr_df.empty else [],
    }
    save_json(summary, out_dir / "summary.json")

    return corr_df


def focus6_cross_model(df: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    """
    Cross-model behavioral alignment analysis.
    
    Three sub-analyses:
      A. Difficulty agreement (do models find the same prompts hard?)
      B. Signal alignment (do signals behave similarly across models?)
      C. Fingerprint structure (noted as redundant with named features)
    """
    log.info("Focus 6: Cross-model behavioral alignment")
    out_dir = ensure_dir(base_dir / "focus_06_cross_model")
    fig_dir = ensure_dir(out_dir / "figures")

    cdf = correctness_subset(df)
    models = sorted(cdf["model"].unique())
    if len(models) < 2:
        log.warning("  Need ≥2 models")
        save_json({"finding": "Fewer than 2 models available"}, out_dir / "summary.json")
        return pd.DataFrame()

    # ── A. Difficulty agreement (Spearman on pass rates) ──
    model_diff = cdf.groupby(["model", "dataset", "question_id"])["correct"].mean().reset_index()
    model_diff.columns = ["model", "dataset", "question_id", "pass_rate"]

    agreement_rows = []
    tables_dir = ensure_dir(out_dir / "tables")
    for dataset in sorted(cdf["dataset"].unique()):
        sub = model_diff[model_diff["dataset"] == dataset]
        pivot = sub.pivot_table(index="question_id", columns="model", values="pass_rate")
        if pivot.shape[1] < 2:
            continue

        corr = pivot.corr(method="spearman")
        save_table(corr.reset_index(), out_dir / "tables" / f"focus6_difficulty_corr_{dataset}.csv")

        # Pairwise correlations
        for i, m1 in enumerate(corr.columns):
            for j, m2 in enumerate(corr.columns):
                if j <= i:
                    continue
                agreement_rows.append({
                    "dataset": dataset, "model_a": m1, "model_b": m2,
                    "spearman_rho": corr.loc[m1, m2],
                    "n_questions": pivot[[m1, m2]].dropna().shape[0],
                })

        # Heatmap
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(corr, annot=True, fmt=".3f", cmap="viridis",
                    vmin=0, vmax=1, ax=ax, linewidths=0.5)
        ax.set_title(f"Cross-Model Difficulty Agreement (Spearman) — {dataset}")
        fig.tight_layout()
        save_figure(fig, fig_dir / f"focus6_agreement_{dataset}.png")

        print(f"\n  {dataset} — pass-rate Spearman correlation:")
        print(corr.to_string())

    agree_df = pd.DataFrame(agreement_rows) if agreement_rows else pd.DataFrame()

    # ── B. Signal alignment (per-question signal correlation across models) ──
    signal_align_rows = []
    key_signals = ["entropy_mean", "surprisal_mean", "margin_mean", "risk_score"]
    for dataset in sorted(cdf["dataset"].unique()):
        for sig in key_signals:
            if sig not in cdf.columns:
                continue
            agg = cdf[cdf["dataset"] == dataset].groupby(
                ["model", "question_id"]
            )[sig].mean().reset_index()
            pivot = agg.pivot_table(index="question_id", columns="model", values=sig)
            if pivot.shape[1] < 2:
                continue
            corr = pivot.corr(method="spearman")
            for i, m1 in enumerate(corr.columns):
                for j, m2 in enumerate(corr.columns):
                    if j <= i:
                        continue
                    signal_align_rows.append({
                        "dataset": dataset, "signal": sig,
                        "model_a": m1, "model_b": m2,
                        "spearman_rho": corr.loc[m1, m2],
                    })

    signal_df = pd.DataFrame(signal_align_rows) if signal_align_rows else pd.DataFrame()

    # ── C. Fingerprint note ──
    fp_cols = [c for c in df.columns if c.startswith("fp_")]
    fp_note = (
        f"Fingerprint has {len(fp_cols)} dimensions. Inspection shows these are "
        "concatenated summary statistics (fp_00=entropy_mean, fp_01=entropy_std, etc.), "
        "NOT a learned embedding. Fingerprint analysis is therefore redundant with "
        "named feature analysis and should not be treated as independent evidence."
    )

    # ── Save tables ──
    tables = {}
    if not agree_df.empty:
        tables["difficulty_agreement"] = agree_df
    if not signal_df.empty:
        tables["signal_alignment"] = signal_df

    save_focus_outputs(out_dir, "focus6", tables, summary={}, logger=log)

    summary = {
        "description": "Cross-model behavioral alignment",
        "difficulty_agreement": agree_df.to_dict("records") if not agree_df.empty else [],
        "signal_alignment_summary": signal_df.to_dict("records") if not signal_df.empty else [],
        "fingerprint_note": fp_note,
    }
    save_json(summary, out_dir / "summary.json")

    return agree_df


# ── Ranking Evaluation ───────────────────────────────────────

def ranking_evaluation(df: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    """
    Best-of-k selection using run-varying CoreVital signals.

    Restricts to signals that vary across candidate runs for the same prompt,
    infers signal direction per model/dataset cell, and treats ties as a
    random tie-break expectation instead of picking the first row.
    """
    log.info("Ranking evaluation: best-of-k selection")
    out_dir = ensure_dir(base_dir / "ranking")
    fig_dir = ensure_dir(out_dir / "figures")

    from sklearn.metrics import roc_auc_score

    cdf = correctness_subset(df)
    signals = get_analysis_signals(cdf)

    results = []
    for model in sorted(cdf["model"].unique()):
        for dataset in sorted(cdf["dataset"].unique()):
            cell = cdf[(cdf["model"] == model) & (cdf["dataset"] == dataset)]
            prompts = cell.groupby("question_id")

            for signal in signals:
                if signal not in cell.columns:
                    continue

                signal_vals = cell[signal].astype(float)
                valid_mask = signal_vals.notna()
                if valid_mask.sum() < 50:
                    continue
                if signal_vals[valid_mask].std() < 1e-12:
                    continue

                try:
                    raw_auc = roc_auc_score(
                        cell.loc[valid_mask, "correct"].astype(int),
                        signal_vals[valid_mask],
                    )
                except Exception:
                    raw_auc = 0.5
                higher_is_better = raw_auc >= 0.5

                random_correct = 0.0
                signal_correct = 0.0
                oracle_correct = 0.0
                n_prompts = 0
                n_ties = 0

                for _, group in prompts:
                    group = group.dropna(subset=[signal, "correct"])
                    if len(group) < 3:
                        continue
                    if group["correct"].sum() == 0:
                        continue

                    random_correct += float(group["correct"].mean())
                    oracle_correct += 1.0

                    if higher_is_better:
                        best_val = group[signal].max()
                        candidates = group[group[signal] == best_val]
                    else:
                        best_val = group[signal].min()
                        candidates = group[group[signal] == best_val]

                    if len(candidates) > 1:
                        n_ties += 1
                    signal_correct += float(candidates["correct"].mean())
                    n_prompts += 1

                if n_prompts < 10:
                    continue

                random_acc = random_correct / n_prompts
                signal_acc = signal_correct / n_prompts
                oracle_acc = oracle_correct / n_prompts

                results.append({
                    "model": model,
                    "dataset": dataset,
                    "signal": signal,
                    "direction": "higher_better" if higher_is_better else "lower_better",
                    "random_acc": random_acc,
                    "signal_acc": signal_acc,
                    "oracle_acc": oracle_acc,
                    "lift_vs_random": signal_acc - random_acc,
                    "lift_vs_oracle": signal_acc - oracle_acc,
                    "n_prompts": n_prompts,
                    "n_ties": n_ties,
                    "tie_rate": n_ties / n_prompts,
                })

    res_df = pd.DataFrame(results)

    temp_rows = []
    if "temperature" in cdf.columns and cdf["temperature"].nunique() > 1:
        for temp in sorted(cdf["temperature"].dropna().unique()):
            t_cell = cdf[cdf["temperature"] == temp]
            for model in sorted(t_cell["model"].unique()):
                for dataset in sorted(t_cell["dataset"].unique()):
                    cell = t_cell[(t_cell["model"] == model) & (t_cell["dataset"] == dataset)]
                    for signal in ["entropy_mean", "surprisal_mean", "margin_mean", "risk_score"]:
                        if signal not in cell.columns:
                            continue

                        vals = cell[signal].astype(float)
                        valid_mask = vals.notna()
                        if valid_mask.sum() < 20 or vals[valid_mask].std() < 1e-12:
                            continue
                        try:
                            raw_auc = roc_auc_score(
                                cell.loc[valid_mask, "correct"].astype(int),
                                vals[valid_mask],
                            )
                        except Exception:
                            raw_auc = 0.5
                        higher_is_better = raw_auc >= 0.5

                        prompts = cell.groupby("question_id")
                        sc = 0.0
                        rc = 0.0
                        np_ = 0
                        for _, group in prompts:
                            group = group.dropna(subset=[signal, "correct"])
                            if len(group) < 2 or group["correct"].sum() == 0:
                                continue
                            rc += float(group["correct"].mean())
                            if higher_is_better:
                                best_val = group[signal].max()
                                candidates = group[group[signal] == best_val]
                            else:
                                best_val = group[signal].min()
                                candidates = group[group[signal] == best_val]
                            sc += float(candidates["correct"].mean())
                            np_ += 1

                        if np_ >= 5:
                            temp_rows.append({
                                "temperature": temp,
                                "model": model,
                                "dataset": dataset,
                                "signal": signal,
                                "direction": "higher_better" if higher_is_better else "lower_better",
                                "random_acc": rc / np_,
                                "signal_acc": sc / np_,
                                "lift": (sc - rc) / np_,
                                "n_prompts": np_,
                            })

    temp_rank_df = pd.DataFrame(temp_rows) if temp_rows else pd.DataFrame()

    tables = {"ranking_results": res_df}
    if not temp_rank_df.empty:
        tables["ranking_by_temperature"] = temp_rank_df
    save_focus_outputs(out_dir, "ranking", tables, summary={}, logger=log)

    if not res_df.empty:
        for model in sorted(cdf["model"].unique()):
            sub = res_df[res_df["model"] == model].sort_values("lift_vs_random", ascending=False)
            print(f"\n  {model} - Top ranking signals:")
            for _, r in sub.head(10).iterrows():
                icon = "*" if r["lift_vs_random"] > 0.02 else " "
                print(
                    f"    {icon} {r['signal']:35s} {r['dataset']:10s} "
                    f"random={r['random_acc']:.1%} signal={r['signal_acc']:.1%} "
                    f"lift={r['lift_vs_random']:+.1%} ties={r['tie_rate']:.0%}"
                )

    if not res_df.empty:
        for dataset in sorted(cdf["dataset"].unique()):
            sub = res_df[res_df["dataset"] == dataset]
            top_sigs = (
                sub.groupby("signal")["lift_vs_random"].mean()
                .sort_values(ascending=False).head(15).index
            )
            plot_data = sub[sub["signal"].isin(top_sigs)]
            if len(plot_data) < 5:
                continue

            fig, ax = plt.subplots(figsize=(12, 6))
            pivot = plot_data.pivot_table(
                index="signal", columns="model", values="lift_vs_random"
            ).sort_values(by=plot_data["model"].unique()[0], ascending=True)
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdBu_r", center=0, ax=ax, linewidths=0.5)
            ax.set_title(f"Best-of-k Lift vs Random - {dataset}")
            fig.tight_layout()
            save_figure(fig, fig_dir / f"ranking_lift_{dataset}.png")

    summary = {
        "description": "Best-of-k selection using run-varying CoreVital signals",
        "restriction": "Only prompts with at least 1 correct run",
        "n_results": len(res_df),
        "top_10_by_lift": res_df.sort_values("lift_vs_random", ascending=False).head(10).to_dict("records") if not res_df.empty else [],
    }
    save_json(summary, out_dir / "summary.json")

    return res_df


def signal_ablation(df: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    """
    Incremental signal ablation: entropy -> baseline -> full.

    Uses grouped CV by question_id so prompt-level telemetry is evaluated on
    unseen prompts rather than leaked across repeated runs.
    """
    log.info("Signal ablation: entropy -> baseline -> full")
    out_dir = ensure_dir(base_dir / "ablation")
    fig_dir = ensure_dir(out_dir / "figures")

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    cdf = correctness_subset(df)
    signals = get_analysis_signals(cdf, include_prompt=True, include_performance=False)

    prompt_sigs = get_prompt_signals(signals)
    early_signals = [s for s in signals if s.startswith("early")]
    health_sigs = [
        s for s in signals
        if any(s.startswith(prefix) for prefix in [
            "risk_", "failure_", "n_warning", "n_compound", "max_compound", "cs_"
        ])
    ]

    tier_defs = [
        ("T1: entropy_only", ["entropy_mean"]),
        ("T2: confidence_baseline", BASELINE_FEATURES),
        ("T3: + prompt_signals", BASELINE_FEATURES + prompt_sigs),
        ("T4: + early_window", BASELINE_FEATURES + prompt_sigs + early_signals),
        ("T5: + health_signals", BASELINE_FEATURES + prompt_sigs + early_signals + health_sigs),
        ("T6: full_corevital", signals),
    ]

    results = []
    fold_results = []

    for model in sorted(cdf["model"].unique()):
        for dataset in sorted(cdf["dataset"].unique()):
            cell = cdf[(cdf["model"] == model) & (cdf["dataset"] == dataset)].dropna(subset=["correct"])
            y = cell["correct"].astype(int).to_numpy()
            if len(np.unique(y)) < 2 or len(y) < 100:
                continue

            splits, n_groups, split_kind = iter_group_splits(cell, y, ("question_id",), desired_splits=5)
            if not splits:
                log.warning(f"  Skipping {model}/{dataset}: insufficient prompt groups for grouped CV")
                continue

            for tier_name, tier_cols in tier_defs:
                cols = [c for c in tier_cols if c in cell.columns]
                if not cols:
                    continue

                X = cell[cols]
                fold_aucs = []

                for fold_idx, (tr, te) in enumerate(splits):
                    y_train = y[tr]
                    y_test = y[te]
                    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                        continue
                    try:
                        model_cv = HistGradientBoostingClassifier(
                            max_iter=100,
                            max_depth=4,
                            class_weight="balanced",
                            random_state=42,
                        )
                        model_cv.fit(X.iloc[tr], y_train)
                        proba = model_cv.predict_proba(X.iloc[te])[:, 1]
                        auc = roc_auc_score(y_test, proba)
                        fold_aucs.append(auc)
                        fold_results.append({
                            "model": model,
                            "dataset": dataset,
                            "tier": tier_name,
                            "fold": fold_idx,
                            "auroc": auc,
                            "splitter": split_kind,
                            "n_prompt_groups": n_groups,
                        })
                    except Exception as exc:
                        log.warning(f"  Ablation fold error: {model}/{dataset}/{tier_name}: {exc}")

                if fold_aucs:
                    mean_auc = float(np.mean(fold_aucs))
                    std_auc = float(np.std(fold_aucs))
                    se_auc = std_auc / max(1.0, np.sqrt(len(fold_aucs)))
                    results.append({
                        "model": model,
                        "dataset": dataset,
                        "tier": tier_name,
                        "n_features": len(cols),
                        "mean_auroc": mean_auc,
                        "std_auroc": std_auc,
                        "ci_lower": mean_auc - 1.96 * se_auc,
                        "ci_upper": mean_auc + 1.96 * se_auc,
                        "n_folds": len(fold_aucs),
                        "n_samples": len(y),
                        "n_prompt_groups": n_groups,
                        "splitter": split_kind,
                    })

    res_df = pd.DataFrame(results)
    fold_df = pd.DataFrame(fold_results)

    tables = {"ablation_summary": res_df}
    if not fold_df.empty:
        tables["ablation_folds"] = fold_df
    save_focus_outputs(out_dir, "ablation", tables, summary={}, logger=log)

    if not res_df.empty:
        for model in sorted(cdf["model"].unique()):
            sub = res_df[res_df["model"] == model]
            print(f"\n  {model}:")
            for _, row in sub.iterrows():
                print(
                    f"    {row['tier']:30s} ({row['n_features']:3d} feat) "
                    f"AUROC={row['mean_auroc']:.3f} +/-{row['std_auroc']:.3f} "
                    f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}] "
                    f"groups={int(row['n_prompt_groups'])}"
                )

    if not res_df.empty:
        for dataset in sorted(cdf["dataset"].unique()):
            sub = res_df[res_df["dataset"] == dataset]
            if len(sub) < 3:
                continue
            fig, ax = plt.subplots(figsize=(10, 6))
            for model in sorted(sub["model"].unique()):
                m_data = sub[sub["model"] == model]
                ax.errorbar(
                    m_data["tier"],
                    m_data["mean_auroc"],
                    yerr=m_data["std_auroc"],
                    marker="o",
                    linewidth=2,
                    capsize=4,
                    label=model,
                )
            ax.set_xlabel("Feature Tier")
            ax.set_ylabel("AUROC (grouped prompt CV)")
            ax.set_title(f"Signal Ablation - {dataset}")
            ax.tick_params(axis="x", rotation=25)
            ax.legend()
            fig.tight_layout()
            save_figure(fig, fig_dir / f"ablation_curve_{dataset}.png")

    summary = {
        "description": "Incremental signal ablation using grouped prompt-level CV",
        "tiers": [name for name, _ in tier_defs],
        "n_results": len(res_df),
        "results": res_df.to_dict("records") if not res_df.empty else [],
    }
    save_json(summary, out_dir / "summary.json")

    return res_df


def format_failure_analysis(df: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    """
    Analyze format failures as a separate prediction target.
    
    Instead of globally discarding format failures, this section asks:
    can CoreVital signals predict format failure?
    """
    log.info("Format failure analysis")
    out_dir = ensure_dir(base_dir / "format_failure")
    fig_dir = ensure_dir(out_dir / "figures")

    if "format_failure" not in df.columns:
        save_json({"finding": "No format_failure column"}, out_dir / "summary.json")
        return pd.DataFrame()

    ff_rate = df["format_failure"].fillna(False).astype(bool)
    total_ff = ff_rate.sum()
    log.info(f"  {total_ff} format failures out of {len(df)} runs ({total_ff/len(df):.1%})")

    if total_ff < 10:
        save_json({
            "finding": f"Only {total_ff} format failures — too few for analysis",
            "total_runs": len(df),
        }, out_dir / "summary.json")
        return pd.DataFrame()

    # ── Prevalence by model/dataset ──
    prev = df.groupby(["model", "dataset"]).agg(
        n_runs=("format_failure", "count"),
        n_failures=("format_failure", lambda x: x.fillna(False).astype(bool).sum()),
    ).reset_index()
    prev["failure_rate"] = prev["n_failures"] / prev["n_runs"]

    # ── Signal association with format failure ──
    signals = get_analysis_signals(df)
    assoc_rows = []
    y = ff_rate.astype(int).values

    for model in sorted(df["model"].unique()):
        m_mask = df["model"] == model
        m_y = y[m_mask]
        if len(np.unique(m_y)) < 2:
            continue
        for sig in signals:
            if sig not in df.columns:
                continue
            vals = df.loc[m_mask, sig].values.astype(float)
            mask = ~np.isnan(vals)
            if mask.sum() < 50:
                continue
            if np.std(vals[mask]) < 1e-12:
                continue
            try:
                r, p = pointbiserialr(m_y[mask], vals[mask])
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(m_y[mask], vals[mask])
                pred_power = max(auc, 1 - auc)
            except Exception:
                continue

            assoc_rows.append({
                "model": model, "signal": sig,
                "correlation": r, "p_value": p,
                "raw_auroc": auc, "predictive_power": pred_power,
                "direction": "higher_more_failure" if auc >= 0.5 else "lower_more_failure",
                "n": int(mask.sum()),
            })

    assoc_df = pd.DataFrame(assoc_rows) if assoc_rows else pd.DataFrame()

    # ── Length analysis ──
    length_rows = []
    for model in sorted(df["model"].unique()):
        m = df[df["model"] == model]
        ff = m[ff_rate[m.index]]
        ok = m[~ff_rate[m.index]]
        if len(ff) < 5 or len(ok) < 5:
            continue
        length_rows.append({
            "model": model,
            "ff_mean_tokens": ff["generated_tokens"].mean(),
            "ok_mean_tokens": ok["generated_tokens"].mean(),
            "ff_median_tokens": ff["generated_tokens"].median(),
            "ok_median_tokens": ok["generated_tokens"].median(),
        })
    length_df = pd.DataFrame(length_rows) if length_rows else pd.DataFrame()

    # ── Save tables ──
    tables = {"prevalence": prev}
    if not assoc_df.empty:
        tables["signal_association"] = assoc_df
    if not length_df.empty:
        tables["length_comparison"] = length_df

    save_focus_outputs(out_dir, "ff", tables, summary={}, logger=log)

    # ── Console ──
    print(f"\n  Format failure prevalence:")
    for _, r in prev.iterrows():
        print(f"    {r['model']:15s} {r['dataset']:10s} "
              f"{r['n_failures']}/{r['n_runs']} ({r['failure_rate']:.1%})")
    if not assoc_df.empty:
        print(f"\n  Top signals predicting format failure:")
        for _, r in assoc_df.sort_values("predictive_power", ascending=False).head(10).iterrows():
            print(f"    {r['signal']:35s} {r['model']:10s} PP={r['predictive_power']:.3f}")

    # ── Summary ──
    summary = {
        "description": "Format failure analysis — can CoreVital signals predict output format failures?",
        "total_format_failures": int(total_ff),
        "total_runs": len(df),
        "prevalence": prev.to_dict("records"),
        "top_predictive_signals": assoc_df.sort_values("predictive_power", ascending=False).head(5).to_dict("records") if not assoc_df.empty else [],
    }
    save_json(summary, out_dir / "summary.json")

    return assoc_df


# ── Risk Calibration Analysis (NEW) ──────────────────────────

def risk_calibration(df: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    """
    Analyze whether risk_score and failure_risk are calibrated.
    
    Questions:
      - Does risk_score = 0.8 actually mean ~80% failure rate?
      - How discretized are the scores?
      - What is the saturation rate?
    """
    log.info("Risk calibration analysis")
    out_dir = ensure_dir(base_dir / "risk_calibration")
    fig_dir = ensure_dir(out_dir / "figures")

    cdf = correctness_subset(df)

    calibration_rows = []
    for score_col in ["risk_score", "failure_risk"]:
        if score_col not in cdf.columns:
            continue

        valid = cdf[[score_col, "correct", "model"]].dropna()
        if len(valid) < 100:
            continue

        # Unique value analysis
        n_unique = valid[score_col].nunique()
        value_counts = valid[score_col].value_counts().head(20)

        # Saturation
        n_zero = (valid[score_col] == 0.0).sum()
        n_one = (valid[score_col] == 1.0).sum()
        n_total = len(valid)

        # Binned calibration
        n_bins = min(10, n_unique)
        if n_bins >= 3:
            valid["score_bin"] = pd.qcut(valid[score_col], q=n_bins, duplicates="drop")
            cal = valid.groupby("score_bin").agg(
                mean_score=(score_col, "mean"),
                failure_rate=("correct", lambda x: 1 - x.mean()),
                n=("correct", "count"),
            ).reset_index()
            cal["score_name"] = score_col
            calibration_rows.append(cal)

            # Calibration plot
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(cal["mean_score"], cal["failure_rate"],
                    "o-", linewidth=2, markersize=8)
            ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="perfect calibration")
            ax.set_xlabel(f"Mean {score_col}")
            ax.set_ylabel("Actual Failure Rate")
            ax.set_title(f"Calibration: {score_col}")
            ax.legend()
            fig.tight_layout()
            save_figure(fig, fig_dir / f"calibration_{score_col}.png")

            # Per-model calibration
            for model in sorted(valid["model"].unique()):
                m_valid = valid[valid["model"] == model].copy()
                if m_valid[score_col].nunique() < 3:
                    continue
                m_valid["score_bin"] = pd.qcut(m_valid[score_col], q=min(5, m_valid[score_col].nunique()), duplicates="drop")
                m_cal = m_valid.groupby("score_bin").agg(
                    mean_score=(score_col, "mean"),
                    failure_rate=("correct", lambda x: 1 - x.mean()),
                    n=("correct", "count"),
                ).reset_index()
                m_cal["model"] = model
                m_cal["score_name"] = score_col
                calibration_rows.append(m_cal)

        # Summary stats per model
        for model in sorted(valid["model"].unique()):
            m = valid[valid["model"] == model]
            calibration_rows_meta = {
                "score": score_col, "model": model,
                "n": len(m), "n_unique_values": m[score_col].nunique(),
                "pct_zero": n_zero / n_total * 100,
                "pct_one": n_one / n_total * 100,
                "mean": m[score_col].mean(),
                "std": m[score_col].std(),
            }

    cal_df = pd.concat(calibration_rows, ignore_index=True) if calibration_rows else pd.DataFrame()

    # ── Score distribution ──
    dist_rows = []
    for score_col in ["risk_score", "failure_risk"]:
        if score_col not in cdf.columns:
            continue
        for model in sorted(cdf["model"].unique()):
            vals = cdf[cdf["model"] == model][score_col].dropna()
            if len(vals) < 10:
                continue
            dist_rows.append({
                "score": score_col, "model": model,
                "n": len(vals), "n_unique": vals.nunique(),
                "pct_zero": (vals == 0).mean() * 100,
                "pct_one": (vals == 1).mean() * 100,
                "mean": vals.mean(), "std": vals.std(),
                "median": vals.median(),
                "p10": vals.quantile(0.1), "p90": vals.quantile(0.9),
            })

    dist_df = pd.DataFrame(dist_rows) if dist_rows else pd.DataFrame()

    # ── Save tables ──
    tables = {}
    if not cal_df.empty:
        tables["calibration"] = cal_df
    if not dist_df.empty:
        tables["score_distributions"] = dist_df

    save_focus_outputs(out_dir, "risk_cal", tables, summary={}, logger=log)

    # ── Console ──
    if not dist_df.empty:
        print(f"\n  Score distribution summary:")
        for _, r in dist_df.iterrows():
            print(f"    {r['score']:15s} {r['model']:10s} "
                  f"unique={r['n_unique']:3d} %zero={r['pct_zero']:.1f} %one={r['pct_one']:.1f} "
                  f"mean={r['mean']:.3f}")

    # ── Summary ──
    summary = {
        "description": "Risk score calibration and saturation analysis",
        "scores_analyzed": ["risk_score", "failure_risk"],
        "distributions": dist_df.to_dict("records") if not dist_df.empty else [],
        "calibration_available": not cal_df.empty,
    }
    save_json(summary, out_dir / "summary.json")

    return cal_df


# ── Three-Way Outcome Profiling (NEW) ────────────────────────

def outcome_signal_profiling(df: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    """
    Three-way signal profiling: correct vs incorrect vs format failure.
    
    Instead of treating correctness and format failure as separate binary
    problems, this section asks: do the three outcome categories have
    distinct signal signatures?
    
    Outputs:
      1. Per-signal distribution table across three outcome categories
         (mean, median, std, p10, p90 for each category)
      2. One-vs-rest AUROC for each signal × category
         (does this signal specifically identify THIS outcome?)
      3. Signal specificity table: which signals are specific to one
         category vs shared across failure modes
      4. Outcome profile: what does a "typical" correct/incorrect/format-failure
         run look like across the top signals
    """
    log.info("Outcome signal profiling: correct vs incorrect vs format failure")
    out_dir = ensure_dir(base_dir / "outcome_profiling")
    fig_dir = ensure_dir(out_dir / "figures")
    tables_dir = ensure_dir(out_dir / "tables")

    from sklearn.metrics import roc_auc_score

    # ── Build three-way outcome column ──
    df = df.copy()
    def classify_outcome(row):
        if row.get("format_failure") == True or row.get("format_failure") == 1:
            return "format_failure"
        if row.get("correct") == True or row.get("correct") == 1:
            return "correct"
        if row.get("correct") == False or row.get("correct") == 0:
            return "incorrect"
        return None

    df["outcome"] = df.apply(classify_outcome, axis=1)
    df = df[df["outcome"].notna()].copy()

    outcome_counts = df["outcome"].value_counts()
    log.info(f"  Outcome distribution: {outcome_counts.to_dict()}")

    if outcome_counts.min() < 20:
        log.warning("  Too few samples in one category for reliable analysis")

    signals = get_analysis_signals(df)

    # Exclude timing signals (they dominate by scale, not by information)
    timing_exclude = {
        "total_run_ms", "inference_ms", "overhead_ms", "unattributed_ms",
        "known_non_inference_ms", "report_build_ms", "tokenize_ms",
        "overhead_pct", "inference_pct",
    }
    signals = [s for s in signals if s not in timing_exclude]

    # ── 1. Per-signal distribution across outcome categories ──
    dist_rows = []
    for sig in signals:
        if sig not in df.columns:
            continue
        for outcome in ["correct", "incorrect", "format_failure"]:
            vals = df[df["outcome"] == outcome][sig].dropna()
            if len(vals) < 10:
                continue
            dist_rows.append({
                "signal": sig, "outcome": outcome,
                "n": len(vals),
                "mean": vals.mean(), "median": vals.median(),
                "std": vals.std(),
                "p10": vals.quantile(0.1), "p90": vals.quantile(0.9),
            })

    dist_df = pd.DataFrame(dist_rows) if dist_rows else pd.DataFrame()

    # ── 2. One-vs-rest AUROC per signal per outcome ──
    # For each outcome, how well does each signal discriminate
    # "this outcome" from "everything else"?
    ovr_rows = []
    for model in sorted(df["model"].unique()):
        m_df = df[df["model"] == model]

        for outcome in ["correct", "incorrect", "format_failure"]:
            y = (m_df["outcome"] == outcome).astype(int).values
            if len(np.unique(y)) < 2:
                continue
            n_pos = y.sum()
            if n_pos < 10:
                continue

            for sig in signals:
                if sig not in m_df.columns:
                    continue
                vals = m_df[sig].values.astype(float)
                mask = ~np.isnan(vals)
                if mask.sum() < 50:
                    continue
                if np.std(vals[mask]) < 1e-12:
                    continue

                try:
                    raw_auc = roc_auc_score(y[mask], vals[mask])
                    pred_power = max(raw_auc, 1.0 - raw_auc)
                    direction = "higher" if raw_auc >= 0.5 else "lower"
                except Exception:
                    continue

                ovr_rows.append({
                    "model": model, "outcome": outcome, "signal": sig,
                    "raw_auroc": raw_auc,
                    "predictive_power": pred_power,
                    "direction": direction,
                    "n_total": int(mask.sum()),
                    "n_outcome": int(n_pos),
                })

    ovr_df = pd.DataFrame(ovr_rows) if ovr_rows else pd.DataFrame()

    # ── 3. Signal specificity analysis ──
    # A signal is "specific" to an outcome if its PP for that outcome
    # is much higher than for the other two.
    specificity_rows = []
    if not ovr_df.empty:
        for model in sorted(ovr_df["model"].unique()):
            m_ovr = ovr_df[ovr_df["model"] == model]
            for sig in m_ovr["signal"].unique():
                sig_data = m_ovr[m_ovr["signal"] == sig]
                if len(sig_data) < 2:
                    continue

                pp_by_outcome = {}
                dir_by_outcome = {}
                for _, row in sig_data.iterrows():
                    pp_by_outcome[row["outcome"]] = row["predictive_power"]
                    dir_by_outcome[row["outcome"]] = row["direction"]

                if len(pp_by_outcome) < 2:
                    continue

                # Find which outcome this signal is best at detecting
                best_outcome = max(pp_by_outcome, key=pp_by_outcome.get)
                best_pp = pp_by_outcome[best_outcome]
                other_pps = [v for k, v in pp_by_outcome.items() if k != best_outcome]
                mean_other_pp = np.mean(other_pps) if other_pps else 0.5

                specificity_rows.append({
                    "model": model, "signal": sig,
                    "best_outcome": best_outcome,
                    "best_pp": best_pp,
                    "mean_other_pp": mean_other_pp,
                    "specificity_gap": best_pp - mean_other_pp,
                    "pp_correct": pp_by_outcome.get("correct", None),
                    "pp_incorrect": pp_by_outcome.get("incorrect", None),
                    "pp_format_failure": pp_by_outcome.get("format_failure", None),
                    "dir_correct": dir_by_outcome.get("correct", None),
                    "dir_incorrect": dir_by_outcome.get("incorrect", None),
                    "dir_format_failure": dir_by_outcome.get("format_failure", None),
                })

    spec_df = pd.DataFrame(specificity_rows) if specificity_rows else pd.DataFrame()

    # ── 4. Outcome profiles: what does a "typical" run look like? ──
    # Pick the top 15 most discriminative signals and show the
    # mean value for each outcome category, per model
    profile_rows = []
    if not ovr_df.empty:
        # Find top signals by max predictive power across any outcome
        top_sigs = (
            ovr_df.groupby("signal")["predictive_power"]
            .max().sort_values(ascending=False).head(20).index
        )

        for model in sorted(df["model"].unique()):
            m_df = df[df["model"] == model]
            for sig in top_sigs:
                if sig not in m_df.columns:
                    continue
                for outcome in ["correct", "incorrect", "format_failure"]:
                    vals = m_df[m_df["outcome"] == outcome][sig].dropna()
                    if len(vals) < 5:
                        continue
                    profile_rows.append({
                        "model": model, "signal": sig, "outcome": outcome,
                        "mean": vals.mean(), "median": vals.median(),
                        "std": vals.std(), "n": len(vals),
                    })

    profile_df = pd.DataFrame(profile_rows) if profile_rows else pd.DataFrame()

    # ── Save tables ──
    tables = {}
    if not dist_df.empty:
        tables["distributions"] = dist_df
    if not ovr_df.empty:
        tables["one_vs_rest_auroc"] = ovr_df
    if not spec_df.empty:
        tables["signal_specificity"] = spec_df
    if not profile_df.empty:
        tables["outcome_profiles"] = profile_df

    save_focus_outputs(out_dir, "outcome", tables, summary={}, logger=log)

    # ── Console output ──
    if not spec_df.empty:
        for model in sorted(spec_df["model"].unique()):
            m_spec = spec_df[spec_df["model"] == model]
            print(f"\n  {model} — Signal specificity (top by gap):")
            top = m_spec.sort_values("specificity_gap", ascending=False).head(15)
            for _, r in top.iterrows():
                pp_c = f"{r['pp_correct']:.3f}" if r['pp_correct'] is not None else "  n/a"
                pp_i = f"{r['pp_incorrect']:.3f}" if r['pp_incorrect'] is not None else "  n/a"
                pp_f = f"{r['pp_format_failure']:.3f}" if r['pp_format_failure'] is not None else "  n/a"
                print(f"    {r['signal']:40s} best={r['best_outcome']:15s} "
                      f"PP: correct={pp_c} incorrect={pp_i} ff={pp_f} "
                      f"gap={r['specificity_gap']:.3f}")

    # ── Figures ──

    # Heatmap: per-model, show PP for top signals across the three outcomes
    if not ovr_df.empty:
        for model in sorted(ovr_df["model"].unique()):
            m_ovr = ovr_df[ovr_df["model"] == model]
            # Get top 20 signals by max PP
            top_sigs = (
                m_ovr.groupby("signal")["predictive_power"]
                .max().sort_values(ascending=False).head(20).index
            )
            sub = m_ovr[m_ovr["signal"].isin(top_sigs)]
            pivot = sub.pivot_table(
                index="signal", columns="outcome", values="predictive_power"
            )
            # Sort by max across outcomes
            pivot["_max"] = pivot.max(axis=1)
            pivot = pivot.sort_values("_max", ascending=False).drop("_max", axis=1)

            if pivot.empty:
                continue

            fig, ax = plt.subplots(figsize=(8, max(6, len(pivot) * 0.35)))
            sns.heatmap(
                pivot, annot=True, fmt=".3f", cmap="magma",
                vmin=0.5, vmax=0.85, ax=ax, linewidths=0.5,
            )
            ax.set_title(f"Signal Predictive Power by Outcome — {model}")
            fig.tight_layout()
            save_figure(fig, fig_dir / f"outcome_heatmap_{model}.png")

            # Save heatmap source
            save_table(pivot.reset_index(), tables_dir / f"outcome_heatmap_{model}.csv")

    # Bar chart: for the most specific signals, show how the mean value
    # differs across the three outcomes
    if not profile_df.empty:
        for model in sorted(profile_df["model"].unique()):
            m_prof = profile_df[profile_df["model"] == model]
            # Get top 10 signals
            if spec_df.empty:
                continue
            m_spec = spec_df[spec_df["model"] == model]
            top_sigs = m_spec.sort_values("specificity_gap", ascending=False).head(8)["signal"].values
            sub = m_prof[m_prof["signal"].isin(top_sigs)]
            if len(sub) < 6:
                continue

            # Normalize values per signal for visualization (z-score across outcomes)
            plot_rows = []
            for sig in top_sigs:
                sig_data = sub[sub["signal"] == sig]
                all_mean = sig_data["mean"].mean()
                all_std = sig_data["mean"].std()
                if all_std < 1e-12:
                    continue
                for _, r in sig_data.iterrows():
                    plot_rows.append({
                        "signal": sig, "outcome": r["outcome"],
                        "z_mean": (r["mean"] - all_mean) / all_std,
                    })

            if not plot_rows:
                continue
            plot_df = pd.DataFrame(plot_rows)

            fig, ax = plt.subplots(figsize=(10, 6))
            # Use outcome as hue
            outcome_colors = {
                "correct": CORRECT_COLOR,
                "incorrect": INCORRECT_COLOR,
                "format_failure": "#ffd43b",
            }
            sns.barplot(
                data=plot_df, x="signal", y="z_mean", hue="outcome",
                palette=outcome_colors, ax=ax,
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
            ax.set_ylabel("Normalized Signal Value (z-score)")
            ax.set_title(f"Outcome Signal Profiles — {model}")
            ax.axhline(0, color="white", linewidth=0.5, alpha=0.5)
            fig.tight_layout()
            save_figure(fig, fig_dir / f"outcome_profiles_{model}.png")

    # ── Summary ──
    # Build concise summary of key findings
    specificity_summary = []
    if not spec_df.empty:
        # For each model, which signals are most specific to each outcome?
        for model in sorted(spec_df["model"].unique()):
            m_spec = spec_df[spec_df["model"] == model]
            for outcome in ["correct", "incorrect", "format_failure"]:
                outcome_sigs = m_spec[m_spec["best_outcome"] == outcome]
                top = outcome_sigs.sort_values("specificity_gap", ascending=False).head(3)
                for _, r in top.iterrows():
                    specificity_summary.append({
                        "model": model, "outcome": outcome,
                        "signal": r["signal"],
                        "best_pp": r["best_pp"],
                        "specificity_gap": r["specificity_gap"],
                    })

    summary = {
        "description": (
            "Three-way outcome profiling: do correct, incorrect, and format-failure "
            "runs have distinct signal signatures, or is failure a single spectrum?"
        ),
        "outcome_counts": outcome_counts.to_dict(),
        "n_signals_tested": len(signals),
        "n_one_vs_rest_results": len(ovr_df),
        "n_specificity_results": len(spec_df),
        "key_finding_preview": (
            "Signals that are specific to ONE outcome (high gap) indicate distinct "
            "failure modes. Signals that are strong for multiple outcomes indicate "
            "a shared failure spectrum."
        ),
        "top_specific_signals": specificity_summary,
    }
    save_json(summary, out_dir / "summary.json")

    print(f"\n  Outcome profiling: {len(ovr_df)} one-vs-rest tests, "
          f"{len(spec_df)} specificity comparisons")

    return spec_df


# ── Signal Redundancy Mapping ────────────────────────────────

def signal_redundancy_mapping(df: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    """
    Identify which signals are redundant and which are independent.
    
    Computes pairwise correlation among all analysis signals, clusters
    them into families, and identifies the best representative from each.
    This directly informs a minimal feature set for a v2 risk formula.
    
    Outputs:
      1. Pairwise correlation matrix for top signals
      2. Signal families (clusters of r > 0.8)
      3. Recommended minimal signal set (one per family)
    """
    log.info("Signal redundancy mapping")
    out_dir = ensure_dir(base_dir / "signal_redundancy")
    fig_dir = ensure_dir(out_dir / "figures")
    tables_dir = ensure_dir(out_dir / "tables")

    cdf = correctness_subset(df)
    signals = get_analysis_signals(cdf)

    # Exclude timing signals
    timing_exclude = {
        "total_run_ms", "inference_ms", "overhead_ms", "unattributed_ms",
        "known_non_inference_ms", "report_build_ms", "tokenize_ms",
        "overhead_pct", "inference_pct",
    }
    signals = [s for s in signals if s not in timing_exclude]

    # Compute correlation matrix across all data (pooled)
    sig_data = cdf[signals].dropna(axis=1, how="all")
    available_sigs = list(sig_data.columns)
    corr_matrix = sig_data.corr(method="spearman")

    # ── Cluster into families using greedy agglomeration ──
    # Two signals are in the same family if |r| > threshold
    threshold = 0.80
    assigned = {}  # signal -> family_id
    families = {}  # family_id -> list of signals
    family_id = 0

    # Sort signals by their mean absolute correlation with others (most connected first)
    mean_abs_corr = corr_matrix.abs().mean().sort_values(ascending=False)

    for sig in mean_abs_corr.index:
        if sig in assigned:
            continue
        # Start new family
        families[family_id] = [sig]
        assigned[sig] = family_id

        # Find all unassigned signals correlated above threshold
        for other in mean_abs_corr.index:
            if other in assigned:
                continue
            if sig in corr_matrix.columns and other in corr_matrix.columns:
                r = corr_matrix.loc[sig, other]
                if abs(r) > threshold:
                    families[family_id].append(other)
                    assigned[other] = family_id

        family_id += 1

    # ── Pick best representative per family ──
    # Use mean predictive power from Focus 1 results if available
    focus1_path = base_dir / "focus_01_metric_correlation" / "tables" / "focus1_results.csv"
    if focus1_path.exists():
        f1 = pd.read_csv(focus1_path)
        sig_pp = f1.groupby("signal")["predictive_power"].mean().to_dict()
    else:
        sig_pp = {}

    family_rows = []
    representatives = []
    for fid, members in sorted(families.items()):
        # Rank members by predictive power
        member_pp = [(s, sig_pp.get(s, 0.5)) for s in members]
        member_pp.sort(key=lambda x: -x[1])
        best = member_pp[0][0]
        best_pp = member_pp[0][1]
        representatives.append(best)

        for s in members:
            family_rows.append({
                "family_id": fid,
                "signal": s,
                "is_representative": s == best,
                "mean_predictive_power": sig_pp.get(s, None),
                "family_size": len(members),
                "representative": best,
            })

    family_df = pd.DataFrame(family_rows)

    # ── Summary table ──
    summary_rows = []
    for fid, members in sorted(families.items()):
        member_pp = [(s, sig_pp.get(s, 0.5)) for s in members]
        member_pp.sort(key=lambda x: -x[1])
        best = member_pp[0]
        summary_rows.append({
            "family_id": fid,
            "representative": best[0],
            "representative_pp": best[1],
            "family_size": len(members),
            "members": ", ".join(members[:5]) + ("..." if len(members) > 5 else ""),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("representative_pp", ascending=False)

    # ── Save tables ──
    tables = {
        "families": family_df,
        "family_summary": summary_df,
    }
    # Save correlation matrix for top 30 signals
    top_sigs = mean_abs_corr.head(30).index.tolist()
    top_corr = corr_matrix.loc[top_sigs, top_sigs]
    save_table(top_corr.reset_index(), tables_dir / "redundancy_correlation_top30.csv")

    save_focus_outputs(out_dir, "redundancy", tables, summary={}, logger=log)

    # ── Console ──
    print(f"\n  Signal redundancy: {len(available_sigs)} signals → "
          f"{len(families)} families (threshold |r| > {threshold})")
    print(f"\n  Recommended minimal set ({len(representatives)} signals):")
    for _, r in summary_df.head(20).iterrows():
        print(f"    Family {r['family_id']:2d}: {r['representative']:40s} "
              f"PP={r['representative_pp']:.3f} (family size={r['family_size']})")

    # ── Figure: correlation heatmap of top 30 ──
    if len(top_sigs) >= 5:
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(
            top_corr, annot=False, cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, ax=ax, linewidths=0.3,
        )
        ax.set_title(f"Signal Correlation (Spearman) — Top 30 by connectivity")
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        save_figure(fig, fig_dir / f"redundancy_corr_heatmap.png")

    # ── Summary JSON ──
    summary = {
        "description": (
            "Signal redundancy mapping. Clusters correlated signals (|r| > 0.80) "
            "into families and identifies the best representative per family."
        ),
        "n_signals_analyzed": len(available_sigs),
        "n_families": len(families),
        "correlation_threshold": threshold,
        "recommended_minimal_set": [
            {"signal": r["representative"], "pp": r["representative_pp"],
             "family_size": r["family_size"]}
            for _, r in summary_df.head(20).iterrows()
        ],
    }
    save_json(summary, out_dir / "summary.json")

    return family_df


# ── Confidence-Correctness Calibration ───────────────────────

def confidence_correctness_calibration(df: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    """
    When the model is confident, is it actually right?
    
    Uses margin_mean as a confidence proxy. Partitions runs into
    confidence quintiles and shows accuracy per bin per model.
    Then asks: do confident-but-wrong runs have different CoreVital
    signatures than confident-and-right runs?
    
    This answers: "can CoreVital catch failures that the model's own
    confidence doesn't flag?"
    """
    log.info("Confidence-correctness calibration")
    out_dir = ensure_dir(base_dir / "confidence_calibration")
    fig_dir = ensure_dir(out_dir / "figures")
    tables_dir = ensure_dir(out_dir / "tables")

    from sklearn.metrics import roc_auc_score

    cdf = correctness_subset(df)

    # Use margin_mean as primary confidence proxy (higher = more confident)
    conf_signal = "margin_mean"
    if conf_signal not in cdf.columns:
        log.warning("  margin_mean not available")
        save_json({"finding": "margin_mean not available"}, out_dir / "summary.json")
        return pd.DataFrame()

    # ── 1. Binned confidence → accuracy ──
    bin_rows = []
    for model in sorted(cdf["model"].unique()):
        for dataset in sorted(cdf["dataset"].unique()):
            cell = cdf[(cdf["model"] == model) & (cdf["dataset"] == dataset)]
            cell = cell.dropna(subset=[conf_signal, "correct"])
            if len(cell) < 50:
                continue

            # Quintile bins
            try:
                cell["conf_bin"] = pd.qcut(cell[conf_signal], q=5, duplicates="drop")
            except ValueError:
                continue

            for bin_label, group in cell.groupby("conf_bin"):
                bin_rows.append({
                    "model": model, "dataset": dataset,
                    "confidence_bin": str(bin_label),
                    "bin_median_confidence": group[conf_signal].median(),
                    "n": len(group),
                    "accuracy": group["correct"].mean(),
                    "n_correct": int(group["correct"].sum()),
                    "n_incorrect": int((~group["correct"].astype(bool)).sum()),
                })

    bin_df = pd.DataFrame(bin_rows) if bin_rows else pd.DataFrame()

    # ── 2. Confident-but-wrong analysis ──
    # Focus on the top confidence quintile per model/dataset
    confident_wrong_rows = []
    signals = get_analysis_signals(cdf)
    timing_exclude = {
        "total_run_ms", "inference_ms", "overhead_ms", "unattributed_ms",
        "known_non_inference_ms", "report_build_ms", "tokenize_ms",
        "overhead_pct", "inference_pct",
    }
    signals = [s for s in signals if s not in timing_exclude]

    for model in sorted(cdf["model"].unique()):
        for dataset in sorted(cdf["dataset"].unique()):
            cell = cdf[(cdf["model"] == model) & (cdf["dataset"] == dataset)]
            cell = cell.dropna(subset=[conf_signal, "correct"])
            if len(cell) < 100:
                continue

            # Top 20% most confident runs
            threshold = cell[conf_signal].quantile(0.80)
            confident = cell[cell[conf_signal] >= threshold]
            conf_correct = confident[confident["correct"] == True]
            conf_wrong = confident[confident["correct"] == False]

            if len(conf_correct) < 10 or len(conf_wrong) < 10:
                continue

            for sig in signals:
                if sig not in cell.columns or sig == conf_signal:
                    continue
                c_vals = conf_correct[sig].dropna()
                w_vals = conf_wrong[sig].dropna()
                if len(c_vals) < 10 or len(w_vals) < 10:
                    continue

                # Can this signal distinguish confident-correct from confident-wrong?
                all_vals = confident[sig].dropna()
                y = confident.loc[all_vals.index, "correct"].astype(int)
                if len(np.unique(y)) < 2:
                    continue
                if np.std(all_vals.values) < 1e-12:
                    continue

                try:
                    auc = roc_auc_score(y, all_vals.values)
                    pp = max(auc, 1 - auc)
                except Exception:
                    continue

                confident_wrong_rows.append({
                    "model": model, "dataset": dataset, "signal": sig,
                    "predictive_power_in_confident": pp,
                    "confident_correct_mean": c_vals.mean(),
                    "confident_wrong_mean": w_vals.mean(),
                    "delta": w_vals.mean() - c_vals.mean(),
                    "n_confident_correct": len(c_vals),
                    "n_confident_wrong": len(w_vals),
                })

    cw_df = pd.DataFrame(confident_wrong_rows) if confident_wrong_rows else pd.DataFrame()

    # ── Save tables ──
    tables = {}
    if not bin_df.empty:
        tables["confidence_bins"] = bin_df
    if not cw_df.empty:
        tables["confident_wrong_signals"] = cw_df

    save_focus_outputs(out_dir, "confidence", tables, summary={}, logger=log)

    # ── Console ──
    if not bin_df.empty:
        print(f"\n  Confidence → Accuracy (top quintile):")
        for model in sorted(bin_df["model"].unique()):
            m_bins = bin_df[bin_df["model"] == model].sort_values("bin_median_confidence")
            top_bin = m_bins.iloc[-1] if len(m_bins) > 0 else None
            if top_bin is not None:
                print(f"    {model}: most confident bin accuracy = {top_bin['accuracy']:.1%} "
                      f"(n={int(top_bin['n'])})")

    if not cw_df.empty:
        print(f"\n  Signals that catch confident-but-wrong runs:")
        top_cw = cw_df.sort_values("predictive_power_in_confident", ascending=False).head(10)
        for _, r in top_cw.iterrows():
            print(f"    {r['signal']:40s} {r['model']:10s}/{r['dataset']:10s} "
                  f"PP={r['predictive_power_in_confident']:.3f}")

    # ── Figure: confidence calibration curves ──
    if not bin_df.empty:
        for dataset in sorted(bin_df["dataset"].unique()):
            sub = bin_df[bin_df["dataset"] == dataset]
            if len(sub) < 5:
                continue
            fig, ax = plt.subplots(figsize=(8, 6))
            for model in sorted(sub["model"].unique()):
                m_data = sub[sub["model"] == model].sort_values("bin_median_confidence")
                ax.plot(m_data["bin_median_confidence"], m_data["accuracy"],
                        "o-", label=model, linewidth=2, markersize=6)
            ax.set_xlabel("Model Confidence (margin_mean)")
            ax.set_ylabel("Actual Accuracy")
            ax.set_title(f"Confidence Calibration — {dataset}")
            ax.legend()
            ax.set_ylim(0, 1.05)
            fig.tight_layout()
            save_figure(fig, fig_dir / f"confidence_calibration_{dataset}.png")

    # ── Summary ──
    summary = {
        "description": (
            "Confidence-correctness calibration. Tests whether model confidence "
            "(margin_mean) reliably predicts accuracy, and identifies CoreVital signals "
            "that catch confident-but-wrong runs."
        ),
        "n_confidence_bins": len(bin_df),
        "n_confident_wrong_tests": len(cw_df),
        "top_signals_catching_confident_wrong": (
            cw_df.sort_values("predictive_power_in_confident", ascending=False)
            .head(5).to_dict("records")
        ) if not cw_df.empty else [],
    }
    save_json(summary, out_dir / "summary.json")

    return cw_df


# ── Temperature Effect Decomposition ─────────────────────────

def temperature_decomposition(df: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    """
    Do CoreVital signals shift systematically with temperature?
    Does predictive power change between temp 0.7 and 0.8?
    
    If signals are equally strong at both temperatures, that's a
    robustness finding. If they differ, it's a limitation.
    """
    log.info("Temperature effect decomposition")
    out_dir = ensure_dir(base_dir / "temperature_effects")
    fig_dir = ensure_dir(out_dir / "figures")

    from sklearn.metrics import roc_auc_score

    cdf = correctness_subset(df)

    if "temperature" not in cdf.columns or cdf["temperature"].nunique() < 2:
        log.warning("  Temperature data not available or single-valued")
        save_json({"finding": "Insufficient temperature variation"}, out_dir / "summary.json")
        return pd.DataFrame()

    temps = sorted(cdf["temperature"].dropna().unique())
    signals = get_analysis_signals(cdf)
    timing_exclude = {
        "total_run_ms", "inference_ms", "overhead_ms", "unattributed_ms",
        "known_non_inference_ms", "report_build_ms", "tokenize_ms",
        "overhead_pct", "inference_pct",
    }
    signals = [s for s in signals if s not in timing_exclude]

    # ── 1. Signal distribution shift with temperature ──
    shift_rows = []
    for model in sorted(cdf["model"].unique()):
        for dataset in sorted(cdf["dataset"].unique()):
            cell = cdf[(cdf["model"] == model) & (cdf["dataset"] == dataset)]
            for sig in signals[:50]:  # Top 50 to keep manageable
                if sig not in cell.columns:
                    continue
                means_by_temp = {}
                for temp in temps:
                    vals = cell[cell["temperature"] == temp][sig].dropna()
                    if len(vals) < 20:
                        continue
                    means_by_temp[temp] = vals.mean()

                if len(means_by_temp) >= 2:
                    temp_vals = list(means_by_temp.values())
                    shift_rows.append({
                        "model": model, "dataset": dataset, "signal": sig,
                        **{f"mean_temp_{t}": v for t, v in means_by_temp.items()},
                        "temp_delta": temp_vals[-1] - temp_vals[0],
                    })

    shift_df = pd.DataFrame(shift_rows) if shift_rows else pd.DataFrame()

    # ── 2. Predictive power by temperature ──
    pp_rows = []
    # Focus on the most important signals
    key_signals = [
        "entropy_mean", "surprisal_mean", "margin_mean", "risk_score",
        "l2_norm_slope", "compound_density_per_100t",
        "early10_surprisal_mean", "early10_entropy_mean",
        "concentration_min_mean", "focused_head_rate",
    ]
    key_signals = [s for s in key_signals if s in cdf.columns]

    for model in sorted(cdf["model"].unique()):
        for dataset in sorted(cdf["dataset"].unique()):
            for temp in temps:
                cell = cdf[
                    (cdf["model"] == model) &
                    (cdf["dataset"] == dataset) &
                    (cdf["temperature"] == temp)
                ]
                y = cell["correct"].astype(int).values
                if len(np.unique(y)) < 2 or len(y) < 30:
                    continue

                for sig in key_signals:
                    if sig not in cell.columns:
                        continue
                    vals = cell[sig].values.astype(float)
                    mask = ~np.isnan(vals)
                    if mask.sum() < 30 or np.std(vals[mask]) < 1e-12:
                        continue
                    try:
                        auc = roc_auc_score(y[mask], vals[mask])
                        pp = max(auc, 1 - auc)
                    except Exception:
                        continue

                    pp_rows.append({
                        "model": model, "dataset": dataset,
                        "temperature": temp, "signal": sig,
                        "predictive_power": pp, "n": int(mask.sum()),
                    })

    pp_df = pd.DataFrame(pp_rows) if pp_rows else pd.DataFrame()

    # ── 3. Robustness: PP difference between temperatures ──
    robustness_rows = []
    if not pp_df.empty and len(temps) == 2:
        for model in sorted(pp_df["model"].unique()):
            for dataset in sorted(pp_df["dataset"].unique()):
                for sig in pp_df["signal"].unique():
                    sub = pp_df[
                        (pp_df["model"] == model) &
                        (pp_df["dataset"] == dataset) &
                        (pp_df["signal"] == sig)
                    ]
                    if len(sub) != 2:
                        continue
                    pp_vals = sub.sort_values("temperature")["predictive_power"].values
                    robustness_rows.append({
                        "model": model, "dataset": dataset, "signal": sig,
                        f"pp_temp_{temps[0]}": pp_vals[0],
                        f"pp_temp_{temps[1]}": pp_vals[1],
                        "pp_delta": pp_vals[1] - pp_vals[0],
                        "pp_abs_delta": abs(pp_vals[1] - pp_vals[0]),
                    })

    robust_df = pd.DataFrame(robustness_rows) if robustness_rows else pd.DataFrame()

    # ── Save tables ──
    tables = {}
    if not shift_df.empty:
        tables["signal_shift"] = shift_df
    if not pp_df.empty:
        tables["pp_by_temperature"] = pp_df
    if not robust_df.empty:
        tables["robustness"] = robust_df

    save_focus_outputs(out_dir, "temperature", tables, summary={}, logger=log)

    # ── Console ──
    if not robust_df.empty:
        print(f"\n  Temperature robustness (PP stability between {temps[0]} and {temps[1]}):")
        mean_delta = robust_df["pp_abs_delta"].mean()
        print(f"    Mean |PP delta| across all signals: {mean_delta:.4f}")
        least_robust = robust_df.sort_values("pp_abs_delta", ascending=False).head(5)
        print(f"    Least robust signals:")
        for _, r in least_robust.iterrows():
            print(f"      {r['signal']:35s} {r['model']:10s}/{r['dataset']:10s} "
                  f"Δ={r['pp_delta']:+.3f}")
        most_robust = robust_df.sort_values("pp_abs_delta", ascending=True).head(5)
        print(f"    Most robust signals:")
        for _, r in most_robust.iterrows():
            print(f"      {r['signal']:35s} {r['model']:10s}/{r['dataset']:10s} "
                  f"Δ={r['pp_delta']:+.3f}")

    # ── Figure ──
    if not pp_df.empty and len(temps) == 2:
        for dataset in sorted(pp_df["dataset"].unique()):
            sub = pp_df[pp_df["dataset"] == dataset]
            if len(sub) < 10:
                continue
            fig, ax = plt.subplots(figsize=(10, 6))
            pivot = sub.pivot_table(
                index=["model", "signal"], columns="temperature",
                values="predictive_power"
            ).dropna()
            if pivot.empty or pivot.shape[1] < 2:
                plt.close(fig)
                continue

            ax.scatter(pivot.iloc[:, 0], pivot.iloc[:, 1], alpha=0.6, s=30)
            lims = [0.45, max(pivot.max().max() + 0.05, 0.85)]
            ax.plot(lims, lims, "--", color="gray", alpha=0.5, label="perfect agreement")
            ax.set_xlabel(f"PP at temp={temps[0]}")
            ax.set_ylabel(f"PP at temp={temps[1]}")
            ax.set_title(f"Signal Predictive Power: temp {temps[0]} vs {temps[1]} — {dataset}")
            ax.legend()
            fig.tight_layout()
            save_figure(fig, fig_dir / f"temperature_pp_scatter_{dataset}.png")

    # ── Summary ──
    summary = {
        "description": (
            "Temperature effect decomposition. Tests whether CoreVital signal "
            "distributions and predictive power change between temperatures."
        ),
        "temperatures": [float(t) for t in temps],
        "n_shift_tests": len(shift_df),
        "n_pp_tests": len(pp_df),
        "mean_pp_abs_delta": float(robust_df["pp_abs_delta"].mean()) if not robust_df.empty else None,
        "finding": (
            "Robust" if (not robust_df.empty and robust_df["pp_abs_delta"].mean() < 0.03)
            else "Some sensitivity" if not robust_df.empty
            else "Insufficient data"
        ),
    }
    save_json(summary, out_dir / "summary.json")

    return robust_df


# ── Difficulty-Stratified Signal Analysis ─────────────────────

def difficulty_stratified_analysis(df: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    """
    Are CoreVital signals equally predictive on easy vs hard questions?
    
    Stratifies by difficulty band and re-computes predictive power.
    If signals only work on medium-difficulty questions, that's a
    practical limitation worth knowing.
    """
    log.info("Difficulty-stratified signal analysis")
    out_dir = ensure_dir(base_dir / "difficulty_stratified")
    fig_dir = ensure_dir(out_dir / "figures")

    from sklearn.metrics import roc_auc_score

    cdf = correctness_subset(df)
    diff_col = "empirical_difficulty_pooled" if "empirical_difficulty_pooled" in cdf.columns else "question_difficulty"

    if diff_col not in cdf.columns:
        log.warning("  No difficulty data available")
        save_json({"finding": "No difficulty data"}, out_dir / "summary.json")
        return pd.DataFrame()

    # Define difficulty bands
    bands = [
        ("easy", 0.0, 0.33),
        ("medium", 0.33, 0.67),
        ("hard", 0.67, 1.01),
    ]

    key_signals = [
        "entropy_mean", "surprisal_mean", "margin_mean",
        "l2_norm_slope", "compound_density_per_100t",
        "early10_surprisal_mean", "early10_entropy_mean",
        "risk_score", "concentration_min_mean", "focused_head_rate",
        "entropy_std", "margin_std", "perplexity_mean",
        "collapsed_rate_mean", "hidden_max_abs_last_layer_mean",
    ]
    key_signals = [s for s in key_signals if s in cdf.columns]

    results = []
    for model in sorted(cdf["model"].unique()):
        for dataset in sorted(cdf["dataset"].unique()):
            cell = cdf[(cdf["model"] == model) & (cdf["dataset"] == dataset)]
            cell = cell.dropna(subset=[diff_col, "correct"])

            for band_name, lo, hi in bands:
                band = cell[(cell[diff_col] >= lo) & (cell[diff_col] < hi)]
                y = band["correct"].astype(int).values
                if len(np.unique(y)) < 2 or len(y) < 30:
                    continue

                for sig in key_signals:
                    if sig not in band.columns:
                        continue
                    vals = band[sig].values.astype(float)
                    mask = ~np.isnan(vals)
                    if mask.sum() < 30 or np.std(vals[mask]) < 1e-12:
                        continue

                    try:
                        auc = roc_auc_score(y[mask], vals[mask])
                        pp = max(auc, 1 - auc)
                    except Exception:
                        continue

                    results.append({
                        "model": model, "dataset": dataset,
                        "difficulty_band": band_name,
                        "diff_lo": lo, "diff_hi": hi,
                        "signal": sig,
                        "predictive_power": pp,
                        "n": int(mask.sum()),
                        "class_balance": float(y[mask].mean()),
                    })

    res_df = pd.DataFrame(results)

    # ── Compute PP variation across bands ──
    variation_rows = []
    if not res_df.empty:
        for (model, dataset, sig), grp in res_df.groupby(["model", "dataset", "signal"]):
            if len(grp) < 2:
                continue
            pp_vals = grp.set_index("difficulty_band")["predictive_power"]
            variation_rows.append({
                "model": model, "dataset": dataset, "signal": sig,
                "pp_easy": pp_vals.get("easy", None),
                "pp_medium": pp_vals.get("medium", None),
                "pp_hard": pp_vals.get("hard", None),
                "pp_range": pp_vals.max() - pp_vals.min(),
                "best_band": pp_vals.idxmax(),
            })

    var_df = pd.DataFrame(variation_rows) if variation_rows else pd.DataFrame()

    # ── Save tables ──
    tables = {"stratified_results": res_df}
    if not var_df.empty:
        tables["band_variation"] = var_df

    save_focus_outputs(out_dir, "difficulty_strat", tables, summary={}, logger=log)

    # ── Console ──
    if not var_df.empty:
        print(f"\n  Difficulty-stratified PP (signals with largest band variation):")
        top_var = var_df.sort_values("pp_range", ascending=False).head(10)
        for _, r in top_var.iterrows():
            easy = f"{r['pp_easy']:.3f}" if r['pp_easy'] is not None else " n/a "
            med = f"{r['pp_medium']:.3f}" if r['pp_medium'] is not None else " n/a "
            hard = f"{r['pp_hard']:.3f}" if r['pp_hard'] is not None else " n/a "
            print(f"    {r['signal']:35s} {r['model']:10s}/{r['dataset']:10s} "
                  f"easy={easy} med={med} hard={hard} range={r['pp_range']:.3f} "
                  f"best={r['best_band']}")

        mean_range = var_df["pp_range"].mean()
        print(f"\n  Mean PP range across difficulty bands: {mean_range:.3f}")

    # ── Figure: PP by difficulty band ──
    if not res_df.empty:
        for dataset in sorted(res_df["dataset"].unique()):
            sub = res_df[res_df["dataset"] == dataset]
            # Show top 8 signals by mean PP
            top_sigs = sub.groupby("signal")["predictive_power"].mean().sort_values(ascending=False).head(8).index
            plot_data = sub[sub["signal"].isin(top_sigs)]
            if len(plot_data) < 6:
                continue

            fig, ax = plt.subplots(figsize=(12, 6))
            band_order = ["easy", "medium", "hard"]
            sns.barplot(
                data=plot_data, x="signal", y="predictive_power",
                hue="difficulty_band", hue_order=band_order,
                palette={"easy": CORRECT_COLOR, "medium": "#ffd43b", "hard": INCORRECT_COLOR},
                ax=ax,
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
            ax.set_ylabel("Predictive Power")
            ax.set_title(f"Signal Strength by Question Difficulty — {dataset}")
            ax.axhline(0.5, color="white", linewidth=0.5, alpha=0.3)
            fig.tight_layout()
            save_figure(fig, fig_dir / f"difficulty_stratified_{dataset}.png")

    # ── Summary ──
    summary = {
        "description": (
            "Difficulty-stratified signal analysis. Tests whether CoreVital signals "
            "are equally predictive on easy, medium, and hard questions."
        ),
        "n_results": len(res_df),
        "mean_pp_range_across_bands": float(var_df["pp_range"].mean()) if not var_df.empty else None,
        "finding": (
            "Signals are difficulty-invariant" if (not var_df.empty and var_df["pp_range"].mean() < 0.05)
            else "Some difficulty sensitivity" if not var_df.empty
            else "Insufficient data"
        ),
        "most_variable_signals": (
            var_df.sort_values("pp_range", ascending=False).head(5).to_dict("records")
        ) if not var_df.empty else [],
    }
    save_json(summary, out_dir / "summary.json")

    return var_df


# ── Summary & Manifest ───────────────────────────────────────

def write_key_findings(
    base_dir: Path,
    focus1_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
):
    """
    Write key_findings.json — the single most important AI-oriented artifact.
    
    Contains the top findings from each section with enough numeric detail
    for an LLM to write accurate docs without hallucinating.
    """
    findings = {
        "experiment": {
            "design": "Pass@k (k=10) under sampling (temp 0.7 + 0.8)",
            "datasets": "GSM8K (200 prompts) + HumanEval (164 prompts)",
            "models": "Llama-3.1-8B-Instruct, Qwen-2.5-7B-Instruct, Mistral-7B-Instruct-v0.3, Mixtral-8x7B-Instruct-v0.1",
            "traces": "~14,500 instrumented generation traces",
        },
        "sections": {},
    }

    # Load each section's summary.json
    for section_dir in sorted(base_dir.iterdir()):
        if not section_dir.is_dir():
            continue
        summary_path = section_dir / "summary.json"
        if summary_path.exists():
            import json
            with open(summary_path) as f:
                findings["sections"][section_dir.name] = json.load(f)

    save_json(findings, base_dir / "key_findings.json")
    log.info(f"Saved key_findings.json ({len(findings['sections'])} sections)")


def write_results_summary(df: pd.DataFrame, base_dir: Path):
    """Write a comprehensive human-readable markdown summary."""
    path = base_dir / "RESULTS_SUMMARY.md"

    cdf = correctness_subset(df)
    with open(path, "w") as f:
        f.write("# CoreVital Validation Experiment — Results Summary\n\n")
        f.write("## Experiment Design\n\n")
        f.write("- **Design:** Pass@k (k=10) under sampling (temp 0.7 + 0.8)\n")
        f.write("- **Datasets:** GSM8K (200 prompts) + HumanEval (164 prompts)\n")
        f.write("- **Models:** Llama-3.1-8B, Qwen-2.5-7B, Mistral-7B, Mixtral-8x7B (8-bit)\n")
        f.write(f"- **Total traces:** {len(df)}\n")
        f.write(f"- **After format-failure exclusion:** {len(cdf)}\n\n")

        f.write("## Per-Model Accuracy\n\n")
        f.write("| Model | Dataset | Prompts | Runs | Accuracy | Format Fail % |\n")
        f.write("|-------|---------|---------|------|----------|---------------|\n")
        for (m, d), g in df.groupby(["model", "dataset"]):
            acc = g["correct"].mean() if g["correct"].notna().any() else float("nan")
            ff = g["format_failure"].fillna(False).astype(bool).mean() * 100
            f.write(f"| {m} | {d} | {g['question_id'].nunique()} | {len(g)} | {acc:.1%} | {ff:.1f}% |\n")

        f.write("\n## Section Summaries\n\n")
        f.write("Each section below has its own directory with CSV tables, figures, and a summary.json.\n\n")

        sections = [
            ("Focus 1: Metric Correlation", "focus_01_metric_correlation"),
            ("Focus 2: MoE vs Dense", "focus_02_moe_vs_dense"),
            ("Focus 3: Self-Consistency", "focus_03_self_consistency"),
            ("Focus 4: Layer Analysis", "focus_04_layer_analysis"),
            ("Focus 5: Difficulty Profiling", "focus_05_difficulty"),
            ("Focus 6: Cross-Model Alignment", "focus_06_cross_model"),
            ("Ranking Evaluation", "ranking"),
            ("Signal Ablation", "ablation"),
            ("Format Failure Analysis", "format_failure"),
            ("Risk Calibration", "risk_calibration"),
            ("Outcome Signal Profiling", "outcome_profiling"),
            ("Signal Redundancy Mapping", "signal_redundancy"),
            ("Confidence-Correctness Calibration", "confidence_calibration"),
            ("Temperature Effects", "temperature_effects"),
            ("Difficulty-Stratified Analysis", "difficulty_stratified"),
        ]
        for title, dirname in sections:
            summary_path = base_dir / dirname / "summary.json"
            f.write(f"### {title}\n\n")
            f.write(f"See `{dirname}/summary.json` for machine-readable findings.\n\n")
            if summary_path.exists():
                import json
                with open(summary_path) as sf:
                    s = json.load(sf)
                desc = s.get("description", "")
                if desc:
                    f.write(f"_{desc}_\n\n")

        f.write("## Artifact Index\n\n")
        f.write("- `key_findings.json` — All section findings in one file (AI-oriented)\n")
        f.write("- `global_manifest.json` — Complete artifact listing\n")
        for title, dirname in sections:
            f.write(f"- `{dirname}/` — {title}\n")

    log.info(f"Saved {path}")


def write_global_manifest(base_dir: Path):
    """List all generated artifacts."""
    artifacts = []
    for p in sorted(base_dir.rglob("*")):
        if p.is_file():
            artifacts.append({
                "path": str(p.relative_to(base_dir)),
                "size_bytes": p.stat().st_size,
                "type": p.suffix,
            })

    manifest = build_manifest(
        script_name="analyze.py",
        inputs={"features": str((RESULTS_DIR / "features.parquet").relative_to(REPO_ROOT))},
        outputs=[a["path"] for a in artifacts],
        row_counts={"artifacts": len(artifacts)},
        extra={"artifacts": artifacts},
    )
    save_json(manifest, base_dir / "global_manifest.json")
    log.info(f"Saved global_manifest.json ({len(artifacts)} artifacts)")


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CoreVital validation experiment analysis"
    )
    parser.add_argument("--features", type=Path, default=RESULTS_DIR / "features.parquet")
    parser.add_argument("--prompt-level", type=Path, default=RESULTS_DIR / "prompt_level.parquet")
    args = parser.parse_args()

    # ── Setup ──
    apply_dark_theme()
    ensure_dir(ANALYSIS_DIR)

    # ── Load & enrich ──
    df = load_data(args.features)
    df = enrich_features(df)
    prompt_df = pd.DataFrame()
    if args.prompt_level.exists():
        prompt_df = load_data(args.prompt_level, label="prompt_level")
    else:
        log.warning(f"Prompt-level parquet not found: {args.prompt_level}; falling back to grouped run data")

    print(f"\n{'='*60}")
    print(f"COREVITAL VALIDATION ANALYSIS")
    print(f"  {len(df)} rows, {df['model'].nunique()} models, {df['dataset'].nunique()} datasets")
    print(f"  Format failures: {df['format_failure'].fillna(False).astype(bool).sum()}")
    print(f"  Features: {len(df.columns)} columns")
    print(f"{'='*60}")

    # ── Run all analyses ──
    focus1_df = focus1_metric_correlation(df, ANALYSIS_DIR)
    focus2_df = focus2_moe_vs_dense(df, ANALYSIS_DIR)
    focus3_df = focus3_self_consistency(df, ANALYSIS_DIR)
    focus4_df = focus4_layer_analysis(df, ANALYSIS_DIR)
    focus5_df = focus5_difficulty(df, prompt_df, ANALYSIS_DIR)
    focus6_df = focus6_cross_model(df, ANALYSIS_DIR)
    ranking_df = ranking_evaluation(df, ANALYSIS_DIR)
    ablation_df = signal_ablation(df, ANALYSIS_DIR)
    ff_df = format_failure_analysis(df, ANALYSIS_DIR)
    cal_df = risk_calibration(df, ANALYSIS_DIR)
    outcome_df = outcome_signal_profiling(df, ANALYSIS_DIR)
    redundancy_df = signal_redundancy_mapping(df, ANALYSIS_DIR)
    confidence_df = confidence_correctness_calibration(df, ANALYSIS_DIR)
    temp_df = temperature_decomposition(df, ANALYSIS_DIR)
    diff_strat_df = difficulty_stratified_analysis(df, ANALYSIS_DIR)

    # ── Write summaries ──
    write_key_findings(ANALYSIS_DIR, focus1_df, ranking_df, ablation_df)
    write_results_summary(df, ANALYSIS_DIR)
    write_global_manifest(ANALYSIS_DIR)

    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE")
    print(f"  Results: {ANALYSIS_DIR}")
    print(f"  key_findings.json — AI-oriented summary")
    print(f"  RESULTS_SUMMARY.md — human-oriented summary")
    print(f"  global_manifest.json — complete artifact index")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()