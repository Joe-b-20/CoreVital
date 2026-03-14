#!/usr/bin/env python3
"""
CoreVital Risk Calibration - using Experiment v2 labeled data.

This version uses grouped held-out evaluation by question_id so repeated runs
of the same prompt do not leak across train/test folds.

Three calibration steps:
  1. Evaluate current heuristic scores with grouped ECE + Platt scaling
  2. Build calibration profiles per model from correct runs
  3. Derive grouped, data-driven failure-risk weights via logistic regression

Usage:
    python3 calibrate_risk.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = EXPERIMENT_DIR.parent  # For relative paths in saved JSON (public-friendly)
RESULTS_DIR = EXPERIMENT_DIR / "results"
TRACES_DIR = EXPERIMENT_DIR / "traces"
CALIBRATION_DIR = EXPERIMENT_DIR / "calibration"

MIN_CELL_ROWS = 100
MIN_PROFILE_TRACES = 20
PROFILE_SAMPLE_SIZE = 200
DESIRED_SPLITS = 5
RANDOM_STATE = 42

RISK_COMPONENT_FEATURES = [
    "entropy_mean",
    "margin_mean",
    "topk_mass_mean",
    "surprisal_mean",
    "entropy_slope",
    "high_entropy_frac",
    "collapsed_rate_mean",
    "l2_norm_last_layer_mean",
    "n_compound_signals",
    "max_compound_severity",
    "nan_detected",
    "repetition_detected",
    "mid_layer_anomaly",
    "attention_collapse_detected",
]

PROMPT_FEATURES = [
    "prompt_surprisal_mean",
    "basin_score_min",
    "basin_score_mean",
    "layer_transform_mean",
    "layer_transform_std",
]


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=json_default)


def fmt(value: Optional[float], digits: int = 3, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    if not np.isfinite(value):
        return "n/a"
    spec = f"{'+' if signed else ''}.{digits}f"
    return format(value, spec)


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    mask = np.isfinite(scores)
    if mask.sum() < 2:
        return None

    y_eval = y_true[mask]
    score_eval = scores[mask]
    if len(np.unique(y_eval)) < 2:
        return None
    if np.std(score_eval) < 1e-12:
        return 0.5

    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_eval, score_eval))


def summarize_fold_metric(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "std": None, "ci95": None, "n_folds": 0}

    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    ci95 = float(1.96 * std / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "ci95": ci95,
        "n_folds": int(len(arr)),
    }


def build_group_labels(df: pd.DataFrame, columns: Tuple[str, ...]) -> pd.Series:
    return df.loc[:, list(columns)].astype(str).agg("::".join, axis=1)


def iter_group_splits(
    df: pd.DataFrame,
    y: np.ndarray,
    group_cols: Tuple[str, ...] = ("question_id",),
    desired_splits: int = DESIRED_SPLITS,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], int, str]:
    from sklearn.model_selection import GroupKFold

    groups = build_group_labels(df, group_cols).to_numpy()
    n_groups = int(pd.Series(groups).nunique())
    n_splits = min(desired_splits, n_groups)
    if n_splits < 3:
        return [], n_groups, "insufficient_groups"

    split_kind = "GroupKFold"
    try:
        from sklearn.model_selection import StratifiedGroupKFold

        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        raw_splits = list(splitter.split(df, y, groups))
        split_kind = "StratifiedGroupKFold"
    except Exception:
        splitter = GroupKFold(n_splits=n_splits)
        raw_splits = list(splitter.split(df, y, groups))

    valid_splits = []
    for train_idx, test_idx in raw_splits:
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        valid_splits.append((train_idx, test_idx))

    return valid_splits, n_groups, split_kind


def evaluate_probability_column(
    cell: pd.DataFrame,
    score_col: str,
    compute_ece,
    fit_platt_scaling,
    apply_platt_scaling,
) -> Optional[Dict[str, Any]]:
    data = cell.dropna(subset=["correct", score_col]).copy()
    if len(data) < MIN_CELL_ROWS:
        return None

    labels = (1 - data["correct"].astype(int)).to_numpy()
    if len(np.unique(labels)) < 2:
        return None

    raw_scores = data[score_col].astype(float).to_numpy()
    splits, n_groups, split_kind = iter_group_splits(data, labels, ("question_id",), DESIRED_SPLITS)
    if not splits:
        return None

    raw_oos = np.full(len(data), np.nan, dtype=float)
    cal_oos = np.full(len(data), np.nan, dtype=float)
    fold_rows: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        train_scores = raw_scores[train_idx]
        test_scores = raw_scores[test_idx]
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]

        raw_oos[test_idx] = test_scores
        fold_info: Dict[str, Any] = {
            "fold": fold_idx,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "ece_raw": float(compute_ece(test_scores.tolist(), test_labels.tolist())),
            "auroc_raw": safe_auc(test_labels, test_scores),
        }

        try:
            a, b = fit_platt_scaling(train_scores.tolist(), train_labels.tolist())
            calibrated = np.array(
                [apply_platt_scaling(float(score), a, b) for score in test_scores],
                dtype=float,
            )
            cal_oos[test_idx] = calibrated
            fold_info.update({
                "ece_calibrated": float(compute_ece(calibrated.tolist(), test_labels.tolist())),
                "auroc_calibrated": safe_auc(test_labels, calibrated),
                "platt_a": float(a),
                "platt_b": float(b),
            })
        except Exception as exc:
            fold_info.update({
                "ece_calibrated": None,
                "auroc_calibrated": None,
                "platt_error": str(exc),
            })

        fold_rows.append(fold_info)

    raw_mask = np.isfinite(raw_oos)
    cal_mask = np.isfinite(cal_oos)
    if raw_mask.sum() < 20:
        return None

    raw_ece_values = [row["ece_raw"] for row in fold_rows if row.get("ece_raw") is not None]
    raw_auc_values = [row["auroc_raw"] for row in fold_rows if row.get("auroc_raw") is not None]
    cal_ece_values = [row["ece_calibrated"] for row in fold_rows if row.get("ece_calibrated") is not None]
    cal_auc_values = [row["auroc_calibrated"] for row in fold_rows if row.get("auroc_calibrated") is not None]

    result: Dict[str, Any] = {
        "score_column": score_col,
        "evaluation_protocol": "Grouped held-out CV by question_id; Platt scaling fit on train folds and scored on held-out prompts.",
        "n_samples": int(raw_mask.sum()),
        "n_prompt_groups": n_groups,
        "splitter": split_kind,
        "n_folds": int(len(fold_rows)),
        "failure_rate": float(labels[raw_mask].mean()),
        "ece_raw_grouped_cv": float(compute_ece(raw_oos[raw_mask].tolist(), labels[raw_mask].tolist())),
        "auroc_raw_grouped_cv": safe_auc(labels[raw_mask], raw_oos[raw_mask]),
        "fold_ece_raw": summarize_fold_metric(raw_ece_values),
        "fold_auroc_raw": summarize_fold_metric(raw_auc_values),
        "folds": fold_rows,
    }

    if cal_mask.any():
        result.update({
            "ece_calibrated_grouped_cv": float(compute_ece(cal_oos[cal_mask].tolist(), labels[cal_mask].tolist())),
            "auroc_calibrated_grouped_cv": safe_auc(labels[cal_mask], cal_oos[cal_mask]),
            "fold_ece_calibrated": summarize_fold_metric(cal_ece_values),
            "fold_auroc_calibrated": summarize_fold_metric(cal_auc_values),
        })

    try:
        full_a, full_b = fit_platt_scaling(raw_scores.tolist(), labels.tolist())
        result["platt_a_full_fit"] = float(full_a)
        result["platt_b_full_fit"] = float(full_b)
    except Exception as exc:
        result["platt_fit_error"] = str(exc)

    return result


def fit_grouped_failure_model(
    cell: pd.DataFrame,
    feature_cols: List[str],
    compute_ece,
    group_cols: Tuple[str, ...] = ("question_id",),
) -> Optional[Dict[str, Any]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    data = cell.dropna(subset=["correct"]).copy()
    data = data.reset_index(drop=True)
    if len(data) < MIN_CELL_ROWS:
        return None

    features = [col for col in feature_cols if col in data.columns]
    if not features:
        return None

    y_failure = (1 - data["correct"].astype(int)).to_numpy()
    if len(np.unique(y_failure)) < 2:
        return None

    X = data[features].fillna(0.0)
    splits, n_groups, split_kind = iter_group_splits(data, y_failure, group_cols, DESIRED_SPLITS)
    if not splits:
        return None

    oos_probs = np.full(len(data), np.nan, dtype=float)
    fold_rows: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        y_train = y_failure[train_idx]
        y_test = y_failure[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X.iloc[train_idx])
        X_test = scaler.transform(X.iloc[test_idx])

        model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
        )
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        oos_probs[test_idx] = probs

        fold_rows.append({
            "fold": fold_idx,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "auroc": safe_auc(y_test, probs),
            "ece": float(compute_ece(probs.tolist(), y_test.tolist())),
        })

    mask = np.isfinite(oos_probs)
    if mask.sum() < 20:
        return None

    current_risk = data["risk_score"].fillna(0.5).astype(float).to_numpy() if "risk_score" in data.columns else np.full(len(data), np.nan, dtype=float)
    failure_risk = data["failure_risk"].fillna(0.5).astype(float).to_numpy() if "failure_risk" in data.columns else np.full(len(data), np.nan, dtype=float)

    scaler_full = StandardScaler()
    X_full = scaler_full.fit_transform(X)
    model_full = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
    )
    model_full.fit(X_full, y_failure)

    coefficients: Dict[str, Dict[str, float]] = {}
    for feat, coef, mean, std in zip(features, model_full.coef_[0], scaler_full.mean_, scaler_full.scale_):
        coefficients[feat] = {
            "raw_coef": float(coef),
            "importance": float(abs(coef)),
            "direction": "+" if coef > 0 else "-",
            "feature_mean": float(mean),
            "feature_std": float(std),
        }

    fold_auc_values = [row["auroc"] for row in fold_rows if row.get("auroc") is not None]
    fold_ece_values = [row["ece"] for row in fold_rows if row.get("ece") is not None]

    summary = {
        "evaluation_protocol": f"Grouped held-out CV by {'/'.join(group_cols)}; coefficients fit on all labeled rows after evaluation.",
        "target": "failure",
        "n_samples": int(mask.sum()),
        "n_prompt_groups": n_groups,
        "splitter": split_kind,
        "n_folds": int(len(fold_rows)),
        "features": features,
        "data_driven_auroc_grouped_cv": safe_auc(y_failure[mask], oos_probs[mask]),
        "data_driven_ece_grouped_cv": float(compute_ece(oos_probs[mask].tolist(), y_failure[mask].tolist())),
        "current_risk_auroc_grouped_cv": safe_auc(y_failure[mask], current_risk[mask]),
        "current_risk_ece_grouped_cv": float(compute_ece(current_risk[mask].tolist(), y_failure[mask].tolist())) if np.isfinite(current_risk[mask]).all() else None,
        "current_failure_risk_auroc_grouped_cv": safe_auc(y_failure[mask], failure_risk[mask]),
        "current_failure_risk_ece_grouped_cv": float(compute_ece(failure_risk[mask].tolist(), y_failure[mask].tolist())) if np.isfinite(failure_risk[mask]).all() else None,
        "improvement_vs_risk_score": None,
        "fold_auroc": summarize_fold_metric(fold_auc_values),
        "fold_ece": summarize_fold_metric(fold_ece_values),
    }

    if summary["current_risk_auroc_grouped_cv"] is not None and summary["data_driven_auroc_grouped_cv"] is not None:
        summary["improvement_vs_risk_score"] = (
            float(summary["data_driven_auroc_grouped_cv"] - summary["current_risk_auroc_grouped_cv"])
        )

    prediction_frame = data[["model", "dataset", "question_id", "run_idx", "correct"]].copy()
    prediction_frame["failure_label"] = y_failure
    prediction_frame["proposed_risk_oos"] = oos_probs
    prediction_frame["risk_score"] = current_risk
    prediction_frame["failure_risk"] = failure_risk

    return {
        "summary": summary,
        "coefficients": coefficients,
        "intercept": float(model_full.intercept_[0]),
        "folds": fold_rows,
        "prediction_frame": prediction_frame,
    }


def build_profiles(df: pd.DataFrame, calibrate_from_runs) -> Dict[str, Any]:
    profile_manifest: Dict[str, Any] = {}
    grades = None
    grades_path = RESULTS_DIR / "grades.jsonl"
    if grades_path.exists():
        try:
            grades = pd.read_json(grades_path, lines=True)
        except Exception:
            grades = None

    for model in sorted(df["model"].unique()):
        model_df = df[(df["model"] == model) & (df["correct"] == True)]
        trace_dir = TRACES_DIR / model
        if not trace_dir.exists():
            print(f"  {model}: trace directory not found, skipping")
            continue

        if len(model_df) == 0:
            print(f"  {model}: no correct runs available, skipping")
            continue

        sample = model_df.sample(n=min(PROFILE_SAMPLE_SIZE, len(model_df)), random_state=RANDOM_STATE)
        model_traces = []
        loaded = 0

        for _, row in sample.iterrows():
            qid = row["question_id"]
            run_idx = int(row["run_idx"])
            dataset = row["dataset"]
            trace_path = trace_dir / dataset / f"{qid}_run{run_idx:02d}.json"
            if not trace_path.exists():
                continue

            try:
                with open(trace_path, encoding="utf-8") as f:
                    trace = json.load(f)
                model_traces.append(trace)
                loaded += 1
            except Exception:
                continue

        if loaded < MIN_PROFILE_TRACES:
            print(f"  {model}: only {loaded} traces loaded, need >= {MIN_PROFILE_TRACES}, skipping")
            continue

        if grades is not None and not grades[grades["model"] == model].empty:
            hf_id = grades[grades["model"] == model]["model_id"].iloc[0]
        else:
            hf_id = model

        print(f"  {model}: building profile from {loaded} correct traces...")
        profile = calibrate_from_runs(hf_id, model_traces)
        profile_path = CALIBRATION_DIR / f"profile_{model}.json"
        profile.save(profile_path)

        profile_manifest[model] = {
            "profile_path": str(profile_path.relative_to(REPO_ROOT)),
            "n_traces": int(loaded),
            "entropy_mean": float(profile.entropy_per_step.mean),
            "entropy_std": float(profile.entropy_per_step.std),
            "margin_mean": float(profile.margin_per_step.mean),
            "margin_std": float(profile.margin_per_step.std),
            "n_layers_with_l2": int(len(profile.l2_norm_per_layer)),
        }

        print(f"    Saved to {profile_path}")
        print(f"    Entropy: mean={profile.entropy_per_step.mean:.4f}, std={profile.entropy_per_step.std:.4f}")
        print(f"    Margin:  mean={profile.margin_per_step.mean:.4f}, std={profile.margin_per_step.std:.4f}")
        print(f"    Layers:  {len(profile.l2_norm_per_layer)} layers with L2 data")

    save_json(CALIBRATION_DIR / "step2_profile_manifest.json", profile_manifest)
    return profile_manifest


def print_step_header(index: int, title: str) -> None:
    print("\n" + "=" * 60)
    print(f"STEP {index}: {title}")
    print("=" * 60)


def main() -> None:
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)

    features_path = RESULTS_DIR / "features.parquet"
    print(f"Loading {features_path}...")
    df = pd.read_parquet(features_path)
    df = df[df["format_failure"] != True].copy()
    print(
        f"  {len(df)} rows, {df['model'].nunique()} models, "
        f"{df['dataset'].nunique()} datasets"
    )

    sys.path.insert(0, str(EXPERIMENT_DIR.parent / "src"))
    from CoreVital.calibration import calibrate_from_runs
    from CoreVital.calibration_risk import (
        apply_platt_scaling,
        compute_ece,
        fit_platt_scaling,
    )

    print_step_header(1, "EVALUATE CURRENT HEURISTIC SCORES")
    step1_outputs = {
        "risk_score": CALIBRATION_DIR / "step1_ece_results.json",
        "failure_risk": CALIBRATION_DIR / "step1_failure_risk_results.json",
    }

    step1_results: Dict[str, Dict[str, Any]] = {}
    for score_col, output_path in step1_outputs.items():
        print(f"\n  Evaluating {score_col}...")
        score_results: Dict[str, Any] = {}

        for model in sorted(df["model"].unique()):
            for dataset in sorted(df["dataset"].unique()):
                cell = df[(df["model"] == model) & (df["dataset"] == dataset)]
                result = evaluate_probability_column(
                    cell,
                    score_col,
                    compute_ece,
                    fit_platt_scaling,
                    apply_platt_scaling,
                )
                if result is None:
                    continue

                key = f"{model}/{dataset}"
                score_results[key] = result

                raw_ece = result.get("ece_raw_grouped_cv")
                cal_ece = result.get("ece_calibrated_grouped_cv")
                quality = "+" if cal_ece is not None and cal_ece < 0.10 else "~" if cal_ece is not None and cal_ece < 0.20 else "!"
                print(
                    f"  {quality} {key}: raw ECE={fmt(raw_ece, 4)}, "
                    f"cal ECE={fmt(cal_ece, 4)}, raw AUROC={fmt(result.get('auroc_raw_grouped_cv'))}, "
                    f"cal AUROC={fmt(result.get('auroc_calibrated_grouped_cv'))}, "
                    f"folds={result.get('n_folds')}, groups={result.get('n_prompt_groups')}"
                )

        save_json(output_path, score_results)
        step1_results[score_col] = score_results

    print_step_header(2, "BUILD CALIBRATION PROFILES FROM CORRECT RUNS")
    build_profiles(df, calibrate_from_runs)

    print_step_header(3, "DERIVE DATA-DRIVEN RISK WEIGHTS")
    available_features = [
        feature for feature in (RISK_COMPONENT_FEATURES + PROMPT_FEATURES)
        if feature in df.columns
    ]
    print(f"\n  Using {len(available_features)} features for grouped weight derivation")

    weight_results: Dict[str, Any] = {}
    for model in sorted(df["model"].unique()):
        for dataset in sorted(df["dataset"].unique()):
            cell = df[(df["model"] == model) & (df["dataset"] == dataset)]
            result = fit_grouped_failure_model(
                cell,
                available_features,
                compute_ece,
                group_cols=("question_id",),
            )
            if result is None:
                continue

            key = f"{model}/{dataset}"
            summary = result["summary"]
            weight_results[key] = {
                **summary,
                "coefficients": result["coefficients"],
            }

            sorted_coefs = sorted(
                result["coefficients"].items(),
                key=lambda item: item[1]["importance"],
                reverse=True,
            )

            print(f"\n  {key}:")
            print(f"    Current risk_score AUROC:  {fmt(summary['current_risk_auroc_grouped_cv'])}")
            print(f"    Current failure_risk AUROC:{fmt(summary['current_failure_risk_auroc_grouped_cv'])}")
            print(f"    Data-driven LR AUROC:      {fmt(summary['data_driven_auroc_grouped_cv'])} (delta={fmt(summary['improvement_vs_risk_score'], signed=True)})")
            print(f"    Data-driven LR ECE:        {fmt(summary['data_driven_ece_grouped_cv'], 4)}")
            print("    Top 10 features (standardized |coef|, predicting failure):")
            for feat, info in sorted_coefs[:10]:
                print(f"      {info['direction']} {feat:<35s} |coef|={info['importance']:.4f}")

    save_json(CALIBRATION_DIR / "step3_data_driven_weights.json", weight_results)

    print_step_header(4, "CROSS-MODEL FEATURE IMPORTANCE CONSENSUS")
    feature_importance_agg: Dict[str, List[float]] = defaultdict(list)
    feature_direction_agg: Dict[str, List[int]] = defaultdict(list)

    for result in weight_results.values():
        for feat, info in result["coefficients"].items():
            feature_importance_agg[feat].append(float(info["importance"]))
            feature_direction_agg[feat].append(1 if info["direction"] == "+" else -1)

    print(f"\n  {'Feature':<35s} {'mean |coef|':>12s} {'consistency':>12s} {'direction':>18s}")
    print(f"  {'-' * 83}")

    consensus: Dict[str, Any] = {}
    for feat in sorted(feature_importance_agg.keys(), key=lambda name: np.mean(feature_importance_agg[name]), reverse=True):
        importances = feature_importance_agg[feat]
        directions = feature_direction_agg[feat]
        mean_importance = float(np.mean(importances))
        majority_direction = 1 if sum(directions) >= 0 else -1
        consistency = float(sum(1 for direction in directions if direction == majority_direction) / len(directions))
        direction_label = "higher failure risk" if majority_direction > 0 else "lower failure risk"

        consensus[feat] = {
            "mean_importance": mean_importance,
            "consistency": consistency,
            "majority_direction": direction_label,
            "n_cells": int(len(importances)),
        }

        if mean_importance > 0.05:
            print(f"  {feat:<35s} {mean_importance:>12.4f} {consistency:>11.0%} {direction_label:>18s}")

    save_json(CALIBRATION_DIR / "step4_feature_consensus.json", consensus)

    print_step_header(5, "PROPOSED NEW RISK FORMULA")
    proposed_features = [
        feat for feat, info in consensus.items()
        if info["mean_importance"] > 0.10 and info["consistency"] >= 0.70
    ]
    if not proposed_features:
        proposed_features = [
            feat for feat, _ in sorted(
                consensus.items(),
                key=lambda item: item[1]["mean_importance"],
                reverse=True,
            )[:8]
        ]
        print("\n  No features cleared the default threshold; falling back to the top consensus features.")

    available_proposed = [feat for feat in proposed_features if feat in df.columns]
    print(f"\n  Proposed risk features ({len(available_proposed)}):")
    for feat in available_proposed:
        info = consensus[feat]
        print(
            f"    {feat:<35s} importance={info['mean_importance']:.3f}, "
            f"consistency={info['consistency']:.0%}, {info['majority_direction']}"
        )

    global_result = fit_grouped_failure_model(
        df,
        available_proposed,
        compute_ece,
        group_cols=("model", "dataset", "question_id"),
    )
    if global_result is None:
        raise RuntimeError("Unable to fit grouped global failure model with the selected features")

    global_summary = global_result["summary"]
    prediction_frame = global_result["prediction_frame"]
    prediction_path = CALIBRATION_DIR / "step5_proposed_risk_predictions.parquet"
    prediction_frame.to_parquet(prediction_path, index=False)

    final_model = {
        "evaluation_protocol": "Grouped held-out CV by model/dataset/question_id; coefficients fit on all labeled rows after evaluation.",
        "target": "failure",
        "selected_features": available_proposed,
        "intercept": global_result["intercept"],
        "features": global_result["coefficients"],
        "auroc_grouped_cv": global_summary["data_driven_auroc_grouped_cv"],
        "ece_grouped_cv": global_summary["data_driven_ece_grouped_cv"],
        "current_risk_auroc_grouped_cv": global_summary["current_risk_auroc_grouped_cv"],
        "current_risk_ece_grouped_cv": global_summary["current_risk_ece_grouped_cv"],
        "current_failure_risk_auroc_grouped_cv": global_summary["current_failure_risk_auroc_grouped_cv"],
        "current_failure_risk_ece_grouped_cv": global_summary["current_failure_risk_ece_grouped_cv"],
        "n_samples": global_summary["n_samples"],
        "n_prompt_groups": global_summary["n_prompt_groups"],
        "splitter": global_summary["splitter"],
        "description": "Logistic regression: P(failure) = sigmoid(intercept + sum(coef_i * (feat_i - mean_i) / std_i))",
        "predictions_path": str(prediction_path.relative_to(REPO_ROOT)),
    }
    save_json(CALIBRATION_DIR / "step5_proposed_risk_model.json", final_model)

    print(f"\n  Proposed model grouped-CV AUROC: {fmt(final_model['auroc_grouped_cv'])}")
    print(f"  Proposed model grouped-CV ECE:   {fmt(final_model['ece_grouped_cv'], 4)}")
    print(f"  Current risk_score AUROC:        {fmt(final_model['current_risk_auroc_grouped_cv'])}")
    print(f"  Current risk_score ECE:          {fmt(final_model['current_risk_ece_grouped_cv'], 4)}")
    print(f"  Proposed model saved to {CALIBRATION_DIR / 'step5_proposed_risk_model.json'}")

    print_step_header(6, "PER-MODEL EVALUATION OF PROPOSED FORMULA")
    per_model_eval: Dict[str, Any] = {}
    for (model, dataset), cell in prediction_frame.groupby(["model", "dataset"]):
        if len(cell) < 50 or cell["failure_label"].nunique() < 2:
            continue

        y_failure = cell["failure_label"].to_numpy(dtype=int)
        proposed = cell["proposed_risk_oos"].to_numpy(dtype=float)
        current_risk = cell["risk_score"].to_numpy(dtype=float)
        failure_risk = cell["failure_risk"].to_numpy(dtype=float)

        proposed_auc = safe_auc(y_failure, proposed)
        current_auc = safe_auc(y_failure, current_risk)
        failure_auc = safe_auc(y_failure, failure_risk)

        proposed_mask = np.isfinite(proposed)
        current_mask = np.isfinite(current_risk)
        failure_mask = np.isfinite(failure_risk)

        proposed_ece = float(compute_ece(proposed[proposed_mask].tolist(), y_failure[proposed_mask].tolist())) if proposed_mask.any() else None
        current_ece = float(compute_ece(current_risk[current_mask].tolist(), y_failure[current_mask].tolist())) if current_mask.any() else None
        failure_ece = float(compute_ece(failure_risk[failure_mask].tolist(), y_failure[failure_mask].tolist())) if failure_mask.any() else None

        delta = None
        if proposed_auc is not None and current_auc is not None:
            delta = float(proposed_auc - current_auc)

        key = f"{model}/{dataset}"
        per_model_eval[key] = {
            "n_samples": int(len(cell)),
            "proposed_auroc_grouped_cv": proposed_auc,
            "proposed_ece_grouped_cv": proposed_ece,
            "current_risk_auroc_grouped_cv": current_auc,
            "current_risk_ece_grouped_cv": current_ece,
            "current_failure_risk_auroc_grouped_cv": failure_auc,
            "current_failure_risk_ece_grouped_cv": failure_ece,
            "delta_vs_risk_score": delta,
        }

        icon = "+" if delta is not None and delta > 0.05 else "~" if delta is not None and delta > 0 else "!"
        print(
            f"  {icon} {key}: current={fmt(current_auc)} -> proposed={fmt(proposed_auc)} "
            f"(delta={fmt(delta, signed=True)})"
        )

    save_json(CALIBRATION_DIR / "step6_per_model_evaluation.json", per_model_eval)

    print("\n" + "=" * 60)
    print("CALIBRATION COMPLETE")
    print("=" * 60)
    print(f"\n  Output files in {CALIBRATION_DIR}:")
    for path in sorted(CALIBRATION_DIR.glob("*")):
        size_kb = path.stat().st_size / 1024
        print(f"    {path.name} ({size_kb:.1f} KB)")

    print("\n  Next steps:")
    print("  1. Review grouped CV metrics in calibration/*.json")
    print("  2. Use step4_feature_consensus.json to refine heuristic formulas")
    print("  3. Treat step5_proposed_risk_model.json as the learned failure-risk baseline")
    print("  4. If the grouped results hold up, fold the formula back into risk.py in a separate PR")


if __name__ == "__main__":
    main()
