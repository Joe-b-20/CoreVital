#!/usr/bin/env python3
"""
CoreVital Validation Experiment - Feature Extraction

Reads JSON traces and grades.jsonl, extracts all CoreVital features into
a single DataFrame, and saves as features.parquet for analysis.

Usage:
    python3 extract_features.py
    python3 extract_features.py --traces-dir ~/experiment/perf_traces  # for perf subset
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

EXPERIMENT_DIR = Path.home() / "experiment"
TRACES_DIR = EXPERIMENT_DIR / "traces"
RESULTS_DIR = EXPERIMENT_DIR / "results"
GRADES_FILE = RESULTS_DIR / "grades.jsonl"


# ---------------------------------------------------------------------------
# Feature extraction from a single trace
# ---------------------------------------------------------------------------

def safe_mean(vals: List[float]) -> Optional[float]:
    """Mean of finite values, or None if empty."""
    finite = [v for v in vals if v is not None and math.isfinite(v)]
    return sum(finite) / len(finite) if finite else None


def safe_std(vals: List[float]) -> Optional[float]:
    finite = [v for v in vals if v is not None and math.isfinite(v)]
    if len(finite) < 2:
        return None
    mean = sum(finite) / len(finite)
    return (sum((v - mean) ** 2 for v in finite) / (len(finite) - 1)) ** 0.5


def safe_slope(vals: List[float]) -> Optional[float]:
    """Least-squares slope of values over their index."""
    finite = [(i, v) for i, v in enumerate(vals) if v is not None and math.isfinite(v)]
    if len(finite) < 3:
        return None
    n = len(finite)
    x_mean = sum(i for i, _ in finite) / n
    y_mean = sum(v for _, v in finite) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in finite)
    den = sum((i - x_mean) ** 2 for i, _ in finite)
    return num / den if den > 1e-12 else 0.0


def extract_logits_values(timeline: list, field: str) -> List[float]:
    """Extract a numeric field from each timeline step's logits_summary."""
    vals = []
    for step in timeline:
        ls = step.get("logits_summary", {})
        if ls is None:
            ls = {}
        v = ls.get(field)
        if v is not None:
            vals.append(float(v))
    return vals


def extract_features_from_trace(trace: dict) -> Dict[str, Any]:
    """
    Extract all analysis features from a single CoreVital JSON trace.

    Returns a flat dict of feature_name → value.
    """
    features = {}
    timeline = trace.get("timeline", []) or []
    extensions = trace.get("extensions", {}) or {}
    health_flags = trace.get("health_flags", {}) or {}
    summary = trace.get("summary", {}) or {}
    prompt_analysis = trace.get("prompt_analysis", {}) or {}

    total_steps = len(timeline)
    gen_tokens = summary.get("generated_tokens", total_steps)
    features["generated_tokens"] = gen_tokens
    features["total_steps"] = total_steps

    # === Logits features ===
    entropies = extract_logits_values(timeline, "entropy")
    surprisals = extract_logits_values(timeline, "surprisal")
    margins = extract_logits_values(timeline, "top_k_margin")
    masses = extract_logits_values(timeline, "topk_mass")
    perplexities = extract_logits_values(timeline, "perplexity")

    features["entropy_mean"] = safe_mean(entropies)
    features["entropy_std"] = safe_std(entropies)
    features["entropy_slope"] = safe_slope(entropies)
    features["entropy_max"] = max(entropies) if entropies else None

    features["surprisal_mean"] = safe_mean(surprisals)
    features["surprisal_std"] = safe_std(surprisals)
    s_mean = features["surprisal_mean"]
    s_std = features["surprisal_std"]
    features["surprisal_volatility"] = (s_std / s_mean) if (s_mean and s_std and s_mean > 1e-9) else None

    features["margin_mean"] = safe_mean(margins)
    features["margin_slope"] = safe_slope(margins)

    features["topk_mass_mean"] = safe_mean(masses)
    features["perplexity_mean"] = safe_mean(perplexities)

    # === Hidden state features ===
    # Collect l2_norm_mean from last layer at each step
    l2_norms_last = []
    l2_norms_all = []  # For cross-layer max
    for step in timeline:
        layers = step.get("layers", []) or []
        if layers:
            last_layer = layers[-1]
            hs = last_layer.get("hidden_summary", {}) or {}
            norm = hs.get("l2_norm_mean")
            if norm is not None:
                l2_norms_last.append(float(norm))
        # All layers for cross-layer stats
        for layer in layers:
            hs = layer.get("hidden_summary", {}) or {}
            norm = hs.get("l2_norm_mean")
            if norm is not None:
                l2_norms_all.append(float(norm))

    features["l2_norm_last_layer_mean"] = safe_mean(l2_norms_last)
    features["l2_norm_slope"] = safe_slope(l2_norms_last)
    features["l2_norm_cross_layer_max"] = max(l2_norms_all) if l2_norms_all else None

    # === Attention features ===
    collapsed_rates = []
    focused_counts = []
    for step in timeline:
        for layer in (step.get("layers") or []):
            attn = layer.get("attention_summary", {}) or {}
            cr = attn.get("collapsed_head_rate")
            if cr is not None:
                collapsed_rates.append(float(cr))
            fc = attn.get("focused_head_count")
            if fc is not None:
                focused_counts.append(float(fc))

    features["collapsed_rate_mean"] = safe_mean(collapsed_rates)
    features["focused_head_mean"] = safe_mean(focused_counts)

    # === Health flags ===
    features["nan_detected"] = int(health_flags.get("nan_detected", False))
    features["inf_detected"] = int(health_flags.get("inf_detected", False))
    features["repetition_detected"] = int(health_flags.get("repetition_loop_detected", False))
    features["mid_layer_anomaly"] = int(health_flags.get("mid_layer_anomaly_detected", False))
    features["attention_collapse_detected"] = int(health_flags.get("attention_collapse_detected", False))
    features["attention_collapse_severity"] = health_flags.get("attention_collapse_severity")

    high_ent_steps = health_flags.get("high_entropy_steps", 0) or 0
    features["high_entropy_steps"] = high_ent_steps
    features["high_entropy_frac"] = high_ent_steps / max(1, gen_tokens)

    # === Risk score ===
    risk_ext = extensions.get("risk", {}) or {}
    features["risk_score"] = risk_ext.get("risk_score", 0.0)
    features["risk_factors"] = json.dumps(risk_ext.get("risk_factors", []))

    # === Early warning ===
    ew_ext = extensions.get("early_warning", {}) or {}
    features["failure_risk"] = ew_ext.get("failure_risk", 0.0)
    features["warning_signals"] = json.dumps(ew_ext.get("warning_signals", []))
    features["n_warning_signals"] = len(ew_ext.get("warning_signals", []))

    # === Compound signals ===
    cs_ext = extensions.get("compound_signals", []) or []
    features["n_compound_signals"] = len(cs_ext)
    features["max_compound_severity"] = max((s.get("severity", 0) for s in cs_ext), default=0.0)
    # One-hot for each known compound signal
    cs_names = {s.get("name", "") for s in cs_ext}
    for name in ["context_loss", "confident_confusion", "degenerating_generation",
                 "attention_bottleneck", "confident_repetition_risk"]:
        features[f"cs_{name}"] = int(name in cs_names)

    # === Prompt telemetry ===
    prompt_surprisals = prompt_analysis.get("prompt_surprisals", []) or []
    features["prompt_surprisal_mean"] = safe_mean(prompt_surprisals)

    basin_scores_all = []
    for layer_data in (prompt_analysis.get("layers") or []):
        for bs in (layer_data.get("basin_scores") or []):
            if bs is not None:
                basin_scores_all.append(float(bs))
    features["basin_score_min"] = min(basin_scores_all) if basin_scores_all else None
    features["basin_score_mean"] = safe_mean(basin_scores_all)

    layer_transforms = prompt_analysis.get("layer_transformations", []) or []
    features["layer_transform_mean"] = safe_mean(layer_transforms)

    # === Fingerprint ===
    fp_ext = extensions.get("fingerprint", {}) or {}
    fp_vec = fp_ext.get("vector", []) or []
    for i, v in enumerate(fp_vec):
        features[f"fp_{i:02d}"] = v
    # Pad if vector is shorter than 25
    for i in range(len(fp_vec), 25):
        features[f"fp_{i:02d}"] = None

    # === Early warning features (first 50% of timeline) ===
    if total_steps >= 4:
        half = total_steps // 2
        half_timeline = timeline[:half]
        half_ent = extract_logits_values(half_timeline, "entropy")
        half_surp = extract_logits_values(half_timeline, "surprisal")
        half_marg = extract_logits_values(half_timeline, "top_k_margin")

        features["early_entropy_mean"] = safe_mean(half_ent)
        features["early_surprisal_mean"] = safe_mean(half_surp)
        features["early_margin_mean"] = safe_mean(half_marg)
        features["early_entropy_slope"] = safe_slope(half_ent)
    else:
        features["early_entropy_mean"] = None
        features["early_surprisal_mean"] = None
        features["early_margin_mean"] = None
        features["early_entropy_slope"] = None

    return features


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------

def load_grades() -> Dict[str, dict]:
    """Load grades keyed by (model/dataset/question_id)."""
    grades = {}
    if not GRADES_FILE.exists():
        print(f"WARNING: {GRADES_FILE} not found.")
        return grades
    with open(GRADES_FILE) as f:
        for line in f:
            record = json.loads(line)
            grades[record["key"]] = record
    return grades


def find_all_traces(traces_dir: Path) -> List[Path]:
    """Find all .json trace files recursively."""
    return sorted(traces_dir.rglob("*.json"))


def main():
    parser = argparse.ArgumentParser(description="Extract features from CoreVital traces")
    parser.add_argument("--traces-dir", type=Path, default=TRACES_DIR)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "features.parquet")
    args = parser.parse_args()

    print(f"Loading grades from {GRADES_FILE}...")
    grades = load_grades()
    print(f"  Found {len(grades)} grade records.")

    print(f"Finding traces in {args.traces_dir}...")
    trace_files = find_all_traces(args.traces_dir)
    print(f"  Found {len(trace_files)} trace files.")

    rows = []
    errors = 0

    for trace_path in tqdm(trace_files, desc="Extracting features"):
        try:
            with open(trace_path) as f:
                trace = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  Skipping {trace_path}: {e}")
            errors += 1
            continue

        # Reconstruct the key from the path structure
        # Expected: traces_dir/model_short/dataset/question_id.json
        try:
            rel = trace_path.relative_to(args.traces_dir)
            parts = rel.parts  # (model_short, dataset, filename)
            model_short = parts[0]
            dataset = parts[1]
            qid = parts[2].replace(".json", "")
            key = f"{model_short}/{dataset}/{qid}"
        except (ValueError, IndexError):
            print(f"  Skipping {trace_path}: unexpected path structure")
            errors += 1
            continue

        # Extract features
        try:
            features = extract_features_from_trace(trace)
        except Exception as e:
            print(f"  Feature extraction failed for {key}: {e}")
            errors += 1
            continue

        # Add metadata
        features["key"] = key
        features["model"] = model_short
        features["dataset"] = dataset
        features["question_id"] = qid

        # Add grade info
        grade = grades.get(key, {})
        features["correct"] = grade.get("correct", None)
        features["format_failure"] = grade.get("format_failure", None)
        features["gold_answer"] = grade.get("gold_answer", None)
        features["extracted_answer"] = grade.get("extracted_answer", None)

        rows.append(features)

    # Build DataFrame
    df = pd.DataFrame(rows)
    print(f"\nExtracted {len(df)} rows, {errors} errors.")

    if len(df) == 0:
        print("ERROR: No features extracted. Check traces directory and grades file.")
        sys.exit(1)

    # Compute question difficulty (fraction of models that got it wrong)
    if "correct" in df.columns:
        difficulty = df.groupby(["dataset", "question_id"])["correct"].apply(
            lambda x: 1.0 - x.mean()
        ).reset_index()
        difficulty.columns = ["dataset", "question_id", "question_difficulty"]
        df = df.merge(difficulty, on=["dataset", "question_id"], how="left")

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Saved to {args.output}")

    # Summary
    print(f"\nDataset summary:")
    for (model, dataset), group in df.groupby(["model", "dataset"]):
        n = len(group)
        correct = group["correct"].sum() if "correct" in group.columns else "?"
        acc = group["correct"].mean() if "correct" in group.columns else "?"
        if isinstance(acc, float):
            print(f"  {model}/{dataset}: {n} traces, {correct:.0f} correct ({acc:.1%})")
        else:
            print(f"  {model}/{dataset}: {n} traces")

    print(f"\nFeature columns ({len(df.columns)} total):")
    for col in sorted(df.columns):
        non_null = df[col].notna().sum()
        print(f"  {col}: {non_null}/{len(df)} non-null")


if __name__ == "__main__":
    main()