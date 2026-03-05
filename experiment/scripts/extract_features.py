#!/usr/bin/env python3
"""
CoreVital Validation Experiment — Feature Extraction

Reads JSON traces and grades.jsonl, extracts CoreVital features including
prompt analysis metrics, and saves as features.parquet.

Key changes from v1:
  - Includes run_idx, temperature, seed per trace
  - Extracts prompt analysis features (surprisal, basin scores, layer transforms)
  - Computes within-prompt pass rate for difficulty profiling
  - Per-layer signal extraction for layer analysis

Usage:
    python3 extract_features.py
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


def safe_mean(vals):
    finite = [v for v in vals if v is not None and np.isfinite(v)]
    return np.mean(finite) if finite else None

def safe_std(vals):
    finite = [v for v in vals if v is not None and np.isfinite(v)]
    return np.std(finite, ddof=1) if len(finite) >= 2 else None

def safe_slope(vals):
    finite = [(i, v) for i, v in enumerate(vals) if v is not None and np.isfinite(v)]
    if len(finite) < 3: return None
    x = np.array([i for i, _ in finite])
    y = np.array([v for _, v in finite])
    return np.polyfit(x, y, 1)[0]


def extract_logits_field(timeline, field):
    return [float(s.get("logits_summary", {}).get(field))
            for s in timeline
            if s.get("logits_summary", {}).get(field) is not None]


def extract_features(trace: dict) -> Dict[str, Any]:
    """Extract all features from a single trace."""
    f = {}
    timeline = trace.get("timeline", []) or []
    ext = trace.get("extensions", {}) or {}
    hf = trace.get("health_flags", {}) or {}
    summary = trace.get("summary", {}) or {}
    prompt_analysis = trace.get("prompt_analysis", {}) or {}

    total_steps = len(timeline)
    gen_tokens = summary.get("generated_tokens", total_steps)
    f["generated_tokens"] = gen_tokens
    f["total_steps"] = total_steps

    # === Logits ===
    ent = extract_logits_field(timeline, "entropy")
    surp = extract_logits_field(timeline, "surprisal")
    marg = extract_logits_field(timeline, "top_k_margin")
    mass = extract_logits_field(timeline, "topk_mass")
    perp = extract_logits_field(timeline, "perplexity")

    f["entropy_mean"] = safe_mean(ent)
    f["entropy_std"] = safe_std(ent)
    f["entropy_slope"] = safe_slope(ent)
    f["entropy_max"] = max(ent) if ent else None
    f["surprisal_mean"] = safe_mean(surp)
    f["surprisal_std"] = safe_std(surp)
    f["surprisal_volatility"] = (f["surprisal_std"] / f["surprisal_mean"]) if f["surprisal_mean"] and f["surprisal_std"] and f["surprisal_mean"] > 1e-9 else None
    f["margin_mean"] = safe_mean(marg)
    f["margin_slope"] = safe_slope(marg)
    f["topk_mass_mean"] = safe_mean(mass)
    f["perplexity_mean"] = safe_mean(perp)

    # === Hidden states ===
    l2_last, l2_all = [], []
    for step in timeline:
        for li, layer in enumerate(step.get("layers") or []):
            hs = layer.get("hidden_summary", {}) or {}
            norm = hs.get("l2_norm_mean")
            if norm is not None:
                l2_all.append(float(norm))
                if li == len(step.get("layers", [])) - 1:
                    l2_last.append(float(norm))

    f["l2_norm_last_layer_mean"] = safe_mean(l2_last)
    f["l2_norm_slope"] = safe_slope(l2_last)
    f["l2_norm_cross_layer_max"] = max(l2_all) if l2_all else None

    # === Attention ===
    collapsed, focused = [], []
    for step in timeline:
        for layer in (step.get("layers") or []):
            attn = layer.get("attention_summary", {}) or {}
            if attn.get("collapsed_head_rate") is not None: collapsed.append(float(attn["collapsed_head_rate"]))
            if attn.get("focused_head_count") is not None: focused.append(float(attn["focused_head_count"]))

    f["collapsed_rate_mean"] = safe_mean(collapsed)
    f["focused_head_mean"] = safe_mean(focused)

    # === Per-layer aggregates (for layer analysis) ===
    # Compute mean entropy and attention stats per layer across all steps
    if timeline and timeline[0].get("layers"):
        n_layers = len(timeline[0].get("layers", []))
        for li in range(min(n_layers, 40)):  # Cap at 40 layers
            layer_entropies = []
            layer_l2 = []
            for step in timeline:
                layers = step.get("layers") or []
                if li < len(layers):
                    attn = layers[li].get("attention_summary", {}) or {}
                    ae = attn.get("entropy_mean")
                    if ae is not None: layer_entropies.append(float(ae))
                    hs = layers[li].get("hidden_summary", {}) or {}
                    norm = hs.get("l2_norm_mean")
                    if norm is not None: layer_l2.append(float(norm))
            f[f"layer_{li:02d}_attn_entropy"] = safe_mean(layer_entropies)
            f[f"layer_{li:02d}_l2_norm"] = safe_mean(layer_l2)

    # === Health flags ===
    f["nan_detected"] = int(hf.get("nan_detected", False))
    f["inf_detected"] = int(hf.get("inf_detected", False))
    f["repetition_detected"] = int(hf.get("repetition_loop_detected", False))
    f["mid_layer_anomaly"] = int(hf.get("mid_layer_anomaly_detected", False))
    f["attention_collapse_detected"] = int(hf.get("attention_collapse_detected", False))
    high_ent = hf.get("high_entropy_steps", 0) or 0
    f["high_entropy_steps"] = high_ent
    f["high_entropy_frac"] = high_ent / max(1, gen_tokens)

    # === Risk / early warning ===
    risk = ext.get("risk", {}) or {}
    f["risk_score"] = risk.get("risk_score", 0.0)

    ew = ext.get("early_warning", {}) or {}
    f["failure_risk"] = ew.get("failure_risk", 0.0)
    f["n_warning_signals"] = len(ew.get("warning_signals", []))

    # === Compound signals ===
    cs = ext.get("compound_signals", []) or []
    f["n_compound_signals"] = len(cs)
    f["max_compound_severity"] = max((s.get("severity", 0) for s in cs), default=0.0)
    cs_names = {s.get("name", "") for s in cs}
    for name in ["context_loss", "confident_confusion", "degenerating_generation",
                 "attention_bottleneck", "confident_repetition_risk"]:
        f[f"cs_{name}"] = int(name in cs_names)

    # === Prompt analysis ===
    prompt_surprisals = prompt_analysis.get("prompt_surprisals", []) or []
    f["prompt_surprisal_mean"] = safe_mean(prompt_surprisals)
    f["prompt_surprisal_max"] = max(prompt_surprisals) if prompt_surprisals else None
    f["prompt_surprisal_std"] = safe_std(prompt_surprisals)

    basin_all = []
    for ld in (prompt_analysis.get("layers") or []):
        for bs in (ld.get("basin_scores") or []):
            if bs is not None: basin_all.append(float(bs))
    f["basin_score_min"] = min(basin_all) if basin_all else None
    f["basin_score_mean"] = safe_mean(basin_all)
    f["basin_score_std"] = safe_std(basin_all)

    layer_transforms = prompt_analysis.get("layer_transformations", []) or []
    f["layer_transform_mean"] = safe_mean(layer_transforms)
    f["layer_transform_std"] = safe_std(layer_transforms)
    f["layer_transform_max"] = max(layer_transforms) if layer_transforms else None

    sparse_heads = prompt_analysis.get("sparse_attention_heads", []) or []
    f["n_sparse_heads"] = len(sparse_heads)

    # === Fingerprint ===
    fp = ext.get("fingerprint", {}).get("vector", []) or []
    for i in range(25):
        f[f"fp_{i:02d}"] = fp[i] if i < len(fp) else None

    # === Performance timing ===
    perf = ext.get("performance", {}) or {}
    f["total_run_ms"] = perf.get("total_run_ms")
    parent_ops = perf.get("parent_operations", {}) or {}
    f["inference_ms"] = parent_ops.get("model_inference", {}).get("total_ms") if parent_ops else None
    f["report_build_ms"] = parent_ops.get("report_build", {}).get("total_ms") if parent_ops else None
    f["tokenize_ms"] = parent_ops.get("tokenize", {}).get("total_ms") if parent_ops else None
    if f["total_run_ms"] and f["inference_ms"]:
        f["overhead_ms"] = f["total_run_ms"] - f["inference_ms"]
        f["overhead_pct"] = (f["overhead_ms"] / f["total_run_ms"]) * 100
    else:
        f["overhead_ms"] = None
        f["overhead_pct"] = None

    # === Early warning features (first 50% of timeline) ===
    if total_steps >= 6:
        half = total_steps // 2
        f["early_entropy_mean"] = safe_mean(extract_logits_field(timeline[:half], "entropy"))
        f["early_surprisal_mean"] = safe_mean(extract_logits_field(timeline[:half], "surprisal"))
        f["early_margin_mean"] = safe_mean(extract_logits_field(timeline[:half], "top_k_margin"))
        f["early_entropy_slope"] = safe_slope(extract_logits_field(timeline[:half], "entropy"))
    else:
        f["early_entropy_mean"] = None
        f["early_surprisal_mean"] = None
        f["early_margin_mean"] = None
        f["early_entropy_slope"] = None

    return f


def load_grades() -> Dict[str, list]:
    """Load grades keyed by (prompt_key, run_idx)."""
    grades = {}
    if not GRADES_FILE.exists():
        return grades
    with open(GRADES_FILE) as f:
        for line in f:
            r = json.loads(line)
            key = (r["prompt_key"], r["run_idx"])
            grades[key] = r
    return grades


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces-dir", type=Path, default=TRACES_DIR)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "features.parquet")
    args = parser.parse_args()

    print(f"Loading grades...")
    grades = load_grades()
    print(f"  {len(grades)} grade records.")

    print(f"Finding traces in {args.traces_dir}...")
    trace_files = sorted(args.traces_dir.rglob("*.json"))
    print(f"  {len(trace_files)} trace files.")

    rows = []
    errors = 0

    for tp in tqdm(trace_files, desc="Extracting"):
        try:
            with open(tp) as f:
                trace = json.load(f)
        except Exception:
            errors += 1; continue

        try:
            rel = tp.relative_to(args.traces_dir)
            parts = rel.parts  # model/dataset/qid_runNN.json
            model_short = parts[0]
            dataset = parts[1]
            fname = parts[2].replace(".json", "")
            # Parse: gsm8k_0001_run03
            run_match = fname.rsplit("_run", 1)
            qid = run_match[0]
            run_idx = int(run_match[1]) if len(run_match) > 1 else 0
            prompt_key = f"{model_short}/{dataset}/{qid}"
        except Exception:
            errors += 1; continue

        try:
            features = extract_features(trace)
        except Exception as e:
            errors += 1; continue

        features["prompt_key"] = prompt_key
        features["model"] = model_short
        features["dataset"] = dataset
        features["question_id"] = qid
        features["run_idx"] = run_idx

        grade = grades.get((prompt_key, run_idx), {})
        features["correct"] = grade.get("correct")
        features["format_failure"] = grade.get("format_failure")
        features["temperature"] = grade.get("temperature")
        features["seed"] = grade.get("seed")

        rows.append(features)

    df = pd.DataFrame(rows)
    print(f"\nExtracted {len(df)} rows, {errors} errors.")

    if len(df) == 0:
        print("ERROR: No features extracted.")
        sys.exit(1)

    # Compute within-prompt pass rate (difficulty proxy)
    if "correct" in df.columns:
        pass_rate = df.groupby(["model", "dataset", "question_id"])["correct"].mean().reset_index()
        pass_rate.columns = ["model", "dataset", "question_id", "pass_rate"]
        df = df.merge(pass_rate, on=["model", "dataset", "question_id"], how="left")

        # Cross-model difficulty
        diff = df.groupby(["dataset", "question_id"])["correct"].apply(lambda x: 1.0 - x.mean()).reset_index()
        diff.columns = ["dataset", "question_id", "question_difficulty"]
        df = df.merge(diff, on=["dataset", "question_id"], how="left")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Saved to {args.output}")

    print(f"\nSummary:")
    for (m, d), g in df.groupby(["model", "dataset"]):
        n_prompts = g["question_id"].nunique()
        n_runs = len(g)
        acc = g["correct"].mean() if "correct" in g.columns else "?"
        print(f"  {m}/{d}: {n_prompts} prompts, {n_runs} runs, {acc:.1%} accuracy")


if __name__ == "__main__":
    main()