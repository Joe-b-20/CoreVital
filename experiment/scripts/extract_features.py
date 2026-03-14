#!/usr/bin/env python3
"""
CoreVital Validation Experiment — Feature Extraction (v2)
==========================================================
Reads JSON traces and grades.jsonl, extracts CoreVital features,
and produces structured artifacts for downstream analysis.

Outputs:
  results/features.parquet         — run-level feature table (backward-compatible)
  results/prompt_level.parquet     — one row per (model, dataset, question_id)
  results/layer_long.parquet       — long-form per-layer per-run table
  results/extraction_manifest.json — schema, row counts, null rates, errors
  results/extraction_summary.md    — human-readable extraction report

Key improvements over v1:
  - Provenance flags (trace_has_*, grade_found)
  - Architecture metadata from trace model block
  - Saturation / discretization flags for risk_score, failure_risk
  - Multi-window early features (first 10, 25%, 50%) + late 50%
  - Algebraic slope (no polyfit), surprisal_diff_std, surprisal_range
  - Normalized attention features (entropy_norm, focused_head_rate)
  - Hidden state features (std, max_abs)
  - Layer shape summaries (peak, trough, early-late delta)
  - Performance decomposition (report_build, tokenize, unattributed)
  - Structured error log with trace paths and exception types
  - Prompt-level table with invariance validation
  - Long-form layer table for layer analysis

Usage:
    python3 extract_features.py [--traces-dir PATH] [--output PATH]
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Import shared helpers — expected alongside this script
try:
    from helpers import (
        setup_logging, safe_mean, safe_std, safe_min, safe_max,
        safe_percentile, safe_slope, safe_diff_std, save_json, ensure_dir,
        build_manifest,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from helpers import (
        setup_logging, safe_mean, safe_std, safe_min, safe_max,
        safe_percentile, safe_slope, safe_diff_std, save_json, ensure_dir,
        build_manifest,
    )

log = setup_logging("extract")

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = EXPERIMENT_DIR.parent  # For relative paths in saved manifest (public-friendly)
TRACES_DIR = EXPERIMENT_DIR / "traces"
RESULTS_DIR = EXPERIMENT_DIR / "results"
GRADES_FILE = RESULTS_DIR / "grades.jsonl"


def _path_rel_to_repo(p: Path) -> str:
    """Return path relative to repo root for manifest; fallback to absolute."""
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(Path(p).resolve())

PROMPT_INVARIANT_COLUMNS = [
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

PERFORMANCE_COLUMNS = [
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


# ── Trace Field Extraction ──────────────────────────────────

def extract_logits_field(timeline: list, field: str) -> List[float]:
    """Pull a named field from logits_summary across all timeline steps."""
    return [
        float(s["logits_summary"][field])
        for s in timeline
        if s.get("logits_summary", {}).get(field) is not None
    ]


def extract_time_series_stats(vals: List[float], prefix: str) -> Dict[str, Any]:
    """
    Compute a standard set of time-series summary statistics.
    
    Applied to entropy, surprisal, margin, etc. Returns dict with
    keys like {prefix}_mean, {prefix}_std, {prefix}_slope, etc.
    """
    f = {}
    f[f"{prefix}_mean"] = safe_mean(vals)
    f[f"{prefix}_std"] = safe_std(vals)
    f[f"{prefix}_slope"] = safe_slope(vals)
    f[f"{prefix}_min"] = safe_min(vals)
    f[f"{prefix}_max"] = safe_max(vals)
    f[f"{prefix}_p90"] = safe_percentile(vals, 90)
    f[f"{prefix}_range"] = None
    if f[f"{prefix}_max"] is not None and f[f"{prefix}_min"] is not None:
        f[f"{prefix}_range"] = f[f"{prefix}_max"] - f[f"{prefix}_min"]
    f[f"{prefix}_diff_std"] = safe_diff_std(vals)

    # Early-late delta (last 25% mean minus first 25% mean)
    if vals and len(vals) >= 8:
        q = max(1, len(vals) // 4)
        early_m = safe_mean(vals[:q])
        late_m = safe_mean(vals[-q:])
        f[f"{prefix}_early_late_delta"] = (
            late_m - early_m if early_m is not None and late_m is not None else None
        )
    else:
        f[f"{prefix}_early_late_delta"] = None

    return f


def extract_window_stats(
    vals: List[float], total_steps: int, prefix: str
) -> Dict[str, Any]:
    """
    Multi-window early/late features for a signal.
    
    Windows: first 10 steps, first 25%, first 50%, last 50%.
    """
    f = {}
    windows = {
        "early10": min(10, len(vals)),
        "early25p": max(1, len(vals) // 4),
        "early50p": max(1, len(vals) // 2),
    }
    for w_name, w_size in windows.items():
        if len(vals) >= 6:
            f[f"{w_name}_{prefix}_mean"] = safe_mean(vals[:w_size])
            f[f"{w_name}_{prefix}_slope"] = safe_slope(vals[:w_size])
        else:
            f[f"{w_name}_{prefix}_mean"] = None
            f[f"{w_name}_{prefix}_slope"] = None

    # Late window (last 50%)
    if len(vals) >= 6:
        half = max(1, len(vals) // 2)
        f[f"late50p_{prefix}_mean"] = safe_mean(vals[half:])
    else:
        f[f"late50p_{prefix}_mean"] = None

    return f


# ── Main Feature Extractor ──────────────────────────────────

def extract_features(trace: dict) -> Tuple[Dict[str, Any], List[Dict]]:
    """
    Extract all features from a single trace.
    
    Returns:
        (features_dict, layer_rows_list)
        
    features_dict: flat dict of run-level features
    layer_rows_list: list of dicts for the long-form layer table
    """
    f = {}
    warnings_list = []
    timeline = trace.get("timeline", []) or []
    ext = trace.get("extensions", {}) or {}
    hf = trace.get("health_flags", {}) or {}
    summary = trace.get("summary", {}) or {}
    prompt_analysis = trace.get("prompt_analysis", {}) or {}
    model_info = trace.get("model", {}) or {}

    total_steps = len(timeline)
    gen_tokens = summary.get("generated_tokens", total_steps)
    f["generated_tokens"] = gen_tokens
    f["total_steps"] = total_steps

    # ── Provenance Flags ──
    f["trace_has_prompt_analysis"] = int(bool(prompt_analysis))
    f["trace_has_fingerprint"] = int(bool(ext.get("fingerprint", {}).get("vector")))
    f["trace_has_performance"] = int(bool(ext.get("performance")))
    f["trace_has_extensions"] = int(bool(ext))
    f["trace_has_health_flags"] = int(bool(hf))
    f["trace_has_early_warning"] = int(bool(ext.get("early_warning")))
    f["trace_has_risk"] = int(bool(ext.get("risk")))

    # ── Architecture Metadata ──
    f["arch_num_layers"] = model_info.get("num_layers") or model_info.get("config", {}).get("num_hidden_layers")
    f["arch_num_heads"] = model_info.get("num_attention_heads") or model_info.get("config", {}).get("num_attention_heads")
    f["arch_hidden_size"] = model_info.get("hidden_size") or model_info.get("config", {}).get("hidden_size")
    f["arch_is_moe"] = int(bool(
        model_info.get("is_moe")
        or model_info.get("config", {}).get("num_local_experts")
        or model_info.get("num_experts")
    ))

    # ── Logits Time Series ──
    ent = extract_logits_field(timeline, "entropy")
    surp = extract_logits_field(timeline, "surprisal")
    marg = extract_logits_field(timeline, "top_k_margin")
    mass = extract_logits_field(timeline, "topk_mass")
    perp = extract_logits_field(timeline, "perplexity")

    # Standard stats for each signal
    f.update(extract_time_series_stats(ent, "entropy"))
    f.update(extract_time_series_stats(surp, "surprisal"))
    f.update(extract_time_series_stats(marg, "margin"))

    # Lighter stats for mass and perplexity (less analytically central)
    f["topk_mass_mean"] = safe_mean(mass)
    f["topk_mass_std"] = safe_std(mass)
    f["perplexity_mean"] = safe_mean(perp)
    f["perplexity_max"] = safe_max(perp)

    # Surprisal volatility — kept for backward compat, but flagged as unstable
    # when surprisal_mean is near zero
    surp_mean = f.get("surprisal_mean")
    surp_std = f.get("surprisal_std")
    if surp_mean is not None and surp_std is not None and surp_mean > 0.1:
        f["surprisal_volatility"] = surp_std / surp_mean
    else:
        f["surprisal_volatility"] = None

    # Multi-window early/late features
    f.update(extract_window_stats(ent, total_steps, "entropy"))
    f.update(extract_window_stats(surp, total_steps, "surprisal"))
    f.update(extract_window_stats(marg, total_steps, "margin"))

    # ── Hidden States ──
    l2_last, l2_all = [], []
    hidden_std_last, hidden_maxabs_last = [], []
    for step in timeline:
        layers = step.get("layers") or []
        for li, layer in enumerate(layers):
            hs = layer.get("hidden_summary") or {}
            norm = hs.get("l2_norm_mean")
            if norm is not None:
                l2_all.append(float(norm))
                if li == len(layers) - 1:
                    l2_last.append(float(norm))
            # Additional hidden features (last layer only)
            if li == len(layers) - 1:
                hstd = hs.get("std")
                hmax = hs.get("max_abs")
                if hstd is not None:
                    hidden_std_last.append(float(hstd))
                if hmax is not None:
                    hidden_maxabs_last.append(float(hmax))

    f["l2_norm_last_layer_mean"] = safe_mean(l2_last)
    f["l2_norm_slope"] = safe_slope(l2_last)
    f["l2_norm_cross_layer_max"] = safe_max(l2_all)
    f["hidden_std_last_layer_mean"] = safe_mean(hidden_std_last)
    f["hidden_max_abs_last_layer_mean"] = safe_mean(hidden_maxabs_last)

    # ── Attention ──
    collapsed_rates, focused_counts = [], []
    attn_ent_norm_all = []
    conc_max_all, conc_min_all = [], []
    num_heads_seen = None
    for step in timeline:
        for layer in (step.get("layers") or []):
            attn = layer.get("attention_summary") or {}
            cr = attn.get("collapsed_head_rate")
            if cr is not None:
                collapsed_rates.append(float(cr))
            fc = attn.get("focused_head_count")
            if fc is not None:
                focused_counts.append(float(fc))
            # Normalized attention entropy
            ae_norm = attn.get("entropy_mean_normalized")
            if ae_norm is not None:
                attn_ent_norm_all.append(float(ae_norm))
            # Concentration
            cm = attn.get("concentration_max")
            if cm is not None:
                conc_max_all.append(float(cm))
            cn = attn.get("concentration_min")
            if cn is not None:
                conc_min_all.append(float(cn))
            # Track head count for normalization
            hc = attn.get("num_heads") or attn.get("focused_head_count")
            if hc is not None and num_heads_seen is None:
                # Rough: total heads = collapsed + focused + other
                total = attn.get("num_heads")
                if total is not None:
                    num_heads_seen = int(total)

    f["collapsed_rate_mean"] = safe_mean(collapsed_rates)
    f["focused_head_mean"] = safe_mean(focused_counts)
    f["attn_entropy_norm_mean"] = safe_mean(attn_ent_norm_all)
    f["attn_entropy_norm_std"] = safe_std(attn_ent_norm_all)
    f["concentration_max_mean"] = safe_mean(conc_max_all)
    f["concentration_min_mean"] = safe_mean(conc_min_all)

    # Focused head rate (normalized by head count if available)
    n_heads = f.get("arch_num_heads") or num_heads_seen
    if n_heads and n_heads > 0 and f["focused_head_mean"] is not None:
        f["focused_head_rate"] = f["focused_head_mean"] / n_heads
    else:
        f["focused_head_rate"] = None

    # ── Per-Layer Aggregates → Long-Form Table + Shape Summaries ──
    # Discover actual layer count from all steps
    max_layers = 0
    for step in timeline:
        n = len(step.get("layers") or [])
        if n > max_layers:
            max_layers = n

    layer_rows = []
    layer_attn_ent_profile = []   # per-layer mean attn entropy (for shape)
    layer_attn_norm_profile = []  # per-layer mean normalized attn entropy
    layer_l2_profile = []         # per-layer mean l2 norm

    for li in range(max_layers):
        le_vals, ll2_vals, le_norm_vals = [], [], []
        for step in timeline:
            layers = step.get("layers") or []
            if li < len(layers):
                attn = layers[li].get("attention_summary") or {}
                hs = layers[li].get("hidden_summary") or {}
                ae = attn.get("entropy_mean")
                ae_norm = attn.get("entropy_mean_normalized")
                norm = hs.get("l2_norm_mean")
                if ae is not None:
                    le_vals.append(float(ae))
                if ae_norm is not None:
                    le_norm_vals.append(float(ae_norm))
                if norm is not None:
                    ll2_vals.append(float(norm))

        ae_mean = safe_mean(le_vals)
        l2_mean = safe_mean(ll2_vals)
        ae_norm_mean = safe_mean(le_norm_vals)

        # Wide columns (backward compatible)
        f[f"layer_{li:02d}_attn_entropy"] = ae_mean
        f[f"layer_{li:02d}_l2_norm"] = l2_mean

        # Track profiles for shape summaries
        layer_attn_ent_profile.append(ae_mean)
        layer_attn_norm_profile.append(ae_norm_mean)
        layer_l2_profile.append(l2_mean)

        # Long-form row
        layer_rows.append({
            "layer_idx": li,
            "layer_pos_norm": li / max(1, max_layers - 1),
            "attn_entropy_mean": ae_mean,
            "attn_entropy_norm_mean": ae_norm_mean,
            "l2_norm_mean": l2_mean,
        })

    # Layer shape summaries (from profiles)
    for profile, name in [
        (layer_attn_ent_profile, "layer_attn_entropy"),
        (layer_l2_profile, "layer_l2_norm"),
    ]:
        valid = [(i, v) for i, v in enumerate(profile) if v is not None]
        if valid:
            vals_only = [v for _, v in valid]
            peak_idx = valid[np.argmax(vals_only)][0]
            trough_idx = valid[np.argmin(vals_only)][0]
            f[f"{name}_peak_layer"] = peak_idx
            f[f"{name}_trough_layer"] = trough_idx
            f[f"{name}_cross_range"] = max(vals_only) - min(vals_only)
            f[f"{name}_slope"] = safe_slope([v for _, v in valid])
            # Early-mid-late
            n = len(valid)
            third = max(1, n // 3)
            f[f"{name}_early_mean"] = safe_mean(vals_only[:third])
            f[f"{name}_mid_mean"] = safe_mean(vals_only[third:2*third])
            f[f"{name}_late_mean"] = safe_mean(vals_only[2*third:])
            early_m = f[f"{name}_early_mean"]
            late_m = f[f"{name}_late_mean"]
            f[f"{name}_early_late_delta"] = (
                late_m - early_m if early_m is not None and late_m is not None else None
            )
        else:
            for suffix in ["_peak_layer", "_trough_layer", "_cross_range", "_slope",
                           "_early_mean", "_mid_mean", "_late_mean", "_early_late_delta"]:
                f[f"{name}{suffix}"] = None

    # ── Health Flags ──
    f["nan_detected"] = int(hf.get("nan_detected", False))
    f["inf_detected"] = int(hf.get("inf_detected", False))
    f["repetition_detected"] = int(hf.get("repetition_loop_detected", False))
    f["mid_layer_anomaly"] = int(hf.get("mid_layer_anomaly_detected", False))
    f["attention_collapse_detected"] = int(hf.get("attention_collapse_detected", False))
    high_ent = hf.get("high_entropy_steps", 0) or 0
    f["high_entropy_steps"] = high_ent
    f["high_entropy_frac"] = high_ent / max(1, gen_tokens)

    # ── Risk / Early Warning + Saturation ──
    risk_block = ext.get("risk") or {}
    ew_block = ext.get("early_warning") or {}

    risk_score = risk_block.get("risk_score")
    failure_risk = ew_block.get("failure_risk")

    # Use None when block is absent (provenance-aware)
    f["risk_score"] = risk_score if f["trace_has_risk"] else None
    f["failure_risk"] = failure_risk if f["trace_has_early_warning"] else None

    # Saturation flags (only meaningful when score exists)
    f["risk_score_is_zero"] = int(risk_score == 0.0) if risk_score is not None else None
    f["risk_score_is_one"] = int(risk_score == 1.0) if risk_score is not None else None
    f["failure_risk_is_zero"] = int(failure_risk == 0.0) if failure_risk is not None else None
    f["failure_risk_is_one"] = int(failure_risk == 1.0) if failure_risk is not None else None

    f["n_warning_signals"] = len(ew_block.get("warning_signals", []))

    # ── Compound Signals ──
    cs = ext.get("compound_signals", []) or []
    f["n_compound_signals"] = len(cs)
    f["max_compound_severity"] = max((s.get("severity", 0) for s in cs), default=0.0)
    # Length-normalized densities
    f["warning_density_per_100t"] = f["n_warning_signals"] / max(1, gen_tokens) * 100
    f["compound_density_per_100t"] = len(cs) / max(1, gen_tokens) * 100

    cs_names = {s.get("name", "") for s in cs}
    for name in ["context_loss", "confident_confusion", "degenerating_generation",
                 "attention_bottleneck", "confident_repetition_risk"]:
        f[f"cs_{name}"] = int(name in cs_names)

    # ── Prompt Analysis ──
    prompt_surprisals = prompt_analysis.get("prompt_surprisals", []) or []
    f["prompt_surprisal_mean"] = safe_mean(prompt_surprisals)
    f["prompt_surprisal_max"] = safe_max(prompt_surprisals)
    f["prompt_surprisal_std"] = safe_std(prompt_surprisals)
    f["prompt_surprisal_p90"] = safe_percentile(prompt_surprisals, 90)
    f["prompt_token_count"] = len(prompt_surprisals) if prompt_surprisals else None

    basin_all = []
    for ld in (prompt_analysis.get("layers") or []):
        for bs in (ld.get("basin_scores") or []):
            if bs is not None:
                basin_all.append(float(bs))
    f["basin_score_min"] = safe_min(basin_all)
    f["basin_score_mean"] = safe_mean(basin_all)
    f["basin_score_std"] = safe_std(basin_all)

    layer_transforms = prompt_analysis.get("layer_transformations", []) or []
    f["layer_transform_mean"] = safe_mean(layer_transforms)
    f["layer_transform_std"] = safe_std(layer_transforms)
    f["layer_transform_max"] = safe_max(layer_transforms)

    sparse_heads = prompt_analysis.get("sparse_attention_heads", []) or []
    f["n_sparse_heads"] = len(sparse_heads)

    # ── Fingerprint ──
    fp = ext.get("fingerprint", {}).get("vector", []) or []
    f["fingerprint_dim_total"] = len(fp)
    f["fingerprint_dim_kept"] = min(len(fp), 25)
    f["fingerprint_truncated"] = int(len(fp) > 25)
    for i in range(25):
        f[f"fp_{i:02d}"] = float(fp[i]) if i < len(fp) else None

    # ── Performance Timing ──
    perf = ext.get("performance") or {}
    total_ms = perf.get("total_wall_time_ms") or perf.get("total_run_ms")
    f["total_run_ms"] = total_ms

    parent_ops_raw = perf.get("parent_operations", [])
    if isinstance(parent_ops_raw, list):
        parent_ops = {op["name"]: op for op in parent_ops_raw
                      if isinstance(op, dict) and "name" in op}
    else:
        parent_ops = parent_ops_raw or {}

    inference_ms = parent_ops.get("model_inference", {}).get("ms")
    report_ms = parent_ops.get("report_build", {}).get("ms")
    tokenize_ms = parent_ops.get("tokenize", {}).get("ms")

    f["inference_ms"] = inference_ms
    f["report_build_ms"] = report_ms
    f["tokenize_ms"] = tokenize_ms

    # Decomposed overhead
    if total_ms is not None and inference_ms is not None:
        f["overhead_ms"] = total_ms - inference_ms
        f["overhead_pct"] = (f["overhead_ms"] / total_ms) * 100 if total_ms > 0 else None
        known_non_inference = sum(
            v for v in [report_ms, tokenize_ms] if v is not None
        )
        f["known_non_inference_ms"] = known_non_inference
        f["unattributed_ms"] = total_ms - inference_ms - known_non_inference
        f["inference_pct"] = (inference_ms / total_ms) * 100 if total_ms > 0 else None
    else:
        f["overhead_ms"] = None
        f["overhead_pct"] = None
        f["known_non_inference_ms"] = None
        f["unattributed_ms"] = None
        f["inference_pct"] = None

    return f, layer_rows


# ── Grades ──────────────────────────────────────────────────

def load_grades(path: Path) -> Dict[Tuple[str, int], dict]:
    """Load grades keyed by (prompt_key, run_idx)."""
    grades = {}
    if not path.exists():
        log.warning(f"Grades file not found: {path}")
        return grades
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            grades[(r["prompt_key"], r["run_idx"])] = r
    log.info(f"Loaded {len(grades)} grade records from {path.name}")
    return grades


# ── Main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CoreVital feature extraction from traces + grades"
    )
    parser.add_argument("--traces-dir", type=Path, default=TRACES_DIR)
    parser.add_argument("--grades", type=Path, default=GRADES_FILE)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=None,
                        help="Override parquet output path")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    parquet_path = args.output or (output_dir / "features.parquet")

    grades = load_grades(args.grades)

    log.info(f"Scanning traces in {args.traces_dir}...")
    trace_files = sorted(args.traces_dir.rglob("*.json"))
    log.info(f"Found {len(trace_files)} trace files")

    rows = []
    layer_rows_all = []
    error_log = []

    for tp in tqdm(trace_files, desc="Extracting"):
        # ── Parse file path ──
        try:
            rel = tp.relative_to(args.traces_dir)
            parts = rel.parts  # model/dataset/qid_runNN.json
            model_short = parts[0]
            dataset = parts[1]
            fname = parts[2].replace(".json", "")
            run_match = fname.rsplit("_run", 1)
            qid = run_match[0]
            run_idx = int(run_match[1]) if len(run_match) > 1 else 0
            prompt_key = f"{model_short}/{dataset}/{qid}"
        except Exception as e:
            error_log.append({
                "file": str(tp), "stage": "path_parse",
                "error_type": type(e).__name__, "error": str(e),
            })
            continue

        # ── Load trace ──
        try:
            with open(tp) as fh:
                trace = json.load(fh)
        except Exception as e:
            error_log.append({
                "file": str(tp), "stage": "json_load",
                "error_type": type(e).__name__, "error": str(e),
            })
            continue

        # ── Extract features ──
        try:
            features, layer_rows_for_run = extract_features(trace)
        except Exception as e:
            error_log.append({
                "file": str(tp), "stage": "feature_extraction",
                "error_type": type(e).__name__, "error": str(e),
                "traceback": traceback.format_exc(limit=3),
            })
            continue

        # ── Metadata ──
        features["prompt_key"] = prompt_key
        features["model"] = model_short
        features["dataset"] = dataset
        features["question_id"] = qid
        features["run_idx"] = run_idx
        features["trace_path"] = str(rel)

        # ── Grade join ──
        grade = grades.get((prompt_key, run_idx))
        features["grade_found"] = int(grade is not None)
        if grade:
            features["correct"] = grade.get("correct")
            features["format_failure"] = grade.get("format_failure")
            features["temperature"] = grade.get("temperature")
            features["seed"] = grade.get("seed")
        else:
            features["correct"] = None
            features["format_failure"] = None
            features["temperature"] = None
            features["seed"] = None

        rows.append(features)

        # ── Layer rows ──
        for lr in layer_rows_for_run:
            lr["prompt_key"] = prompt_key
            lr["model"] = model_short
            lr["dataset"] = dataset
            lr["question_id"] = qid
            lr["run_idx"] = run_idx
            lr["correct"] = features["correct"]
            lr["format_failure"] = features["format_failure"]
            lr["temperature"] = features["temperature"]
            layer_rows_all.append(lr)

    # ── Build DataFrames ──
    df = pd.DataFrame(rows)
    log.info(f"Extracted {len(df)} rows, {len(error_log)} errors")

    if len(df) == 0:
        log.error("No features extracted. Exiting.")
        sys.exit(1)

    # ── Derived: pass rate and difficulty ──
    if "correct" in df.columns:
        # Per-model pass rate
        df["pass_rate"] = df.groupby(
            ["model", "dataset", "question_id"]
        )["correct"].transform("mean")

        # Pooled difficulty (named honestly)
        df["empirical_difficulty_pooled"] = 1.0 - df.groupby(
            ["dataset", "question_id"]
        )["correct"].transform("mean")

        # Backward compat alias
        df["question_difficulty"] = df["empirical_difficulty_pooled"]

    # ── Save run-level features ──
    df.to_parquet(parquet_path, index=False)
    log.info(f"Saved {parquet_path} ({len(df)} rows, {len(df.columns)} columns)")

    # ── Save layer long-form table ──
    if layer_rows_all:
        layer_df = pd.DataFrame(layer_rows_all)
        layer_path = output_dir / "layer_long.parquet"
        layer_df.to_parquet(layer_path, index=False)
        log.info(f"Saved {layer_path} ({len(layer_df)} rows)")
    else:
        layer_df = pd.DataFrame()

    # ── Build prompt-level table ──
    prompt_cols_available = [c for c in PROMPT_INVARIANT_COLUMNS if c in df.columns]
    if prompt_cols_available and "correct" in df.columns:
        prompt_group = df.groupby(["model", "dataset", "question_id"])
        prompt_agg = {"pass_rate": "first"}
        for c in prompt_cols_available:
            prompt_agg[c] = "first"
        prompt_df = prompt_group.agg(prompt_agg).reset_index()
        # Add n_runs and difficulty
        prompt_df["n_runs"] = prompt_group.size().values
        prompt_df = prompt_df.merge(
            df.groupby(["dataset", "question_id"])["empirical_difficulty_pooled"]
            .first().reset_index(),
            on=["dataset", "question_id"], how="left",
        )
        prompt_path = output_dir / "prompt_level.parquet"
        prompt_df.to_parquet(prompt_path, index=False)
        log.info(f"Saved {prompt_path} ({len(prompt_df)} rows)")
    else:
        prompt_df = pd.DataFrame()

    # ── Manifest ──
    null_rates = (df.isnull().sum() / len(df)).to_dict()
    manifest = build_manifest(
        script_name="extract_features.py",
        inputs={
            "traces_dir": _path_rel_to_repo(args.traces_dir),
            "grades_file": _path_rel_to_repo(args.grades),
            "n_trace_files": len(trace_files),
            "n_grade_records": len(grades),
        },
        outputs=[
            _path_rel_to_repo(parquet_path),
            _path_rel_to_repo(output_dir / "layer_long.parquet"),
            _path_rel_to_repo(output_dir / "prompt_level.parquet"),
        ],
        row_counts={
            "features": len(df),
            "layer_long": len(layer_df) if not layer_df.empty else 0,
            "prompt_level": len(prompt_df) if not prompt_df.empty else 0,
            "errors": len(error_log),
        },
        extra={
            "columns": list(df.columns),
            "prompt_invariant_columns": [c for c in PROMPT_INVARIANT_COLUMNS if c in df.columns],
            "performance_columns": [c for c in PERFORMANCE_COLUMNS if c in df.columns],
            "null_rates_top20": dict(
                sorted(null_rates.items(), key=lambda x: -x[1])[:20]
            ),
            "error_types": pd.DataFrame(error_log)["error_type"].value_counts().to_dict()
                if error_log else {},
            "error_sample": error_log[:20],
            "per_model_dataset": {
                f"{m}/{d}": {
                    "n_prompts": int(g["question_id"].nunique()),
                    "n_runs": len(g),
                    "accuracy": float(g["correct"].mean()) if "correct" in g.columns and g["correct"].notna().any() else None,
                    "format_failure_rate": float(g["format_failure"].mean()) if "format_failure" in g.columns and g["format_failure"].notna().any() else None,
                }
                for (m, d), g in df.groupby(["model", "dataset"])
            },
        },
    )
    manifest_path = output_dir / "extraction_manifest.json"
    save_json(manifest, manifest_path)
    log.info(f"Saved {manifest_path}")

    # ── Human-readable summary ──
    summary_path = output_dir / "extraction_summary.md"
    with open(summary_path, "w") as fh:
        fh.write("# CoreVital Feature Extraction Summary\n\n")
        fh.write(f"- **Traces scanned:** {len(trace_files)}\n")
        fh.write(f"- **Rows extracted:** {len(df)}\n")
        fh.write(f"- **Extraction errors:** {len(error_log)}\n")
        fh.write(f"- **Features:** {len(df.columns)} columns\n")
        fh.write(f"- **Layer rows:** {len(layer_df) if not layer_df.empty else 0}\n")
        fh.write(f"- **Prompt-level rows:** {len(prompt_df) if not prompt_df.empty else 0}\n\n")
        fh.write("## Per Model/Dataset\n\n")
        fh.write("| Model | Dataset | Prompts | Runs | Accuracy | Format Fail % |\n")
        fh.write("|-------|---------|---------|------|----------|---------------|\n")
        for (m, d), g in df.groupby(["model", "dataset"]):
            acc = g["correct"].mean() if "correct" in g.columns and g["correct"].notna().any() else float("nan")
            ff = g["format_failure"].mean() * 100 if "format_failure" in g.columns and g["format_failure"].notna().any() else float("nan")
            fh.write(f"| {m} | {d} | {g['question_id'].nunique()} | {len(g)} | {acc:.1%} | {ff:.1f}% |\n")
        if error_log:
            fh.write(f"\n## Errors ({len(error_log)} total)\n\n")
            for stage, count in pd.DataFrame(error_log)["stage"].value_counts().items():
                fh.write(f"- {stage}: {count}\n")
    log.info(f"Saved {summary_path}")

    # ── Console summary ──
    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"  {len(df)} rows, {len(error_log)} errors, {len(df.columns)} features")
    for (m, d), g in df.groupby(["model", "dataset"]):
        n_prompts = g["question_id"].nunique()
        acc = g["correct"].mean() if "correct" in g.columns and g["correct"].notna().any() else float("nan")
        print(f"  {m}/{d}: {n_prompts} prompts, {len(g)} runs, {acc:.1%} accuracy")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()