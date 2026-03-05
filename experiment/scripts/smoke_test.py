#!/usr/bin/env python3
"""
CoreVital Validation Experiment v2 — Smoke Test

Run 5 GSM8K problems × 3 runs each = 15 traces under sampling.
Verify that signals have variance, compound signals fire at different rates,
and risk_score actually spreads.

Run this BEFORE the full experiment. Costs ~$2 and takes ~20 minutes.

Usage:
    python3 smoke_test.py
"""

import json
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch

EXPERIMENT_DIR = Path.home() / "experiment"
DATA_DIR = EXPERIMENT_DIR / "data"
SMOKE_DIR = EXPERIMENT_DIR / "smoke_test"


def main():
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load 5 GSM8K questions ---
    gsm_path = DATA_DIR / "gsm8k.jsonl"
    if not gsm_path.exists():
        print(f"ERROR: {gsm_path} not found. Run setup first.")
        sys.exit(1)

    questions = []
    with open(gsm_path) as f:
        for line in f:
            questions.append(json.loads(line))
            if len(questions) >= 5:
                break

    print(f"Loaded {len(questions)} GSM8K questions.")

    # --- Run 15 traces ---
    from CoreVital.config import Config
    from CoreVital.instrumentation.collector import InstrumentationCollector
    from CoreVital.reporting.report_builder import ReportBuilder
    from CoreVital.utils.serialization import serialize_report_to_json

    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    print(f"\nLoading model: {model_id}")

    config = Config.from_default()
    config.model.hf_id = model_id
    config.device.requested = "auto"
    config.generation.max_new_tokens = 512
    config.generation.do_sample = True
    config.generation.temperature = 0.7
    config.generation.top_p = 0.95
    config.generation.top_k = 50
    config.capture.capture_mode = "full"
    config.prompt_telemetry.enabled = True

    collector = InstrumentationCollector(config)
    builder = ReportBuilder(config)

    all_reports = []

    for qi, question in enumerate(questions):
        prompt = (
            f"Solve the following math problem step by step. "
            f"After your solution, write the final numerical answer "
            f"on the last line preceded by \"####\".\n\n"
            f"Problem: {question['question']}"
        )

        print(f"\n--- Question {qi+1}/{len(questions)} ---")
        print(f"  {question['question'][:100]}...")

        # Run 3 times with different seeds (batch GPU work)
        batch_results = []
        for run_i in range(3):
            config.generation.seed = 42 + run_i
            result = collector.run(prompt)
            batch_results.append((result, prompt, run_i))
            print(f"  Run {run_i+1}/3: {len(result.generated_token_ids)} tokens generated")

        # Now build reports and serialize (CPU work, batched)
        for result, prompt_text, run_i in batch_results:
            report = builder.build(result, prompt_text)
            report_dict = json.loads(serialize_report_to_json(report))

            # Save trace
            trace_path = SMOKE_DIR / f"q{qi}_run{run_i}.json"
            with open(trace_path, "w") as f:
                json.dump(report_dict, f, indent=2)

            all_reports.append({
                "question_idx": qi,
                "run_idx": run_i,
                "report": report_dict,
                "generated_tokens": len(result.generated_token_ids),
            })

    # --- Analyze the 15 traces ---
    print(f"\n{'='*60}")
    print("SMOKE TEST ANALYSIS")
    print(f"{'='*60}")

    # Build a comprehensive summary dict that gets saved as JSON
    summary = {
        "meta": {
            "model": model_id,
            "n_questions": len(questions),
            "n_runs_per_question": 3,
            "n_traces": len(all_reports),
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 50,
            "max_new_tokens": 512,
        },
        "per_trace": [],
        "signal_stats": {},
        "within_question": [],
        "compound_signal_rates": {},
        "health_flag_rates": {},
        "prompt_analysis": [],
        "verdict": {},
    }

    # Extract all signals per trace
    from collections import Counter
    compound_counter = Counter()
    health_flags_counter = Counter()

    for item in all_reports:
        r = item["report"]
        ext = r.get("extensions", {})
        timeline = r.get("timeline", [])
        hf = r.get("health_flags", {}) or {}
        prompt_analysis_data = r.get("prompt_analysis", {}) or {}

        # Logits from timeline
        entropies = [s.get("logits_summary", {}).get("entropy") for s in timeline if s.get("logits_summary")]
        entropies = [e for e in entropies if e is not None]
        surprisals = [s.get("logits_summary", {}).get("surprisal") for s in timeline if s.get("logits_summary")]
        surprisals = [s for s in surprisals if s is not None]
        margins = [s.get("logits_summary", {}).get("top_k_margin") for s in timeline if s.get("logits_summary")]
        margins = [m for m in margins if m is not None]
        perplexities = [s.get("logits_summary", {}).get("perplexity") for s in timeline if s.get("logits_summary")]
        perplexities = [p for p in perplexities if p is not None]

        # Attention stats from layers
        collapsed_rates = []
        focused_counts = []
        attn_entropies_by_layer = {}
        l2_norms_by_layer = {}

        for step in timeline:
            for li, layer in enumerate(step.get("layers") or []):
                attn = layer.get("attention_summary", {}) or {}
                hs = layer.get("hidden_summary", {}) or {}
                cr = attn.get("collapsed_head_rate")
                if cr is not None: collapsed_rates.append(cr)
                fc = attn.get("focused_head_count")
                if fc is not None: focused_counts.append(fc)
                ae = attn.get("entropy_mean")
                if ae is not None:
                    attn_entropies_by_layer.setdefault(li, []).append(ae)
                norm = hs.get("l2_norm_mean")
                if norm is not None:
                    l2_norms_by_layer.setdefault(li, []).append(norm)

        # Risk and early warning
        risk = ext.get("risk", {}) or {}
        ew = ext.get("early_warning", {}) or {}

        # Compound signals
        cs = ext.get("compound_signals", []) or []
        cs_names = [s.get("name", "?") for s in cs]
        for name in cs_names:
            compound_counter[name] += 1

        # Health flags
        for flag in ["nan_detected", "inf_detected", "repetition_loop_detected",
                     "mid_layer_anomaly_detected", "attention_collapse_detected"]:
            if hf.get(flag):
                health_flags_counter[flag] += 1

        # Prompt analysis
        prompt_surprisals = prompt_analysis_data.get("prompt_surprisals", []) or []
        basin_scores = []
        for ld in (prompt_analysis_data.get("layers") or []):
            for bs in (ld.get("basin_scores") or []):
                if bs is not None: basin_scores.append(bs)
        layer_transforms = prompt_analysis_data.get("layer_transformations", []) or []

        # Entropy slope
        slope = None
        if len(entropies) >= 3:
            x = np.arange(len(entropies))
            slope = float(np.polyfit(x, entropies, 1)[0])

        trace_summary = {
            "question_idx": item["question_idx"],
            "run_idx": item["run_idx"],
            "generated_tokens": item["generated_tokens"],
            "n_timeline_steps": len(timeline),
            "entropy_mean": float(np.mean(entropies)) if entropies else None,
            "entropy_std": float(np.std(entropies)) if len(entropies) >= 2 else None,
            "entropy_slope": slope,
            "surprisal_mean": float(np.mean(surprisals)) if surprisals else None,
            "surprisal_std": float(np.std(surprisals)) if len(surprisals) >= 2 else None,
            "margin_mean": float(np.mean(margins)) if margins else None,
            "perplexity_mean": float(np.mean(perplexities)) if perplexities else None,
            "collapsed_rate_mean": float(np.mean(collapsed_rates)) if collapsed_rates else None,
            "focused_head_mean": float(np.mean(focused_counts)) if focused_counts else None,
            "l2_norm_last_layer_mean": float(np.mean(l2_norms_by_layer[max(l2_norms_by_layer)])) if l2_norms_by_layer else None,
            "risk_score": risk.get("risk_score", 0),
            "risk_factors": risk.get("risk_factors", []),
            "failure_risk": ew.get("failure_risk", 0),
            "warning_signals": ew.get("warning_signals", []),
            "n_compound_signals": len(cs),
            "compound_signal_names": cs_names,
            "compound_signal_severities": [s.get("severity", 0) for s in cs],
            "high_entropy_steps": hf.get("high_entropy_steps", 0),
            "health_flags": {k: v for k, v in hf.items() if v},
            # Prompt analysis
            "prompt_surprisal_mean": float(np.mean(prompt_surprisals)) if prompt_surprisals else None,
            "prompt_surprisal_max": float(max(prompt_surprisals)) if prompt_surprisals else None,
            "basin_score_min": float(min(basin_scores)) if basin_scores else None,
            "basin_score_mean": float(np.mean(basin_scores)) if basin_scores else None,
            "layer_transform_mean": float(np.mean(layer_transforms)) if layer_transforms else None,
            "n_layers_captured": len(attn_entropies_by_layer),
        }
        summary["per_trace"].append(trace_summary)

    # Aggregate signal statistics
    signal_names = [
        "entropy_mean", "entropy_std", "entropy_slope", "surprisal_mean", "surprisal_std",
        "margin_mean", "perplexity_mean", "collapsed_rate_mean", "focused_head_mean",
        "l2_norm_last_layer_mean", "risk_score", "failure_risk", "n_compound_signals",
        "prompt_surprisal_mean", "basin_score_min", "basin_score_mean", "layer_transform_mean",
    ]

    pass_fail = True
    warnings_list = []

    print(f"\n  Total traces: {len(all_reports)}")
    print(f"\n  {'Signal':<30s} {'mean':>8s} {'std':>8s} {'min':>8s} {'max':>8s} {'range':>8s}")
    print(f"  {'-'*72}")

    for sig in signal_names:
        vals = [t[sig] for t in summary["per_trace"] if t.get(sig) is not None]
        if not vals:
            print(f"  {sig:<30s} {'NO DATA':>8s}")
            summary["signal_stats"][sig] = {"status": "NO_DATA"}
            warnings_list.append(f"{sig}: no data")
            continue

        arr = np.array(vals, dtype=float)
        stats = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "range": float(np.max(arr) - np.min(arr)),
            "n": len(vals),
        }
        summary["signal_stats"][sig] = stats
        print(f"  {sig:<30s} {stats['mean']:>8.4f} {stats['std']:>8.4f} "
              f"{stats['min']:>8.4f} {stats['max']:>8.4f} {stats['range']:>8.4f}")

        # Degenerate checks
        if sig == "risk_score" and stats["std"] < 0.01:
            w = f"risk_score nearly flat (std={stats['std']:.4f})"
            warnings_list.append(w)
            print(f"    ⚠ WARNING: {w}")
            pass_fail = False
        if sig == "entropy_mean" and stats["std"] < 0.01:
            w = f"entropy_mean nearly flat (std={stats['std']:.4f})"
            warnings_list.append(w)
            print(f"    ⚠ WARNING: {w}")
            pass_fail = False
        if sig == "risk_score" and stats["range"] < 0.05:
            w = f"risk_score range only {stats['range']:.4f}"
            warnings_list.append(w)
            print(f"    ⚠ WARNING: {w}")

    # Compound signals
    summary["compound_signal_rates"] = {}
    print(f"\n  Compound signals:")
    for name, count in compound_counter.most_common():
        rate = count / len(all_reports)
        summary["compound_signal_rates"][name] = {"count": count, "rate": round(rate, 3)}
        print(f"    {name}: {count}/{len(all_reports)} ({rate:.0%})")
        if rate > 0.95:
            w = f"{name} fires on {rate:.0%} of traces — uninformative"
            warnings_list.append(w)
            print(f"    ⚠ WARNING: {w}")
        elif rate < 0.05 and rate > 0:
            print(f"    (rare)")
    if not compound_counter:
        print(f"    None fired")
        summary["compound_signal_rates"]["NONE"] = {"count": 0, "rate": 0}

    # Health flags
    summary["health_flag_rates"] = {}
    print(f"\n  Health flags:")
    for flag, count in health_flags_counter.most_common():
        rate = count / len(all_reports)
        summary["health_flag_rates"][flag] = {"count": count, "rate": round(rate, 3)}
        print(f"    {flag}: {count}/{len(all_reports)} ({rate:.0%})")
    if not health_flags_counter:
        print(f"    None fired")

    # Within-question variance
    print(f"\n  Within-question variance (same prompt, different seeds):")
    for qi in range(len(questions)):
        traces = [t for t in summary["per_trace"] if t["question_idx"] == qi]
        q_data = {
            "question_idx": qi,
            "tokens": [t["generated_tokens"] for t in traces],
            "entropy_mean": [t["entropy_mean"] for t in traces if t["entropy_mean"] is not None],
            "risk_score": [t["risk_score"] for t in traces],
            "failure_risk": [t["failure_risk"] for t in traces],
            "margin_mean": [t["margin_mean"] for t in traces if t["margin_mean"] is not None],
            "n_compound_signals": [t["n_compound_signals"] for t in traces],
        }
        summary["within_question"].append(q_data)
        print(f"    Q{qi}: tokens={q_data['tokens']} "
              f"entropy={[f'{e:.4f}' for e in q_data['entropy_mean']]} "
              f"risk={[f'{r:.4f}' for r in q_data['risk_score']]} "
              f"failure_risk={[f'{fr:.4f}' for fr in q_data['failure_risk']]}")

    # Prompt analysis summary
    print(f"\n  Prompt analysis:")
    for qi in range(len(questions)):
        # All runs of same prompt share the same prompt analysis, so take first
        traces = [t for t in summary["per_trace"] if t["question_idx"] == qi]
        if traces:
            t = traces[0]
            print(f"    Q{qi}: prompt_surprisal={t['prompt_surprisal_mean']:.4f}" if t["prompt_surprisal_mean"] else f"    Q{qi}: prompt_surprisal=N/A", end="")
            print(f"  basin_min={t['basin_score_min']:.4f}" if t["basin_score_min"] else "  basin_min=N/A", end="")
            print(f"  layers_captured={t['n_layers_captured']}")

    # Verdict
    summary["verdict"] = {
        "passed": pass_fail,
        "warnings": warnings_list,
    }

    print(f"\n{'='*60}")
    if pass_fail:
        print("✓ SMOKE TEST PASSED — signals have variance under sampling.")
        print("  Proceed to full experiment.")
    else:
        print("✗ SMOKE TEST FAILED — see warnings above.")
        print("  DO NOT run the full experiment until issues are resolved.")
    print(f"{'='*60}")

    # Save the summary JSON (this is what you send to Claude)
    summary_path = SMOKE_DIR / "smoke_test_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary saved to: {summary_path}")
    print(f"  Size: {summary_path.stat().st_size / 1024:.1f} KB")
    print(f"\n  >>> Send THIS FILE to Claude for review: {summary_path}")
    print(f"  >>> (NOT the raw traces — they're too large)")

    # Also save the full console output
    # The user can copy-paste the terminal output too


if __name__ == "__main__":
    main()