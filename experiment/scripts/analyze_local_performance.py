#!/usr/bin/env python3
"""
Analyze performance data from local trace files (runs/ or custom dir).
Use this on your LOCAL machine to compare with pod timings.

Finds trace_*.json from the last N days that have extensions.performance,
and prints total_wall_time_ms, generated_tokens, ms/token, and parent_operations
breakdown. If you have pod numbers, compare ms/token to see if local is faster.

Usage:
  python3 analyze_local_performance.py [--runs-dir PATH] [--days 2]
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

def main():
    parser = argparse.ArgumentParser(description="Analyze performance in local trace JSONs")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"),
                        help="Directory containing trace_*.json (default: runs)")
    parser.add_argument("--days", type=int, default=2,
                        help="Only include traces modified in last N days (default: 2)")
    args = parser.parse_args()

    runs_dir = args.runs_dir.resolve()
    if not runs_dir.exists():
        print(f"Directory not found: {runs_dir}", file=sys.stderr)
        sys.exit(1)

    cutoff = datetime.now() - timedelta(days=args.days)
    cutoff_ts = cutoff.timestamp()

    traces = []
    for f in runs_dir.glob("trace_*.json"):
        if f.stat().st_mtime < cutoff_ts:
            continue
        try:
            with open(f) as fp:
                data = json.load(fp)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Skipping {f.name}: {e}", file=sys.stderr)
            continue
        perf = (data.get("extensions") or {}).get("performance")
        if not perf:
            continue
        summary = data.get("summary") or {}
        gen_tokens = summary.get("generated_tokens", 0)
        total_ms = perf.get("total_wall_time_ms") or 0
        traces.append({
            "file": f.name,
            "total_ms": total_ms,
            "generated_tokens": gen_tokens,
            "ms_per_token": total_ms / gen_tokens if gen_tokens else None,
            "perf": perf,
        })

    if not traces:
        print(f"No traces with extensions.performance in {runs_dir} from last {args.days} days.")
        print("Run CoreVital with --perf summary (or strict) to get performance data.")
        return

    print(f"Found {len(traces)} trace(s) with performance data (last {args.days} days)\n")
    print(f"{'File':<30} {'total_ms':>12} {'tokens':>8} {'ms/tok':>10}")
    print("-" * 62)
    for t in traces:
        ms_tok = f"{t['ms_per_token']:.1f}" if t["ms_per_token"] is not None else "—"
        print(f"{t['file']:<30} {t['total_ms']:>12.0f} {t['generated_tokens']:>8} {ms_tok:>10}")

    avg_ms = sum(t["total_ms"] for t in traces) / len(traces)
    avg_tok = sum(t["generated_tokens"] for t in traces) / len(traces)
    avg_ms_per_tok = avg_ms / avg_tok if avg_tok else 0
    print("-" * 62)
    print(f"{'AVG':<30} {avg_ms:>12.0f} {avg_tok:>8.0f} {avg_ms_per_tok:>10.1f}")

    # Parent operations from first trace (same config usually)
    print("\n--- parent_operations (first trace) ---")
    for p in traces[0]["perf"].get("parent_operations", []):
        print(f"  {p.get('name', '?'):<25} {p.get('ms', 0):>10.0f} ms  ({p.get('pct', 0):.1f}%)")

    print("\nCompare: pod dry run was ~97–113s per MMLU question (32 tokens) ≈ 3000–3500 ms/token.")
    print("If local ms/token is much lower, host CPU/transfer is likely the pod bottleneck.")

if __name__ == "__main__":
    main()
