#!/usr/bin/env python3
"""
Real-world test: CoreVitalMonitor with meta-llama/Llama-3.1-8B (4-bit).
Runs one instrumented run, checks report/risk/health, then replays via async stream.
Usage: python scripts/test_llama_stream.py [--8bit] [--cpu]
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from CoreVital import CoreVitalMonitor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--8bit", dest="use_8bit", action="store_true", help="Use 8-bit instead of 4-bit")
    ap.add_argument("--cpu", action="store_true", help="Use CPU (no quantization; may OOM on 8B)")
    args = ap.parse_args()

    model_id = "meta-llama/Llama-3.1-8B"
    device = "cpu" if args.cpu else "auto"
    load_in_4bit = not args.use_8bit and not args.cpu
    load_in_8bit = args.use_8bit and not args.cpu

    print("CoreVital real-world test: Llama-3.1-8B")
    print(f"  device={device}, 4bit={load_in_4bit}, 8bit={load_in_8bit}")
    print()

    monitor = CoreVitalMonitor(
        capture_mode="summary",
        intervene_on_risk_above=0.8,
        max_new_tokens=12,
        device=device,
        seed=42,
    )

    run_kw = dict(max_new_tokens=12, device=device, seed=42)
    if load_in_4bit:
        run_kw["load_in_4bit"] = True
    if load_in_8bit:
        run_kw["load_in_8bit"] = True

    prompt = "What is 2 + 2? Answer in one word."

    # Run (CLI-equivalent: full instrumentation + report)
    print("Running instrumented generation...")
    try:
        monitor.run(model_id, prompt, **run_kw)
    except Exception as e:
        print(f"Run failed: {e}")
        return 1

    # Check report and summary
    report = monitor.get_report()
    if report is None:
        print("No report produced.")
        return 1
    print(
        "Report: trace_id=%s model=%s total_steps=%s"
        % (
            report.trace_id[:12],
            report.model.hf_id,
            report.summary.total_steps,
        )
    )
    risk = monitor.get_risk_score()
    flags = monitor.get_health_flags()
    print("Risk score: %.4f" % risk)
    print("Health flags: %s" % flags)
    print("Should intervene: %s" % monitor.should_intervene())
    summary = monitor.get_summary()
    assert "risk_score" in summary and "health_flags" in summary
    if "narrative" in summary:
        print("Narrative: %s" % (summary["narrative"].get("summary", "")[:80] + "..."))
    print()

    # Async stream: runs generation again in thread, then replays timeline as events
    async def consume_stream():
        events = []
        async for ev in monitor.stream(model_id, prompt, **run_kw):
            events.append(ev)
        return events

    print("Replaying stream (async)...")
    events = asyncio.run(consume_stream())
    print("Stream events: %d (expected ~12 generated steps)" % len(events))
    if events:
        first = events[0]
        print("  First event keys: %s" % list(first.keys()))
        print(
            "  Sample: step_index=%s token_text=%r entropy=%s"
            % (
                first.get("step_index"),
                first.get("token_text"),
                first.get("entropy"),
            )
        )
    assert len(events) == report.summary.generated_tokens, "Stream length %d vs generated_tokens %d" % (
        len(events),
        report.summary.generated_tokens,
    )
    print()
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
