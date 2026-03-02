#!/usr/bin/env python3
"""
Profile a single CoreVital run to see where time is spent (CPU vs transfer vs GPU).

Run on the POD to pin down the bottleneck. Usage:
  python3 profile_one_trace.py

Output: one trace with cProfile, then print top 30 functions by cumulative time.
Also prints parent_operations from performance extension if --perf was used.
"""

import cProfile
import io
import pstats
import sys
from pathlib import Path

# Add repo to path if needed
repo = Path(__file__).resolve().parents[2]
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))

def run_one_trace():
    from CoreVital.config import Config
    from CoreVital.instrumentation.collector import InstrumentationCollector
    from CoreVital.reporting.report_builder import ReportBuilder

    config = Config.from_default()
    config.model.hf_id = "Qwen/Qwen2.5-3B-Instruct"
    config.model.trust_remote_code = True
    config.device.requested = "auto"
    config.generation.max_new_tokens = 32
    config.generation.do_sample = False
    config.generation.seed = 42
    config.capture.capture_mode = "full"
    config.prompt_telemetry.enabled = True
    config.performance.mode = "strict"

    collector = InstrumentationCollector(config)
    prompt = (
        "The following is a multiple choice question.\n\n"
        "Question: What is 2+2?\nA. 3\nB. 4\nC. 5\nD. 6\n\n"
        "Answer with just the letter (A, B, C, or D):"
    )
    results = collector.run(prompt)
    report = ReportBuilder(config).build(results, prompt)
    return report

def main():
    print("Profiling one trace (Qwen 2.5 3B, 32 tokens, full capture)...")
    prof = cProfile.Profile()
    prof.enable()
    report = run_one_trace()
    prof.disable()

    # Summary stats
    if report.summary:
        print(f"\nGenerated tokens: {report.summary.generated_tokens}")
    if report.extensions.get("performance"):
        perf = report.extensions["performance"]
        print(f"\n--- extensions.performance ---")
        print(f"total_wall_time_ms: {perf.get('total_wall_time_ms')}")
        for p in perf.get("parent_operations", []):
            print(f"  {p.get('name')}: {p.get('ms')} ms ({p.get('pct')}%)")

    # Top 30 by cumulative time
    s = io.StringIO()
    ps = pstats.Stats(prof, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats(30)
    print("\n--- cProfile top 30 (cumulative time) ---")
    print(s.getvalue())

if __name__ == "__main__":
    main()
