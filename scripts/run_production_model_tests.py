#!/usr/bin/env python3
"""
Production model test suite for RunPod / GPU.

Runs CoreVital on all listed instruct models, writes to SQLite only (no per-run JSON),
and appends runtime metrics to a log file. Run AFTER push so the pod installs the
latest CoreVital. See docs/production-model-test-suite.md.

Usage:
  python scripts/run_production_model_tests.py --out runs --log runs/runtime_metrics.log
  # Optional: sync after run
  # aws s3 cp runs/corevital.db s3://your-bucket/corevital-runs/runpod-$(date +%Y%m%d-%H%M).db
"""

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Instruct models (coherent answers so metrics reflect good/bad output)
PRODUCTION_INSTRUCT_MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen2-0.5B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
]

PROMPTS = [
    "What is 2+2? Answer in one short sentence.",
    "Name one benefit of exercise in one sentence.",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CoreVital on production instruct models (RunPod).")
    parser.add_argument("--out", type=str, default="runs", help="Output directory (DB and log)")
    parser.add_argument(
        "--log", type=str, default=None, help="Runtime metrics log file (default: <out>/runtime_metrics.log)"
    )
    parser.add_argument("--models", type=str, nargs="+", default=None, help="Override model list")
    parser.add_argument("--max-new-tokens", type=int, default=30, help="Max new tokens per run")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log) if args.log else out / "runtime_metrics.log"
    models = args.models or PRODUCTION_INSTRUCT_MODELS

    for model_id in models:
        for prompt in PROMPTS:
            cmd = [
                sys.executable,
                "-m",
                "CoreVital.cli",
                "run",
                "--model",
                model_id,
                "--prompt",
                prompt,
                "--max_new_tokens",
                str(args.max_new_tokens),
                "--sink",
                "sqlite",
                "--out",
                str(out),
                "--perf",
                "summary",
            ]
            if args.dry_run:
                print(" ".join(cmd))
                continue
            start = datetime.now(timezone.utc).isoformat()
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                # One-line summary for quick scan
                line = (
                    f"{start}\tmodel={model_id}\tprompt_len={len(prompt)}"
                    f"\texit={result.returncode}\tstdout_len={len(result.stdout)}"
                    f"\tstderr_len={len(result.stderr)}\n"
                )
                with open(log_path, "a") as f:
                    f.write(line)
                # Append runtime metrics (--perf summary and any stderr) so you can inspect without re-running
                with open(log_path, "a") as f:
                    f.write("--- stderr (runtime/perf) ---\n")
                    f.write(result.stderr or "(none)\n")
                if result.returncode != 0:
                    print(f"FAILED: {model_id} | {prompt[:40]}...", file=sys.stderr)
                    return 1
            except subprocess.TimeoutExpired:
                with open(log_path, "a") as f:
                    f.write(f"{start}\tmodel={model_id}\tTIMEOUT\n")
                print(f"TIMEOUT: {model_id}", file=sys.stderr)
                return 1
            except Exception as e:
                with open(log_path, "a") as f:
                    f.write(f"{start}\tmodel={model_id}\tERROR\t{e}\n")
                raise

    print(f"All {len(models)} models x {len(PROMPTS)} prompts OK. DB: {out / 'corevital.db'}, log: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
