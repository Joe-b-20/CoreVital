#!/usr/bin/env python3
"""Generate demo trace JSON files for the CoreVital dashboard.

Runs CoreVital on 4 models (Llama 3.1, Mistral, Llama 3.2, FLAN-T5) with CUDA,
4-bit quantization, full capture, and strict performance. Writes individual
trace JSON files (and performance breakdown files) to a demo directory.

Default output directory:
  /home/joebachir20/corevital-dashboard/public/demo

Requirements: conda env llm_hm (with CUDA, bitsandbytes, CoreVital deps), and
HuggingFace model access. Run from repo root:

  conda activate llm_hm
  pip install -e .   # if not already
  python scripts/gen_demo_traces.py

The script uses conda env llm_hm when invoking corevital (via conda run -n llm_hm).

Optional:
  --dry-run        Print commands only
  --out DIR        Output directory (default: /home/joebachir20/corevital-dashboard/public/demo)
  --conda-env ENV  Conda environment name (default: llm_hm). Use "" to use current Python.
  --timeout SEC    Timeout per model run in seconds (default: 1800). Use 3600 for slow GPUs.
  --keep-existing  Do not delete existing trace_*.json files in output dir
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# (model_id, prompt) — interesting prompts, not trivial. Qwen not supported (attention capture).
DEMO_RUNS = [
    (
        "meta-llama/Llama-3.1-8B-Instruct",
        "You are a historian. In exactly three short paragraphs, compare the causes of "
        "the French Revolution and the American Revolution, then state one way they diverged.",
    ),
    (
        "mistralai/Mistral-7B-Instruct-v0.2",
        "A user asks: 'What's the best way to debug a race condition in distributed systems?' "
        "Reply as a senior engineer: one concrete strategy, one tool, and one pitfall to avoid.",
    ),
    (
        "meta-llama/Llama-3.2-3B-Instruct",
        "List five speculative but plausible advances in quantum error correction by 2030. "
        "For each, one sentence only.",
    ),
    (
        "google/flan-t5-large",
        "Summarize: The Mediterranean diet has been linked to lower cardiovascular risk and "
        "better cognitive function in studies, but confounding factors make causal claims hard. "
        "Recent randomized trials have had mixed results.",
    ),
]


def _clean_output_dir(out_dir: Path) -> int:
    removed = 0
    for path in out_dir.glob("trace_*.json"):
        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate demo CoreVital trace JSON files for the dashboard."
    )
    parser.add_argument(
        "--out",
        type=str,
        default="/home/joebachir20/corevital-dashboard/public/demo",
        help="Output directory for trace JSON files",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Max new tokens per run (default: 128)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument(
        "--conda-env",
        type=str,
        default="llm_hm",
        help="Conda environment to run corevital in (default: llm_hm). Use empty string to use current Python.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        metavar="SECONDS",
        help="Timeout per model run in seconds (default: 1800 = 30 min). Increase if load+inference is slow.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not delete existing trace_*.json files in output dir",
    )
    args = parser.parse_args()

    out_dir = Path(args.out).expanduser().resolve()

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        if not args.keep_existing:
            removed = _clean_output_dir(out_dir)
            if removed:
                print(f"Removed {removed} existing trace file(s) from {out_dir}")

    for i, (model_id, prompt) in enumerate(DEMO_RUNS, 1):
        run_args = [
            "--model",
            model_id,
            "--prompt",
            prompt,
            "--max_new_tokens",
            str(args.max_new_tokens),
            "--device",
            "cuda",
            "--quantize-4",
            "--capture",
            "full",
            "--perf",
            "strict",
            "--sink",
            "local",
            "--out",
            str(out_dir),
        ]
        if args.conda_env:
            cmd = [
                "conda",
                "run",
                "-n",
                args.conda_env,
                "--no-capture-output",
                "python",
                "-m",
                "CoreVital.cli",
                "run",
            ] + run_args
        else:
            cmd = [sys.executable, "-m", "CoreVital.cli", "run"] + run_args
        if args.dry_run:
            print(f"# Run {i}/{len(DEMO_RUNS)}: {model_id}")
            print(" ".join(cmd))
            continue
        print(f"Run {i}/{len(DEMO_RUNS)}: {model_id} ...")
        env = os.environ.copy()
        if SRC.exists():
            env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
        if result.returncode != 0:
            print(result.stderr or result.stdout, file=sys.stderr)
            return result.returncode

    if args.dry_run:
        print(f"# Final: traces written to {out_dir}")
    else:
        print(f"Wrote demo traces to {out_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
