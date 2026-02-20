#!/usr/bin/env python3
"""Generate docs/demo/corevital_demo.db with curated runs for the dashboard.

Runs CoreVital on 4 models (Llama 3.1, Mistral, Llama 3.2, FLAN-T5) with CUDA,
4-bit quantization, full capture, and strict performance. Replaces docs/demo/corevital_demo.db.

Requirements: conda env llm_hm (with CUDA, bitsandbytes, CoreVital deps), and HuggingFace
model access. Run from repo root:

  conda activate llm_hm
  pip install -e .   # if not already
  python scripts/gen_demo_db.py

The script uses conda env llm_hm when invoking corevital (via conda run -n llm_hm).

Optional:
  --dry-run       Print commands only
  --out DIR       Use DIR for DB (default: docs/demo); final DB is <out>/corevital_demo.db
  --conda-env ENV Conda environment name (default: llm_hm). Use "" to use current Python.
  --timeout SEC   Timeout per model run in seconds (default: 1800). Use 3600 for slow GPUs.
"""

import argparse
import os
import shutil
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate docs/demo/corevital_demo.db with curated CoreVital runs."
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(REPO_ROOT / "docs" / "demo"),
        help="Output directory; DB will be <out>/corevital.db then copied to <out>/corevital_demo.db",
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
    args = parser.parse_args()

    out_dir = Path(args.out).resolve()
    db_path = out_dir / "corevital.db"
    demo_db_path = out_dir / "corevital_demo.db"

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        # Start fresh: remove existing corevital.db so we don't append to old runs
        if db_path.exists():
            db_path.unlink()
        # Backup existing demo DB if present
        if demo_db_path.exists():
            backup = demo_db_path.with_suffix(".db.bak")
            shutil.copy2(demo_db_path, backup)
            print(f"Backed up existing demo DB to {backup}")

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
            "sqlite",
            "--out",
            str(out_dir),
        ]
        if args.conda_env:
            cmd = ["conda", "run", "-n", args.conda_env, "--no-capture-output", "python", "-m", "CoreVital.cli", "run"] + run_args
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

    if not args.dry_run and db_path.exists():
        shutil.copy2(db_path, demo_db_path)
        print(f"Wrote {demo_db_path} with {len(DEMO_RUNS)} runs.")
    elif args.dry_run:
        print(f"# Final: cp {db_path} {demo_db_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
