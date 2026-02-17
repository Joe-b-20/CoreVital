#!/usr/bin/env python3
"""
Run many prompt × config combinations on a single model (e.g. Llama-3.1-8B-Instruct).

For each prompt, runs multiple times with different generation configs (temperature,
top_k, top_p, seed). Results go to the same sink as a normal corevital run (default:
SQLite in runs/corevital.db). Use `corevital compare --db runs/corevital.db` to view.

Default grid: 5 prompts × 6 config variants = 30 runs. Use --limit N for a quick test.

Usage:
  conda activate llm_hm
  python scripts/run_prompt_sweep.py --model meta-llama/Llama-3.1-8B-Instruct --device cuda --quantize-4
  python scripts/run_prompt_sweep.py --model meta-llama/Llama-3.1-8B-Instruct --limit 5 --dry-run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Default prompts: same as docs/test-prompts.md (Llama health-flag targets + clean)
PROMPTS = [
    {
        "id": "repetition",
        "name": "Repetition (target: repetition_loop_detected)",
        "text": (
            "Repeat the word the exactly 25 times: "
            "the the the the the the the the the the the the "
            "the the the the the the the the the the the the the"
        ),
        "max_new_tokens": 60,
    },
    {
        "id": "high_entropy",
        "name": "High entropy",
        "text": "The meaning of life, the universe, and everything is",
        "max_new_tokens": 15,
    },
    {
        "id": "attention_collapse",
        "name": "Yes/no (target: attention_collapse)",
        "text": "Yes or no? Answer with one word:",
        "max_new_tokens": 10,
    },
    {
        "id": "mid_layer",
        "name": "Ambiguous (target: mid_layer_anomaly)",
        "text": "The correct answer to this question is",
        "max_new_tokens": 20,
    },
    {
        "id": "clean",
        "name": "Clean baseline",
        "text": "The capital of France is",
        "max_new_tokens": 10,
    },
]

# Generation config variants: vary temperature, top_k, top_p, seed
CONFIG_VARIANTS = [
    {"name": "default", "temperature": 0.8, "top_k": 50, "top_p": 0.95, "seed": 42},
    {"name": "low_temp", "temperature": 0.2, "top_k": 50, "top_p": 0.95, "seed": 42},
    {"name": "high_temp", "temperature": 1.2, "top_k": 50, "top_p": 0.95, "seed": 42},
    {"name": "low_top_k", "temperature": 0.8, "top_k": 10, "top_p": 0.95, "seed": 42},
    {"name": "low_top_p", "temperature": 0.8, "top_k": 50, "top_p": 0.5, "seed": 42},
    {"name": "seed_123", "temperature": 0.8, "top_k": 50, "top_p": 0.95, "seed": 123},
]


@dataclass
class RunSpec:
    prompt_id: str
    prompt_name: str
    config_name: str
    temperature: float
    top_k: int
    top_p: float
    seed: int
    prompt_text: str
    max_new_tokens: int


def build_specs() -> list[RunSpec]:
    specs: list[RunSpec] = []
    for p in PROMPTS:
        for c in CONFIG_VARIANTS:
            specs.append(
                RunSpec(
                    prompt_id=p["id"],
                    prompt_name=p["name"],
                    config_name=c["name"],
                    temperature=c["temperature"],
                    top_k=c["top_k"],
                    top_p=c["top_p"],
                    seed=c["seed"],
                    prompt_text=p["text"],
                    max_new_tokens=p["max_new_tokens"],
                )
            )
    return specs


def build_argv(
    spec: RunSpec,
    *,
    invoker: str,
    model: str,
    device: str,
    out: str | None,
    sink: str,
    quantize_4: bool,
    quantize_8: bool,
) -> list[str]:
    cmd = invoker.strip().split()
    cmd.extend(["run", "--model", model, "--device", device])
    cmd.extend(["--prompt", spec.prompt_text])
    cmd.extend(["--max_new_tokens", str(spec.max_new_tokens)])
    cmd.extend(["--temperature", str(spec.temperature)])
    cmd.extend(["--top_k", str(spec.top_k)])
    cmd.extend(["--top_p", str(spec.top_p)])
    cmd.extend(["--seed", str(spec.seed)])
    cmd.extend(["--sink", sink])
    if out:
        cmd.extend(["--out", out])
    if quantize_4:
        cmd.append("--quantize-4")
    if quantize_8:
        cmd.append("--quantize-8")
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run prompt × config sweep for CoreVital (e.g. Llama-3.1-8B-Instruct)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Hugging Face model ID",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="cuda",
        help="Device (default: cuda)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory (default: runs). Passed to corevital as --out.",
    )
    parser.add_argument(
        "--sink",
        type=str,
        choices=["sqlite", "local"],
        default="sqlite",
        help="Sink type (default: sqlite)",
    )
    parser.add_argument("--quantize-4", action="store_true", help="Load model in 4-bit")
    parser.add_argument("--quantize-8", action="store_true", help="Load model in 8-bit")
    parser.add_argument(
        "--invoker",
        type=str,
        default="corevital",
        help='CLI invoker (default: "corevital"). Use "python -m CoreVital" if corevital not on PATH.',
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of runs (default: all). Useful for a quick test.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only, do not run",
    )
    args = parser.parse_args()

    specs = build_specs()
    if args.limit is not None:
        specs = specs[: args.limit]

    out_dir = args.out or "runs"
    print(f"Prompt sweep: {len(specs)} runs (model={args.model}, device={args.device})")
    if args.dry_run:
        print("(dry-run: commands only)\n")
    else:
        print()

    failed: list[tuple[RunSpec, str]] = []

    for i, spec in enumerate(specs, 1):
        argv = build_argv(
            spec,
            invoker=args.invoker,
            model=args.model,
            device=args.device,
            out=args.out,
            sink=args.sink,
            quantize_4=args.quantize_4,
            quantize_8=args.quantize_8,
        )
        label = f"[{i}/{len(specs)}] {spec.prompt_id} + {spec.config_name}"
        if args.dry_run:
            print(label)
            print("  " + " ".join(argv))
            print()
            continue
        print(label, end=" ... ", flush=True)
        project_root = Path(__file__).resolve().parent.parent
        env = {**os.environ, "PYTHONPATH": str(project_root / "src")}
        try:
            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=600,
                cwd=project_root,
                env=env,
            )
            if result.returncode != 0:
                failed.append((spec, result.stderr or result.stdout or "unknown"))
                print("FAIL")
            else:
                print("OK")
        except subprocess.TimeoutExpired:
            failed.append((spec, "timeout"))
            print("TIMEOUT")
        except FileNotFoundError as e:
            failed.append((spec, str(e)))
            print("NOT FOUND")
        except Exception as e:
            failed.append((spec, str(e)))
            print("ERROR")

    if failed:
        print(f"\n{len(failed)} run(s) failed:")
        for spec, err in failed[:10]:
            print(f"  {spec.prompt_id} + {spec.config_name}: {err[:80]}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
        return 1

    if not args.dry_run and args.sink == "sqlite":
        db_path = Path(out_dir) / "corevital.db"
        if db_path.exists():
            print(f"\nResults in {db_path}. Compare with:")
            print(f"  corevital compare --db {db_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
