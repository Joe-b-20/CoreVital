#!/usr/bin/env python3
"""
CoreVital experiment monitor — one command to see progress, errors, and status.

Usage (from anywhere after setup):
  check_run
  python3 ~/experiment/scripts/check_run.py
"""

import json
import os
import re
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = EXPERIMENT_DIR / "data"
RESULTS_DIR = EXPERIMENT_DIR / "results"
LOGS_DIR = EXPERIMENT_DIR / "logs"
CHECKPOINT_FILE = RESULTS_DIR / "checkpoint.json"
GRADES_FILE = RESULTS_DIR / "grades.jsonl"
TRACES_DIR = EXPERIMENT_DIR / "traces"

MODELS = ["llama", "qwen", "mistral7b", "mixtral"]
DATASETS = ["gsm8k", "humaneval"]


def count_questions():
    out = {}
    for name in DATASETS:
        p = DATA_DIR / f"{name}.jsonl"
        if p.exists():
            with open(p) as f:
                out[name] = sum(1 for _ in f)
        else:
            out[name] = 0
    return out


def total_expected(counts):
    return sum(len(MODELS) * c for c in counts.values())


def load_checkpoint():
    if not CHECKPOINT_FILE.exists():
        return set()
    with open(CHECKPOINT_FILE) as f:
        return set(json.load(f).get("completed", []))


def count_grades_by_cell():
    """Count grade records per (model, dataset)."""
    by_cell = {}
    if not GRADES_FILE.exists():
        return by_cell
    with open(GRADES_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                key = (r.get("model"), r.get("dataset"))
                if key not in by_cell:
                    by_cell[key] = {"runs": 0, "correct": 0, "errors": 0}
                by_cell[key]["runs"] += 1
                if r.get("correct"):
                    by_cell[key]["correct"] += 1
            except Exception:
                pass
    return by_cell


def recent_errors(log_dir, num_lines=50):
    """Last N lines of the most recent log, and any ERROR/OOM/Traceback lines."""
    if not log_dir.exists():
        return [], []
    logs = sorted(log_dir.glob("run_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        return [], []
    path = logs[0]
    with open(path) as f:
        lines = f.readlines()
    last_n = lines[-num_lines:] if len(lines) > num_lines else lines
    err_pattern = re.compile(r"ERROR|OOM|OutOfMemory|Traceback|Exception|Error on")
    err_lines = [l.strip() for l in lines if err_pattern.search(l)]
    return last_n[-15:], err_lines[-20:]  # last 15 of log, last 20 error lines


def is_experiment_running():
    try:
        with open("/proc/self/cgroup") as f:
            # In container we can check for python run_experiment
            pass
    except Exception:
        pass
    # Simple check: look for python process running run_experiment
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p / "cmdline").read_text().replace("\x00", " ")
            if "run_experiment" in cmd and "python" in cmd:
                return True
        except Exception:
            continue
    return False


def main():
    print()
    print("=" * 60)
    print("  CoreVital experiment — status")
    print("=" * 60)

    # Data counts
    counts = count_questions()
    total_prompts_expected = total_expected(counts)
    if total_prompts_expected == 0:
        print("\n  No data found. Run setup first.")
        print(f"  Data dir: {DATA_DIR}")
        sys.exit(0)

    completed = load_checkpoint()
    num_done = len(completed)
    num_left = total_prompts_expected - num_done
    pct = (num_done / total_prompts_expected * 100) if total_prompts_expected else 0

    # Running?
    running = is_experiment_running()
    print(f"\n  Status:    {'RUNNING' if running else 'IDLE'}")
    print(f"  Progress:  {num_done} / {total_prompts_expected} prompt-groups ({pct:.1f}%)")
    print(f"  Remaining: {num_left}")

    # Per-dataset expected
    print("\n  Expected (prompt-groups per model):")
    for ds in DATASETS:
        print(f"    {ds}: {counts[ds]} problems")

    # Grades summary
    by_cell = count_grades_by_cell()
    if by_cell:
        print("\n  Done so far (by model × dataset):")
        for model in MODELS:
            for ds in DATASETS:
                key = (model, ds)
                c = by_cell.get(key, {})
                runs = c.get("runs", 0)
                correct = c.get("correct", 0)
                exp = counts.get(ds, 0) * 10  # 10 runs per prompt
                if exp > 0:
                    pct_cell = runs / exp * 100
                    acc = (correct / runs * 100) if runs else 0
                    print(f"    {model:12} / {ds:8}  runs: {runs:5} / {exp}  ({pct_cell:5.1f}%)  run-acc: {acc:.1f}%")
    else:
        print("\n  No grades yet (experiment not started or no completions).")

    # Traces on disk
    if TRACES_DIR.exists():
        trace_files = list(TRACES_DIR.rglob("*.json"))
        print(f"\n  Traces on disk: {len(trace_files)} files")

    # Recent log and errors
    last_lines, err_lines = recent_errors(LOGS_DIR)
    if err_lines:
        print("\n  --- Recent errors (from latest log) ---")
        for line in err_lines[-10:]:
            print(f"    {line[:100]}")
        print()
    elif last_lines:
        print("\n  --- Last lines of latest log ---")
        for line in last_lines[-5:]:
            print(f"    {line.rstrip()[:100]}")
        print()

    # Checkpoint and grades paths
    print("  Paths:")
    print(f"    Checkpoint: {CHECKPOINT_FILE}  (exists: {CHECKPOINT_FILE.exists()})")
    print(f"    Grades:     {GRADES_FILE}  (exists: {GRADES_FILE.exists()})")
    print(f"    Logs:       {LOGS_DIR}")
    print()
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
