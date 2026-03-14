#!/usr/bin/env python3
"""
CoreVital Validation Experiment — Runner

Pass@k experiment: for each (model × prompt), run k=10 generations under
sampling (5 @ temp 0.7, 5 @ temp 0.8), capture CoreVital traces, grade outputs.

Key difference from v1:
  - Sampling, not greedy
  - Batched GPU work: 10 inferences per prompt before CPU serialization
  - HumanEval sandboxed execution for code grading
  - All runs include prompt analysis

Usage:
    python3 run_experiment.py                   # Full experiment
    python3 run_experiment.py --dry-run          # 5 problems per model per dataset
    python3 run_experiment.py --model llama       # One model only
    python3 run_experiment.py --dataset gsm8k     # One dataset only
"""

import argparse
import gc
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = EXPERIMENT_DIR / "data"
TRACES_DIR = EXPERIMENT_DIR / "traces"
RESULTS_DIR = EXPERIMENT_DIR / "results"
LOGS_DIR = EXPERIMENT_DIR / "logs"
CHECKPOINT_FILE = RESULTS_DIR / "checkpoint.json"
GRADES_FILE = RESULTS_DIR / "grades.jsonl"

MODELS = {
    "llama":     {"hf_id": "meta-llama/Llama-3.1-8B-Instruct", "quantize": None},
    "qwen":      {"hf_id": "Qwen/Qwen2.5-7B-Instruct",        "quantize": None},
    "mistral7b": {"hf_id": "mistralai/Mistral-7B-Instruct-v0.3", "quantize": None},
    "mixtral":   {"hf_id": "mistralai/Mixtral-8x7B-Instruct-v0.1", "quantize": None},
}

DATASETS = ["gsm8k", "humaneval"]

# Pass@k configuration
K_RUNS = 10  # 5 at temp 0.7 + 5 at temp 0.8
TEMP_SCHEDULE = [0.7] * 5 + [0.8] * 5
SEED_SCHEDULE = list(range(10))  # seeds 0-9

# Shared generation parameters
BASE_GEN_PARAMS = {
    "do_sample": True,
    "top_p": 0.95,
    "top_k": 50,
    "max_new_tokens": 768,
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_file: Optional[str] = None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
        force=True,
    )

logger = logging.getLogger("experiment")


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def load_checkpoint() -> set:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return set(json.load(f).get("completed", []))
    return set()


def save_checkpoint(completed: set):
    tmp = CHECKPOINT_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump({"completed": sorted(completed)}, f)
    tmp.rename(CHECKPOINT_FILE)


def append_grades(records: List[dict]):
    with open(GRADES_FILE, "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_questions(dataset_name: str) -> List[dict]:
    path = DATA_DIR / f"{dataset_name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Run setup first.")
    questions = []
    with open(path) as f:
        for line in f:
            questions.append(json.loads(line))
    return questions


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_prompt(question: dict, dataset_name: str) -> str:
    if dataset_name == "gsm8k":
        return (
            f"Solve the following math problem step by step. "
            f"After your solution, write the final numerical answer "
            f"on the last line preceded by \"####\".\n\n"
            f"Problem: {question['question']}"
        )
    elif dataset_name == "humaneval":
        return question["prompt"]  # HumanEval provides the function signature + docstring
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# ---------------------------------------------------------------------------
# Output grading
# ---------------------------------------------------------------------------

def grade_gsm8k(output_text: str, question: dict) -> dict:
    matches = re.findall(r'####\s*([-+]?\d[\d,]*\.?\d*)', output_text)
    if matches:
        extracted = matches[-1].replace(",", "")
    else:
        return {"correct": False, "extracted_answer": None,
                "gold_answer": question["gold_answer"], "format_failure": True}
    try:
        is_correct = abs(float(extracted) - float(question["gold_answer"])) < 1e-6
    except ValueError:
        is_correct = extracted.strip() == question["gold_answer"].strip()
    return {"correct": is_correct, "extracted_answer": extracted,
            "gold_answer": question["gold_answer"], "format_failure": False}


def grade_humaneval(output_text: str, question: dict) -> dict:
    """
    Grade HumanEval by executing the generated code with the test cases.

    The model generates the function body. We prepend the prompt (function signature)
    and append the test cases, then execute in a sandboxed subprocess.
    """
    full_code = question["prompt"] + output_text

    # Append test cases
    test_code = question.get("test", "")
    entry_point = question.get("entry_point", "")
    exec_code = full_code + "\n\n" + test_code
    if entry_point:
        exec_code += f"\n\ncheck({entry_point})\n"

    # Execute in sandboxed subprocess with timeout
    try:
        result = subprocess.run(
            [sys.executable, "-c", exec_code],
            capture_output=True,
            timeout=10,  # 10 second timeout
            text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        is_correct = result.returncode == 0
        error_msg = result.stderr[:500] if result.stderr else None
    except subprocess.TimeoutExpired:
        is_correct = False
        error_msg = "TIMEOUT"
    except Exception as e:
        is_correct = False
        error_msg = str(e)[:500]

    return {
        "correct": is_correct,
        "extracted_answer": "PASS" if is_correct else (error_msg or "FAIL"),
        "gold_answer": "PASS",
        "format_failure": False,
    }


def grade_output(output_text: str, question: dict, dataset_name: str) -> dict:
    if dataset_name == "gsm8k":
        return grade_gsm8k(output_text, question)
    elif dataset_name == "humaneval":
        return grade_humaneval(output_text, question)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# ---------------------------------------------------------------------------
# CoreVital batched runner
# ---------------------------------------------------------------------------

def run_prompt_batch(
    collector,
    builder,
    config,
    prompt: str,
    k: int = K_RUNS,
) -> List[Tuple[Any, Any]]:
    """
    Run k inferences for one prompt, batching GPU work.

    Returns list of (report_object, InstrumentationResults) tuples.
    """
    from CoreVital.instrumentation.performance import PerformanceMonitor

    # Phase 1: GPU work — run all k inferences back to back
    raw_results = []
    for i in range(k):
        config.generation.seed = SEED_SCHEDULE[i]
        config.generation.temperature = TEMP_SCHEDULE[i]

        # Lightweight perf monitoring (summary mode = just timers, no warmup)
        monitor = PerformanceMonitor(mode="summary")
        monitor.mark_run_start()

        result = collector.run(prompt, monitor=monitor)
        raw_results.append((result, monitor))

    # Phase 2: CPU work — build all reports
    reports = []
    for result, monitor in raw_results:
        report = builder.build(result, prompt)

        # Finalize perf data and inject into report
        monitor.mark_run_end()
        perf_summary = monitor.build_summary_dict()
        report.extensions["performance"] = perf_summary

        reports.append((report, result))

    return reports


def save_trace(report, key: str, run_idx: int, output_dir: Path) -> Path:
    """Save a single trace as JSON."""
    from CoreVital.utils.serialization import serialize_report_to_json

    parts = key.split("/")
    trace_dir = output_dir / parts[0] / parts[1]
    trace_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{parts[2]}_run{run_idx:02d}.json"
    filepath = trace_dir / filename

    json_str = serialize_report_to_json(report, indent=None)
    with open(filepath, "w") as f:
        f.write(json_str)

    return filepath


# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------

def make_config(model_spec: dict) -> "Config":
    """Create a CoreVital Config for a given model spec."""
    from CoreVital.config import Config

    config = Config.from_default()
    config.model.hf_id = model_spec["hf_id"]
    config.device.requested = "auto"
    config.capture.capture_mode = "full"
    config.prompt_telemetry.enabled = True

    # Qwen2.5 can produce inf/nan logits in float16 during sampling; use bfloat16
    if "Qwen" in model_spec.get("hf_id", ""):
        config.model.dtype = "bfloat16"

    # Sampling defaults (will be overridden per-run)
    config.generation.do_sample = True
    config.generation.top_p = 0.95
    config.generation.top_k = 50
    config.generation.max_new_tokens = 512

    # Quantization
    if model_spec.get("quantize") == "8bit":
        config.model.load_in_8bit = True

    return config


def clear_gpu():
    """Force clear GPU memory."""
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(
    models_to_run: List[str],
    datasets_to_run: List[str],
    max_per_cell: Optional[int] = None,
):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    completed = load_checkpoint()
    logger.info(f"Checkpoint: {len(completed)} prompt-groups already completed.")

    total_prompts = 0
    total_runs = 0
    total_errors = 0
    cell_stats = {}

    for model_short in models_to_run:
        model_spec = MODELS[model_short]
        logger.info(f"\n{'='*60}")
        logger.info(f"MODEL: {model_short} ({model_spec['hf_id']})")
        if model_spec.get("quantize"):
            logger.info(f"  Quantization: {model_spec['quantize']}")
        logger.info(f"{'='*60}")

        # Create config and collector (loads model once)
        config = make_config(model_spec)
        from CoreVital.instrumentation.collector import InstrumentationCollector
        from CoreVital.reporting.report_builder import ReportBuilder

        collector = InstrumentationCollector(config)
        builder = ReportBuilder(config)

        # Force model load
        logger.info("Loading model...")
        config.generation.seed = 0
        config.generation.temperature = 0.7

        for dataset_name in datasets_to_run:
            questions = load_questions(dataset_name)
            if max_per_cell is not None:
                questions = questions[:max_per_cell]

            cell_key = (model_short, dataset_name)
            cell_stats[cell_key] = {
                "prompts": 0, "runs": 0,
                "correct_runs": 0, "incorrect_runs": 0,
                "format_fails": 0, "errors": 0,
            }

            logger.info(f"\n  Dataset: {dataset_name} ({len(questions)} problems × {K_RUNS} runs = {len(questions) * K_RUNS} traces)")

            pbar = tqdm(questions, desc=f"  {model_short}/{dataset_name}", unit="prompt")

            for question in pbar:
                qid = question["id"]
                prompt_key = f"{model_short}/{dataset_name}/{qid}"

                # Skip if all k runs for this prompt are already done
                if prompt_key in completed:
                    continue

                try:
                    prompt = format_prompt(question, dataset_name)

                    # Batched: run all k inferences, then build reports
                    reports = run_prompt_batch(collector, builder, config, prompt)

                    # Grade and save each run
                    grade_records = []
                    for run_idx, (report, result) in enumerate(reports):
                        output_text = ""
                        if report.generated:
                            output_text = report.generated.output_text or ""

                        grade = grade_output(output_text, question, dataset_name)

                        trace_path = save_trace(report, prompt_key, run_idx, TRACES_DIR)

                        record = {
                            "prompt_key": prompt_key,
                            "model": model_short,
                            "model_id": model_spec["hf_id"],
                            "dataset": dataset_name,
                            "question_id": qid,
                            "run_idx": run_idx,
                            "temperature": TEMP_SCHEDULE[run_idx],
                            "seed": SEED_SCHEDULE[run_idx],
                            "correct": grade["correct"],
                            "extracted_answer": grade["extracted_answer"],
                            "gold_answer": grade["gold_answer"],
                            "format_failure": grade["format_failure"],
                            "output_text": output_text[:500],
                            "generated_tokens": len(result.generated_token_ids),
                            "trace_path": str(trace_path),
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        }
                        grade_records.append(record)

                        if grade["format_failure"]:
                            cell_stats[cell_key]["format_fails"] += 1
                        elif grade["correct"]:
                            cell_stats[cell_key]["correct_runs"] += 1
                        else:
                            cell_stats[cell_key]["incorrect_runs"] += 1

                    # Write all grades for this prompt
                    append_grades(grade_records)
                    cell_stats[cell_key]["runs"] += K_RUNS
                    cell_stats[cell_key]["prompts"] += 1
                    total_prompts += 1
                    total_runs += K_RUNS

                    # Checkpoint per prompt
                    completed.add(prompt_key)
                    save_checkpoint(completed)

                    # Progress
                    c = cell_stats[cell_key]
                    total_cell = c["correct_runs"] + c["incorrect_runs"] + c["format_fails"]
                    acc = c["correct_runs"] / total_cell if total_cell > 0 else 0
                    pass_k = c["prompts"]  # Will compute real pass@k in analysis
                    pbar.set_postfix(acc=f"{acc:.0%}", prompts=c["prompts"])

                except torch.cuda.OutOfMemoryError:
                    logger.error(f"  OOM on {prompt_key}. Clearing GPU.")
                    clear_gpu()
                    cell_stats[cell_key]["errors"] += 1
                    total_errors += 1
                    completed.add(prompt_key)
                    save_checkpoint(completed)

                except Exception as e:
                    logger.error(f"  Error on {prompt_key}: {e}")
                    logger.debug(traceback.format_exc())
                    cell_stats[cell_key]["errors"] += 1
                    total_errors += 1
                    completed.add(prompt_key)
                    save_checkpoint(completed)

            pbar.close()

            # Cell summary
            c = cell_stats[cell_key]
            total_cell = c["correct_runs"] + c["incorrect_runs"] + c["format_fails"]
            acc = c["correct_runs"] / total_cell if total_cell > 0 else 0
            logger.info(
                f"  Cell: {c['prompts']} prompts, {c['runs']} runs, "
                f"{acc:.1%} run accuracy, {c['format_fails']} format fails, "
                f"{c['errors']} errors"
            )

        # Clear model before next
        del collector
        del builder
        clear_gpu()
        logger.info(f"  GPU cleared.")

    save_checkpoint(completed)

    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total prompts: {total_prompts}, Total runs: {total_runs}, Errors: {total_errors}")
    for (model, dataset), c in sorted(cell_stats.items()):
        total_cell = c["correct_runs"] + c["incorrect_runs"] + c["format_fails"]
        acc = c["correct_runs"] / total_cell if total_cell > 0 else 0
        logger.info(f"  {model}/{dataset}: {c['prompts']} prompts, {acc:.1%} accuracy, "
                     f"{c['format_fails']} format fails")

    # Ceiling/floor warnings
    for (model, dataset), c in sorted(cell_stats.items()):
        total_cell = c["correct_runs"] + c["incorrect_runs"] + c["format_fails"]
        if total_cell == 0:
            continue
        acc = c["correct_runs"] / total_cell
        if acc > 0.95:
            logger.warning(f"  ⚠ {model}/{dataset}: {acc:.1%} accuracy — almost all correct under sampling")
        elif acc < 0.05:
            logger.warning(f"  ⚠ {model}/{dataset}: {acc:.1%} accuracy — model can't do this task at all")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CoreVital Validation Runner")
    parser.add_argument("--dry-run", action="store_true",
                        help="5 problems per cell (quick test)")
    parser.add_argument("--model", type=str, choices=list(MODELS.keys()))
    parser.add_argument("--dataset", type=str, choices=DATASETS)
    parser.add_argument("--max-per-cell", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(str(LOGS_DIR / f"run_{timestamp}.log"))

    logger.info(f"Args: {vars(args)}")

    if args.no_resume and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        if GRADES_FILE.exists():
            GRADES_FILE.unlink()

    models_to_run = [args.model] if args.model else list(MODELS.keys())
    datasets_to_run = [args.dataset] if args.dataset else DATASETS

    max_per_cell = args.max_per_cell
    if args.dry_run:
        max_per_cell = 5
        logger.info("DRY RUN: 5 problems per cell")

    run_experiment(models_to_run, datasets_to_run, max_per_cell)


if __name__ == "__main__":
    main()