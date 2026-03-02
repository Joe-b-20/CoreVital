#!/usr/bin/env python3
"""
CoreVital Validation Experiment - Runner

Iterates over (model × dataset × question), runs CoreVital instrumented
generation, grades the output, saves the trace, and logs results.

The model is loaded ONCE per model and stays in VRAM for all datasets and
questions.  Between datasets only the generation parameters (max_new_tokens)
change on the shared Config object, which the collector reads at each run().

Usage:
    python3 run_experiment.py                 # Full experiment
    python3 run_experiment.py --dry-run       # 20 questions per model per dataset
    python3 run_experiment.py --perf-only     # 50 traces per model with --perf strict
    python3 run_experiment.py --model llama   # Run only one model
    python3 run_experiment.py --dataset gsm8k # Run only one dataset
    python3 run_experiment.py --resume        # Resume from checkpoint (default: on)
"""

import argparse
import gc
import json
import logging
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXPERIMENT_DIR = Path.home() / "experiment"
DATA_DIR = EXPERIMENT_DIR / "data"
TRACES_DIR = EXPERIMENT_DIR / "traces"
PERF_DIR = EXPERIMENT_DIR / "perf_traces"
RESULTS_DIR = EXPERIMENT_DIR / "results"
LOGS_DIR = EXPERIMENT_DIR / "logs"
CHECKPOINT_FILE = RESULTS_DIR / "checkpoint.json"
GRADES_FILE = RESULTS_DIR / "grades.jsonl"

# Model registry — each entry carries every setting that varies per model so
# the runner never needs ad-hoc if/else branches.
MODELS: Dict[str, Dict[str, Any]] = {
    "phi": {
        "hf_id": "microsoft/Phi-3.5-mini-instruct",
        "trust_remote_code": True,
    },
    "llama": {
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "trust_remote_code": False,
    },
    "mistral7b": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "trust_remote_code": False,
    },
    "nemo": {
        "hf_id": "mistralai/Mistral-Nemo-Instruct-2407",
        "trust_remote_code": False,
    },
}

DATASETS = ["mmlu", "gsm8k", "truthfulqa"]

# Generation parameters (greedy, deterministic)
GEN_PARAMS: Dict[str, Dict[str, Any]] = {
    "mmlu":       {"max_new_tokens": 32,  "do_sample": False, "temperature": 1.0, "top_k": 0, "top_p": 1.0},
    "gsm8k":      {"max_new_tokens": 512, "do_sample": False, "temperature": 1.0, "top_k": 0, "top_p": 1.0},
    "truthfulqa": {"max_new_tokens": 32,  "do_sample": False, "temperature": 1.0, "top_k": 0, "top_p": 1.0},
}

SEED = 42

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_file: Optional[str] = None):
    handlers: list = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )

logger = logging.getLogger("experiment")


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def load_checkpoint() -> set:
    """Load set of completed (model, dataset, question_id) keys."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            data = json.load(f)
        return set(data.get("completed", []))
    return set()


def save_checkpoint(completed: set):
    """Save checkpoint atomically."""
    tmp = CHECKPOINT_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump({"completed": sorted(completed)}, f)
    tmp.rename(CHECKPOINT_FILE)


def append_grade(record: dict):
    """Append a single grade record to the grades JSONL file."""
    with open(GRADES_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_questions(dataset_name: str) -> List[dict]:
    """Load questions from the prepared JSONL file."""
    path = DATA_DIR / f"{dataset_name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}. Run setup.sh first.")
    questions = []
    with open(path) as f:
        for line in f:
            questions.append(json.loads(line))
    return questions


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_prompt(question: dict, dataset_name: str) -> str:
    """Format a question into a prompt string for the model."""

    if dataset_name == "mmlu":
        choices = question["choices"]
        letters = "ABCD"
        choice_text = "\n".join(f"{letters[i]}. {choices[i]}" for i in range(len(choices)))
        return (
            f"The following is a multiple choice question about {question['subject']}.\n\n"
            f"Question: {question['question']}\n"
            f"{choice_text}\n\n"
            f"Answer with just the letter (A, B, C, or D):"
        )

    elif dataset_name == "gsm8k":
        return (
            f"Solve the following math problem step by step. "
            f"After your solution, write the final numerical answer "
            f"on the last line preceded by \"####\".\n\n"
            f"Problem: {question['question']}"
        )

    elif dataset_name == "truthfulqa":
        choices = question["choices"]
        letters = "ABCD"
        choice_text = "\n".join(f"{letters[i]}. {choices[i]}" for i in range(len(choices)))
        return (
            f"Answer the following question by selecting the best option.\n\n"
            f"Question: {question['question']}\n"
            f"{choice_text}\n\n"
            f"Answer with just the letter (A, B, C, or D):"
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# ---------------------------------------------------------------------------
# Output grading
# ---------------------------------------------------------------------------

def extract_mc_answer(text: str) -> Optional[str]:
    """Extract a single letter (A-D) from model output for MC tasks."""
    match = re.search(r'\b([A-D])\b', text)
    if match:
        return match.group(1)
    match = re.search(r'[A-D]', text)
    if match:
        return match.group(0)
    return None


def extract_gsm8k_answer(text: str) -> Optional[str]:
    """Extract the final numerical answer after #### from GSM8K output."""
    matches = re.findall(r'####\s*([-+]?\d[\d,]*\.?\d*)', text)
    if matches:
        answer = matches[-1].replace(",", "")
        return answer
    return None


def grade_output(output_text: str, question: dict, dataset_name: str) -> dict:
    """Grade a model output against the ground truth."""
    if dataset_name in ("mmlu", "truthfulqa"):
        extracted = extract_mc_answer(output_text)
        gold = question["answer_letter"]
        if extracted is None:
            return {
                "correct": False,
                "extracted_answer": None,
                "gold_answer": gold,
                "format_failure": True,
            }
        return {
            "correct": extracted == gold,
            "extracted_answer": extracted,
            "gold_answer": gold,
            "format_failure": False,
        }

    elif dataset_name == "gsm8k":
        extracted = extract_gsm8k_answer(output_text)
        gold = question["gold_answer"]
        if extracted is None:
            return {
                "correct": False,
                "extracted_answer": None,
                "gold_answer": gold,
                "format_failure": True,
            }
        try:
            ext_num = float(extracted)
            gold_num = float(gold)
            is_correct = abs(ext_num - gold_num) < 1e-6
        except ValueError:
            is_correct = extracted.strip() == gold.strip()
        return {
            "correct": is_correct,
            "extracted_answer": extracted,
            "gold_answer": gold,
            "format_failure": False,
        }

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# ---------------------------------------------------------------------------
# CoreVital model lifecycle
# ---------------------------------------------------------------------------

class ModelSession:
    """Owns a single model's Config + InstrumentationCollector.

    The model is loaded into VRAM on the first collector.run() call and
    stays there for every subsequent run().  Between datasets the caller
    updates generation params via update_gen_params() — the collector
    reads from the same Config reference at each run().
    """

    def __init__(self, model_short: str, model_entry: Dict[str, Any]):
        from CoreVital.config import Config
        from CoreVital.instrumentation.collector import InstrumentationCollector

        self.model_short = model_short
        self.model_id = model_entry["hf_id"]

        self.config = Config.from_default()
        self.config.model.hf_id = self.model_id
        self.config.model.trust_remote_code = model_entry.get("trust_remote_code", False)
        self.config.device.requested = "auto"
        self.config.generation.seed = SEED
        self.config.capture.capture_mode = "full"
        self.config.prompt_telemetry.enabled = True
        self.config.summaries.logits.topk = 10

        self.collector = InstrumentationCollector(self.config)
        self._loaded = False

    def warmup(self, prompt: str = "Hello"):
        """Force the model into VRAM with a throwaway run."""
        if self._loaded:
            return
        logger.info(f"Loading {self.model_id} into VRAM (first run)...")
        t0 = time.time()
        self.collector.run(prompt)
        elapsed = time.time() - t0
        self._loaded = True
        logger.info(f"Model loaded and warmed up in {elapsed:.1f}s")

    def update_gen_params(self, gen_params: Dict[str, Any]):
        """Update generation parameters on the shared config for the next dataset."""
        self.config.generation.max_new_tokens = gen_params["max_new_tokens"]
        self.config.generation.do_sample = gen_params.get("do_sample", False)
        self.config.generation.temperature = gen_params.get("temperature", 1.0)
        self.config.generation.top_k = gen_params.get("top_k", 0)
        self.config.generation.top_p = gen_params.get("top_p", 1.0)

    def run_trace(self, prompt: str) -> Tuple[Any, str]:
        """Run a single CoreVital trace and build the report.

        Returns (report, output_text).
        """
        from CoreVital.reporting.report_builder import ReportBuilder

        results = self.collector.run(prompt)
        builder = ReportBuilder(self.config)
        report = builder.build(results, prompt)

        output_text = ""
        if report.generated:
            output_text = report.generated.output_text or ""

        return report, output_text

    def close(self):
        """Release GPU memory held by this model."""
        if self.collector.model_bundle is not None:
            del self.collector.model_bundle.model
            self.collector.model_bundle = None
        del self.collector
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Released GPU memory for {self.model_id}")


# ---------------------------------------------------------------------------
# Trace persistence
# ---------------------------------------------------------------------------

def save_trace(report: Any, key: str, output_dir: Path) -> Path:
    """Save a report as a JSON file."""
    from CoreVital.utils.serialization import serialize_report_to_json

    parts = key.split("/")
    trace_dir = output_dir / parts[0] / parts[1]
    trace_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{parts[2]}.json"
    filepath = trace_dir / filename

    json_str = serialize_report_to_json(report, indent=None)
    with open(filepath, "w") as f:
        f.write(json_str)

    return filepath


# ---------------------------------------------------------------------------
# Performance overhead subset
# ---------------------------------------------------------------------------

def run_perf_subset(models_to_run: List[str], datasets_to_run: List[str]):
    """Run a small subset of traces with --perf strict for overhead measurement.

    Uses its own collector/config with perf enabled to avoid contaminating
    the main signal data.
    """
    from CoreVital.config import Config
    from CoreVital.instrumentation.collector import InstrumentationCollector
    from CoreVital.instrumentation.performance import PerformanceMonitor
    from CoreVital.reporting.report_builder import ReportBuilder
    from CoreVital.utils.serialization import serialize_report_to_json

    logger.info("=== PERFORMANCE OVERHEAD MEASUREMENT ===")
    per_cell = 17  # ~50 total across 3 datasets

    for model_short in models_to_run:
        entry = MODELS[model_short]
        model_id = entry["hf_id"]
        logger.info(f"\n  Perf run: {model_short} ({model_id})")

        config = Config.from_default()
        config.model.hf_id = model_id
        config.model.trust_remote_code = entry.get("trust_remote_code", False)
        config.device.requested = "auto"
        config.generation.seed = SEED
        config.capture.capture_mode = "full"
        config.prompt_telemetry.enabled = True
        config.performance.mode = "strict"

        collector = InstrumentationCollector(config)

        for dataset_name in datasets_to_run:
            questions = load_questions(dataset_name)[:per_cell]
            gen_params = GEN_PARAMS[dataset_name]

            config.generation.max_new_tokens = gen_params["max_new_tokens"]
            config.generation.do_sample = gen_params.get("do_sample", False)
            config.generation.temperature = gen_params.get("temperature", 1.0)
            config.generation.top_k = gen_params.get("top_k", 0)
            config.generation.top_p = gen_params.get("top_p", 1.0)

            for question in tqdm(questions, desc=f"  perf/{model_short}/{dataset_name}", unit="q"):
                prompt = format_prompt(question, dataset_name)
                try:
                    monitor = PerformanceMonitor(mode="strict")
                    monitor.mark_run_start()

                    results = collector.run(prompt, monitor=monitor)
                    builder = ReportBuilder(config)
                    report = builder.build(results, prompt)

                    monitor.mark_run_end()
                    perf_summary = monitor.build_summary_dict()
                    perf_summary["detailed_breakdown"] = monitor.build_detailed_breakdown()
                    report.extensions["performance"] = perf_summary

                    key = f"{model_short}/{dataset_name}/{question['id']}"
                    trace_dir = PERF_DIR / model_short / dataset_name
                    trace_dir.mkdir(parents=True, exist_ok=True)
                    json_str = serialize_report_to_json(report, indent=None)
                    with open(trace_dir / f"{question['id']}.json", "w") as f:
                        f.write(json_str)

                except Exception as e:
                    logger.error(f"  Perf error on {question['id']}: {e}")

        # Free model
        if collector.model_bundle is not None:
            del collector.model_bundle.model
            collector.model_bundle = None
        del collector
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("=== PERF MEASUREMENT COMPLETE ===")


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(
    models_to_run: List[str],
    datasets_to_run: List[str],
    max_per_cell: Optional[int] = None,
):
    """Main experiment loop.

    For each model, a single ModelSession is created.  The model loads into
    VRAM on the first trace and stays there for all datasets and questions.
    Between datasets, only generation parameters are updated.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    completed = load_checkpoint()
    logger.info(f"Checkpoint loaded: {len(completed)} traces already completed.")

    total_traces = 0
    total_correct = 0
    total_errors = 0
    cell_stats: Dict[Tuple[str, str], Dict[str, int]] = {}

    for model_short in models_to_run:
        entry = MODELS[model_short]
        logger.info(f"\n{'='*60}")
        logger.info(f"MODEL: {model_short} ({entry['hf_id']})")
        logger.info(f"{'='*60}")

        session = ModelSession(model_short, entry)

        # Warm up: force model into VRAM before timing any real traces.
        session.warmup()

        for dataset_name in datasets_to_run:
            questions = load_questions(dataset_name)
            gen_params = GEN_PARAMS[dataset_name]

            if max_per_cell is not None:
                questions = questions[:max_per_cell]

            # Point the session at this dataset's generation params
            session.update_gen_params(gen_params)

            cell_key = (model_short, dataset_name)
            cell_stats[cell_key] = {"correct": 0, "incorrect": 0, "format_fail": 0, "error": 0}

            logger.info(f"\n  Dataset: {dataset_name} ({len(questions)} questions)")
            logger.info(f"  Gen params: max_tokens={gen_params['max_new_tokens']}, greedy")

            pbar = tqdm(questions, desc=f"  {model_short}/{dataset_name}", unit="q")

            for question in pbar:
                qid = question["id"]
                key = f"{model_short}/{dataset_name}/{qid}"

                if key in completed:
                    continue

                try:
                    prompt = format_prompt(question, dataset_name)
                    report, output_text = session.run_trace(prompt)

                    grade_result = grade_output(output_text, question, dataset_name)

                    trace_path = save_trace(report, key, TRACES_DIR)

                    grade_record = {
                        "key": key,
                        "model": model_short,
                        "model_id": entry["hf_id"],
                        "dataset": dataset_name,
                        "question_id": qid,
                        "correct": grade_result["correct"],
                        "extracted_answer": grade_result["extracted_answer"],
                        "gold_answer": grade_result["gold_answer"],
                        "format_failure": grade_result["format_failure"],
                        "output_text": output_text[:500],
                        "generated_tokens": report.summary.generated_tokens if report.summary else 0,
                        "trace_path": str(trace_path),
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                    append_grade(grade_record)

                    if grade_result["format_failure"]:
                        cell_stats[cell_key]["format_fail"] += 1
                    elif grade_result["correct"]:
                        cell_stats[cell_key]["correct"] += 1
                        total_correct += 1
                    else:
                        cell_stats[cell_key]["incorrect"] += 1

                    total_traces += 1
                    completed.add(key)

                    if total_traces % 10 == 0:
                        save_checkpoint(completed)

                    c = cell_stats[cell_key]
                    total_cell = c["correct"] + c["incorrect"] + c["format_fail"]
                    acc = c["correct"] / total_cell if total_cell > 0 else 0
                    pbar.set_postfix(acc=f"{acc:.1%}", done=total_cell)

                except torch.cuda.OutOfMemoryError:
                    logger.error(f"  OOM on {key}. Clearing cache and skipping.")
                    torch.cuda.empty_cache()
                    gc.collect()
                    cell_stats[cell_key]["error"] += 1
                    total_errors += 1
                    completed.add(key)

                except Exception as e:
                    logger.error(f"  Error on {key}: {e}")
                    logger.debug(traceback.format_exc())
                    cell_stats[cell_key]["error"] += 1
                    total_errors += 1
                    completed.add(key)

            pbar.close()

            c = cell_stats[cell_key]
            total_cell = c["correct"] + c["incorrect"] + c["format_fail"]
            acc = c["correct"] / total_cell if total_cell > 0 else 0
            logger.info(
                f"  Cell summary: {c['correct']} correct, {c['incorrect']} incorrect, "
                f"{c['format_fail']} format failures, {c['error']} errors "
                f"(accuracy: {acc:.1%})"
            )

            save_checkpoint(completed)

        # Release model from GPU before loading the next one
        session.close()

    save_checkpoint(completed)

    # --- Overall summary -------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total traces: {total_traces}")
    logger.info(f"Total errors: {total_errors}")
    logger.info(f"\nPer-cell breakdown:")
    for (model, dataset), c in sorted(cell_stats.items()):
        total_cell = c["correct"] + c["incorrect"] + c["format_fail"]
        acc = c["correct"] / total_cell if total_cell > 0 else 0
        logger.info(
            f"  {model}/{dataset}: {total_cell} total, "
            f"{acc:.1%} accuracy, {c['format_fail']} format fails, {c['error']} errors"
        )

    logger.info("\n  Accuracy warnings:")
    for (model, dataset), c in sorted(cell_stats.items()):
        total_cell = c["correct"] + c["incorrect"] + c["format_fail"]
        if total_cell == 0:
            continue
        acc = c["correct"] / total_cell
        if acc > 0.90:
            logger.warning(
                f"    WARNING {model}/{dataset}: accuracy {acc:.1%} — too few incorrect samples."
            )
        elif acc < 0.10:
            logger.warning(
                f"    WARNING {model}/{dataset}: accuracy {acc:.1%} — too few correct samples."
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CoreVital Validation Experiment Runner")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run 20 questions per model per dataset (quick test)")
    parser.add_argument("--perf-only", action="store_true",
                        help="Run 50 traces per model with --perf strict (overhead measurement)")
    parser.add_argument("--model", type=str, choices=list(MODELS.keys()),
                        help="Run only this model (default: all)")
    parser.add_argument("--dataset", type=str, choices=DATASETS,
                        help="Run only this dataset (default: all)")
    parser.add_argument("--max-per-cell", type=int, default=None,
                        help="Max questions per (model, dataset) cell")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh (ignore checkpoint)")
    args = parser.parse_args()

    # Setup
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"run_{timestamp}.log"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(str(log_file))

    logger.info(f"Log file: {log_file}")
    logger.info(f"Args: {vars(args)}")

    if args.no_resume and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info("Checkpoint cleared.")

    models_to_run = [args.model] if args.model else list(MODELS.keys())
    datasets_to_run = [args.dataset] if args.dataset else DATASETS

    max_per_cell = args.max_per_cell
    if args.dry_run:
        max_per_cell = 20
        logger.info("DRY RUN MODE: 20 questions per cell")

    if args.perf_only:
        run_perf_subset(models_to_run, datasets_to_run)
    else:
        run_experiment(
            models_to_run=models_to_run,
            datasets_to_run=datasets_to_run,
            max_per_cell=max_per_cell,
        )


if __name__ == "__main__":
    main()
