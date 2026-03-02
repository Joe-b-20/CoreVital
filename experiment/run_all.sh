#!/usr/bin/env bash
# ==============================================================================
# CoreVital Validation Experiment - Hands-Off Orchestrator
#
# Runs the entire experiment pipeline end-to-end:
#   1. Dry run (20 questions per cell) — validates prompts, grading, traces
#   2. Full experiment (8,000 traces) with checkpoint/resume
#   3. Performance overhead measurement (50 traces per model)
#   4. Feature extraction → features.parquet
#   5. Statistical analysis → analysis/ directory
#   6. S3 upload (if S3_BUCKET is set)
#
# Prerequisites:
#   - setup.sh has been run successfully
#   - (Local) conda env llm_hm with CoreVital deps — script auto-activates if conda is available
#   - HuggingFace login for gated models (Llama)
#   - (Optional) AWS credentials configured, S3_BUCKET env var set
#
# Usage (on RunPod):
#   # Full pipeline:
#   nohup bash ~/CoreVital/experiment/run_all.sh 2>&1 | tee ~/experiment/logs/pipeline.log &
#
#   # Skip dry run (you already validated):
#   nohup bash ~/CoreVital/experiment/run_all.sh --skip-dry-run 2>&1 | tee ~/experiment/logs/pipeline.log &
#
#   # Dry run only (just validate):
#   bash ~/CoreVital/experiment/run_all.sh --dry-run-only
# ==============================================================================
set -euo pipefail

# Use conda env llm_hm if available (where CoreVital deps live)
if command -v conda &>/dev/null; then
    _conda_base="$(conda info --base 2>/dev/null)"
    if [ -n "$_conda_base" ] && [ -f "$_conda_base/etc/profile.d/conda.sh" ]; then
        set +u
        source "$_conda_base/etc/profile.d/conda.sh"
        conda activate llm_hm 2>/dev/null || true
        set -u
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR=~/experiment
SCRIPTS_DIR="$EXPERIMENT_DIR/scripts"
LOGS_DIR="$EXPERIMENT_DIR/logs"
RESULTS_DIR="$EXPERIMENT_DIR/results"
ANALYSIS_DIR="$EXPERIMENT_DIR/analysis"

# Ensure scripts symlink is accessible
if [ ! -d "$SCRIPTS_DIR" ]; then
    SCRIPTS_DIR="$SCRIPT_DIR/scripts"
fi

mkdir -p "$LOGS_DIR"

# --- Parse arguments -------------------------------------------------------
SKIP_DRY_RUN=false
DRY_RUN_ONLY=false
SKIP_PERF=false

for arg in "$@"; do
    case $arg in
        --skip-dry-run)  SKIP_DRY_RUN=true ;;
        --dry-run-only)  DRY_RUN_ONLY=true ;;
        --skip-perf)     SKIP_PERF=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

fail() {
    log "FATAL: $*"
    exit 1
}

# --- Pre-flight checks ------------------------------------------------------
log "============================================"
log "CoreVital Experiment Pipeline — Pre-flight"
log "============================================"

python3 -c "
import torch, sys
assert torch.cuda.is_available(), 'No CUDA GPU detected'
gpu = torch.cuda.get_device_name(0)
props = torch.cuda.get_device_properties(0)
mem_gb = (getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)) / 1e9
print(f'  GPU: {gpu} ({mem_gb:.0f} GB)')
assert mem_gb > 30, f'Need >=40GB VRAM, got {mem_gb:.0f}GB'
" || fail "GPU check failed"

python3 -c "
from CoreVital.config import Config
from CoreVital.instrumentation.collector import InstrumentationCollector
from CoreVital.reporting.report_builder import ReportBuilder
print('  CoreVital imports: OK')
" || fail "CoreVital import check failed"

for ds in mmlu gsm8k truthfulqa; do
    [ -f "$EXPERIMENT_DIR/data/$ds.jsonl" ] || fail "Missing dataset: $ds.jsonl — run setup.sh"
done
log "  Datasets: OK"

log "Pre-flight passed."
echo ""

# ===========================================================================
# PHASE 1: Dry Run
# ===========================================================================
if [ "$SKIP_DRY_RUN" = false ]; then
    log "============================================"
    log "PHASE 1: Dry Run (20 questions per cell)"
    log "============================================"

    # Clear any prior dry run data
    rm -rf "$EXPERIMENT_DIR"/traces/*
    rm -f "$RESULTS_DIR/grades.jsonl" "$RESULTS_DIR/checkpoint.json"

    python3 "$SCRIPTS_DIR/run_experiment.py" --dry-run \
        2>&1 | tee "$LOGS_DIR/dry_run_$TIMESTAMP.log"

    # Validate dry run results
    log "Validating dry run..."
    python3 << 'VALIDATE_DRY'
import json, sys
from pathlib import Path

grades_file = Path.home() / "experiment" / "results" / "grades.jsonl"
if not grades_file.exists():
    print("FATAL: No grades.jsonl produced by dry run")
    sys.exit(1)

grades = [json.loads(line) for line in open(grades_file)]
if len(grades) == 0:
    print("FATAL: grades.jsonl is empty")
    sys.exit(1)

print(f"  Dry run produced {len(grades)} graded traces")

from collections import Counter
by_cell = Counter()
format_fails = Counter()
for g in grades:
    cell = f"{g['model']}/{g['dataset']}"
    by_cell[cell] += 1
    if g.get("format_failure"):
        format_fails[cell] += 1

issues = []
for cell, count in sorted(by_cell.items()):
    ff = format_fails.get(cell, 0)
    ff_rate = ff / count if count > 0 else 0
    correct = sum(1 for g in grades if f"{g['model']}/{g['dataset']}" == cell and g.get("correct"))
    acc = correct / count if count > 0 else 0
    status = "OK"
    if ff_rate > 0.10:
        status = f"WARNING: {ff_rate:.0%} format failures"
        issues.append(f"{cell}: {status}")
    if acc > 0.95 or acc < 0.05:
        status = f"WARNING: accuracy {acc:.0%} (ceiling/floor)"
        issues.append(f"{cell}: {status}")
    print(f"  {cell}: {count} traces, {acc:.0%} accuracy, {ff:.0%} format fail — {status}")

# Also verify trace files exist
traces_dir = Path.home() / "experiment" / "traces"
trace_count = len(list(traces_dir.rglob("*.json")))
print(f"  Trace files on disk: {trace_count}")

if trace_count == 0:
    print("FATAL: No trace files written")
    sys.exit(1)

# Spot-check a trace
import random
trace_files = list(traces_dir.rglob("*.json"))
sample = random.choice(trace_files)
trace = json.load(open(sample))
required_keys = ["timeline", "extensions", "health_flags", "summary"]
missing = [k for k in required_keys if k not in trace]
if missing:
    print(f"FATAL: Trace {sample.name} missing keys: {missing}")
    sys.exit(1)

risk = trace.get("extensions", {}).get("risk", {}).get("risk_score")
if risk is None:
    print(f"WARNING: Trace {sample.name} has no risk_score in extensions.risk")

print(f"  Spot-check trace {sample.name}: timeline={len(trace.get('timeline',[]))} steps, risk_score={risk}")

if issues:
    print(f"\n  {len(issues)} issue(s) detected (non-fatal):")
    for i in issues:
        print(f"    - {i}")

print("\n  Dry run validation PASSED")
VALIDATE_DRY

    if [ "$DRY_RUN_ONLY" = true ]; then
        log "Dry-run-only mode — stopping here."
        exit 0
    fi

    # Clear dry-run data so the full run starts fresh
    log "Clearing dry-run data before full experiment..."
    rm -rf "$EXPERIMENT_DIR"/traces/*
    rm -f "$RESULTS_DIR/grades.jsonl" "$RESULTS_DIR/checkpoint.json"

    echo ""
fi

# ===========================================================================
# PHASE 2: Full Experiment
# ===========================================================================
log "============================================"
log "PHASE 2: Full Experiment (~8,000 traces)"
log "============================================"

# NOTE: data is NOT cleared here — if --skip-dry-run is set we want
# checkpoint/resume to work for interrupted runs.

python3 "$SCRIPTS_DIR/run_experiment.py" \
    2>&1 | tee "$LOGS_DIR/full_run_$TIMESTAMP.log"

# Quick sanity check
TRACE_COUNT=$(find "$EXPERIMENT_DIR/traces" -name "*.json" | wc -l)
GRADE_COUNT=$(wc -l < "$RESULTS_DIR/grades.jsonl" 2>/dev/null || echo 0)
log "Full run complete: $TRACE_COUNT traces, $GRADE_COUNT grades"

if [ "$TRACE_COUNT" -lt 100 ]; then
    fail "Too few traces ($TRACE_COUNT). Something went wrong."
fi

echo ""

# ===========================================================================
# PHASE 3: Performance Overhead Measurement
# ===========================================================================
if [ "$SKIP_PERF" = false ]; then
    log "============================================"
    log "PHASE 3: Performance Overhead (~50 per model)"
    log "============================================"

    python3 "$SCRIPTS_DIR/run_experiment.py" --perf-only \
        2>&1 | tee "$LOGS_DIR/perf_run_$TIMESTAMP.log"

    echo ""
fi

# ===========================================================================
# PHASE 4: Feature Extraction
# ===========================================================================
log "============================================"
log "PHASE 4: Feature Extraction"
log "============================================"

python3 "$SCRIPTS_DIR/extract_features.py" \
    2>&1 | tee "$LOGS_DIR/extract_$TIMESTAMP.log"

if [ ! -f "$RESULTS_DIR/features.parquet" ]; then
    fail "features.parquet not created"
fi

FEATURE_ROWS=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('$RESULTS_DIR/features.parquet')))")
log "Extracted $FEATURE_ROWS feature rows"

echo ""

# ===========================================================================
# PHASE 5: Statistical Analysis
# ===========================================================================
log "============================================"
log "PHASE 5: Statistical Analysis"
log "============================================"

python3 "$SCRIPTS_DIR/analyze.py" \
    2>&1 | tee "$LOGS_DIR/analyze_$TIMESTAMP.log"

if [ ! -f "$ANALYSIS_DIR/RESULTS_SUMMARY.md" ]; then
    fail "RESULTS_SUMMARY.md not created"
fi

VIZ_COUNT=$(ls "$ANALYSIS_DIR"/*.png 2>/dev/null | wc -l)
log "Analysis complete: $VIZ_COUNT visualizations generated"

echo ""

# ===========================================================================
# PHASE 6: S3 Upload
# ===========================================================================
S3_BUCKET="${S3_BUCKET:-}"

if [ -n "$S3_BUCKET" ]; then
    log "============================================"
    log "PHASE 6: S3 Upload → $S3_BUCKET"
    log "============================================"

    S3_PREFIX="s3://$S3_BUCKET/corevital-validation"

    aws s3 sync "$EXPERIMENT_DIR/traces/"      "$S3_PREFIX/traces/"      --quiet
    aws s3 sync "$EXPERIMENT_DIR/perf_traces/" "$S3_PREFIX/perf_traces/" --quiet
    aws s3 sync "$EXPERIMENT_DIR/results/"     "$S3_PREFIX/results/"     --quiet
    aws s3 sync "$EXPERIMENT_DIR/analysis/"    "$S3_PREFIX/analysis/"    --quiet
    aws s3 sync "$EXPERIMENT_DIR/metadata/"    "$S3_PREFIX/metadata/"    --quiet
    aws s3 sync "$EXPERIMENT_DIR/logs/"        "$S3_PREFIX/logs/"        --quiet

    log "S3 upload complete"
    echo ""
else
    log "S3_BUCKET not set — skipping upload."
    log "To upload later: S3_BUCKET=your-bucket bash ~/CoreVital/experiment/run_all.sh --skip-dry-run --skip-perf"
    echo ""
fi

# ===========================================================================
# DONE
# ===========================================================================
log "============================================"
log "PIPELINE COMPLETE"
log "============================================"
log ""
log "Results:"
log "  Traces:       $EXPERIMENT_DIR/traces/"
log "  Features:     $RESULTS_DIR/features.parquet"
log "  Analysis:     $ANALYSIS_DIR/"
log "  Summary:      $ANALYSIS_DIR/RESULTS_SUMMARY.md"
log "  Perf traces:  $EXPERIMENT_DIR/perf_traces/"
log "  Logs:         $LOGS_DIR/"
log ""
log "Read the summary:"
log "  cat $ANALYSIS_DIR/RESULTS_SUMMARY.md"
