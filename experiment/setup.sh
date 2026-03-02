#!/usr/bin/env bash
# ==============================================================================
# CoreVital Validation Experiment - RunPod Setup
#
# Run this ONCE after provisioning your RunPod instance.
# Prerequisites: RunPod A100 80GB with PyTorch template
#
# Usage (on RunPod):
#   git clone -b experiment/validation https://github.com/Joe-b-20/CoreVital.git ~/CoreVital
#   cd ~/CoreVital
#   chmod +x experiment/setup.sh
#   ./experiment/setup.sh
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

echo "=========================================="
echo "CoreVital Validation Experiment Setup"
echo "=========================================="

# Detect repo root (this script lives in <repo>/experiment/setup.sh)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EXPERIMENT_DIR=~/experiment

echo "Repo:       $REPO_DIR"
echo "Experiment: $EXPERIMENT_DIR"

# ---------------------------------------------------------------------------
# 1. System info (recorded for reproducibility)
# ---------------------------------------------------------------------------
echo ""
echo "[1/8] Recording system info..."
mkdir -p "$EXPERIMENT_DIR/metadata"

cat > "$EXPERIMENT_DIR/metadata/system_info.json" << SYSEOF
{
  "setup_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "hostname": "$(hostname)",
  "gpu": "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')",
  "gpu_memory_mb": "$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || echo 'unknown')",
  "cuda_version": "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo 'unknown')",
  "python_version": "$(python3 --version 2>&1)",
  "torch_version": "will be recorded after install"
}
SYSEOF
cat "$EXPERIMENT_DIR/metadata/system_info.json"
echo ""

# ---------------------------------------------------------------------------
# 2. Install CoreVital from this repo
# ---------------------------------------------------------------------------
echo "[2/8] Installing CoreVital from repo..."
cd "$REPO_DIR"

# Pin the commit SHA for reproducibility
COREVITAL_SHA=$(git rev-parse HEAD)
echo "CoreVital commit: $COREVITAL_SHA"
echo "CoreVital branch: $(git branch --show-current)"
echo "{\"corevital_commit\": \"$COREVITAL_SHA\", \"branch\": \"$(git branch --show-current)\"}" \
    > "$EXPERIMENT_DIR/metadata/corevital_version.json"

# Install CoreVital in editable mode (so we can import it)
pip install -e ".[all]" --quiet

# ---------------------------------------------------------------------------
# 3. Install experiment dependencies
# ---------------------------------------------------------------------------
echo ""
echo "[3/8] Installing experiment dependencies..."

pip install -r "$SCRIPT_DIR/requirements-experiment.txt" --quiet

# Record exact versions
pip freeze > "$EXPERIMENT_DIR/metadata/pip_freeze.txt"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# ---------------------------------------------------------------------------
# 4. Download datasets
# ---------------------------------------------------------------------------
echo ""
echo "[4/8] Downloading datasets..."

EXPERIMENT_DIR_PY="$EXPERIMENT_DIR" python3 << 'PYEOF'
import json
import os
from pathlib import Path
from datasets import load_dataset

data_dir = Path(os.environ["EXPERIMENT_DIR_PY"]) / "data"
data_dir.mkdir(parents=True, exist_ok=True)

# --- MMLU ---
print("  Downloading MMLU (test split)...")
mmlu = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
print(f"  MMLU total: {len(mmlu)} questions across {len(set(mmlu['subject']))} subjects")

# Stratified sample: ~1000 questions across all subjects
import random
random.seed(42)
by_subject = {}
for i, row in enumerate(mmlu):
    subj = row["subject"]
    if subj not in by_subject:
        by_subject[subj] = []
    by_subject[subj].append(i)

n_subjects = len(by_subject)
per_subject = max(1, 1000 // n_subjects)  # ~17 per subject
sampled_indices = []
for subj, indices in sorted(by_subject.items()):
    k = min(per_subject, len(indices))
    sampled_indices.extend(random.sample(indices, k))

# If we're under 1000, top up from remaining
if len(sampled_indices) < 1000:
    remaining = set(range(len(mmlu))) - set(sampled_indices)
    extra = random.sample(list(remaining), min(1000 - len(sampled_indices), len(remaining)))
    sampled_indices.extend(extra)

random.shuffle(sampled_indices)
mmlu_sample = mmlu.select(sampled_indices[:1000])

# Save as JSONL
out = data_dir / "mmlu.jsonl"
with open(out, "w") as f:
    for i, row in enumerate(mmlu_sample):
        item = {
            "id": f"mmlu_{i:04d}",
            "dataset": "mmlu",
            "subject": row["subject"],
            "question": row["question"],
            "choices": row["choices"],
            "answer_index": row["answer"],  # 0-3 index
            "answer_letter": "ABCD"[row["answer"]],
        }
        f.write(json.dumps(item) + "\n")
print(f"  Saved {i+1} MMLU questions to {out}")

# --- GSM8K ---
print("  Downloading GSM8K (test split)...")
gsm = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)
print(f"  GSM8K total: {len(gsm)} questions")

indices = random.sample(range(len(gsm)), min(500, len(gsm)))
gsm_sample = gsm.select(indices)

out = data_dir / "gsm8k.jsonl"
with open(out, "w") as f:
    for i, row in enumerate(gsm_sample):
        # Extract the gold answer from the "#### <number>" format
        answer_text = row["answer"]
        gold = answer_text.split("####")[-1].strip().replace(",", "")
        item = {
            "id": f"gsm8k_{i:04d}",
            "dataset": "gsm8k",
            "question": row["question"],
            "gold_answer": gold,
            "full_solution": answer_text,
        }
        f.write(json.dumps(item) + "\n")
print(f"  Saved {i+1} GSM8K questions to {out}")

# --- TruthfulQA ---
print("  Downloading TruthfulQA (validation split, MC format)...")
tqa = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation",
                   trust_remote_code=True)
print(f"  TruthfulQA total: {len(tqa)} questions")

indices = random.sample(range(len(tqa)), min(500, len(tqa)))
tqa_sample = tqa.select(indices)

out = data_dir / "truthfulqa.jsonl"
with open(out, "w") as f:
    for i, row in enumerate(tqa_sample):
        # mc1_targets: {"choices": [...], "labels": [0,1,0,...]}
        mc1 = row["mc1_targets"]
        choices = mc1["choices"]
        labels = mc1["labels"]
        correct_idx = labels.index(1) if 1 in labels else 0
        # Limit to first 4 choices (A-D) for consistency
        choices_limited = choices[:4]
        if correct_idx >= 4:
            # If correct answer is beyond D, skip this question
            continue
        item = {
            "id": f"tqa_{i:04d}",
            "dataset": "truthfulqa",
            "question": row["question"],
            "choices": choices_limited,
            "answer_index": correct_idx,
            "answer_letter": "ABCD"[correct_idx],
        }
        f.write(json.dumps(item) + "\n")
print(f"  Saved TruthfulQA questions to {out}")

# --- Summary ---
print("\n  Dataset files:")
for p in sorted(data_dir.glob("*.jsonl")):
    lines = sum(1 for _ in open(p))
    print(f"    {p.name}: {lines} questions")

PYEOF

# ---------------------------------------------------------------------------
# 5. Download models (this takes a while)
# ---------------------------------------------------------------------------
echo ""
echo "[5/8] Downloading models (this may take 20-30 minutes)..."

EXPERIMENT_DIR_PY="$EXPERIMENT_DIR" python3 << 'PYEOF'
import json
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# Must match the MODELS registry in run_experiment.py
models = {
    "microsoft/Phi-3.5-mini-instruct":       {"trust_remote_code": True},
    "meta-llama/Llama-3.1-8B-Instruct":      {"trust_remote_code": False},
    "mistralai/Mistral-7B-Instruct-v0.3":    {"trust_remote_code": False},
    "mistralai/Mistral-Nemo-Instruct-2407":   {"trust_remote_code": False},
}

meta_dir = Path(os.environ["EXPERIMENT_DIR_PY"]) / "metadata"
model_shas = {}

for model_id, opts in models.items():
    trc = opts["trust_remote_code"]
    print(f"\n  Downloading {model_id} (trust_remote_code={trc})...")
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trc)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="cpu",
            trust_remote_code=trc,
        )
        sha = getattr(model.config, '_commit_hash', 'unknown')
        model_shas[model_id] = sha
        print(f"    OK Downloaded. Commit: {sha}")
        del model
        del tok
        import gc; gc.collect()
    except Exception as e:
        print(f"    FAILED: {e}")
        print(f"    If this is a gated model, run: huggingface-cli login")
        model_shas[model_id] = f"FAILED: {e}"

with open(meta_dir / "model_versions.json", "w") as f:
    json.dump(model_shas, f, indent=2)

print("\n  Model versions saved to metadata/model_versions.json")
PYEOF

# ---------------------------------------------------------------------------
# 6. Create directory structure and symlink scripts
# ---------------------------------------------------------------------------
echo ""
echo "[6/8] Creating directory structure..."

mkdir -p "$EXPERIMENT_DIR"/{traces,results,analysis,logs,perf_traces}
mkdir -p "$EXPERIMENT_DIR"/traces/{phi,llama,mistral7b,nemo}

# Symlink scripts so they're accessible from ~/experiment/scripts/
ln -sfn "$SCRIPT_DIR/scripts" "$EXPERIMENT_DIR/scripts"

echo "  ~/experiment/"
echo "  ├── scripts/ → $SCRIPT_DIR/scripts/ (symlink)"
echo "  ├── data/               (datasets)"
echo "  ├── traces/             (CoreVital JSON traces)"
echo "  ├── perf_traces/        (overhead measurement traces)"
echo "  ├── results/            (features.parquet, labels, etc.)"
echo "  ├── analysis/           (notebooks, figures)"
echo "  ├── logs/               (run logs)"
echo "  └── metadata/           (versions, system info)"

# ---------------------------------------------------------------------------
# 7. Verify scripts are accessible
# ---------------------------------------------------------------------------
echo ""
echo "[7/8] Verifying scripts..."
for script in run_experiment.py extract_features.py analyze.py; do
    if [ -f "$EXPERIMENT_DIR/scripts/$script" ]; then
        echo "  ✓ $script"
    else
        echo "  ✗ $script NOT FOUND — check symlink"
    fi
done

# ---------------------------------------------------------------------------
# 8. Verify everything works
# ---------------------------------------------------------------------------
echo ""
echo "[8/8] Verification..."

EXPERIMENT_DIR_PY="$EXPERIMENT_DIR" python3 << 'PYEOF'
import os
import sys

checks = []

# Check CoreVital imports
try:
    from CoreVital.monitor import CoreVitalMonitor
    from CoreVital.config import Config
    from CoreVital.instrumentation.collector import InstrumentationCollector
    from CoreVital.reporting.report_builder import ReportBuilder
    checks.append(("CoreVital imports", True))
except Exception as e:
    checks.append(("CoreVital imports", f"FAIL: {e}"))

# Check analysis imports
try:
    import pandas, sklearn, scipy, statsmodels, pingouin
    import matplotlib, seaborn, pyarrow
    checks.append(("Analysis imports", True))
except Exception as e:
    checks.append(("Analysis imports", f"FAIL: {e}"))

# Check datasets
try:
    from pathlib import Path
    data_dir = Path(os.environ["EXPERIMENT_DIR_PY"]) / "data"
    for name in ["mmlu.jsonl", "gsm8k.jsonl", "truthfulqa.jsonl"]:
        assert (data_dir / name).exists(), f"Missing {name}"
    checks.append(("Datasets present", True))
except Exception as e:
    checks.append(("Datasets present", f"FAIL: {e}"))

# Check GPU
try:
    import torch
    assert torch.cuda.is_available(), "No CUDA"
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    checks.append((f"GPU: {gpu_name} ({gpu_mem:.0f}GB)", True))
except Exception as e:
    checks.append(("GPU", f"FAIL: {e}"))

print("\n  Verification Results:")
all_ok = True
for name, status in checks:
    icon = "✓" if status is True else "✗"
    detail = "" if status is True else f" — {status}"
    print(f"    {icon} {name}{detail}")
    if status is not True:
        all_ok = False

if all_ok:
    print("\n  ✓ All checks passed. Ready to run experiments.")
else:
    print("\n  ✗ Some checks failed. Fix issues before proceeding.")
    sys.exit(1)
PYEOF

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. If using gated models (Llama), run: huggingface-cli login"
echo ""
echo "  HANDS-OFF (recommended) — runs everything end-to-end:"
echo "    S3_BUCKET=your-bucket nohup bash ~/CoreVital/experiment/run_all.sh 2>&1 | tee ~/experiment/logs/pipeline.log &"
echo ""
echo "  MANUAL — run each step yourself:"
echo "    cd ~/experiment"
echo "    python3 scripts/run_experiment.py --dry-run        # Validate prompts/grading"
echo "    python3 scripts/run_experiment.py                  # Full 8,000-trace run"
echo "    python3 scripts/run_experiment.py --perf-only      # Overhead measurement"
echo "    python3 scripts/extract_features.py                # Build features.parquet"
echo "    python3 scripts/analyze.py                         # H1/H2/H2b/H3 + plots"