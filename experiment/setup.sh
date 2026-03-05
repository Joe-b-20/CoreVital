#!/usr/bin/env bash
# ==============================================================================
# CoreVital Validation Experiment - RunPod Setup
#
# Usage (on RunPod):
#   git clone -b experiment/validation https://github.com/Joe-b-20/CoreVital.git ~/CoreVital
#   cd ~/CoreVital
#   chmod +x experiment/setup.sh
#   ./experiment/setup.sh
# ==============================================================================
set -euo pipefail

echo "=========================================="
echo "CoreVital Validation Experiment Setup"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EXPERIMENT_DIR=~/experiment

echo "Repo:       $REPO_DIR"
echo "Experiment: $EXPERIMENT_DIR"

# ---------------------------------------------------------------------------
# 0. Credentials: HuggingFace + AWS (you only enter token + keys)
#    Optional: set HF_TOKEN, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY to skip prompts.
# ---------------------------------------------------------------------------
echo ""
echo "[0/10] Credentials and auth (HuggingFace + AWS)..."

pip install -q huggingface_hub "awscli" 2>/dev/null || true

# HuggingFace token (use env var from Configure Deployment to skip prompt)
if [ -z "${HF_TOKEN:-}" ]; then
  echo "  Enter your HuggingFace token (get one at https://huggingface.co/settings/tokens):"
  read -r HF_TOKEN
else
  echo "  Using existing HF_TOKEN from environment."
fi
if [ -n "${HF_TOKEN:-}" ]; then
  huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || true
  echo "  ✓ HuggingFace login done."
else
  echo "  (Skipped HF login; set HF_TOKEN or run: huggingface-cli login)"
fi

# AWS: access key + secret only; bucket/region/output are fixed
AWS_BUCKET="${AWS_BUCKET:-corevital-validation}"
AWS_REGION="${AWS_REGION:-us-east-1}"

if [ -z "${AWS_ACCESS_KEY_ID:-}" ]; then
  echo "  Enter AWS Access Key ID:"
  read -r AWS_ACCESS_KEY_ID
else
  echo "  Using existing AWS_ACCESS_KEY_ID from environment."
fi
if [ -z "${AWS_SECRET_ACCESS_KEY:-}" ]; then
  echo "  Enter AWS Secret Access Key:"
  read -s -r AWS_SECRET_ACCESS_KEY
  echo ""
else
  echo "  Using existing AWS_SECRET_ACCESS_KEY from environment."
fi

mkdir -p ~/.aws
if [ -n "${AWS_ACCESS_KEY_ID:-}" ] && [ -n "${AWS_SECRET_ACCESS_KEY:-}" ]; then
  cat > ~/.aws/credentials << AWSCRED
[default]
aws_access_key_id = ${AWS_ACCESS_KEY_ID}
aws_secret_access_key = ${AWS_SECRET_ACCESS_KEY}
AWSCRED
  cat > ~/.aws/config << AWSCFG
[default]
region = ${AWS_REGION}
output = json
AWSCFG
  echo "  ✓ AWS configured (bucket=$AWS_BUCKET, region=$AWS_REGION, output=json)."
else
  echo "  (Skipped AWS config; set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY if you need S3 sync)"
fi

# ---------------------------------------------------------------------------
# 1. System info
# ---------------------------------------------------------------------------
echo ""
echo "[1/10] Recording system info..."
mkdir -p "$EXPERIMENT_DIR/metadata"

cat > "$EXPERIMENT_DIR/metadata/system_info.json" << SYSEOF
{
  "setup_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "hostname": "$(hostname)",
  "gpu": "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')",
  "gpu_memory_mb": "$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || echo 'unknown')",
  "python_version": "$(python3 --version 2>&1)"
}
SYSEOF
cat "$EXPERIMENT_DIR/metadata/system_info.json"

# ---------------------------------------------------------------------------
# 2. Install CoreVital
# ---------------------------------------------------------------------------
echo ""
echo "[2/10] Installing CoreVital..."
cd "$REPO_DIR"

COREVITAL_SHA=$(git rev-parse HEAD)
echo "CoreVital commit: $COREVITAL_SHA"
echo "{\"corevital_commit\": \"$COREVITAL_SHA\", \"branch\": \"$(git branch --show-current)\"}" \
    > "$EXPERIMENT_DIR/metadata/corevital_version.json"

pip install -e ".[all]" --quiet

# ---------------------------------------------------------------------------
# 3. Install experiment dependencies
# ---------------------------------------------------------------------------
echo ""
echo "[3/10] Installing experiment dependencies..."

pip install datasets pandas scikit-learn scipy statsmodels pingouin \
    matplotlib seaborn pyarrow tqdm --quiet

pip freeze > "$EXPERIMENT_DIR/metadata/pip_freeze.txt"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# ---------------------------------------------------------------------------
# 4. Download datasets
# ---------------------------------------------------------------------------
echo ""
echo "[4/10] Downloading datasets..."

EXPERIMENT_DIR_PY="$EXPERIMENT_DIR" python3 << 'PYEOF'
import json
import os
import random
from pathlib import Path
from datasets import load_dataset

data_dir = Path(os.environ["EXPERIMENT_DIR_PY"]) / "data"
data_dir.mkdir(parents=True, exist_ok=True)
random.seed(42)

# --- GSM8K ---
print("  Downloading GSM8K (test split)...")
gsm = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)
print(f"  GSM8K total: {len(gsm)} questions")

indices = random.sample(range(len(gsm)), min(200, len(gsm)))
gsm_sample = gsm.select(indices)

out = data_dir / "gsm8k.jsonl"
with open(out, "w") as f:
    for i, row in enumerate(gsm_sample):
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        item = {
            "id": f"gsm8k_{i:04d}",
            "dataset": "gsm8k",
            "question": row["question"],
            "gold_answer": gold,
            "full_solution": row["answer"],
        }
        f.write(json.dumps(item) + "\n")
print(f"  Saved {i+1} GSM8K questions to {out}")

# --- HumanEval ---
print("  Downloading HumanEval...")
he = load_dataset("openai/openai_humaneval", split="test", trust_remote_code=True)
print(f"  HumanEval total: {len(he)} problems")

out = data_dir / "humaneval.jsonl"
with open(out, "w") as f:
    for i, row in enumerate(he):
        item = {
            "id": f"he_{i:04d}",
            "dataset": "humaneval",
            "task_id": row["task_id"],
            "prompt": row["prompt"],
            "canonical_solution": row["canonical_solution"],
            "test": row["test"],
            "entry_point": row["entry_point"],
        }
        f.write(json.dumps(item) + "\n")
print(f"  Saved {i+1} HumanEval problems to {out}")

# --- Summary ---
print("\n  Dataset files:")
for p in sorted(data_dir.glob("*.jsonl")):
    lines = sum(1 for _ in open(p))
    print(f"    {p.name}: {lines} problems")
PYEOF

# ---------------------------------------------------------------------------
# 5. Download models
# ---------------------------------------------------------------------------
echo ""
echo "[5/10] Downloading models (this takes 20-40 minutes)..."

python3 << 'PYEOF'
import json, gc
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

models = [
    ("meta-llama/Llama-3.1-8B-Instruct", False),
    ("Qwen/Qwen2.5-7B-Instruct", False),
    ("mistralai/Mistral-7B-Instruct-v0.3", False),
    ("mistralai/Mixtral-8x7B-Instruct-v0.1", False),  # Weights only; 8bit loaded at runtime
]

for model_id, _ in models:
    print(f"\n  Downloading {model_id}...")
    try:
        AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype="auto", device_map="cpu", trust_remote_code=True,
        )
        print(f"    ✓ Downloaded.")
        del model; gc.collect()
    except Exception as e:
        print(f"    ✗ FAILED: {e}")
        print(f"    Re-run setup or: huggingface-cli login")

print("\n  All models downloaded.")
PYEOF

# ---------------------------------------------------------------------------
# 6. Create directory structure
# ---------------------------------------------------------------------------
echo ""
echo "[6/10] Creating directories..."

mkdir -p "$EXPERIMENT_DIR"/{traces,results,analysis,logs,smoke_test}
ln -sfn "$SCRIPT_DIR/scripts" "$EXPERIMENT_DIR/scripts"

echo "  ~/experiment/"
echo "  ├── scripts/ → (symlink to repo)"
echo "  ├── data/         (datasets)"
echo "  ├── traces/       (CoreVital traces)"
echo "  ├── results/      (features, grades)"
echo "  ├── analysis/     (figures, results)"
echo "  ├── smoke_test/   (15 verification traces)"
echo "  └── logs/"

# ---------------------------------------------------------------------------
# 7. Verify
# ---------------------------------------------------------------------------
echo ""
echo "[7/10] Verification..."

python3 << 'PYEOF'
import os, sys
from pathlib import Path

checks = []
try:
    from CoreVital.instrumentation.collector import InstrumentationCollector
    from CoreVital.config import Config
    checks.append(("CoreVital imports", True))
except Exception as e:
    checks.append(("CoreVital imports", f"FAIL: {e}"))

try:
    import pandas, sklearn, scipy, matplotlib, seaborn, pyarrow
    checks.append(("Analysis imports", True))
except Exception as e:
    checks.append(("Analysis imports", f"FAIL: {e}"))

try:
    data_dir = Path(os.environ.get("EXPERIMENT_DIR_PY", Path.home() / "experiment")) / "data"
    for name in ["gsm8k.jsonl", "humaneval.jsonl"]:
        assert (data_dir / name).exists(), f"Missing {name}"
    checks.append(("Datasets present", True))
except Exception as e:
    checks.append(("Datasets present", f"FAIL: {e}"))

try:
    import torch
    assert torch.cuda.is_available()
    gpu = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    mem = (getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)) / 1e9
    checks.append((f"GPU: {gpu} ({mem:.0f}GB)", True))
except Exception as e:
    checks.append(("GPU", f"FAIL: {e}"))

print("\n  Verification:")
ok = True
for name, status in checks:
    icon = "✓" if status is True else "✗"
    print(f"    {icon} {name}" + ("" if status is True else f" — {status}"))
    if status is not True: ok = False

if ok: print("\n  ✓ All checks passed.")
else: print("\n  ✗ Fix issues before proceeding."); sys.exit(1)
PYEOF

# ---------------------------------------------------------------------------
# 8. Sync from S3 (pull existing experiment state if any)
# ---------------------------------------------------------------------------
echo ""
echo "[8/10] Syncing from S3 (pull existing state)..."

if command -v aws &>/dev/null && [ -n "${AWS_ACCESS_KEY_ID:-}" ]; then
  if aws s3 ls "s3://${AWS_BUCKET}/" &>/dev/null; then
    aws s3 sync "s3://${AWS_BUCKET}/experiment" "$EXPERIMENT_DIR" \
      --exclude "*.pyc" --exclude "*__pycache__*" \
      --no-progress --only-show-errors 2>/dev/null || true
    echo "  ✓ Pull from s3://${AWS_BUCKET}/experiment done."
  else
    echo "  (Bucket empty or first run; nothing to pull.)"
  fi
else
  echo "  (AWS not configured; skip sync.)"
fi

# ---------------------------------------------------------------------------
# 9. Monitoring and sync helpers (check_run, check_resources, sync_to_s3)
# ---------------------------------------------------------------------------
echo ""
echo "[9/10] Installing monitoring commands..."

mkdir -p "$EXPERIMENT_DIR/bin"
chmod +x "$SCRIPT_DIR/scripts/check_run.py" "$SCRIPT_DIR/scripts/check_resources.sh" "$SCRIPT_DIR/scripts/sync_to_s3.sh" 2>/dev/null || true
cat > "$EXPERIMENT_DIR/bin/check_run" << 'BINRUN'
#!/usr/bin/env bash
exec python3 ~/experiment/scripts/check_run.py "$@"
BINRUN
cat > "$EXPERIMENT_DIR/bin/check_resources" << 'BINRES'
#!/usr/bin/env bash
exec bash ~/experiment/scripts/check_resources.sh "$@"
BINRES
cat > "$EXPERIMENT_DIR/bin/sync_to_s3" << 'BINSYNC'
#!/usr/bin/env bash
exec bash ~/experiment/scripts/sync_to_s3.sh "$@"
BINSYNC
chmod +x "$EXPERIMENT_DIR/bin/check_run" "$EXPERIMENT_DIR/bin/check_resources" "$EXPERIMENT_DIR/bin/sync_to_s3"

# Add to PATH for current and future shells
ADD_PATH="export PATH=\"\$HOME/experiment/bin:\$PATH\""
if ! grep -q 'experiment/bin' ~/.bashrc 2>/dev/null; then
  echo "" >> ~/.bashrc
  echo "# CoreVital experiment helpers" >> ~/.bashrc
  echo "$ADD_PATH" >> ~/.bashrc
fi
eval "$ADD_PATH" 2>/dev/null || true
echo "  ✓ Commands available: check_run, check_resources, sync_to_s3"
echo "    (If this is your first shell, run: source ~/.bashrc  or  export PATH=\"\$HOME/experiment/bin:\$PATH\")"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps (hands-off):"
echo "  1. SMOKE TEST:  cd ~/experiment && python3 scripts/smoke_test.py"
echo "  2. DRY RUN:     cd ~/experiment && python3 scripts/run_experiment.py --dry-run"
echo "  3. FULL RUN:    cd ~/experiment && nohup python3 scripts/run_experiment.py > logs/full.log 2>&1 &"
echo "  4. MONITOR:     check_run          # progress, errors, what's done/left"
echo "                  check_resources   # GPU and CPU usage"
echo "  5. UPLOAD:      sync_to_s3        # after run, push results to S3"