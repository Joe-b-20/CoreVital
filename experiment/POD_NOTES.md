# RunPod setup – fixes applied

Quick reference so the next pod run goes smoothly.

## Before setup

- **HuggingFace (gated models):**  
  `huggingface-cli` is often not in PATH in containers. Use:  
  `python3 -c "from huggingface_hub import login; login()"`

- **Conda:**  
  `setup.sh` and `run_all.sh` auto-activate `llm_hm` if conda is available. Create that env and install deps there if you use conda.

## Model list (Phi → Qwen 3B)

- **Phi-3.5-mini** was replaced with **Qwen/Qwen2.5-3B-Instruct** in the experiment (Phi had a `DynamicCache.from_legacy_cache` compatibility issue with the current Transformers version).
- `run_experiment.py` and `setup.sh` both use: **qwen3b**, llama, mistral7b, nemo. Trace dirs: `traces/{qwen3b,llama,mistral7b,nemo}`.

## Python 3.11

- `pyproject.toml` allows Python >=3.11 so RunPod’s 3.11 works without editing on the pod.

## GPU memory check

- Scripts use `getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)` so both older and newer PyTorch work.

## Summary work on GPU

- Per-step summary math (attention, hidden, `.max()` etc.) now runs on the same device as the model (GPU when available), with only final scalars brought to CPU. This reduces CPU bottleneck on weak-CPU pods.
