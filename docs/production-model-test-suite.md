# Production Model Test Suite (RunPod / GPU)

**Purpose:** Run production-level tests for **all** listed CoreVital models (Llama, Mistral, Mixtral, Qwen, etc.) on a GPU pod (e.g. RunPod), with **instruct** variants for coherent answers, runtime metrics logged to a file, and optional S3 sync so results can be inspected after shutting down the pod. Tests must be **flawless** so no debugging is needed on the expensive GPU.

**When to run:** **After** the next push. Install the latest CoreVital from PyPI (or from the repo tag) on the pod so the run reflects the released version.

---

## Goals

1. **Coverage:** Run CoreVital on every production model we list (see [Model compatibility](model-compatibility.md)).
2. **Instruct variants:** Use instruct/chat models so outputs are coherent and a human reader can see how metrics reflect good vs bad answers.
3. **Runtime metrics:** Write runtime metrics (timing, memory, etc.) to a **log file** during inference so you can verify behavior without re-running.
4. **No JSON trace per run by default:** Use SQLite-only (no `--write-json`) unless you explicitly want a file; optionally sync the **database** to S3.
5. **Zero on-pod debugging:** Script is idempotent, well-tested locally (e.g. with small models or mocks), and fails fast with clear errors.

---

## Model list (instruct where available)

Prefer **instruct** (or chat) variants so that outputs are interpretable (good/bad/weird). Example mapping (adjust to actual HF IDs and availability):

| Family   | Base example           | Instruct example (preferred for test)     |
|----------|------------------------|-------------------------------------------|
| Llama 3  | Llama-3.2-1B/3B        | meta-llama/Llama-3.2-1B-Instruct (or 3B-Instruct) |
| Mistral  | Mistral-7B-v0.1        | mistralai/Mistral-7B-Instruct-v0.2        |
| Mixtral  | Mixtral-8x7B-v0.1      | mistralai/Mixtral-8x7B-Instruct-v0.1      |
| Qwen2    | Qwen2-0.5B/7B         | Qwen/Qwen2-0.5B-Instruct or Qwen2-7B-Instruct |

Use the same list as in `docs/model-compatibility.md` and `tests/test_models_production.py`, but **instruct** IDs for this suite.

---

## RunPod workflow

1. **Spin up a RunPod pod** with enough GPU memory for the largest model (e.g. Mixtral).
2. **Install once:**  
   `pip install corevital[dashboard]` (after publish) or `pip install -e .[dashboard]` from a fresh clone of the **tagged release**.
3. **Environment:** Set `HF_TOKEN` if using gated models (Llama, etc.).
4. **Run the test script** (see below). It should:
   - Create a single SQLite DB (e.g. `runs/corevital.db`) and a log file (e.g. `runs/runtime_metrics.log`).
   - For each model: load → run a fixed set of prompts (2–3 short instruct prompts) → write to DB only (no per-run JSON).
   - Log runtime metrics (time per step, memory, any `--perf` output) to the log file.
   - Not fail silently: exit non-zero and print clear errors if something breaks.
5. **Optional: sync to S3**  
   After the script exits successfully, sync the DB (and optionally the log file) to S3 so you can download and inspect after shutdown:
   - `aws s3 cp runs/corevital.db s3://your-bucket/corevital-runs/runpod-YYYYMMDD-HHMM.db`
   - `aws s3 cp runs/runtime_metrics.log s3://your-bucket/corevital-runs/runpod-YYYYMMDD-HHMM.log`
6. **Shut down the pod** to avoid cost.

---

## Runtime metrics log

The test script should enable **performance logging** so that:

- Every run logs at least: model_id, prompt (truncated), duration, max_new_tokens, and any `--perf` summary (or a single line per run).
- Optionally, use CoreVital’s `--perf detailed` and redirect stderr to the log file, e.g.:
  - `python -m CoreVital.cli run ... --perf detailed 2>>runs/runtime_metrics.log`
- Or use a small wrapper that runs CoreVital and appends one line per run to `runs/runtime_metrics.log` with: timestamp, model_id, trace_id, wall_time_sec, peak_memory_mb (if available). That way you can confirm runtime metrics without re-running.

Design choice: either rely on CoreVital’s existing `--perf` output (and capture it in the log file), or add a minimal “run summary” logger that writes one line per run. The important part is: **no need to re-run to see how runtime metrics behaved**; it’s all in the log file.

---

## Test script requirements

- **Single entrypoint:** e.g. `scripts/run_production_model_tests.py` or a shell script that calls the CLI in a loop.
- **Idempotent:** Can be re-run; uses one DB and appends to one log file.
- **Model list:** Read from a config file or a constant list (instruct model IDs only).
- **Prompts:** 2–3 short, fixed instruct prompts (e.g. “What is 2+2? Answer in one sentence.”, “List one benefit of exercise.”) so outputs are comparable and obviously good/bad.
- **No per-run JSON:** Use default sink (SQLite). Do **not** use `--write-json` unless you explicitly want a file for a specific run.
- **Exit code:** 0 only if all models and all prompts succeed; otherwise 1 and print which model/prompt failed.
- **Tested locally first:** Run with a small model (e.g. Qwen2-0.5B or gpt2) and/or with mocks before using on RunPod so the script is known to work.

---

## Checklist before RunPod run

- [ ] CoreVital is pushed (or tagged) and installable on the pod (`pip install corevital` or install from tag).
- [ ] Script runs successfully locally with at least one small model (e.g. CPU or small GPU).
- [ ] Log file path and format are documented (e.g. in this doc or in script `--help`).
- [ ] S3 sync commands (or a small script) are documented so you can upload `runs/corevital.db` and `runs/runtime_metrics.log` after the run.
- [ ] Instruct model IDs are confirmed (HF catalog) and match the script/list.

---

## References

- Model list and notes: [model-compatibility.md](model-compatibility.md)
- Production smoke tests (current): `tests/test_models_production.py` (run with `pytest -m slow` or `-m "slow and gpu"`)
- Default sink and JSON: SQLite default; optional `--write-json` / `--json-pretty` (see README and CLI help)
