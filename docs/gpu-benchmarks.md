# GPU Overhead Benchmarks

CoreVital adds overhead to inference by enabling `output_hidden_states`, `output_attentions`, and `output_scores`, and by computing summary statistics. This document describes how to measure overhead and (when run) reports numbers for production-scale models. For benchmark **validation** (whether CoreVital signals predict task correctness on GSM8K and HumanEval), see the [Validation Report](validation-report.md) and `experiment/`; the validation experiment used the hardware documented below.

## How to measure

Use **strict performance mode** so that baseline (no instrumentation) and instrumented runs are comparable:

```bash
corevital run --model <model_id> --prompt "Your prompt" --perf strict --max_new_tokens 20
```

The report (or logs) include:

- `original_model_load_ms`
- `warmup_ms`
- `baseline_ms` (raw inference, no instrumentation)
- `instrumented_inference_ms` (instrumentation + generation)
- Overhead is roughly `(instrumented_inference_ms - baseline_ms) / baseline_ms * 100%`

For CSV or scripted runs, parse the report JSON or use `--out` to write results to a file.

## Expected overhead

- **GPT-2 (CPU)**: See the [Measured Overhead](README.md#measured-overhead) table in the README.
- **Production models (GPU)**: Benchmarks on Llama-3.1-8B, Mixtral-8x7B, etc. are in the results table below. The **validation experiment** (GSM8K and HumanEval, four models: Llama-3.1-8B, Qwen-2.5-7B, Mistral-7B, Mixtral-8x7B at full precision) ran on the hardware in the next section; see the [Validation Report](validation-report.md) for accuracy, calibration, and methodology—not overhead. Overhead is dominated by **attention weight materialization** (`output_attentions=True`) rather than CoreVital’s summary computation.
- **Reducing overhead**: Use `--capture summary` to skip per-layer data and keep overhead low (e.g. under ~5% in many setups).
- **Weak-CPU hosts (e.g. RunPod):** By default, report/summary computation runs on CPU after generation so the GPU stays free. On hosts where CPU is the bottleneck, use `--report-on-gpu` so tensors stay on the model device and summary ops run on GPU. See [README CLI options](README.md#cli-options-run) and `config.device.report_on_gpu` / `COREVITAL_DEVICE_REPORT_ON_GPU`.

## Validation experiment: scale and models

The validation experiment (see [Validation Report](validation-report.md) and `experiment/`) produced **~14,500 instrumented traces** on two benchmarks with four models:

| Model (short) | Full model ID | Dataset | Prompts | Runs |
|---------------|----------------|---------|---------|------|
| Llama | meta-llama/Llama-3.1-8B-Instruct | GSM8K | 200 | 2,000 |
| Llama | meta-llama/Llama-3.1-8B-Instruct | HumanEval | 164 | 1,640 |
| Qwen | Qwen/Qwen2.5-7B-Instruct | GSM8K | 200 | 2,000 |
| Qwen | Qwen/Qwen2.5-7B-Instruct | HumanEval | 164 | 1,640 |
| Mistral 7B | mistralai/Mistral-7B-Instruct-v0.3 | GSM8K | 200 | 2,000 |
| Mistral 7B | mistralai/Mistral-7B-Instruct-v0.3 | HumanEval | 164 | 1,640 |
| Mixtral | mistralai/Mixtral-8x7B-Instruct-v0.1 | GSM8K | 201 | 2,001 |
| Mixtral | mistralai/Mixtral-8x7B-Instruct-v0.1 | HumanEval | 162 | 1,620 |

All GPU models were run at full precision (no quantization). Run configuration and counts are in `experiment/results/extraction_summary.md` and `experiment/results/extraction_manifest.json`.

## Validation experiment hardware

The validation experiment was run on the following environment. Full metadata: `experiment/metadata/system_info.json`.

| Item | Value |
|------|--------|
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition |
| GPU memory | 97,887 MB |
| Python | 3.12.3 |
| Setup date | 2026-03-08 |

This doc does not report validation outcomes (accuracy, ECE, AUROC); those are in the validation report and experiment artifacts.

## Results table (strict mode)

Measured with `--perf strict`, short prompt, 10 `--max_new_tokens`. GPU models in the validation experiment were run at full precision. The validation experiment used Llama-3.1-8B, Qwen-2.5-7B, Mistral-7B, and Mixtral-8x7B; the table below includes overhead for those models where measured. The Qwen2-0.5B row is a smaller variant tested separately. **Note:** Overhead is computed as `(instrumented - baseline) / baseline`. Strict mode runs two warmups before baseline, so both baseline and instrumented runs are warm. When the raw value is negative (baseline run slower than instrumented in that single measurement), it's timing variance—we report **0%** because instrumentation cannot actually reduce inference time. For stable numbers, run multiple times and average.

| Model | Device | Baseline (ms) | Instrumented (ms) | Overhead (%) |
|-------|--------|----------------|-------------------|--------------|
| gpt2 | CPU | 314 | 282 | 0 (variance) |
| meta-llama/Llama-3.1-8B-Instruct | CUDA | 1,248 | 1,243 | 0 (variance) |
| mistralai/Mistral-7B-Instruct-v0.2 | CUDA | 1,345 | 1,338 | 0 (variance) |
| Qwen/Qwen2-0.5B-Instruct | CUDA | 601 | 662 | +10 |
| mistralai/Mixtral-8x7B-Instruct-v0.1 | CUDA | TBD | TBD | TBD |

To contribute strict-mode numbers: run with `--perf strict`, fixed seed and short `--max_new_tokens`, and fill in the table (or open a PR with updated docs). The report JSON (e.g. with `--sink local --out <dir>`) includes `extensions.performance` with `baseline_ms`, `instrumented_inference_ms`, and `inference_overhead_pct`.

## Overhead from the validation experiment (features.parquet)

The experiment pipeline writes timing into `experiment/results/features.parquet` for each trace: `total_run_ms`, `inference_ms`, `overhead_ms`, `overhead_pct` (overhead = total − inference, as % of total). These come from `extensions.performance` when present in the trace. 

| Model | Device | Median inference (ms) | Median total (ms) | Overhead (%) | n runs |
|-------|--------|------------------------|-------------------|--------------|--------|
| Llama-3.1-8B | CUDA | 14,184 | 71,500 | 82.5 | 3,640 |
| Qwen-2.5-7B | CUDA | 12,055 | 53,712 | 83.0 | 3,640 |
| Mistral-7B-v0.3 | CUDA | 5,654 | 33,949 | 83.3 | 3,640 |
| Mixtral-8x7B-v0.1 | CUDA | 13,542 | 79,284 | 82.2 | 3,621 |

Column definitions: **Median inference** = median `inference_ms` (model forward pass); **Median total** = median `total_run_ms` (end-to-end); **Overhead (%)** = median `overhead_pct` (100 × (total − inference) / total). The high overhead in this table reflects full capture (per-step summaries, report build, I/O); it is *not* the same as strict-mode baseline-vs-instrumented inference overhead above.
