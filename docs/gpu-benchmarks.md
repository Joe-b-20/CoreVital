# GPU Overhead Benchmarks

CoreVital adds overhead to inference by enabling `output_hidden_states`, `output_attentions`, and `output_scores`, and by computing summary statistics. This document describes how to measure overhead and (when run) reports numbers for production-scale models.

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
- **Production models (GPU)**: Benchmarks on Llama-3.1-8B, Mixtral-8x7B, etc. are planned. Early testing suggests overhead is dominated by **attention weight materialization** (`output_attentions=True`) rather than CoreVital’s summary computation.
- **Reducing overhead**: Use `--capture summary` to skip per-layer data and keep overhead low (e.g. under ~5% in many setups).

## Results table

Measured with `--perf strict`, short prompt, 10 `--max_new_tokens`. GPU models used 4-bit quantization. **Note:** Overhead is computed as `(instrumented - baseline) / baseline`. Strict mode runs two warmups before baseline, so both baseline and instrumented runs are warm. When the raw value is negative (baseline run slower than instrumented in that single measurement), it's timing variance—we report **0%** because instrumentation cannot actually reduce inference time. For stable numbers, run multiple times and average.

| Model | Device | Baseline (ms) | Instrumented (ms) | Overhead (%) |
|-------|--------|----------------|-------------------|--------------|
| gpt2 | CPU | 314 | 282 | 0 (variance) |
| meta-llama/Llama-3.1-8B-Instruct | CUDA | 1,248 | 1,243 | 0 (variance) |
| mistralai/Mistral-7B-Instruct-v0.2 | CUDA | 1,345 | 1,338 | 0 (variance) |
| Qwen/Qwen2-0.5B-Instruct | CUDA | 601 | 662 | +10 |
| Mixtral-8x7B | CUDA | TBD | TBD | TBD |

To contribute numbers: run with `--perf strict`, fixed seed and short `--max_new_tokens`, and fill in the table (or open a PR with updated docs). The report JSON (e.g. with `--sink local --out <dir>`) includes `extensions.performance` with `baseline_ms`, `instrumented_inference_ms`, and `inference_overhead_pct`.
