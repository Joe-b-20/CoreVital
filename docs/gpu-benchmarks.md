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

## Results table (template)

When benchmarks are run, update the table below (or the README table) with measured values.

| Model | Device | Baseline (ms) | Instrumented (ms) | Overhead (%) |
|-------|--------|----------------|-------------------|--------------|
| gpt2 | CPU | (see README) | (see README) | (see README) |
| meta-llama/Llama-3.2-1B | CUDA | TBD | TBD | TBD |
| Llama-3.1-8B | CUDA | TBD | TBD | TBD |
| Mixtral-8x7B | CUDA | TBD | TBD | TBD |

To contribute numbers: run with `--perf strict`, fixed seed and short `--max_new_tokens`, and fill in the table (or open a PR with updated docs). The report JSON (e.g. with `--sink local_file --out <dir>`) includes `extensions.performance` with `baseline_ms`, `instrumented_inference_ms`, and `inference_overhead_pct`.
