# CoreVital Model Compatibility (Foundation F4)

CoreVital is tested and designed to work with production-oriented open-weight models. This document notes model-specific quirks and suggested config.

## Tested With

| Family | Example model IDs | Notes |
|--------|-------------------|--------|
| **Llama 3** | `meta-llama/Llama-3.2-1B`, `meta-llama/Llama-3.2-3B` | CausalLM; SDPA by default — for full attention capture you may need `attn_implementation="eager"` when loading. |
| **Mistral** | `mistralai/Mistral-7B-v0.1` | CausalLM; standard attention. |
| **Mixtral** | `mistralai/Mixtral-8x7B-v0.1` | MoE CausalLM; higher memory. |
| **Qwen2** | `Qwen/Qwen2-0.5B`, `Qwen/Qwen2-7B` | CausalLM; standard. |

Smoke tests: `tests/test_models_production.py` (run with `pytest -m slow` or `-m "slow and gpu"`).

## Attention Capture

- **SDPA / Flash Attention:** Some models use scaled dot-product attention that does not return full attention weights. To get per-head attention in the report, load with `attn_implementation="eager"` (or the model’s equivalent option). This can increase memory and runtime.
- **Default:** CoreVital works with whatever the model returns; if attention tensors are omitted, attention summaries will be empty or partial.

## Quantization

- **4-bit / 8-bit:** Use `--quantize-4` or `--quantize-8` with CUDA. Reported `dtype` may show as `quantized_unknown`; health checks (e.g. NaN/Inf) still apply.

## Device

- **CPU:** All tested models run on CPU with small `max_new_tokens`; use for smoke tests and debugging.
- **CUDA:** Recommended for 7B+ and for quantization.

## References

- Execution plan: `docs/phase-2-through-8-execution-plan.md` (Foundation F4).
- Phase-1 metrics: `docs/Phase1 metrics analysis.md`.
