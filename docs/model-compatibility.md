# CoreVital Model Compatibility

CoreVital is tested with production-oriented open-weight models. This document covers supported architectures, attention capture, quantization, and model-specific notes.

## Tested Models

| Family | Example model IDs | Notes |
|--------|-------------------|--------|
| **Llama 3** | `meta-llama/Llama-3.2-1B`, `meta-llama/Llama-3.1-8B-Instruct` | CausalLM; SDPA by default -- for full attention capture use `attn_implementation="eager"` when loading. |
| **Mistral** | `mistralai/Mistral-7B-v0.1`, `mistralai/Mistral-7B-Instruct-v0.2` | CausalLM; standard attention. |
| **Mixtral** | `mistralai/Mixtral-8x7B-v0.1`, `mistralai/Mixtral-8x7B-Instruct-v0.1` | MoE CausalLM; higher memory. |
| **Qwen2** | `Qwen/Qwen2-0.5B`, `Qwen/Qwen2-0.5B-Instruct` | CausalLM; standard. |
| **GPT-2** | `gpt2` | CausalLM; used for CI smoke tests (small, no gating). |
| **T5 / Flan-T5** | `google/flan-t5-small` | Seq2Seq; full encoder/decoder/cross-attention support. |

Smoke tests: `tests/test_models_production.py` (run with `pytest -m slow` or `-m "slow and gpu"`).

## Supported Architectures

**Causal Language Models (GPT-2, Llama, Mistral, Qwen, etc.):**
Fully supported with automatic detection. CoreVital automatically switches attention implementation from SDPA to `eager` for Llama models to enable attention weight capture. This may slightly increase inference time but is necessary for attention analysis.

**Sequence-to-Sequence Models (T5, BART, etc.):**
Fully supported with automatic detection and deep instrumentation. CoreVital uses manual generation to capture hidden states and attentions, as Seq2Seq models don't return these via the standard `generate()` method. For Seq2Seq models, the tool captures:
- **Encoder outputs**: Encoder hidden states and encoder self-attention (computed once, fixed for the entire run)
- **Decoder outputs**: Decoder hidden states and decoder self-attention (computed at each generation step)
- **Cross-attention**: How the decoder attends to encoder outputs at each generation step

**Other Models:**
Models using eager attention by default will work without modification. Models that don't support attention output will log warnings and attention summaries will be empty or partial.

## Attention Capture

- **SDPA / Flash Attention:** Some models use scaled dot-product attention that does not return full attention weights. To get per-head attention in the report, load with `attn_implementation="eager"` (or the model's equivalent option). This can increase memory and runtime.
- **Default:** CoreVital works with whatever the model returns; if attention tensors are omitted, attention summaries will be empty or partial.

## Quantization

- **4-bit / 8-bit:** Use `--quantize-4` or `--quantize-8` with CUDA. Reported `dtype` may show as `quantized_unknown`; health checks (e.g. NaN/Inf) still apply.
- Quantization requires CUDA and will automatically fall back to CPU without quantization if CUDA is unavailable.
- The quantization status is reflected in the report's `model.quantization` field.

## Device

- **CPU:** All tested models run on CPU with small `max_new_tokens`; use for smoke tests and debugging.
- **CUDA:** Recommended for 7B+ and for quantization.

## Per-model threshold profiles

Detection thresholds (repetition cosine similarity, L2 explosion multiplier, high-entropy cutoff, collapsed/focused head thresholds) can be overridden per model family so that different architectures use appropriate values.

- **Location:** `configs/model_profiles/`. Each file is a YAML with keys: `l2_explosion_multiplier`, `high_entropy_threshold_bits`, `repetition_cosine_threshold`, `collapsed_head_entropy_threshold`, `focused_head_concentration_threshold`.
- **Resolution:** At report build time, CoreVital maps the model’s HuggingFace architecture string (e.g. `GPT2LMHeadModel`, `LlamaForCausalLM`) to a profile key and loads `configs/model_profiles/<key>.yaml`. If that file is missing, `default.yaml` is used.
- **Mapping:** `GPT2*` → `gpt2`, `Llama*` → `llama`, `Mistral*` → `mistral`, `T5*` → `t5`, `Bart*` → `bart`; anything else → `default`.
- **Override:** You can set `config.model_profile` (e.g. from a custom YAML or code) to bypass auto-loading and use that profile for all runs with that config.

See `configs/model_profiles/default.yaml` for the default values and add or edit files (e.g. `gpt2.yaml`, `llama.yaml`) to tune behavior per family.

## Numerical stability and attention detail

- **Hidden states:** Before computing mean, std, L2 norm, and max abs, values are clamped to `[-1e6, 1e6]` to avoid NaN propagation. When clamping was applied, `hidden_summary.clipped` is `true`.
- **Attention per-head max:** Each layer's `attention_summary` includes `max_weight_per_head` (one float per head: the maximum attention weight that head assigns to any key). This helps spot specialist heads (e.g. Voita et al. 2019: ~80% max weight) and individual head failures that mean-only aggregation can hide.

## References

- Phase-1 metrics research: [Phase1 metrics analysis](Phase1%20metrics%20analysis.md)
- Design decisions: [Design Journey](design-journey.md)
