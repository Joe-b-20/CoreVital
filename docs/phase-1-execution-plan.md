# Phase-1 Execution Plan

**Source of truth for metrics:** `docs/Phase1 metrics analysis.md`
**Source of truth for implementation:** This document.

**Branch strategy:** 4 sequential sub-phases, each merged to `main` before the next starts.

---

## Phase-1a: Schema + Enhanced Metrics ✅
**Branch:** `phase-1a` | **Merge target:** `main` | **Status: COMPLETE**

Metrics that operate on data we already have. No new forward passes.

| # | What | Files | Status | Notes |
|---|------|-------|--------|-------|
| 1 | Schema v0.3.0 | `schema.py`, `__init__.py`, `pyproject.toml` | ✅ | 5 new models, 4 extended models, version bump 0.2.0→0.3.0 |
| 2 | Numerical stability | `summaries.py` | ✅ | `log_softmax` + `nan_to_num` for `0*(-inf)` edge case. Full-vocab entropy in bits. |
| 3 | Enhanced logit metrics | `summaries.py` | ✅ | Shared intermediates: `log_softmax`→`probs`→`topk` computed once. |
| 4 | Enhanced attention aggregation | `summaries.py` | ✅ | Per-head entropy/concentration as shared intermediates, then min/max/count. |
| 5 | NaN/Inf detection | `summaries.py`, `report_builder.py` | ✅ | `detect_tensor_anomalies()` wired per layer in `_build_layer_summaries`. |
| 6 | Report builder + validation | `report_builder.py`, `validation.py` | ✅ | v0.3.0, `actual_token_id` passed for surprisal, `prompt_analysis=None`, `health_flags=None`. |
| 7 | Perf wrapping | — | ⚠️ | Metrics computed within existing `_build_timeline` operation (per-step timing captured). Individual metric wrapping deferred — each metric is <1ms, wrapping overhead would exceed metric cost. |
| 8 | Tests | `tests/` | ✅ | 42/42 pass. Version assertions updated. |

**Also fixed:**
- `configs/default.yaml` — added new stat names (YAML was overriding Python defaults)
- `config.py` — updated default stats lists for logits and attention

**Exit criteria results:**
- ✅ `pytest` green (42/42)
- ✅ `ruff` clean (lint + format)
- ✅ New metrics in JSON output verified with real GPT-2 CPU run
- ⚠️ Per-metric perf costs captured in aggregate (per-step timing), not individually separated

---

## Phase-1b: Prompt Telemetry
**Branch:** `phase-1b` | **Merge target:** `main` (after 1a merged)

New forward pass for prompt analysis. Each piece independently testable.

| # | What | Files | Details |
|---|------|-------|---------|
| 1 | CausalLM prompt forward | `collector.py` | `model(input_ids)` before `generate()`. CLI: `--no-prompt-telemetry`. |
| 2 | Seq2Seq encoder reuse | `collector.py` | Explicit encoder call, pass to `generate(encoder_outputs=...)`. |
| 3 | Vectorized sparse extraction | `summaries.py` | `torch.where(attn > threshold)` → `SparseAttentionHead` (SoA). |
| 4 | Basin score | `summaries.py` | Vectorized middle/boundary ratio per head. |
| 5 | Layer transformation | `summaries.py` | `F.cosine_similarity(h[:-1], h[1:])` per prompt token. |
| 6 | Prompt surprisal | `summaries.py` | `CrossEntropyLoss(reduction='none')` on prompt logits. |
| 7 | Report wiring | `report_builder.py` | Populate `report.prompt_analysis`. |

**Exit criteria:** `prompt_analysis` populated for both CausalLM and Seq2Seq, `--no-prompt-telemetry` skips it, perf breakdown shows costs.

---

## Phase-1c: Health Flags + Transient Buffer
**Branch:** `phase-1c` | **Merge target:** `main` (after 1b merged)

Post-processing aggregation. Transient buffer with explicit lifecycle.

| # | What | Files | Details |
|---|------|-------|---------|
| 1 | Transient buffer infra | `collector.py` | `List[Tensor]` capacity 5, FIFO. Born before generate, fed during steps, killed after health flags. Never serialized. |
| 2 | Repetition loop detection | `summaries.py` / new | `cosine_sim > 0.99` across buffer entries → `repetition_loop_detected`. |
| 3 | Mid-layer anomaly detection | `summaries.py` / new | Dynamic 5× early-layer L2 baseline. Mid-layer NaN/Inf + collapse check. |
| 4 | Aggregate health flags | `report_builder.py` | Combine all signals into `HealthFlags`. |
| 5 | Buffer teardown | `collector.py` | `del buffer; torch.cuda.empty_cache()`. Assert no leaks. |

**Buffer lifecycle:** Allocate → Feed (per step) → Consume (health flags) → Kill. Never touches schema.

**Exit criteria:** Health flags in JSON, repetition detected on constructed repetitive prompt, no buffer data in output.

---

## Phase-1d: Dashboard + Sinks
**Branch:** `phase-1d` | **Merge target:** `main` (after 1c merged)

Output and visualization.

| # | What | Files | Details |
|---|------|-------|---------|
| 1 | Streamlit dashboard | `dashboard.py` | Entropy chart, attention heatmap, latency flags, prompt analysis, health flags. |
| 2 | DatadogSink | `sinks/datadog_sink.py` | Replace stub. `datadog-api-client`. |
| 3 | PrometheusSink | `sinks/prometheus_sink.py` | Replace stub. `prometheus_client`. |
| 4 | Sink CLI routing | `cli.py` | `--sink local\|datadog\|prometheus`. |

**Exit criteria:** Dashboard renders real report, sinks pass mocked tests, CLI routes correctly.

---

## Rules

1. **Performance wrapping:** Every new computation wrapped with `monitor.start_operation()` / `monitor.end_operation()`.
2. **Vectorized:** All tensor operations use `torch` ops, no Python loops over individual elements.
3. **No serialization leaks:** Transient data (buffers, intermediate tensors) never enters Pydantic models.
4. **Backward compat:** v0.2.0 reports still loadable (validation accepts both versions).
5. **Tests before merge:** `pytest` + `ruff` + real model run must pass before any merge to `main`.
