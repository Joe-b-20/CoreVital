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

## Phase-1b: Prompt Telemetry ✅
**Branch:** `phase-1b` | **Merge target:** `main` | **Status: COMPLETE**

New forward pass for prompt analysis. Each piece independently testable.

| # | What | Files | Status | Notes |
|---|------|-------|--------|-------|
| 1 | CausalLM prompt forward | `collector.py` | ✅ | `model(input_ids)` before `generate()`. CLI: `--no-prompt-telemetry`. Config: `PromptTelemetryConfig`. |
| 2 | Seq2Seq encoder reuse | `collector.py` | ✅ | Reuses existing encoder outputs from `_generate_seq2seq_manual` (zero-cost). |
| 3 | Vectorized sparse extraction | `summaries.py` | ✅ | `torch.where` per head on full seq×seq matrix. Threshold configurable (default 0.01). |
| 4 | Basin score | `summaries.py` | ✅ | Middle/boundary ratio per head. Short sequences (< 3 tokens) default to 1.0. |
| 5 | Layer transformation | `summaries.py` | ✅ | `F.cosine_similarity` between consecutive layers, averaged across tokens. |
| 6 | Prompt surprisal | `summaries.py` | ✅ | `CrossEntropyLoss(reduction='none')` on shifted logits/labels. CausalLM only (empty for Seq2Seq). |
| 7 | Report wiring | `report_builder.py` | ✅ | `_build_prompt_analysis()` populates `report.prompt_analysis`. |

**Also added:**
- `PromptTelemetryConfig` in `config.py` with `enabled` and `sparse_threshold`
- `prompt_telemetry` section in `configs/default.yaml`
- `PromptForwardData` dataclass in `collector.py`
- Mock forward pass (`__call__`) for CausalLM in test fixtures

**Exit criteria results:**
- ✅ `prompt_analysis` populated for CausalLM (GPT-2): 12 layers, sparse heads, basin scores, 11 transformations, 4 surprisals
- ✅ `prompt_analysis` populated for Seq2Seq (flan-t5-small): 8 layers, sparse heads, basin scores, 7 transformations, 0 surprisals (correct)
- ✅ `--no-prompt-telemetry` → `prompt_analysis: null`
- ✅ `--perf detailed` shows `prompt_forward_pass` (115ms) and `_build_prompt_analysis` (10ms)
- ✅ `pytest` 42/42, `ruff` clean

---

## Phase-1c: Health Flags + Transient Buffer
**Branch:** `phase-1c` | **Merge target:** `main` | **Status: COMPLETE**

Post-processing aggregation. Transient buffer with explicit lifecycle.

| # | What | Files | Status | Notes |
|---|------|-------|--------|-------|
| 1 | Transient buffer infra | `report_builder.py` | ✅ | Buffer built inside `_build_health_flags()` as local variable. Last 5 generated steps, last-layer last-token hidden vectors. |
| 2 | Repetition loop detection | `summaries.py` | ✅ | `cosine_sim > 0.9995` (threshold raised from 0.99 to account for float16 anisotropy). |
| 3 | Mid-layer anomaly detection | `summaries.py` | ✅ | Per-step dynamic 5× L2 baseline. NaN/Inf only (attention collapse excluded — it's structural, not runtime). |
| 4 | Aggregate health flags | `report_builder.py` | ✅ | `_build_health_flags()` combines: NaN/Inf, collapse, high entropy (>4.0), repetition, mid-layer anomaly. |
| 5 | Buffer teardown | `report_builder.py` | ✅ | `del buffer; torch.cuda.empty_cache()`. Verified no leak in serialized JSON. |

**Implementation notes:**
- Buffer lives in `_build_health_flags()` (report builder), not in collector — cleaner since timeline data already contains raw tensors
- Repetition threshold raised to **0.9995**: GPT-2 float16 produces cosine sims of 0.992-0.999 for non-repetitive text due to last-layer anisotropy; true repetition gives ~1.0
- Mid-layer anomaly uses **per-step baselines** (not global): step 0 in CausalLM processes full prompt (different L2 scale than single-token steps 1+)
- Attention collapse removed from mid-layer check: GPT-2 has 62 collapsed head occurrences across 10 steps — this is model architecture, not a runtime anomaly

**Exit criteria results:**
- ✅ Health flags populated for GPT-2: `{nan: false, inf: false, collapse: true, high_entropy: 0, repetition: false, mid_layer: false}`
- ✅ Health flags populated for flan-t5-small: `{nan: false, inf: false, collapse: true, high_entropy: 5, repetition: false, mid_layer: false}`
- ✅ `--no-prompt-telemetry` → prompt_analysis null, health_flags still populated
- ✅ Repetition detection triggers on identical vectors, does not false-positive on normal text
- ✅ Mid-layer anomaly triggers on NaN/Inf and L2 explosion, does not false-positive on GPT-2
- ✅ No buffer data in serialized JSON output
- ✅ `_build_health_flags` cost: 1.23ms (GPT-2), 0.76ms (flan-t5-small)
- ✅ `pytest` 54/54, `ruff` clean

### Key Design Decisions

**Decision 1 — Transient buffer lives in report builder, not collector**

The original plan placed the buffer in `collector.py` ("Born before generate, fed during steps, killed after health flags"). During implementation, we realized that `_process_timeline` already stores raw `hidden_states` tensors in each `StepData` object, which are passed to the report builder via `InstrumentationResults.timeline`. Building the buffer from existing data inside `_build_health_flags()` is cleaner: no buffer management threaded through the generation loop, no new state on the collector, and the "transient" contract is enforced by being a local variable in a single method.

**Decision 2 — Repetition threshold raised from 0.99 to 0.9995 (float16 anisotropy)**

The metrics analysis document proposed `cosine_sim > 0.99` based on the intuition that "vectors pointing in the same direction = repetition." E2E testing with GPT-2 on CUDA float16 revealed this causes **false positives on 100% normal text**: non-repetitive tokens like "picture", "below", ",", "are", "actually" produce cosine similarities of 0.992–0.999 between consecutive last-layer hidden states. This is the well-documented **anisotropy problem** in transformers — last-layer representations cluster in a narrow cone in high-dimensional space, and float16 quantization exacerbates it.

Empirical separation:
- Non-repetitive GPT-2 float16 CUDA: 0.992–0.999 (below 0.9995)
- True repetition (identical/near-identical tokens): ~1.0 (above 0.9995)
- Threshold 0.9995 provides clean separation with margin on both sides

The threshold remains a parameter in `detect_repetition_loop()` for future tuning.

**Decision 3 — Attention collapse removed from mid-layer anomaly detection**

The metrics analysis document included `collapsed_head_count > 0` as a mid-layer anomaly signal. Testing revealed GPT-2 has **62 collapsed-head occurrences** across 10 steps × 12 layers — nearly every mid-layer has at least one collapsed head. This is a structural property of GPT-2's architecture (well-documented in "Are Sixteen Heads Really Better Than One?"), not a runtime anomaly indicating hallucination. Including it caused 100% false-positive rate.

The separation:
- `attention_collapse_detected` (top-level health flag): reports that the model HAS collapsed heads (structural signal)
- `mid_layer_anomaly_detected`: reports **runtime** anomalies only — NaN/Inf, L2 explosion

**Decision 4 — Per-step L2 baselines instead of global baseline**

The metrics analysis proposed computing a global baseline from "early layers of the first few steps." Testing revealed that CausalLM step 0 processes the **full prompt** (4+ tokens, shape `(1, seq_len, hidden_dim)`), while steps 1+ process **single tokens** (shape `(1, 1, hidden_dim)`) using KV cache. This creates wildly different L2 norm scales:

| | Step 0 (full prompt) | Steps 1+ (single token) |
|---|---|---|
| Early layers (L0–L3) | 4.7 – 694.1 | 4.0 – 30.0 |
| Mid-layers (L4–L7) | 743.2 – 835.9 | 64.4 – 92.3 |

A global baseline of ~84 (mixing both regimes) yields 5× threshold = 420 — step 0's mid-layers (743–835) exceed this, causing a false positive. Per-step baselines correctly normalize: step 0's early mean ~246 gives threshold 1230 (mid-layers 743–835 don't trigger), while step 1's early mean ~14 gives threshold 70 (mid-layers 64–92 don't trigger either). Only genuine L2 explosions cross the per-step threshold.

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
