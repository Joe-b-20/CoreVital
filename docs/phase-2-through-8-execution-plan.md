# CoreVital Phase-2 Through Phase-8 Execution Plan

**Purpose:** Detailed, instruction-style plan for implementing CoreVital from Phase-2 through end-state (risk scores, fingerprints, early warning, health-aware decoding, cross-model comparison, narratives, dashboard, library API, integrations). Use this document as the source of truth for implementation order and design decisions.

**Source of truth for existing metrics:** `docs/Phase1 metrics analysis.md`  
**Source of truth for Phase-1 implementation:** `docs/phase-1-execution-plan.md`  
**Source of truth for roadmap:** `README.md` (Roadmap section)

**Document conventions:**
- Each phase has: **What** (goal), **How to** (step-by-step instructions), **Files**, **Research / citations**, **Exit criteria**.
- Status values: ⬜ Not started | 🔄 In progress | ✅ Complete

---

## Table of Contents

1. [Foundation (Pre-Phase-2)](#foundation-pre-phase-2)
2. [Phase-2: Risk Scores + Layer Blame](#phase-2-risk-scores--layer-blame)
3. [Phase-3: Prompt Fingerprints](#phase-3-prompt-fingerprints)
4. [Phase-4: Early Warning / Failure-Horizon](#phase-4-early-warning--failure-horizon)
5. [Phase-5: Health-Aware Decoding](#phase-5-health-aware-decoding)
6. [Phase-6: Cross-Model Comparison](#phase-6-cross-model-comparison)
7. [Phase-7: Human-Readable Narratives](#phase-7-human-readable-narratives)
8. [Phase-8: Dashboard + Packaging](#phase-8-dashboard--packaging)
9. [Library API (Streaming + Post-Run)](#library-api-streaming--post-run)
10. [Integrations (OpenTelemetry, Langfuse)](#integrations-opentelemetry-langfuse)
11. [Documentation and Positioning](#documentation-and-positioning)

---

## Foundation (Pre-Phase-2)

**Goal:** Database-first storage, tiered capture (runtime vs deep-dive), RAG context metadata, and focus on production models (Llama, Mistral, Mixtral, Qwen). These unblock later phases and keep payloads small.

### F1. Database-first (JSON as export)

| # | What | Files | Status | Notes |
|---|------|-------|--------|-------|
| F1.1 | Make SQLite default sink | `config.py`, `configs/default.yaml`, `cli.py` | ✅ | Default `sink.type = "sqlite"`; `--sink local` for JSON. Branch: foundation-f1. |
| F1.2 | JSON export path | `cli.py`, `sinks/local_file.py` | ✅ | When `--sink local`, write to `output_dir`. With `--sink sqlite`, no JSON unless `--write-json`; `--json-pretty` uses indented JSON (larger file). |
| F1.3 | Migrate existing JSON to DB | `cli.py` (migrate command) | ✅ | `corevital migrate --from-dir ./runs --to-db runs/corevital.db`; `--dry-run` supported. Tests in test_sinks.py. |
| F1.4 | Dashboard default to DB | `dashboard.py` | ✅ | Database source with default path `runs/corevital.db`. |

**How to (F1.3 migration):**
1. Add subparser `migrate` to CLI.
2. Args: `--from-dir`, `--to-db`, optional `--dry-run`.
3. Glob `from-dir/trace_*.json` (exclude `*_performance_*`).
4. For each file: `deserialize_report_from_json(path.read_text())`, then `SQLiteSink`-style insert (reuse `write()` logic or a shared `insert_report(conn, report)`).
5. Log count migrated; on conflict (trace_id) use INSERT OR REPLACE.

**Exit criteria:** CLI default is SQLite; `corevital migrate` runs without error; dashboard lists runs from DB by default.

---

### F2. Split metrics: runtime vs deep-dive (capture_mode)

**Goal:** Always store a small "summary" (health flags, time series, risk score). Store full trace (layer summaries, sparse attention, sketches) only when `capture_mode` is `full` or `on_risk` (risk above threshold or any health flag set).

| # | What | Files | Status | Notes |
|---|------|-------|--------|-------|
| F2.1 | CaptureConfig | `config.py` | ✅ | `CaptureConfig` added with `capture_mode: Literal["summary", "full", "on_risk"]` and `risk_threshold: float = 0.7`. **Default is currently `"full"`** for backward-compat; refine later if we want summary-by-default. |
| F2.2 | Summary-only build path | `report_builder.py` | ✅ | When `capture_mode in ("summary", "on_risk")`: timeline steps have `layers=[]` but internal per-layer summaries are still computed for health flags; `prompt_analysis.layers[].heads` is empty (no sparse heads) while `basin_scores`, `layer_transformations`, and `prompt_surprisals` remain populated. |
| F2.3 | On-risk trigger | `report_builder.py` | ✅ | When `capture_mode == "on_risk"`: build summary first; after health_flags and risk_score (Phase-2), if `risk_score >= risk_threshold` or any health flag True, attach full layer data (timeline + prompt_analysis sparse heads). Test: `test_report_builder_on_risk_attaches_full_layers_when_triggered`. |
| F2.4 | Schema for summary vs full | `reporting/schema.py` | ⬜ | Optional/deferred: `Report.summary_only: bool` or separate `RunSummary` model. Current: single Report with empty `timeline[].layers` in summary mode. |
| F2.5 | Sink writes | `sinks/sqlite_sink.py` | ⬜ | Deferred: optional split of summary_json vs full_trace_json in DB. Current: single report blob per run. |

**How to (F2.2 summary-only):**
1. In `_build_timeline`, if `config.capture_mode == "summary"`: for each step append `TimelineStep(step_index, token, logits_summary, layers=[], extensions={})`. Compute logits_summary as today (entropy, perplexity, surprisal); do not call `_build_layer_summaries`.
2. In `_build_prompt_analysis`, if summary: still compute layer_transformations, prompt_surprisals, basin_scores per layer; omit `heads` (sparse attention) or set to `[]`.
3. In `_build_health_flags`, you need last-layer hidden states for repetition. So either: (a) keep a minimal buffer (last 5 steps, last-layer last-token) only when capture_mode != "full" and use it for health flags then discard, or (b) compute health flags from existing timeline layer data—but in summary mode there are no layers. Therefore: for summary mode, still collect last-layer last-token hidden states in the collector (or in report_builder from results.timeline step data) for repetition and mid-layer check; do not store them in the report. So collector must expose minimal buffer for health flags even in summary mode.

**Exit criteria:** `capture_mode=summary` produces a report that is much smaller (no per-layer data); `on_risk` produces full trace only when risk or flags trigger; DB stores one row per run with summary and optional full blob.

---

### F3. RAG context metadata

**Goal:** When the user's pipeline is RAG (retrieval + LLM), accept optional context metadata and store it so runs can be correlated with context length, source, etc. No instrumentation of retrieval or embedding models yet.

| # | What | Files | Status | Notes |
|---|------|-------|--------|-------|
| F3.1 | RAG metadata schema | `reporting/schema.py` | ✅ | `RAGContext` model added with `context_token_count`, `retrieved_doc_ids`, `retrieved_doc_titles`, `retrieval_metadata`. Stored in `report.extensions["rag"]`. |
| F3.2 | CLI / API input | `cli.py`, future `monitor.py` | ✅ | CLI: `--rag-context PATH` loads JSON and sets `config.rag_context`; API `set_rag_context` deferred to Library API. |
| F3.3 | Report builder | `report_builder.py` | ✅ | If `config.rag_context` is set, `report.extensions["rag"] = RAGContext(**rag_dict).model_dump()`. |
| F3.4 | Dashboard | `dashboard.py` | ✅ | When `extensions.rag` present, show RAG Context section: context tokens, doc count, titles/IDs, retrieval_metadata expander. |

**How to (F3.1):**
1. Define Pydantic model `RAGContext` in `schema.py` (or in `reporting/extensions.py` to keep schema clean). Add to Report's extensions docstring.
2. In CLI, add `--rag-context` optional path; if set, load JSON and pass dict to ReportBuilder or Config.
3. In ReportBuilder.build(), if rag_context in config/results, assign to report.extensions["rag"].

**Exit criteria:** Runs can carry optional RAG context; dashboard displays it when present; no retrieval instrumentation.

---

### F4. Focus on production models (Llama, Mistral, Mixtral, Qwen)

| # | What | Files | Status | Notes |
|---|------|-------|--------|-------|
| F4.1 | Test matrix | `tests/test_models_production.py` | ✅ | Smoke tests for Llama-3.2-1B, Qwen2-0.5B (CPU/slow); parametrized GPU tests for all four families. Marked `@pytest.mark.slow` and `@pytest.mark.gpu`. |
| F4.2 | README "Tested with" | `README.md` | ✅ | "Tested with" line and link to model-compatibility.md and test_models_production.py. |
| F4.3 | Model-specific notes | `docs/model-compatibility.md` (new) | ✅ | Table of families, attention capture (SDPA/eager), quantization, device. |
| F4.4 | Examples and demo | `README.md`, `docs/` | ✅ | README examples remain GPT-2/flan-t5 for quick start; model-compatibility.md and "Tested with" cover production models. |

**Exit criteria:** Tests exist for Llama, Mistral, Mixtral, Qwen; README and docs reflect production-model focus.

---

## Phase-2: Risk Scores + Layer Blame

**Goal:** Compute a single risk score per run (0–1) from health flags and scalar metrics, and attribute blame to layers where possible (e.g. which layers had collapse or anomaly).

**Source of truth for health flags:** Phase-1c (NaN/Inf, attention collapse, high entropy steps, repetition loop, mid-layer anomaly).

### P2.1 Risk score formula

**Formula (weighted combination):**
- Map each signal to a contribution in [0, 1]:
  - `nan_detected` or `inf_detected` → 1.0 (max risk).
  - `repetition_loop_detected` → 0.9.
  - `mid_layer_anomaly_detected` → 0.7.
  - `attention_collapse_detected` → 0.3 (structural; often present in healthy runs).
  - `high_entropy_steps`: normalize by step count, e.g. `min(1.0, high_entropy_steps / max(1, total_steps)) * 0.5`.
- Aggregate: e.g. `risk_score = min(1.0, sum(contributions))` or use max of any single contribution. Prefer a simple rule: if any NaN/Inf → 1.0; else if repetition → 0.9; else if mid_layer → 0.7; else base + entropy_component.

**Research:** Internal states and entropy are established signals for hallucination and failure. Semantic entropy and internal-state methods (e.g. INSIDE, MIND) show that combining multiple internal signals improves detection (Nature 2024, OpenReview INSIDE; arxiv 2403.06448 MIND). We use a hand-crafted combination of existing CoreVital flags to avoid training; can be tuned later.

**How to calculate (reference implementation):**
```python
def compute_risk_score(health_flags: HealthFlags, summary: Summary, timeline_entropies: List[float]) -> tuple[float, list[str]]:
    factors = []
    if health_flags.nan_detected or health_flags.inf_detected:
        return 1.0, ["nan_or_inf"]
    score = 0.0
    if health_flags.repetition_loop_detected:
        score = max(score, 0.9)
        factors.append("repetition_loop")
    if health_flags.mid_layer_anomaly_detected:
        score = max(score, 0.7)
        factors.append("mid_layer_anomaly")
    if health_flags.attention_collapse_detected:
        score = max(score, 0.3)
        factors.append("attention_collapse")
    total_steps = max(1, summary.total_steps)
    entropy_component = min(1.0, health_flags.high_entropy_steps / total_steps) * 0.5
    score = min(1.0, score + entropy_component)
    if health_flags.high_entropy_steps > 0:
        factors.append("high_entropy_steps")
    return score, factors
```

**How to:**
1. New file `src/CoreVital/risk.py`: `compute_risk_score(health_flags: HealthFlags, summary: Summary, timeline_entropies: List[float]) -> tuple[float, List[str]]`.
2. Implement the mapping above (or the reference); return score in [0, 1] and risk_factors list.
3. Call from ReportBuilder after `_build_health_flags`; set `report.extensions["risk"] = {"risk_score": ..., "risk_factors": [...]}` or add top-level `Report.risk_score` in schema.

**Files:** `risk.py`, `report_builder.py`, `schema.py` (optional new field or extensions).

### P2.2 Layer blame

**Goal:** List which layers contributed to risk (e.g. layers with collapse, or with mid-layer anomaly, or highest entropy variance).

**Approach (simple, no gradients):**
- For each layer and step we already have: attention_summary (collapsed_head_count, entropy_mean), hidden_summary (l2_norm_mean), anomalies (has_nan, has_inf).
- Layer blame = which layers had anomalies or collapse. E.g. `blamed_layers: List[int]` = layer indices where `anomalies.has_nan or anomalies.has_inf or attention_summary.collapsed_head_count > 0` (or use a threshold). For mid-layer anomaly we have a per-step signal; attribute to layers that exceeded the L2 baseline (store in report_builder when computing mid-layer check).
- More advanced: gradient-based attribution (e.g. AttnLRP, Ali et al.) is possible but out of scope for v1; document as future work.

**Research:** Layer-wise attribution (e.g. AttnLRP, LRP for transformers) identifies which layers contribute to predictions or failures (Achtibat et al., MLR 2024; Ali et al., MLR 2022). We use a rule-based blame from existing summaries first.

**How to:**
1. In `_build_health_flags` or a new `_build_layer_blame`, iterate timeline steps and layers; collect layer indices where anomalies or collapse occurred. Optionally add "entropy variance" per layer across steps and blame top-k highest variance.
2. Set `report.extensions["risk"]["blamed_layers"] = [...]` or add `Report.blamed_layers`.
3. Dashboard: show blamed layers in risk section.

**Files:** `report_builder.py`, `risk.py` (optional helper), `schema.py`, `dashboard.py`.

### Phase-2 exit criteria

- [x] `compute_risk_score()` returns a value in [0, 1]; report contains risk_score and risk_factors in `report.extensions["risk"]`.
- [x] Blamed layers list populated via `compute_layer_blame(layers_by_step)`; stored in `extensions["risk"]["blamed_layers"]`; dashboard shows risk section when present.
- [x] Tests: `tests/test_risk.py` (unit tests for risk score and layer blame); mock report builder test asserts `extensions["risk"]` shape.
- [ ] Docs: short description in README (optional follow-up).

---

## Phase-3: Prompt Fingerprints

**Goal:** A compact, comparable representation of a "prompt" or "run" so that runs can be clustered (e.g. which prompt types lead to high risk) and compared.

**Options:**
- **A) Run-summary vector:** Fixed-size vector from run: e.g. [mean_entropy, max_entropy, frac_high_entropy_steps, risk_score, nan, inf, collapse, repetition, mid_layer]. No model call; works offline.
- **B) Prompt embedding:** Embed the prompt text with the same model’s embedding layer or an external embedder; store vector. Enables semantic clustering.
- **C) Hash-based:** Hash of prompt text + model id for exact duplicate detection.

**Recommendation for v1:** Implement A and C. A gives immediate value (cluster by behavior); C is trivial. Add B later if users need semantic similarity.

**Research:** Clustering prompts by behavior or embedding is used in controllable clustering and failure analysis (e.g. ACL 2025 EMNLP Industry, ClusterLLM). We avoid requiring an extra embedder by using run-summary vectors first.

**How to:**
1. **Fingerprint (run-summary):** In ReportBuilder, after risk score, compute `fingerprint_vector: List[float]` from summary + health_flags + risk_score (fixed order, fixed length). Store in `report.extensions["fingerprint"] = {"vector": [...], "prompt_hash": "sha256..."}`.
2. **prompt_hash:** SHA256 of normalized prompt string (strip, lower if desired) + model_id; store for exact duplicate detection.
3. **API:** Expose `get_fingerprint(report) -> dict` for downstream clustering (e.g. in Phase-6 or external scripts).
4. **Dashboard:** Optional "similar runs" later (e.g. k-NN on fingerprint vector); for Phase-3, storing is enough.

**Implemented:** `fingerprint.py`: `compute_fingerprint_vector(timeline, summary, health_flags, risk_score)` (9-dim vector), `compute_prompt_hash(prompt_text, model_id)`, `get_fingerprint(report)`. ReportBuilder sets `extensions["fingerprint"]` for every report. Tests in `test_fingerprint.py`; mock report asserts fingerprint present.

**Exit criteria:** ✅ Every report has a fingerprint (vector + prompt_hash); doc and tests.

---

## Phase-4: Early Warning / Failure-Horizon

**Goal:** Surface signals that indicate "this run is heading toward failure" (e.g. entropy spiking, repetition starting, attention collapsing) in near real time—without promising exact "failure in N steps." Frame as "early warning" or "failure risk" not "failure-horizon prediction."

**Signals to use (all from existing data):**
- Entropy trend: rising entropy over last k steps.
- Repetition precursor: cosine similarity of last-layer hidden states approaching 0.9995 (use same buffer as Phase-1c).
- Attention collapse: increase in collapsed_head_count over steps.
- Mid-layer L2: sudden jump in L2 norms in mid layers.

**Research:** Gnosis (arxiv 2512.20578) shows that monitoring internal states during inference allows early detection of failing trajectories with low overhead. Repetition neurons and feature-level analysis (arxiv 2504.01100, 2504.14218) show that internal activations predict repetition. We use simple heuristics (trends, thresholds) rather than a learned predictor in v1.

**How to:**
1. **Online (streaming):** In monitor (see Library API section), after each step compute: (a) rolling mean entropy over last 5 steps; (b) cosine sim between last two hidden states in buffer. If entropy_trend > 0.2 (rising) or cosine > 0.999, emit `warning_signals: ["entropy_rising", "repetition_risk"]`. No need to store full history; keep last 5 steps’ scalars and last 2 hidden vectors.
2. **Post-run:** In ReportBuilder, compute `failure_risk: float` and `warning_signals: List[str]` from timeline: e.g. if repetition_loop_detected → failure_risk = 0.9; else if entropy trend positive and max_entropy > 4 → failure_risk = 0.6; else 0.3. Store in `report.extensions["early_warning"]`.
3. **Optional:** `estimated_steps_to_failure: Optional[int]` only if we have a calibrated heuristic (e.g. "repetition always within 3 steps after cosine > 0.999"); otherwise leave null.

**Implemented:** `early_warning.py`: `compute_early_warning(timeline, health_flags)` → (failure_risk, warning_signals). ReportBuilder sets `extensions["early_warning"]` for every report. Dashboard shows Early Warning section when present. Streaming deferred to Library API (monitor).

**Exit criteria:** ✅ Post-run report includes failure_risk and warning_signals; ⬜ streaming monitor (Library API).

---

## Phase-5: Health-Aware Decoding

**Goal:** Allow the caller to intervene during generation when risk or early-warning signals exceed a threshold (e.g. resample, lower temperature, or abort). CoreVital does not own the sampling logic; it exposes `should_intervene()` and optional callbacks.

**How to:**
1. **API:** In `CoreVitalMonitor`, after each step: if `get_risk_so_far()` or `get_warning_signals()` exceed config thresholds, set a flag. Method `should_intervene() -> bool` returns that flag; optional callback `on_intervene(step, reason)`.
2. **Caller responsibility:** Caller checks `should_intervene()` after each step (or in callback); if True, they may call `model.generate(..., temperature=0.3)` again or stop.
3. **Config:** `intervene_on_risk_above: float = 0.8`, `intervene_on_signals: List[str] = ["repetition_risk"]`.
4. **No change to HF generate:** We do not modify sampling inside the model; we only recommend intervention. Phase-5 is thus thin: compute running risk and signals, expose API.

**Implemented:** `CoreVitalMonitor` (Library API) exposes `should_intervene()` post-run: True if `risk_score >= intervene_on_risk_above` or any `intervene_on_signals` in `early_warning.warning_signals`. Caller runs `monitor.run(model_id, prompt)` then `if monitor.should_intervene(): ...`. Per-step callback deferred to streaming API.

**Exit criteria:** ✅ Monitor exposes `should_intervene()`; doc/example in execution plan.

---

## Phase-6: Cross-Model Comparison

**Goal:** Compare the same prompt (or prompt set) across different models (e.g. Llama vs Mistral) on risk, health flags, and scalar metrics.

**How to:**
1. **Data:** Runs are already stored with trace_id, model_id, prompt (or prompt_hash), risk_score, health_flags, fingerprint. No new storage; need a "comparison" view.
2. **API or dashboard:** (a) Query runs by prompt_hash (or prompt text) and optional model list. (b) Aggregate per model: mean risk, count of runs with repetition, mean entropy, etc. (c) Present table or chart: model A vs B vs C.
3. **Dashboard:** New view "Compare models": select prompt(s) or fingerprint cluster; select models; show table of metrics (risk, flags, mean entropy). Optional: side-by-side timelines (entropy over steps for model A vs B).
4. **CLI:** Optional `corevital compare --prompt "..." --models llama,mistral,qwen` that runs both (or reads from DB) and prints comparison table.

**Implemented:** SQLite: added `prompt_hash` and `risk_score` columns (populated on write from report.extensions); `list_traces(db_path, model_id=..., prompt_hash=...)` filters. Dashboard: when source is Database, "Compare runs" view shows table of trace_id, model_id, created_at_utc, risk_score, prompt_hash with sidebar filters (model, prompt_hash) and "Export as CSV". Single-run "Export report (JSON)" in sidebar. CLI compare: `corevital compare --db runs/corevital.db [--limit N] [--prompt-hash HASH]` prints per-model run counts and basic risk statistics from SQLite.

**Exit criteria:** ✅ User can compare runs (filter by model/prompt_hash, table, CSV export); ✅ CLI compare.

---

## Phase-7: Human-Readable Narratives

**Goal:** Turn a report (health flags, risk, timeline summary, blamed layers) into a short natural-language narrative, e.g. "The model was confident for the first 10 steps. At step 11 entropy spiked and repetition was detected in the last 3 steps. Layer 4 had attention collapse."

**How to:**
1. **Template-based (v1):** No LLM call. Build a narrative from templates: if repetition_loop_detected → add "Repetition was detected in the last few steps."; if risk_score > 0.7 → add "This run was high risk."; list blamed layers; summarize entropy trend (stable / rising / high in N steps). Concatenate into 2–4 sentences.
2. **Schema:** `report.extensions["narrative"] = {"summary": "..."}` or `Report.narrative: Optional[str]`.
3. **Dashboard:** Show narrative at top of run view.
4. **Optional later:** Use an LLM to generate a longer narrative from the same data (cite model, ensure no PII).

**Implemented:** `narrative.py`: `build_narrative(health_flags, risk_score, blamed_layers, warning_signals)` → summary string. ReportBuilder sets `extensions["narrative"]` after risk and early_warning. Dashboard shows narrative in info box when present. Tests in `test_narrative.py`.

**Exit criteria:** ✅ Every report has a 2–4 sentence narrative; dashboard displays it; tests for template coverage.

---

## Phase-8: Dashboard + Packaging

**Goal:** Polish the Streamlit dashboard (filter by risk, model, date; narratives; comparison view; run detail) and package the project for easy install and run.

**How to:**
1. **Dashboard:** Add filters (model, risk range, date range, prompt_hash); add "Compare" view (Phase-6); ensure narrative and risk/blame are prominent; add "Export" (JSON/csv of runs).
2. **Packaging:** Ensure `pip install -e .` and `pip install corevital` (when published) install CLI and deps; optional extras `[dashboard]`, `[datadog]`, `[prometheus]`, `[all]`. Document in README.
3. **Docker (optional):** Dockerfile for dashboard + SQLite volume; document in production guide.

**Implemented:** Dashboard: model and prompt_hash filters in sidebar (Database source); "Compare runs" view with table and CSV export; "Export report (JSON)" for single run. Packaging: `pyproject.toml` has optional extras `[dashboard]`, `[datadog]`, `[prometheus]`, `[all]`; README documents install. Docker/production guide deferred.

**Exit criteria:** ✅ Dashboard list/filter/detail/compare + export; ✅ package installs with extras; ✅ production guide (D3).

---

## Library API (Streaming + Post-Run)

**Goal:** Embeddable `CoreVitalMonitor` that wraps generation, exposes streaming events (per-step metrics, risk, warnings) and post-run summary (risk_score, health_flags, narrative). Async/streaming for runtime; post-run for full metrics that need complete data.

### L1. CoreVitalMonitor class

**Location:** `src/CoreVital/monitor.py`.

**Constructor:** `CoreVitalMonitor(capture_mode="summary", risk_threshold=0.7, enable_early_warning=True, intervene_on_risk_above=0.8, ...)`.

**Methods:**
- `wrap_generation(model, prompt, **generate_kwargs)`: Context manager that runs `model.generate` with instrumentation (reuse InstrumentationCollector logic), yields control to caller. After each step, update internal state and optionally call `on_step_callback(step_event)`.
- `stream(model, prompt, **generate_kwargs)`: Async iterator that yields per-step events: `{step_index, token_id, token_text, entropy, surprisal, risk_so_far, warning_signals}`. Consumes generation in a thread or async wrapper so that each step is emitted as soon as it’s available.
- `get_risk_score() -> float`: Available after run; from post-run summary.
- `get_health_flags() -> dict`: After run.
- `get_summary() -> dict`: Full summary (risk, flags, fingerprint, narrative if Phase-7).
- `should_intervene() -> bool`: During run; True if running risk or warning signals exceed thresholds.

**How to (streaming):**
1. Reuse `InstrumentationCollector`’s step loop: run generation with `output_hidden_states=True`, etc., and after each step compute logits summary (entropy, surprisal), update a small buffer (last 5 steps’ entropy, last 2 hidden states for cosine). Compute running risk and warning signals from these; do not store full layer data in streaming path.
2. Emit event dict; if callback provided, call it; if async iterator, yield.
3. After loop, run report_builder in "summary" mode on collected data to get final risk_score, health_flags, and optional narrative. Expose via `get_summary()`.

**Implemented:** `src/CoreVital/monitor.py`: `CoreVitalMonitor(capture_mode, risk_threshold, intervene_on_risk_above, ...)`. `run(model_id, prompt, **kwargs)` runs InstrumentationCollector + ReportBuilder and stores report. `wrap_generation(model_id, prompt, **kwargs)` context manager runs and yields self. `get_risk_score()`, `get_health_flags()`, `get_summary()`, `should_intervene()` (post-run). Async stream: `async def stream(...)` runs the instrumented generation in a thread, then replays the built `report.timeline` as an async iterator of per-step events `{step_index, token_id, token_text, entropy, surprisal, risk_so_far, warning_signals}` (v1 = post-run replay). Exported from `CoreVital`. Tests in `tests/test_monitor.py`.

**Exit criteria:** ✅ wrap_generation + get_summary + should_intervene (post-run); ✅ async stream (post-run replay).

---

## Integrations (OpenTelemetry, Langfuse)

**Goal:** Export CoreVital metrics/traces to OpenTelemetry so that Langfuse, OpenLIT, or any OTLP backend can consume them.

**Research:** OpenTelemetry Python SDK: TracerProvider, SpanProcessor, BatchSpanProcessor, OTLP exporter (opentelemetry.io/docs/languages/python/instrumentation, readthedocs opentelemetry.sdk.metrics.export). Custom metrics: implement `MetricExporter` or use OTLP metric exporter.

**How to:**
1. **Optional dependency:** `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp` (or console for testing). Add to `pyproject.toml` extra `[otel]`.
2. **New file:** `integrations/opentelemetry.py`. Functions: `export_run_to_otel(report: Report, tracer, meter)` — create a span per run, add attributes (model_id, risk_score, health flags); record metrics (gauge risk_score, count high_entropy_steps). Use standard OTLP export so Collector or Langfuse can receive.
3. **Config:** `export_otel: bool = False`, `otel_endpoint: Optional[str] = None`. When True, after sink.write() call export_run_to_otel().
4. **Docs:** `docs/integrations.md`: "Using CoreVital with Langfuse/OpenLIT" — set env or config for OTLP endpoint; link to Langfuse/OpenLIT docs for receiving OTLP.

**Files:** `integrations/opentelemetry.py`, `config.py`, `cli.py` or `report_builder.py` (call export), `docs/integrations.md`, `pyproject.toml`.

**Implemented:** `pyproject.toml` extra `[otel]`; `src/CoreVital/integrations/opentelemetry.py` (`get_otel_tracer_meter`, `export_run_to_otel`); `OtelConfig` in `config.py`; CLI `--export-otel`, `--otel-endpoint`; after `sink.write(report)` call export when `config.otel.export_otel`; `docs/integrations.md`.

**Exit criteria:** ✅ With export_otel=True, a run produces OTLP traces/metrics; doc explains how to connect to Langfuse/OpenLIT.

---

## Documentation and Positioning

**Goal:** README and docs that explain "why CoreVital exists," what works today, and how to run in production; plus a demo.

| # | What | Files | Status |
|---|------|-------|--------|
| D1 | "Why CoreVital exists" | `README.md` | ✅ |
| D2 | "What works today" vs "Planned" | `README.md` | ✅ (Roadmap section) |
| D3 | Production deployment guide | `docs/production-deployment.md` | ✅ |
| D4 | Demo (pre-generated report + screenshot) | `README.md`, optional `docs/demo/` | ✅ |
| D5 | Case study template | `docs/case-studies/` | ✅ |

**How to (D1):** Add a short section at top: problem (LLM outputs fail; debugging internal behavior is hard); solution (instrument inference, summarize tensors, health flags, risk, narratives); value (debug faster, monitor production, compare models).  
**How to (D3):** Sampling strategy, DB setup, metrics export (Prometheus/Datadog), alerting on risk or flags, optional Docker/K8s.  
**How to (D4):** Generate one report (e.g. Llama 3) and commit a small JSON or link to hosted dashboard screenshot; "Try CoreVital in 5 minutes" steps.

**Exit criteria:** New visitor understands why and what; production guide is actionable; demo is linked.

---

## Implementation Order (Summary)

1. **Foundation:** F1 (DB default, migrate), F2 (capture_mode), F3 (RAG metadata), F4 (production models).
2. **Phase-2:** Risk score + layer blame.
3. **Phase-3:** Prompt fingerprints.
4. **Phase-4:** Early warning (post-run + streaming).
5. **Phase-5:** Health-aware decoding API.
6. **Phase-7:** Narratives (template-based).
7. **Phase-6:** Cross-model comparison (dashboard + optional CLI).
8. **Phase-8:** Dashboard polish + packaging.
9. **Library API:** CoreVitalMonitor (streaming + post-run) — can start after F2 and P2.
10. **Integrations:** OpenTelemetry after Library API.
11. **Documentation:** D1–D5 in parallel and before/after each phase.

---

## Rules (apply to all phases)

1. **Backward compatibility:** Existing report schema and CLI flags remain valid; new fields are additive (extensions or optional fields).
2. **Tests before merge:** New code has unit and/or integration tests; `pytest` and `ruff` pass.
3. **No serialization leaks:** Transient buffers and raw tensors are never stored in reports; only summaries and scalars.
4. **Vectorized ops:** Tensor math uses PyTorch ops; avoid Python loops over elements where possible.
5. **Document as you go:** Update this plan with status (⬜→🔄→✅) and any design decisions or post-merge fixes.

---

## Tensor summarization (current vs optional extensions)

**Current (Phase-1):** Hidden states → mean, std, l2_norm_mean, max_abs, optional sketch (random projection). Attention → entropy mean/min/max, concentration max/min, collapsed/focused head counts. Logits → entropy, perplexity, surprisal, top-k, margins. No change required for Phase-2–8.

**Optional extensions (later):** For earlier warning and richer blame: (a) **temporal:** entropy rate (change per step) or variance over steps; (b) **layer correlation:** which layers co-vary (collapse together); (c) **variance metrics:** e.g. entropy_variance per layer across steps. These can be added in Phase-2 or Phase-4 without changing the core summarization pipeline; implement when needed.

---

## References (Cited in this document)

- **Hallucination / internal states:** Nature 2024 (semantic entropy); OpenReview INSIDE (EigenScore, internal states); arxiv 2403.06448 (MIND).
- **Layer attribution:** Achtibat et al., AttnLRP, MLR 2024; Ali et al., Conservative Propagation, MLR 2022.
- **Prompt clustering:** ACL 2025 EMNLP Industry (controllable clustering); ClusterLLM.
- **Early warning / repetition:** arxiv 2512.20578 (Gnosis); arxiv 2504.01100, 2504.14218 (repetition mechanisms).
- **OpenTelemetry:** opentelemetry.io/docs/languages/python/instrumentation; OpenTelemetry Python SDK (TraceProvider, MetricExporter, OTLP).

---

*Last updated: Execution plan created. Implementation status to be updated as phases complete.*
