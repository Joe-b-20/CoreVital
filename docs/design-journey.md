# Design Journey

How CoreVital went from "I need to see inside my model" to a production-ready instrumentation toolkit. This document captures the key architectural decisions, trade-offs, and lessons learned across eight development phases.

---

## The Problem

LLM inference fails in ways that are invisible from the outside. A model can produce repetitive text, nonsensical output, or subtly wrong answers — and the only signal you get from the standard `generate()` API is the final token sequence. You can't see that attention collapsed in layer 7, or that entropy spiked at step 15, or that hidden state norms exploded in the middle layers.

Existing observability tools (LangSmith, Langfuse, OpenLIT) work at the API level — they trace prompts, responses, latency, and cost. None of them instrument the model's forward pass. CoreVital was built to fill that gap.

---

## Phase 0: Can We Hook Into the Forward Pass?

**Question:** Is it feasible to capture hidden states, attention weights, and logits during generation without modifying the model itself?

**Decision:** Use PyTorch forward hooks to intercept tensors at each layer during the model's forward pass. This approach is non-invasive (no model code changes), works with any Hugging Face transformer, and can be toggled on/off per run.

**Trade-off:** Hugging Face's `generate()` method doesn't return internal tensors for Seq2Seq models. Rather than limiting support to CausalLM-only, we implemented manual token-by-token generation for Seq2Seq, calling the model's forward method directly in a loop. More code, but universal model support.

**Key decision: summary-only storage.** Storing raw tensors (hidden states are `[batch, seq_len, hidden_dim]` per layer per step) would produce gigabyte-scale traces for a single run. Instead, we compute lightweight summaries in-memory — mean, std, L2 norm, entropy, concentration — and discard the raw tensors immediately. This keeps traces at 200KB–5MB while preserving the diagnostic signal.

---

## Phase 0.5: Hardening the Schema

**Question:** How do we structure the output so it's stable across versions and useful for downstream tools?

**Decision:** Pydantic-validated schema with explicit versioning (`schema_version: "0.3.0"`). Every report field is typed and documented. Optional `extensions` dicts on Report, TimelineStep, and LayerSummary allow future metrics without breaking the schema.

**Lesson learned:** The schema evolved twice (0.1.0 to 0.2.0 to 0.3.0) before stabilizing. Breaking changes early (removing deprecated fields, restructuring encoder/decoder separation) were worth the pain — they prevented confusion later when health flags and risk scores needed clean extension points.

---

## Phase 0.75: Measuring Our Own Overhead

**Question:** How much does instrumentation cost, and where does the time go?

**Decision:** Built a `--perf` flag with three modes:
- **Summary:** Total wall time and breakdown by parent operation (config, model load, inference, report build).
- **Detailed:** Nested child operations and per-step statistics.
- **Strict:** Adds warmup passes and uninstrumented baseline to isolate CoreVital's overhead from model inference time.

**Finding:** For small models (GPT-2), overhead is 2–5%. For medium models (Llama-3.1-8B), report building dominates at ~15–20% of wall time because per-layer summary computation scales as O(steps × layers × heads). The inference hooks themselves add under 2%.

**Trade-off:** We could reduce report-building overhead by computing summaries lazily or in a background thread, but that complicates the architecture. For production use, `--capture summary` skips per-layer data entirely, reducing overhead to near-zero.

---

## Phase 1: The Metrics That Matter

**Phase 1a — Schema v0.3.0 and enhanced metrics.** Added per-step perplexity, surprisal, top-k margins, voter agreement, and per-layer collapsed/focused head counts. These metrics were chosen based on the ML literature (see `docs/Phase1 metrics analysis.md`) — each has a documented interpretation and a "what to do when this is abnormal" playbook.

**Phase 1b — Prompt telemetry.** An extra forward pass over the prompt (before generation) captures how the model processes the input: sparse attention profiles, basin scores (does the model attend to the middle of the prompt?), layer transformations (how much each layer changes the representation), and per-token surprisal.

**Key decision: sparse attention storage.** Full attention matrices for 12 layers × 12 heads × N² positions would be enormous. We store only attention weights above a threshold (default 0.01), using a Structure-of-Arrays format (query_indices, key_indices, weights). This achieves ~680× compression compared to the naive approach while preserving the diagnostically interesting connections.

**Phase 1c — Health flags.** Aggregated boolean indicators: NaN/Inf detected, attention collapse, high entropy steps, repetition loop, mid-layer anomaly. These are the "glanceable" signals — you look at health flags first, then drill into the timeline if something is flagged.

**Design choice: transient buffer for repetition detection.** Repetition loops are detected by comparing cosine similarity of last-layer hidden states across consecutive steps. But we don't store all past hidden states — we keep a rolling buffer of the last 3 steps and check for similarity above 0.9995. This catches repetition without accumulating memory.

**Phase 1d — Dashboard and SQLite default.** The Streamlit dashboard visualizes entropy/perplexity/surprisal curves, attention heatmaps, health flags, prompt analysis, and performance breakdowns. SQLite replaced JSON files as the default sink — one DB per project directory is cleaner than hundreds of JSON files, and it enables the Compare view.

---

## Phases 2–8: From Metrics to Actionable Intelligence

**Phase 2 — Risk scores and layer blame.** A single 0–1 risk score computed from health flags, weighted by severity (NaN/Inf > repetition > mid-layer anomaly > entropy). Layer blame identifies which specific layers contributed to the risk — useful for debugging quantization issues or fine-tuning regressions.

**Phase 3 — Prompt fingerprints.** A 16-dimensional vector summarizing each run's behavior (entropy distribution, risk, token count, attention patterns) plus a prompt hash. Fingerprints enable clustering similar runs and detecting behavioral drift over time.

**Phase 4 — Early warning.** Failure risk score and warning signals (entropy rising, high entropy) derived from the timeline. Unlike the post-hoc risk score, early warning is designed for streaming scenarios where you want to intervene mid-generation.

**Phase 5 — Health-aware decoding.** The `CoreVitalMonitor.should_intervene()` API returns true when risk exceeds a configurable threshold. This enables patterns like "if the model is struggling, switch to a fallback model" or "resample with lower temperature."

**Phase 6 — Cross-model comparison.** Dashboard Compare view and `corevital compare` CLI command. Select two or more runs from the SQLite database and see metrics side-by-side with differences highlighted.

**Phase 7 — Human-readable narratives.** Template-based 2–4 sentence summaries of each run ("This run was low risk. Two steps showed elevated entropy. No repetition detected."). No LLM call — fast, deterministic, and always available.

**Phase 8 — Packaging and library API.** `CoreVitalMonitor` provides an embeddable interface: `run()`, `wrap_generation()` (context manager), and `stream()` (async per-step events). The monitor exposes `get_risk_score()`, `get_health_flags()`, `get_summary()`, and `should_intervene()` for programmatic use. Optional extras for dashboard, Datadog, Prometheus, and OpenTelemetry integration.

---

## Recurring Design Principles

**Compute summaries, not tensors.** Every metric in CoreVital is a lightweight summary of the underlying tensor. We never store raw activations. This keeps storage small, overhead low, and avoids the privacy/security concerns of persisting model internals.

**Pluggable sinks.** The `Sink` interface (one method: `write(report) -> location`) allows swapping persistence backends without changing instrumentation logic. Built-in: SQLite, local JSON, Datadog, Prometheus, HTTP, OpenTelemetry.

**Schema-first development.** Every feature starts with the schema change — define the Pydantic model, the field names, the types, and the semantics. Then implement the computation. This prevents schema drift and ensures the dashboard, CLI, and library API all agree on the data shape.

**Graceful degradation.** Optional dependencies (Plotly, Datadog client, OpenTelemetry SDK) are handled with try/except imports. Missing features show "install X for Y" messages rather than crashes. Quantization falls back to CPU without quantization if CUDA is unavailable.

**Test without models.** The mock testing suite (`tests/conftest.py`) provides mock `ModelBundle` fixtures that return properly shaped tensors. This enables fast CI testing of the full instrumentation-to-report-to-sink pipeline without downloading or loading any model.

---

## What's Next

- **True online streaming:** Currently, `stream()` replays events post-run. Real-time per-step events during generation would enable live dashboards and immediate intervention.
- **Attention pattern clustering:** Grouping runs by attention fingerprint to detect behavioral regimes.
- **Multi-request aggregation:** Dashboard views that summarize health across hundreds of production runs, not just individual traces.
- **Expanded model support:** Vision-language models, mixture-of-experts architectures, and custom model classes beyond Hugging Face.
