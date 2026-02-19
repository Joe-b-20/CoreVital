# CoreVital — agent/LLM manifest

Dense reference only. Human README: README.md. (README points AIs here.)

---

## 1. Identity & scope

- **What:** LLM inference health monitor. Hooks HF Transformers forward; computes summary stats (no raw tensor storage); outputs structured report (risk 0–1, health flags, timeline metrics, prompt telemetry).
- **Output:** Report = schema 0.4.0, persisted to JSON/SQLite/Datadog/Prometheus/W&B/OTLP. Dashboard = Streamlit (dashboard.py).
- **Python:** 3.12+. Core: torch, transformers, numpy, pyyaml, pydantic. Optional extras: dashboard, datadog, prometheus, wandb, otel, quantization (pyproject.toml).
- **License:** Apache-2.0. Status: Beta.

---

## 2. Entry points

- **CLI:** `corevital` or `python -m CoreVital.cli` → subcommands: `run`, `migrate`, `compare`. Entry: `src/CoreVital/cli.py` → `main()`, `create_parser()`, `run_command()`.
- **Library:** `from CoreVital import CoreVitalMonitor`; `monitor.run(model_id, prompt, ...)`, `get_risk_score()`, `get_health_flags()`, `wrap_generation()`, `stream()`. Entry: `src/CoreVital/monitor.py`.
- **Dashboard:** `streamlit run dashboard.py` (root). Requires `CoreVital[dashboard]`.

---

## 3. Repo layout (where to edit)

```
src/CoreVital/
  cli.py              # CLI; run_command: config → collector.run() → ReportBuilder.build() → sink.write()
  config.py           # Config, SinkConfig, ModelConfig, SummariesConfig; load_model_profile()
  monitor.py          # CoreVitalMonitor (run, wrap_generation, stream)
  risk.py             # compute_risk_score(), compute_layer_blame()
  narrative.py        # build_narrative()
  fingerprint.py      # compute_fingerprint_vector()
  early_warning.py    # early warning signals
  errors.py           # CoreVitalError, SinkError, ValidationError
  backends/
    base.py           # Backend (ABC), BackendCapabilities; run(config, prompt, monitor) → InstrumentationResults
    huggingface.py   # HuggingFaceBackend (default path; full hidden_states/attentions/prompt_telemetry)
    vllm_backend.py  # VLLMBackend (stub); llama_cpp_backend.py, tgi_backend.py (stubs)
  instrumentation/
    collector.py      # InstrumentationCollector(config, backend=None).run(prompt) → InstrumentationResults; delegates to backend or _run_impl (HF)
    hooks.py          # register hooks, capture hidden/attn/logits
    summaries.py      # compute_logits_summary, compute_attention_summary, extract_sparse_attention,
                      # compute_basin_scores, compute_prompt_surprisal, detect_repetition_loop,
                      # detect_mid_layer_anomaly, compute_hidden_summary
    performance.py    # perf timing
  reporting/
    schema.py         # Report, TimelineStep, HealthFlags, PromptAnalysis, etc. (Pydantic)
    report_builder.py # ReportBuilder.build(results, prompt) → Report
    validation.py    # validate_report(report); supported_versions 0.3.0, 0.4.0
    attention_queries.py # get_attention_to_token, get_attention_from_token, get_top_connections, get_basin_anomalies
  sinks/
    base.py           # Sink (ABC), write(report)->str
    sqlite_sink.py    # default DB; list_traces(); compress blob
    local_file.py     # JSON per run
    datadog_sink.py   # metrics API
    prometheus_sink.py
    wandb_sink.py
    http_sink.py      # base for POST
  models/
    registry.py       # ModelCapabilities, model type detection
    hf_loader.py      # load HF model
  integrations/
    opentelemetry.py  # export_run_to_otel
  utils/
    serialization.py  # serialize_report_to_json
tests/                # pytest; test_*.py; markers: slow, gpu
configs/              # default.yaml; model_profiles/*.yaml (gpt2, llama, default, ...)
docs/                 # model-compatibility.md, metrics-interpretation.md, visual-examples.md, etc.
dashboard.py          # Streamlit; load_report(); plotly optional
.github/workflows/    # test.yaml (lint, typecheck, pytest)
```

---

## 4. Data flow (pin down logic)

- **End-to-end:** prompt → `InstrumentationCollector(config, backend?).run(prompt)` → (if backend set, `backend.run(config, prompt)` else built-in HF `_run_impl`) → `InstrumentationResults` → `ReportBuilder.build(results, prompt)` → `Report` → `sink.write(report)`.
- **risk_score:** `risk.compute_risk_score(health_flags, summary)`; called in `report_builder.build()`; result in `report.extensions["risk"]["risk_score"]`.
- **health_flags:** `report_builder._build_health_flags()` (report_builder.py ~L757) aggregates: nan_detected/inf_detected from layer anomalies; attention_collapse_detected from timeline attention summaries; high_entropy_steps from timeline logits; repetition_loop_detected = `summaries.detect_repetition_loop(hidden_state_buffer)`; mid_layer_anomaly_detected = `summaries.detect_mid_layer_anomaly(layers_to_aggregate, num_layers)`. Buffer for repetition is built inside _build_health_flags from last-layer hidden states.
- **timeline:** `report_builder._build_timeline()` from collector results; each step = logits_summary (summaries.compute_logits_summary), layers[] (compute_hidden_summary, compute_attention_summary). Raw data from collector hooks.
- **prompt_analysis:** `report_builder._build_prompt_analysis()` (report_builder.py ~L676) from `results.prompt_forward_data` (collector._run_prompt_forward): layer_transformations, prompt_surprisals (summaries.compute_prompt_surprisal), layers[].heads (extract_sparse_attention), layers[].basin_scores (compute_basin_scores).
- **Model load:** models/hf_loader.py; registry in models/registry.py. Config in config.py + configs/default.yaml; per-model thresholds in configs/model_profiles/*.yaml (keyed by architecture string).

---

## 5. “Where is X?” quick map

| What | Where (file → function or flow) |
|------|----------------------------------|
| Risk score | risk.py → compute_risk_score; report_builder.build() calls it, stores in report.extensions["risk"] |
| Health flags | report_builder.py → _build_health_flags; uses summaries.detect_repetition_loop, detect_mid_layer_anomaly; timeline logits for high_entropy; attention for collapse |
| Timeline steps | report_builder.py → _build_timeline; data from collector; logits_summary = compute_logits_summary; layers = compute_hidden_summary, compute_attention_summary |
| Prompt analysis | report_builder.py → _build_prompt_analysis; data from results.prompt_forward_data (collector); compute_prompt_surprisal, extract_sparse_attention, compute_basin_scores |
| Repetition detection | summaries.py → detect_repetition_loop (cosine sim > 0.9995, 3+ consecutive) |
| Mid-layer anomaly | summaries.py → detect_mid_layer_anomaly (NaN/Inf or L2 > 8× baseline in middle third layers) |
| Logits metrics | summaries.py → compute_logits_summary (entropy, perplexity, surprisal, top_k_margin, voter_agreement) |
| Attention metrics | summaries.py → compute_attention_summary (entropy_mean/min/max, concentration, collapsed/focused counts) |
| Sink selection | cli.py run_command: --sink → config.sink.type → instantiate SQLiteSink | LocalFileSink | DatadogSink | PrometheusSink | WandBSink |
| Config load | config.py Config.from_yaml / from_default; env COREVITAL_<SECTION>_<KEY> override |
| Model profile | config.py load_model_profile(architecture); configs/model_profiles/<key>.yaml (gpt2, llama, mistral, t5, bart, default) |
| Narrative | narrative.py → build_narrative(); report_builder puts in report (or extensions); human-readable summary of risk/flags |
| Fingerprint | fingerprint.py → compute_fingerprint_vector(); report_builder; used for comparison (extensions / compare view) |
| Backends | backends/base.py → Backend (ABC), BackendCapabilities; HuggingFaceBackend (default); VLLMBackend, LlamaCppBackend, TGIBackend stubs in backends/*. Collector accepts backend= in __init__; run() delegates to backend.run() or _run_impl. |
| Beam search | generation.num_beams > 1 (config / --num_beams); CausalLM only. collector passes num_beams/early_stopping to generate(); _process_timeline uses beam_indices to index scores/hidden/attn for best beam. |
| Real-time intervention | Seq2Seq only: collector.run(prompt, step_callback=...) — callback(step, generated_ids, last_layer_hidden_buffer, last_logits) -> bool; return True to stop. Use with summaries.detect_repetition_loop(buffer). CausalLM uses model.generate() (no per-step hook). |
| Risk calibration | docs/risk-calibration.md; thresholds heuristic; ECE/benchmark validation planned. |
| GPU benchmarks | docs/gpu-benchmarks.md; measure with --perf strict; report.extensions["performance"] has baseline_ms, inference_overhead_pct. |

---

## 6. Commands (validate / CI)

- **Install (editable):** `pip install -e ".[dev]"`. Optional: `[dashboard]`, `[wandb]`, etc.
- **Lint:** `ruff check src/ tests/` ; `ruff format --check src/ tests/` (fix: `ruff format src/ tests/`).
- **Typecheck:** `mypy src/` (or `mypy src/CoreVital/ --ignore-missing-imports --warn-return-any --warn-unused-configs`).
- **Tests:** `pytest tests/ -m 'not slow'` (fast, default). Exclude gpu: `pytest tests/ -m "not gpu"`. Slow: `pytest tests/test_smoke_gpt2_cpu.py tests/test_integration.py` or `pytest -m slow`. Production models: `pytest -m slow` (test_models_production.py).
- **CI:** .github/workflows/test.yaml → ruff check, ruff format --check, mypy, pytest (not gpu). Install: `pip install -e ".[dev]"` (no PYTHONPATH).

---

## 7. Config & environment

- **Config files:** configs/default.yaml (model, device, generation, summaries, sink, prompt_telemetry, etc.). Per-model: configs/model_profiles/<key>.yaml (l2_explosion_multiplier, high_entropy_threshold_bits, repetition_cosine_threshold, collapsed_head_entropy_threshold, focused_head_concentration_threshold). Key from architecture: GPT2*→gpt2, Llama*→llama, Mistral*→mistral, T5*→t5, Bart*→bart, else default.
- **Env override:** COREVITAL_<SECTION>_<KEY>=value (e.g. COREVITAL_DEVICE_REQUESTED=cuda, COREVITAL_PROMPT_TELEMETRY_ENABLED=0). See config.py from_env logic.
- **Sink/API env:** DD_API_KEY, DD_SITE (Datadog); WANDB_PROJECT, WANDB_ENTITY (W&B); OTEL_EXPORTER_OTLP_ENDPOINT (OTLP). CLI also accepts --datadog_api_key, --wandb_project, --wandb_entity, --otel-endpoint.

---

## 8. Conventions (codebase)

- **Style:** Ruff (E, F, I, W, B); line-length 120; isort known-first-party CoreVital. dashboard.py: E501, W291 ignored.
- **Types:** Mypy on src/; warn_return_any, warn_unused_configs. Return types on public APIs.
- **Schema:** reporting/schema.py. Default schema_version 0.4.0; validation accepts 0.3.0 and 0.4.0.
- **Errors:** CoreVitalError, SinkError, ValidationError in errors.py; raise with details for debugging.

---

## 9. Use cases (how to use the repo)

- **Run one monitored generation:** `corevital run --model gpt2 --prompt "Hello" --max_new_tokens 5` → report in runs/corevital.db (or --sink local for JSON). Optional: --device cuda, --quantize-4, --perf, --capture summary|full|on_risk.
- **Use in Python:** `CoreVitalMonitor(capture_mode="summary").run("gpt2", "Hello", max_new_tokens=5)` then `get_risk_score()`, `get_health_flags()`, `should_intervene()`. Or `wrap_generation()` context manager.
- **Dashboard:** `streamlit run dashboard.py`; sidebar: load from runs/, DB, or upload JSON. Uses load_report(path) or SQLiteSink.load_report(db_path, trace_id).
- **Compare runs:** `corevital compare --db runs/corevital.db` (or --db path); lists traces, optional `--prompt-hash` filter. Compare view in dashboard.
- **Migrate JSON to DB:** `corevital migrate --from-dir <dir> --to-db <path>`.
- **Export to W&B/Datadog/OTLP:** --sink wandb (and WANDB_* or --wandb_project/entity); --sink datadog (DD_API_KEY); --export-otel (OTEL_EXPORTER_OTLP_ENDPOINT).

---

## 10. Contributing (how to contribute)

- **Setup:** Clone; `pip install -e ".[dev]"`. Optional conda env (e.g. llm_hm) for local runs.
- **Before PR:** Run `ruff check src/ tests/ && ruff format --check src/ tests/ && mypy src/ && pytest tests/ -m 'not slow'`. Fix format with `ruff format src/ tests/`.
- **Tests:** Add under tests/; name test_*.py, class Test*, function test_*. Use pytest fixtures (conftest.py); mock external deps (e.g. sinks). Unit tests for summaries: tests/test_summaries.py. Integration: tests/test_integration.py. Smoke: tests/test_smoke_gpt2_cpu.py.
- **Add a sink:** Implement Sink in src/CoreVital/sinks/ (write(report)->str); register in cli.py (--sink X, instantiate); export in sinks/__init__.py. Optional: add to pyproject.toml optional-dependencies and config.py SinkConfig.
- **Add a metric:** Implement in instrumentation/summaries.py if summary stat; call from report_builder.py; add field to schema if stored in Report.
- **Change schema:** reporting/schema.py; if breaking, bump schema_version and add to validation supported_versions for backward compat; update report_builder default.

---

## 11. Due diligence (hiring / evaluation)

- **License:** Apache-2.0.
- **Python:** 3.12 only (requires-python >=3.12).
- **Test scope:** pytest; markers `slow` (model load), `gpu` (CUDA); default run excludes slow. test_smoke_gpt2_cpu.py = full CPU pipeline; test_integration.py = pipeline + schema + dashboard load; test_summaries.py = unit tests for summary functions; test_sinks.py = sink logic and CLI.
- **CI:** GitHub Actions on push/PR; lint (Ruff), typecheck (MyPy), pytest (not gpu). No coverage gate in manifest; coverage reported in CI.
- **Maturity:** Beta; schema 0.4.0; backward compat 0.3.0 for load. HF Transformers only (no vLLM/TGI/llama.cpp).

---

## 12. Debugging & common issues

- **Logging:** --log_level DEBUG (cli) or set logging in config. Loggers via CoreVital.logging_utils.get_logger.
- **No attention weights:** Models with SDPA/Flash often don’t return attention; use attn_implementation="eager" when loading (see docs/model-compatibility.md). CoreVital may switch automatically for Llama.
- **Import errors for optional sinks:** Datadog/W&B/etc. are optional; install e.g. pip install CoreVital[datadog]. Sinks raise ImportError with install hint if used without deps.
- **Report validation fails:** Check schema_version in supported_versions (validation.py); required fields (trace_id, model.hf_id, prompt.text, etc.). See reporting/validation.py.

---

## 13. Key abstractions

- **Sink:** write(report: Report) -> str. Implementations: SQLiteSink, LocalFileSink, DatadogSink, PrometheusSink, WandBSink, HTTPSink (base).
- **Report:** Pydantic; trace_id, model, prompt, generated, timeline[], summary, health_flags, prompt_analysis, extensions (risk, performance, …).
- **Timeline step:** step_index, token, logits_summary (entropy, perplexity, surprisal, top_k_margin, voter_agreement), layers[] (hidden_summary, attention_summary).
- **Prompt analysis:** layers[] (basin_scores, heads with query_indices, key_indices, weights); layer_transformations; prompt_surprisals.

---

## 14. Docs map (humans + deep dives)

- **README.md** — human intro, quick start, CLI, library API, roadmap.
- **docs/v0.4.0-launch.md** — technical launch post: what CoreVital is, v0.4.0 features, how to try it.
- **docs/model-compatibility.md** — HF only; CausalLM vs Seq2Seq; SDPA vs eager; model_profiles; limitations (no vLLM/TGI/llama.cpp).
- **docs/metrics-interpretation.md** — metric definitions, thresholds, citations (Shannon, Voita, Attention Basin).
- **docs/visual-examples.md** — good vs bad runs in dashboard.
- **docs/audit-existing-features.md** — verification notes for roadmap items.
- **docs/integrations.md** — OTLP, Langfuse, etc.
- **docs/production-deployment.md** — deployment, sinks, env.

---

## 15. One-shot “help a user” checklist

- **Explain the repo:** Sections 1–2, 3 (layout), 4 (data flow), 13 (abstractions).
- **Run / use:** Section 9 (use cases); CLI and library entry points in 2.
- **Find logic:** Section 5 (“Where is X?”); section 4 (data flow).
- **Contribute:** Section 10 (contributing); section 6 (commands); section 8 (conventions).
- **Evaluate (e.g. hiring):** Section 11 (due diligence); section 1 (identity, license).
- **Debug:** Section 12 (debugging); section 7 (config/env).
- **Extend:** Section 10 (add sink, metric, schema); section 5 for where to wire.
- **Deploy in production:** Section 16.
- **Tune thresholds:** Section 17.
- **User says “model repeats” / “too many alerts” / “production cost”:** Section 18.

---

## 16. Deploy in production

- **Capture mode:** `--capture summary` = minimal (health flags, timeline scalars, prompt scalars; no per-layer). `--capture full` = full trace. `--capture on_risk` = summary until risk/flag triggers then full. Use summary for high volume; see docs/production-deployment.md.
- **Sampling:** CLI does not sample. In app: run CoreVital on a subset of requests (e.g. 1% or every Nth) via CoreVitalMonitor.
- **SQLite:** Set `--out /path` so DB is at `<out>/corevital.db`. Back up DB (cron, sqlite3 .backup). No auto-retention: delete old rows (e.g. `DELETE FROM reports WHERE created_at_utc < ?`) yourself.
- **Metrics export:** `--sink prometheus` + `--prometheus_port 9091` (scrape /metrics); or `--sink datadog` (DD_API_KEY); or `--export-otel` (OTEL_EXPORTER_OTLP_ENDPOINT). One process per instance or sidecar.
- **Alerting:** Alert on risk_score > threshold (e.g. 0.7) and on health flags (nan_detected, repetition_loop_detected, etc.). Metric names in sinks (e.g. corevital_risk_score, corevital_health_*).
- **Docker/K8s:** No built-in Dockerfile/Helm. Use volume for --out; expose Prometheus port if used; Secrets for API keys; env for OTEL endpoint. See docs/production-deployment.md checklist.

---

## 17. Tune thresholds

- **Where:** configs/model_profiles/*.yaml. Key = architecture mapping: GPT2*→gpt2, Llama*→llama, Mistral*→mistral, T5*→t5, Bart*→bart, else default. File names: default.yaml, gpt2.yaml, llama.yaml, etc.
- **Keys (all optional in YAML):** l2_explosion_multiplier (default 8.0), high_entropy_threshold_bits (4.0), repetition_cosine_threshold (0.9995), collapsed_head_entropy_threshold (0.1), focused_head_concentration_threshold (0.9). See configs/model_profiles/default.yaml.
- **Meaning:** l2 = mid-layer anomaly L2 vs early-layer baseline; high_entropy = step counts as “high entropy”; repetition = cosine sim threshold for “same” hidden state; collapsed = head entropy below = collapsed; focused = max attention above = focused. Details: docs/metrics-interpretation.md.
- **Override in code:** Set config.model_profile to a loaded profile (e.g. from your own YAML) to bypass auto-load by architecture. Or edit configs/model_profiles/<key>.yaml and restart.
- **Basin threshold:** Used in dashboard and get_basin_anomalies(); not in model_profiles; pass threshold to get_basin_anomalies(layers, threshold=0.3).

---

## 18. User intent → action & errors

- **“Model repeats” / “stuck in a loop”:** repetition_loop_detected; logic in summaries.detect_repetition_loop. Suggest: shorten max_new_tokens, change prompt, or use should_intervene() to resample. Tune: repetition_cosine_threshold in model_profiles (lower = stricter).
- **“Weird middle layers” / “mid-layer anomaly”:** detect_mid_layer_anomaly; L2 explosion or NaN/Inf in middle third. Suggest: check inputs/precision; tune l2_explosion_multiplier in model_profiles.
- **“Too many false positives” (e.g. attention collapse):** Tune collapsed_head_entropy_threshold, focused_head_concentration_threshold, or high_entropy_threshold_bits in model_profiles for that architecture.
- **“Production too expensive”:** Use --capture summary; sample (run CoreVital on subset of requests in app). See §16.
- **“Where do I alert?”:** risk_score and health flags; export to Prometheus/Datadog/OTLP and alert there. See §16.
- **Errors (where raised):** ConfigurationError (cli.py: missing API key, unknown sink). SinkError (sinks/*.py: write/upload failed). ValidationError (reporting/validation.py: invalid report). All in errors.py; inherit CoreVitalError. Use to suggest: check env (DD_API_KEY, etc.), check report schema_version, check sink deps.

End of manifest.
