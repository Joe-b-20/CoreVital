# CoreVital — agent/LLM manifest

Dense reference only. Human README: README.md. (README points AIs here.)

---

## 1. Identity & scope

- **What:** LLM inference health monitor. Uses HF-output-based capture (no PyTorch hooks): configures HF models to return hidden states, attention weights, and logits at every generation step. Computes summary stats (no raw tensor storage); outputs structured report (risk 0–1, health flags, compound signals, early warning, layer blame, fingerprint, narrative, optional calibration divergence score).
- **Output:** Report = schema 0.4.0, persisted to JSON/SQLite/Datadog/Prometheus/W&B/OTLP. Viewing: hosted React dashboard + local API (corevital serve), or enterprise sinks only (no dashboard).
- **Python:** 3.12+. Core: torch, transformers, numpy, pyyaml, pydantic. Optional extras: serve (FastAPI+uvicorn), datadog, prometheus, wandb, otel, quantization (pyproject.toml).
- **License:** Apache-2.0. Status: v0.5.0-rc. Refactor complete (Phases 0–5, 436 tests).

---

## 2. Entry points

- **CLI:** `corevital` or `python -m CoreVital.cli` → subcommands: `run`, `calibrate`, `serve`, `migrate`, `compare`. Entry: `src/CoreVital/cli.py` → `main()`, `create_parser()`, `run_command()`, `calibrate_command()`, `serve_command()`.
- **Library:** `from CoreVital import CoreVitalMonitor`; `monitor.run(model_id, prompt, ...)`, `get_risk_score()`, `get_health_flags()`, `wrap_generation()`, `stream()`. Entry: `src/CoreVital/monitor.py`.
- **Local API:** `corevital serve` (uvicorn). Requires `CoreVital[serve]`. App: `src/CoreVital/api.py`; GET /api/traces, GET /api/traces/{id}. Used by hosted React dashboard; data stays local.

---

## 3. Repo layout (where to edit)

```
src/CoreVital/
  api.py              # FastAPI app; GET /api/traces, GET /api/traces/{id}; SQLiteSink.list_traces, load_report; CORS allow_origins=["*"]
  cli.py              # CLI; run_command: config → collector.run() → ReportBuilder.build() → sink.write(); calibrate_command → calibrate_from_runs → save profile; serve_command → uvicorn
  config.py           # Config, SinkConfig, ModelConfig, SummariesConfig, ModelProfile (with typical_entropy_range, typical_l2_norm_range), LogitsSummariesConfig (entropy_mode); load_model_profile(); _architecture_to_profile_key (GPT2, Llama, Mixtral, Mistral, Qwen2, Phi3, T5, Bart)
  monitor.py          # CoreVitalMonitor (run, wrap_generation, stream)
  risk.py             # compute_risk_score() (composite: boolean + continuous + compound), compute_risk_score_legacy(), compute_layer_blame() (enriched: List[dict] with layer/reasons/severity), compute_layer_blame_flat() (backward compat: List[int])
  compound_signals.py # CompoundSignal dataclass, detect_compound_signals() (5 patterns: context_loss, confident_confusion, degenerating_generation, attention_bottleneck, confident_repetition_risk)
  early_warning.py    # compute_early_warning() with 5 trend detectors: entropy_accelerating, margin_collapsed, margin_declining, surprisal_volatile, entropy_margin_divergence; health flag passthrough
  narrative.py        # build_narrative() — data-specific 2–6 sentences citing actual values, step indices, token text, compound signals, recommendations
  fingerprint.py      # compute_fingerprint_vector() (25-element v2), compute_prompt_hash(), FINGERPRINT_VERSION=2, FINGERPRINT_LENGTH=25, is_legacy_fingerprint()
  calibration.py      # MetricDistribution, CalibrationProfile (save/load JSON), calibrate_from_runs(), compute_divergence_score()
  calibration_risk.py # compute_ece(), fit_platt_scaling(), apply_platt_scaling(), evaluate_calibration(), RiskCalibrationResult
  errors.py           # CoreVitalError, SinkError, ValidationError, InstrumentationError
  backends/
    base.py           # Backend (ABC), BackendCapabilities; run(config, prompt, monitor) → InstrumentationResults
    huggingface.py   # HuggingFaceBackend (default path; full hidden_states/attentions/prompt_telemetry)
    vllm_backend.py  # VLLMBackend (stub); llama_cpp_backend.py, tgi_backend.py (stubs)
  instrumentation/
    __init__.py       # Exports InstrumentationCollector, StepSummary, etc.
    collector.py      # InstrumentationCollector (~250-line orchestrator). run() → delegates to causal_lm or seq2seq. Per-request torch.Generator (no global seed). Batch-size assertion, prompt-length check, try/finally with cuda.empty_cache.
    causal_lm.py      # run_causal_generation() — wraps model.generate(), token extraction, calls step_processor per step
    seq2seq.py        # run_seq2seq_generation() — manual decoder loop with KV cache, step_processor per step, raw tensors never accumulate
    step_processor.py # NormalizedStepPayload, StepSummary (scalars only), normalize_step_tensors() (strip embedding, slice attn, detach/cpu), process_step() (compute summaries, discard tensors)
    baselines.py      # _resolve_special_token(), _normalize_eos(), _build_logits_processor() (HF LogitsProcessorList), run_warmup(), run_baseline(), run_baseline_causal(), run_baseline_seq2seq(), run_prompt_forward()
    hooks.py          # Unused stub (documented: HF output flags, not PyTorch hooks)
    summaries/        # Subpackage
      __init__.py     # Re-exports everything; backward-compatible imports
      logits.py       # compute_logits_summary (entropy_mode: full|topk_approx), compute_prompt_surprisal (detach/cpu/no_grad), MIN_TOPK_FOR_STATS, VOTER_AGREEMENT_TOP_K
      attention.py    # compute_attention_summary (entropy_mean + entropy_mean_normalized), compute_basin_scores (vectorized), extract_sparse_attention; collapse detection: NORMALIZED_COLLAPSED_THRESHOLD=0.03 (normalized, not raw nats); clamp-based log (not epsilon addition); division normalization (not softmax)
      hidden_states.py # compute_hidden_summary (clip_fraction, clip_max_before when clipped), compute_encoder_hidden_states_summaries, compute_layer_transformations, detect_mid_layer_anomaly (median baseline, not mean), detect_repetition_loop (optional token_id_buffer, consecutive_required), detect_tensor_anomalies, L2_EXPLOSION_MULTIPLIER
      utils.py        # _random_projection_sketch (np.random.default_rng, no global seed)
    performance.py    # perf timing
  reporting/
    schema.py         # Report, TimelineStep, HealthFlags, PromptAnalysis, AttentionSummary (entropy_mean_normalized), HiddenSummary (clip_fraction, clip_max_before), etc. (Pydantic)
    report_builder.py # ReportBuilder.build(results, prompt) → Report; calls risk, compound_signals, early_warning, narrative, fingerprint, calibration, validation
    validation.py    # validate_report(report); validate_metric_consistency() (perplexity=2^entropy, margin≤mass, non-negative, etc.); supported_versions 0.3.0, 0.4.0
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
    registry.py       # ModelCapabilities (attentions_available: Optional[bool]), model type detection
    hf_loader.py      # load HF model; _probe_attentions_available() at load time
  integrations/
    opentelemetry.py  # export_run_to_otel
  utils/
    serialization.py  # serialize_report_to_json
tests/                # pytest; 436 tests; markers: slow, gpu
configs/              # default.yaml; model_profiles/*.yaml (gpt2, llama, mistral, mixtral, qwen2, phi3, default)
docs/                 # metrics-interpretation.md, risk-calibration.md, model-compatibility.md, visual-examples.md, etc.
CHANGELOG.md          # v0.5.0-rc changelog: all phases, schema/config/CLI changes, breaking changes, deprecations
.github/workflows/    # test.yaml (lint, typecheck, pytest)
```

---

## 4. Data flow (pin down logic)

- **End-to-end:** prompt → `InstrumentationCollector(config, backend?).run(prompt)` → (if backend set, `backend.run()` else built-in HF path) → `causal_lm.run_causal_generation()` or `seq2seq.run_seq2seq_generation()` → per-step: `step_processor.normalize_step_tensors()` → `step_processor.process_step()` (compute summaries, discard raw tensors) → `List[StepSummary]` → `InstrumentationResults` → `ReportBuilder.build(results, prompt)` → `Report` → `sink.write(report)`.
- **StepSummary pipeline:** Raw tensors → `normalize_step_tensors` (strip embedding, slice attention to last query, detach/cpu) → `process_step` (logits/attention/hidden summaries per layer, detect anomalies, extract repetition-detection vector) → `StepSummary` (scalars only, `_last_layer_hidden_vec` for repetition). Raw tensors never survive past `process_step()`.
- **risk_score:** `risk.compute_risk_score(health_flags, summary, timeline, layers_by_step, compound_signals)` → composite of boolean ceilings (NaN/Inf=1.0, repetition=0.9, mid-layer=0.7, collapse=0.15) + continuous additive (entropy/margin/mass/surprisal) + compound signal severities; capped at 1.0. Fallback to `compute_risk_score_legacy` when no timeline.
- **compound_signals:** `compound_signals.detect_compound_signals(timeline, layers_by_step, basin_scores)` → List[CompoundSignal]. 5 patterns: context_loss, confident_confusion, degenerating_generation, attention_bottleneck, confident_repetition_risk. Stored in `extensions.compound_signals`, fed into risk score.
- **early_warning:** `early_warning.compute_early_warning(health_flags, timeline, high_entropy_threshold)` → (failure_risk, warning_signals). 5 trend detectors. Stored in `extensions.early_warning`.
- **layer_blame:** `risk.compute_layer_blame(timeline_layers, health_flags)` → List[dict] with {layer, reasons, severity}. 4 conditions: NaN/Inf (1.0), collapse rate >50% (0.4), L2 z-score outlier (0.5), L2 instability CV>0.5 (0.3). `compute_layer_blame_flat()` for backward compat List[int]. Both stored in `extensions.risk`.
- **health_flags:** `report_builder._build_health_flags()` aggregates: nan/inf from layer anomalies; attention_collapse from timeline attention summaries (normalized entropy < 0.03); high_entropy_steps from timeline logits (model profile threshold); repetition_loop = `detect_repetition_loop(hidden_state_buffer)` (optional token_id_buffer cross-check); mid_layer_anomaly = `detect_mid_layer_anomaly(layers, num_layers)` (median baseline).
- **timeline:** `report_builder._build_timeline()` from `List[StepSummary]` (pre-computed summaries via `_build_layers_from_step_summary`); legacy `StepData` path kept.
- **prompt_analysis:** `report_builder._build_prompt_analysis()` from `results.prompt_forward_data`: layer_transformations, prompt_surprisals, layers[].heads (extract_sparse_attention), layers[].basin_scores (compute_basin_scores, vectorized).
- **narrative:** `narrative.build_narrative(health_flags, risk_score, risk_factors, blamed_layers, warning_signals, timeline, compound_signals, summary)` → data-specific text with peak entropy, step index, token text, compound signals (capped at 2), blamed layers, actionable recommendations.
- **fingerprint:** `fingerprint.compute_fingerprint_vector(timeline, health_flags, summary, risk_score, layers_by_step)` → 25-element v2 vector (entropy/margin/surprisal/agreement profiles with slopes, correlations, temporal features). `FINGERPRINT_VERSION=2`.
- **calibration:** When `config.calibration_profile` is set: `CalibrationProfile.load(path)` → `compute_divergence_score(trace, profile)` → (score, anomalies). Stored in `extensions.calibration`.
- **metric_consistency:** When DEBUG logging: `validate_metric_consistency(report)` → warnings list. Stored in `extensions.metric_consistency`.
- **Model load:** models/hf_loader.py (includes `_probe_attentions_available()`); registry in models/registry.py. Config in config.py + configs/default.yaml; per-model thresholds in configs/model_profiles/*.yaml.

---

## 5. "Where is X?" quick map

| What | Where (file → function or flow) |
|------|----------------------------------|
| Risk score | risk.py → compute_risk_score (composite) or compute_risk_score_legacy (boolean only); report_builder.build() calls it, stores in report.extensions["risk"] |
| Risk factors | risk.py → compute_risk_score returns (score, factors); factors include: nan_or_inf, repetition_loop, mid_layer_anomaly, attention_collapse, elevated_entropy, entropy_rising, low_confidence_margin, low_topk_mass, elevated_surprisal, compound:* |
| Compound signals | compound_signals.py → detect_compound_signals; 5 patterns (context_loss, confident_confusion, degenerating_generation, attention_bottleneck, confident_repetition_risk); called from report_builder |
| Early warning | early_warning.py → compute_early_warning; 5 trend detectors; called from report_builder |
| Layer blame (enriched) | risk.py → compute_layer_blame → List[dict] with {layer, reasons, severity}; compute_layer_blame_flat → List[int] |
| Health flags | report_builder.py → _build_health_flags; uses summaries.detect_repetition_loop, detect_mid_layer_anomaly; timeline logits for high_entropy; attention for collapse (normalized threshold 0.03) |
| Timeline steps | report_builder.py → _build_timeline; data from List[StepSummary] (pre-computed in step_processor) |
| StepSummary | step_processor.py → normalize_step_tensors + process_step → StepSummary (scalars only); replaces StepData |
| Prompt analysis | report_builder.py → _build_prompt_analysis; compute_prompt_surprisal, extract_sparse_attention, compute_basin_scores (vectorized) |
| Repetition detection | summaries/hidden_states.py → detect_repetition_loop (cosine sim > threshold, 3+ consecutive); optional token_id_buffer for n-gram cross-check |
| Mid-layer anomaly | summaries/hidden_states.py → detect_mid_layer_anomaly (NaN/Inf or L2 > multiplier × median of early-layer norms) |
| Logits metrics | summaries/logits.py → compute_logits_summary (entropy, perplexity, surprisal, top_k_margin, topk_mass, topk_probs); supports entropy_mode full/topk_approx |
| Attention metrics | summaries/attention.py → compute_attention_summary (entropy_mean, entropy_mean_normalized, entropy_min/max, concentration, collapsed/focused counts); collapse = normalized entropy < 0.03; clamp-based log; division normalization |
| Hidden diagnostics | summaries/hidden_states.py → compute_hidden_summary; clip_fraction and clip_max_before when clipping fires |
| Basin scores | summaries/attention.py → compute_basin_scores (vectorized batched tensor ops) |
| Narrative | narrative.py → build_narrative(); data-specific text with actual entropy values, step indices, token text, compound signals, recommendations |
| Fingerprint | fingerprint.py → compute_fingerprint_vector() (25-element v2); FINGERPRINT_VERSION=2; is_legacy_fingerprint() |
| Calibration profile | calibration.py → CalibrationProfile, calibrate_from_runs(), compute_divergence_score() |
| ECE / Platt scaling | calibration_risk.py → compute_ece(), fit_platt_scaling(), evaluate_calibration() |
| Metric consistency | reporting/validation.py → validate_metric_consistency() (perplexity=2^entropy, margin≤mass, etc.) |
| Sink selection | cli.py run_command: --sink → config.sink.type → instantiate SQLiteSink | LocalFileSink | DatadogSink | PrometheusSink | WandBSink |
| Config load | config.py Config.from_yaml / from_default; env COREVITAL_<SECTION>_<KEY> override |
| Model profile | config.py load_model_profile(architecture); configs/model_profiles/<key>.yaml (gpt2, llama, mistral, mixtral, qwen2, phi3, t5, bart, default) |
| CausalLM generation | instrumentation/causal_lm.py → run_causal_generation(); wraps model.generate() |
| Seq2Seq generation | instrumentation/seq2seq.py → run_seq2seq_generation(); manual decoder loop with KV cache |
| Step processing | instrumentation/step_processor.py → normalize_step_tensors + process_step |
| Baseline/warmup | instrumentation/baselines.py → run_warmup, run_baseline, run_prompt_forward |
| Backends | backends/base.py → Backend (ABC); HuggingFaceBackend (default); VLLMBackend, LlamaCppBackend, TGIBackend stubs |
| Beam search | generation.num_beams > 1; CausalLM only via model.generate(). Raises InstrumentationError if hidden/attention capture enabled with beams > 1 |
| Real-time intervention | Seq2Seq only: collector.run(prompt, step_callback=...) — callback(step, generated_ids, last_layer_hidden_buffer, last_logits) -> bool; return True to stop |
| Risk calibration | docs/risk-calibration.md; calibration.py for profiles; calibration_risk.py for ECE |
| GPU benchmarks | docs/gpu-benchmarks.md; measure with --perf strict; report.extensions["performance"] |

---

## 6. Commands (validate / CI)

- **Install (editable):** `pip install -e ".[dev]"`. Optional: `[serve]`, `[wandb]`, etc.
- **Lint:** `ruff check src/ tests/` ; `ruff format --check src/ tests/` (fix: `ruff format src/ tests/`).
- **Typecheck:** `mypy src/` (or `mypy src/CoreVital/ --ignore-missing-imports --warn-return-any --warn-unused-configs`).
- **Tests:** `pytest tests/ -m 'not slow'` (fast, default; 436 tests). Exclude gpu: `pytest tests/ -m "not gpu"`. Slow: `pytest tests/test_smoke_gpt2_cpu.py tests/test_integration.py` or `pytest -m slow` (includes 6 performance benchmarks). Production models: `pytest -m slow`.
- **CI:** .github/workflows/test.yaml → ruff check, ruff format --check, mypy, pytest (not gpu). Install: `pip install -e ".[dev]"` (no PYTHONPATH).

---

## 7. Config & environment

- **Config files:** configs/default.yaml (model, device, generation, summaries, sink, prompt_telemetry, calibration_profile, etc.). Logits: entropy_mode ("full" or "topk_approx"), topk_mass/topk_probs in stats. Attention: entropy_mean_normalized in stats. Per-model: configs/model_profiles/<key>.yaml (l2_explosion_multiplier, high_entropy_threshold_bits, repetition_cosine_threshold, collapsed_head_entropy_threshold [deprecated for collapse detection], focused_head_concentration_threshold, typical_entropy_range, typical_l2_norm_range). Key from architecture: GPT2*→gpt2, Llama*→llama, Mixtral*→mixtral (before Mistral), Mistral*→mistral, Qwen2*→qwen2, Phi3*/Phi-3*→phi3, T5*→t5, Bart*→bart, else default.
- **Calibration:** `calibration_profile` in config or `--calibration <path>` on CLI. Build with `corevital calibrate --model <id> --prompts <file> --out <path>`.
- **Env override:** COREVITAL_<SECTION>_<KEY>=value (e.g. COREVITAL_DEVICE_REQUESTED=cuda, COREVITAL_PROMPT_TELEMETRY_ENABLED=0). See config.py from_env logic.
- **Sink/API env:** DD_API_KEY, DD_SITE (Datadog); WANDB_PROJECT, WANDB_ENTITY (W&B); OTEL_EXPORTER_OTLP_ENDPOINT (OTLP). CLI also accepts --datadog_api_key, --wandb_project, --wandb_entity, --otel-endpoint.

---

## 8. Conventions (codebase)

- **Style:** Ruff (E, F, I, W, B); line-length 120; isort known-first-party CoreVital.
- **Types:** Mypy on src/; warn_return_any, warn_unused_configs. Return types on public APIs.
- **Schema:** reporting/schema.py. Default schema_version 0.4.0; validation accepts 0.3.0 and 0.4.0.
- **Errors:** CoreVitalError, SinkError, ValidationError, InstrumentationError in errors.py; raise with details for debugging.
- **Tensors:** All tensor operations inside torch.no_grad() unless training. All captured tensors get .detach().cpu() before storage. Never store raw GPU tensors beyond the function that captured them.
- **Randomness:** torch.Generator (not torch.manual_seed). np.random.default_rng (not np.random.seed).
- **Data structures:** dataclasses for new data structures, not dicts.

---

## 9. Use cases (how to use the repo)

- **Run one monitored generation:** `corevital run --model gpt2 --prompt "Hello" --max_new_tokens 5` → report in runs/corevital.db (or --sink local for JSON). Optional: --device cuda, --quantize-4, --perf, --capture summary|full|on_risk, --calibration <path>.
- **Build calibration profile:** `corevital calibrate --model gpt2 --prompts prompts.txt --out calibration/gpt2.json`. Then use: `corevital run --model gpt2 --prompt "..." --calibration calibration/gpt2.json`.
- **Use in Python:** `CoreVitalMonitor(capture_mode="summary").run("gpt2", "Hello", max_new_tokens=5)` then `get_risk_score()`, `get_health_flags()`, `should_intervene()`. Or `wrap_generation()` context manager.
- **Local API:** corevital serve → uvicorn CoreVital.api:app. GET /api/traces (SQLiteSink.list_traces), GET /api/traces/{id} (SQLiteSink.load_report). DB path: COREVITAL_DB_PATH or --db (default runs/corevital.db).
- **Compare runs:** `corevital compare --db runs/corevital.db` (or --db path); lists traces, optional `--prompt-hash` filter. Compare view in dashboard.
- **Migrate JSON to DB:** `corevital migrate --from-dir <dir> --to-db <path>`.
- **Export to W&B/Datadog/OTLP:** --sink wandb (and WANDB_* or --wandb_project/entity); --sink datadog (DD_API_KEY); --export-otel (OTEL_EXPORTER_OTLP_ENDPOINT).
- **Validate risk calibration:** Use calibration_risk.py: `evaluate_calibration(raw_scores, labels)` → ECE, Platt params. See docs/risk-calibration.md for 5-step workflow.

---

## 10. Contributing (how to contribute)

- **Setup:** Clone; `pip install -e ".[dev]"`. Optional conda env (e.g. llm_hm) for local runs.
- **Before PR:** Run `ruff check src/ tests/ && ruff format --check src/ tests/ && mypy src/ && pytest tests/ -m 'not slow'`. Fix format with `ruff format src/ tests/`.
- **Tests:** Add under tests/; name test_*.py, class Test*, function test_*. Use pytest fixtures (conftest.py); mock external deps (e.g. sinks). Unit tests for summaries: tests/test_summaries.py. Integration: tests/test_metric_integration.py. Smoke: tests/test_smoke_gpt2_cpu.py. Risk: test_risk.py. Compound: test_compound_signals.py. Early warning: test_early_warning.py. Narrative: test_narrative.py. Fingerprint: test_fingerprint.py. Calibration: test_calibration.py, test_calibration_risk.py. Metric consistency: test_metric_consistency.py.
- **Add a sink:** Implement Sink in src/CoreVital/sinks/ (write(report)->str); register in cli.py (--sink X, instantiate); export in sinks/__init__.py.
- **Add a metric:** Implement in instrumentation/summaries/<appropriate>.py; call from step_processor.py (per-step) or report_builder.py (aggregate); add field to schema if stored in Report.
- **Add a compound signal:** Add detection logic in compound_signals.py → detect_compound_signals(); it feeds into risk score automatically.
- **Change schema:** reporting/schema.py; if breaking, bump schema_version and add to validation supported_versions for backward compat; update report_builder default.
- **Add a model profile:** Create configs/model_profiles/<key>.yaml; add architecture mapping in config.py _architecture_to_profile_key; add tests in test_model_profiles.py.

---

## 11. Due diligence (hiring / evaluation)

- **License:** Apache-2.0.
- **Python:** 3.12 only (requires-python >=3.12).
- **Test scope:** pytest; 436 tests; markers `slow` (model load, 6 perf benchmarks), `gpu` (CUDA); default run excludes slow. Key test files: test_smoke_gpt2_cpu.py (full CPU pipeline), test_mock_instrumentation.py (mock pipeline), test_summaries.py (summary functions), test_risk.py (risk scoring + layer blame), test_compound_signals.py, test_early_warning.py, test_narrative.py, test_fingerprint.py (45 tests), test_calibration.py (24 tests), test_calibration_risk.py (34 tests), test_metric_integration.py (38 integration tests), test_metric_consistency.py (34 tests), test_model_profiles.py (41 tests).
- **CI:** GitHub Actions on push/PR; lint (Ruff), typecheck (MyPy), pytest (not gpu). No coverage gate in manifest; coverage reported in CI.
- **Maturity:** v0.5.0-rc; schema 0.4.0; backward compat 0.3.0 for load. HF Transformers only (no vLLM/TGI/llama.cpp). CHANGELOG.md covers all changes.

---

## 12. Debugging & common issues

- **Logging:** --log_level DEBUG (cli) or set logging in config. Loggers via CoreVital.logging_utils.get_logger. DEBUG enables metric consistency validation (extensions.metric_consistency).
- **No attention weights:** Models with SDPA/Flash often don't return attention; use attn_implementation="eager" when loading (see docs/model-compatibility.md). CoreVital probes attention availability at load time (`_probe_attentions_available` in hf_loader.py) and logs WARNING if not returned.
- **Beam search + hidden/attention:** Raises InstrumentationError. Use num_beams=1 or disable hidden/attention capture.
- **Import errors for optional sinks:** Datadog/W&B/etc. are optional; install e.g. pip install CoreVital[datadog]. Sinks raise ImportError with install hint if used without deps.
- **Report validation fails:** Check schema_version in supported_versions (validation.py); required fields (trace_id, model.hf_id, prompt.text, etc.). See reporting/validation.py.
- **Fingerprint length mismatch:** Old 9-element vectors vs new 25-element v2. Use `is_legacy_fingerprint()` to detect and handle both.

---

## 13. Key abstractions

- **Sink:** write(report: Report) -> str. Implementations: SQLiteSink, LocalFileSink, DatadogSink, PrometheusSink, WandBSink, HTTPSink (base).
- **Report:** Pydantic; trace_id, model, prompt, generated, timeline[], summary, health_flags, prompt_analysis, extensions (risk, compound_signals, early_warning, fingerprint, narrative, calibration, metric_consistency, performance, …).
- **StepSummary:** Replaces StepData. Scalars only: logits_summary, layer_summaries (List[LayerSummary]), _last_layer_hidden_vec (small 1-D derived CPU vector for repetition detection). No raw tensors.
- **Timeline step:** step_index, token, logits_summary (entropy, perplexity, surprisal, top_k_margin, topk_mass, topk_probs), layers[] (hidden_summary with optional clip_fraction/clip_max_before, attention_summary with entropy_mean_normalized).
- **CompoundSignal:** name, description, severity (0–1), evidence (list of strings).
- **CalibrationProfile:** Per-metric distributions (entropy, margin, surprisal per step; L2 norm, attention entropy per layer). Save/load JSON. `compute_divergence_score` returns (score, anomalies).
- **ModelProfile:** Per-model threshold overrides: l2_explosion_multiplier, high_entropy_threshold_bits, repetition_cosine_threshold, typical_entropy_range, typical_l2_norm_range.
- **Prompt analysis:** layers[] (basin_scores, heads with query_indices, key_indices, weights); layer_transformations; prompt_surprisals.

---

## 14. Docs map (humans + deep dives)

- **README.md** — human intro, quick start, CLI, library API, features, architecture diagram, glossary, roadmap.
- **CHANGELOG.md** — v0.5.0-rc: all phases, schema/config/CLI changes, breaking changes, deprecations.
- **docs/v0.4.0-launch.md** — technical launch post: what CoreVital is, v0.4.0 features, how to try it.
- **docs/model-compatibility.md** — HF only; CausalLM vs Seq2Seq; SDPA vs eager; model_profiles; limitations (no vLLM/TGI/llama.cpp).
- **docs/metrics-interpretation.md** — metric definitions, renamed fields (topk_mass, topk_probs), new metrics (entropy_mean_normalized, clip_fraction, clip_max_before), compound signals, early warning, enriched layer blame, fingerprint v2, calibration, metric consistency, thresholds, citations.
- **docs/risk-calibration.md** — composite scoring formula, ECE/Platt scaling, 5-step benchmark workflow, calibration profiles, per-model threshold table, calibrate CLI.
- **docs/visual-examples.md** — good vs bad runs in dashboard.
- **docs/audit-existing-features.md** — verification notes for roadmap items.
- **docs/integrations.md** — OTLP, Langfuse, etc.
- **docs/production-deployment.md** — deployment, sinks, env.

---

## 15. One-shot "help a user" checklist

- **Explain the repo:** Sections 1–2, 3 (layout), 4 (data flow), 13 (abstractions).
- **Run / use:** Section 9 (use cases); CLI and library entry points in 2.
- **Find logic:** Section 5 ("Where is X?"); section 4 (data flow).
- **Contribute:** Section 10 (contributing); section 6 (commands); section 8 (conventions).
- **Evaluate (e.g. hiring):** Section 11 (due diligence); section 1 (identity, license).
- **Debug:** Section 12 (debugging); section 7 (config/env).
- **Extend:** Section 10 (add sink, metric, compound signal, model profile, schema); section 5 for where to wire.
- **Deploy in production:** Section 16.
- **Tune thresholds:** Section 17.
- **User says "model repeats" / "too many alerts" / "production cost":** Section 18.

---

## 16. Deploy in production

- **Capture mode:** `--capture summary` = minimal (health flags, timeline scalars, prompt scalars; no per-layer). `--capture full` = full trace. `--capture on_risk` = summary until risk/flag triggers then full. Use summary for high volume; see docs/production-deployment.md.
- **Calibration:** Build calibration profile offline (`corevital calibrate`); deploy with `--calibration <path>` for divergence scoring alongside heuristic risk.
- **Sampling:** CLI does not sample. In app: run CoreVital on a subset of requests (e.g. 1% or every Nth) via CoreVitalMonitor.
- **SQLite:** Set `--out /path` so DB is at `<out>/corevital.db`. Back up DB (cron, sqlite3 .backup). No auto-retention: delete old rows (e.g. `DELETE FROM reports WHERE created_at_utc < ?`) yourself.
- **Metrics export:** `--sink prometheus` + `--prometheus_port 9091` (scrape /metrics); or `--sink datadog` (DD_API_KEY); or `--export-otel` (OTEL_EXPORTER_OTLP_ENDPOINT). One process per instance or sidecar.
- **Alerting:** Alert on risk_score > threshold (e.g. 0.7) and on health flags (nan_detected, repetition_loop_detected, etc.). Compound signals (context_loss, degenerating_generation) at severity > 0.5. Metric names in sinks (e.g. corevital_risk_score, corevital_health_*).
- **Docker/K8s:** No built-in Dockerfile/Helm. Use volume for --out; expose Prometheus port if used; Secrets for API keys; env for OTEL endpoint. See docs/production-deployment.md checklist.

---

## 17. Tune thresholds

- **Where:** configs/model_profiles/*.yaml. Key = architecture mapping: GPT2*→gpt2, Llama*→llama, Mixtral*→mixtral (checked before Mistral), Mistral*→mistral, Qwen2*→qwen2, Phi3*/Phi-3*→phi3, T5*→t5, Bart*→bart, else default. Files: default.yaml, gpt2.yaml, llama.yaml, mistral.yaml, mixtral.yaml, qwen2.yaml, phi3.yaml.
- **Keys (all optional in YAML):** l2_explosion_multiplier (default 8.0), high_entropy_threshold_bits (4.0), repetition_cosine_threshold (0.9995), collapsed_head_entropy_threshold (0.1, deprecated for collapse detection), focused_head_concentration_threshold (0.9), typical_entropy_range (Optional [p10, p90]), typical_l2_norm_range (Optional [p10, p90]).
- **Calibrated values:** GPT-2: entropy 5.0, l2 5; LLaMA: entropy 3.5, l2 10, cosine 0.9998; Mixtral: entropy 4.5; Qwen2: entropy 4.5; Phi-3: entropy 3.8, l2 7.0; Mistral: defaults (stub).
- **Collapse detection:** Now uses normalized entropy threshold (NORMALIZED_COLLAPSED_THRESHOLD=0.03 in summaries/attention.py), NOT collapsed_head_entropy_threshold from profiles. The profile field is deprecated for this purpose.
- **Mid-layer anomaly:** Uses median of early-layer L2 norms as baseline (not mean). Robust to outlier early layers.
- **Override in code:** Set config.model_profile to a loaded profile to bypass auto-load. Or edit configs/model_profiles/<key>.yaml and restart.
- **Data-driven approach:** Use `corevital calibrate` to build a CalibrationProfile from known-healthy runs; then divergence scoring replaces static thresholds. See docs/risk-calibration.md.
- **Basin threshold:** Used in dashboard and get_basin_anomalies(); not in model_profiles; pass threshold to get_basin_anomalies(layers, threshold=0.3).

---

## 18. User intent → action & errors

- **"Model repeats" / "stuck in a loop":** repetition_loop_detected; logic in summaries/hidden_states.py detect_repetition_loop (optional token_id_buffer for n-gram cross-check). Suggest: shorten max_new_tokens, change prompt, or use should_intervene() to resample. Tune: repetition_cosine_threshold in model_profiles (lower = stricter). Check compound signal: confident_repetition_risk.
- **"Weird middle layers" / "mid-layer anomaly":** detect_mid_layer_anomaly; L2 explosion vs median of early-layer norms or NaN/Inf in middle third. Suggest: check inputs/precision; tune l2_explosion_multiplier in model_profiles. Check layer blame: blamed_layers has {layer, reasons, severity}.
- **"Too many false positives" (e.g. attention collapse):** Tune high_entropy_threshold_bits in model_profiles for that architecture. Collapse detection uses normalized threshold (0.03); tune focused_head_concentration_threshold. Consider building a calibration profile to replace static thresholds.
- **"Model losing context":** Check compound signal context_loss (high entropy + low basin). Check basin scores in prompt analysis.
- **"Generation quality declining":** Check compound signal degenerating_generation (rising entropy + declining margin). Check early warning signals: entropy_accelerating, margin_declining.
- **"Production too expensive":** Use --capture summary; sample (run CoreVital on subset of requests in app). See §16.
- **"Where do I alert?":** risk_score and health flags; compound signals at severity > 0.5; early_warning failure_risk. Export to Prometheus/Datadog/OTLP and alert there. See §16.
- **"How do I validate the risk score?":** Use calibration_risk.py with labeled benchmark data. See docs/risk-calibration.md 5-step workflow. Target ECE < 0.10.
- **Errors (where raised):** ConfigurationError (cli.py: missing API key, unknown sink). SinkError (sinks/*.py: write/upload failed). ValidationError (reporting/validation.py: invalid report). InstrumentationError (collector/causal_lm: beam search + capture, prompt too long, batch size != 1). All in errors.py; inherit CoreVitalError.

End of manifest.
