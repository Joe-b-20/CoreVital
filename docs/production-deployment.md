# Production deployment guide

This guide covers how to run CoreVital in production: sampling, persistence, metrics export, and alerting. It also explains the **Two Viewing Paths** (open-source/individual vs enterprise) for visualizing your data.

## ⚠️ Critical: Built-in scores are not production-calibrated

**Before deploying to production with risk-based alerts:**

The built-in **`risk_score`** and **`failure_risk`** are **heuristic research placeholders**, not production-calibrated predictors. Validation on GSM8K/HumanEval showed:

- **risk_score saturates at 1.0** for Mistral (96%) and Mixtral (94%)
- **Poor calibration:** ECE 0.24-0.70 before Platt scaling
- **Weak discrimination:** AUROC 0.48-0.62 (near chance in some cells)
- **failure_risk is discrete** (2-5 unique values), AUROC near chance

**Do NOT:**
- Set production SLAs based on raw risk_score thresholds without calibration
- Trigger automated rollbacks or circuit breakers on failure_risk
- Assume risk_score=0.8 means 80% failure probability (it doesn't)

**DO:**
- Use built-in scores as **indicative signals** for debugging and exploration
- Run your own calibration workflow (see [Risk Calibration](risk-calibration.md)) with labeled data from your domain
- Implement per-model calibration (validation showed pooled models don't transfer well)
- Use CoreVital metrics (entropy, margin, surprisal, attention stats) as inputs to your own learned failure detector

See [Risk Calibration](risk-calibration.md) for experiment findings, calibration workflow, and recommendations. See [Validation Report](validation-report.md) for full methodology and per-model results.

## Sampling strategy

Instrumenting every request can be expensive. Use **capture mode** and **sampling** to control cost and payload size.

- **`--capture summary`** — Store only health flags, time series (entropy, perplexity, surprisal), and prompt scalars. No per-layer data. Best for high-volume production.
- **`--capture full`** — Full trace (all layers, attention summaries). Use for debugging or a small fraction of traffic.
- **`--capture on_risk`** — Summary by default; when `risk_score` or health flags exceed thresholds, also persist a full trace. Requires Phase-2 risk; balance between observability and storage.

**Sampling in your app:** If you use the Library API (`CoreVitalMonitor`), run CoreVital only on a subset of requests (e.g. 1% random, or every N-th request, or only when latency &gt; P95). The CLI does not sample; wrap it in your own script or scheduler that chooses which prompts to monitor.

**Report on GPU:** By default, report and summary computation runs on CPU (tensors are offloaded after generation so the GPU is free for inference). On weak-CPU cloud hosts (e.g. RunPod), use `--report-on-gpu` (or `COREVITAL_DEVICE_REPORT_ON_GPU=1`) so summary ops run on the model device and avoid a CPU bottleneck.

## Database setup (SQLite)

Default sink is SQLite. For production:

1. **Path and backups**
   - Set a dedicated path, e.g. `--out /var/lib/corevital/runs` so the DB is at `/var/lib/corevital/runs/corevital.db`.
   - Back up the DB regularly (e.g. cron + `sqlite3 .backup` or filesystem snapshots).

2. **Migration**
   - Ensure the DB is initialized with `corevital migrate` if you ever need to create or alter the schema. Normal `run` commands create the DB and tables on first write.

3. **Retention**
   - CoreVital does not auto-delete old traces. Implement retention (e.g. delete rows older than 30 days) via cron or a small script that runs `DELETE FROM reports WHERE created_at_utc < ?` (table name is `reports`).

## Metrics export (Prometheus / Datadog)

- **Prometheus:** Use `--sink prometheus` and `--prometheus_port 9091`. Scrape `/metrics` from your Prometheus config. See `src/CoreVital/sinks/prometheus_sink.py` for metric names (e.g. `corevital_risk_score`, `corevital_health_*`).
- **Datadog:** Use `--sink datadog` with `--datadog_api_key` or `DD_API_KEY`. Metrics and events are sent to Datadog; see `src/CoreVital/sinks/datadog_sink.py`.
- **OpenTelemetry:** Use `--export-otel` and `--otel-endpoint` (or `OTEL_EXPORTER_OTLP_ENDPOINT`) to send traces and metrics to any OTLP backend (e.g. Langfuse, OpenLIT). See [Integrations](integrations.md).

Run one CoreVital process per app instance (or a dedicated sidecar) so metrics reflect that instance; aggregate in Prometheus/Datadog/OTLP.

## Alerting on risk and health flags

⚠️ **See warning above** — Built-in risk_score is not production-calibrated. If using risk-based alerts:

- **Run calibration first:** Use the workflow in [Risk Calibration](risk-calibration.md) to fit Platt scaling or a learned model on your labeled data
- **Use health flags for critical issues:** `nan_detected`, `repetition_loop_detected` are boolean and more reliable than continuous risk scores
- **Combine with domain metrics:** Don't rely solely on CoreVital scores; use them alongside task-specific quality checks

**If you choose to alert on risk_score** (after calibration):
- **Prometheus:** `corevital_risk_score > <calibrated_threshold>`
- **Datadog:** Similar on the exported metric
- **Calibrated thresholds:** Determine from ECE analysis, not arbitrary cutoffs

**Health flags (more reliable):**
- `corevital_health_nan_detected` — Catastrophic numerical issues (alert immediately)
- `corevital_health_repetition_loop_detected` — Degenerate output (high-severity alert)
- `corevital_health_attention_collapse_detected` — Common in healthy runs; use with caution
- `corevital_health_mid_layer_anomaly` — Suggests processing failure (medium severity)

Combine with your existing incident pipeline (PagerDuty, Slack, etc.).

## Optional: Docker / K8s

- **Docker:** Build an image that installs CoreVital and your config. Run the CLI as the main process or as a sidecar that reads prompts from a queue or shared volume. Mount a volume for `--out` (SQLite) and expose the Prometheus port if using `--sink prometheus`.
- **Kubernetes:** Deploy as a Deployment or DaemonSet. Use a PVC for SQLite if you use the DB sink. Expose metrics via a Service and scrape with Prometheus Operator or similar. Use Secrets for API keys (Datadog, etc.) and env for `OTEL_EXPORTER_OTLP_ENDPOINT` if using OTLP.

CoreVital does not ship a ready-made Dockerfile or Helm chart; you can add a minimal Dockerfile that runs `corevital run` with your desired args and config.

## Two Viewing Paths (open-source vs enterprise)

- **Path A (open-source / individual developers):** Use the hosted React dashboard ([https://main.d2maxwaq575qed.amplifyapp.com](https://main.d2maxwaq575qed.amplifyapp.com)). Run `corevital serve` locally so the dashboard can list and load traces from your SQLite DB. Data never leaves your machine.
- **Path B (enterprise teams):** You do not need the React dashboard. Configure native sinks in `config.yaml` (or via `--sink datadog`, `--sink prometheus`, `--sink wandb`) so metrics and reports go directly to Datadog, Prometheus, or Weights & Biases. View and alert on CoreVital data in the observability tools your company already uses.

## Checklist

- [ ] Choose capture mode: `summary`, `full`, or `on_risk`.
- [ ] Decide sampling rate if not monitoring every request.
- [ ] Set SQLite path and backup/retention.
- [ ] Configure one of: Prometheus, Datadog, or OTLP (see [integrations](integrations.md)).
- [ ] Define alerts on `risk_score` and critical health flags.
- [ ] (Optional) Run in Docker/K8s with persistent storage and secrets for API/OTLP.
