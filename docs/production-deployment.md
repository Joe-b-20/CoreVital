# Production deployment guide

This guide covers how to run CoreVital in production: sampling, persistence, metrics export, and alerting.

## Sampling strategy

Instrumenting every request can be expensive. Use **capture mode** and **sampling** to control cost and payload size.

- **`--capture summary`** — Store only health flags, time series (entropy, perplexity, surprisal), and prompt scalars. No per-layer data. Best for high-volume production.
- **`--capture full`** — Full trace (all layers, attention summaries). Use for debugging or a small fraction of traffic.
- **`--capture on_risk`** — Summary by default; when `risk_score` or health flags exceed thresholds, also persist a full trace. Requires Phase-2 risk; balance between observability and storage.

**Sampling in your app:** If you use the Library API (`CoreVitalMonitor`), run CoreVital only on a subset of requests (e.g. 1% random, or every N-th request, or only when latency &gt; P95). The CLI does not sample; wrap it in your own script or scheduler that chooses which prompts to monitor.

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

- **Risk score:** Alert when `risk_score` exceeds a threshold (e.g. > 0.7). Prometheus: `corevital_risk_score > 0.7`. Datadog: similar on the metric you export.
- **Health flags:** Use series like `corevital_health_nan_detected`, `corevital_health_attention_collapse_detected`, `corevital_health_repetition_loop_detected` (names may vary; check the sink implementation). Alert on “any true” or "> 0" depending on how they’re encoded.

Combine with your existing incident pipeline (PagerDuty, Slack, etc.) by having Prometheus/Datadog send alerts when these conditions fire.

## Optional: Docker / K8s

- **Docker:** Build an image that installs CoreVital and your config. Run the CLI as the main process or as a sidecar that reads prompts from a queue or shared volume. Mount a volume for `--out` (SQLite) and expose the Prometheus port if using `--sink prometheus`.
- **Kubernetes:** Deploy as a Deployment or DaemonSet. Use a PVC for SQLite if you use the DB sink. Expose metrics via a Service and scrape with Prometheus Operator or similar. Use Secrets for API keys (Datadog, etc.) and env for `OTEL_EXPORTER_OTLP_ENDPOINT` if using OTLP.

CoreVital does not ship a ready-made Dockerfile or Helm chart; you can add a minimal Dockerfile that runs `corevital run` with your desired args and config.

## Checklist

- [ ] Choose capture mode: `summary`, `full`, or `on_risk`.
- [ ] Decide sampling rate if not monitoring every request.
- [ ] Set SQLite path and backup/retention.
- [ ] Configure one of: Prometheus, Datadog, or OTLP (see [integrations](integrations.md)).
- [ ] Define alerts on `risk_score` and critical health flags.
- [ ] (Optional) Run in Docker/K8s with persistent storage and secrets for API/OTLP.
