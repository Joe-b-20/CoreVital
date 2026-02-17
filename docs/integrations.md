# CoreVital integrations

This document describes how to send CoreVital telemetry to external observability backends.

## OpenTelemetry (OTLP)

CoreVital can export each run as OpenTelemetry **traces** and **metrics** via OTLP (gRPC). This lets you:

- Send data to **Langfuse**, **OpenLIT**, or any OTLP-compatible backend
- Correlate CoreVital runs with other spans in your tracing system
- Build dashboards and alerts on `corevital.risk_score` and health flags

### Setup

1. **Install the optional dependency**

   ```bash
   pip install CoreVital[otel]
   ```

2. **Run with export enabled**

   - Use `--export-otel` and either `--otel-endpoint HOST:PORT` or set the environment variable:
     - `OTEL_EXPORTER_OTLP_ENDPOINT` (e.g. `http://localhost:4317` or `localhost:4317` for gRPC)

   **CLI example:**

   ```bash
   corevital run --model gpt2 --prompt "Hello" --export-otel --otel-endpoint localhost:4317
   ```

   Or with env:

   ```bash
   export OTEL_EXPORTER_OTLP_ENDPOINT=localhost:4317
   corevital run --model gpt2 --prompt "Hello" --export-otel
   ```

### What is exported

- **Span:** `corevital.run`  
  Attributes: `corevital.model_id`, `corevital.trace_id`, `corevital.risk_score`, `corevital.health.*` (e.g. `high_entropy_steps`, `attention_collapse_detected`).
- **Metrics:**
  - `corevital.risk_score` (histogram) — risk score per run
  - `corevital.high_entropy_steps` (counter) — number of high-entropy generation steps

### Using with Langfuse / OpenLIT

- **Langfuse:** Use an OTLP collector or Langfuse’s OTLP ingestion (see [Langfuse docs](https://langfuse.com/docs)) and point `OTEL_EXPORTER_OTLP_ENDPOINT` at that collector or ingestion endpoint.
- **OpenLIT:** Configure OpenLIT to accept OTLP and set the same endpoint. See [OpenLIT documentation](https://docs.openlit.io/) for receiving OTLP traces and metrics.

If the endpoint is not set, CoreVital still creates spans and metrics locally; they will only be sent when an OTLP endpoint is configured (e.g. via env or `--otel-endpoint`).

### Config (YAML / API)

In configuration you can set:

- `otel.export_otel: true`
- `otel.otel_endpoint: "localhost:4317"` (optional; env `OTEL_EXPORTER_OTLP_ENDPOINT` can be used instead)

The CLI flags `--export-otel` and `--otel-endpoint` override these when running from the command line.
