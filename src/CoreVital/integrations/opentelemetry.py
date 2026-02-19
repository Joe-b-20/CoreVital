# ============================================================================
# CoreVital - OpenTelemetry integration
#
# Purpose: Export run as OTLP trace span + optional metrics for Langfuse,
#          OpenLIT, or any OTLP backend. Optional dependency: pip install CoreVital[otel]
# ============================================================================

from typing import Any, Optional, Tuple

from CoreVital.logging_utils import get_logger

logger = get_logger(__name__)


def get_otel_tracer_meter(otel_endpoint: Optional[str] = None) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Return (tracer, meter) for CoreVital export. If otel_endpoint is set, configures
    OTLP trace and metric exporters. Otherwise uses existing global provider or no-op.

    Returns (None, None) if OpenTelemetry is not installed.
    """
    trace_mod, metrics_mod = _get_otel()
    if trace_mod is None or metrics_mod is None:
        return None, None
    if otel_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            trace_provider = TracerProvider()
            exporter = OTLPSpanExporter(endpoint=otel_endpoint, insecure=True)
            trace_provider.add_span_processor(BatchSpanProcessor(exporter))
            trace_mod.set_tracer_provider(trace_provider)

            metric_exporter = OTLPMetricExporter(endpoint=otel_endpoint, insecure=True)
            reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=5000)
            meter_provider = MeterProvider(metric_readers=[reader])
            metrics_mod.set_meter_provider(meter_provider)
        except ImportError as e:
            logger.warning("OTLP gRPC exporter not available (%s). Install: pip install CoreVital[otel]", e)
        except Exception as e:
            logger.warning("OTLP provider setup failed (%s); using no-op export", e)
    tracer = trace_mod.get_tracer("corevital", "0.4.0")
    meter = metrics_mod.get_meter("corevital", "0.4.0")
    return tracer, meter


def _get_otel():
    """Optional import of OpenTelemetry API. Returns (trace, metrics) or (None, None)."""
    try:
        from opentelemetry import metrics, trace

        return trace, metrics
    except ImportError:
        return None, None


def export_run_to_otel(
    report: Any,
    tracer: Optional[Any] = None,
    meter: Optional[Any] = None,
) -> bool:
    """
    Export one CoreVital run as an OpenTelemetry span and optional metrics.

    Creates a span "corevital.run" with attributes: model_id, trace_id, risk_score,
    and health flags. If meter is provided, records risk_score as a histogram
    and high_entropy_steps as a counter.

    Args:
        report: CoreVital Report (object or dict).
        tracer: OpenTelemetry Tracer (if None, uses trace.get_tracer when available).
        meter: OpenTelemetry Meter (optional; if None, only span is emitted).

    Returns:
        True if export was attempted and succeeded; False if OTel not installed or export failed.
    """
    trace_mod, metrics_mod = _get_otel()
    if trace_mod is None:
        logger.debug("OpenTelemetry not installed; skip OTLP export. Install with: pip install CoreVital[otel]")
        return False

    # Normalize report to dict-like for attribute extraction
    if hasattr(report, "model"):
        model_id = getattr(report.model, "hf_id", None)
        trace_id = getattr(report, "trace_id", None)
    else:
        model_id = report.get("model", {}).get("hf_id") if isinstance(report, dict) else None
        trace_id = report.get("trace_id") if isinstance(report, dict) else None

    ext = getattr(report, "extensions", None) if not isinstance(report, dict) else report.get("extensions") or {}
    ext = ext or {}
    risk = ext.get("risk") or {}
    risk_score = risk.get("risk_score")
    health = getattr(report, "health_flags", None) if not isinstance(report, dict) else report.get("health_flags")
    if health is None:
        health_dict = {}
        high_entropy_steps = 0
    elif hasattr(health, "model_dump"):
        health_dict = health.model_dump()
        high_entropy_steps = getattr(health, "high_entropy_steps", 0) or 0
    else:
        health_dict = dict(health) if health else {}
        high_entropy_steps = health_dict.get("high_entropy_steps", 0) or 0

    try:
        if tracer is None:
            tracer = trace_mod.get_tracer("corevital", "0.4.0")
        with tracer.start_as_current_span("corevital.run") as span:
            if model_id is not None:
                span.set_attribute("corevital.model_id", str(model_id))
            if trace_id is not None:
                span.set_attribute("corevital.trace_id", str(trace_id))
            if risk_score is not None:
                span.set_attribute("corevital.risk_score", float(risk_score))
            span.set_attribute("corevital.health.high_entropy_steps", int(high_entropy_steps))
            for k, v in health_dict.items():
                if v is not None and k != "high_entropy_steps":
                    span.set_attribute(f"corevital.health.{k}", str(v) if not isinstance(v, (bool, int, float)) else v)

        if meter is not None:
            if risk_score is not None:
                hist = meter.create_histogram(
                    "corevital.risk_score",
                    description="CoreVital risk score (0-1) per run",
                )
                hist.record(float(risk_score), {"model_id": model_id or "unknown"})
            counter = meter.create_counter(
                "corevital.high_entropy_steps",
                description="Number of generation steps with high entropy",
            )
            counter.add(int(high_entropy_steps), {"model_id": model_id or "unknown"})
    except Exception as e:
        logger.warning("OTLP export failed: %s", e)
        return False
    return True
