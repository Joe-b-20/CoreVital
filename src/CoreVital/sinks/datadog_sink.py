# ============================================================================
# CoreVital - Datadog Sink
#
# Purpose: Decompose Report into Datadog custom metrics via DogStatsD protocol
# Inputs: Report objects
# Outputs: Metrics pushed to Datadog via datadog-api-client
# Dependencies: base, reporting.schema, datadog_api_client (optional)
# Usage: sink = DatadogSink(api_key="...", site="datadoghq.com"); sink.write(report)
#
# Changelog:
#   2026-02-07: Initial skeleton in pre-phase-1
#   2026-02-11: Phase-1d — Full implementation:
#               - Decomposes Report into per-step and aggregate metrics
#               - Uses Datadog Metrics API v2 (submit_metrics)
#               - Tags: model, device, trace_id, quantized
#               - Graceful ImportError with clear install instructions
#               - Always writes local file first (LocalFileSink), then pushes metrics
# ============================================================================

import time
from typing import Any, List

from CoreVital.errors import SinkError
from CoreVital.logging_utils import get_logger
from CoreVital.reporting.schema import Report
from CoreVital.sinks.base import Sink

logger = get_logger(__name__)

# Metric prefix — all metrics appear under this namespace in Datadog
_PREFIX = "corevital"


def _try_import_datadog():
    """Lazy import of datadog_api_client with clear error message."""
    try:
        from datadog_api_client import ApiClient, Configuration
        from datadog_api_client.v2.api.metrics_api import MetricsApi
        from datadog_api_client.v2.model.metric_intake_type import MetricIntakeType
        from datadog_api_client.v2.model.metric_payload import MetricPayload
        from datadog_api_client.v2.model.metric_point import MetricPoint
        from datadog_api_client.v2.model.metric_resource import MetricResource
        from datadog_api_client.v2.model.metric_series import MetricSeries

        return {
            "ApiClient": ApiClient,
            "Configuration": Configuration,
            "MetricsApi": MetricsApi,
            "MetricIntakeType": MetricIntakeType,
            "MetricPayload": MetricPayload,
            "MetricPoint": MetricPoint,
            "MetricResource": MetricResource,
            "MetricSeries": MetricSeries,
        }
    except ImportError as exc:
        raise ImportError(
            "DatadogSink requires the 'datadog-api-client' package.\n"
            "Install it with: pip install corevital[datadog]\n"
            "Or directly: pip install datadog-api-client"
        ) from exc


class DatadogSink(Sink):
    """
    Sink that decomposes Report into Datadog custom metrics.

    Metrics submitted:
        Aggregate (per-run):
            corevital.generation.total_steps     gauge
            corevital.generation.elapsed_ms      gauge
            corevital.health.nan_detected         gauge (0/1)
            corevital.health.inf_detected         gauge (0/1)
            corevital.health.attention_collapse    gauge (0/1)
            corevital.health.high_entropy_steps   gauge (count)
            corevital.health.repetition_loop      gauge (0/1)
            corevital.health.mid_layer_anomaly    gauge (0/1)

        Per-step (tagged with step index):
            corevital.step.entropy               gauge
            corevital.step.perplexity            gauge
            corevital.step.surprisal             gauge

    Tags on all metrics:
        model:<hf_id>, device:<device>, trace_id:<id>, quantized:<true/false>
    """

    def __init__(
        self,
        api_key: str,
        site: str = "datadoghq.com",
        *,
        local_output_dir: str = "runs",
    ):
        """
        Initialize Datadog sink.

        Args:
            api_key: Datadog API key (or set DD_API_KEY env var)
            site: Datadog site (e.g. "datadoghq.com", "datadoghq.eu")
            local_output_dir: Also write JSON locally (always-on backup)
        """
        self.api_key = api_key
        self.site = site
        self.local_output_dir = local_output_dir
        logger.info(f"DatadogSink initialized: site={self.site}")

    def _build_tags(self, report: Report) -> List[str]:
        """Build Datadog tags from report metadata."""
        tags = [
            f"model:{report.model.hf_id}",
            f"device:{report.model.device}",
            f"trace_id:{report.trace_id[:8]}",
        ]
        if report.model.quantization and report.model.quantization.enabled:
            tags.append("quantized:true")
            if report.model.quantization.method:
                tags.append(f"quant_method:{report.model.quantization.method}")
        else:
            tags.append("quantized:false")
        return tags

    def _build_series(self, report: Report) -> List[Any]:
        """Build Datadog MetricSeries list from report."""
        dd = _try_import_datadog()
        MetricSeries = dd["MetricSeries"]
        MetricPoint = dd["MetricPoint"]
        MetricIntakeType = dd["MetricIntakeType"]

        now = int(time.time())
        tags = self._build_tags(report)
        series: List[Any] = []

        def _gauge(name: str, value: float, extra_tags: List[str] | None = None) -> None:
            all_tags = tags + (extra_tags or [])
            series.append(
                MetricSeries(
                    metric=f"{_PREFIX}.{name}",
                    type=MetricIntakeType.GAUGE,
                    points=[MetricPoint(timestamp=now, value=value)],
                    tags=all_tags,
                )
            )

        # --- Aggregate metrics ---
        _gauge("generation.total_steps", float(report.summary.total_steps))
        _gauge("generation.elapsed_ms", float(report.summary.elapsed_ms))
        _gauge("generation.prompt_tokens", float(report.summary.prompt_tokens))

        # Health flags
        hf = report.health_flags
        if hf:
            _gauge("health.nan_detected", 1.0 if hf.nan_detected else 0.0)
            _gauge("health.inf_detected", 1.0 if hf.inf_detected else 0.0)
            _gauge("health.attention_collapse", 1.0 if hf.attention_collapse_detected else 0.0)
            _gauge("health.high_entropy_steps", float(hf.high_entropy_steps))
            _gauge("health.repetition_loop", 1.0 if hf.repetition_loop_detected else 0.0)
            _gauge("health.mid_layer_anomaly", 1.0 if hf.mid_layer_anomaly_detected else 0.0)

        # --- Per-step metrics ---
        for step in report.timeline:
            step_tags = [f"step:{step.step_index}"]
            ls = step.logits_summary
            if ls:
                if ls.entropy is not None:
                    _gauge("step.entropy", ls.entropy, step_tags)
                if ls.perplexity is not None:
                    _gauge("step.perplexity", ls.perplexity, step_tags)
                if ls.surprisal is not None:
                    _gauge("step.surprisal", ls.surprisal, step_tags)

        return series

    def write(self, report: Report) -> str:
        """
        Decompose report into Datadog metrics and push them.

        Also writes a local JSON backup via LocalFileSink.

        Args:
            report: Report to decompose and push

        Returns:
            Identifier string (trace_id + metric count)

        Raises:
            SinkError: If metric submission fails
        """
        # Always write locally as backup
        from CoreVital.sinks.local_file import LocalFileSink

        local_sink = LocalFileSink(self.local_output_dir)
        local_path = local_sink.write(report)
        logger.info(f"Local backup written to {local_path}")

        # Build and submit metrics
        try:
            dd = _try_import_datadog()
            Configuration = dd["Configuration"]
            ApiClient = dd["ApiClient"]
            MetricsApi = dd["MetricsApi"]
            MetricPayload = dd["MetricPayload"]

            series = self._build_series(report)

            configuration = Configuration()
            configuration.api_key["apiKeyAuth"] = self.api_key
            configuration.server_variables["site"] = self.site

            with ApiClient(configuration) as api_client:
                api_instance = MetricsApi(api_client)
                payload = MetricPayload(series=series)
                api_instance.submit_metrics(body=payload)

            logger.info(f"Submitted {len(series)} metrics to Datadog (site={self.site})")
            return f"datadog:{report.trace_id[:8]} ({len(series)} metrics) + {local_path}"

        except ImportError:
            raise
        except Exception as e:
            raise SinkError(
                f"Failed to submit metrics to Datadog: {e}",
                details=f"site={self.site}, series_count={len(series) if 'series' in dir() else '?'}",
            ) from e
