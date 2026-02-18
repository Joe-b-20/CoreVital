# ============================================================================
# CoreVital - Prometheus Sink
#
# Purpose: Expose Report metrics on a /metrics endpoint for Prometheus scraping
# Inputs: Report objects
# Outputs: Prometheus gauge metrics on HTTP endpoint
# Dependencies: base, reporting.schema, prometheus_client (optional)
# Usage: sink = PrometheusSink(port=9091); sink.write(report)
#
# Changelog:
#   2026-02-07: Initial skeleton in pre-phase-1
#   2026-02-11: Phase-1d — Full implementation:
#               - Exposes gauges for health flags, entropy, perplexity, generation stats
#               - Labels: model, device, trace_id
#               - HTTP server starts on first write(), stays alive for scraping
#               - Always writes local file first (LocalFileSink)
#               - Graceful ImportError with clear install instructions
# ============================================================================

from typing import Any, Dict

from CoreVital.errors import SinkError
from CoreVital.logging_utils import get_logger
from CoreVital.reporting.schema import Report
from CoreVital.sinks.base import Sink

logger = get_logger(__name__)

# Metric prefix — all metrics appear under this namespace
_PREFIX = "corevital"


def _try_import_prometheus():
    """Lazy import of prometheus_client with clear error message."""
    try:
        from prometheus_client import Gauge, Info, start_http_server

        return {
            "Gauge": Gauge,
            "Info": Info,
            "start_http_server": start_http_server,
        }
    except ImportError as exc:
        raise ImportError(
            "PrometheusSink requires the 'prometheus-client' package.\n"
            "Install it with: pip install corevital[prometheus]\n"
            "Or directly: pip install prometheus-client"
        ) from exc


class PrometheusSink(Sink):
    """
    Sink that exposes Report metrics for Prometheus scraping.

    Metrics exposed (all gauges, labeled with model/device/trace_id):
        corevital_generation_total_steps
        corevital_generation_elapsed_ms
        corevital_generation_prompt_tokens
        corevital_health_nan_detected           (0/1)
        corevital_health_inf_detected           (0/1)
        corevital_health_attention_collapse      (0/1)
        corevital_health_high_entropy_steps
        corevital_health_repetition_loop         (0/1)
        corevital_health_mid_layer_anomaly       (0/1)
        corevital_step_entropy_last             (entropy of the last step)
        corevital_step_perplexity_last          (perplexity of the last step)
        corevital_entropy_mean                  (mean entropy across all steps)

    The HTTP server starts on the first write() call and stays alive
    for Prometheus to scrape. Subsequent write() calls update the gauge values.
    """

    # Class-level storage for gauges (shared across instances to avoid
    # prometheus_client duplicate registration errors)
    _gauges: Dict[str, Any] = {}
    _server_started: bool = False

    def __init__(self, port: int = 9091, *, local_output_dir: str = "runs"):
        """
        Initialize Prometheus sink.

        Args:
            port: Port for the /metrics HTTP endpoint (default: 9091)
            local_output_dir: Also write JSON locally (always-on backup)
        """
        self.port = port
        self.local_output_dir = local_output_dir
        logger.info(f"PrometheusSink initialized: port={self.port}")

    def _ensure_server(self) -> None:
        """Start the HTTP server if not already running."""
        if PrometheusSink._server_started:
            return
        prom = _try_import_prometheus()
        start_http_server = prom["start_http_server"]
        start_http_server(self.port)
        PrometheusSink._server_started = True
        logger.info(f"Prometheus metrics server started on :{self.port}/metrics")

    def _get_or_create_gauge(self, name: str, description: str, labels: list[str]) -> Any:
        """Get an existing gauge or create a new one."""
        full_name = f"{_PREFIX}_{name}"
        if full_name not in PrometheusSink._gauges:
            prom = _try_import_prometheus()
            Gauge = prom["Gauge"]
            PrometheusSink._gauges[full_name] = Gauge(full_name, description, labels)
        return PrometheusSink._gauges[full_name]

    def _set_gauge(
        self,
        name: str,
        description: str,
        value: float,
        labels: Dict[str, str],
    ) -> None:
        """Set a gauge value with labels."""
        label_keys = sorted(labels.keys())
        gauge = self._get_or_create_gauge(name, description, label_keys)
        gauge.labels(**labels).set(value)

    def write(self, report: Report) -> str:
        """
        Update Prometheus metrics from report and ensure server is running.

        Also writes a local JSON backup via LocalFileSink.

        Args:
            report: Report to expose as metrics

        Returns:
            Metrics endpoint URL

        Raises:
            SinkError: If metric update fails
        """
        # Always write locally as backup
        from CoreVital.sinks.local_file import LocalFileSink

        local_sink = LocalFileSink(self.local_output_dir)
        local_path = local_sink.write(report)
        logger.info(f"Local backup written to {local_path}")

        try:
            # Ensure HTTP server is running
            self._ensure_server()

            # Common labels
            labels = {
                "model": report.model.hf_id,
                "device": report.model.device,
                "trace_id": report.trace_id[:8],
            }

            # --- Generation metrics ---
            self._set_gauge(
                "generation_total_steps",
                "Total generation steps",
                float(report.summary.total_steps),
                labels,
            )
            self._set_gauge(
                "generation_elapsed_ms",
                "Total generation time in milliseconds",
                float(report.summary.elapsed_ms),
                labels,
            )
            self._set_gauge(
                "generation_prompt_tokens",
                "Number of prompt tokens",
                float(report.summary.prompt_tokens),
                labels,
            )

            # --- Health flags ---
            hf = report.health_flags
            if hf:
                self._set_gauge("health_nan_detected", "NaN detected (0/1)", 1.0 if hf.nan_detected else 0.0, labels)
                self._set_gauge("health_inf_detected", "Inf detected (0/1)", 1.0 if hf.inf_detected else 0.0, labels)
                self._set_gauge(
                    "health_attention_collapse",
                    "Attention collapse detected (0/1)",
                    1.0 if hf.attention_collapse_detected else 0.0,
                    labels,
                )
                self._set_gauge(
                    "health_high_entropy_steps",
                    "Number of high entropy steps",
                    float(hf.high_entropy_steps),
                    labels,
                )
                self._set_gauge(
                    "health_repetition_loop",
                    "Repetition loop detected (0/1)",
                    1.0 if hf.repetition_loop_detected else 0.0,
                    labels,
                )
                self._set_gauge(
                    "health_mid_layer_anomaly",
                    "Mid-layer anomaly detected (0/1)",
                    1.0 if hf.mid_layer_anomaly_detected else 0.0,
                    labels,
                )

            # --- Aggregate step metrics ---
            entropies = [
                s.logits_summary.entropy
                for s in report.timeline
                if s.logits_summary and s.logits_summary.entropy is not None
            ]
            if entropies:
                self._set_gauge(
                    "entropy_mean",
                    "Mean entropy across all generation steps",
                    sum(entropies) / len(entropies),
                    labels,
                )
                # Last step metrics (most recent signal)
                last_step = report.timeline[-1]
                if last_step.logits_summary:
                    if last_step.logits_summary.entropy is not None:
                        self._set_gauge(
                            "step_entropy_last",
                            "Entropy of the last generation step",
                            last_step.logits_summary.entropy,
                            labels,
                        )
                    if last_step.logits_summary.perplexity is not None:
                        self._set_gauge(
                            "step_perplexity_last",
                            "Perplexity of the last generation step",
                            last_step.logits_summary.perplexity,
                            labels,
                        )

            endpoint = f"http://localhost:{self.port}/metrics"
            logger.info(f"Prometheus metrics updated: {endpoint}")
            return f"{endpoint} + {local_path}"

        except ImportError:
            raise
        except Exception as e:
            raise SinkError(
                f"Failed to update Prometheus metrics: {e}",
                details=f"port={self.port}",
            ) from e
