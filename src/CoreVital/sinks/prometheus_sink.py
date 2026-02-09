# ============================================================================
# CoreVital - Prometheus Sink (Stub)
#
# Purpose: Expose Report metrics on a /metrics endpoint for Prometheus scraping
# Inputs: Report objects
# Outputs: Prometheus gauge/histogram metrics
# Dependencies: base, reporting.schema
# Usage: sink = PrometheusSink(port=9090); sink.write(report)
#
# Changelog:
#   2026-02-07: Stub created in pre-phase-1
#               Architecture is ready — implementation deferred until schema stabilizes
# ============================================================================

from CoreVital.logging_utils import get_logger
from CoreVital.reporting.schema import Report
from CoreVital.sinks.base import Sink

logger = get_logger(__name__)


class PrometheusSink(Sink):
    """
    Sink that exposes Report metrics for Prometheus scraping.

    The intended approach (post-stabilization):
        - Start a lightweight HTTP server on a configurable port
        - Each numeric field becomes a Prometheus gauge or histogram
        - Labels derived from model info
        - Example: corevital_entropy_mean{model="gpt2",layer="3",step="12"} 0.73

    This is a stub — calling write() raises NotImplementedError.
    """

    def __init__(self, port: int = 9090):
        """
        Initialize Prometheus sink.

        Args:
            port: Port for the /metrics HTTP endpoint
        """
        self.port = port
        logger.info(f"PrometheusSink initialized (stub): port={self.port}")

    def write(self, report: Report) -> str:
        """
        Update Prometheus metrics from report.

        Not yet implemented — raises NotImplementedError.

        Args:
            report: Report to expose as metrics

        Returns:
            Metrics endpoint URL

        Raises:
            NotImplementedError: Always, until post-stabilization implementation
        """
        raise NotImplementedError(
            "PrometheusSink is not yet implemented. Planned for post-schema-stabilization (phase-2+)."
        )
