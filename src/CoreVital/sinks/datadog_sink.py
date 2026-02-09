# ============================================================================
# CoreVital - Datadog Sink (Stub)
#
# Purpose: Decompose Report into Datadog custom metrics
# Inputs: Report objects
# Outputs: Metrics pushed to Datadog via DogStatsD / HTTP API
# Dependencies: base, reporting.schema
# Usage: sink = DatadogSink(api_key="..."); sink.write(report)
#
# Changelog:
#   2026-02-07: Stub created in pre-phase-1
#               Architecture is ready — implementation deferred until schema stabilizes
# ============================================================================

from CoreVital.logging_utils import get_logger
from CoreVital.reporting.schema import Report
from CoreVital.sinks.base import Sink

logger = get_logger(__name__)


class DatadogSink(Sink):
    """
    Sink that decomposes Report into Datadog custom metrics.

    The intended approach (post-stabilization):
        - Each numeric field becomes a Datadog gauge/histogram
        - Tags derived from model info (model, device, quantization)
        - Example metric: corevital.entropy.mean{model:gpt2, layer:3, step:12}

    This is a stub — calling write() raises NotImplementedError.
    """

    def __init__(self, api_key: str, host: str = "https://api.datadoghq.com"):
        """
        Initialize Datadog sink.

        Args:
            api_key: Datadog API key
            host: Datadog API host (default: US site)
        """
        self.api_key = api_key
        self.host = host
        logger.info(f"DatadogSink initialized (stub): host={self.host}")

    def write(self, report: Report) -> str:
        """
        Decompose report into Datadog metrics and push them.

        Not yet implemented — raises NotImplementedError.

        Args:
            report: Report to decompose and push

        Returns:
            Identifier string (e.g. trace URL)

        Raises:
            NotImplementedError: Always, until post-stabilization implementation
        """
        raise NotImplementedError(
            "DatadogSink is not yet implemented. Planned for post-schema-stabilization (phase-2+)."
        )
