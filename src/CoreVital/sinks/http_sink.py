# ============================================================================
# CoreVital - HTTP Sink
#
# Purpose: Base class for POSTing reports to a remote HTTP endpoint
# Inputs: Report objects
# Outputs: HTTP POST to remote URL
# Dependencies: base
# Usage: Subclass HTTPSink and implement _post() with requests or httpx
#
# Changelog:
#   2026-01-13: Initial HTTPSink for Phase-0
#   2026-02-04: Phase-0.75 - performance data arrives inside report.extensions
# ============================================================================

from CoreVital.logging_utils import get_logger
from CoreVital.reporting.schema import Report
from CoreVital.sinks.base import Sink

logger = get_logger(__name__)


class HTTPSink(Sink):
    """Sink that POSTs reports to a remote HTTP endpoint.

    This base implementation raises ``NotImplementedError`` on write.
    Subclass and override :meth:`write` with your preferred HTTP client
    (``requests``, ``httpx``, etc.) to add retry logic, authentication,
    and error handling for your deployment.
    """

    def __init__(self, url: str):
        self.url = url
        logger.info(f"HTTPSink initialized: {self.url}")

    def write(self, report: Report) -> str:
        """POST report to remote endpoint.

        Raises:
            NotImplementedError: Always. Subclass to provide HTTP logic.
        """
        raise NotImplementedError(
            "HTTPSink.write() is not implemented. Subclass HTTPSink and "
            "override write() with your HTTP client (requests, httpx), "
            "or use a built-in sink (sqlite, local, datadog, prometheus)."
        )
