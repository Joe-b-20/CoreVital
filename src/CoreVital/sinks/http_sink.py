# ============================================================================
# CoreVital - HTTP Sink (Stub)
#
# Purpose: POST reports to remote HTTP endpoint
# Inputs: Report objects
# Outputs: HTTP POST to remote URL
# Dependencies: base, utils.serialization
# Usage: sink = HTTPSink("https://api.example.com/traces"); sink.write(report)
#
# Changelog:
#   2026-01-13: Initial HTTPSink stub for Phase-0
#   2026-02-04: Phase-0.75 - added note: performance data is injected by CLI after write
# ============================================================================

from CoreVital.reporting.schema import Report
from CoreVital.sinks.base import Sink
from CoreVital.utils.serialization import serialize_report_to_json
from CoreVital.errors import SinkError
from CoreVital.logging_utils import get_logger


logger = get_logger(__name__)


class HTTPSink(Sink):
    """
    Sink that POSTs reports to a remote HTTP endpoint.
    
    Note: This is a stub implementation for Phase-0. Full implementation
    would include retry logic, authentication, etc.
    """
    
    def __init__(self, url: str):
        """
        Initialize HTTP sink.
        
        Args:
            url: Target URL for POST requests
        """
        self.url = url
        logger.info(f"HTTPSink initialized: {self.url}")
    
    def write(self, report: Report) -> str:
        """
        POST report to remote endpoint.
        
        Args:
            report: Report to send
            
        Returns:
            URL or identifier of posted resource
            
        Raises:
            SinkError: If POST fails
        """
        try:
            # Serialize report
            # Note: Performance data is added by CLI after sink_write completes
            json_str = serialize_report_to_json(report, indent=None)
            
            # TODO: Implement actual HTTP POST with requests or httpx
            # For Phase-0, this is a stub that logs intent
            logger.warning(f"HTTPSink.write() is a stub - would POST to {self.url}")
            logger.debug(f"Would send {len(json_str)} bytes")
            
            # Return mock URL
            return f"{self.url}/{report.trace_id}"
            
        except Exception as e:
            logger.exception("Failed to POST report")
            raise SinkError(
                f"Failed to POST report to {self.url}",
                details=str(e)
            ) from e