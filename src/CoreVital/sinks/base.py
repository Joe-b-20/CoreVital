# ============================================================================
# CoreVital - Base Sink Interface
#
# Purpose: Abstract base class for persistence sinks
# Inputs: Report objects
# Outputs: Location identifier string
# Dependencies: abc, reporting.schema
# Usage: class MySink(Sink): ...
#
# Changelog:
#   2026-01-13: Initial Sink interface for Phase-0
# ============================================================================

from abc import ABC, abstractmethod

from CoreVital.reporting.schema import Report


class Sink(ABC):
    """
    Abstract base class for report persistence sinks.

    Sinks are responsible for persisting Report objects to various backends
    (local filesystem, HTTP endpoints, databases, cloud storage, etc.).
    """

    @abstractmethod
    def write(self, report: Report) -> str:
        """
        Write a report to the sink.

        Args:
            report: Report object to persist

        Returns:
            Location identifier (file path, URL, etc.)

        Raises:
            SinkError: If write operation fails
        """
        pass
