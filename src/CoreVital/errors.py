# ============================================================================
# CoreVital - Error Classes
#
# Purpose: Custom exception hierarchy for the package
# Inputs: Error messages and context
# Outputs: Structured exceptions
# Dependencies: None
# Usage: raise ModelLoadError("Failed to load model")
#
# Changelog:
#   2026-01-13: Initial error classes for Phase-0
# ============================================================================

from typing import Optional


class CoreVitalError(Exception):
    """Base exception for all CoreVital errors."""

    def __init__(self, message: str, details: Optional[str] = None):
        """
        Initialize error.

        Args:
            message: Error message
            details: Optional detailed error information
        """
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        """String representation."""
        if self.details:
            return f"{self.message}\nDetails: {self.details}"
        return self.message


class ConfigurationError(CoreVitalError):
    """Raised when configuration is invalid or missing."""

    pass


class ModelLoadError(CoreVitalError):
    """Raised when model loading fails."""

    pass


class InstrumentationError(CoreVitalError):
    """Raised when instrumentation fails during inference."""

    pass


class SummaryComputationError(CoreVitalError):
    """Raised when summary computation fails."""

    pass


class SinkError(CoreVitalError):
    """Raised when sink write operation fails."""

    pass


class ValidationError(CoreVitalError):
    """Raised when report validation fails."""

    pass
