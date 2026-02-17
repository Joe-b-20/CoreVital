# ============================================================================
# CoreVital - Package Initialization
#
# Purpose: Package-level exports and version information
# Inputs: None
# Outputs: Public API exports
# Dependencies: None
# Usage: from CoreVital import __version__
#
# Changelog:
#   2026-01-13: Initial package setup for Phase-0
# ============================================================================

__version__ = "0.3.0"
__author__ = "Joe Bachir"
__license__ = "Apache-2.0"

from CoreVital.config import Config
from CoreVital.monitor import CoreVitalMonitor
from CoreVital.reporting.schema import Report
from CoreVital.sinks.base import Sink
from CoreVital.sinks.local_file import LocalFileSink

__all__ = [
    "__version__",
    "Config",
    "CoreVitalMonitor",
    "Report",
    "Sink",
    "LocalFileSink",
]
