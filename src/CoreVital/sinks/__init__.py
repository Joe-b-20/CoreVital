# ============================================================================
# CoreVital - Sinks Package
#
# Purpose: Persistence backends for reports
# Inputs: None
# Outputs: Public API exports
# Dependencies: None
# Usage: from CoreVital.sinks import Sink, LocalFileSink, HTTPSink
#
# Changelog:
#   2026-01-13: Initial sinks package for Phase-0
# ============================================================================

from CoreVital.sinks.base import Sink
from CoreVital.sinks.local_file import LocalFileSink
from CoreVital.sinks.http_sink import HTTPSink

__all__ = ["Sink", "LocalFileSink", "HTTPSink"]