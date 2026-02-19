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
#   2026-02-07: Pre-phase-1 - added DatadogSink and PrometheusSink stubs
#   2026-02-11: Phase-1d - DatadogSink and PrometheusSink fully implemented
# ============================================================================

from CoreVital.sinks.base import Sink
from CoreVital.sinks.datadog_sink import DatadogSink
from CoreVital.sinks.http_sink import HTTPSink
from CoreVital.sinks.local_file import LocalFileSink
from CoreVital.sinks.prometheus_sink import PrometheusSink
from CoreVital.sinks.sqlite_sink import SQLiteSink
from CoreVital.sinks.wandb_sink import WandBSink

__all__ = [
    "Sink",
    "LocalFileSink",
    "HTTPSink",
    "DatadogSink",
    "PrometheusSink",
    "SQLiteSink",
    "WandBSink",
]
