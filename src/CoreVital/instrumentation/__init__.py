# ============================================================================
# CoreVital - Instrumentation Package
#
# Purpose: Instrumentation, hooks, and collection logic
# Inputs: None
# Outputs: Public API exports
# Dependencies: None
# Usage: from CoreVital.instrumentation import InstrumentationCollector
#
# Changelog:
#   2026-01-13: Initial instrumentation package for Phase-0
# ============================================================================

from CoreVital.instrumentation.collector import InstrumentationCollector, InstrumentationResults

__all__ = ["InstrumentationCollector", "InstrumentationResults"]