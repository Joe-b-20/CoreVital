# ============================================================================
# CoreVital - Utils Package
#
# Purpose: Shared utility functions
# Inputs: None
# Outputs: Public API exports
# Dependencies: None
# Usage: from CoreVital.utils import serialize_report_to_json, get_utc_timestamp
#
# Changelog:
#   2026-01-13: Initial utils package for Phase-0
# ============================================================================

from CoreVital.utils.serialization import serialize_report_to_json
from CoreVital.utils.time import get_utc_timestamp

__all__ = ["serialize_report_to_json", "get_utc_timestamp"]
