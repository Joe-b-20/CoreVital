# ============================================================================
# CoreVital - Time Utilities
#
# Purpose: Time-related utility functions
# Inputs: None
# Outputs: Timestamps
# Dependencies: datetime
# Usage: timestamp = get_utc_timestamp()
#
# Changelog:
#   2026-01-13: Initial time utilities for Phase-0
# ============================================================================

from datetime import datetime, timezone


def get_utc_timestamp() -> str:
    """
    Get current UTC timestamp in ISO-8601 format.

    Returns:
        ISO-8601 formatted timestamp string (e.g., "2026-01-11T15:22:08Z")
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
