# ============================================================================
# CoreVital - Serialization Utilities
#
# Purpose: Serialize Report objects to JSON
# Inputs: Report objects
# Outputs: JSON strings
# Dependencies: json, pydantic
# Usage: json_str = serialize_report_to_json(report)
#
# Changelog:
#   2026-01-13: Initial serialization for Phase-0
#   2026-02-11: JSON size optimization — compact format by default (indent=None, separators),
#               exclude None fields (exclude_none=True). Saves ~63% file size vs pretty-printed.
#               Dashboard provides on-demand pretty-printing for inspection.
# ============================================================================

import json
from typing import Optional

from CoreVital.reporting.schema import Report


def serialize_report_to_json(report: Report, indent: Optional[int] = None) -> str:
    """
    Serialize a Report object to JSON string.

    Args:
        report: Report to serialize
        indent: JSON indentation (None for compact, 2 for pretty-print)

    Returns:
        JSON string (compact by default for smaller file sizes)
    """
    # Exclude None fields to reduce file size (Pydantic will use defaults on deserialize)
    report_dict = report.model_dump(mode="json", exclude_none=True)
    if indent is None:
        # Compact JSON: no indent, minimal separators (saves ~63% file size)
        json_str = json.dumps(report_dict, separators=(",", ":"), ensure_ascii=False)
    else:
        # Pretty-print JSON: with indentation
        json_str = json.dumps(report_dict, indent=indent, ensure_ascii=False)
    return json_str


def deserialize_report_from_json(json_str: str) -> Report:
    """
    Deserialize a Report object from JSON string.

    Args:
        json_str: JSON string

    Returns:
        Report object
    """
    data = json.loads(json_str)
    return Report(**data)
