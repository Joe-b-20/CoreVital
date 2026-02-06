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
# ============================================================================

from typing import Optional
import json
from CoreVital.reporting.schema import Report


def serialize_report_to_json(report: Report, indent: Optional[int] = 2) -> str:
    """
    Serialize a Report object to JSON string.

    Args:
        report: Report to serialize
        indent: JSON indentation (None for compact)

    Returns:
        JSON string
    """
    report_dict = report.model_dump(mode="json")
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