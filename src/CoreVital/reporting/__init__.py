# ============================================================================
# CoreVital - Reporting Package
#
# Purpose: Report schema, building, and validation
# Inputs: None
# Outputs: Public API exports
# Dependencies: None
# Usage: from CoreVital.reporting import Report, ReportBuilder, validate_report
#
# Changelog:
#   2026-01-13: Initial reporting package for Phase-0
# ============================================================================

from CoreVital.reporting.report_builder import ReportBuilder
from CoreVital.reporting.schema import Report
from CoreVital.reporting.validation import validate_metric_consistency, validate_report

__all__ = ["Report", "ReportBuilder", "validate_metric_consistency", "validate_report"]
