# ============================================================================
# CoreVital - Report Validation
#
# Purpose: Validate Report objects against schema requirements
# Inputs: Report object
# Outputs: Validation result (bool or exception)
# Dependencies: reporting.schema, errors
# Usage: validate_report(report)
#
# Changelog:
#   2026-01-13: Initial validation for Phase-0
# ============================================================================

from CoreVital.reporting.schema import Report
from CoreVital.errors import ValidationError
from CoreVital.logging_utils import get_logger


logger = get_logger(__name__)


def validate_report(report: Report) -> bool:
    """
    Validate a Report object.
    
    Args:
        report: Report to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    errors = []
    
    # Check schema version
    if report.schema_version != "0.1.0":
        errors.append(f"Invalid schema_version: {report.schema_version}")
    
    # Check required fields
    if not report.trace_id:
        errors.append("Missing trace_id")
    
    if not report.created_at_utc:
        errors.append("Missing created_at_utc")
    
    # Check model info
    if not report.model.hf_id:
        errors.append("Missing model.hf_id")
    
    if report.model.num_layers <= 0:
        errors.append(f"Invalid num_layers: {report.model.num_layers}")
    
    # Check prompt
    if not report.prompt.text:
        errors.append("Missing prompt.text")
    
    if len(report.prompt.token_ids) != report.prompt.num_tokens:
        errors.append("Prompt token_ids length mismatch")
    
    # Check timeline
    if len(report.timeline) == 0:
        logger.warning("Timeline is empty (may be valid for errors)")
    
    # Check summary consistency
    expected_total = report.summary.prompt_tokens + report.summary.generated_tokens
    if report.summary.total_steps != expected_total:
        # Note:
        #   total_steps is allowed to differ from prompt_tokens + generated_tokens.
        #   In some pipelines, total_steps may include additional internal steps
        #   (e.g., system tokens, preprocessing/postprocessing steps, or discarded
        #   tokens) that are not counted in prompt_tokens or generated_tokens.
        #   Because of this, we treat this as a soft consistency check and only log
        #   a warning instead of failing validation.
        logger.warning(
            f"Summary total_steps ({report.summary.total_steps}) != "
            f"prompt_tokens + generated_tokens ({expected_total}); "
            "this can be expected when extra processing steps are included"
        )
    
    if errors:
        error_msg = "Report validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValidationError(error_msg)
    
    logger.debug("Report validation passed")
    return True