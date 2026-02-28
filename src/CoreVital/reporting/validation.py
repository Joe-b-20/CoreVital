# ============================================================================
# CoreVital - Report Validation
#
# Purpose: Validate Report objects against schema requirements and check
#          metric self-consistency (Issue 15)
# Inputs: Report object
# Outputs: Validation result (bool or exception), consistency warnings
# Dependencies: reporting.schema, errors
# Usage: validate_report(report); validate_metric_consistency(report)
#
# Changelog:
#   2026-01-13: Initial validation for Phase-0
#   2026-02-28: Phase 5, Issue 15 — validate_metric_consistency(): perplexity ==
#               2^entropy, entropy/concentration correlation, margin ≤ topk_mass
# ============================================================================

import math
from typing import List, Optional

from CoreVital.errors import ValidationError
from CoreVital.logging_utils import get_logger
from CoreVital.reporting.schema import Report

logger = get_logger(__name__)

PERPLEXITY_REL_TOL = 0.01
CONCENTRATION_ENTROPY_THRESHOLD = 0.95
CONCENTRATION_ENTROPY_MAX_NATS = 1.0


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

    # Check schema version (0.3.0 accepted for backward compatibility when loading from DB)
    supported_versions = {"0.3.0", "0.4.0"}
    if report.schema_version not in supported_versions:
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


def validate_metric_consistency(
    report: Report,
    *,
    perplexity_rel_tol: float = PERPLEXITY_REL_TOL,
    concentration_entropy_threshold: float = CONCENTRATION_ENTROPY_THRESHOLD,
    concentration_entropy_max_nats: float = CONCENTRATION_ENTROPY_MAX_NATS,
) -> List[str]:
    """Check internal consistency of computed metrics. Returns list of warnings.

    Invariants checked per timeline step:
      1. perplexity ≈ 2^entropy  (relative tolerance ``perplexity_rel_tol``)
      2. top_k_margin ≤ topk_mass (voter_agreement) — margin is top-1 minus top-2
         probability; topk_mass is the sum of the top-K probabilities, always ≥ margin
      3. Per-layer: high concentration_max with non-low entropy_min is suspicious —
         a very peaked attention distribution should correspond to low entropy

    This function never raises; it collects warnings for logging.

    Args:
        report: A fully-built Report object.
        perplexity_rel_tol: Relative tolerance for perplexity vs 2^entropy.
        concentration_entropy_threshold: Concentration above which entropy should
            be low.
        concentration_entropy_max_nats: Maximum expected entropy_min (nats) when
            concentration_max exceeds the threshold.

    Returns:
        List of human-readable warning strings. Empty list = all consistent.
    """
    warnings_list: List[str] = []

    for step in report.timeline:
        step_idx = step.step_index
        logits = step.logits_summary

        # 1. perplexity == 2^entropy
        entropy: Optional[float] = logits.entropy if logits else None
        perplexity: Optional[float] = logits.perplexity if logits else None

        if entropy is not None and perplexity is not None:
            expected_ppl = 2.0 ** entropy
            denom = max(expected_ppl, 1e-10)
            if abs(perplexity - expected_ppl) / denom > perplexity_rel_tol:
                warnings_list.append(
                    f"Step {step_idx}: perplexity ({perplexity:.4f}) != "
                    f"2^entropy ({expected_ppl:.4f}), "
                    f"relative error {abs(perplexity - expected_ppl) / denom:.4f}"
                )

        # 2. top_k_margin ≤ topk_mass (voter_agreement)
        margin: Optional[float] = logits.top_k_margin if logits else None
        topk_mass: Optional[float] = logits.voter_agreement if logits else None

        if margin is not None and topk_mass is not None:
            if margin > topk_mass + 1e-6:
                warnings_list.append(
                    f"Step {step_idx}: top_k_margin ({margin:.4f}) > "
                    f"topk_mass ({topk_mass:.4f}) — impossible"
                )

        # 3. Per-layer: concentration_max vs entropy_min correlation
        for layer in step.layers:
            attn = layer.attention_summary
            if attn is None:
                continue
            ent_min = attn.entropy_min
            conc_max = attn.concentration_max
            if ent_min is not None and conc_max is not None:
                if conc_max > concentration_entropy_threshold and ent_min > concentration_entropy_max_nats:
                    warnings_list.append(
                        f"Step {step_idx}, Layer {layer.layer_index}: "
                        f"concentration_max={conc_max:.4f} but entropy_min={ent_min:.4f} — "
                        f"high concentration should correlate with low entropy"
                    )

        # 4. Entropy should be non-negative (information-theoretic invariant)
        if entropy is not None and entropy < -1e-6:
            warnings_list.append(
                f"Step {step_idx}: negative entropy ({entropy:.4f})"
            )

        # 5. Perplexity should be >= 1 (2^0 = 1 is the minimum for zero entropy)
        if perplexity is not None and perplexity < 1.0 - 1e-6:
            warnings_list.append(
                f"Step {step_idx}: perplexity ({perplexity:.4f}) < 1.0"
            )

        # 6. Surprisal should be non-negative
        surprisal: Optional[float] = logits.surprisal if logits else None
        if surprisal is not None and surprisal < -1e-6:
            warnings_list.append(
                f"Step {step_idx}: negative surprisal ({surprisal:.4f})"
            )

        # 7. Entropy ≤ log2(vocab_size) — checked loosely via perplexity
        #    max perplexity = vocab_size; we skip this if we don't know vocab size

    return warnings_list


def validate_metric_consistency_and_log(report: Report) -> List[str]:
    """Run metric consistency validation and log any warnings at WARNING level.

    Convenience wrapper for use in report_builder debug mode.

    Returns:
        The list of warning strings (also logged).
    """
    warnings_list = validate_metric_consistency(report)
    if warnings_list:
        logger.warning(
            "Metric consistency issues detected (%d):\n  %s",
            len(warnings_list),
            "\n  ".join(warnings_list),
        )
    else:
        logger.debug("Metric consistency validation passed — all invariants hold")
    return warnings_list
