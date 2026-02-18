# ============================================================================
# CoreVital - Prompt Fingerprints (Phase-3)
#
# Purpose: Compact run-summary vector and prompt hash for clustering and
#          duplicate detection. No model call; works offline.
# ============================================================================

import hashlib
from typing import Any, Dict, List

from CoreVital.reporting.schema import HealthFlags, Summary, TimelineStep


def compute_fingerprint_vector(
    timeline: List[TimelineStep],
    summary: Summary,
    health_flags: HealthFlags,
    risk_score: float,
) -> List[float]:
    """Build a fixed-size run-summary vector for clustering (Phase-3 Option A).

    Order: [mean_entropy, max_entropy, frac_high_entropy_steps, risk_score,
            nan, inf, collapse, repetition, mid_layer].
    Length: 9. All values in [0, 1] or small floats (entropy in bits).
    """
    entropies: List[float] = []
    for step in timeline:
        if step.logits_summary and step.logits_summary.entropy is not None:
            entropies.append(step.logits_summary.entropy)

    mean_entropy = sum(entropies) / len(entropies) if entropies else 0.0
    max_entropy = max(entropies) if entropies else 0.0
    total_steps = max(1, summary.total_steps)
    frac_high_entropy = health_flags.high_entropy_steps / total_steps

    nan = 1.0 if health_flags.nan_detected else 0.0
    inf = 1.0 if health_flags.inf_detected else 0.0
    collapse = 1.0 if health_flags.attention_collapse_detected else 0.0
    repetition = 1.0 if health_flags.repetition_loop_detected else 0.0
    mid_layer = 1.0 if health_flags.mid_layer_anomaly_detected else 0.0

    return [
        mean_entropy,
        max_entropy,
        frac_high_entropy,
        risk_score,
        nan,
        inf,
        collapse,
        repetition,
        mid_layer,
    ]


def compute_prompt_hash(prompt_text: str, model_id: str) -> str:
    """SHA256 of normalized prompt + model_id for exact duplicate detection (Phase-3 Option C)."""
    normalized = (prompt_text.strip().lower() + "\n" + model_id).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def get_fingerprint(report: Any) -> Dict[str, Any]:
    """Return fingerprint dict from a report for downstream clustering (Phase-3 API).

    Args:
        report: Report object or dict with .extensions or ["extensions"].

    Returns:
        dict with "vector" and "prompt_hash" if present; otherwise {}.
    """
    if hasattr(report, "extensions"):
        ext = report.extensions or {}
    elif isinstance(report, dict):
        ext = report.get("extensions") or {}
    else:
        return {}
    return ext.get("fingerprint") or {}
