# ============================================================================
# CoreVital - Prompt Fingerprints (Phase-4.3)
#
# Purpose: Compact run-summary vector and prompt hash for clustering and
#          duplicate detection. No model call; works offline.
#
# Vector v2 (25 elements): encodes temporal patterns, cross-metric
# correlations, and trend slopes alongside scalar summaries.
# ============================================================================

import hashlib
import statistics
from typing import Any, Dict, List, Optional, Tuple

from CoreVital.reporting.schema import HealthFlags, Summary, TimelineStep

FINGERPRINT_VERSION = 2
FINGERPRINT_LENGTH = 25


def _correlation(xs: List[float], ys: List[float]) -> float:
    """Pearson correlation between two equal-length sequences.

    Returns 0.0 when either sequence has zero variance or lengths differ.
    """
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0
    x_mean = sum(xs[:n]) / n
    y_mean = sum(ys[:n]) / n
    cov = sum((xs[i] - x_mean) * (ys[i] - y_mean) for i in range(n))
    var_x = sum((xs[i] - x_mean) ** 2 for i in range(n))
    var_y = sum((ys[i] - y_mean) ** 2 for i in range(n))
    denom = (var_x * var_y) ** 0.5
    if denom == 0.0:
        return 0.0
    return float(cov / denom)


def _safe_stats(vals: List[float]) -> Tuple[float, float, float, float, float]:
    """Return (mean, std, min, max, trend_slope) for a list of values."""
    if not vals:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    mean = sum(vals) / len(vals)
    std = statistics.stdev(vals) if len(vals) >= 2 else 0.0
    if len(vals) >= 2:
        n = len(vals)
        x_mean = (n - 1) / 2.0
        numerator = sum((i - x_mean) * (v - mean) for i, v in enumerate(vals))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        slope = numerator / denominator if denominator > 0 else 0.0
    else:
        slope = 0.0
    return (mean, std, min(vals), max(vals), slope)


def compute_fingerprint_vector(
    timeline: List[TimelineStep],
    summary: Summary,
    health_flags: HealthFlags,
    risk_score: float,
    layers_by_step: Optional[List[List[Any]]] = None,
) -> List[float]:
    """Build a fixed-size run-summary vector for clustering (v2, 25 elements).

    Encodes temporal patterns, cross-metric relationships, and layer-level
    signals alongside the original scalar summaries.

    Elements:
        0-4:   Entropy profile (mean, std, min, max, slope)
        5-7:   Margin profile (mean, std, slope)
        8-10:  Surprisal profile (mean, std, slope)
        11-12: Agreement profile (mean, std)
        13:    Risk score
        14:    Fraction of high-entropy steps
        15-19: Boolean flags (nan, inf, collapse, repetition, mid_layer)
        20:    Entropy-margin correlation
        21:    Entropy coefficient of variation
        22-23: First-quarter / last-quarter entropy means
        24:    [reserved, currently 0.0]
    """
    entropies = [
        s.logits_summary.entropy for s in timeline if s.logits_summary and s.logits_summary.entropy is not None
    ]
    margins = [
        s.logits_summary.top_k_margin
        for s in timeline
        if s.logits_summary and hasattr(s.logits_summary, "top_k_margin") and s.logits_summary.top_k_margin is not None
    ]
    surprisals = [
        s.logits_summary.surprisal
        for s in timeline
        if s.logits_summary and hasattr(s.logits_summary, "surprisal") and s.logits_summary.surprisal is not None
    ]
    agreements = []
    for s in timeline:
        if not s.logits_summary:
            continue
        val = getattr(s.logits_summary, "topk_mass", None)
        if val is None:
            val = getattr(s.logits_summary, "voter_agreement", None)
        if val is not None:
            agreements.append(val)

    ent_stats = _safe_stats(entropies)
    margin_stats = _safe_stats(margins)
    surprisal_stats = _safe_stats(surprisals)
    agreement_stats = _safe_stats(agreements)

    total_steps = max(1, summary.total_steps)

    # First-quarter / last-quarter entropy means
    if len(entropies) >= 4:
        q_len = len(entropies) // 4
        first_q_mean = sum(entropies[:q_len]) / max(1, q_len)
        last_q_mean = sum(entropies[-q_len:]) / max(1, q_len)
    else:
        first_q_mean = 0.0
        last_q_mean = 0.0

    return [
        # Entropy profile (5 values: indices 0-4)
        ent_stats[0],
        ent_stats[1],
        ent_stats[2],
        ent_stats[3],
        ent_stats[4],
        # Margin profile (3 values: indices 5-7)
        margin_stats[0],
        margin_stats[1],
        margin_stats[4],
        # Surprisal profile (3 values: indices 8-10)
        surprisal_stats[0],
        surprisal_stats[1],
        surprisal_stats[4],
        # Agreement profile (2 values: indices 11-12)
        agreement_stats[0],
        agreement_stats[1],
        # Aggregate signals (indices 13-14)
        risk_score,
        health_flags.high_entropy_steps / total_steps,
        # Boolean flags (5 values: indices 15-19)
        1.0 if health_flags.nan_detected else 0.0,
        1.0 if health_flags.inf_detected else 0.0,
        1.0 if health_flags.attention_collapse_detected else 0.0,
        1.0 if health_flags.repetition_loop_detected else 0.0,
        1.0 if health_flags.mid_layer_anomaly_detected else 0.0,
        # Cross-metric features (2 values: indices 20-21)
        _correlation(entropies, margins) if entropies and margins else 0.0,
        (ent_stats[1] / ent_stats[0]) if ent_stats[0] > 0 else 0.0,
        # Temporal features (3 values: indices 22-24)
        first_q_mean,
        last_q_mean,
        0.0,  # reserved slot for future use
    ]


def is_legacy_fingerprint(vector: List[float]) -> bool:
    """Return True if the vector is a v1 9-element fingerprint."""
    return len(vector) == 9


def compute_prompt_hash(prompt_text: str, model_id: str) -> str:
    """SHA256 of normalized prompt + model_id for exact duplicate detection."""
    normalized = (prompt_text.strip().lower() + "\n" + model_id).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def get_fingerprint(report: Any) -> Dict[str, Any]:
    """Return fingerprint dict from a report for downstream clustering.

    Args:
        report: Report object or dict with .extensions or ["extensions"].

    Returns:
        dict with "vector", "prompt_hash", and "version" if present;
        otherwise {}.
    """
    if hasattr(report, "extensions"):
        ext = report.extensions or {}
    elif isinstance(report, dict):
        ext = report.get("extensions") or {}
    else:
        return {}
    return ext.get("fingerprint") or {}
