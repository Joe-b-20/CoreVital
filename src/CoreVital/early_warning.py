# ============================================================================
# CoreVital - Early Warning / Failure Risk (Phase-2.5)
#
# Purpose: Genuine trend detection that identifies degradation patterns
#          *before* they trip boolean health-flag thresholds.
#          Unlike risk_score (which aggregates what already happened),
#          early_warning detects patterns suggesting the NEXT N tokens
#          are likely to fail.
#
# Signals:
#   entropy_accelerating   — entropy rate-of-change is itself increasing
#   margin_collapsed       — top_k_margin near zero for recent window
#   margin_declining       — margin dropped to <30% of its early level
#   surprisal_volatile     — coefficient of variation > 1.5 in recent window
#   entropy_margin_divergence — entropy above threshold while margin stays high
#                              (confident but confused)
#   repetition_loop        — carried from health_flags (hard ceiling)
#   mid_layer_anomaly      — carried from health_flags
#   attention_collapse     — carried from health_flags
# ============================================================================

import statistics
from typing import List, Tuple

from CoreVital.reporting.schema import HealthFlags, TimelineStep

DEFAULT_HIGH_ENTROPY_THRESHOLD = 4.0


def _extract_entropies(timeline: List[TimelineStep]) -> List[float]:
    return [
        s.logits_summary.entropy
        for s in timeline
        if s.logits_summary and s.logits_summary.entropy is not None
    ]


def _extract_margins(timeline: List[TimelineStep]) -> List[float]:
    return [
        s.logits_summary.top_k_margin
        for s in timeline
        if s.logits_summary
        and s.logits_summary.top_k_margin is not None
    ]


def _extract_surprisals(timeline: List[TimelineStep]) -> List[float]:
    return [
        s.logits_summary.surprisal
        for s in timeline
        if s.logits_summary
        and s.logits_summary.surprisal is not None
    ]


def _detect_entropy_acceleration(
    entropies: List[float], window_size: int,
) -> bool:
    """True when the rate of entropy increase is itself increasing."""
    if len(entropies) < window_size * 2:
        return False
    windows = [
        entropies[i : i + window_size]
        for i in range(0, len(entropies) - window_size + 1)
    ]
    window_means = [sum(w) / len(w) for w in windows]
    if len(window_means) < 3:
        return False
    deltas = [window_means[i + 1] - window_means[i] for i in range(len(window_means) - 1)]
    recent_deltas = deltas[-3:]
    return all(d > 0 for d in recent_deltas) and recent_deltas[-1] > recent_deltas[0]


def _detect_margin_collapse(
    margins: List[float], window_size: int,
) -> Tuple[bool, bool]:
    """Return (collapsed, declining)."""
    collapsed = False
    declining = False
    if len(margins) >= window_size:
        recent = margins[-window_size:]
        if all(m < 0.1 for m in recent):
            collapsed = True
        elif len(margins) >= window_size * 2:
            early = sum(margins[:window_size]) / window_size
            late = sum(margins[-window_size:]) / window_size
            if early > 0 and late / early < 0.3:
                declining = True
    return collapsed, declining


def _detect_surprisal_volatility(
    surprisals: List[float], window_size: int,
) -> bool:
    """True when coefficient of variation exceeds 1.5 in recent window."""
    if len(surprisals) < window_size:
        return False
    recent = surprisals[-window_size:]
    if len(recent) < 2:
        return False
    vol = statistics.stdev(recent)
    mean_s = sum(recent) / len(recent)
    if mean_s <= 0:
        return False
    return vol / mean_s > 1.5


def _detect_entropy_margin_divergence(
    entropies: List[float],
    margins: List[float],
    window_size: int,
    high_entropy_threshold: float,
) -> bool:
    """True when entropy is high but margin stays high — confident yet confused."""
    if len(entropies) < window_size or len(margins) < window_size:
        return False
    recent_ent = sum(entropies[-window_size:]) / window_size
    recent_margin = sum(margins[-window_size:]) / window_size
    return recent_ent > high_entropy_threshold and recent_margin > 0.3


def compute_early_warning(
    timeline: List[TimelineStep],
    health_flags: HealthFlags,
    high_entropy_threshold: float = DEFAULT_HIGH_ENTROPY_THRESHOLD,
    window_size: int = 5,
) -> Tuple[float, List[str]]:
    """Detect degradation trends that predict imminent failure.

    Unlike risk_score (which aggregates what happened), early_warning
    identifies patterns suggesting the NEXT N tokens are likely to fail.

    Args:
        timeline: Generation timeline steps with logits summaries.
        health_flags: Aggregated boolean health flags from the run.
        high_entropy_threshold: Entropy threshold in bits (from model profile).
        window_size: Sliding window size for trend detection.

    Returns:
        (failure_risk, warning_signals) where failure_risk is in [0, 1].
    """
    signals: List[str] = []
    failure_risk = 0.0

    entropies = _extract_entropies(timeline)
    margins = _extract_margins(timeline)
    surprisals = _extract_surprisals(timeline)

    # 1. Entropy acceleration: rate of change is itself increasing
    if _detect_entropy_acceleration(entropies, window_size):
        signals.append("entropy_accelerating")
        failure_risk = max(failure_risk, 0.7)

    # 2. Margin collapse: top-k margin dropping toward zero
    collapsed, declining = _detect_margin_collapse(margins, window_size)
    if collapsed:
        signals.append("margin_collapsed")
        failure_risk = max(failure_risk, 0.6)
    elif declining:
        signals.append("margin_declining")
        failure_risk = max(failure_risk, 0.5)

    # 3. Surprisal volatility: erratic surprisal suggests instability
    if _detect_surprisal_volatility(surprisals, window_size):
        signals.append("surprisal_volatile")
        failure_risk = max(failure_risk, 0.5)

    # 4. Entropy-margin divergence: high entropy + high margin = about to break
    if _detect_entropy_margin_divergence(entropies, margins, window_size, high_entropy_threshold):
        signals.append("entropy_margin_divergence")
        failure_risk = max(failure_risk, 0.55)

    # 5. Boolean health flags as hard ceilings (don't let them dominate the signal list)
    if health_flags.repetition_loop_detected:
        failure_risk = max(failure_risk, 0.9)
        if "repetition_loop" not in signals:
            signals.append("repetition_loop")
    if health_flags.mid_layer_anomaly_detected:
        failure_risk = max(failure_risk, 0.6)
        if "mid_layer_anomaly" not in signals:
            signals.append("mid_layer_anomaly")
    if health_flags.attention_collapse_detected:
        failure_risk = max(failure_risk, 0.4)
        if "attention_collapse" not in signals:
            signals.append("attention_collapse")

    return failure_risk, signals
