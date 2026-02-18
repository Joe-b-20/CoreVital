# ============================================================================
# CoreVital - Early Warning / Failure Risk (Phase-4)
#
# Purpose: Post-run failure_risk and warning_signals from timeline and
#          health flags. Streaming part deferred to Library API (monitor).
# ============================================================================

from typing import List, Tuple

from CoreVital.reporting.schema import HealthFlags, TimelineStep


def compute_early_warning(
    timeline: List[TimelineStep],
    health_flags: HealthFlags,
) -> Tuple[float, List[str]]:
    """Compute failure_risk (0–1) and warning_signals from timeline and health flags (Phase-4 post-run).

    Rules:
    - repetition_loop_detected → failure_risk 0.9, signal "repetition_loop"
    - Else if entropy trend positive and max_entropy > 4 → failure_risk 0.6, "entropy_rising", "high_entropy"
    - Else → failure_risk 0.3; add "high_entropy" if max_entropy > 4
    """
    warning_signals: List[str] = []
    entropies: List[float] = []
    for step in timeline:
        if step.logits_summary and step.logits_summary.entropy is not None:
            entropies.append(step.logits_summary.entropy)

    max_entropy = max(entropies) if entropies else 0.0
    # Entropy trend: rising if mean of last 5 > mean of first 5 (or linear trend if few steps)
    k = 5
    if len(entropies) >= k:
        first_k = sum(entropies[:k]) / k
        last_k = sum(entropies[-k:]) / k
        entropy_rising = last_k > first_k
    elif len(entropies) >= 2:
        entropy_rising = entropies[-1] > entropies[0]
    else:
        entropy_rising = False

    if health_flags.repetition_loop_detected:
        failure_risk = 0.9
        warning_signals.append("repetition_loop")
    elif entropy_rising and max_entropy > 4:
        failure_risk = 0.6
        warning_signals.extend(["entropy_rising", "high_entropy"])
    else:
        failure_risk = 0.3
        if max_entropy > 4:
            warning_signals.append("high_entropy")

    if health_flags.mid_layer_anomaly_detected and "mid_layer_anomaly" not in warning_signals:
        warning_signals.append("mid_layer_anomaly")
    if health_flags.attention_collapse_detected and "attention_collapse" not in warning_signals:
        warning_signals.append("attention_collapse")

    return failure_risk, warning_signals
