# ============================================================================
# CoreVital - Risk Score and Layer Blame (Phase-2)
#
# Purpose: Compute a single risk score per run from health flags and
#          attribute blame to layers (anomalies, collapse).
# Inputs: HealthFlags, Summary, optional layer data for blame
# Outputs: risk_score in [0, 1], risk_factors list, blamed_layers list
# ============================================================================

from typing import List, Tuple

from CoreVital.reporting.schema import HealthFlags, LayerSummary, Summary


def compute_risk_score(
    health_flags: HealthFlags,
    summary: Summary,
) -> Tuple[float, List[str]]:
    """Compute a single risk score in [0, 1] from health flags and summary.

    Formula (hand-crafted combination; can be tuned later):
    - NaN/Inf → 1.0 (max risk).
    - Repetition loop → 0.9.
    - Mid-layer anomaly → 0.7.
    - Attention collapse → 0.3 (structural; often present in healthy runs).
    - High entropy steps: normalized by total_steps, contribution up to 0.5.

    Returns:
        (risk_score, risk_factors) where risk_factors lists which signals contributed.
    """
    factors: List[str] = []

    if health_flags.nan_detected or health_flags.inf_detected:
        return 1.0, ["nan_or_inf"]

    score = 0.0
    if health_flags.repetition_loop_detected:
        score = max(score, 0.9)
        factors.append("repetition_loop")
    if health_flags.mid_layer_anomaly_detected:
        score = max(score, 0.7)
        factors.append("mid_layer_anomaly")
    if health_flags.attention_collapse_detected:
        score = max(score, 0.3)
        factors.append("attention_collapse")

    total_steps = max(1, summary.total_steps)
    entropy_component = min(1.0, health_flags.high_entropy_steps / total_steps) * 0.5
    score = min(1.0, score + entropy_component)
    if health_flags.high_entropy_steps > 0:
        factors.append("high_entropy_steps")

    return score, factors


def compute_layer_blame(layers_by_step: List[List[LayerSummary]]) -> List[int]:
    """Identify layer indices that contributed to risk (anomalies or collapse).

    Rule-based: any layer that had NaN/Inf (anomalies) or attention collapse
    (collapsed_head_count > 0) in any step is blamed.

    Args:
        layers_by_step: For each step, a list of LayerSummary (same order as layer_index).

    Returns:
        Sorted unique list of layer indices (0-based) that had anomalies or collapse.
    """
    blamed: set[int] = set()
    for step_layers in layers_by_step:
        for layer in step_layers:
            if layer.anomalies and (layer.anomalies.has_nan or layer.anomalies.has_inf):
                blamed.add(layer.layer_index)
            if layer.attention_summary and getattr(layer.attention_summary, "collapsed_head_count", 0) > 0:
                blamed.add(layer.layer_index)
    return sorted(blamed)
