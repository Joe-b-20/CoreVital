# ============================================================================
# CoreVital - Risk Score and Layer Blame (Phase-2)
#
# Purpose: Compute a single risk score per run from health flags and
#          attribute blame to layers (anomalies, collapse).
# Inputs: HealthFlags, Summary, timeline (optional), layers_by_step (optional)
# Outputs: risk_score in [0, 1], risk_factors list, blamed_layers list
# ============================================================================

from typing import TYPE_CHECKING, List, Optional, Tuple

from CoreVital.reporting.schema import HealthFlags, LayerSummary, Summary, TimelineStep

if TYPE_CHECKING:
    from CoreVital.compound_signals import CompoundSignal


def compute_risk_score_legacy(
    health_flags: HealthFlags,
    summary: Summary,
) -> Tuple[float, List[str]]:
    """Legacy risk score from health flags and summary only (no timeline).

    Formula: NaN/Inf → 1.0; repetition 0.9; mid-layer 0.7; attention_collapse 0.3;
    high_entropy_steps / total_steps * 0.5 additive.

    Returns:
        (risk_score, risk_factors).
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


def compute_risk_score(
    health_flags: HealthFlags,
    summary: Summary,
    timeline: Optional[List[TimelineStep]] = None,
    layers_by_step: Optional[List[List[LayerSummary]]] = None,
    compound_signals: Optional[List["CompoundSignal"]] = None,
) -> Tuple[float, List[str]]:
    """Composite risk score from health flags, continuous timeline metrics, and compound signals.

    When timeline is None, falls back to compute_risk_score_legacy.
    When timeline is empty, returns 0.0 and no factors (unless NaN/Inf).
    Boolean flags (NaN/Inf, repetition, mid-layer anomaly) act as hard ceilings via max();
    continuous metrics (entropy, top_k_margin, topk_mass, surprisal) and compound signal
    severities are additive (capped at 1.0).

    Returns:
        (score, factors) so downstream knows why the score is what it is.
    """
    # Hard ceiling: NaN/Inf is always catastrophic
    if health_flags.nan_detected or health_flags.inf_detected:
        return 1.0, ["nan_or_inf"]

    if timeline is None:
        return compute_risk_score_legacy(health_flags, summary)

    if not timeline:
        return 0.0, []

    factors: List[str] = []
    components: List[float] = []

    # --- Boolean flag components (weighted; act as ceilings via max later) ---
    if health_flags.repetition_loop_detected:
        components.append(0.9)
        factors.append("repetition_loop")
    if health_flags.mid_layer_anomaly_detected:
        components.append(0.7)
        factors.append("mid_layer_anomaly")
    if health_flags.attention_collapse_detected:
        components.append(0.15)
        factors.append("attention_collapse")

    # --- Continuous metric components (additive) ---
    entropies = [
        s.logits_summary.entropy
        for s in timeline
        if s.logits_summary and s.logits_summary.entropy is not None
    ]
    if entropies:
        mean_ent = sum(entropies) / len(entropies)
        entropy_component = min(1.0, mean_ent / 8.0) * 0.3
        if entropy_component > 0.05:
            components.append(entropy_component)
            factors.append("elevated_entropy")
        if len(entropies) >= 6:
            k = len(entropies) // 3
            first_third = sum(entropies[:k]) / k
            last_third = sum(entropies[-k:]) / k
            if last_third > first_third * 1.3:
                trend_component = min(0.2, (last_third - first_third) / first_third * 0.1)
                components.append(trend_component)
                factors.append("entropy_rising")

    margins = [
        s.logits_summary.top_k_margin
        for s in timeline
        if s.logits_summary
        and getattr(s.logits_summary, "top_k_margin", None) is not None
    ]
    if margins:
        mean_margin = sum(margins) / len(margins)
        margin_component = max(0.0, (1.0 - mean_margin * 5.0)) * 0.2
        if margin_component > 0.05:
            components.append(margin_component)
            factors.append("low_confidence_margin")

    # topk_mass (prefer) or voter_agreement
    agreements = []
    for s in timeline:
        if not s.logits_summary:
            continue
        val = getattr(s.logits_summary, "topk_mass", None) or getattr(
            s.logits_summary, "voter_agreement", None
        )
        if val is not None:
            agreements.append(val)
    if agreements:
        mean_agreement = sum(agreements) / len(agreements)
        agreement_component = max(0.0, (1.0 - mean_agreement)) * 0.15
        if agreement_component > 0.03:
            components.append(agreement_component)
            factors.append("low_topk_mass")

    surprisals = [
        s.logits_summary.surprisal
        for s in timeline
        if s.logits_summary and getattr(s.logits_summary, "surprisal", None) is not None
    ]
    if surprisals:
        mean_surprisal = sum(surprisals) / len(surprisals)
        surprisal_component = min(0.1, mean_surprisal / 10.0)
        if surprisal_component > 0.02:
            components.append(surprisal_component)
            factors.append("elevated_surprisal")

    # --- Compound signals (Issue 6): severity as additive component ---
    if compound_signals:
        for cs in compound_signals:
            components.append(cs.severity)
            factors.append(f"compound:{cs.name}")

    # Combine: boolean flags dominate via max; continuous are additive (capped at 1.0)
    if components:
        bool_components = [c for c in components if c >= 0.5]
        continuous_components = [c for c in components if c < 0.5]
        bool_max = max(bool_components) if bool_components else 0.0
        continuous_sum = sum(continuous_components)
        score = min(1.0, bool_max + continuous_sum)
    else:
        score = 0.0

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
