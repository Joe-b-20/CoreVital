# ============================================================================
# CoreVital - Compound Signal Detection (Phase-2, Issue 6)
#
# Purpose: Detect multi-metric failure patterns (e.g. high entropy + low
#          basin = context loss) from timeline and optional layer/basin data.
# Inputs: timeline, layers_by_step (optional), basin_scores (optional)
# Outputs: List[CompoundSignal] for report extensions and risk scoring
# ============================================================================

from dataclasses import dataclass
from typing import List, Optional

from CoreVital.reporting.schema import LayerSummary, TimelineStep


@dataclass
class CompoundSignal:
    """A detected compound failure pattern with evidence."""

    name: str
    description: str
    severity: float  # 0-1
    evidence: List[str]  # human-readable metric citations


def detect_compound_signals(
    timeline: List[TimelineStep],
    layers_by_step: Optional[List[List[LayerSummary]]] = None,
    basin_scores: Optional[List[List[float]]] = None,
) -> List[CompoundSignal]:
    """Detect multi-metric failure patterns from timeline and optional layer/basin data."""
    signals: List[CompoundSignal] = []

    entropies = _extract(timeline, "entropy")
    margins = _extract(timeline, "top_k_margin")
    surprisals = _extract(timeline, "surprisal")
    agreements = _extract_agreement(timeline)

    # Pattern 1: Context Loss
    # High entropy + low basin scores — model losing track of context
    if entropies and basin_scores:
        mean_ent = _mean(entropies[-5:])
        flat_basins = [s for layer_scores in basin_scores for s in layer_scores]
        mean_basin = _mean(flat_basins) if flat_basins else 1.0
        if mean_ent > 4.0 and mean_basin < 0.3:
            signals.append(
                CompoundSignal(
                    name="context_loss",
                    description="Model is losing track of context: high entropy combined with "
                    "low basin scores suggests middle-context tokens are being ignored.",
                    severity=0.75,
                    evidence=[
                        f"Mean entropy (last 5 steps): {mean_ent:.2f}",
                        f"Mean basin score: {mean_basin:.2f}",
                    ],
                )
            )

    # Pattern 2: Confident Confusion
    # High entropy but high margin — one strong pick but otherwise lost
    if entropies and margins:
        recent_ent = _mean(entropies[-5:])
        recent_margin = _mean(margins[-5:])
        if recent_ent > 4.0 and recent_margin > 0.3:
            signals.append(
                CompoundSignal(
                    name="confident_confusion",
                    description="Model shows high overall uncertainty but still has a dominant "
                    "token choice — may produce plausible-looking but unreliable output.",
                    severity=0.5,
                    evidence=[
                        f"Mean entropy (last 5): {recent_ent:.2f}",
                        f"Mean margin (last 5): {recent_margin:.2f}",
                    ],
                )
            )

    # Pattern 3: Degenerating Generation
    # Entropy rising + margin falling + surprisal rising over time
    if len(entropies) >= 10 and len(margins) >= 10 and len(surprisals) >= 10:
        ent_slope = _slope(entropies)
        margin_slope = _slope(margins)
        surprisal_slope = _slope(surprisals)
        if ent_slope > 0.05 and margin_slope < -0.01 and surprisal_slope > 0.05:
            signals.append(
                CompoundSignal(
                    name="degenerating_generation",
                    description="Output quality is degrading over time: entropy and surprisal "
                    "are rising while confidence margin is falling.",
                    severity=0.7,
                    evidence=[
                        f"Entropy slope: +{ent_slope:.3f}/step",
                        f"Margin slope: {margin_slope:.3f}/step",
                        f"Surprisal slope: +{surprisal_slope:.3f}/step",
                    ],
                )
            )

    # Pattern 4: Attention Bottleneck
    # Many collapsed heads + high entropy = model can't route information
    if layers_by_step and entropies:
        total_collapsed = 0
        total_heads_observed = 0
        for step_layers in layers_by_step:
            for layer in step_layers:
                if layer.attention_summary:
                    cc = getattr(layer.attention_summary, "collapsed_head_count", 0) or 0
                    total_collapsed += cc
                    mw = getattr(layer.attention_summary, "max_weight_per_head", None)
                    if mw is not None:
                        total_heads_observed += len(mw)
                    else:
                        fc = getattr(layer.attention_summary, "focused_head_count", 0) or 0
                        total_heads_observed += max(cc, fc, 1)
        if total_heads_observed > 0:
            collapse_rate = min(1.0, total_collapsed / total_heads_observed)
            mean_ent = _mean(entropies)
            if collapse_rate > 0.2 and mean_ent > 3.5:
                signals.append(
                    CompoundSignal(
                        name="attention_bottleneck",
                        description="Widespread attention collapse combined with high output "
                        "entropy suggests information flow is restricted.",
                        severity=0.65,
                        evidence=[
                            f"Collapse rate: {collapse_rate:.1%} of observed heads",
                            f"Mean entropy: {mean_ent:.2f}",
                        ],
                    )
                )

    # Pattern 5: Confident Repetition Risk
    # Low entropy + low surprisal + very high voter agreement — stuck but confident
    if entropies and surprisals:
        mean_ent = _mean(entropies[-5:])
        mean_surp = _mean(surprisals[-5:])
        if mean_ent < 2.0 and mean_surp < 1.0:
            if agreements:
                mean_agree = _mean(agreements[-5:])
                if mean_agree > 0.95:
                    signals.append(
                        CompoundSignal(
                            name="confident_repetition_risk",
                            description="Model is extremely confident with near-unanimous token "
                            "selection — may be locked into a repetitive pattern.",
                            severity=0.4,
                            evidence=[
                                f"Mean entropy: {mean_ent:.2f}",
                                f"Mean surprisal: {mean_surp:.2f}",
                                f"Mean voter agreement: {mean_agree:.2f}",
                            ],
                        )
                    )

    return signals


def _extract(timeline: List[TimelineStep], field: str) -> List[float]:
    """Extract a numeric field from each step's logits_summary."""
    vals: List[float] = []
    for s in timeline:
        if s.logits_summary:
            v = getattr(s.logits_summary, field, None)
            if v is not None:
                vals.append(float(v))
    return vals


def _extract_agreement(timeline: List[TimelineStep]) -> List[float]:
    """Extract topk_mass or voter_agreement from each step (prefer topk_mass)."""
    vals: List[float] = []
    for s in timeline:
        if not s.logits_summary:
            continue
        v = getattr(s.logits_summary, "topk_mass", None)
        if v is None:
            v = getattr(s.logits_summary, "voter_agreement", None)
        if v is not None:
            vals.append(float(v))
    return vals


def _mean(vals: List[float]) -> float:
    """Mean of values; 0.0 if empty."""
    return sum(vals) / len(vals) if vals else 0.0


def _slope(vals: List[float]) -> float:
    """Linear regression slope (least squares)."""
    if len(vals) < 2:
        return 0.0
    n = len(vals)
    mean_v = sum(vals) / n
    x_mean = (n - 1) / 2.0
    num = sum((i - x_mean) * (v - mean_v) for i, v in enumerate(vals))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den > 0 else 0.0
