# ============================================================================
# CoreVital - Risk Score and Layer Blame (Phase-2)
#
# Purpose: Compute a single risk score per run from health flags and
#          attribute blame to layers (anomalies, collapse).
# Inputs: HealthFlags, Summary, timeline (optional), layers_by_step (optional)
# Outputs: risk_score in [0, 1], risk_factors list, blamed_layers list
# ============================================================================

import math
import statistics
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

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
        collapse_sev = health_flags.attention_collapse_severity
        score = max(score, collapse_sev if collapse_sev is not None else 0.3)
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

    # --- Boolean flag ceilings (hard floors via max, never additive) ---
    bool_ceilings: List[float] = []
    if health_flags.repetition_loop_detected:
        bool_ceilings.append(0.9)
        factors.append("repetition_loop")
    if health_flags.mid_layer_anomaly_detected:
        bool_ceilings.append(0.7)
        factors.append("mid_layer_anomaly")
    if health_flags.attention_collapse_detected:
        collapse_sev = health_flags.attention_collapse_severity
        bool_ceilings.append(collapse_sev if collapse_sev is not None else 0.15)
        factors.append("attention_collapse")

    # --- Continuous metric components (additive) ---
    # Filter non-finite values: NaN/Inf can reach logits_summary when
    # detect_tensor_anomalies doesn't inspect logits tensors directly.
    entropies = [
        s.logits_summary.entropy
        for s in timeline
        if s.logits_summary and s.logits_summary.entropy is not None and math.isfinite(s.logits_summary.entropy)
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
            if first_third > 1e-9 and last_third > first_third * 1.3:
                trend_component = min(0.2, (last_third - first_third) / first_third * 0.1)
                components.append(trend_component)
                factors.append("entropy_rising")

    margins: List[float] = [
        s.logits_summary.top_k_margin  # type: ignore[misc]
        for s in timeline
        if s.logits_summary
        and s.logits_summary.top_k_margin is not None
        and math.isfinite(s.logits_summary.top_k_margin)
    ]
    if margins:
        mean_margin = sum(margins) / len(margins)
        margin_component = max(0.0, (1.0 - mean_margin * 5.0)) * 0.2
        if margin_component > 0.05:
            components.append(margin_component)
            factors.append("low_confidence_margin")

    agreements: List[float] = []
    for s in timeline:
        if not s.logits_summary:
            continue
        val = getattr(s.logits_summary, "topk_mass", None)
        if val is None:
            val = getattr(s.logits_summary, "voter_agreement", None)
        if val is not None and math.isfinite(val):
            agreements.append(val)
    if agreements:
        mean_agreement = sum(agreements) / len(agreements)
        agreement_component = max(0.0, (1.0 - mean_agreement)) * 0.15
        if agreement_component > 0.03:
            components.append(agreement_component)
            factors.append("low_topk_mass")

    surprisals: List[float] = [
        s.logits_summary.surprisal  # type: ignore[misc]
        for s in timeline
        if s.logits_summary and s.logits_summary.surprisal is not None and math.isfinite(s.logits_summary.surprisal)
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

    # Combine: boolean ceilings dominate via max; continuous + compound are additive
    bool_max = max(bool_ceilings) if bool_ceilings else 0.0
    continuous_sum = sum(components)
    score = min(1.0, bool_max + continuous_sum) if (bool_ceilings or components) else 0.0

    return score, factors


def compute_layer_blame(
    layers_by_step: List[List[LayerSummary]],
    num_layers: Optional[int] = None,
) -> List[Dict]:
    """Identify layers that contributed to risk with structured evidence.

    Checks four conditions per layer:
    1. NaN/Inf in anomalies (severity 1.0)
    2. Attention collapse rate across steps (severity 0.4 when >50%)
    3. L2 norm z-score outlier vs cross-layer baseline (severity 0.5 when z>2.5)
    4. L2 norm instability within layer across steps (severity 0.3 when CV>0.5)

    Args:
        layers_by_step: For each step, a list of LayerSummary (same order as layer_index).
        num_layers: Optional total layer count (unused currently; reserved for future use).

    Returns:
        List of dicts: [{"layer": int, "reasons": [str], "severity": float}]
        Sorted by layer index. Only layers with at least one reason are included.
    """
    if not layers_by_step:
        return []

    layer_stats: Dict[int, Dict] = {}

    for step_layers in layers_by_step:
        for layer in step_layers:
            idx = layer.layer_index
            if idx not in layer_stats:
                layer_stats[idx] = {
                    "nan_inf": False,
                    "collapsed_steps": 0,
                    "l2_norms": [],
                    "entropy_means": [],
                    "total_steps": 0,
                }
            stats = layer_stats[idx]
            stats["total_steps"] += 1

            if layer.anomalies and (layer.anomalies.has_nan or layer.anomalies.has_inf):
                stats["nan_inf"] = True

            if layer.attention_summary:
                cc = getattr(layer.attention_summary, "collapsed_head_count", 0)
                if cc and cc > 0:
                    stats["collapsed_steps"] += 1
                ent = getattr(layer.attention_summary, "entropy_mean", None)
                if ent is not None:
                    stats["entropy_means"].append(ent)

            if layer.hidden_summary:
                norm = getattr(layer.hidden_summary, "l2_norm_mean", None)
                if norm is not None:
                    stats["l2_norms"].append(norm)

    # Cross-layer L2 baseline for z-score computation
    all_l2_means: List[Tuple[int, float]] = []
    for idx, stats in layer_stats.items():
        if stats["l2_norms"]:
            all_l2_means.append((idx, sum(stats["l2_norms"]) / len(stats["l2_norms"])))

    global_l2_mean = (sum(m for _, m in all_l2_means) / len(all_l2_means)) if all_l2_means else 0.0
    global_l2_std = 0.0
    if len(all_l2_means) >= 2:
        global_l2_std = statistics.stdev(m for _, m in all_l2_means)

    blamed: List[Dict] = []
    for idx, stats in sorted(layer_stats.items()):
        reasons: List[str] = []
        severity = 0.0

        # 1. NaN/Inf
        if stats["nan_inf"]:
            reasons.append("NaN/Inf detected")
            severity = max(severity, 1.0)

        # 2. Attention collapse rate
        total = max(1, stats["total_steps"])
        collapse_rate = stats["collapsed_steps"] / total
        if collapse_rate > 0.5:
            reasons.append(f"Attention collapse in {collapse_rate:.0%} of steps")
            severity = max(severity, 0.4)

        # 3. L2 norm z-score outlier
        if stats["l2_norms"] and global_l2_std > 0:
            layer_l2_mean = sum(stats["l2_norms"]) / len(stats["l2_norms"])
            z_score = (layer_l2_mean - global_l2_mean) / global_l2_std
            if z_score > 2.5:
                reasons.append(f"L2 norm outlier (z={z_score:.1f})")
                severity = max(severity, 0.5)

        # 4. L2 norm instability (coefficient of variation within layer across steps)
        if len(stats["l2_norms"]) >= 3:
            l2_cv = statistics.stdev(stats["l2_norms"]) / max(1e-10, statistics.mean(stats["l2_norms"]))
            if l2_cv > 0.5:
                reasons.append(f"Unstable L2 norms (CV={l2_cv:.2f})")
                severity = max(severity, 0.3)

        if reasons:
            blamed.append(
                {
                    "layer": idx,
                    "reasons": reasons,
                    "severity": round(severity, 2),
                }
            )

    return blamed


def compute_layer_blame_flat(layers_by_step: List[List[LayerSummary]]) -> List[int]:
    """Backward-compatible wrapper: return flat sorted list of blamed layer indices.

    Delegates to compute_layer_blame and extracts the "layer" field.
    """
    return [b["layer"] for b in compute_layer_blame(layers_by_step)]
