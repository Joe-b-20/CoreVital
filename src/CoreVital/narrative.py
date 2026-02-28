# ============================================================================
# CoreVital - Human-Readable Narratives
#
# Purpose: Build actionable, data-specific 2-6 sentence narratives from
#          health flags, risk score, timeline metrics, compound signals,
#          and layer blame. No LLM call — pure template logic referencing
#          actual step indices, entropy values, and metric trends.
# ============================================================================

from typing import Any, List, Optional

from CoreVital.reporting.schema import HealthFlags, Summary, TimelineStep


def build_narrative(
    health_flags: HealthFlags,
    risk_score: float,
    risk_factors: List[str],
    blamed_layers: List[Any],
    warning_signals: List[str],
    timeline: List[TimelineStep],
    compound_signals: Optional[List[Any]] = None,
    summary: Optional[Summary] = None,
) -> str:
    """Build a specific, actionable narrative from run data."""
    parts: List[str] = []

    # Lead with risk level and primary cause
    if risk_score > 0.7:
        primary = risk_factors[0] if risk_factors else "multiple signals"
        parts.append(f"High risk (score: {risk_score:.2f}), primarily driven by {_humanize_factor(primary)}.")
    elif risk_score > 0.3:
        parts.append(f"Moderate risk (score: {risk_score:.2f}).")
    else:
        parts.append(f"Low risk (score: {risk_score:.2f}). No significant anomalies detected.")

    # Entropy specifics
    entropies = [
        s.logits_summary.entropy for s in timeline if s.logits_summary and s.logits_summary.entropy is not None
    ]
    if entropies:
        mean_ent = sum(entropies) / len(entropies)
        max_ent = max(entropies)
        max_step_obj = next(s for s in timeline if s.logits_summary and s.logits_summary.entropy == max_ent)
        max_step_idx = max_step_obj.step_index
        if max_ent > 4.0:
            token_text = max_step_obj.token.token_text if max_step_obj.token else "?"
            parts.append(
                f"Peak entropy of {max_ent:.1f} bits at step {max_step_idx} "
                f"(mean: {mean_ent:.1f}); the model was most uncertain "
                f'when generating "{token_text or "?"}".'
            )
        if "entropy_rising" in warning_signals and len(entropies) >= 6:
            early = sum(entropies[:3]) / 3
            late = sum(entropies[-3:]) / 3
            parts.append(
                f"Entropy rose from ~{early:.1f} to ~{late:.1f} bits over the "
                f"course of generation, suggesting progressive degradation."
            )

    # Compound signals (from Issue 6)
    if compound_signals:
        for cs in compound_signals[:2]:
            parts.append(f"{cs.description}")

    # Layer blame specifics — handles both List[dict] (Issue 7 rich blame)
    # and List[int] (current simple blame)
    if blamed_layers:
        if isinstance(blamed_layers[0], dict):
            top_blame = sorted(blamed_layers, key=lambda b: b.get("severity", 0), reverse=True)[:3]
            for b in top_blame:
                reasons_str = "; ".join(b.get("reasons", []))
                if reasons_str:
                    parts.append(f"Layer {b['layer']}: {reasons_str}.")
        else:
            layer_str = ", ".join(f"L{i}" for i in blamed_layers[:5])
            if len(blamed_layers) > 5:
                layer_str += f" (+{len(blamed_layers) - 5} more)"
            parts.append(f"Layers with anomalies or collapse: {layer_str}.")

    # Actionable recommendation
    if risk_score > 0.7:
        if "repetition_loop" in risk_factors:
            parts.append("Consider: lower temperature, add repetition penalty, or shorten max_new_tokens.")
        elif "mid_layer_anomaly" in risk_factors:
            parts.append(
                "Consider: check input encoding, try different precision (fp32 vs fp16), or use a different model."
            )
        elif "elevated_entropy" in risk_factors:
            parts.append(
                "Consider: refine the prompt for clarity, provide more context, "
                "or try a model better suited to this domain."
            )

    return " ".join(parts) if parts else "No notable issues detected."


def _humanize_factor(factor: str) -> str:
    """Convert internal risk factor names to readable text."""
    mapping = {
        "repetition_loop": "a repetition loop in the last generated tokens",
        "mid_layer_anomaly": "anomalous behavior in middle transformer layers",
        "attention_collapse": "attention head collapse",
        "elevated_entropy": "elevated output uncertainty",
        "nan_or_inf": "numerical instability (NaN/Inf)",
        "entropy_rising": "rising entropy over generation",
        "low_confidence_margin": "low confidence between top token choices",
        "low_topk_mass": "low top-K probability mass (dispersed predictions)",
        "elevated_surprisal": "elevated surprisal (unexpected token choices)",
    }
    return mapping.get(factor, factor.replace("_", " "))


def _token_at_step(timeline: List[TimelineStep], step_idx: int) -> str:
    """Return the token text generated at a given step index."""
    if step_idx < len(timeline) and timeline[step_idx].token:
        return timeline[step_idx].token.token_text or "?"
    return "?"
