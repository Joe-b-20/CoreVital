# ============================================================================
# CoreVital - Human-Readable Narratives (Phase-7)
#
# Purpose: Template-based 2–4 sentence summary from health flags, risk,
#          blamed layers, and warning signals. No LLM call.
# ============================================================================

from typing import List

from CoreVital.reporting.schema import HealthFlags


def build_narrative(
    health_flags: HealthFlags,
    risk_score: float,
    blamed_layers: List[int],
    warning_signals: List[str],
) -> str:
    """Build a short natural-language narrative from run data (Phase-7 template-based v1)."""
    parts: List[str] = []

    if risk_score > 0.7:
        parts.append("This run was high risk.")
    elif risk_score > 0.3:
        parts.append("This run had moderate risk.")
    else:
        parts.append("This run was low risk.")

    if health_flags.repetition_loop_detected:
        parts.append("Repetition was detected in the last few steps.")

    if health_flags.nan_detected or health_flags.inf_detected:
        parts.append("Numerical anomalies (NaN/Inf) were present.")

    if blamed_layers:
        layer_str = ", ".join(f"L{i}" for i in blamed_layers[:10])
        if len(blamed_layers) > 10:
            layer_str += f" (+{len(blamed_layers) - 10} more)"
        parts.append(f"Layers with anomalies or collapse: {layer_str}.")

    if warning_signals:
        if "entropy_rising" in warning_signals:
            parts.append("Entropy rose toward the end of generation.")
        if "high_entropy" in warning_signals and "entropy_rising" not in warning_signals:
            parts.append("High entropy was observed in several steps.")

    return " ".join(parts) if parts else "No notable issues detected."
