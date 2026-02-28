# Hidden state summary computation and health detection

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from CoreVital.errors import SummaryComputationError
from CoreVital.logging_utils import get_logger

from .utils import _random_projection_sketch

if TYPE_CHECKING:
    from CoreVital.config import HiddenSummariesConfig

logger = get_logger(__name__)

# Constants
L2_EXPLOSION_MULTIPLIER = 8.0  # Mid-layer L2 norm vs early-layer baseline (flan-t5 peaks at 5.7x)


def compute_hidden_summary(
    hidden_state: torch.Tensor,
    config: "HiddenSummariesConfig",
) -> Dict[str, Any]:
    """
    Compute summary statistics for hidden state tensor.

    Args:
        hidden_state: Hidden state tensor, shape (batch, seq_len, hidden_dim) or (seq_len, hidden_dim)
        config: Hidden summaries configuration

    Returns:
        Dictionary with summary statistics

    Raises:
        SummaryComputationError: If computation fails
    """
    try:
        if not config.enabled:
            return {}

        # Ensure 2D: (seq_len, hidden_dim)
        if hidden_state.dim() == 3:
            hidden_state = hidden_state[0]  # Take first batch

        # Move to CPU for computation
        hidden_state = hidden_state.cpu().float()

        CLIP_BOUND = 1e6
        clamped = torch.clamp(hidden_state, -CLIP_BOUND, CLIP_BOUND)
        was_clipped = not torch.equal(hidden_state, clamped)

        summary: Dict[str, Any] = {"clipped": was_clipped}
        if was_clipped:
            clip_mask = hidden_state.abs() > CLIP_BOUND
            summary["clip_fraction"] = float(clip_mask.float().mean().item())
            summary["clip_max_before"] = float(hidden_state.abs().max().item())

        hidden_state = clamped

        if "mean" in config.stats:
            summary["mean"] = float(hidden_state.mean().item())

        if "std" in config.stats:
            summary["std"] = float(hidden_state.std().item())

        if "l2_norm_mean" in config.stats:
            # L2 norm per token, then average
            l2_norms = torch.norm(hidden_state, p=2, dim=-1)
            summary["l2_norm_mean"] = float(l2_norms.mean().item())

        if "max_abs" in config.stats:
            summary["max_abs"] = float(hidden_state.abs().max().item())

        # Sketch via random projection (optional — disabled by default to minimize payload)
        if getattr(config.sketch, "enabled", False) and config.sketch.method == "randproj":
            sketch = _random_projection_sketch(
                hidden_state,
                config.sketch.dim,
                config.sketch.seed,
            )
            summary["sketch"] = sketch

        return summary

    except Exception as e:
        logger.exception("Failed to compute hidden summary")
        raise SummaryComputationError("Hidden state summary computation failed", details=str(e)) from e


def compute_encoder_hidden_states_summaries(
    encoder_hidden_states: List[torch.Tensor],
    config: "HiddenSummariesConfig",
) -> List[Dict[str, Any]]:
    """
    Compute summaries for encoder hidden states (one per encoder layer).

    Args:
        encoder_hidden_states: List of hidden state tensors, one per encoder layer
                             Each tensor shape: (batch, seq_len, hidden_dim) or (seq_len, hidden_dim)
        config: Hidden summaries configuration

    Returns:
        List of summary dictionaries, one per encoder layer

    Raises:
        SummaryComputationError: If computation fails
    """
    try:
        summaries = []
        for layer_idx, hidden_state in enumerate(encoder_hidden_states):
            if hidden_state is None:
                logger.debug(f"Encoder layer {layer_idx} hidden state is None")
                summaries.append({})
                continue

            # Use the standard hidden summary computation
            layer_summary = compute_hidden_summary(hidden_state, config)
            summaries.append(layer_summary)

        return summaries

    except Exception as e:
        logger.exception("Failed to compute encoder hidden states summaries")
        raise SummaryComputationError("Encoder hidden states summary computation failed", details=str(e)) from e


def compute_layer_transformations(
    hidden_states: List[torch.Tensor],
) -> List[float]:
    """Compute layer-to-layer transformation as 1 - cosine_similarity between consecutive layers.

    Each value represents how much the representation changed from one layer to the next,
    averaged across all prompt tokens. This is a telemetry/fingerprint signal useful for
    comparing runs and models. Interpretation of "high" vs "low" transformation is
    model-dependent and should not be used as a health indicator without per-model calibration.

    Args:
        hidden_states: List of hidden state tensors per layer.
            Each tensor: (batch, seq_len, hidden_dim) or (seq_len, hidden_dim)

    Returns:
        List of transformation values, length = len(hidden_states) - 1.
    """
    if len(hidden_states) < 2:
        return []

    transformations: List[float] = []
    for i in range(1, len(hidden_states)):
        prev_h = hidden_states[i - 1].float()
        curr_h = hidden_states[i].float()

        # Ensure 2D: (seq_len, hidden_dim)
        if prev_h.dim() == 3:
            prev_h = prev_h[0]
        if curr_h.dim() == 3:
            curr_h = curr_h[0]

        # Cosine similarity per token, then average
        cosine_sim = F.cosine_similarity(prev_h, curr_h, dim=-1).mean()
        transformation = 1.0 - cosine_sim.item()
        transformations.append(round(transformation, 6))

    return transformations


def detect_repetition_loop(
    hidden_state_buffer: List[torch.Tensor],
    token_id_buffer: Optional[List[int]] = None,
    threshold: float = 0.9995,
    consecutive_required: int = 3,
    profile: Optional[Any] = None,
) -> bool:
    """Detect if the model is stuck in a repetition loop using cosine similarity
    and optional token-level n-gram confirmation.

    Compares consecutive hidden state vectors in the buffer. If ``consecutive_required``+
    consecutive pairs have cosine_sim > threshold, a repetition loop is flagged via
    hidden states. When ``token_id_buffer`` is provided, the hidden-state signal is
    cross-checked with n-gram repetition in recent token IDs — if both agree, the
    detection is confirmed; if only hidden states fire, the result is still returned
    (backward-compatible) but a future confidence field may distinguish the two.

    Note on threshold: The default of 0.9995 accounts for the anisotropy problem
    in transformer last-layer representations. In float16 models (e.g., GPT-2 on CUDA),
    even non-repetitive tokens produce cosine similarities of 0.992-0.999. True
    repetition (same token repeated) gives ~1.0, so 0.9995 provides clear separation.

    Args:
        hidden_state_buffer: List of 1D hidden state vectors (last layer, last token).
            Typically the last 5 steps. Each tensor shape: (hidden_dim,)
        token_id_buffer: Optional list of recently generated token IDs. When provided,
            hidden-state similarity is cross-checked with n-gram repetition.
        threshold: Cosine similarity threshold for "same direction" (default 0.9995)
        consecutive_required: Number of consecutive high-similarity pairs needed (default 3)
        profile: Optional model profile; if set, threshold = profile.repetition_cosine_threshold

    Returns:
        True if repetition loop detected, False otherwise.
    """
    if profile is not None and hasattr(profile, "repetition_cosine_threshold"):
        threshold = float(profile.repetition_cosine_threshold)
    if len(hidden_state_buffer) < consecutive_required + 1:
        return False

    consecutive_count = 0
    for i in range(1, len(hidden_state_buffer)):
        prev_h = hidden_state_buffer[i - 1].float()
        curr_h = hidden_state_buffer[i].float()

        if prev_h.dim() > 1:
            prev_h = prev_h.flatten()
        if curr_h.dim() > 1:
            curr_h = curr_h.flatten()

        cosine_sim = F.cosine_similarity(prev_h.unsqueeze(0), curr_h.unsqueeze(0)).item()
        if cosine_sim > threshold:
            consecutive_count += 1
            if consecutive_count >= consecutive_required:
                break
        else:
            consecutive_count = 0

    hidden_repetition = consecutive_count >= consecutive_required

    if token_id_buffer is None:
        return hidden_repetition

    # Token-level n-gram confirmation
    if hidden_repetition and len(token_id_buffer) >= consecutive_required + 1:
        recent = token_id_buffer[-(consecutive_required + 3) :]
        for n in (3, 2):
            if len(recent) >= n * 2:
                ngrams = [tuple(recent[i : i + n]) for i in range(len(recent) - n + 1)]
                if len(ngrams) != len(set(ngrams)):
                    return True

    return hidden_repetition


def detect_mid_layer_anomaly(
    timeline_layers: List[List[Any]],
    num_layers: int,
    l2_multiplier: Optional[float] = None,
    profile: Optional[Any] = None,
) -> bool:
    """Detect runtime anomalies in middle layers.

    Empirically, mid-layer anomalies correlate with higher failure rates in generation
    quality; this is a heuristic and model-dependent. Calibration per model family is
    recommended.

    Checks (runtime anomalies only):
    1. NaN/Inf in mid-layer hidden states or attentions
    2. L2 norm explosion (dynamic 8× per-step early-layer baseline)

    Note: Attention collapse is NOT checked here — it's a model architecture
    property (e.g., GPT-2 has many collapsed heads by design), not a runtime
    anomaly. Collapse is already captured by attention_collapse_detected.

    Args:
        timeline_layers: List of step layers. Each entry is a list of LayerSummary-like
            objects with .anomalies, .hidden_summary attributes.
        num_layers: Total number of layers in the model.
        l2_multiplier: Override for L2 explosion multiplier (default from profile or constant).
        profile: Optional model profile; if set, l2_multiplier = profile.l2_explosion_multiplier

    Returns:
        True if mid-layer anomaly detected, False otherwise.
    """
    multiplier = L2_EXPLOSION_MULTIPLIER
    if l2_multiplier is not None:
        multiplier = l2_multiplier
    elif profile is not None and hasattr(profile, "l2_explosion_multiplier"):
        multiplier = float(profile.l2_explosion_multiplier)
    if not timeline_layers or num_layers < 3:
        return False

    mid_start = num_layers // 3
    mid_end = 2 * num_layers // 3

    # Check mid-layers across all steps using per-step baselines
    # Per-step baseline accounts for different sequence lengths across steps
    # (step 0 in CausalLM processes full prompt, steps 1+ process single tokens,
    #  causing very different L2 norm scales — a global baseline would false-positive)
    for step_layers in timeline_layers:
        # NaN/Inf check (no baseline needed)
        for layer_idx in range(mid_start, min(mid_end, len(step_layers))):
            layer = step_layers[layer_idx]
            if hasattr(layer, "anomalies") and layer.anomalies is not None:
                if layer.anomalies.has_nan or layer.anomalies.has_inf:
                    return True

        # Per-step L2 explosion check: baseline from THIS step's early layers
        step_early_norms: list[float] = []
        for layer_idx in range(min(mid_start, len(step_layers))):
            layer = step_layers[layer_idx]
            if hasattr(layer, "hidden_summary") and layer.hidden_summary is not None:
                norm = getattr(layer.hidden_summary, "l2_norm_mean", None)
                if norm is not None and isinstance(norm, (int, float)):
                    step_early_norms.append(float(norm))

        if step_early_norms:
            sorted_norms = sorted(step_early_norms)
            n = len(sorted_norms)
            baseline = sorted_norms[n // 2] if n % 2 == 1 else (sorted_norms[n // 2 - 1] + sorted_norms[n // 2]) / 2
            explosion_threshold = baseline * multiplier
        else:
            explosion_threshold = 1000.0  # Conservative fallback

        for layer_idx in range(mid_start, min(mid_end, len(step_layers))):
            layer = step_layers[layer_idx]
            if hasattr(layer, "hidden_summary") and layer.hidden_summary is not None:
                norm = getattr(layer.hidden_summary, "l2_norm_mean", None)
                if norm is not None and isinstance(norm, (int, float)) and float(norm) > explosion_threshold:
                    return True

    return False


def detect_tensor_anomalies(
    hidden_state: Optional[torch.Tensor] = None,
    attention: Optional[torch.Tensor] = None,
    cross_attention: Optional[torch.Tensor] = None,
) -> Dict[str, bool]:
    """
    Detect NaN/Inf in hidden state, self-attention, and cross-attention tensors for a single layer.

    Args:
        hidden_state: Hidden state tensor (any shape)
        attention: Self-attention tensor (any shape)
        cross_attention: Cross-attention tensor (Seq2Seq only, any shape)

    Returns:
        Dictionary with has_nan and has_inf boolean flags
    """
    has_nan = False
    has_inf = False

    for tensor in (hidden_state, attention, cross_attention):
        if tensor is None:
            continue
        if not isinstance(tensor, torch.Tensor):
            continue
        if torch.isnan(tensor).any().item():
            has_nan = True
        if torch.isinf(tensor).any().item():
            has_inf = True
        # Short-circuit if both already found
        if has_nan and has_inf:
            break

    return {"has_nan": has_nan, "has_inf": has_inf}
