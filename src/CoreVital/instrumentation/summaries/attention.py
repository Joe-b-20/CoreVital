# Attention summary computation: compute_attention_summary, compute_basin_scores, extract_sparse_attention

import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch

from CoreVital.errors import SummaryComputationError
from CoreVital.logging_utils import get_logger

if TYPE_CHECKING:
    from CoreVital.config import AttentionSummariesConfig

logger = get_logger(__name__)

# Constants
COLLAPSED_HEAD_ENTROPY_THRESHOLD = 0.1  # Legacy raw-nats threshold (deprecated; use NORMALIZED_COLLAPSED_THRESHOLD)
NORMALIZED_COLLAPSED_THRESHOLD = 0.03  # Normalized entropy in [0,1]; below this = collapsed head
FOCUSED_HEAD_CONCENTRATION_THRESHOLD = 0.9  # Avg max attention above this = very focused head


def compute_attention_summary(
    attention: Any,  # Changed from torch.Tensor to Any for safe checking
    config: "AttentionSummariesConfig",
    profile: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Compute summary statistics for attention tensor.

    Supports both self-attention and cross-attention tensors:
    - Self-attention: shape (batch, heads, seq_len, seq_len) or (heads, seq_len, seq_len)
    - Cross-attention: shape (batch, heads, target_seq_len, source_seq_len) or (heads, target_seq_len, source_seq_len)
      where target_seq_len (query) and source_seq_len (key) can differ

    Args:
        attention: Attention tensor (self-attention or cross-attention)
        config: Attention summaries configuration
        profile: Optional model profile for collapsed/focused thresholds

    Returns:
        Dictionary with summary statistics

    Raises:
        SummaryComputationError: If computation fails
    """
    # collapsed_threshold from profile is deprecated; collapse detection now uses
    # NORMALIZED_COLLAPSED_THRESHOLD in [0,1] space (Issue 21).
    focused_threshold = (
        float(profile.focused_head_concentration_threshold)
        if profile is not None and hasattr(profile, "focused_head_concentration_threshold")
        else FOCUSED_HEAD_CONCENTRATION_THRESHOLD
    )
    try:
        if not config.enabled:
            logger.debug("Attention summary computation disabled in config")
            return {}

        if attention is None:
            logger.debug("Attention tensor is None")
            return {}

        # Handle tuple/list of layers (shouldn't happen when called from report_builder, but handle it)
        if isinstance(attention, (list, tuple)):
            if len(attention) == 0:
                logger.debug("Attention is empty list/tuple")
                return {}
            # If it's a list/tuple, take the first element (we're already getting per-layer tensors)
            attention = attention[0] if len(attention) > 0 else None
            if attention is None:
                logger.debug("Attention element is None after extraction")
                return {}

        # Safety check: if after extraction it's still None or not a tensor
        if not isinstance(attention, torch.Tensor):
            logger.warning(
                f"Attention is not a tensor (type: {type(attention).__name__}). "
                f"Shape/length: {getattr(attention, 'shape', getattr(attention, '__len__', 'N/A'))}"
            )
            return {}

        # Handle different tensor shapes
        # Expected shapes for self-attention: (batch, heads, seq_len, seq_len) or (heads, seq_len, seq_len)
        # Expected shapes for cross-attention: (batch, heads, target_len, source_len) or (heads, target_len, source_len)
        original_shape = attention.shape

        # Ensure 3D: (heads, target_len, source_len) or (heads, seq_len, seq_len)
        if attention.dim() == 4:
            attention = attention[0]  # Take first batch
        elif attention.dim() != 3:
            logger.warning(f"Unexpected attention tensor shape: {original_shape}, expected 3D or 4D")
            return {}

        # For cross-attention, the last two dimensions may differ (target_seq_len != source_seq_len)
        # This is fine - we compute entropy over the source dimension (last dim) for each target position

        # Move to CPU
        attention = attention.cpu().float()

        # Issue 23: Renormalize by division, not softmax.
        # Attention is already post-softmax; numerical drift means it may not sum to exactly 1.
        # Applying softmax again would exponentially reweight the distribution.
        attention_sum = attention.sum(dim=-1, keepdim=True)
        if not torch.allclose(attention_sum, torch.ones_like(attention_sum), atol=1e-3):
            logger.debug("Attention weights not normalized, renormalizing by sum")
            attention = attention / attention_sum.clamp(min=1e-10)

        summary: Dict[str, Any] = {}

        # ── Per-head entropy (shared intermediate) ────────────────────
        # Entropy of attention distribution over keys for each query position
        # attention shape: (heads, target_len, source_len)
        need_entropy = any(
            s in config.stats
            for s in (
                "entropy_mean", "entropy_min", "entropy_max",
                "entropy_mean_normalized",
                "collapsed_head_count", "focused_head_count",
            )
        )

        per_head_entropy = None
        per_head_entropy_norm = None
        if need_entropy:
            # Issue 10: clamp instead of additive epsilon — preserves actual values
            # for weights > 1e-10 and only floors true zeros.
            log_attn = torch.log(torch.clamp(attention, min=1e-10))
            entropy = -(attention * log_attn).sum(dim=-1)  # (heads, target_len)

            # Issue 21: normalize by log(K) so entropy lives in [0, 1]
            source_len = attention.shape[-1]
            max_entropy = math.log(source_len) if source_len > 1 else 1.0
            normalized_entropy = entropy / max_entropy  # (heads, target_len)

            per_head_entropy = entropy.mean(dim=-1)  # raw nats, (heads,)
            per_head_entropy_norm = normalized_entropy.mean(dim=-1)  # [0,1], (heads,)

            if "entropy_mean" in config.stats:
                summary["entropy_mean"] = float(per_head_entropy.mean().item())

            if "entropy_mean_normalized" in config.stats:
                summary["entropy_mean_normalized"] = float(per_head_entropy_norm.mean().item())

            if "entropy_min" in config.stats:
                summary["entropy_min"] = float(per_head_entropy.min().item())

            if "entropy_max" in config.stats:
                summary["entropy_max"] = float(per_head_entropy.max().item())

            # Issue 21: collapse detection uses normalized entropy
            if "collapsed_head_count" in config.stats:
                norm_threshold = NORMALIZED_COLLAPSED_THRESHOLD
                summary["collapsed_head_count"] = int((per_head_entropy_norm < norm_threshold).sum().item())

            # Focused heads: heads where mean max-attention per query > threshold (concentration).
            # High concentration = sharp/focused attention on specific keys; low entropy = sharp.
            # (High entropy = diffuse/overloaded attention; we use concentration, not entropy, here.)
            if "focused_head_count" in config.stats:
                # focused_head_count: heads with concentration_max > 0.9
                # We compute this from per-head max attention below
                pass  # Will be filled in concentration section

        # ── Per-head concentration (shared intermediate) ──────────────
        need_concentration = any(
            s in config.stats for s in ("concentration_max", "concentration_min", "focused_head_count")
        )

        if need_concentration:
            # Max over keys for each (head, query), then mean over queries
            max_attn_per_query = attention.max(dim=-1)[0]  # (heads, target_len)
            per_head_max = max_attn_per_query.mean(dim=-1)  # (heads,)

            if "concentration_max" in config.stats:
                summary["concentration_max"] = float(max_attn_per_query.max().item())

            if "concentration_min" in config.stats:
                summary["concentration_min"] = float(max_attn_per_query.min().item())

            if "focused_head_count" in config.stats:
                summary["focused_head_count"] = int((per_head_max > focused_threshold).sum().item())

        # Per-head max weight: strongest single connection per head (Voita et al. 2019, #6)
        # Specialist heads often have ~80% max weight; mean-only aggregation hides failures
        # Gated by config.stats so trimming attention stats actually disables this field
        if "max_weight_per_head" in config.stats:
            num_heads = attention.size(0)
            max_per_head = attention.view(num_heads, -1).max(dim=1)[0]  # (heads,)
            summary["max_weight_per_head"] = [round(float(x), 6) for x in max_per_head.tolist()]

        return summary

    except Exception as e:
        logger.exception("Failed to compute attention summary")
        raise SummaryComputationError("Attention summary computation failed", details=str(e)) from e


def extract_sparse_attention(
    attention: torch.Tensor,
    threshold: float = 0.01,
    max_per_head: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Extract sparse attention connections above threshold for one layer (all heads).

    Uses vectorized torch.where — no Python loops over query positions.
    When max_per_head is set, keeps only the top-N connections by weight per head to limit payload.

    Args:
        attention: Attention tensor, shape (batch, heads, seq_len, seq_len) or (heads, seq_len, seq_len)
        threshold: Minimum attention weight to store (default 0.01)
        max_per_head: If set, keep at most this many connections per head (by weight, descending)

    Returns:
        List of dicts (one per head), each with query_indices, key_indices, weights lists.
    """
    # Ensure 3D: (heads, seq_len, seq_len)
    if attention.dim() == 4:
        attention = attention[0]
    attention = attention.cpu().float()

    num_heads = attention.shape[0]
    heads: List[Dict[str, Any]] = []

    for head_idx in range(num_heads):
        head_attn = attention[head_idx]  # (seq_len, seq_len)
        mask = head_attn > threshold
        query_idx, key_idx = torch.where(mask)
        weights = head_attn[mask]

        if max_per_head is not None and weights.numel() > max_per_head:
            top_vals, top_pos = torch.topk(weights, max_per_head, largest=True, sorted=False)
            query_idx = query_idx[top_pos]
            key_idx = key_idx[top_pos]
            weights = top_vals

        heads.append(
            {
                "query_indices": query_idx.tolist(),
                "key_indices": key_idx.tolist(),
                "weights": [round(float(w), 4) for w in weights.tolist()],
            }
        )

    return heads


def compute_basin_scores(
    attention: torch.Tensor,
) -> List[float]:
    """Compute basin score per attention head for one layer.

    Basin score = avg(middle_attention) / avg(boundary_attention)
    Where middle = middle third, boundaries = first + last third,
    averaged across all query positions for this head.

    Low basin_score (< 0.3) = head ignores middle tokens ("Lost in the Middle").

    Args:
        attention: Attention tensor, shape (batch, heads, seq_len, seq_len) or (heads, seq_len, seq_len)

    Returns:
        List of basin scores, one per head.
    """
    if attention.dim() == 4:
        attention = attention[0]
    attention = attention.cpu().float()

    num_heads, seq_len, _ = attention.shape

    if seq_len < 3:
        return [1.0] * num_heads

    mid_start = seq_len // 3
    mid_end = 2 * seq_len // 3

    # Vectorized over all heads: sum over key dim, mean over query dim
    middle_attn = attention[:, :, mid_start:mid_end].sum(dim=-1).mean(dim=-1)  # (heads,)
    boundary_mask = torch.ones(seq_len, dtype=torch.bool)
    boundary_mask[mid_start:mid_end] = False
    boundary_attn = attention[:, :, boundary_mask].sum(dim=-1).mean(dim=-1)  # (heads,)

    scores = (middle_attn / (boundary_attn + 1e-10)).tolist()
    return [round(s, 4) for s in scores]
