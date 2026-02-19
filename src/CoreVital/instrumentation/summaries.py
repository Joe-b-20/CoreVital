# ============================================================================
# CoreVital - Summary Computation
#
# Purpose: Compute lightweight summaries from tensors (hidden states, attention, logits)
# Inputs: Tensors from model forward pass
# Outputs: Summary dictionaries with statistics
# Dependencies: torch, numpy, config
# Usage: summary = compute_hidden_summary(tensor, config)
#
# Changelog:
#   2026-01-13: Initial summary functions for Phase-0
#   2026-01-14: Fixed attention summary computation - improved tensor shape handling and normalization
#                Added better error handling and logging for attention tensor processing
#                Fixed logits summary to properly handle topk extraction
#   2026-01-15: Enhanced attention summary to support cross-attention tensors (different source/target lengths)
#                Added compute_encoder_hidden_states_summaries helper for Seq2Seq models
#   2026-01-21: Phase-0.5 hardening - replaced magic number with MIN_TOPK_FOR_ENTROPY constant
#   2026-02-07: Phase-1a — Enhanced metrics:
#                - Numerical stability: log_softmax for entropy computation
#                - New logit metrics: top_k_margin, voter_agreement, perplexity, surprisal
#                - Enhanced attention: entropy_max, concentration_min, collapsed/focused counts
#                - New: detect_tensor_anomalies() for NaN/Inf detection
#   2026-02-10: Phase-1b — Prompt telemetry summary functions:
#                - extract_sparse_attention(): vectorized torch.where per head → SoA
#                - compute_basin_scores(): middle/boundary ratio per head
#                - compute_layer_transformations(): cosine similarity between consecutive layers
#                - compute_prompt_surprisal(): CrossEntropyLoss(reduction='none') on prompt logits
#   2026-02-10: Phase-1c — Health flag detection functions:
#                - detect_repetition_loop(): cosine similarity > 0.9995 on transient buffer
#                  (threshold accounts for float16 anisotropy — non-repetitive GPT-2 gives 0.992-0.999)
#                - detect_mid_layer_anomaly(): per-step dynamic L2 baseline + NaN/Inf
#                  (attention collapse excluded — structural not runtime; per-step baselines
#                   to handle CausalLM step-0 prompt processing scale difference)
#   2026-02-10: Phase-1c fixes (CI/Codex review):
#                - detect_repetition_loop(): enforce CONSECUTIVE similarity — counter resets
#                  on a non-matching pair (previously counted total, false-positived on
#                  [high, low, high, high] patterns)
#                - compute_logits_summary(): entropy always computed internally so that
#                  stats=["perplexity"] works without "entropy" in the list
#                - detect_tensor_anomalies(): added cross_attention parameter — Seq2Seq
#                  cross-attention NaN/Inf was silently missed
#                - detect_mid_layer_anomaly(): L2 multiplier raised from 5× to 8×
#                  (flan-t5-small peaks at 5.7× mid/early ratio in normal operation,
#                   GPT-2 at 3.1×; 5× false-positived on flan-t5-small)
# ============================================================================

import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from CoreVital.errors import SummaryComputationError
from CoreVital.logging_utils import get_logger

if TYPE_CHECKING:
    from CoreVital.config import AttentionSummariesConfig, HiddenSummariesConfig, LogitsSummariesConfig


logger = get_logger(__name__)

# Constants
MIN_TOPK_FOR_ENTROPY = 50  # Minimum top-k for good entropy estimate
COLLAPSED_HEAD_ENTROPY_THRESHOLD = 0.1  # Entropy below this = nearly all weight on one token
FOCUSED_HEAD_CONCENTRATION_THRESHOLD = 0.9  # Avg max attention above this = very focused head
VOTER_AGREEMENT_TOP_K = 10  # Number of top tokens for voter agreement probability mass
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

        summary: Dict[str, Any] = {}

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


def compute_attention_summary(
    attention: Any,  # Changed from torch.Tensor to Any for safe checking
    config: "AttentionSummariesConfig",
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

    Returns:
        Dictionary with summary statistics

    Raises:
        SummaryComputationError: If computation fails
    """
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

        # Normalize attention weights if needed (they should already be normalized from softmax)
        # But we'll check and normalize if necessary to ensure they sum to 1
        # This handles cases where attention might not be properly normalized
        attention_sum = attention.sum(dim=-1, keepdim=True)
        if not torch.allclose(attention_sum, torch.ones_like(attention_sum), atol=1e-3):
            logger.debug("Attention weights not normalized, applying softmax")
            attention = F.softmax(attention, dim=-1)

        summary: Dict[str, Any] = {}

        # ── Per-head entropy (shared intermediate) ────────────────────
        # Entropy of attention distribution over keys for each query position
        # attention shape: (heads, target_len, source_len)
        need_entropy = any(
            s in config.stats
            for s in ("entropy_mean", "entropy_min", "entropy_max", "collapsed_head_count", "focused_head_count")
        )

        per_head_entropy = None
        if need_entropy:
            # Use log_softmax for numerical stability
            log_attn = torch.log(attention + 1e-10)
            # Entropy per (head, query): -sum(p * log(p)) over keys
            entropy = -(attention * log_attn).sum(dim=-1)  # (heads, target_len)
            # Average over query positions to get per-head scalar
            per_head_entropy = entropy.mean(dim=-1)  # (heads,)

            if "entropy_mean" in config.stats:
                summary["entropy_mean"] = float(per_head_entropy.mean().item())

            if "entropy_min" in config.stats:
                summary["entropy_min"] = float(per_head_entropy.min().item())

            if "entropy_max" in config.stats:
                summary["entropy_max"] = float(per_head_entropy.max().item())

            if "collapsed_head_count" in config.stats:
                summary["collapsed_head_count"] = int(
                    (per_head_entropy < COLLAPSED_HEAD_ENTROPY_THRESHOLD).sum().item()
                )

            # Focused heads: concentration > threshold (using high entropy as proxy)
            # Focused = head has high concentration (entropy > 4.0 = overloaded/diffuse)
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
                summary["focused_head_count"] = int((per_head_max > FOCUSED_HEAD_CONCENTRATION_THRESHOLD).sum().item())

        return summary

    except Exception as e:
        logger.exception("Failed to compute attention summary")
        raise SummaryComputationError("Attention summary computation failed", details=str(e)) from e


def compute_logits_summary(
    logits: torch.Tensor,
    tokenizer: Any,
    config: "LogitsSummariesConfig",
    actual_token_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute summary statistics for logits tensor.

    Args:
        logits: Logits tensor, shape (batch, seq_len, vocab_size) or (vocab_size,)
        tokenizer: Tokenizer for decoding token IDs
        config: Logits summaries configuration
        actual_token_id: The actually generated token ID (for surprisal computation).
                        If None, surprisal is skipped.

    Returns:
        Dictionary with summary statistics

    Raises:
        SummaryComputationError: If computation fails
    """
    try:
        if not config.enabled:
            return {}

        # Get last token logits
        if logits.dim() == 3:
            logits = logits[0, -1, :]  # (vocab_size,)
        elif logits.dim() == 2:
            logits = logits[-1, :]

        # Move to CPU
        logits = logits.cpu().float()

        summary: Dict[str, Any] = {}

        # ── Shared intermediates (compute once, use many) ──────────────
        # Use log_softmax for numerical stability (log-sum-exp trick)
        log_probs_full = F.log_softmax(logits, dim=-1)
        probs_full = torch.exp(log_probs_full)

        # Top-k values (shared by margin, agreement, topk_probs)
        topk_k = max(config.topk, MIN_TOPK_FOR_ENTROPY)
        topk_values, topk_indices = torch.topk(logits, k=min(topk_k, len(logits)))
        topk_probs = probs_full[topk_indices]

        # ── Shannon Entropy (numerically stable via log_softmax) ───────
        # Always compute internally — perplexity depends on it even if "entropy" not in stats
        p_log_p = probs_full * log_probs_full
        entropy_nats = -torch.nan_to_num(p_log_p, nan=0.0).sum()
        entropy_bits = float(entropy_nats.item()) / math.log(2)
        if "entropy" in config.stats:
            summary["entropy"] = entropy_bits

        # ── Top-1 to Top-2 margin (legacy field, kept for backward compat) ──
        if "top1_top2_margin" in config.stats:
            if len(topk_probs) >= 2:
                margin = topk_probs[0] - topk_probs[1]
                summary["top1_top2_margin"] = float(margin.item())
            else:
                summary["top1_top2_margin"] = 0.0

        # ── Top-K Margin (Phase-1a: same as top1_top2_margin but separate field) ──
        if "top_k_margin" in config.stats:
            if len(topk_probs) >= 2:
                summary["top_k_margin"] = float((topk_probs[0] - topk_probs[1]).item())
            else:
                summary["top_k_margin"] = 0.0

        # ── Voter Agreement (top-K probability mass) ───────────────────
        if "voter_agreement" in config.stats:
            top_n = min(VOTER_AGREEMENT_TOP_K, len(topk_probs))
            agreement = float(topk_probs[:top_n].sum().item())
            summary["voter_agreement"] = agreement

        # ── Perplexity (2^entropy) ────────────────────────────────────
        if "perplexity" in config.stats:
            summary["perplexity"] = float(2.0**entropy_bits)

        # ── Surprisal (-log₂(p_actual_token)) ────────────────────────
        if "surprisal" in config.stats and actual_token_id is not None:
            if 0 <= actual_token_id < len(log_probs_full):
                # -log₂(p) = -log_p / log(2)
                log_p_actual = log_probs_full[actual_token_id]
                surprisal = float(-log_p_actual.item()) / math.log(2)
                summary["surprisal"] = surprisal

        # ── Top-k token probabilities ─────────────────────────────────
        if "topk_probs" in config.stats:
            topk_list = []
            for token_id, prob in zip(
                topk_indices[: config.topk],
                topk_probs[: config.topk],
                strict=False,
            ):
                token_id = int(token_id.item())
                token_text = tokenizer.decode([token_id])
                topk_list.append(
                    {
                        "token_id": token_id,
                        "token_text": token_text,
                        "prob": round(float(prob.item()), 3),
                    }
                )
            summary["topk"] = topk_list

        return summary

    except Exception as e:
        logger.exception("Failed to compute logits summary")
        raise SummaryComputationError("Logits summary computation failed", details=str(e)) from e


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
        # Too short to meaningfully split into thirds
        return [1.0] * num_heads

    mid_start = seq_len // 3
    mid_end = 2 * seq_len // 3

    # Build a boolean mask for the middle third of keys
    mid_mask = torch.zeros(seq_len, dtype=torch.bool)
    mid_mask[mid_start:mid_end] = True

    scores: List[float] = []
    for head_idx in range(num_heads):
        head_attn = attention[head_idx]  # (seq_len, seq_len)
        # Sum over key dimension, then average over query dimension
        middle_attn = head_attn[:, mid_mask].sum(dim=-1).mean()  # scalar
        boundary_attn = head_attn[:, ~mid_mask].sum(dim=-1).mean()  # scalar
        basin = (middle_attn / (boundary_attn + 1e-10)).item()
        scores.append(round(basin, 4))

    return scores


def compute_layer_transformations(
    hidden_states: List[torch.Tensor],
) -> List[float]:
    """Compute layer-to-layer transformation as 1 - cosine_similarity between consecutive layers.

    Each value represents how much the representation changed from one layer to the next,
    averaged across all prompt tokens.

    High transformation (>0.5) = layer performing significant work (good).
    Low transformation (<0.1) = layer nearly identity (possible deadness).

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


def compute_prompt_surprisal(
    logits: torch.Tensor,
    prompt_token_ids: List[int],
) -> List[float]:
    """Compute per-token surprisal for prompt tokens using CrossEntropyLoss.

    Surprisal = -log₂(p(actual_token | context))
    Computed via manual shift (autoregressive: predict token[i+1] from logits[i]).

    Args:
        logits: Prompt logits, shape (batch, seq_len, vocab_size) or (seq_len, vocab_size)
        prompt_token_ids: Prompt token IDs for alignment

    Returns:
        List of surprisal values in bits, length = len(prompt_token_ids) - 1
        (first token has no context, so no surprisal).
    """
    if logits.dim() == 3:
        logits = logits[0]  # (seq_len, vocab_size)
    logits = logits.float()

    if len(prompt_token_ids) < 2:
        return []

    # Autoregressive shift: logits[i] predicts token[i+1]
    shift_logits = logits[:-1, :].contiguous()
    shift_labels = torch.tensor(
        prompt_token_ids[1:], dtype=torch.long, device=logits.device
    ).contiguous()

    # CrossEntropyLoss with no reduction → per-token loss in nats
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss_per_token = loss_fct(shift_logits, shift_labels)

    # Convert nats to bits: bits = nats / ln(2)
    surprisals_bits = loss_per_token / math.log(2)

    return [round(s, 4) for s in surprisals_bits.tolist()]


def detect_repetition_loop(
    hidden_state_buffer: List[torch.Tensor],
    threshold: float = 0.9995,
) -> bool:
    """Detect if the model is stuck in a repetition loop using cosine similarity.

    Compares consecutive hidden state vectors in the buffer. If 3+ consecutive
    pairs have cosine_sim > threshold, a repetition loop is detected.

    The buffer should contain the last-layer hidden state vector from the most
    recent generation steps (transient — never serialized).

    Note on threshold: The default of 0.9995 accounts for the anisotropy problem
    in transformer last-layer representations. In float16 models (e.g., GPT-2 on CUDA),
    even non-repetitive tokens produce cosine similarities of 0.992-0.999. True
    repetition (same token repeated) gives ~1.0, so 0.9995 provides clear separation.

    Args:
        hidden_state_buffer: List of 1D hidden state vectors (last layer, last token).
            Typically the last 5 steps. Each tensor shape: (hidden_dim,)
        threshold: Cosine similarity threshold for "same direction" (default 0.9995)

    Returns:
        True if repetition loop detected, False otherwise.
    """
    if len(hidden_state_buffer) < 4:
        return False

    consecutive_count = 0
    for i in range(1, len(hidden_state_buffer)):
        prev_h = hidden_state_buffer[i - 1].float()
        curr_h = hidden_state_buffer[i].float()

        # Ensure 1D
        if prev_h.dim() > 1:
            prev_h = prev_h.flatten()
        if curr_h.dim() > 1:
            curr_h = curr_h.flatten()

        cosine_sim = F.cosine_similarity(prev_h.unsqueeze(0), curr_h.unsqueeze(0)).item()
        if cosine_sim > threshold:
            consecutive_count += 1
            if consecutive_count >= 3:
                return True
        else:
            consecutive_count = 0  # Reset on a non-matching pair

    return False


def detect_mid_layer_anomaly(
    timeline_layers: List[List[Any]],
    num_layers: int,
) -> bool:
    """Detect runtime anomalies in middle layers (hallucination sweet spot).

    Middle layers (middle third) are where "truth processing" occurs.
    Anomalies here are more dangerous than in early (syntactic) or late
    (token selection) layers.

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

    Returns:
        True if mid-layer anomaly detected, False otherwise.
    """
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
            baseline = sum(step_early_norms) / len(step_early_norms)
            explosion_threshold = baseline * L2_EXPLOSION_MULTIPLIER
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


def _random_projection_sketch(
    tensor: torch.Tensor,
    sketch_dim: int,
    seed: int,
) -> List[float]:
    """
    Compute random projection sketch of tensor.

    Args:
        tensor: Input tensor, shape (seq_len, hidden_dim)
        sketch_dim: Target sketch dimension
        seed: Random seed for reproducibility

    Returns:
        List of sketch values
    """
    # Use numpy for deterministic random projection
    np.random.seed(seed)

    # Flatten or average across sequence
    if tensor.dim() == 2:
        # Average across sequence dimension
        vector = tensor.mean(dim=0).numpy()
    else:
        vector = tensor.numpy().flatten()

    hidden_dim = len(vector)

    # Generate random projection matrix
    projection_matrix = np.random.randn(hidden_dim, sketch_dim)
    projection_matrix /= np.sqrt(hidden_dim)  # Normalize

    # Project
    sketch = vector @ projection_matrix

    # Return as list, rounded
    return [round(float(x), 2) for x in sketch.tolist()]


# ============================================================================
# Test Harness
# ============================================================================


def _test_summaries():
    """Test harness for summary computation."""
    print("Testing Summary Computation...")

    from transformers import AutoTokenizer

    from CoreVital.config import Config

    config = Config()

    # Test hidden summary (sketch disabled by default for small payload)
    hidden = torch.randn(1, 10, 768)
    hidden_summary = compute_hidden_summary(hidden, config.summaries.hidden)
    print(f"✓ Hidden summary: {list(hidden_summary.keys())}")
    assert "mean" in hidden_summary
    # Sketch only present when config.sketch.enabled is True
    if getattr(config.summaries.hidden.sketch, "enabled", False):
        assert "sketch" in hidden_summary

    # Test attention summary
    attention = torch.softmax(torch.randn(1, 12, 10, 10), dim=-1)
    attn_summary = compute_attention_summary(attention, config.summaries.attention)
    print(f"✓ Attention summary: {list(attn_summary.keys())}")
    assert "entropy_mean" in attn_summary

    # Test logits summary
    logits = torch.randn(1, 10, 50257)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    logits_summary = compute_logits_summary(logits, tokenizer, config.summaries.logits)
    print(f"✓ Logits summary: {list(logits_summary.keys())}")
    assert "entropy" in logits_summary
    assert "topk" in logits_summary

    print("✓ All summary tests passed!\n")


if __name__ == "__main__":
    _test_summaries()
