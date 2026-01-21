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
# ============================================================================

from typing import Dict, List, Optional, Any, TYPE_CHECKING
import torch
import torch.nn.functional as F
import numpy as np

from CoreVital.errors import SummaryComputationError
from CoreVital.logging_utils import get_logger

if TYPE_CHECKING:
    from CoreVital.config import HiddenSummariesConfig, AttentionSummariesConfig, LogitsSummariesConfig


logger = get_logger(__name__)

# Constants
MIN_TOPK_FOR_ENTROPY = 50  # Minimum top-k for good entropy estimate


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
        
        summary = {}
        
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
        
        # Sketch via random projection
        if config.sketch.method == "randproj":
            sketch = _random_projection_sketch(
                hidden_state,
                config.sketch.dim,
                config.sketch.seed,
            )
            summary["sketch"] = sketch
        
        return summary
        
    except Exception as e:
        logger.exception("Failed to compute hidden summary")
        raise SummaryComputationError(
            "Hidden state summary computation failed",
            details=str(e)
        ) from e


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
        
        summary = {}
        # Compute entropy per head
        # Entropy of attention distribution over keys for each query
        if "entropy_mean" in config.stats or "entropy_min" in config.stats:
            # Add small epsilon to avoid log(0)
            eps = 1e-10
            attention_safe = attention + eps
            
            # Entropy: -sum(p * log(p)) per query position, then average over queries
            # attention shape: (heads, target_len, source_len) for cross-attention
            #                  (heads, seq_len, seq_len) for self-attention
            # We compute entropy over the last dimension (keys/source) for each query/target position
            entropy = -(attention_safe * torch.log(attention_safe)).sum(dim=-1)
            
            if "entropy_mean" in config.stats:
                summary["entropy_mean"] = float(entropy.mean().item())
            
            if "entropy_min" in config.stats:
                summary["entropy_min"] = float(entropy.min().item())
        
        # Concentration: max attention weight per head
        if "concentration_max" in config.stats:
            # Max over keys/source for each query/target, then max over queries/targets, then max over heads
            # Works for both self-attention and cross-attention
            max_attn = attention.max(dim=-1)[0]  # Max over keys/source: (heads, target_len) or (heads, seq_len)
            summary["concentration_max"] = float(max_attn.max().item())
        
        return summary
        
    except Exception as e:
        logger.exception("Failed to compute attention summary")
        raise SummaryComputationError(
            "Attention summary computation failed",
            details=str(e)
        ) from e


def compute_logits_summary(
    logits: torch.Tensor,
    tokenizer: Any,
    config: "LogitsSummariesConfig",
) -> Dict[str, Any]:
    """
    Compute summary statistics for logits tensor.
    
    Args:
        logits: Logits tensor, shape (batch, seq_len, vocab_size) or (vocab_size,)
        tokenizer: Tokenizer for decoding token IDs
        config: Logits summaries configuration
        
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
        
        summary = {}
        
        # Compute probabilities
        # For efficiency, only compute over top-k
        topk_k = max(config.topk, MIN_TOPK_FOR_ENTROPY)  # At least MIN_TOPK_FOR_ENTROPY for good entropy estimate
        topk_values, topk_indices = torch.topk(logits, k=min(topk_k, len(logits)))
        
        # Compute softmax over top-k (approximation for large vocabs)
        topk_probs = F.softmax(topk_values, dim=-1)
        
        # Entropy
        if "entropy" in config.stats:
            eps = 1e-10
            entropy = -(topk_probs * torch.log(topk_probs + eps)).sum()
            summary["entropy"] = float(entropy.item())
        
        # Top-1 to Top-2 margin
        if "top1_top2_margin" in config.stats:
            if len(topk_probs) >= 2:
                margin = topk_probs[0] - topk_probs[1]
                summary["top1_top2_margin"] = float(margin.item())
            else:
                summary["top1_top2_margin"] = 0.0
        
        # Top-k token probabilities
        if "topk_probs" in config.stats:
            topk_list = []
            for idx, (token_id, prob) in enumerate(zip(topk_indices[:config.topk], topk_probs[:config.topk])):
                token_id = int(token_id.item())
                token_text = tokenizer.decode([token_id])
                topk_list.append({
                    "token_id": token_id,
                    "token_text": token_text,
                    "prob": round(float(prob.item()), 3),
                })
            summary["topk"] = topk_list
        
        return summary
        
    except Exception as e:
        logger.exception("Failed to compute logits summary")
        raise SummaryComputationError(
            "Logits summary computation failed",
            details=str(e)
        ) from e


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
        raise SummaryComputationError(
            "Encoder hidden states summary computation failed",
            details=str(e)
        ) from e


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
    
    from CoreVital.config import Config
    from transformers import AutoTokenizer
    
    config = Config()
    
    # Test hidden summary
    hidden = torch.randn(1, 10, 768)
    hidden_summary = compute_hidden_summary(hidden, config.summaries.hidden)
    print(f"✓ Hidden summary: {list(hidden_summary.keys())}")
    assert "mean" in hidden_summary
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