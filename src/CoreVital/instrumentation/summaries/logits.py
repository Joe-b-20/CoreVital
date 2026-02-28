# Logits summary computation: compute_logits_summary, compute_prompt_surprisal

import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from CoreVital.errors import SummaryComputationError
from CoreVital.logging_utils import get_logger

if TYPE_CHECKING:
    from CoreVital.config import LogitsSummariesConfig

logger = get_logger(__name__)

# Constants
MIN_TOPK_FOR_STATS = 50  # Minimum top-k for margin/topk_mass/topk_probs; ensures enough tokens for stats
VOTER_AGREEMENT_TOP_K = 10  # Number of top tokens for topk_mass (probability mass sum)


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

        # Top-k values (shared by margin, topk_mass, topk_probs)
        topk_k = max(config.topk, MIN_TOPK_FOR_STATS)
        topk_values, topk_indices = torch.topk(logits, k=min(topk_k, len(logits)))
        topk_probs = probs_full[topk_indices]

        # ── Shannon Entropy (numerically stable via log_softmax) ───────
        # Always compute internally — perplexity depends on it even if "entropy" not in stats
        p_log_p = probs_full * log_probs_full
        entropy_nats = -torch.nan_to_num(p_log_p, nan=0.0).sum()
        entropy_bits = float(entropy_nats.item()) / math.log(2)
        if "entropy" in config.stats:
            summary["entropy"] = entropy_bits

        # ── Top-1 to Top-2 margin (deprecated: use top_k_margin; kept for backward compat) ──
        if "top1_top2_margin" in config.stats:
            if len(topk_probs) >= 2:
                margin = topk_probs[0] - topk_probs[1]
                summary["top1_top2_margin"] = float(margin.item())
            else:
                summary["top1_top2_margin"] = 0.0

        # ── Top-K Margin ──
        if "top_k_margin" in config.stats:
            if len(topk_probs) >= 2:
                summary["top_k_margin"] = float((topk_probs[0] - topk_probs[1]).item())
            else:
                summary["top_k_margin"] = 0.0

        # ── Top-K mass (sum of top-K token probabilities; was voter_agreement) ──
        if "topk_mass" in config.stats or "voter_agreement" in config.stats:
            top_n = min(VOTER_AGREEMENT_TOP_K, len(topk_probs))
            mass = float(topk_probs[:top_n].sum().item())
            summary["topk_mass"] = mass
            # Backward compat — deprecated; remove in v0.5.0
            summary["voter_agreement"] = mass

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
            summary["topk_probs"] = topk_list
            # Backward compat — deprecated; remove in v0.5.0
            summary["topk"] = topk_list

        return summary

    except Exception as e:
        logger.exception("Failed to compute logits summary")
        raise SummaryComputationError("Logits summary computation failed", details=str(e)) from e


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
    shift_labels = torch.tensor(prompt_token_ids[1:], dtype=torch.long, device=logits.device).contiguous()

    # CrossEntropyLoss with no reduction → per-token loss in nats
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss_per_token = loss_fct(shift_logits, shift_labels)

    # Convert nats to bits: bits = nats / ln(2)
    surprisals_bits = loss_per_token / math.log(2)

    return [round(s, 4) for s in surprisals_bits.tolist()]
