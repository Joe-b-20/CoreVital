# CoreVital - Summary Computation Package
#
# Re-exports all summary functions and constants for backward-compatible imports:
#   from CoreVital.instrumentation.summaries import compute_logits_summary, ...
#

from .attention import (
    COLLAPSED_HEAD_ENTROPY_THRESHOLD,
    FOCUSED_HEAD_CONCENTRATION_THRESHOLD,
    NORMALIZED_COLLAPSED_THRESHOLD,
    compute_attention_summary,
    compute_basin_scores,
    extract_sparse_attention,
)
from .hidden_states import (
    L2_EXPLOSION_MULTIPLIER,
    compute_encoder_hidden_states_summaries,
    compute_hidden_summary,
    compute_layer_transformations,
    detect_mid_layer_anomaly,
    detect_repetition_loop,
    detect_tensor_anomalies,
)
from .logits import (
    MIN_TOPK_FOR_STATS,
    VOTER_AGREEMENT_TOP_K,
    compute_logits_summary,
    compute_prompt_surprisal,
)

__all__ = [
    # logits
    "MIN_TOPK_FOR_STATS",
    "VOTER_AGREEMENT_TOP_K",
    "compute_logits_summary",
    "compute_prompt_surprisal",
    # attention
    "COLLAPSED_HEAD_ENTROPY_THRESHOLD",
    "FOCUSED_HEAD_CONCENTRATION_THRESHOLD",
    "NORMALIZED_COLLAPSED_THRESHOLD",
    "compute_attention_summary",
    "compute_basin_scores",
    "extract_sparse_attention",
    # hidden_states
    "L2_EXPLOSION_MULTIPLIER",
    "compute_encoder_hidden_states_summaries",
    "compute_hidden_summary",
    "compute_layer_transformations",
    "detect_mid_layer_anomaly",
    "detect_repetition_loop",
    "detect_tensor_anomalies",
]


# ============================================================================
# Test Harness
# ============================================================================


def _test_summaries():
    """Test harness for summary computation."""
    import torch
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
