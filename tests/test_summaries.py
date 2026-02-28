# ============================================================================
# CoreVital - Summary Computation Unit Tests
#
# Purpose: Unit test each function in summaries.py with known inputs (#15)
# Dependencies: pytest, torch, CoreVital
# Usage: pytest tests/test_summaries.py -v
# ============================================================================

from unittest.mock import MagicMock

import torch

from CoreVital.config import (
    AttentionSummariesConfig,
    HiddenSummariesConfig,
    LogitsSummariesConfig,
    SketchConfig,
)
from CoreVital.instrumentation.summaries import (
    NORMALIZED_COLLAPSED_THRESHOLD,
    compute_attention_summary,
    compute_basin_scores,
    compute_hidden_summary,
    compute_logits_summary,
    compute_prompt_surprisal,
    detect_mid_layer_anomaly,
    detect_repetition_loop,
    extract_sparse_attention,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_tokenizer():
    t = MagicMock()
    t.decode = lambda ids: f"tok_{ids[0]}" if ids else ""
    return t


# ---------------------------------------------------------------------------
# compute_logits_summary
# ---------------------------------------------------------------------------


class TestComputeLogitsSummary:
    """Unit tests for compute_logits_summary."""

    def test_entropy_perplexity_surprisal(self):
        """Crafted logits -> verify entropy, perplexity, surprisal."""
        config = LogitsSummariesConfig(
            enabled=True,
            stats=["entropy", "perplexity", "surprisal"],
            topk=5,
        )
        # Uniform over 4 tokens -> entropy = 2 bits, perplexity = 4
        # Shape (1, 1, vocab): batch=1, seq_len=1, vocab_size=4
        logits = torch.tensor([[[0.0, 0.0, 0.0, 0.0]]])
        tokenizer = _mock_tokenizer()
        out = compute_logits_summary(logits, tokenizer, config, actual_token_id=0)
        assert "entropy" in out
        assert abs(out["entropy"] - 2.0) < 0.01  # log2(4)=2
        assert "perplexity" in out
        assert abs(out["perplexity"] - 4.0) < 0.1  # 2^2
        assert "surprisal" in out
        assert abs(out["surprisal"] - 2.0) < 0.01  # -log2(1/4)

    def test_top_k_margin_voter_agreement(self):
        """Verify top_k_margin and voter_agreement from top probs."""
        config = LogitsSummariesConfig(
            enabled=True,
            stats=["top_k_margin", "voter_agreement"],
            topk=5,
        )
        # Peak on index 0, then 1; shape (1, 1, 5)
        logits = torch.tensor([[[10.0, 5.0, 0.0, 0.0, 0.0]]])
        tokenizer = _mock_tokenizer()
        out = compute_logits_summary(logits, tokenizer, config)
        assert "top_k_margin" in out
        assert out["top_k_margin"] >= 0  # p0 - p1
        assert "voter_agreement" in out
        assert 0 <= out["voter_agreement"] <= 1.0

    def test_disabled_returns_empty(self):
        """When config.enabled=False, returns {}."""
        config = LogitsSummariesConfig(enabled=False, stats=["entropy"], topk=5)
        logits = torch.randn(1, 10)
        out = compute_logits_summary(logits, _mock_tokenizer(), config)
        assert out == {}


# ---------------------------------------------------------------------------
# compute_attention_summary
# ---------------------------------------------------------------------------


class TestComputeAttentionSummary:
    """Unit tests for compute_attention_summary."""

    def test_entropy_min_max_concentration(self):
        """Verify entropy_min/max, concentration, collapsed/focused counts."""
        config = AttentionSummariesConfig(
            enabled=True,
            stats=[
                "entropy_mean",
                "entropy_mean_normalized",
                "entropy_min",
                "entropy_max",
                "concentration_max",
                "concentration_min",
                "collapsed_head_count",
                "focused_head_count",
                "max_weight_per_head",
            ],
        )
        # 2 heads, 3x3 attention; one head peaked, one uniform
        attn = torch.zeros(2, 3, 3)
        attn[0, :, :] = 1.0 / 3.0  # uniform -> high entropy
        attn[1, 0, 0] = 1.0  # peaked -> low entropy, high concentration
        attn[1, 1, 1] = 1.0
        attn[1, 2, 2] = 1.0
        attn[1] = attn[1] / attn[1].sum(dim=-1, keepdim=True)
        out = compute_attention_summary(attn, config)
        assert "entropy_mean" in out
        assert "entropy_mean_normalized" in out
        assert "entropy_min" in out
        assert "entropy_max" in out
        assert "concentration_max" in out
        assert "concentration_min" in out
        assert "collapsed_head_count" in out
        assert "focused_head_count" in out
        assert "max_weight_per_head" in out
        assert len(out["max_weight_per_head"]) == 2

    def test_none_returns_empty(self):
        """None attention returns {}."""
        config = AttentionSummariesConfig(enabled=True, stats=["entropy_mean"])
        out = compute_attention_summary(None, config)
        assert out == {}

    def test_disabled_returns_empty(self):
        """When config.enabled=False, returns {}."""
        config = AttentionSummariesConfig(enabled=False, stats=["entropy_mean"])
        attn = torch.ones(2, 4, 4) / 4.0
        out = compute_attention_summary(attn, config)
        assert out == {}


# ---------------------------------------------------------------------------
# Issue 10: Clamp vs additive epsilon
# ---------------------------------------------------------------------------


class TestIssue10ClampVsEpsilon:
    """Issue 10: torch.log(clamp(attn)) instead of torch.log(attn + eps)."""

    def test_clamp_preserves_nonzero_weights(self):
        """Weights above 1e-10 contribute the same entropy as exact log(p)."""
        import math
        config = AttentionSummariesConfig(
            enabled=True, stats=["entropy_mean"]
        )
        # Single head, 4 tokens, known distribution
        attn = torch.tensor([[[0.7, 0.2, 0.1, 0.0]]])  # (1, 1, 4)
        out = compute_attention_summary(attn, config)
        # Hand-calculated: H = -(0.7*ln0.7 + 0.2*ln0.2 + 0.1*ln0.1 + 0*ln(1e-10))
        expected = -(0.7 * math.log(0.7) + 0.2 * math.log(0.2) + 0.1 * math.log(0.1))
        assert abs(out["entropy_mean"] - expected) < 1e-4

    def test_zero_weight_does_not_shift_entropy(self):
        """A zero-weight token contributes 0 to entropy (0 * log(1e-10) = 0)."""
        config = AttentionSummariesConfig(
            enabled=True, stats=["entropy_mean"]
        )
        # All weight on one token
        attn = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
        out = compute_attention_summary(attn, config)
        assert abs(out["entropy_mean"]) < 1e-6  # entropy of a delta is 0

    def test_very_small_weight_not_dominated_by_epsilon(self):
        """A weight of 1e-12 should use log(1e-10) via clamp, not log(1e-12 + 1e-10)."""
        import math
        config = AttentionSummariesConfig(
            enabled=True, stats=["entropy_mean"]
        )
        # Near-delta: one big weight, one tiny weight
        p_big = 1.0 - 1e-12
        p_tiny = 1e-12
        attn = torch.tensor([[[p_big, p_tiny]]])
        out = compute_attention_summary(attn, config)
        # With clamp: p_tiny is clamped to 1e-10, so contribution is -1e-12 * log(1e-10)
        # With additive eps: would be -1e-12 * log(1e-12 + 1e-10) ≈ -1e-12 * log(1.01e-10)
        # Both are tiny; key assertion: entropy is very close to 0
        assert out["entropy_mean"] < 0.001


# ---------------------------------------------------------------------------
# Issue 23: Division normalization vs softmax
# ---------------------------------------------------------------------------


class TestIssue23DivisionNormalization:
    """Issue 23: Unnormalized attention uses division, not re-softmax."""

    def test_drifted_attention_renormalized_by_division(self):
        """Attention summing to 0.99 is renormalized by division, preserving ratios."""
        config = AttentionSummariesConfig(
            enabled=True, stats=["entropy_mean", "concentration_max"]
        )
        # Slightly drifted: sums to 0.99, not 1.0
        attn = torch.tensor([[[0.495, 0.495, 0.0]]])  # sum = 0.99
        out = compute_attention_summary(attn, config)
        # After division: [0.5, 0.5, 0.0] — entropy of fair coin in nats
        import math
        expected_entropy = math.log(2)
        assert abs(out["entropy_mean"] - expected_entropy) < 0.01

    def test_softmax_would_change_distribution(self):
        """Verify that softmax on post-softmax values differs from division."""
        import torch.nn.functional as F
        # Drifted distribution
        raw = torch.tensor([0.8, 0.15, 0.04])  # sum = 0.99
        # Division normalization preserves ratios
        div_result = raw / raw.sum()
        # Softmax treats values as logits — exponential reweighting
        sm_result = F.softmax(raw, dim=-1)
        # They should NOT be equal
        assert not torch.allclose(div_result, sm_result, atol=1e-3)

    def test_already_normalized_unchanged(self):
        """Attention already summing to 1.0 is not modified."""
        config = AttentionSummariesConfig(
            enabled=True, stats=["entropy_mean"]
        )
        attn = torch.tensor([[[0.5, 0.3, 0.2]]])  # sum = 1.0
        out = compute_attention_summary(attn, config)
        import math
        expected = -(0.5 * math.log(0.5) + 0.3 * math.log(0.3) + 0.2 * math.log(0.2))
        assert abs(out["entropy_mean"] - expected) < 1e-4


# ---------------------------------------------------------------------------
# Issue 21: Normalized entropy and collapse detection
# ---------------------------------------------------------------------------


class TestIssue21NormalizedEntropy:
    """Issue 21: Entropy normalized by log(K), collapse detection in [0,1] space."""

    def test_uniform_gives_normalized_entropy_one(self):
        """Uniform attention over K tokens should have normalized entropy = 1.0."""
        config = AttentionSummariesConfig(
            enabled=True, stats=["entropy_mean_normalized"]
        )
        K = 16
        attn = torch.ones(1, 1, K) / K  # uniform
        out = compute_attention_summary(attn, config)
        assert abs(out["entropy_mean_normalized"] - 1.0) < 1e-4

    def test_peaked_gives_normalized_entropy_near_zero(self):
        """Delta attention (all weight on one token) → normalized entropy ≈ 0."""
        config = AttentionSummariesConfig(
            enabled=True, stats=["entropy_mean_normalized"]
        )
        attn = torch.zeros(1, 1, 32)
        attn[0, 0, 0] = 1.0
        out = compute_attention_summary(attn, config)
        assert out["entropy_mean_normalized"] < 0.01

    def test_both_raw_and_normalized_stored(self):
        """Both entropy_mean (raw nats) and entropy_mean_normalized (in [0,1]) are returned."""
        config = AttentionSummariesConfig(
            enabled=True, stats=["entropy_mean", "entropy_mean_normalized"]
        )
        K = 8
        attn = torch.ones(2, 4, K) / K
        out = compute_attention_summary(attn, config)
        assert "entropy_mean" in out
        assert "entropy_mean_normalized" in out
        import math
        assert abs(out["entropy_mean"] - math.log(K)) < 0.01
        assert abs(out["entropy_mean_normalized"] - 1.0) < 0.01

    def test_collapse_detection_same_for_short_and_long_sequences(self):
        """A collapsed head (delta) is detected regardless of sequence length.

        This is the key property of length-normalized collapse detection:
        the same "collapsed" pattern should be caught at K=32 and K=2048.
        """
        config = AttentionSummariesConfig(
            enabled=True, stats=["collapsed_head_count", "entropy_mean_normalized"]
        )

        for K in [32, 128, 512, 2048]:
            # Head 0: delta (collapsed), Head 1: uniform (not collapsed)
            attn = torch.zeros(2, 1, K)
            attn[0, 0, 0] = 1.0  # collapsed
            attn[1, 0, :] = 1.0 / K  # uniform
            out = compute_attention_summary(attn, config)
            assert out["collapsed_head_count"] == 1, (
                f"At K={K}: expected 1 collapsed head, got {out['collapsed_head_count']}"
            )

    def test_slightly_focused_not_collapsed(self):
        """A head that's focused but not a delta should NOT be collapsed."""
        config = AttentionSummariesConfig(
            enabled=True, stats=["collapsed_head_count", "entropy_mean_normalized"]
        )
        K = 64
        attn = torch.zeros(1, 1, K)
        # Spread weight across 5 tokens — normalized entropy ~log(5)/log(64) ≈ 0.27 > 0.03
        for i in range(5):
            attn[0, 0, i] = 1.0 / 5.0
        out = compute_attention_summary(attn, config)
        assert out["collapsed_head_count"] == 0
        assert out["entropy_mean_normalized"] > NORMALIZED_COLLAPSED_THRESHOLD

    def test_normalized_threshold_constant(self):
        """NORMALIZED_COLLAPSED_THRESHOLD is 0.03 in [0,1] space."""
        assert NORMALIZED_COLLAPSED_THRESHOLD == 0.03


# ---------------------------------------------------------------------------
# extract_sparse_attention
# ---------------------------------------------------------------------------


class TestExtractSparseAttention:
    """Unit tests for extract_sparse_attention."""

    def test_soa_format_threshold_filtering(self):
        """Verify SoA format and threshold filtering."""
        # (1, 2, 4, 4): 2 heads, 4x4
        attn = torch.zeros(1, 2, 4, 4)
        attn[0, 0, 0, 1] = 0.5
        attn[0, 0, 0, 2] = 0.02  # above 0.01
        attn[0, 0, 1, 0] = 0.03
        # Normalize so each row sums to 1
        for h in range(2):
            for q in range(4):
                row = attn[0, h, q, :]
                s = row.sum()
                if s > 0:
                    attn[0, h, q, :] = row / s
        heads = extract_sparse_attention(attn, threshold=0.01)
        assert len(heads) == 2
        for h in heads:
            assert "query_indices" in h and "key_indices" in h and "weights" in h
            assert len(h["query_indices"]) == len(h["key_indices"]) == len(h["weights"])
        # First head should have at least the 0.5 and 0.02 and 0.03 entries
        n0 = len(heads[0]["weights"])
        assert n0 >= 1

    def test_max_per_head_caps_connections(self):
        """max_per_head limits number of stored connections per head."""
        attn = torch.ones(1, 1, 5, 5) / 5.0  # all 0.2
        heads = extract_sparse_attention(attn, threshold=0.1, max_per_head=3)
        assert len(heads) == 1
        assert len(heads[0]["weights"]) <= 3


# ---------------------------------------------------------------------------
# compute_basin_scores
# ---------------------------------------------------------------------------


class TestComputeBasinScores:
    """Unit tests for compute_basin_scores."""

    def test_middle_boundary_ratio(self):
        """Verify middle/boundary ratio: more weight on middle -> higher score."""
        # (1, 1, 9, 9): 1 head, seq_len 9 -> middle third = indices 3,4,5
        attn = torch.zeros(1, 1, 9, 9)
        # Put weight on middle keys
        attn[0, 0, :, 3] = 0.5
        attn[0, 0, :, 4] = 0.5
        attn[0, 0, :, 5] = 0.5
        attn[0, 0, :, 0] = 0.01
        attn[0, 0, :, 8] = 0.01
        for q in range(9):
            row = attn[0, 0, q, :].clone()
            s = row.sum()
            if s > 0:
                attn[0, 0, q, :] = row / s
        scores = compute_basin_scores(attn)
        assert len(scores) == 1
        assert scores[0] > 0.5  # middle attention higher than boundary

    def test_short_sequence_returns_ones(self):
        """Seq_len < 3 returns [1.0] * num_heads."""
        attn = torch.ones(1, 2, 2, 2) / 2.0
        scores = compute_basin_scores(attn)
        assert scores == [1.0, 1.0]


# ---------------------------------------------------------------------------
# compute_prompt_surprisal
# ---------------------------------------------------------------------------


class TestComputePromptSurprisal:
    """Unit tests for compute_prompt_surprisal."""

    def test_hand_calculated_cross_entropy(self):
        """Surprisal = -log2(p); for uniform p=1/V, surprisal = log2(V)."""
        # 4 tokens, uniform logits -> each token p=0.25, surprisal = 2 bits
        logits = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )  # (2, 4) -> 1 surprisal value (for token at index 1)
        prompt_ids = [0, 1]
        out = compute_prompt_surprisal(logits, prompt_ids)
        assert len(out) == 1
        assert abs(out[0] - 2.0) < 0.01  # -log2(1/4)

    def test_short_prompt_returns_empty(self):
        """len(prompt_token_ids) < 2 returns []."""
        logits = torch.randn(1, 10)
        out = compute_prompt_surprisal(logits, [100])
        assert out == []


# ---------------------------------------------------------------------------
# detect_repetition_loop
# ---------------------------------------------------------------------------


class TestDetectRepetitionLoop:
    """Unit tests for detect_repetition_loop."""

    def test_consecutive_logic_three_high_similarity(self):
        """3+ consecutive pairs with cos_sim > threshold -> True."""
        from CoreVital.instrumentation.summaries import detect_repetition_loop

        v = torch.randn(4, 64)
        # Make last 3 identical so 3 consecutive pairs match
        v[1] = v[0].clone()
        v[2] = v[0].clone()
        v[3] = v[0].clone()
        assert detect_repetition_loop(v, threshold=0.9995) is True

    def test_reset_on_non_matching(self):
        """Counter resets when a pair doesn't meet threshold."""
        # Use orthogonal directions so (1,2) has cos_sim 0; only pairs (0,1) and (2,3),(3,4) match
        v = torch.zeros(5, 64)
        v[0, :32] = 1.0
        v[1, :32] = 1.0  # (0,1) sim=1
        v[2, 32:] = 1.0  # (1,2) sim=0 -> reset
        v[3, 32:] = 1.0  # (2,3) sim=1
        v[4, 32:] = 1.0  # (3,4) sim=1 -> 2 consecutive, not 3
        assert detect_repetition_loop(v, threshold=0.9995) is False

    def test_short_buffer_false(self):
        """len < 4 returns False."""
        assert detect_repetition_loop([torch.randn(64)] * 3, threshold=0.9995) is False


# ---------------------------------------------------------------------------
# detect_mid_layer_anomaly
# ---------------------------------------------------------------------------


class TestDetectMidLayerAnomaly:
    """Unit tests for detect_mid_layer_anomaly."""

    def test_nan_detection(self):
        """Layer with has_nan in anomalies -> True."""
        # num_layers=3 -> mid_start=1, mid_end=2; we need step_layers with indices 0,1,2
        layer0 = MagicMock(anomalies=None, hidden_summary=MagicMock(l2_norm_mean=1.0))
        layer_with_nan = MagicMock(
            anomalies=MagicMock(has_nan=True, has_inf=False),
            hidden_summary=MagicMock(l2_norm_mean=1.0),
        )
        layer2 = MagicMock(anomalies=None, hidden_summary=MagicMock(l2_norm_mean=1.0))
        step_layers = [layer0, layer_with_nan, layer2]
        assert detect_mid_layer_anomaly([step_layers], num_layers=3) is True

    def test_empty_timeline_false(self):
        """Empty timeline or num_layers < 3 -> False."""
        assert detect_mid_layer_anomaly([], num_layers=5) is False
        assert detect_mid_layer_anomaly([[MagicMock()]], num_layers=2) is False


# ---------------------------------------------------------------------------
# compute_hidden_summary (clipping guard Branch 5)
# ---------------------------------------------------------------------------


class TestComputeHiddenSummaryClipping:
    """Test feature clipping guard in compute_hidden_summary (#28)."""

    def test_clipped_flag_when_extreme_values(self):
        """Extreme values get clamped and summary has clipped=True."""
        config = HiddenSummariesConfig(
            enabled=True,
            stats=["mean", "l2_norm_mean"],
            sketch=SketchConfig(enabled=False, method="randproj", dim=8, seed=0),
        )
        hidden = torch.tensor([[1e10, -1e10, 0.0]])  # (1, 3)
        out = compute_hidden_summary(hidden, config)
        assert out.get("clipped") is True
        assert "mean" in out
        assert abs(out["mean"]) <= 1e6  # clamped

    def test_not_clipped_when_normal(self):
        """Normal values -> clipped=False."""
        config = HiddenSummariesConfig(
            enabled=True,
            stats=["mean"],
            sketch=SketchConfig(enabled=False, method="randproj", dim=8, seed=0),
        )
        hidden = torch.randn(1, 64) * 0.1
        out = compute_hidden_summary(hidden, config)
        assert out.get("clipped") is False
