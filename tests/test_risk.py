# ============================================================================
# CoreVital - Risk Score and Layer Blame Tests (Phase-2)
# ============================================================================


from CoreVital.reporting.schema import (
    AttentionSummary,
    HealthFlags,
    HiddenSummary,
    LayerSummary,
    LogitsSummary,
    Summary,
    TensorAnomalies,
    TimelineStep,
    TokenInfo,
)
from CoreVital.risk import compute_layer_blame, compute_risk_score


def _step(step_index: int, entropy: float = None, top_k_margin: float = None, voter_agreement: float = None, surprisal: float = None) -> TimelineStep:
    """Build a minimal TimelineStep with optional logits metrics."""
    logits = LogitsSummary()
    if entropy is not None:
        logits.entropy = entropy
    if top_k_margin is not None:
        logits.top_k_margin = top_k_margin
    if voter_agreement is not None:
        logits.voter_agreement = voter_agreement
    if surprisal is not None:
        logits.surprisal = surprisal
    return TimelineStep(
        step_index=step_index,
        token=TokenInfo(token_id=0, token_text="x", is_prompt_token=False),
        logits_summary=logits,
    )


class TestComputeRiskScore:
    """Unit tests for compute_risk_score."""

    def test_nan_or_inf_returns_one(self):
        summary = Summary(prompt_tokens=1, generated_tokens=5, total_steps=6, elapsed_ms=100)
        flags = HealthFlags(nan_detected=True, inf_detected=False)
        score, factors = compute_risk_score(flags, summary)
        assert score == 1.0
        assert factors == ["nan_or_inf"]

        flags2 = HealthFlags(nan_detected=False, inf_detected=True)
        score2, factors2 = compute_risk_score(flags2, summary)
        assert score2 == 1.0
        assert factors2 == ["nan_or_inf"]

    def test_repetition_loop_contribution(self):
        summary = Summary(prompt_tokens=1, generated_tokens=5, total_steps=6, elapsed_ms=100)
        flags = HealthFlags(repetition_loop_detected=True)
        score, factors = compute_risk_score(flags, summary)
        assert score >= 0.9
        assert "repetition_loop" in factors

    def test_mid_layer_anomaly_contribution(self):
        summary = Summary(prompt_tokens=1, generated_tokens=5, total_steps=6, elapsed_ms=100)
        flags = HealthFlags(mid_layer_anomaly_detected=True)
        score, factors = compute_risk_score(flags, summary)
        assert score >= 0.7
        assert "mid_layer_anomaly" in factors

    def test_attention_collapse_contribution(self):
        summary = Summary(prompt_tokens=1, generated_tokens=5, total_steps=6, elapsed_ms=100)
        flags = HealthFlags(attention_collapse_detected=True)
        score, factors = compute_risk_score(flags, summary)
        assert score >= 0.3
        assert "attention_collapse" in factors

    def test_high_entropy_steps_contribution(self):
        summary = Summary(prompt_tokens=1, generated_tokens=10, total_steps=11, elapsed_ms=100)
        flags = HealthFlags(high_entropy_steps=5)
        score, factors = compute_risk_score(flags, summary)
        assert score <= 1.0
        assert "high_entropy_steps" in factors

    def test_clean_run_low_score(self):
        summary = Summary(prompt_tokens=1, generated_tokens=5, total_steps=6, elapsed_ms=100)
        flags = HealthFlags()
        score, factors = compute_risk_score(flags, summary)
        assert score == 0.0
        assert factors == []

    def test_score_capped_at_one(self):
        summary = Summary(prompt_tokens=1, generated_tokens=2, total_steps=3, elapsed_ms=100)
        flags = HealthFlags(
            repetition_loop_detected=True,
            mid_layer_anomaly_detected=True,
            attention_collapse_detected=True,
            high_entropy_steps=2,
        )
        score, _ = compute_risk_score(flags, summary)
        assert score <= 1.0

    def test_high_entropy_and_low_margin_scores_higher_than_either_alone(self):
        """High entropy + low margin together should score higher than either alone."""
        summary = Summary(prompt_tokens=0, generated_tokens=10, total_steps=10, elapsed_ms=100)
        flags = HealthFlags()

        timeline_high_entropy_only = [_step(i, entropy=5.0, top_k_margin=0.8) for i in range(10)]
        timeline_low_margin_only = [_step(i, entropy=1.0, top_k_margin=0.05) for i in range(10)]
        timeline_both = [_step(i, entropy=5.0, top_k_margin=0.05) for i in range(10)]

        score_entropy, _ = compute_risk_score(flags, summary, timeline=timeline_high_entropy_only)
        score_margin, _ = compute_risk_score(flags, summary, timeline=timeline_low_margin_only)
        score_both, _ = compute_risk_score(flags, summary, timeline=timeline_both)

        assert score_both > score_entropy
        assert score_both > score_margin

    def test_nan_inf_always_returns_one(self):
        """NaN/Inf must always return 1.0 regardless of timeline."""
        summary = Summary(prompt_tokens=1, generated_tokens=5, total_steps=6, elapsed_ms=100)
        timeline = [_step(i, entropy=1.0) for i in range(6)]

        score_nan, factors = compute_risk_score(
            HealthFlags(nan_detected=True, inf_detected=False), summary, timeline=timeline
        )
        assert score_nan == 1.0
        assert "nan_or_inf" in factors

        score_inf, factors2 = compute_risk_score(
            HealthFlags(nan_detected=False, inf_detected=True), summary, timeline=timeline
        )
        assert score_inf == 1.0
        assert "nan_or_inf" in factors2

    def test_factors_contain_expected_strings(self):
        """Factors list should contain the right signal names."""
        summary = Summary(prompt_tokens=0, generated_tokens=8, total_steps=8, elapsed_ms=100)
        flags = HealthFlags()
        timeline = [
            _step(i, entropy=6.0, top_k_margin=0.08, voter_agreement=0.3)
            for i in range(8)
        ]
        score, factors = compute_risk_score(flags, summary, timeline=timeline)
        assert "elevated_entropy" in factors
        assert "low_confidence_margin" in factors
        assert "low_topk_mass" in factors

    def test_empty_timeline_returns_zero(self):
        """Empty timeline should return 0.0 and no factors (no NaN/Inf)."""
        summary = Summary(prompt_tokens=0, generated_tokens=0, total_steps=0, elapsed_ms=0)
        flags = HealthFlags()
        score, factors = compute_risk_score(flags, summary, timeline=[])
        assert score == 0.0
        assert factors == []


class TestComputeLayerBlame:
    """Unit tests for compute_layer_blame."""

    def _layer(
        self,
        layer_index: int,
        has_nan: bool = False,
        has_inf: bool = False,
        collapsed_heads: int = 0,
    ) -> LayerSummary:
        anomalies = None
        if has_nan or has_inf:
            anomalies = TensorAnomalies(has_nan=has_nan, has_inf=has_inf)
        attn = AttentionSummary(collapsed_head_count=collapsed_heads)
        return LayerSummary(
            layer_index=layer_index,
            hidden_summary=HiddenSummary(),
            attention_summary=attn,
            anomalies=anomalies,
        )

    def test_empty_layers_no_blame(self):
        assert compute_layer_blame([]) == []
        assert compute_layer_blame([[]]) == []

    def test_blame_nan_layer(self):
        layers_step0 = [
            self._layer(0),
            self._layer(1, has_nan=True),
            self._layer(2),
        ]
        assert compute_layer_blame([layers_step0]) == [1]

    def test_blame_collapsed_heads(self):
        layers_step0 = [
            self._layer(0, collapsed_heads=1),
            self._layer(1),
            self._layer(2, collapsed_heads=2),
        ]
        assert compute_layer_blame([layers_step0]) == [0, 2]

    def test_blame_merged_across_steps(self):
        layers_s0 = [self._layer(0), self._layer(1, has_inf=True)]
        layers_s1 = [self._layer(0, collapsed_heads=1), self._layer(1)]
        assert compute_layer_blame([layers_s0, layers_s1]) == [0, 1]

    def test_blame_sorted_unique(self):
        # Layer 0: collapse in s1; Layer 1: collapse in s1; Layer 2: NaN in s0
        layers_s0 = [self._layer(0), self._layer(1), self._layer(2, has_nan=True)]
        layers_s1 = [self._layer(0, collapsed_heads=1), self._layer(1, collapsed_heads=1), self._layer(2)]
        assert compute_layer_blame([layers_s0, layers_s1]) == [0, 1, 2]
