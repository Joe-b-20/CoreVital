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
from CoreVital.risk import compute_layer_blame, compute_layer_blame_flat, compute_risk_score


def _step(
    step_index: int,
    entropy: float = None,
    top_k_margin: float = None,
    voter_agreement: float = None,
    surprisal: float = None,
) -> TimelineStep:
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
        flags = HealthFlags(attention_collapse_detected=True, attention_collapse_severity=0.3)
        score, factors = compute_risk_score(flags, summary)
        assert score >= 0.3
        assert "attention_collapse" in factors

    def test_attention_collapse_variable_severity(self):
        """Variable severity from detect_attention_collapse flows through to risk score."""
        summary = Summary(prompt_tokens=1, generated_tokens=5, total_steps=6, elapsed_ms=100)
        for sev in [0.2, 0.4, 0.8]:
            flags = HealthFlags(attention_collapse_detected=True, attention_collapse_severity=sev)
            score, factors = compute_risk_score(flags, summary)
            assert score >= sev, f"severity={sev} but score={score}"
            assert "attention_collapse" in factors

    def test_attention_collapse_none_severity_falls_back_new_path(self):
        """New path: when severity is None with timeline, falls back to 0.15."""
        summary = Summary(prompt_tokens=1, generated_tokens=5, total_steps=6, elapsed_ms=100)
        flags = HealthFlags(attention_collapse_detected=True, attention_collapse_severity=None)
        timeline = [_step(i, entropy=0.0) for i in range(5)]
        score, factors = compute_risk_score(flags, summary, timeline=timeline)
        assert abs(score - 0.15) < 0.01
        assert "attention_collapse" in factors

    def test_attention_collapse_none_severity_falls_back_legacy(self):
        """Legacy path: when severity is None without timeline, falls back to 0.3."""
        summary = Summary(prompt_tokens=1, generated_tokens=5, total_steps=6, elapsed_ms=100)
        flags = HealthFlags(attention_collapse_detected=True, attention_collapse_severity=None)
        score, factors = compute_risk_score(flags, summary)
        assert abs(score - 0.3) < 0.01
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
        timeline = [_step(i, entropy=6.0, top_k_margin=0.08, voter_agreement=0.3) for i in range(8)]
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
    """Unit tests for compute_layer_blame (enriched blame)."""

    def _layer(
        self,
        layer_index: int,
        has_nan: bool = False,
        has_inf: bool = False,
        collapsed_heads: int = 0,
        l2_norm_mean: float = None,
        entropy_mean: float = None,
    ) -> LayerSummary:
        anomalies = None
        if has_nan or has_inf:
            anomalies = TensorAnomalies(has_nan=has_nan, has_inf=has_inf)
        attn = AttentionSummary(collapsed_head_count=collapsed_heads)
        if entropy_mean is not None:
            attn.entropy_mean = entropy_mean
        hidden = HiddenSummary()
        if l2_norm_mean is not None:
            hidden.l2_norm_mean = l2_norm_mean
        return LayerSummary(
            layer_index=layer_index,
            hidden_summary=hidden,
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
        result = compute_layer_blame([layers_step0])
        assert len(result) == 1
        assert result[0]["layer"] == 1
        assert "NaN/Inf detected" in result[0]["reasons"]
        assert result[0]["severity"] == 1.0

    def test_blame_collapsed_heads_requires_majority(self):
        """Collapse must occur in >50% of steps to trigger blame."""
        # 1 step with collapse out of 1 step -> 100% -> blamed
        layers_step0 = [
            self._layer(0, collapsed_heads=1),
            self._layer(1),
            self._layer(2, collapsed_heads=2),
        ]
        result = compute_layer_blame([layers_step0])
        blamed_indices = [b["layer"] for b in result]
        assert 0 in blamed_indices
        assert 2 in blamed_indices

    def test_collapse_below_threshold_not_blamed(self):
        """Collapse in <50% of steps should not trigger blame."""
        # Layer 0: collapse in 1 of 3 steps -> 33% -> NOT blamed
        layers_s0 = [self._layer(0, collapsed_heads=1), self._layer(1)]
        layers_s1 = [self._layer(0), self._layer(1)]
        layers_s2 = [self._layer(0), self._layer(1)]
        result = compute_layer_blame([layers_s0, layers_s1, layers_s2])
        blamed_indices = [b["layer"] for b in result]
        assert 0 not in blamed_indices

    def test_blame_merged_across_steps(self):
        layers_s0 = [self._layer(0), self._layer(1, has_inf=True)]
        layers_s1 = [self._layer(0, collapsed_heads=1), self._layer(1)]
        result = compute_layer_blame([layers_s0, layers_s1])
        blamed_indices = [b["layer"] for b in result]
        # Layer 1: NaN/Inf; Layer 0: collapse in 1/2=50%, not >50%, so NOT blamed
        assert 1 in blamed_indices

    def test_blame_sorted_by_layer_index(self):
        layers_s0 = [self._layer(0), self._layer(1), self._layer(2, has_nan=True)]
        layers_s1 = [self._layer(0, has_nan=True), self._layer(1), self._layer(2)]
        result = compute_layer_blame([layers_s0, layers_s1])
        indices = [b["layer"] for b in result]
        assert indices == sorted(indices)

    def test_l2_norm_outlier_blamed(self):
        """Layer with L2 norm z-score > 2.5 should be blamed."""
        # 9 layers needed: z-score requires enough samples to avoid small-sample cap.
        # 8 normal at 10.0, 1 outlier at 100.0 -> z ≈ 2.67
        layers = [self._layer(i, l2_norm_mean=10.0) for i in range(8)]
        layers.append(self._layer(8, l2_norm_mean=100.0))
        result = compute_layer_blame([layers])
        blamed_indices = [b["layer"] for b in result]
        assert 8 in blamed_indices
        layer_8 = next(b for b in result if b["layer"] == 8)
        assert any("L2 norm outlier" in r for r in layer_8["reasons"])
        assert layer_8["severity"] >= 0.5

    def test_l2_norm_outlier_not_blamed_when_uniform(self):
        """When all L2 norms are similar, no layer should be blamed for L2."""
        layers = [
            self._layer(0, l2_norm_mean=10.0),
            self._layer(1, l2_norm_mean=10.1),
            self._layer(2, l2_norm_mean=10.2),
        ]
        result = compute_layer_blame([layers])
        assert all("L2 norm outlier" not in r for b in result for r in b["reasons"])

    def test_l2_instability_blamed(self):
        """Layer with high L2 norm CV across steps should be blamed."""
        # Layer 0: wildly varying L2 norms across 4 steps
        layers_s0 = [self._layer(0, l2_norm_mean=5.0), self._layer(1, l2_norm_mean=10.0)]
        layers_s1 = [self._layer(0, l2_norm_mean=50.0), self._layer(1, l2_norm_mean=10.5)]
        layers_s2 = [self._layer(0, l2_norm_mean=5.0), self._layer(1, l2_norm_mean=11.0)]
        layers_s3 = [self._layer(0, l2_norm_mean=100.0), self._layer(1, l2_norm_mean=10.0)]
        result = compute_layer_blame([layers_s0, layers_s1, layers_s2, layers_s3])
        layer_0 = next((b for b in result if b["layer"] == 0), None)
        assert layer_0 is not None
        assert any("Unstable L2 norms" in r for r in layer_0["reasons"])
        assert layer_0["severity"] >= 0.3

    def test_severity_is_max_of_reasons(self):
        """Severity should be the max across all reason severities."""
        # Layer with NaN (1.0) and collapse (0.4) -> severity 1.0
        layers = [self._layer(0, has_nan=True, collapsed_heads=3)]
        result = compute_layer_blame([layers])
        assert len(result) == 1
        assert result[0]["severity"] == 1.0

    def test_return_structure(self):
        """Each blamed entry has layer, reasons, severity."""
        layers = [self._layer(0, has_nan=True)]
        result = compute_layer_blame([layers])
        assert len(result) == 1
        entry = result[0]
        assert "layer" in entry
        assert "reasons" in entry
        assert "severity" in entry
        assert isinstance(entry["layer"], int)
        assert isinstance(entry["reasons"], list)
        assert isinstance(entry["severity"], float)


class TestComputeLayerBlameFlat:
    """Tests for backward-compatible flat layer blame."""

    def _layer(self, layer_index: int, has_nan: bool = False) -> LayerSummary:
        anomalies = TensorAnomalies(has_nan=has_nan) if has_nan else None
        return LayerSummary(
            layer_index=layer_index,
            hidden_summary=HiddenSummary(),
            attention_summary=AttentionSummary(),
            anomalies=anomalies,
        )

    def test_flat_returns_list_of_ints(self):
        layers = [self._layer(0), self._layer(1, has_nan=True), self._layer(2)]
        result = compute_layer_blame_flat([layers])
        assert result == [1]
        assert all(isinstance(x, int) for x in result)

    def test_flat_empty(self):
        assert compute_layer_blame_flat([]) == []

    def test_flat_matches_enriched(self):
        layers = [self._layer(0, has_nan=True), self._layer(1), self._layer(2, has_nan=True)]
        enriched = compute_layer_blame([layers])
        flat = compute_layer_blame_flat([layers])
        assert flat == [b["layer"] for b in enriched]
