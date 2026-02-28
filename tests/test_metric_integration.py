# ============================================================================
# CoreVital - Metric Integration Tests (Phase 5, Issue 13)
#
# Covers:
#   1. TestCompoundMetricStates  — multiple metrics anomalous simultaneously
#   2. TestThresholdBoundary     — exact-boundary behavior for thresholds
#   3. TestEndToEndTraceRiskNarrative — full pipeline trace→risk→narrative
#   4. TestProfileLoading        — model-specific profile thresholds
#   5. TestPerformanceRegression — timing benchmarks for realistic traces
# ============================================================================

import time
from pathlib import Path
from typing import List, Optional

import pytest

from CoreVital.compound_signals import CompoundSignal, detect_compound_signals
from CoreVital.config import ModelProfile, load_model_profile
from CoreVital.early_warning import (
    compute_early_warning,
)
from CoreVital.narrative import build_narrative
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

# ---------------------------------------------------------------------------
# Helpers — mirrors existing pattern from test_risk.py / test_compound_signals.py
# ---------------------------------------------------------------------------


def _step(
    step_index: int,
    entropy: Optional[float] = None,
    top_k_margin: Optional[float] = None,
    voter_agreement: Optional[float] = None,
    surprisal: Optional[float] = None,
    token_text: str = "x",
) -> TimelineStep:
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
        token=TokenInfo(token_id=step_index, token_text=token_text, is_prompt_token=False),
        logits_summary=logits,
    )


def _layer(
    layer_index: int,
    has_nan: bool = False,
    has_inf: bool = False,
    collapsed_head_count: int = 0,
    l2_norm_mean: Optional[float] = None,
    entropy_mean: Optional[float] = None,
) -> LayerSummary:
    anomalies = None
    if has_nan or has_inf:
        anomalies = TensorAnomalies(has_nan=has_nan, has_inf=has_inf)
    attn = AttentionSummary(collapsed_head_count=collapsed_head_count)
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


def _summary(generated: int = 10) -> Summary:
    return Summary(
        prompt_tokens=4,
        generated_tokens=generated,
        total_steps=generated,
        elapsed_ms=100,
    )


# ============================================================================
# 1. Compound metric state tests
# ============================================================================


class TestCompoundMetricStates:
    """Multiple metrics anomalous at the same time produce correct downstream signals."""

    def test_high_entropy_low_margin_scores_higher_than_either_alone(self):
        flags = HealthFlags()
        summary = _summary(10)

        tl_entropy_only = [_step(i, entropy=6.0, top_k_margin=0.8) for i in range(10)]
        tl_margin_only = [_step(i, entropy=1.0, top_k_margin=0.02) for i in range(10)]
        tl_both = [_step(i, entropy=6.0, top_k_margin=0.02) for i in range(10)]

        s_ent, _ = compute_risk_score(flags, summary, timeline=tl_entropy_only)
        s_mar, _ = compute_risk_score(flags, summary, timeline=tl_margin_only)
        s_both, f_both = compute_risk_score(flags, summary, timeline=tl_both)

        assert s_both > s_ent, "Both should exceed entropy-only"
        assert s_both > s_mar, "Both should exceed margin-only"
        assert "elevated_entropy" in f_both
        assert "low_confidence_margin" in f_both

    def test_all_continuous_metrics_anomalous(self):
        """Every continuous metric bad → all four factor names present."""
        flags = HealthFlags()
        summary = _summary(10)
        tl = [_step(i, entropy=7.0, top_k_margin=0.02, voter_agreement=0.2, surprisal=5.0) for i in range(10)]
        score, factors = compute_risk_score(flags, summary, timeline=tl)
        assert score > 0.3
        for f in ("elevated_entropy", "low_confidence_margin", "low_topk_mass", "elevated_surprisal"):
            assert f in factors, f"Expected factor {f}"

    def test_continuous_plus_boolean_flag_stacking(self):
        """Boolean flag (repetition_loop 0.9) + continuous entropy should push score higher."""
        flags_base = HealthFlags(repetition_loop_detected=True)
        summary = _summary(10)
        tl_low = [_step(i, entropy=1.0) for i in range(10)]
        tl_high = [_step(i, entropy=7.0, top_k_margin=0.02) for i in range(10)]

        s_low, _ = compute_risk_score(flags_base, summary, timeline=tl_low)
        s_high, f_high = compute_risk_score(flags_base, summary, timeline=tl_high)

        assert s_high >= s_low
        assert "repetition_loop" in f_high

    def test_compound_signals_stack_with_risk(self):
        """Compound signals (from detect_compound_signals) add to the risk score."""
        flags = HealthFlags()
        summary = _summary(10)
        tl = [_step(i, entropy=5.0, top_k_margin=0.5) for i in range(10)]
        compound = detect_compound_signals(tl)
        assert any(c.name == "confident_confusion" for c in compound)

        s_no_compound, _ = compute_risk_score(flags, summary, timeline=tl)
        s_with, f_with = compute_risk_score(
            flags,
            summary,
            timeline=tl,
            compound_signals=compound,
        )
        assert s_with >= s_no_compound
        assert any(f.startswith("compound:") for f in f_with)

    def test_nan_overrides_all_continuous(self):
        """NaN/Inf flag always returns 1.0 regardless of good continuous metrics."""
        flags = HealthFlags(nan_detected=True)
        summary = _summary(10)
        tl = [_step(i, entropy=0.5, top_k_margin=0.9, surprisal=0.1) for i in range(10)]
        score, factors = compute_risk_score(flags, summary, timeline=tl)
        assert score == 1.0
        assert factors == ["nan_or_inf"]

    def test_entropy_rising_trend_adds_to_score(self):
        """When entropy has a clear rising trend (last_third > first_third * 1.3),
        entropy_rising is added as a factor."""
        flags = HealthFlags()
        summary = _summary(12)
        tl = [_step(i, entropy=2.0 + i * 0.5) for i in range(12)]
        score, factors = compute_risk_score(flags, summary, timeline=tl)
        assert "entropy_rising" in factors
        assert score > 0


# ============================================================================
# 2. Threshold boundary tests
# ============================================================================


class TestThresholdBoundary:
    """Document and test metric values at exact thresholds."""

    def test_entropy_component_threshold(self):
        """Entropy component triggers when mean_ent / 8.0 * 0.3 > 0.05.
        Minimum mean_ent ≈ 1.34 → check just below and at threshold."""
        flags = HealthFlags()
        summary = _summary(10)

        tl_below = [_step(i, entropy=1.0) for i in range(10)]
        tl_above = [_step(i, entropy=2.0) for i in range(10)]

        _, f_below = compute_risk_score(flags, summary, timeline=tl_below)
        _, f_above = compute_risk_score(flags, summary, timeline=tl_above)

        assert "elevated_entropy" not in f_below
        assert "elevated_entropy" in f_above

    def test_margin_component_threshold(self):
        """Margin component triggers when (1 - mean_margin * 5) * 0.2 > 0.05.
        Threshold: mean_margin < 0.15 → component > 0.05."""
        flags = HealthFlags()
        summary = _summary(10)

        tl_ok = [_step(i, entropy=0.5, top_k_margin=0.3) for i in range(10)]
        tl_bad = [_step(i, entropy=0.5, top_k_margin=0.05) for i in range(10)]

        _, f_ok = compute_risk_score(flags, summary, timeline=tl_ok)
        _, f_bad = compute_risk_score(flags, summary, timeline=tl_bad)

        assert "low_confidence_margin" not in f_ok
        assert "low_confidence_margin" in f_bad

    def test_agreement_component_threshold(self):
        """topk_mass/voter_agreement component triggers when
        (1 - mean_agreement) * 0.15 > 0.03 → mean_agreement < 0.80."""
        flags = HealthFlags()
        summary = _summary(10)

        tl_ok = [_step(i, entropy=0.5, voter_agreement=0.95) for i in range(10)]
        tl_bad = [_step(i, entropy=0.5, voter_agreement=0.3) for i in range(10)]

        _, f_ok = compute_risk_score(flags, summary, timeline=tl_ok)
        _, f_bad = compute_risk_score(flags, summary, timeline=tl_bad)

        assert "low_topk_mass" not in f_ok
        assert "low_topk_mass" in f_bad

    def test_surprisal_component_threshold(self):
        """Surprisal component triggers when mean / 10 > 0.02 → mean > 0.2."""
        flags = HealthFlags()
        summary = _summary(10)

        tl_ok = [_step(i, entropy=0.5, surprisal=0.1) for i in range(10)]
        tl_bad = [_step(i, entropy=0.5, surprisal=5.0) for i in range(10)]

        _, f_ok = compute_risk_score(flags, summary, timeline=tl_ok)
        _, f_bad = compute_risk_score(flags, summary, timeline=tl_bad)

        assert "elevated_surprisal" not in f_ok
        assert "elevated_surprisal" in f_bad

    def test_entropy_rising_requires_at_least_6_steps(self):
        """entropy_rising needs ≥6 steps; 5 steps should not trigger."""
        flags = HealthFlags()
        summary = _summary(5)
        tl = [_step(i, entropy=2.0 + i * 2.0) for i in range(5)]
        _, factors = compute_risk_score(flags, summary, timeline=tl)
        assert "entropy_rising" not in factors

    def test_collapse_rate_boundary_for_layer_blame(self):
        """Attention collapse blame requires >50% of steps; exactly 50% should not trigger."""
        layers_s0 = [_layer(0, collapsed_head_count=1)]
        layers_s1 = [_layer(0)]
        result = compute_layer_blame([layers_s0, layers_s1])
        assert not any(b["layer"] == 0 and any("collapse" in r.lower() for r in b["reasons"]) for b in result), (
            "Exactly 50% collapse should NOT trigger blame"
        )

        layers_s2 = [_layer(0, collapsed_head_count=1)]
        result2 = compute_layer_blame([layers_s0, layers_s2, layers_s1])
        blamed_zero = [b for b in result2 if b["layer"] == 0]
        assert any(any("collapse" in r.lower() for r in b["reasons"]) for b in blamed_zero), (
            "67% collapse should trigger blame"
        )

    def test_early_warning_default_entropy_threshold(self):
        """Default high_entropy_threshold is 4.0; verify entropy-margin divergence
        fires above 4.0 but not at 3.9."""
        flags = HealthFlags()

        tl_below = [_step(i, entropy=3.9, top_k_margin=0.5) for i in range(10)]
        _, sigs_below = compute_early_warning(tl_below, flags)
        assert "entropy_margin_divergence" not in sigs_below

        tl_above = [_step(i, entropy=4.5, top_k_margin=0.5) for i in range(10)]
        _, sigs_above = compute_early_warning(tl_above, flags)
        assert "entropy_margin_divergence" in sigs_above

    def test_risk_score_always_capped_at_one(self):
        """Even with every factor maxed out, score must not exceed 1.0."""
        flags = HealthFlags(
            repetition_loop_detected=True,
            mid_layer_anomaly_detected=True,
            attention_collapse_detected=True,
        )
        summary = _summary(12)
        tl = [_step(i, entropy=7.0, top_k_margin=0.01, voter_agreement=0.1, surprisal=9.0) for i in range(12)]
        compound = [
            CompoundSignal(name="test", description="t", severity=0.8, evidence=[]),
        ]
        score, _ = compute_risk_score(
            flags,
            summary,
            timeline=tl,
            compound_signals=compound,
        )
        assert score <= 1.0


# ============================================================================
# 3. End-to-end trace → risk → narrative tests
# ============================================================================


class TestEndToEndTraceRiskNarrative:
    """Full pipeline: build timeline → risk score → narrative text."""

    def _run_pipeline(
        self,
        timeline: List[TimelineStep],
        flags: HealthFlags,
        layers_by_step: Optional[List[List[LayerSummary]]] = None,
        basin_scores: Optional[List[List[float]]] = None,
    ):
        summary = _summary(len(timeline))
        compound = detect_compound_signals(
            timeline,
            layers_by_step=layers_by_step,
            basin_scores=basin_scores,
        )
        score, factors = compute_risk_score(
            flags,
            summary,
            timeline=timeline,
            compound_signals=compound,
        )
        blamed = compute_layer_blame(layers_by_step or [])
        _, warning_signals = compute_early_warning(timeline, flags)
        narrative = build_narrative(
            health_flags=flags,
            risk_score=score,
            risk_factors=factors,
            blamed_layers=blamed,
            warning_signals=warning_signals,
            timeline=timeline,
            compound_signals=compound,
            summary=summary,
        )
        return score, factors, narrative, compound

    def test_healthy_trace(self):
        tl = [_step(i, entropy=1.5, top_k_margin=0.6, surprisal=0.5, voter_agreement=0.85) for i in range(10)]
        score, factors, narrative, _ = self._run_pipeline(tl, HealthFlags())
        assert score < 0.3, f"Healthy trace should be low risk, got {score}"
        assert "Low risk" in narrative
        assert "No significant anomalies" in narrative

    def test_known_bad_repetition_trace(self):
        """Simulated repetition loop: repetition flag on → high risk, narrative mentions it."""
        tl = [_step(i, entropy=0.5, top_k_margin=0.9, surprisal=0.2, voter_agreement=0.99) for i in range(10)]
        flags = HealthFlags(repetition_loop_detected=True)
        score, factors, narrative, _ = self._run_pipeline(tl, flags)
        assert score >= 0.9
        assert "repetition_loop" in factors
        assert "repetition" in narrative.lower()

    def test_known_bad_high_entropy_low_margin(self):
        """High entropy + low margin → elevated risk and narrative mentions uncertainty."""
        tl = [_step(i, entropy=6.0, top_k_margin=0.03, surprisal=4.0, voter_agreement=0.25) for i in range(10)]
        score, factors, narrative, _ = self._run_pipeline(tl, HealthFlags())
        assert score > 0.3
        assert "elevated_entropy" in factors
        assert "entropy" in narrative.lower()

    def test_nan_trace_score_one(self):
        tl = [_step(i, entropy=1.0) for i in range(5)]
        flags = HealthFlags(nan_detected=True)
        score, factors, narrative, _ = self._run_pipeline(tl, flags)
        assert score == 1.0
        assert factors == ["nan_or_inf"]
        assert "High risk" in narrative

    def test_degenerating_trace_fires_compound_signal(self):
        """Entropy rising + margin falling + surprisal rising → degenerating_generation."""
        tl = [
            _step(
                i,
                entropy=2.0 + i * 0.6,
                top_k_margin=0.5 - i * 0.04,
                surprisal=0.5 + i * 0.6,
                voter_agreement=0.7,
            )
            for i in range(12)
        ]
        score, factors, narrative, compound = self._run_pipeline(tl, HealthFlags())
        assert any(c.name == "degenerating_generation" for c in compound)
        assert any("compound:" in f for f in factors)

    def test_context_loss_compound_with_basin_scores(self):
        """High entropy + low basin scores → context_loss fires and feeds narrative."""
        tl = [_step(i, entropy=5.5, top_k_margin=0.1) for i in range(6)]
        basin_scores = [[0.1, 0.05, 0.08]]
        score, factors, narrative, compound = self._run_pipeline(
            tl,
            HealthFlags(),
            basin_scores=basin_scores,
        )
        assert any(c.name == "context_loss" for c in compound)
        assert "context" in narrative.lower()

    def test_blamed_layers_appear_in_narrative(self):
        """Blamed layers with NaN should appear in narrative."""
        tl = [_step(i, entropy=3.0) for i in range(5)]
        layers = [[_layer(0), _layer(1, has_nan=True), _layer(2)] for _ in range(5)]
        score, _, narrative, _ = self._run_pipeline(tl, HealthFlags(), layers_by_step=layers)
        assert "Layer 1" in narrative

    def test_high_risk_narrative_includes_recommendation(self):
        """High risk from repetition → narrative includes actionable recommendation."""
        tl = [_step(i, entropy=1.0, top_k_margin=0.9) for i in range(10)]
        flags = HealthFlags(repetition_loop_detected=True)
        score, _, narrative, _ = self._run_pipeline(tl, flags)
        assert score >= 0.9
        assert "temperature" in narrative.lower() or "repetition penalty" in narrative.lower()

    def test_peak_entropy_cited_in_narrative(self):
        """When entropy > 4.0 at some step, narrative should cite the peak step."""
        tl = [_step(i, entropy=2.0, token_text=f"tok{i}") for i in range(10)]
        tl[7] = _step(7, entropy=6.5, token_text="outlier")
        score, _, narrative, _ = self._run_pipeline(tl, HealthFlags())
        assert "step 7" in narrative.lower() or "6.5" in narrative

    def test_empty_timeline_graceful(self):
        """Empty timeline should not crash the pipeline."""
        score, factors, narrative, compound = self._run_pipeline([], HealthFlags())
        assert score == 0.0
        assert factors == []
        assert compound == []


# ============================================================================
# 4. Profile loading tests
# ============================================================================


class TestProfileLoading:
    """Model profiles load correctly and override default thresholds."""

    @pytest.fixture(autouse=True)
    def _profile_dir(self):
        self.profile_dir = Path(__file__).resolve().parent.parent / "configs" / "model_profiles"

    def test_gpt2_profile_loads(self):
        profile = load_model_profile("GPT2LMHeadModel", base_path=self.profile_dir)
        assert profile.high_entropy_threshold_bits == 5.0
        assert profile.l2_explosion_multiplier == 5.0

    def test_llama_profile_loads(self):
        profile = load_model_profile("LlamaForCausalLM", base_path=self.profile_dir)
        assert profile.high_entropy_threshold_bits == 3.5
        assert profile.l2_explosion_multiplier == 10.0

    def test_gpt2_differs_from_llama(self):
        gpt2 = load_model_profile("GPT2LMHeadModel", base_path=self.profile_dir)
        llama = load_model_profile("LlamaForCausalLM", base_path=self.profile_dir)
        assert gpt2.high_entropy_threshold_bits != llama.high_entropy_threshold_bits
        assert gpt2.l2_explosion_multiplier != llama.l2_explosion_multiplier

    def test_unknown_architecture_falls_back_to_default(self):
        profile = load_model_profile("UnknownArchitecture", base_path=self.profile_dir)
        default = ModelProfile()
        assert profile.high_entropy_threshold_bits == default.high_entropy_threshold_bits
        assert profile.l2_explosion_multiplier == default.l2_explosion_multiplier

    def test_profile_threshold_suppresses_early_warning_divergence(self):
        """GPT-2 threshold (5.0) should suppress entropy-margin divergence
        for entropy=4.5, while default (4.0) fires."""
        flags = HealthFlags()
        tl = [_step(i, entropy=4.5, top_k_margin=0.5) for i in range(10)]

        _, sigs_default = compute_early_warning(tl, flags, high_entropy_threshold=4.0)
        _, sigs_gpt2 = compute_early_warning(tl, flags, high_entropy_threshold=5.0)

        assert "entropy_margin_divergence" in sigs_default
        assert "entropy_margin_divergence" not in sigs_gpt2

    def test_llama_lower_threshold_catches_subtler_issues(self):
        """LLaMA threshold (3.5) catches entropy=3.8 that default (4.0) misses."""
        flags = HealthFlags()
        tl = [_step(i, entropy=3.8, top_k_margin=0.5) for i in range(10)]

        _, sigs_default = compute_early_warning(tl, flags, high_entropy_threshold=4.0)
        _, sigs_llama = compute_early_warning(tl, flags, high_entropy_threshold=3.5)

        assert "entropy_margin_divergence" not in sigs_default
        assert "entropy_margin_divergence" in sigs_llama

    def test_gpt2_has_typical_ranges(self):
        gpt2 = load_model_profile("GPT2LMHeadModel", base_path=self.profile_dir)
        assert gpt2.typical_entropy_range is not None
        assert len(gpt2.typical_entropy_range) == 2
        assert gpt2.typical_l2_norm_range is not None
        assert len(gpt2.typical_l2_norm_range) == 2

    def test_llama_has_typical_ranges(self):
        llama = load_model_profile("LlamaForCausalLM", base_path=self.profile_dir)
        assert llama.typical_entropy_range is not None
        assert llama.typical_l2_norm_range is not None


# ============================================================================
# 5. Performance regression benchmarks
# ============================================================================


@pytest.mark.slow
class TestPerformanceRegression:
    """Timing tests — marked slow so they can be skipped in fast CI."""

    @staticmethod
    def _build_large_trace(num_steps: int, num_layers: int):
        timeline = [
            _step(
                i,
                entropy=2.0 + (i % 5) * 0.3,
                top_k_margin=0.4 + (i % 3) * 0.1,
                voter_agreement=0.7 + (i % 4) * 0.05,
                surprisal=1.0 + (i % 6) * 0.2,
            )
            for i in range(num_steps)
        ]
        layers_by_step = [
            [_layer(j, l2_norm_mean=10.0 + j * 0.5, entropy_mean=2.0 + j * 0.1) for j in range(num_layers)]
            for _ in range(num_steps)
        ]
        return timeline, layers_by_step

    def test_risk_score_under_10ms_50_steps(self):
        """50-step trace with 32 layers: risk score should compute in < 10 ms."""
        tl, layers = self._build_large_trace(50, 32)
        summary = _summary(50)
        flags = HealthFlags()

        start = time.perf_counter()
        for _ in range(100):
            compute_risk_score(flags, summary, timeline=tl, layers_by_step=layers)
        elapsed_ms = (time.perf_counter() - start) * 1000 / 100

        assert elapsed_ms < 10, f"Risk score took {elapsed_ms:.1f} ms (limit: 10 ms)"

    def test_compound_signals_under_10ms_50_steps(self):
        """50-step, 32-layer trace: compound signal detection in < 10 ms."""
        tl, layers = self._build_large_trace(50, 32)

        start = time.perf_counter()
        for _ in range(100):
            detect_compound_signals(tl, layers_by_step=layers)
        elapsed_ms = (time.perf_counter() - start) * 1000 / 100

        assert elapsed_ms < 10, f"Compound signals took {elapsed_ms:.1f} ms (limit: 10 ms)"

    def test_early_warning_under_10ms_50_steps(self):
        """50-step trace: early warning computation in < 10 ms."""
        tl, _ = self._build_large_trace(50, 32)
        flags = HealthFlags()

        start = time.perf_counter()
        for _ in range(100):
            compute_early_warning(tl, flags)
        elapsed_ms = (time.perf_counter() - start) * 1000 / 100

        assert elapsed_ms < 10, f"Early warning took {elapsed_ms:.1f} ms (limit: 10 ms)"

    def test_layer_blame_under_20ms_32_layers_50_steps(self):
        """32-layer, 50-step trace: layer blame computation in < 20 ms."""
        _, layers = self._build_large_trace(50, 32)

        start = time.perf_counter()
        for _ in range(100):
            compute_layer_blame(layers)
        elapsed_ms = (time.perf_counter() - start) * 1000 / 100

        assert elapsed_ms < 20, f"Layer blame took {elapsed_ms:.1f} ms (limit: 20 ms)"

    def test_full_pipeline_under_50ms_50_steps_32_layers(self):
        """Full pipeline (risk + compound + early_warning + narrative + blame)
        for a 50-step, 32-layer trace should complete in < 50 ms."""
        tl, layers = self._build_large_trace(50, 32)
        summary = _summary(50)
        flags = HealthFlags()

        start = time.perf_counter()
        for _ in range(50):
            compound = detect_compound_signals(tl, layers_by_step=layers)
            score, factors = compute_risk_score(
                flags,
                summary,
                timeline=tl,
                layers_by_step=layers,
                compound_signals=compound,
            )
            blamed = compute_layer_blame(layers)
            _, warnings = compute_early_warning(tl, flags)
            build_narrative(
                health_flags=flags,
                risk_score=score,
                risk_factors=factors,
                blamed_layers=blamed,
                warning_signals=warnings,
                timeline=tl,
                compound_signals=compound,
                summary=summary,
            )
        elapsed_ms = (time.perf_counter() - start) * 1000 / 50

        assert elapsed_ms < 50, f"Full pipeline took {elapsed_ms:.1f} ms (limit: 50 ms)"

    def test_scales_with_200_steps_64_layers(self):
        """Larger trace (200 steps, 64 layers) should still be under 200 ms."""
        tl, layers = self._build_large_trace(200, 64)
        summary = _summary(200)
        flags = HealthFlags()

        start = time.perf_counter()
        for _ in range(10):
            compound = detect_compound_signals(tl, layers_by_step=layers)
            score, factors = compute_risk_score(
                flags,
                summary,
                timeline=tl,
                layers_by_step=layers,
                compound_signals=compound,
            )
            compute_layer_blame(layers)
            compute_early_warning(tl, flags)
            build_narrative(
                health_flags=flags,
                risk_score=score,
                risk_factors=factors,
                blamed_layers=[],
                warning_signals=[],
                timeline=tl,
                compound_signals=compound,
                summary=summary,
            )
        elapsed_ms = (time.perf_counter() - start) * 1000 / 10

        assert elapsed_ms < 200, f"200-step pipeline took {elapsed_ms:.1f} ms (limit: 200 ms)"
