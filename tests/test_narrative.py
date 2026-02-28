# ============================================================================
# CoreVital - Narrative Tests (Issue 8 rewrite)
# ============================================================================

from dataclasses import dataclass
from typing import List

from CoreVital.narrative import (
    _dominant_factor,
    _humanize_factor,
    _token_at_step,
    build_narrative,
)
from CoreVital.reporting.schema import (
    HealthFlags,
    LogitsSummary,
    TimelineStep,
    TokenInfo,
)


def _step(idx: int, token_text: str = "tok", entropy: float = 2.0) -> TimelineStep:
    """Helper to build a minimal TimelineStep."""
    return TimelineStep(
        step_index=idx,
        token=TokenInfo(token_id=idx, token_text=token_text, is_prompt_token=False),
        logits_summary=LogitsSummary(entropy=entropy),
        layers=[],
        extensions={},
    )


@dataclass
class FakeCompoundSignal:
    name: str
    description: str
    severity: float
    evidence: List[str]


class TestBuildNarrative:
    def test_low_risk(self):
        flags = HealthFlags()
        text = build_narrative(flags, 0.1, [], [], [], [])
        assert "low risk" in text.lower()
        assert "0.10" in text
        assert "no significant anomalies" in text.lower()

    def test_low_risk_with_signals_does_not_claim_no_anomalies(self):
        """Low risk but with warning signals should not say 'no anomalies'."""
        flags = HealthFlags()
        text = build_narrative(flags, 0.2, ["elevated_entropy"], [], ["entropy_accelerating"], [])
        assert "low risk" in text.lower()
        assert "no significant anomalies" not in text.lower()
        assert "mild signals" in text.lower()

    def test_low_risk_with_blamed_layers_does_not_claim_no_anomalies(self):
        flags = HealthFlags()
        text = build_narrative(flags, 0.2, [], [0, 3], [], [])
        assert "no significant anomalies" not in text.lower()
        assert "mild signals" in text.lower()

    def test_moderate_risk(self):
        flags = HealthFlags()
        text = build_narrative(flags, 0.5, [], [], [], [])
        assert "moderate risk" in text.lower()
        assert "0.50" in text

    def test_high_risk_with_primary_factor(self):
        flags = HealthFlags()
        text = build_narrative(flags, 0.9, ["elevated_entropy"], [], [], [])
        assert "high risk" in text.lower()
        assert "0.90" in text
        assert "elevated output uncertainty" in text

    def test_high_risk_picks_dominant_factor(self):
        """Should pick repetition_loop over elevated_entropy regardless of order."""
        flags = HealthFlags()
        text = build_narrative(flags, 0.9, ["elevated_entropy", "repetition_loop"], [], [], [])
        assert "repetition loop" in text.lower()

    def test_high_risk_no_factors(self):
        flags = HealthFlags()
        text = build_narrative(flags, 0.9, [], [], [], [])
        assert "multiple signals" in text

    def test_entropy_peak_referenced(self):
        flags = HealthFlags()
        timeline = [
            _step(0, "Hello", 2.0),
            _step(1, "world", 5.5),
            _step(2, "!", 3.0),
        ]
        text = build_narrative(flags, 0.5, [], [], [], timeline)
        assert "5.5" in text
        assert "step 1" in text
        assert '"world"' in text

    def test_entropy_peak_with_offset_step_index(self):
        """step_index offset by prompt length should be cited correctly."""
        flags = HealthFlags()
        timeline = [
            _step(10, "Hello", 2.0),
            _step(11, "world", 5.5),
            _step(12, "!", 3.0),
        ]
        text = build_narrative(flags, 0.5, [], [], [], timeline)
        assert "step 11" in text
        assert '"world"' in text

    def test_no_entropy_peak_when_below_threshold(self):
        flags = HealthFlags()
        timeline = [_step(0, "a", 2.0), _step(1, "b", 3.5)]
        text = build_narrative(flags, 0.2, [], [], [], timeline)
        assert "peak entropy" not in text.lower()

    def test_entropy_accelerating_trend(self):
        """New signal name from early_warning redesign."""
        flags = HealthFlags()
        entropies = [1.0, 1.2, 1.1, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
        timeline = [_step(i, f"t{i}", e) for i, e in enumerate(entropies)]
        text = build_narrative(flags, 0.4, [], [], ["entropy_accelerating"], timeline)
        assert "entropy rose" in text.lower()
        assert "progressive degradation" in text.lower()

    def test_entropy_rising_legacy_still_works(self):
        """Old signal name still triggers the trend narrative."""
        flags = HealthFlags()
        entropies = [1.0, 1.2, 1.1, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
        timeline = [_step(i, f"t{i}", e) for i, e in enumerate(entropies)]
        text = build_narrative(flags, 0.4, [], [], ["entropy_rising"], timeline)
        assert "entropy rose" in text.lower()

    def test_entropy_trend_needs_min_steps(self):
        """Trend only reported when >= 6 entropy values."""
        flags = HealthFlags()
        timeline = [_step(0, "a", 1.0), _step(1, "b", 5.0)]
        text = build_narrative(flags, 0.4, [], [], ["entropy_accelerating"], timeline)
        assert "entropy rose" not in text.lower()

    def test_compound_signals_included(self):
        flags = HealthFlags()
        cs = [
            FakeCompoundSignal(
                "context_loss",
                "Context loss detected after step 5.",
                0.6,
                ["entropy > 4"],
            ),
            FakeCompoundSignal(
                "confident_confusion",
                "Model showed confident confusion.",
                0.5,
                [],
            ),
        ]
        text = build_narrative(flags, 0.5, [], [], [], [], compound_signals=cs)
        assert "context loss detected" in text.lower()
        assert "confident confusion" in text.lower()

    def test_compound_signals_capped_at_two(self):
        flags = HealthFlags()
        cs = [
            FakeCompoundSignal("a", "Signal A.", 0.3, []),
            FakeCompoundSignal("b", "Signal B.", 0.3, []),
            FakeCompoundSignal("c", "Signal C.", 0.3, []),
        ]
        text = build_narrative(flags, 0.5, [], [], [], [], compound_signals=cs)
        assert "signal a" in text.lower()
        assert "signal b" in text.lower()
        assert "signal c" not in text.lower()

    def test_blamed_layers_simple_list(self):
        """Current format: List[int] from compute_layer_blame."""
        flags = HealthFlags()
        text = build_narrative(flags, 0.5, [], [0, 4, 7], [], [])
        assert "L0" in text
        assert "L4" in text
        assert "L7" in text

    def test_blamed_layers_rich_dict(self):
        """Future format from Issue 7: List[dict] with severity/reasons."""
        flags = HealthFlags()
        blamed = [
            {
                "layer": 3,
                "severity": 0.8,
                "reasons": ["NaN detected", "attention collapse"],
            },
            {"layer": 7, "severity": 0.5, "reasons": ["high L2 norm"]},
        ]
        text = build_narrative(flags, 0.5, [], blamed, [], [])
        assert "Layer 3" in text
        assert "NaN detected" in text
        assert "attention collapse" in text
        assert "Layer 7" in text

    def test_recommendation_repetition_loop(self):
        flags = HealthFlags()
        text = build_narrative(flags, 0.9, ["repetition_loop"], [], [], [])
        assert "lower temperature" in text.lower()
        assert "repetition penalty" in text.lower()

    def test_recommendation_mid_layer(self):
        flags = HealthFlags()
        text = build_narrative(flags, 0.9, ["mid_layer_anomaly"], [], [], [])
        assert "precision" in text.lower()

    def test_recommendation_entropy(self):
        flags = HealthFlags()
        text = build_narrative(flags, 0.9, ["elevated_entropy"], [], [], [])
        assert "refine the prompt" in text.lower()

    def test_no_recommendation_below_threshold(self):
        flags = HealthFlags()
        text = build_narrative(flags, 0.5, ["elevated_entropy"], [], [], [])
        assert "consider:" not in text.lower()

    def test_empty_timeline(self):
        flags = HealthFlags()
        text = build_narrative(flags, 0.0, [], [], [], [])
        assert "low risk" in text.lower()
        assert len(text) > 0

    def test_empty_everything_returns_fallback(self):
        flags = HealthFlags()
        text = build_narrative(flags, 0.0, [], [], [], [])
        assert text


class TestDominantFactor:
    def test_prefers_boolean_flags_over_continuous(self):
        factors = ["elevated_entropy", "repetition_loop", "low_topk_mass"]
        assert _dominant_factor(factors) == "repetition_loop"

    def test_nan_highest_priority(self):
        factors = ["repetition_loop", "nan_or_inf"]
        assert _dominant_factor(factors) == "nan_or_inf"

    def test_compound_fallback(self):
        factors = ["compound:context_loss"]
        assert _dominant_factor(factors) == "compound:context_loss"

    def test_empty_returns_multiple_signals(self):
        assert _dominant_factor([]) == "multiple signals"

    def test_unknown_factor_returned_as_is(self):
        factors = ["some_new_factor"]
        assert _dominant_factor(factors) == "some_new_factor"


class TestHelpers:
    def test_humanize_known_factors(self):
        assert "repetition loop" in _humanize_factor("repetition_loop")
        assert "NaN/Inf" in _humanize_factor("nan_or_inf")
        assert "top-K" in _humanize_factor("low_topk_mass")
        assert "accelerating" in _humanize_factor("entropy_accelerating")

    def test_humanize_compound_factor(self):
        assert _humanize_factor("compound:context_loss") == "context loss"

    def test_humanize_unknown_factor(self):
        assert _humanize_factor("some_new_thing") == "some new thing"

    def test_token_at_step_valid(self):
        timeline = [_step(0, "hello")]
        assert _token_at_step(timeline, 0) == "hello"

    def test_token_at_step_offset_index(self):
        """step_index=10 should be found even at list position 0."""
        timeline = [_step(10, "hello")]
        assert _token_at_step(timeline, 10) == "hello"

    def test_token_at_step_out_of_range(self):
        assert _token_at_step([], 5) == "?"

    def test_token_at_step_missing_index(self):
        timeline = [_step(0, "hello")]
        assert _token_at_step(timeline, 99) == "?"

    def test_token_at_step_empty_text(self):
        step = TimelineStep(
            step_index=0,
            token=TokenInfo(token_id=0, token_text="", is_prompt_token=False),
            logits_summary=LogitsSummary(),
            layers=[],
            extensions={},
        )
        assert _token_at_step([step], 0) == "?"
