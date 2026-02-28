# ============================================================================
# CoreVital - Early Warning Tests (Phase-2.5, Issues 3 & 16)
# ============================================================================

import pytest

from CoreVital.early_warning import (
    DEFAULT_HIGH_ENTROPY_THRESHOLD,
    _detect_entropy_acceleration,
    _detect_entropy_margin_divergence,
    _detect_margin_collapse,
    _detect_surprisal_volatility,
    compute_early_warning,
)
from CoreVital.reporting.schema import (
    HealthFlags,
    LogitsSummary,
    TimelineStep,
    TokenInfo,
)


def _step(
    entropy: float = 2.0,
    step_index: int = 0,
    top_k_margin: float | None = None,
    surprisal: float | None = None,
    token_text: str = "x",
) -> TimelineStep:
    return TimelineStep(
        step_index=step_index,
        token=TokenInfo(token_id=0, token_text=token_text, is_prompt_token=False),
        logits_summary=LogitsSummary(
            entropy=entropy,
            top_k_margin=top_k_margin,
            surprisal=surprisal,
        ),
        layers=[],
    )


# ---- Entropy acceleration ----

class TestEntropyAcceleration:
    def test_accelerating_entropy_fires(self):
        """Entropy whose rate of increase is itself increasing."""
        # 10 steps: slow rise then accelerating
        entropies = [1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.2, 2.8, 3.6, 4.8]
        assert _detect_entropy_acceleration(entropies, window_size=3)

    def test_flat_entropy_does_not_fire(self):
        entropies = [2.0] * 10
        assert not _detect_entropy_acceleration(entropies, window_size=3)

    def test_too_few_steps(self):
        assert not _detect_entropy_acceleration([1.0, 2.0, 3.0], window_size=3)

    def test_linearly_rising_does_not_fire(self):
        """Constant rate of increase: deltas are equal, so no acceleration."""
        entropies = [float(i) for i in range(10)]  # 0,1,2,...,9
        assert not _detect_entropy_acceleration(entropies, window_size=3)

    def test_integration_entropy_accelerating(self):
        """compute_early_warning returns entropy_accelerating signal."""
        timeline = [_step(e, i) for i, e in enumerate(
            [1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.2, 2.8, 3.6, 4.8]
        )]
        risk, signals = compute_early_warning(timeline, HealthFlags(), window_size=3)
        assert "entropy_accelerating" in signals
        assert risk >= 0.7


# ---- Margin collapse ----

class TestMarginCollapse:
    def test_collapsed_all_below_threshold(self):
        collapsed, declining = _detect_margin_collapse(
            [0.05, 0.04, 0.03, 0.02, 0.01], window_size=5,
        )
        assert collapsed
        assert not declining

    def test_declining_margin(self):
        early = [0.8] * 5
        late = [0.2] * 5
        collapsed, declining = _detect_margin_collapse(early + late, window_size=5)
        assert not collapsed
        assert declining

    def test_healthy_margin(self):
        collapsed, declining = _detect_margin_collapse(
            [0.5, 0.5, 0.5, 0.5, 0.5], window_size=5,
        )
        assert not collapsed
        assert not declining

    def test_too_few_steps(self):
        collapsed, declining = _detect_margin_collapse([0.01, 0.01], window_size=5)
        assert not collapsed
        assert not declining

    def test_integration_margin_collapsed(self):
        timeline = [_step(top_k_margin=0.05, step_index=i) for i in range(5)]
        risk, signals = compute_early_warning(timeline, HealthFlags())
        assert "margin_collapsed" in signals
        assert risk >= 0.6

    def test_integration_margin_declining(self):
        timeline = (
            [_step(top_k_margin=0.8, step_index=i) for i in range(5)]
            + [_step(top_k_margin=0.2, step_index=i) for i in range(5, 10)]
        )
        risk, signals = compute_early_warning(timeline, HealthFlags())
        assert "margin_declining" in signals
        assert risk >= 0.5


# ---- Surprisal volatility ----

class TestSurprisalVolatility:
    def test_volatile_surprisal_fires(self):
        surprisals = [0.01, 0.01, 0.01, 0.01, 10.0]
        assert _detect_surprisal_volatility(surprisals, window_size=5)

    def test_stable_surprisal_does_not_fire(self):
        surprisals = [3.0, 3.1, 2.9, 3.0, 3.0]
        assert not _detect_surprisal_volatility(surprisals, window_size=5)

    def test_too_few(self):
        assert not _detect_surprisal_volatility([1.0], window_size=5)

    def test_zero_mean(self):
        assert not _detect_surprisal_volatility([0.0, 0.0, 0.0, 0.0, 0.0], window_size=5)

    def test_integration_surprisal_volatile(self):
        timeline = [_step(surprisal=s, step_index=i) for i, s in enumerate(
            [0.01, 0.01, 0.01, 0.01, 10.0]
        )]
        risk, signals = compute_early_warning(timeline, HealthFlags())
        assert "surprisal_volatile" in signals
        assert risk >= 0.5


# ---- Entropy-margin divergence ----

class TestEntropyMarginDivergence:
    def test_divergence_fires(self):
        assert _detect_entropy_margin_divergence(
            [5.0] * 5, [0.5] * 5, window_size=5, high_entropy_threshold=4.0,
        )

    def test_low_entropy_no_divergence(self):
        assert not _detect_entropy_margin_divergence(
            [2.0] * 5, [0.5] * 5, window_size=5, high_entropy_threshold=4.0,
        )

    def test_low_margin_no_divergence(self):
        assert not _detect_entropy_margin_divergence(
            [5.0] * 5, [0.1] * 5, window_size=5, high_entropy_threshold=4.0,
        )

    def test_custom_threshold(self):
        """Issue 16: threshold from profile, not hardcoded 4.0."""
        assert not _detect_entropy_margin_divergence(
            [5.0] * 5, [0.5] * 5, window_size=5, high_entropy_threshold=6.0,
        )
        assert _detect_entropy_margin_divergence(
            [5.0] * 5, [0.5] * 5, window_size=5, high_entropy_threshold=4.0,
        )

    def test_integration_divergence(self):
        timeline = [
            _step(entropy=5.0, top_k_margin=0.5, step_index=i) for i in range(5)
        ]
        risk, signals = compute_early_warning(timeline, HealthFlags())
        assert "entropy_margin_divergence" in signals
        assert risk >= 0.55


# ---- Profile threshold (Issue 16) ----

class TestProfileThreshold:
    def test_default_threshold_is_4(self):
        assert DEFAULT_HIGH_ENTROPY_THRESHOLD == 4.0

    def test_custom_threshold_changes_divergence(self):
        """Raising threshold suppresses entropy_margin_divergence."""
        timeline = [
            _step(entropy=5.0, top_k_margin=0.5, step_index=i) for i in range(5)
        ]
        _, signals_default = compute_early_warning(timeline, HealthFlags())
        assert "entropy_margin_divergence" in signals_default

        _, signals_high = compute_early_warning(
            timeline, HealthFlags(), high_entropy_threshold=6.0,
        )
        assert "entropy_margin_divergence" not in signals_high


# ---- Health flags pass-through ----

class TestHealthFlagPassthrough:
    def test_repetition_loop(self):
        flags = HealthFlags(repetition_loop_detected=True)
        risk, signals = compute_early_warning([_step()], flags)
        assert risk >= 0.9
        assert "repetition_loop" in signals

    def test_mid_layer_anomaly(self):
        flags = HealthFlags(mid_layer_anomaly_detected=True)
        risk, signals = compute_early_warning([_step()], flags)
        assert "mid_layer_anomaly" in signals
        assert risk >= 0.6

    def test_attention_collapse(self):
        flags = HealthFlags(attention_collapse_detected=True)
        risk, signals = compute_early_warning([_step()], flags)
        assert "attention_collapse" in signals
        assert risk >= 0.4


# ---- Edge cases ----

class TestEdgeCases:
    def test_empty_timeline(self):
        risk, signals = compute_early_warning([], HealthFlags())
        assert risk == 0.0
        assert isinstance(signals, list)
        assert len(signals) == 0

    def test_single_step(self):
        risk, signals = compute_early_warning([_step()], HealthFlags())
        assert risk == 0.0
        assert len(signals) == 0

    def test_empty_logits_skipped(self):
        step = TimelineStep(
            step_index=0,
            token=TokenInfo(token_id=0, token_text="x", is_prompt_token=False),
            logits_summary=LogitsSummary(),
            layers=[],
        )
        risk, signals = compute_early_warning([step], HealthFlags())
        assert risk == 0.0

    def test_combined_signals_max_risk(self):
        """Multiple signals active; risk is max of all contributions."""
        timeline = [
            _step(entropy=5.0, top_k_margin=0.05, surprisal=s, step_index=i)
            for i, s in enumerate([1.0, 10.0, 0.5, 12.0, 1.0])
        ]
        flags = HealthFlags(repetition_loop_detected=True)
        risk, signals = compute_early_warning(timeline, flags)
        assert risk == 0.9  # repetition_loop dominates
        assert "repetition_loop" in signals
        assert "margin_collapsed" in signals

    def test_return_type(self):
        risk, signals = compute_early_warning([_step()], HealthFlags())
        assert isinstance(risk, float)
        assert isinstance(signals, list)
        assert 0.0 <= risk <= 1.0
