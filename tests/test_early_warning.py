# ============================================================================
# CoreVital - Early Warning Tests (Phase-4)
# ============================================================================


from CoreVital.early_warning import compute_early_warning
from CoreVital.reporting.schema import (
    HealthFlags,
    LogitsSummary,
    TimelineStep,
    TokenInfo,
)


def _step(entropy: float, step_index: int = 0) -> TimelineStep:
    return TimelineStep(
        step_index=step_index,
        token=TokenInfo(token_id=0, token_text="x", is_prompt_token=False),
        logits_summary=LogitsSummary(entropy=entropy),
        layers=[],
    )


class TestComputeEarlyWarning:
    def test_repetition_loop_high_risk(self):
        flags = HealthFlags(repetition_loop_detected=True)
        timeline = [_step(2.0), _step(2.5)]
        risk, signals = compute_early_warning(timeline, flags)
        assert risk == 0.9
        assert "repetition_loop" in signals

    def test_entropy_rising_and_high_risk_mid(self):
        flags = HealthFlags()
        # Rising: first 5 low, last 5 high; max > 4
        timeline = [_step(1.0, i) for i in range(5)] + [_step(5.0, i) for i in range(5, 10)]
        risk, signals = compute_early_warning(timeline, flags)
        assert risk == 0.6
        assert "entropy_rising" in signals
        assert "high_entropy" in signals

    def test_baseline_low_risk(self):
        flags = HealthFlags()
        timeline = [_step(2.0), _step(2.1)]
        risk, signals = compute_early_warning(timeline, flags)
        assert risk == 0.3
        assert "entropy_rising" not in signals

    def test_high_entropy_only_adds_signal(self):
        flags = HealthFlags()
        timeline = [_step(5.0), _step(5.1)]  # max > 4, not rising (only 2 steps, last > first so rising actually True)
        risk, signals = compute_early_warning(timeline, flags)
        assert risk == 0.6  # entropy_rising and max > 4
        assert "high_entropy" in signals

    def test_mid_layer_anomaly_in_signals(self):
        flags = HealthFlags(mid_layer_anomaly_detected=True)
        timeline = [_step(2.0)]
        risk, signals = compute_early_warning(timeline, flags)
        assert "mid_layer_anomaly" in signals

    def test_empty_timeline(self):
        flags = HealthFlags()
        risk, signals = compute_early_warning([], flags)
        assert risk == 0.3
        assert isinstance(signals, list)
