# ============================================================================
# CoreVital - Narrative Tests (Phase-7)
# ============================================================================


from CoreVital.narrative import build_narrative
from CoreVital.reporting.schema import HealthFlags


class TestBuildNarrative:
    def test_low_risk(self):
        flags = HealthFlags()
        text = build_narrative(flags, 0.2, [], [])
        assert "low risk" in text.lower()

    def test_high_risk(self):
        flags = HealthFlags()
        text = build_narrative(flags, 0.9, [], [])
        assert "high risk" in text.lower()

    def test_repetition_mentioned(self):
        flags = HealthFlags(repetition_loop_detected=True)
        text = build_narrative(flags, 0.5, [], [])
        assert "repetition" in text.lower()

    def test_nan_inf_mentioned(self):
        flags = HealthFlags(nan_detected=True)
        text = build_narrative(flags, 0.5, [], [])
        assert "nan" in text.lower() or "numerical" in text.lower()

    def test_blamed_layers_mentioned(self):
        flags = HealthFlags()
        text = build_narrative(flags, 0.5, [0, 4], [])
        assert "L0" in text and "L4" in text

    def test_entropy_rising_mentioned(self):
        flags = HealthFlags()
        text = build_narrative(flags, 0.3, [], ["entropy_rising"])
        assert "entropy" in text.lower()

    def test_empty_input(self):
        flags = HealthFlags()
        text = build_narrative(flags, 0.0, [], [])
        assert "low risk" in text.lower()
        assert len(text) > 0
