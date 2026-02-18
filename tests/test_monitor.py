# ============================================================================
# CoreVital - Library API (CoreVitalMonitor) Tests
# ============================================================================

from unittest.mock import patch

from CoreVital.config import Config
from CoreVital.instrumentation.collector import InstrumentationCollector
from CoreVital.monitor import CoreVitalMonitor


class TestCoreVitalMonitor:
    """Test CoreVitalMonitor with mock model (no real HF load)."""

    def test_run_and_get_risk_score(self, mock_model_bundle):
        """run() then get_risk_score() returns value in [0, 1]."""
        config = Config()
        config.model.hf_id = "mock-causal"
        config.device.requested = "cpu"
        config.generation.max_new_tokens = 2
        config.generation.seed = 42
        collector = InstrumentationCollector(config)
        collector.model_bundle = mock_model_bundle
        from CoreVital.reporting.report_builder import ReportBuilder

        results = collector.run("Hello")
        monitor = CoreVitalMonitor(capture_mode="summary", max_new_tokens=2)
        monitor._results = results
        monitor._report = ReportBuilder(config).build(results, "Hello")
        assert 0 <= monitor.get_risk_score() <= 1

    def test_run_via_run_method(self, mock_model_bundle):
        """Monitor.run(model_id, prompt) runs collector and builds report when load_model is patched."""
        with patch("CoreVital.instrumentation.collector.load_model", return_value=mock_model_bundle):
            monitor = CoreVitalMonitor(capture_mode="summary", max_new_tokens=2)
            monitor.run("mock-causal", "Hi")
        assert monitor.get_risk_score() is not None
        assert isinstance(monitor.get_health_flags(), dict)
        assert "risk_score" in monitor.get_summary()
        assert "fingerprint" in monitor.get_summary()

    def test_should_intervene_below_threshold(self, mock_model_bundle):
        """should_intervene() False when risk below intervene_on_risk_above."""
        config = Config()
        config.model.hf_id = "mock-causal"
        config.device.requested = "cpu"
        config.generation.max_new_tokens = 2
        config.generation.seed = 42
        collector = InstrumentationCollector(config)
        collector.model_bundle = mock_model_bundle
        results = collector.run("Hello")
        from CoreVital.reporting.report_builder import ReportBuilder

        report = ReportBuilder(config).build(results, "Hello")
        monitor = CoreVitalMonitor(intervene_on_risk_above=0.99)
        monitor._report = report
        risk = monitor.get_risk_score()
        if risk < 0.99:
            assert monitor.should_intervene() is False, f"should_intervene() True but risk={risk} < 0.99"
        else:
            assert monitor.should_intervene() is True, f"should_intervene() False but risk={risk} >= 0.99"

    def test_should_intervene_true_when_risk_high(self, mock_model_bundle):
        """should_intervene() True when risk >= intervene_on_risk_above."""
        config = Config()
        config.model.hf_id = "mock-causal"
        config.device.requested = "cpu"
        config.generation.max_new_tokens = 2
        collector = InstrumentationCollector(config)
        collector.model_bundle = mock_model_bundle
        results = collector.run("Hello")
        from CoreVital.reporting.report_builder import ReportBuilder

        report = ReportBuilder(config).build(results, "Hello")
        # Force high risk in extensions
        report.extensions["risk"] = {"risk_score": 0.95, "risk_factors": [], "blamed_layers": []}
        report.extensions["early_warning"] = {"failure_risk": 0.9, "warning_signals": []}
        monitor = CoreVitalMonitor(intervene_on_risk_above=0.8)
        monitor._report = report
        assert monitor.should_intervene() is True

    def test_get_risk_score_no_run(self):
        """get_risk_score() returns 0.0 when no run."""
        monitor = CoreVitalMonitor()
        assert monitor.get_risk_score() == 0.0

    def test_get_health_flags_no_run(self):
        """get_health_flags() returns {} when no run."""
        monitor = CoreVitalMonitor()
        assert monitor.get_health_flags() == {}

    def test_wrap_generation_context(self, mock_model_bundle):
        """wrap_generation is a context manager that yields self after running."""
        with patch("CoreVital.instrumentation.collector.load_model", return_value=mock_model_bundle):
            monitor = CoreVitalMonitor(max_new_tokens=2)
            with monitor.wrap_generation("mock-causal", "Hi") as m:
                assert m is monitor
                assert m.get_report() is not None
            assert monitor.get_summary()["trace_id"] is not None
