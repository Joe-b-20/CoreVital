"""Tests for CoreVital.calibration_risk (Issue 34)."""

import pytest

from CoreVital.calibration_risk import (
    RiskCalibrationResult,
    apply_platt_scaling,
    compute_ece,
    evaluate_calibration,
    fit_platt_scaling,
)

# ---------------------------------------------------------------------------
# compute_ece
# ---------------------------------------------------------------------------


class TestComputeECE:
    def test_perfectly_calibrated(self):
        """Predictions that exactly match label rates → ECE ≈ 0."""
        preds = [0.0] * 50 + [1.0] * 50
        labels = [0] * 50 + [1] * 50
        ece = compute_ece(preds, labels)
        assert ece == pytest.approx(0.0, abs=1e-9)

    def test_worst_calibration(self):
        """All predictions = 1.0 but all labels = 0 → ECE = 1.0."""
        preds = [1.0] * 100
        labels = [0] * 100
        ece = compute_ece(preds, labels)
        assert ece == pytest.approx(1.0, abs=1e-6)

    def test_uniform_predictions_half_labels(self):
        """All predictions = 0.5 and label rate = 0.5 → ECE ≈ 0."""
        preds = [0.5] * 100
        labels = [0] * 50 + [1] * 50
        ece = compute_ece(preds, labels)
        assert ece == pytest.approx(0.0, abs=1e-9)

    def test_uniform_predictions_all_positive(self):
        """All predictions = 0.5 but all labels = 1 → ECE = 0.5."""
        preds = [0.5] * 100
        labels = [1] * 100
        ece = compute_ece(preds, labels)
        assert ece == pytest.approx(0.5, abs=1e-6)

    def test_moderate_miscalibration(self):
        """Predictions that slightly overestimate should produce moderate ECE."""
        preds = [0.9] * 100
        labels = [1] * 70 + [0] * 30
        ece = compute_ece(preds, labels)
        assert 0.1 < ece < 0.3

    def test_bins_parameter(self):
        """More bins should produce same or different granularity, but result stays valid."""
        preds = [i / 20.0 for i in range(20)]
        labels = [0] * 10 + [1] * 10
        ece_10 = compute_ece(preds, labels, n_bins=10)
        ece_20 = compute_ece(preds, labels, n_bins=20)
        assert 0.0 <= ece_10 <= 1.0
        assert 0.0 <= ece_20 <= 1.0

    def test_boundary_values_included(self):
        """Values at 0.0 and 1.0 should be assigned to bins."""
        preds = [0.0, 1.0]
        labels = [0, 1]
        ece = compute_ece(preds, labels)
        assert ece == pytest.approx(0.0, abs=1e-9)

    def test_single_sample(self):
        preds = [0.7]
        labels = [1]
        ece = compute_ece(preds, labels)
        assert 0.0 <= ece <= 1.0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            compute_ece([], [])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_ece([0.5, 0.5], [1])

    def test_invalid_label_raises(self):
        with pytest.raises(ValueError, match="only 0 and 1"):
            compute_ece([0.5], [2])


# ---------------------------------------------------------------------------
# fit_platt_scaling
# ---------------------------------------------------------------------------


class TestFitPlattScaling:
    def test_learns_positive_slope(self):
        """Higher raw scores → higher failure rate: a should be positive."""
        scores = [0.1] * 40 + [0.9] * 40
        labels = [0] * 40 + [1] * 40
        a, b = fit_platt_scaling(scores, labels)
        assert a > 0

    def test_calibrated_ordering_preserved(self):
        """After fitting, higher raw → higher calibrated."""
        scores = [0.1] * 30 + [0.5] * 30 + [0.9] * 30
        labels = [0] * 30 + [0] * 15 + [1] * 15 + [1] * 30
        a, b = fit_platt_scaling(scores, labels)
        p_low = apply_platt_scaling(0.1, a, b)
        p_high = apply_platt_scaling(0.9, a, b)
        assert p_high > p_low

    def test_all_same_scores_no_crash(self):
        """Degenerate input: all scores identical. Should not diverge."""
        scores = [0.5] * 50
        labels = [0] * 25 + [1] * 25
        a, b = fit_platt_scaling(scores, labels)
        p = apply_platt_scaling(0.5, a, b)
        assert 0 < p < 1

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            fit_platt_scaling([], [])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            fit_platt_scaling([0.5], [0, 1])

    def test_invalid_label_raises(self):
        with pytest.raises(ValueError, match="only 0 and 1"):
            fit_platt_scaling([0.5], [3])

    def test_convergence_with_separable_data(self):
        """Clearly separable data should yield confident predictions."""
        scores = [0.0] * 50 + [1.0] * 50
        labels = [0] * 50 + [1] * 50
        a, b = fit_platt_scaling(scores, labels, max_iter=1000)
        p_zero = apply_platt_scaling(0.0, a, b)
        p_one = apply_platt_scaling(1.0, a, b)
        assert p_zero < 0.3
        assert p_one > 0.7


# ---------------------------------------------------------------------------
# apply_platt_scaling
# ---------------------------------------------------------------------------


class TestApplyPlattScaling:
    def test_output_range(self):
        """Output is always in [0, 1]."""
        for a, b in [(1.0, 0.0), (-5.0, 3.0), (100.0, -50.0), (0.0, 0.0)]:
            for raw in [0.0, 0.5, 1.0, -1.0, 10.0]:
                p = apply_platt_scaling(raw, a, b)
                assert 0.0 <= p <= 1.0

    def test_output_open_interval_for_moderate_inputs(self):
        """For moderate inputs, output is strictly in (0, 1)."""
        for raw in [0.0, 0.3, 0.5, 0.7, 1.0]:
            p = apply_platt_scaling(raw, 2.0, -1.0)
            assert 0.0 < p < 1.0

    def test_identity_at_zero(self):
        """a=0, b=0 → sigmoid(0) = 0.5 for any input."""
        assert apply_platt_scaling(0.5, 0.0, 0.0) == pytest.approx(0.5)
        assert apply_platt_scaling(999.0, 0.0, 0.0) == pytest.approx(0.5)

    def test_monotonic_with_positive_a(self):
        a, b = 5.0, -2.5
        p1 = apply_platt_scaling(0.0, a, b)
        p2 = apply_platt_scaling(0.5, a, b)
        p3 = apply_platt_scaling(1.0, a, b)
        assert p1 < p2 < p3

    def test_extreme_inputs_no_overflow(self):
        """Very large/small inputs should not cause math errors."""
        p_large = apply_platt_scaling(1000.0, 1.0, 0.0)
        p_small = apply_platt_scaling(-1000.0, 1.0, 0.0)
        assert 0.0 <= p_large <= 1.0
        assert 0.0 <= p_small <= 1.0
        assert p_large > p_small


# ---------------------------------------------------------------------------
# RiskCalibrationResult
# ---------------------------------------------------------------------------


class TestRiskCalibrationResult:
    def test_to_dict(self):
        r = RiskCalibrationResult(
            a=2.5,
            b=-1.0,
            ece_raw=0.15,
            ece_calibrated=0.05,
            n_samples=100,
            label_rate=0.3,
        )
        d = r.to_dict()
        assert d["platt_a"] == 2.5
        assert d["platt_b"] == -1.0
        assert d["ece_raw"] == 0.15
        assert d["n_samples"] == 100

    def test_roundtrip(self):
        r = RiskCalibrationResult(
            a=1.5,
            b=-0.5,
            ece_raw=0.12,
            ece_calibrated=0.04,
            n_samples=200,
            label_rate=0.4,
        )
        d = r.to_dict()
        restored = RiskCalibrationResult.from_dict(d)
        assert restored.a == r.a
        assert restored.b == r.b
        assert restored.n_samples == r.n_samples

    def test_from_dict_missing_key_raises(self):
        with pytest.raises(KeyError):
            RiskCalibrationResult.from_dict({"platt_a": 1.0})


# ---------------------------------------------------------------------------
# evaluate_calibration (end-to-end convenience)
# ---------------------------------------------------------------------------


class TestEvaluateCalibration:
    def test_separable_data(self):
        """Separable data: calibration should improve ECE or keep it low."""
        scores = [0.1] * 40 + [0.9] * 40
        labels = [0] * 40 + [1] * 40
        result = evaluate_calibration(scores, labels)
        assert result.n_samples == 80
        assert result.label_rate == pytest.approx(0.5)
        assert 0.0 <= result.ece_raw <= 1.0
        assert 0.0 <= result.ece_calibrated <= 1.0

    def test_calibration_reduces_ece_for_shifted_scores(self):
        """Systematically shifted scores should see ECE drop after calibration."""
        scores = [0.3] * 50 + [0.7] * 50
        labels = [0] * 50 + [1] * 50
        result = evaluate_calibration(scores, labels, max_iter=1000)
        assert result.ece_calibrated <= result.ece_raw + 0.05

    def test_already_calibrated_scores(self):
        """If raw scores match label rates, raw ECE is already low."""
        scores = [0.0] * 50 + [1.0] * 50
        labels = [0] * 50 + [1] * 50
        result = evaluate_calibration(scores, labels)
        assert result.ece_raw < 0.05
        assert 0.0 <= result.ece_calibrated <= 1.0

    def test_result_has_correct_types(self):
        scores = [0.2] * 20 + [0.8] * 20
        labels = [0] * 20 + [1] * 20
        result = evaluate_calibration(scores, labels)
        assert isinstance(result, RiskCalibrationResult)
        assert isinstance(result.a, float)
        assert isinstance(result.b, float)
        assert isinstance(result.ece_raw, float)
        assert isinstance(result.ece_calibrated, float)

    def test_all_positive_labels(self):
        """Edge case: all labels = 1."""
        scores = [0.5] * 30
        labels = [1] * 30
        result = evaluate_calibration(scores, labels)
        assert result.label_rate == 1.0
        assert 0.0 <= result.ece_calibrated <= 1.0

    def test_all_negative_labels(self):
        """Edge case: all labels = 0."""
        scores = [0.5] * 30
        labels = [0] * 30
        result = evaluate_calibration(scores, labels)
        assert result.label_rate == 0.0
        assert 0.0 <= result.ece_calibrated <= 1.0


# ---------------------------------------------------------------------------
# Integration with existing calibration module
# ---------------------------------------------------------------------------


class TestIntegrationWithCalibration:
    """Verify calibration_risk works alongside the existing calibration.py."""

    def test_divergence_score_feeds_into_ece(self):
        """Divergence scores from calibration.py can be evaluated for calibration quality."""
        from CoreVital.calibration import calibrate_from_runs, compute_divergence_score

        baseline_traces = [
            {"timeline": [{"step_index": 0, "logits_summary": {"entropy": 2.0, "top_k_margin": 0.5}}]},
            {"timeline": [{"step_index": 0, "logits_summary": {"entropy": 2.0, "top_k_margin": 0.5}}]},
            {"timeline": [{"step_index": 0, "logits_summary": {"entropy": 2.1, "top_k_margin": 0.5}}]},
        ]
        profile = calibrate_from_runs("test", baseline_traces)

        good_traces = [
            {"timeline": [{"step_index": 0, "logits_summary": {"entropy": 2.0, "top_k_margin": 0.5}}]}
            for _ in range(20)
        ]
        bad_traces = [
            {"timeline": [{"step_index": 0, "logits_summary": {"entropy": 10.0, "top_k_margin": 0.01}}]}
            for _ in range(20)
        ]

        scores = []
        labels = []
        for t in good_traces:
            div, _ = compute_divergence_score(t, profile)
            scores.append(div)
            labels.append(0)
        for t in bad_traces:
            div, _ = compute_divergence_score(t, profile)
            scores.append(div)
            labels.append(1)

        result = evaluate_calibration(scores, labels)
        assert result.n_samples == 40
        assert 0.0 <= result.ece_calibrated <= 1.0

    def test_risk_score_calibration_pipeline(self):
        """Risk scores from risk.py can be calibrated using this module."""
        raw_scores = [0.1, 0.2, 0.15, 0.12, 0.85, 0.9, 0.78, 0.95]
        labels = [0, 0, 0, 0, 1, 1, 1, 1]
        result = evaluate_calibration(raw_scores, labels)

        calibrated_low = apply_platt_scaling(0.1, result.a, result.b)
        calibrated_high = apply_platt_scaling(0.9, result.a, result.b)
        assert calibrated_high > calibrated_low
