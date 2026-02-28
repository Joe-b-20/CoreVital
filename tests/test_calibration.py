"""Tests for CoreVital.calibration (Issue 33)."""

import json
import math

import pytest

from CoreVital.calibration import (
    CalibrationProfile,
    MetricDistribution,
    calibrate_from_runs,
    compute_divergence_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trace(entropy_values, margin_values=None, surprisal_values=None, layers=None):
    """Build a minimal trace dict matching the report schema."""
    timeline = []
    for i, ent in enumerate(entropy_values):
        step = {
            "step_index": i,
            "logits_summary": {"entropy": ent},
        }
        if margin_values and i < len(margin_values):
            step["logits_summary"]["top_k_margin"] = margin_values[i]
        if surprisal_values and i < len(surprisal_values):
            step["logits_summary"]["surprisal"] = surprisal_values[i]
        if layers and i < len(layers):
            step["layers"] = layers[i]
        timeline.append(step)
    return {"timeline": timeline}


def _make_layer(layer_index, l2_norm_mean=None, entropy_mean=None):
    layer = {"layer_index": layer_index, "hidden_summary": {}, "attention_summary": {}}
    if l2_norm_mean is not None:
        layer["hidden_summary"]["l2_norm_mean"] = l2_norm_mean
    if entropy_mean is not None:
        layer["attention_summary"]["entropy_mean"] = entropy_mean
    return layer


# ---------------------------------------------------------------------------
# MetricDistribution
# ---------------------------------------------------------------------------


class TestMetricDistribution:
    def test_empty(self):
        md = MetricDistribution()
        assert md.mean == 0.0
        assert md.std == 0.0
        assert md.median == 0.0
        assert md.percentile(50) == 0.0
        assert md.z_score(5.0) == 0.0

    def test_single_value(self):
        md = MetricDistribution(values=[3.0])
        assert md.mean == 3.0
        assert md.std == 0.0
        assert md.median == 3.0

    def test_basic_stats(self):
        md = MetricDistribution(values=[1.0, 2.0, 3.0, 4.0, 5.0])
        assert md.mean == 3.0
        assert md.median == 3.0
        assert md.std > 0

    def test_z_score(self):
        md = MetricDistribution(values=[10.0, 10.0, 10.0, 10.0, 10.0, 12.0])
        z = md.z_score(10.0)
        assert z < 0  # below mean (mean is slightly > 10)

    def test_z_score_zero_std(self):
        md = MetricDistribution(values=[5.0, 5.0, 5.0])
        assert md.z_score(5.0) == 0.0
        assert md.z_score(100.0) == 0.0

    def test_is_anomalous(self):
        md = MetricDistribution(values=[1.0, 1.0, 1.0, 1.0, 1.0, 1.1])
        assert not md.is_anomalous(1.0)
        assert md.is_anomalous(100.0)

    def test_percentile(self):
        md = MetricDistribution(values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        p5 = md.percentile(5)
        p95 = md.percentile(95)
        assert p5 <= p95
        assert p5 == 1.0
        assert p95 == 10.0


# ---------------------------------------------------------------------------
# calibrate_from_runs
# ---------------------------------------------------------------------------


class TestCalibrateFromRuns:
    def test_three_traces(self):
        traces = [
            _make_trace([2.0, 2.1, 2.2], margin_values=[0.5, 0.6, 0.4]),
            _make_trace([2.1, 2.0, 2.3], margin_values=[0.5, 0.5, 0.5]),
            _make_trace([2.2, 2.1, 2.0], margin_values=[0.6, 0.4, 0.5]),
        ]
        profile = calibrate_from_runs("test-model", traces)

        assert profile.model_id == "test-model"
        assert profile.num_runs == 3
        assert len(profile.entropy_per_step.values) == 9
        assert len(profile.margin_per_step.values) == 9
        assert abs(profile.entropy_per_step.mean - 2.1) < 0.02

    def test_with_layer_data(self):
        layers_per_step = [[_make_layer(0, l2_norm_mean=100.0, entropy_mean=2.5)]]
        traces = [_make_trace([2.0], layers=layers_per_step)]
        profile = calibrate_from_runs("m", traces)

        assert 0 in profile.l2_norm_per_layer
        assert profile.l2_norm_per_layer[0].values == [100.0]
        assert 0 in profile.attention_entropy_per_layer
        assert profile.attention_entropy_per_layer[0].values == [2.5]

    def test_empty_traces(self):
        profile = calibrate_from_runs("m", [])
        assert profile.num_runs == 0
        assert profile.entropy_per_step.values == []

    def test_surprisal_collected(self):
        traces = [_make_trace([2.0], surprisal_values=[5.5])]
        profile = calibrate_from_runs("m", traces)
        assert profile.surprisal_per_step.values == [5.5]


# ---------------------------------------------------------------------------
# compute_divergence_score
# ---------------------------------------------------------------------------


class TestComputeDivergenceScore:
    def _build_baseline(self):
        """Baseline with tight distribution around entropy=2.0, margin=0.5."""
        traces = [
            _make_trace([2.0, 2.0, 2.0], margin_values=[0.5, 0.5, 0.5]),
            _make_trace([2.0, 2.0, 2.0], margin_values=[0.5, 0.5, 0.5]),
            _make_trace([2.1, 1.9, 2.0], margin_values=[0.5, 0.5, 0.5]),
        ]
        return calibrate_from_runs("test", traces)

    def test_matching_trace_low_divergence(self):
        baseline = self._build_baseline()
        trace = _make_trace([2.0, 2.0, 2.0], margin_values=[0.5, 0.5, 0.5])
        score, anomalies = compute_divergence_score(trace, baseline)
        assert score < 0.1
        assert len(anomalies) == 0

    def test_anomalous_trace_high_divergence(self):
        baseline = self._build_baseline()
        trace = _make_trace([10.0, 10.0, 10.0], margin_values=[0.5, 0.5, 0.5])
        score, anomalies = compute_divergence_score(trace, baseline)
        assert score > 0.3
        assert len(anomalies) > 0

    def test_empty_trace(self):
        baseline = self._build_baseline()
        score, anomalies = compute_divergence_score({"timeline": []}, baseline)
        assert score == 0.0
        assert anomalies == []

    def test_z_scores_reflect_distance(self):
        baseline = self._build_baseline()
        trace_close = _make_trace([2.1])
        trace_far = _make_trace([8.0])
        score_close, _ = compute_divergence_score(trace_close, baseline)
        score_far, _ = compute_divergence_score(trace_far, baseline)
        assert score_far > score_close

    def test_divergence_capped_at_one(self):
        baseline = self._build_baseline()
        trace = _make_trace([1000.0, 1000.0])
        score, _ = compute_divergence_score(trace, baseline)
        assert score <= 1.0

    def test_anomaly_messages_contain_step_index(self):
        baseline = self._build_baseline()
        trace = _make_trace([50.0])
        _, anomalies = compute_divergence_score(trace, baseline)
        assert any("Step 0" in a for a in anomalies)


# ---------------------------------------------------------------------------
# Save / Load roundtrip
# ---------------------------------------------------------------------------


class TestSaveLoadRoundtrip:
    def test_roundtrip(self, tmp_path):
        traces = [
            _make_trace([2.0, 2.1, 2.2], margin_values=[0.5, 0.6, 0.4], surprisal_values=[3.0, 3.1, 3.2]),
            _make_trace([2.1, 2.0, 2.3], margin_values=[0.5, 0.5, 0.5], surprisal_values=[3.1, 3.0, 3.3]),
            _make_trace(
                [2.2, 2.1, 2.0],
                margin_values=[0.6, 0.4, 0.5],
                layers=[[_make_layer(0, l2_norm_mean=100.0, entropy_mean=2.5)]],
            ),
        ]
        original = calibrate_from_runs("roundtrip-model", traces)

        path = tmp_path / "profile.json"
        original.save(path)

        loaded = CalibrationProfile.load(path)

        assert loaded.model_id == original.model_id
        assert loaded.num_runs == original.num_runs
        assert loaded.entropy_per_step.values == original.entropy_per_step.values
        assert loaded.margin_per_step.values == original.margin_per_step.values
        assert loaded.surprisal_per_step.values == original.surprisal_per_step.values
        assert abs(loaded.entropy_per_step.mean - original.entropy_per_step.mean) < 1e-9
        assert abs(loaded.entropy_per_step.std - original.entropy_per_step.std) < 1e-9

    def test_roundtrip_layer_data(self, tmp_path):
        layers = [[_make_layer(0, l2_norm_mean=50.0), _make_layer(1, entropy_mean=1.5)]]
        traces = [_make_trace([1.0], layers=layers)]
        original = calibrate_from_runs("layer-model", traces)

        path = tmp_path / "layers.json"
        original.save(path)
        loaded = CalibrationProfile.load(path)

        assert 0 in loaded.l2_norm_per_layer
        assert loaded.l2_norm_per_layer[0].values == [50.0]
        assert 1 in loaded.attention_entropy_per_layer
        assert loaded.attention_entropy_per_layer[1].values == [1.5]

    def test_roundtrip_preserves_z_scores(self, tmp_path):
        traces = [
            _make_trace([2.0, 2.0, 2.0]),
            _make_trace([2.0, 2.0, 2.0]),
            _make_trace([2.1, 1.9, 2.0]),
        ]
        original = calibrate_from_runs("z-model", traces)
        test_value = 5.0
        original_z = original.entropy_per_step.z_score(test_value)

        path = tmp_path / "z.json"
        original.save(path)
        loaded = CalibrationProfile.load(path)
        loaded_z = loaded.entropy_per_step.z_score(test_value)

        assert abs(original_z - loaded_z) < 1e-9

    def test_saved_file_is_valid_json(self, tmp_path):
        profile = calibrate_from_runs("json-model", [_make_trace([1.0])])
        path = tmp_path / "valid.json"
        profile.save(path)

        data = json.loads(path.read_text())
        assert "model_id" in data
        assert data["model_id"] == "json-model"
        assert "entropy" in data

    def test_load_empty_profile(self, tmp_path):
        profile = calibrate_from_runs("empty", [])
        path = tmp_path / "empty.json"
        profile.save(path)
        loaded = CalibrationProfile.load(path)
        assert loaded.num_runs == 0
        assert loaded.entropy_per_step.values == []


# ---------------------------------------------------------------------------
# Integration: calibrate → divergence score
# ---------------------------------------------------------------------------


class TestCalibrationIntegration:
    def test_calibrate_then_score(self):
        """End-to-end: calibrate from 3 traces, score a normal and anomalous trace."""
        baseline_traces = [
            _make_trace([2.0, 2.1, 2.0], margin_values=[0.5, 0.5, 0.5]),
            _make_trace([2.1, 2.0, 2.1], margin_values=[0.5, 0.5, 0.5]),
            _make_trace([2.0, 2.0, 2.0], margin_values=[0.5, 0.5, 0.5]),
        ]
        profile = calibrate_from_runs("integration-model", baseline_traces)

        normal_trace = _make_trace([2.0, 2.1], margin_values=[0.5, 0.5])
        score_normal, anomalies_normal = compute_divergence_score(normal_trace, profile)

        bad_trace = _make_trace([9.0, 9.0], margin_values=[0.01, 0.01])
        score_bad, anomalies_bad = compute_divergence_score(bad_trace, profile)

        assert score_bad > score_normal
        assert len(anomalies_bad) > len(anomalies_normal)

    def test_calibrate_then_score_roundtrip(self, tmp_path):
        """Calibrate → save → load → score."""
        traces = [
            _make_trace([2.0, 2.0], margin_values=[0.5, 0.5]),
            _make_trace([2.0, 2.0], margin_values=[0.5, 0.5]),
        ]
        profile = calibrate_from_runs("rt", traces)
        path = tmp_path / "rt.json"
        profile.save(path)
        loaded = CalibrationProfile.load(path)

        test_trace = _make_trace([2.0])
        s1, _ = compute_divergence_score(test_trace, profile)
        s2, _ = compute_divergence_score(test_trace, loaded)
        assert abs(s1 - s2) < 1e-9
