# ============================================================================
# CoreVital - Fingerprint Tests (Phase-4.3)
# ============================================================================


from CoreVital.fingerprint import (
    FINGERPRINT_LENGTH,
    FINGERPRINT_VERSION,
    _correlation,
    _safe_stats,
    compute_fingerprint_vector,
    compute_prompt_hash,
    get_fingerprint,
    is_legacy_fingerprint,
)
from CoreVital.reporting.schema import (
    HealthFlags,
    LogitsSummary,
    Summary,
    TimelineStep,
    TokenInfo,
)


def _step(
    entropy: float = 2.0,
    margin: float = 0.5,
    surprisal: float = 3.0,
    agreement: float = 0.8,
    idx: int = 0,
) -> TimelineStep:
    return TimelineStep(
        step_index=idx,
        token=TokenInfo(token_id=0, token_text="x", is_prompt_token=False),
        logits_summary=LogitsSummary(
            entropy=entropy,
            top_k_margin=margin,
            surprisal=surprisal,
            voter_agreement=agreement,
        ),
        layers=[],
    )


def _summary(steps: int = 10) -> Summary:
    return Summary(prompt_tokens=5, generated_tokens=steps, total_steps=steps + 5, elapsed_ms=100)


# ---- safe_stats ----


class TestSafeStats:
    def test_empty_returns_zeros(self):
        assert _safe_stats([]) == (0.0, 0.0, 0.0, 0.0, 0.0)

    def test_single_value(self):
        mean, std, mn, mx, slope = _safe_stats([5.0])
        assert mean == 5.0
        assert std == 0.0
        assert mn == 5.0
        assert mx == 5.0
        assert slope == 0.0

    def test_two_values(self):
        mean, std, mn, mx, slope = _safe_stats([1.0, 3.0])
        assert mean == 2.0
        assert mn == 1.0
        assert mx == 3.0
        assert slope > 0  # ascending

    def test_ascending_positive_slope(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        _, _, _, _, slope = _safe_stats(vals)
        assert slope == 1.0

    def test_descending_negative_slope(self):
        vals = [5.0, 4.0, 3.0, 2.0, 1.0]
        _, _, _, _, slope = _safe_stats(vals)
        assert slope == -1.0

    def test_flat_zero_slope(self):
        vals = [3.0, 3.0, 3.0, 3.0]
        _, _, _, _, slope = _safe_stats(vals)
        assert slope == 0.0

    def test_std_correct(self):
        import statistics

        vals = [1.0, 2.0, 3.0, 4.0]
        _, std, _, _, _ = _safe_stats(vals)
        assert abs(std - statistics.stdev(vals)) < 1e-10


# ---- _correlation ----


class TestCorrelation:
    def test_perfect_positive(self):
        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [10.0, 20.0, 30.0, 40.0]
        assert abs(_correlation(xs, ys) - 1.0) < 1e-10

    def test_perfect_negative(self):
        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [40.0, 30.0, 20.0, 10.0]
        assert abs(_correlation(xs, ys) - (-1.0)) < 1e-10

    def test_zero_variance_returns_zero(self):
        xs = [5.0, 5.0, 5.0]
        ys = [1.0, 2.0, 3.0]
        assert _correlation(xs, ys) == 0.0

    def test_too_few_values(self):
        assert _correlation([1.0], [2.0]) == 0.0
        assert _correlation([], []) == 0.0

    def test_unequal_lengths_uses_min(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [10.0, 20.0, 30.0]
        corr = _correlation(xs, ys)
        assert abs(corr - 1.0) < 1e-10


# ---- compute_fingerprint_vector ----


class TestComputeFingerprintVector:
    def test_length_25(self):
        timeline = [_step(entropy=e) for e in [2.0, 3.0, 2.5, 4.0]]
        vec = compute_fingerprint_vector(timeline, _summary(4), HealthFlags(), 0.0)
        assert len(vec) == FINGERPRINT_LENGTH
        assert len(vec) == 25

    def test_entropy_stats_correct(self):
        timeline = [_step(entropy=e) for e in [1.0, 2.0, 3.0, 4.0]]
        vec = compute_fingerprint_vector(timeline, _summary(4), HealthFlags(), 0.0)
        assert vec[0] == 2.5  # mean
        assert vec[2] == 1.0  # min
        assert vec[3] == 4.0  # max
        assert vec[4] == 1.0  # slope (ascending by 1 per step)

    def test_margin_stats_positions(self):
        timeline = [_step(margin=m) for m in [0.2, 0.4, 0.6, 0.8]]
        vec = compute_fingerprint_vector(timeline, _summary(4), HealthFlags(), 0.0)
        assert vec[5] == 0.5  # margin mean
        assert vec[7] > 0  # margin slope (ascending)

    def test_surprisal_stats_positions(self):
        timeline = [_step(surprisal=s) for s in [1.0, 2.0, 3.0, 4.0]]
        vec = compute_fingerprint_vector(timeline, _summary(4), HealthFlags(), 0.0)
        assert vec[8] == 2.5  # surprisal mean
        assert vec[10] == 1.0  # surprisal slope

    def test_agreement_stats_positions(self):
        timeline = [_step(agreement=a) for a in [0.6, 0.8]]
        vec = compute_fingerprint_vector(timeline, _summary(2), HealthFlags(), 0.0)
        assert vec[11] == 0.7  # agreement mean

    def test_risk_score_and_high_entropy_fraction(self):
        flags = HealthFlags(high_entropy_steps=3)
        summary = _summary(7)  # total_steps = 12
        timeline = [_step(entropy=2.0)]
        vec = compute_fingerprint_vector(timeline, summary, flags, 0.75)
        assert vec[13] == 0.75
        assert vec[14] == 3 / 12

    def test_boolean_flags(self):
        flags = HealthFlags(
            nan_detected=True,
            inf_detected=False,
            attention_collapse_detected=True,
            repetition_loop_detected=False,
            mid_layer_anomaly_detected=True,
        )
        timeline = [_step()]
        vec = compute_fingerprint_vector(timeline, _summary(1), flags, 0.0)
        assert vec[15] == 1.0  # nan
        assert vec[16] == 0.0  # inf
        assert vec[17] == 1.0  # collapse
        assert vec[18] == 0.0  # repetition
        assert vec[19] == 1.0  # mid_layer

    def test_entropy_margin_correlation(self):
        # Both rising together → positive correlation
        timeline = [
            _step(entropy=1.0, margin=0.1),
            _step(entropy=2.0, margin=0.2),
            _step(entropy=3.0, margin=0.3),
            _step(entropy=4.0, margin=0.4),
        ]
        vec = compute_fingerprint_vector(timeline, _summary(4), HealthFlags(), 0.0)
        assert vec[20] > 0.99  # strong positive correlation

    def test_entropy_cv(self):
        import statistics

        entropies = [2.0, 2.0, 4.0, 4.0]
        expected_cv = statistics.stdev(entropies) / (sum(entropies) / len(entropies))
        timeline = [_step(entropy=e) for e in entropies]
        vec = compute_fingerprint_vector(timeline, _summary(4), HealthFlags(), 0.0)
        assert abs(vec[21] - expected_cv) < 1e-10

    def test_first_last_quarter_entropy(self):
        # 8 steps: first quarter = [1,2], last quarter = [7,8]
        entropies = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        timeline = [_step(entropy=e) for e in entropies]
        vec = compute_fingerprint_vector(timeline, _summary(8), HealthFlags(), 0.0)
        assert vec[22] == 1.5  # mean of [1, 2]
        assert vec[23] == 7.5  # mean of [7, 8]

    def test_first_last_quarter_too_few_steps(self):
        timeline = [_step(entropy=5.0), _step(entropy=6.0)]
        vec = compute_fingerprint_vector(timeline, _summary(2), HealthFlags(), 0.0)
        assert vec[22] == 0.0
        assert vec[23] == 0.0

    def test_reserved_slot_zero(self):
        timeline = [_step()]
        vec = compute_fingerprint_vector(timeline, _summary(1), HealthFlags(), 0.0)
        assert vec[24] == 0.0

    def test_empty_timeline(self):
        vec = compute_fingerprint_vector([], _summary(0), HealthFlags(), 0.0)
        assert len(vec) == 25
        # All stats should be zero
        assert vec[0] == 0.0  # entropy mean
        assert vec[5] == 0.0  # margin mean
        assert vec[8] == 0.0  # surprisal mean
        assert vec[20] == 0.0  # correlation
        assert vec[22] == 0.0  # first quarter

    def test_missing_logits_summary(self):
        step = TimelineStep(
            step_index=0,
            token=TokenInfo(token_id=0, token_text="x", is_prompt_token=False),
            logits_summary=LogitsSummary(),
            layers=[],
        )
        vec = compute_fingerprint_vector([step], _summary(1), HealthFlags(), 0.0)
        assert len(vec) == 25
        assert vec[0] == 0.0  # no entropy data

    def test_layers_by_step_accepted(self):
        timeline = [_step()]
        vec = compute_fingerprint_vector(
            timeline,
            _summary(1),
            HealthFlags(),
            0.0,
            layers_by_step=[[]],
        )
        assert len(vec) == 25

    def test_all_elements_are_float(self):
        timeline = [_step(entropy=e) for e in [1.0, 2.0, 3.0, 4.0]]
        vec = compute_fingerprint_vector(timeline, _summary(4), HealthFlags(), 0.5)
        assert all(isinstance(v, float) for v in vec)


# ---- is_legacy_fingerprint ----


class TestIsLegacyFingerprint:
    def test_nine_element_is_legacy(self):
        assert is_legacy_fingerprint([0.0] * 9) is True

    def test_25_element_is_not_legacy(self):
        assert is_legacy_fingerprint([0.0] * 25) is False

    def test_other_lengths(self):
        assert is_legacy_fingerprint([]) is False
        assert is_legacy_fingerprint([0.0] * 5) is False


# ---- Version constant ----


class TestFingerprintVersion:
    def test_version_is_2(self):
        assert FINGERPRINT_VERSION == 2

    def test_length_is_25(self):
        assert FINGERPRINT_LENGTH == 25


# ---- compute_prompt_hash ----


class TestComputePromptHash:
    def test_deterministic(self):
        h1 = compute_prompt_hash("Hello", "gpt2")
        h2 = compute_prompt_hash("Hello", "gpt2")
        assert h1 == h2

    def test_different_prompt_different_hash(self):
        h1 = compute_prompt_hash("Hello", "gpt2")
        h2 = compute_prompt_hash("World", "gpt2")
        assert h1 != h2

    def test_different_model_different_hash(self):
        h1 = compute_prompt_hash("Hello", "gpt2")
        h2 = compute_prompt_hash("Hello", "llama")
        assert h1 != h2

    def test_normalized_strip_lower(self):
        h1 = compute_prompt_hash("  Hello  ", "gpt2")
        h2 = compute_prompt_hash("hello", "gpt2")
        assert h1 == h2

    def test_hex_length(self):
        h = compute_prompt_hash("x", "y")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ---- get_fingerprint ----


class TestGetFingerprint:
    def test_from_report_object(self):
        class Report:
            extensions = {"fingerprint": {"vector": [1, 2], "prompt_hash": "abc"}}

        assert get_fingerprint(Report()) == {"vector": [1, 2], "prompt_hash": "abc"}

    def test_from_dict(self):
        report = {"extensions": {"fingerprint": {"vector": [1], "prompt_hash": "x"}}}
        assert get_fingerprint(report) == {"vector": [1], "prompt_hash": "x"}

    def test_missing_returns_empty(self):
        class Report:
            extensions = {}

        assert get_fingerprint(Report()) == {}
        assert get_fingerprint({"extensions": {}}) == {}

    def test_none_extensions(self):
        class Report:
            extensions = None

        assert get_fingerprint(Report()) == {}

    def test_non_report_returns_empty(self):
        assert get_fingerprint(42) == {}


# ---- Discriminative power ----


class TestDiscriminativePower:
    """Verify that qualitatively different failure modes produce different vectors."""

    def test_uniform_confusion_vs_sudden_break(self):
        # Uniform high entropy (confusion throughout)
        uniform = [_step(entropy=6.0, margin=0.1) for _ in range(8)]
        # Low entropy then sudden spike (sudden break)
        sudden = [_step(entropy=1.0, margin=0.9)] * 4 + [_step(entropy=8.0, margin=0.05)] * 4

        vec_uniform = compute_fingerprint_vector(uniform, _summary(8), HealthFlags(), 0.5)
        vec_sudden = compute_fingerprint_vector(sudden, _summary(8), HealthFlags(), 0.5)

        # Entropy std should differ: uniform is low std, sudden is high std
        assert vec_sudden[1] > vec_uniform[1]
        # Entropy slope should differ: uniform flat, sudden positive
        assert abs(vec_sudden[4]) > abs(vec_uniform[4])
        # First-quarter vs last-quarter entropy should differ
        assert vec_sudden[23] > vec_sudden[22]  # last > first for sudden
        assert abs(vec_uniform[22] - vec_uniform[23]) < 1.0  # roughly equal for uniform

    def test_healthy_vs_degrading(self):
        healthy = [_step(entropy=2.0, margin=0.8, surprisal=1.0) for _ in range(8)]
        degrading = [_step(entropy=2.0 + i * 0.5, margin=0.8 - i * 0.1, surprisal=1.0 + i * 0.3) for i in range(8)]

        vec_h = compute_fingerprint_vector(healthy, _summary(8), HealthFlags(), 0.1)
        vec_d = compute_fingerprint_vector(degrading, _summary(8), HealthFlags(), 0.6)

        # Degrading should have positive entropy slope, negative margin slope
        assert vec_d[4] > vec_h[4]  # entropy slope
        assert vec_d[7] < vec_h[7]  # margin slope (negative for degrading)
