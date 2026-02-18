# ============================================================================
# CoreVital - Fingerprint Tests (Phase-3)
# ============================================================================


from CoreVital.fingerprint import (
    compute_fingerprint_vector,
    compute_prompt_hash,
    get_fingerprint,
)
from CoreVital.reporting.schema import (
    HealthFlags,
    LogitsSummary,
    Summary,
    TimelineStep,
    TokenInfo,
)


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


class TestComputeFingerprintVector:
    def _step(self, entropy: float) -> TimelineStep:
        return TimelineStep(
            step_index=0,
            token=TokenInfo(token_id=0, token_text="x", is_prompt_token=False),
            logits_summary=LogitsSummary(entropy=entropy),
            layers=[],
        )

    def test_length_nine(self):
        summary = Summary(prompt_tokens=1, generated_tokens=3, total_steps=4, elapsed_ms=100)
        timeline = [self._step(2.0), self._step(3.0), self._step(2.5)]
        vec = compute_fingerprint_vector(timeline, summary, HealthFlags(), 0.0)
        assert len(vec) == 9

    def test_mean_max_entropy(self):
        summary = Summary(prompt_tokens=1, generated_tokens=2, total_steps=3, elapsed_ms=100)
        timeline = [self._step(2.0), self._step(4.0)]
        vec = compute_fingerprint_vector(timeline, summary, HealthFlags(), 0.5)
        assert vec[0] == 3.0  # mean
        assert vec[1] == 4.0  # max
        assert vec[3] == 0.5  # risk_score

    def test_flags_as_binary(self):
        summary = Summary(prompt_tokens=1, generated_tokens=1, total_steps=2, elapsed_ms=100)
        timeline = [self._step(1.0)]
        flags = HealthFlags(nan_detected=True, repetition_loop_detected=True)
        vec = compute_fingerprint_vector(timeline, summary, flags, 0.9)
        assert vec[4] == 1.0  # nan
        assert vec[7] == 1.0  # repetition
        assert vec[3] == 0.9  # risk_score

    def test_empty_timeline(self):
        summary = Summary(prompt_tokens=1, generated_tokens=0, total_steps=1, elapsed_ms=0)
        vec = compute_fingerprint_vector([], summary, HealthFlags(), 0.0)
        assert len(vec) == 9
        assert vec[0] == 0.0 and vec[1] == 0.0


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
