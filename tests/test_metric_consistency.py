"""Tests for metric consistency validation (Issue 15, Phase 5)."""

import logging

from CoreVital.reporting.schema import (
    AttentionConfig,
    AttentionSummary,
    GeneratedInfo,
    GenerationConfig,
    HiddenConfig,
    LayerSummary,
    LogitsConfig,
    LogitsSummary,
    ModelInfo,
    PromptInfo,
    QuantizationInfo,
    Report,
    RunConfig,
    SinkConfig,
    SketchConfig,
    SummariesConfig,
    Summary,
    TimelineStep,
    TokenInfo,
)
from CoreVital.reporting.validation import (
    CONCENTRATION_ENTROPY_MAX_NATS,
    CONCENTRATION_ENTROPY_THRESHOLD,
    PERPLEXITY_REL_TOL,
    validate_metric_consistency,
    validate_metric_consistency_and_log,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_report() -> Report:
    return Report(
        schema_version="0.4.0",
        trace_id="test-trace-consistency",
        created_at_utc="2026-02-28T00:00:00Z",
        model=ModelInfo(
            hf_id="gpt2",
            revision=None,
            architecture="GPT2LMHeadModel",
            num_layers=12,
            hidden_size=768,
            num_attention_heads=12,
            tokenizer_hf_id="gpt2",
            dtype="float32",
            device="cpu",
            quantization=QuantizationInfo(enabled=False),
        ),
        run_config=RunConfig(
            seed=42,
            device_requested="cpu",
            max_new_tokens=5,
            generation=GenerationConfig(
                do_sample=True,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
            ),
            summaries=SummariesConfig(
                hidden=HiddenConfig(
                    enabled=True,
                    stats=["mean"],
                    sketch=SketchConfig(method="randproj", dim=32, seed=0),
                ),
                attention=AttentionConfig(enabled=True, stats=["entropy_mean"]),
                logits=LogitsConfig(enabled=True, stats=["entropy"], topk=5),
            ),
            sink=SinkConfig(type="local_file", target="runs/test.json"),
        ),
        prompt=PromptInfo(text="Hello", token_ids=[15496], num_tokens=1),
        generated=GeneratedInfo(output_text=" world", token_ids=[995], num_tokens=1),
        timeline=[],
        summary=Summary(prompt_tokens=1, generated_tokens=1, total_steps=2, elapsed_ms=100),
    )


def _step(
    idx: int,
    entropy: float = None,
    perplexity: float = None,
    top_k_margin: float = None,
    voter_agreement: float = None,
    surprisal: float = None,
    layers=None,
) -> TimelineStep:
    logits = LogitsSummary(
        entropy=entropy,
        perplexity=perplexity,
        top_k_margin=top_k_margin,
        voter_agreement=voter_agreement,
        surprisal=surprisal,
    )
    return TimelineStep(
        step_index=idx,
        token=TokenInfo(token_id=0, token_text="x", is_prompt_token=False),
        logits_summary=logits,
        layers=layers or [],
    )


# ============================================================================
# Test: Perplexity == 2^entropy
# ============================================================================


class TestPerplexityEntropy:
    def test_consistent_no_warning(self):
        """When perplexity == 2^entropy exactly, no warnings."""
        report = _minimal_report()
        entropy = 3.0
        report.timeline = [_step(0, entropy=entropy, perplexity=2.0**entropy)]
        assert validate_metric_consistency(report) == []

    def test_slightly_off_within_tolerance(self):
        """Small deviation within 1% tolerance is OK."""
        report = _minimal_report()
        entropy = 3.0
        expected = 2.0**entropy
        report.timeline = [_step(0, entropy=entropy, perplexity=expected * 1.005)]
        assert validate_metric_consistency(report) == []

    def test_violation_fires_warning(self):
        """2% deviation triggers a warning."""
        report = _minimal_report()
        entropy = 3.0
        expected = 2.0**entropy
        report.timeline = [_step(0, entropy=entropy, perplexity=expected * 1.05)]
        warnings = validate_metric_consistency(report)
        assert len(warnings) == 1
        assert "perplexity" in warnings[0]
        assert "2^entropy" in warnings[0]

    def test_zero_entropy_consistent(self):
        """Entropy 0 → perplexity 1."""
        report = _minimal_report()
        report.timeline = [_step(0, entropy=0.0, perplexity=1.0)]
        assert validate_metric_consistency(report) == []

    def test_large_entropy(self):
        """Large entropy value, perplexity consistent."""
        report = _minimal_report()
        entropy = 15.0
        report.timeline = [_step(0, entropy=entropy, perplexity=2.0**entropy)]
        assert validate_metric_consistency(report) == []

    def test_only_entropy_no_perplexity(self):
        """Missing perplexity skips check gracefully."""
        report = _minimal_report()
        report.timeline = [_step(0, entropy=3.0)]
        assert validate_metric_consistency(report) == []

    def test_only_perplexity_no_entropy(self):
        """Missing entropy skips check gracefully."""
        report = _minimal_report()
        report.timeline = [_step(0, perplexity=8.0)]
        assert validate_metric_consistency(report) == []

    def test_custom_tolerance(self):
        """Custom tolerance loosens the check."""
        report = _minimal_report()
        entropy = 3.0
        expected = 2.0**entropy
        report.timeline = [_step(0, entropy=entropy, perplexity=expected * 1.05)]
        assert validate_metric_consistency(report, perplexity_rel_tol=0.1) == []


# ============================================================================
# Test: Margin ≤ TopK mass
# ============================================================================


class TestMarginTopkMass:
    def test_margin_less_than_mass_ok(self):
        report = _minimal_report()
        report.timeline = [_step(0, top_k_margin=0.3, voter_agreement=0.85)]
        assert validate_metric_consistency(report) == []

    def test_margin_equal_to_mass_ok(self):
        """When top-2 prob is 0 the margin equals mass."""
        report = _minimal_report()
        report.timeline = [_step(0, top_k_margin=0.99, voter_agreement=0.99)]
        assert validate_metric_consistency(report) == []

    def test_margin_greater_than_mass_fires(self):
        report = _minimal_report()
        report.timeline = [_step(0, top_k_margin=0.9, voter_agreement=0.5)]
        warnings = validate_metric_consistency(report)
        assert len(warnings) == 1
        assert "impossible" in warnings[0]

    def test_missing_margin_skips(self):
        report = _minimal_report()
        report.timeline = [_step(0, voter_agreement=0.85)]
        assert validate_metric_consistency(report) == []

    def test_missing_mass_skips(self):
        report = _minimal_report()
        report.timeline = [_step(0, top_k_margin=0.3)]
        assert validate_metric_consistency(report) == []


# ============================================================================
# Test: Concentration vs entropy correlation
# ============================================================================


class TestConcentrationEntropy:
    def test_high_concentration_low_entropy_ok(self):
        report = _minimal_report()
        layer = LayerSummary(
            layer_index=0,
            attention_summary=AttentionSummary(concentration_max=0.98, entropy_min=0.05),
        )
        report.timeline = [_step(0, layers=[layer])]
        assert validate_metric_consistency(report) == []

    def test_high_concentration_high_entropy_fires(self):
        report = _minimal_report()
        layer = LayerSummary(
            layer_index=0,
            attention_summary=AttentionSummary(concentration_max=0.98, entropy_min=1.5),
        )
        report.timeline = [_step(0, layers=[layer])]
        warnings = validate_metric_consistency(report)
        assert len(warnings) == 1
        assert "concentration" in warnings[0]
        assert "entropy" in warnings[0]

    def test_low_concentration_high_entropy_ok(self):
        """Below threshold concentration doesn't fire."""
        report = _minimal_report()
        layer = LayerSummary(
            layer_index=0,
            attention_summary=AttentionSummary(concentration_max=0.7, entropy_min=2.0),
        )
        report.timeline = [_step(0, layers=[layer])]
        assert validate_metric_consistency(report) == []

    def test_boundary_concentration_not_fired(self):
        """Exactly at threshold does not fire (> not >=)."""
        report = _minimal_report()
        layer = LayerSummary(
            layer_index=0,
            attention_summary=AttentionSummary(
                concentration_max=CONCENTRATION_ENTROPY_THRESHOLD,
                entropy_min=CONCENTRATION_ENTROPY_MAX_NATS + 0.1,
            ),
        )
        report.timeline = [_step(0, layers=[layer])]
        assert validate_metric_consistency(report) == []

    def test_multiple_layers_multiple_warnings(self):
        report = _minimal_report()
        layers = [
            LayerSummary(layer_index=0, attention_summary=AttentionSummary(concentration_max=0.98, entropy_min=1.5)),
            LayerSummary(layer_index=1, attention_summary=AttentionSummary(concentration_max=0.97, entropy_min=1.2)),
        ]
        report.timeline = [_step(0, layers=layers)]
        warnings = validate_metric_consistency(report)
        assert len(warnings) == 2
        assert "Layer 0" in warnings[0]
        assert "Layer 1" in warnings[1]

    def test_custom_thresholds(self):
        """Custom thresholds change detection sensitivity."""
        report = _minimal_report()
        layer = LayerSummary(
            layer_index=0,
            attention_summary=AttentionSummary(concentration_max=0.80, entropy_min=0.5),
        )
        report.timeline = [_step(0, layers=[layer])]
        # Default would not fire (concentration 0.80 < 0.95)
        assert validate_metric_consistency(report) == []
        # Strict threshold fires
        warnings = validate_metric_consistency(
            report,
            concentration_entropy_threshold=0.7,
            concentration_entropy_max_nats=0.3,
        )
        assert len(warnings) == 1


# ============================================================================
# Test: Negative entropy / perplexity < 1 / negative surprisal
# ============================================================================


class TestInformationTheoreticInvariants:
    def test_negative_entropy_warning(self):
        report = _minimal_report()
        report.timeline = [_step(0, entropy=-0.5)]
        warnings = validate_metric_consistency(report)
        assert any("negative entropy" in w for w in warnings)

    def test_perplexity_below_one_warning(self):
        report = _minimal_report()
        report.timeline = [_step(0, perplexity=0.5)]
        warnings = validate_metric_consistency(report)
        assert any("perplexity" in w and "< 1.0" in w for w in warnings)

    def test_negative_surprisal_warning(self):
        report = _minimal_report()
        report.timeline = [_step(0, surprisal=-0.1)]
        warnings = validate_metric_consistency(report)
        assert any("negative surprisal" in w for w in warnings)

    def test_zero_entropy_and_perplexity_one_ok(self):
        report = _minimal_report()
        report.timeline = [_step(0, entropy=0.0, perplexity=1.0, surprisal=0.0)]
        assert validate_metric_consistency(report) == []


# ============================================================================
# Test: Edge cases
# ============================================================================


class TestEdgeCases:
    def test_empty_timeline(self):
        report = _minimal_report()
        report.timeline = []
        assert validate_metric_consistency(report) == []

    def test_empty_logits_summary(self):
        report = _minimal_report()
        step = TimelineStep(
            step_index=0,
            token=TokenInfo(token_id=0, token_text="x", is_prompt_token=False),
            logits_summary=LogitsSummary(),
        )
        report.timeline = [step]
        assert validate_metric_consistency(report) == []

    def test_no_layers(self):
        report = _minimal_report()
        report.timeline = [_step(0, entropy=3.0, perplexity=8.0)]
        assert validate_metric_consistency(report) == []

    def test_multiple_steps_independent_warnings(self):
        """Each step's violation generates its own warning."""
        report = _minimal_report()
        report.timeline = [
            _step(0, entropy=3.0, perplexity=100.0),
            _step(1, entropy=2.0, perplexity=4.0),
            _step(2, entropy=4.0, perplexity=100.0),
        ]
        warnings = validate_metric_consistency(report)
        step_indices = [w.split(":")[0] for w in warnings]
        assert "Step 0" in step_indices[0]
        assert "Step 2" in step_indices[-1]
        assert len(warnings) == 2

    def test_combined_violations_same_step(self):
        """Multiple violations on one step all reported."""
        report = _minimal_report()
        layer = LayerSummary(
            layer_index=0,
            attention_summary=AttentionSummary(concentration_max=0.99, entropy_min=2.0),
        )
        report.timeline = [
            _step(0, entropy=3.0, perplexity=100.0, top_k_margin=0.9, voter_agreement=0.5, layers=[layer])
        ]
        warnings = validate_metric_consistency(report)
        assert len(warnings) == 3  # perplexity, margin>mass, concentration


# ============================================================================
# Test: Logging wrapper
# ============================================================================


class TestLoggingWrapper:
    def test_clean_report_logs_debug(self, caplog):
        report = _minimal_report()
        report.timeline = [_step(0, entropy=3.0, perplexity=8.0)]
        with caplog.at_level(logging.DEBUG, logger="CoreVital.reporting.validation"):
            result = validate_metric_consistency_and_log(report)
        assert result == []
        assert "passed" in caplog.text.lower()

    def test_inconsistent_report_logs_warning(self, caplog):
        report = _minimal_report()
        report.timeline = [_step(0, entropy=3.0, perplexity=100.0)]
        with caplog.at_level(logging.WARNING, logger="CoreVital.reporting.validation"):
            result = validate_metric_consistency_and_log(report)
        assert len(result) == 1
        assert "consistency" in caplog.text.lower() or "perplexity" in caplog.text.lower()

    def test_returns_same_as_validate(self):
        report = _minimal_report()
        report.timeline = [_step(0, entropy=3.0, perplexity=100.0)]
        direct = validate_metric_consistency(report)
        logged = validate_metric_consistency_and_log(report)
        assert direct == logged


# ============================================================================
# Test: Constants exported
# ============================================================================


class TestConstants:
    def test_perplexity_rel_tol(self):
        assert PERPLEXITY_REL_TOL == 0.01

    def test_concentration_entropy_threshold(self):
        assert CONCENTRATION_ENTROPY_THRESHOLD == 0.95

    def test_concentration_entropy_max_nats(self):
        assert CONCENTRATION_ENTROPY_MAX_NATS == 1.0
