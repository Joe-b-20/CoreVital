# ============================================================================
# CoreVital - Persistence Test
#
# Purpose: Test report persistence and validation
# Inputs: None (creates test Report objects)
# Outputs: Test pass/fail
# Dependencies: pytest, CoreVital, tempfile
# Usage: pytest tests/test_persistence.py -v
#
# Changelog:
#   2026-01-16: Initial persistence test for Phase-0.5 hardening
# ============================================================================

import json

import pytest

from CoreVital.reporting.schema import (
    AttentionConfig,
    AttentionSummary,
    GeneratedInfo,
    GenerationConfig,
    HiddenConfig,
    HiddenSummary,
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
from CoreVital.utils.serialization import serialize_report_to_json


def create_minimal_report() -> Report:
    """Create a minimal valid report for testing."""
    return Report(
        schema_version="0.2.0",
        trace_id="test-trace-persistence-123",
        created_at_utc="2026-01-16T10:00:00Z",
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
            max_new_tokens=3,
            generation=GenerationConfig(
                do_sample=True,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
            ),
            summaries=SummariesConfig(
                hidden=HiddenConfig(
                    enabled=True,
                    stats=["mean", "std"],
                    sketch=SketchConfig(method="randproj", dim=32, seed=0),
                ),
                attention=AttentionConfig(
                    enabled=True,
                    stats=["entropy_mean"],
                ),
                logits=LogitsConfig(
                    enabled=True,
                    stats=["entropy"],
                    topk=5,
                ),
            ),
            sink=SinkConfig(
                type="local_file",
                target="runs/test.json",
            ),
        ),
        prompt=PromptInfo(
            text="Hello",
            token_ids=[15496],
            num_tokens=1,
        ),
        generated=GeneratedInfo(
            output_text=" world",
            token_ids=[995],
            num_tokens=3,
        ),
        timeline=[
            TimelineStep(
                step_index=1,
                token=TokenInfo(
                    token_id=995,
                    token_text=" world",
                    is_prompt_token=False,
                ),
                logits_summary=LogitsSummary(),
                layers=[
                    LayerSummary(
                        layer_index=0,
                        hidden_summary=HiddenSummary(),
                        attention_summary=AttentionSummary(),
                    )
                ],
                extensions={},
            )
        ],
        summary=Summary(
            prompt_tokens=1,
            generated_tokens=3,
            total_steps=4,
            elapsed_ms=100,
        ),
        warnings=[],
        extensions={},
    )


def test_report_persistence(tmp_path):
    """
    Test that reports can be saved and loaded correctly.

    This test verifies:
    1. Report can be serialized to JSON
    2. JSON can be saved to a temporary file
    3. JSON can be loaded and validated using Report.model_validate_json()
    4. extensions field exists on Report and every TimelineStep
    """
    # Create a minimal report
    report = create_minimal_report()

    # Verify extensions field exists on Report
    assert hasattr(report, "extensions"), "Report should have extensions field"
    assert isinstance(report.extensions, dict), "Report.extensions should be a dict"

    # Verify extensions field exists on TimelineStep
    assert len(report.timeline) > 0, "Report should have at least one timeline step"
    for step in report.timeline:
        assert hasattr(step, "extensions"), "TimelineStep should have extensions field"
        assert isinstance(step.extensions, dict), "TimelineStep.extensions should be a dict"

    # Serialize to JSON
    json_str = serialize_report_to_json(report)
    assert isinstance(json_str, str), "Serialization should return a string"

    # Save to temporary file
    temp_file = tmp_path / "test_report.json"
    with open(temp_file, "w") as f:
        f.write(json_str)

    assert temp_file.exists(), "Temporary file should exist"

    # Load JSON from file
    with open(temp_file, "r") as f:
        loaded_json_str = f.read()

    # Validate and load using Report.model_validate_json()
    loaded_report = Report.model_validate_json(loaded_json_str)

    # Verify loaded report structure
    assert loaded_report.schema_version == report.schema_version
    assert loaded_report.trace_id == report.trace_id
    assert loaded_report.model.hf_id == report.model.hf_id

    # Verify extensions field exists on loaded Report
    assert hasattr(loaded_report, "extensions"), "Loaded Report should have extensions field"
    assert isinstance(loaded_report.extensions, dict), "Loaded Report.extensions should be a dict"

    # Verify extensions field exists on loaded TimelineStep
    assert len(loaded_report.timeline) == len(report.timeline), (
        "Loaded report should have same number of timeline steps"
    )
    for step in loaded_report.timeline:
        assert hasattr(step, "extensions"), "Loaded TimelineStep should have extensions field"
        assert isinstance(step.extensions, dict), "Loaded TimelineStep.extensions should be a dict"

    # Verify the report can be serialized again (round-trip test)
    json_str2 = serialize_report_to_json(loaded_report)
    loaded_report2 = Report.model_validate_json(json_str2)

    assert loaded_report2.schema_version == report.schema_version
    assert loaded_report2.trace_id == report.trace_id


def test_report_persistence_with_gpt2_cpu(tmp_path):
    """
    Test persistence with a real GPT-2 CPU run.

    This test generates a real report using GPT-2 and verifies persistence.
    """
    pytest.importorskip("transformers")

    from CoreVital.config import Config
    from CoreVital.instrumentation.collector import InstrumentationCollector
    from CoreVital.reporting.report_builder import ReportBuilder

    # Setup config
    config = Config()
    config.model.hf_id = "gpt2"
    config.device.requested = "cpu"
    config.generation.max_new_tokens = 3
    config.generation.seed = 42
    config.sink.output_dir = str(tmp_path)

    # Run instrumentation
    collector = InstrumentationCollector(config)
    results = collector.run("Hi")

    # Build report
    builder = ReportBuilder(config)
    report = builder.build(results, "Hi")

    # Verify extensions field exists on Report
    assert hasattr(report, "extensions"), "Report should have extensions field"
    assert isinstance(report.extensions, dict), "Report.extensions should be a dict"

    # Verify extensions field exists on TimelineStep
    for step in report.timeline:
        assert hasattr(step, "extensions"), "TimelineStep should have extensions field"
        assert isinstance(step.extensions, dict), "TimelineStep.extensions should be a dict"

    # Serialize to JSON
    json_str = serialize_report_to_json(report)

    # Save to temporary file
    temp_file = tmp_path / "gpt2_report.json"
    with open(temp_file, "w") as f:
        f.write(json_str)

    # Load and validate
    with open(temp_file, "r") as f:
        loaded_json_str = f.read()

    loaded_report = Report.model_validate_json(loaded_json_str)

    # Verify loaded report
    assert loaded_report.schema_version == report.schema_version
    assert loaded_report.trace_id == report.trace_id
    assert hasattr(loaded_report, "extensions")
    assert isinstance(loaded_report.extensions, dict)

    # Verify all timeline steps have extensions
    for step in loaded_report.timeline:
        assert hasattr(step, "extensions")
        assert isinstance(step.extensions, dict)


def test_report_validation_fails_on_invalid_schema():
    """
    Test that validation fails if the JSON doesn't match the schema.
    """
    # Create invalid JSON (missing required field)
    invalid_json = json.dumps(
        {
            "schema_version": "0.1.0",
            "trace_id": "test",
            # Missing created_at_utc and other required fields
        }
    )

    # Should raise ValidationError
    with pytest.raises(ValueError):  # Pydantic ValidationError inherits from ValueError
        Report.model_validate_json(invalid_json)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
