# ============================================================================
# CoreVital - Schema Validation Test
#
# Purpose: Test report schema validation
# Inputs: None (creates test Report objects)
# Outputs: Test pass/fail
# Dependencies: pytest, CoreVital
# Usage: pytest tests/test_schema_validation.py -v
#
# Changelog:
#   2026-01-13: Initial schema validation test for Phase-0
# ============================================================================

import pytest
from CoreVital.reporting.schema import (
    Report,
    ModelInfo,
    QuantizationInfo,
    RunConfig,
    GenerationConfig,
    SketchConfig,
    HiddenConfig,
    AttentionConfig,
    LogitsConfig,
    SummariesConfig,
    SinkConfig,
    PromptInfo,
    GeneratedInfo,
    Summary,
    Warning,
)
from CoreVital.reporting.validation import validate_report
from CoreVital.errors import ValidationError


def create_minimal_valid_report() -> Report:
    """Create a minimal valid report for testing."""
    return Report(
        schema_version="0.1.0",
        trace_id="test-trace-123",
        created_at_utc="2026-01-11T15:22:08Z",
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
            num_tokens=5,
        ),
        timeline=[],
        summary=Summary(
            prompt_tokens=1,
            generated_tokens=5,
            total_steps=6,
            elapsed_ms=100,
        ),
        warnings=[],
    )


def test_valid_report():
    """Test that a valid report passes validation."""
    report = create_minimal_valid_report()
    assert validate_report(report) is True


def test_invalid_schema_version():
    """Test that invalid schema version fails validation."""
    report = create_minimal_valid_report()
    report.schema_version = "999.0.0"
    
    with pytest.raises(ValidationError) as exc_info:
        validate_report(report)
    
    assert "schema_version" in str(exc_info.value)


def test_missing_trace_id():
    """Test that missing trace_id fails validation."""
    report = create_minimal_valid_report()
    report.trace_id = ""
    
    with pytest.raises(ValidationError) as exc_info:
        validate_report(report)
    
    assert "trace_id" in str(exc_info.value)


def test_invalid_num_layers():
    """Test that invalid num_layers fails validation."""
    report = create_minimal_valid_report()
    report.model.num_layers = 0
    
    with pytest.raises(ValidationError) as exc_info:
        validate_report(report)
    
    assert "num_layers" in str(exc_info.value)


def test_prompt_token_mismatch():
    """Test that prompt token count mismatch fails validation."""
    report = create_minimal_valid_report()
    report.prompt.num_tokens = 10  # Mismatch with token_ids length
    
    with pytest.raises(ValidationError) as exc_info:
        validate_report(report)
    
    assert "token_ids" in str(exc_info.value)


def test_report_serialization():
    """Test that report can be serialized to dict."""
    report = create_minimal_valid_report()
    report_dict = report.model_dump()
    
    # Check required keys exist
    assert "schema_version" in report_dict
    assert "trace_id" in report_dict
    assert "model" in report_dict
    assert "timeline" in report_dict
    assert "summary" in report_dict


def test_report_with_warnings():
    """Test report with warnings."""
    report = create_minimal_valid_report()
    report.warnings = [
        Warning(
            code="ATTENTION_NOT_AVAILABLE",
            message="Attention weights not available",
        )
    ]
    
    assert validate_report(report) is True
    assert len(report.warnings) == 1
    assert report.warnings[0].code == "ATTENTION_NOT_AVAILABLE"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])