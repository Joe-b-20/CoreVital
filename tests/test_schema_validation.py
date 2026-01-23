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
#   2026-01-23: Phase-0.5 - Added tests for extensions fields and encoder_layers
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
    TimelineStep,
    LayerSummary,
    TokenInfo,
    HiddenSummary,
    AttentionSummary,
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


# ============================================================================
# Phase-0.5 Tests: Extensions and encoder_layers
# ============================================================================

def test_report_extensions_field():
    """Test that extensions field is writable at Report level."""
    report = create_minimal_valid_report()
    
    # Set extensions with various data types
    report.extensions = {
        "custom_metric": 42.5,
        "custom_flag": True,
        "custom_string": "test",
        "custom_list": [1, 2, 3],
        "custom_dict": {"nested": "value"},
    }
    
    assert validate_report(report) is True
    assert report.extensions["custom_metric"] == 42.5
    assert report.extensions["custom_flag"] is True
    assert report.extensions["custom_string"] == "test"
    assert report.extensions["custom_list"] == [1, 2, 3]
    assert report.extensions["custom_dict"]["nested"] == "value"


def test_report_extensions_default_empty():
    """Test that extensions field defaults to empty dict."""
    report = create_minimal_valid_report()
    
    assert validate_report(report) is True
    assert report.extensions == {}
    assert isinstance(report.extensions, dict)


def test_timeline_step_extensions_field():
    """Test that extensions field is writable at TimelineStep level."""
    report = create_minimal_valid_report()
    
    # Create a timeline step with extensions
    timeline_step = TimelineStep(
        step_index=0,
        token=TokenInfo(
            token_id=123,
            token_text="hello",
            is_prompt_token=False,
        ),
        extensions={
            "step_metric": 1.23,
            "step_flag": False,
        },
    )
    
    report.timeline = [timeline_step]
    
    assert validate_report(report) is True
    assert len(report.timeline) == 1
    assert report.timeline[0].extensions["step_metric"] == 1.23
    assert report.timeline[0].extensions["step_flag"] is False


def test_layer_summary_extensions_field():
    """Test that extensions field is writable at LayerSummary level."""
    report = create_minimal_valid_report()
    
    # Create a timeline step with layer summaries that have extensions
    layer_summary = LayerSummary(
        layer_index=0,
        hidden_summary=HiddenSummary(),
        attention_summary=AttentionSummary(),
        extensions={
            "layer_metric": 4.56,
            "layer_info": "test_layer",
        },
    )
    
    timeline_step = TimelineStep(
        step_index=0,
        token=TokenInfo(
            token_id=123,
            token_text="hello",
            is_prompt_token=False,
        ),
        layers=[layer_summary],
    )
    
    report.timeline = [timeline_step]
    
    assert validate_report(report) is True
    assert len(report.timeline) == 1
    assert len(report.timeline[0].layers) == 1
    assert report.timeline[0].layers[0].extensions["layer_metric"] == 4.56
    assert report.timeline[0].layers[0].extensions["layer_info"] == "test_layer"


def test_extensions_serialization_to_json():
    """Test that extensions fields serialize correctly to JSON."""
    report = create_minimal_valid_report()
    
    # Set extensions at all levels
    report.extensions = {"report_level": "value"}
    
    layer_summary = LayerSummary(
        layer_index=0,
        hidden_summary=HiddenSummary(),
        attention_summary=AttentionSummary(),
        extensions={"layer_level": "value"},
    )
    
    timeline_step = TimelineStep(
        step_index=0,
        token=TokenInfo(
            token_id=123,
            token_text="hello",
            is_prompt_token=False,
        ),
        layers=[layer_summary],
        extensions={"step_level": "value"},
    )
    
    report.timeline = [timeline_step]
    
    # Serialize to dict (simulating JSON serialization)
    report_dict = report.model_dump(mode='json')
    
    # Verify extensions are in the serialized output
    assert "extensions" in report_dict
    assert report_dict["extensions"]["report_level"] == "value"
    
    assert "timeline" in report_dict
    assert len(report_dict["timeline"]) == 1
    assert "extensions" in report_dict["timeline"][0]
    assert report_dict["timeline"][0]["extensions"]["step_level"] == "value"
    
    assert "layers" in report_dict["timeline"][0]
    assert len(report_dict["timeline"][0]["layers"]) == 1
    assert "extensions" in report_dict["timeline"][0]["layers"][0]
    assert report_dict["timeline"][0]["layers"][0]["extensions"]["layer_level"] == "value"


def test_backward_compatibility_no_extensions():
    """Test backward compatibility: reports without extensions still work."""
    report = create_minimal_valid_report()
    
    # Don't set extensions - should default to empty dict
    assert report.extensions == {}
    
    # Create timeline step without extensions
    timeline_step = TimelineStep(
        step_index=0,
        token=TokenInfo(
            token_id=123,
            token_text="hello",
            is_prompt_token=False,
        ),
    )
    
    report.timeline = [timeline_step]
    
    # Should validate successfully
    assert validate_report(report) is True
    
    # Extensions should default to empty dict
    assert report.extensions == {}
    assert report.timeline[0].extensions == {}
    assert len(report.timeline[0].layers) == 0  # No layers, so no layer extensions to check


def test_encoder_layers_null_for_causal():
    """Test that encoder_layers is null for CausalLM models."""
    report = create_minimal_valid_report()
    
    # CausalLM models should have encoder_layers as None
    assert report.encoder_layers is None
    
    # Should validate successfully
    assert validate_report(report) is True
    
    # Serialize and verify
    report_dict = report.model_dump(mode='json')
    assert "encoder_layers" in report_dict
    assert report_dict["encoder_layers"] is None


def test_encoder_layers_populated_for_seq2seq():
    """Test that encoder_layers is populated for Seq2Seq models."""
    report = create_minimal_valid_report()
    
    # Create encoder layers (for Seq2Seq models)
    encoder_layer_1 = LayerSummary(
        layer_index=0,
        hidden_summary=HiddenSummary(mean=0.1, std=0.2),
        attention_summary=AttentionSummary(entropy_mean=2.5),
    )
    
    encoder_layer_2 = LayerSummary(
        layer_index=1,
        hidden_summary=HiddenSummary(mean=0.2, std=0.3),
        attention_summary=AttentionSummary(entropy_mean=2.6),
    )
    
    report.encoder_layers = [encoder_layer_1, encoder_layer_2]
    
    # Should validate successfully
    assert validate_report(report) is True
    
    # Verify encoder_layers is populated
    assert report.encoder_layers is not None
    assert len(report.encoder_layers) == 2
    assert report.encoder_layers[0].layer_index == 0
    assert report.encoder_layers[1].layer_index == 1


def test_encoder_layers_and_encoder_hidden_states_both_work():
    """Test that both encoder_layers (new) and encoder_hidden_states (deprecated) work together."""
    report = create_minimal_valid_report()
    
    # Set both encoder_layers (new Phase-0.5 field) and encoder_hidden_states (deprecated)
    encoder_layer = LayerSummary(
        layer_index=0,
        hidden_summary=HiddenSummary(mean=0.1, std=0.2),
        attention_summary=AttentionSummary(entropy_mean=2.5),
    )
    
    report.encoder_layers = [encoder_layer]
    report.encoder_hidden_states = [HiddenSummary(mean=0.1, std=0.2)]
    
    # Should validate successfully with both fields
    assert validate_report(report) is True
    
    # Both fields should be present
    assert report.encoder_layers is not None
    assert len(report.encoder_layers) == 1
    assert report.encoder_hidden_states is not None
    assert len(report.encoder_hidden_states) == 1
    
    # Serialize and verify both are in JSON
    report_dict = report.model_dump(mode='json')
    assert "encoder_layers" in report_dict
    assert "encoder_hidden_states" in report_dict
    assert len(report_dict["encoder_layers"]) == 1
    assert len(report_dict["encoder_hidden_states"]) == 1


def test_extensions_various_data_types():
    """Test that extensions can contain various data types."""
    report = create_minimal_valid_report()
    
    # Test various data types in extensions
    report.extensions = {
        "string": "test",
        "int": 42,
        "float": 3.14,
        "bool": True,
        "list": [1, 2, 3],
        "dict": {"nested": "value"},
        "null": None,
    }
    
    assert validate_report(report) is True
    
    # Verify all types are preserved
    assert isinstance(report.extensions["string"], str)
    assert isinstance(report.extensions["int"], int)
    assert isinstance(report.extensions["float"], float)
    assert isinstance(report.extensions["bool"], bool)
    assert isinstance(report.extensions["list"], list)
    assert isinstance(report.extensions["dict"], dict)
    assert report.extensions["null"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])