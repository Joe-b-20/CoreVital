# ============================================================================
# CoreVital - Mock Instrumentation Tests
#
# Purpose: Test instrumentation logic without loading heavy models
# Inputs: Mock model bundle fixture
# Outputs: Test pass/fail
# Dependencies: pytest, unittest.mock, CoreVital
# Usage: pytest tests/test_mock_instrumentation.py -v
#
# Changelog:
#   2026-01-16: Initial mock instrumentation tests
#   2026-01-23: Phase-0.5 - Added tests for extensions fields and encoder_layers
# ============================================================================

import json
import pytest
import torch
from pathlib import Path

from CoreVital.config import Config
from CoreVital.instrumentation.collector import InstrumentationCollector, InstrumentationResults
from CoreVital.reporting.report_builder import ReportBuilder
from CoreVital.reporting.schema import Report
from CoreVital.utils.serialization import serialize_report_to_json


class TestMockCausalLMInstrumentation:
    """Test instrumentation with CausalLM mock model."""
    
    def test_collector_runs_with_mock_causal(self, mock_model_bundle):
        """Test that InstrumentationCollector can run generation with mock CausalLM model."""
        # Create config
        config = Config()
        config.model.hf_id = "mock-causal"
        config.device.requested = "cpu"
        config.generation.max_new_tokens = 3
        config.generation.seed = 42
        config.generation.do_sample = False
        
        # Create collector and inject mock bundle
        collector = InstrumentationCollector(config)
        collector.model_bundle = mock_model_bundle
        
        # Run generation
        prompt = "Hello world"
        results = collector.run(prompt)
        
        # Verify results structure
        assert isinstance(results, InstrumentationResults)
        assert results.model_bundle == mock_model_bundle
        assert results.prompt_text == prompt
        assert len(results.prompt_token_ids) > 0
        assert len(results.generated_token_ids) > 0
        assert len(results.generated_token_ids) <= config.generation.max_new_tokens
        assert results.elapsed_ms > 0
        assert len(results.timeline) > 0
        
        # Verify timeline has data
        for step in results.timeline:
            assert step.step_index >= 0
            assert step.token_id >= 0
            assert step.token_text is not None
            assert step.is_prompt_token == False  # Only generated tokens in timeline
            # At least some steps should have logits, hidden_states, or attentions
            assert step.logits is not None or step.hidden_states is not None or step.attentions is not None
    
    def test_report_builder_with_mock_causal(self, mock_model_bundle):
        """Test that ReportBuilder produces valid JSON report with mock CausalLM results."""
        # Create config
        config = Config()
        config.model.hf_id = "mock-causal"
        config.device.requested = "cpu"
        config.generation.max_new_tokens = 3
        config.generation.seed = 42
        
        # Create collector and inject mock bundle
        collector = InstrumentationCollector(config)
        collector.model_bundle = mock_model_bundle
        
        # Run generation
        prompt = "Test prompt"
        results = collector.run(prompt)
        
        # Build report
        builder = ReportBuilder(config)
        report = builder.build(results, prompt)
        
        # Verify report structure
        assert isinstance(report, Report)
        assert report.schema_version == "0.1.0"
        assert report.trace_id is not None
        assert report.model.hf_id == "mock-causal"
        assert report.model.num_layers == mock_model_bundle.num_layers
        assert report.model.hidden_size == mock_model_bundle.hidden_size
        assert report.model.num_attention_heads == mock_model_bundle.num_attention_heads
        assert report.summary.prompt_tokens > 0
        assert report.summary.generated_tokens > 0
        assert report.summary.total_steps > 0
        # elapsed_ms can be 0 for very fast mock execution, so just check it's >= 0
        assert report.summary.elapsed_ms >= 0
        
        # Verify timeline
        assert len(report.timeline) > 0
        for step in report.timeline:
            assert step.token.token_id >= 0
            assert step.token.token_text is not None
        
        # Serialize to JSON
        json_str = serialize_report_to_json(report)
        assert json_str is not None
        assert len(json_str) > 0
        
        # Parse JSON to verify it's valid
        json_data = json.loads(json_str)
        assert json_data["schema_version"] == "0.1.0"
        assert "trace_id" in json_data
        assert "model" in json_data
        assert "timeline" in json_data
        assert "summary" in json_data
        
        # Verify summaries are present
        assert len(json_data["timeline"]) > 0
        first_step = json_data["timeline"][0]
        # At least one of these should be present
        assert "logits" in first_step or "layers" in first_step
    
    def test_mock_causal_output_shapes(self, mock_model_bundle):
        """Test that mock CausalLM model returns correctly shaped tensors."""
        config = Config()
        config.model.hf_id = "mock-causal"
        config.device.requested = "cpu"
        config.generation.max_new_tokens = 2
        config.generation.do_sample = False
        
        collector = InstrumentationCollector(config)
        collector.model_bundle = mock_model_bundle
        
        results = collector.run("Test")
        
        # Check that we have timeline steps
        assert len(results.timeline) > 0
        
        # Check tensor shapes in timeline
        for step in results.timeline:
            if step.logits is not None:
                # Logits should be (batch_size, vocab_size)
                assert step.logits.shape == (1, 50257) or len(step.logits.shape) == 2
            
            if step.hidden_states is not None:
                # Hidden states should be a list of tensors, one per layer
                assert isinstance(step.hidden_states, list)
                assert len(step.hidden_states) == mock_model_bundle.num_layers
                for hidden in step.hidden_states:
                    assert isinstance(hidden, torch.Tensor)
                    # Shape should be (batch_size, seq_len, hidden_size)
                    assert len(hidden.shape) == 3
                    assert hidden.shape[0] == 1  # batch_size
                    assert hidden.shape[2] == mock_model_bundle.hidden_size
            
            if step.attentions is not None:
                # Attentions should be a list of tensors, one per layer
                assert isinstance(step.attentions, list)
                assert len(step.attentions) == mock_model_bundle.num_layers
                for attn in step.attentions:
                    assert isinstance(attn, torch.Tensor)
                    # Shape should be (batch_size, num_heads, seq_len, seq_len)
                    assert len(attn.shape) == 4
                    assert attn.shape[0] == 1  # batch_size
                    assert attn.shape[1] == mock_model_bundle.num_attention_heads


class TestMockSeq2SeqInstrumentation:
    """Test instrumentation with Seq2Seq mock model."""
    
    @pytest.mark.parametrize('mock_model_bundle', ['seq2seq'], indirect=True)
    def test_collector_runs_with_mock_seq2seq(self, mock_model_bundle):
        """Test that InstrumentationCollector can run generation with mock Seq2Seq model."""
        # Create config
        config = Config()
        config.model.hf_id = "mock-seq2seq"
        config.device.requested = "cpu"
        config.generation.max_new_tokens = 3
        config.generation.seed = 42
        config.generation.do_sample = False
        
        # Create collector and inject mock bundle
        collector = InstrumentationCollector(config)
        collector.model_bundle = mock_model_bundle
        
        # Run generation
        prompt = "Translate: Hello"
        results = collector.run(prompt)
        
        # Verify results structure
        assert isinstance(results, InstrumentationResults)
        assert results.model_bundle == mock_model_bundle
        assert results.prompt_text == prompt
        assert len(results.prompt_token_ids) > 0
        assert len(results.generated_token_ids) > 0
        assert len(results.generated_token_ids) <= config.generation.max_new_tokens
        assert results.elapsed_ms > 0
        assert len(results.timeline) > 0
        
        # Seq2Seq models should have encoder outputs
        assert results.encoder_hidden_states is not None
        assert len(results.encoder_hidden_states) == mock_model_bundle.num_layers
        
        # Verify timeline has data
        for step in results.timeline:
            assert step.step_index >= 0
            assert step.token_id >= 0
            assert step.token_text is not None
            assert step.is_prompt_token == False
    
    @pytest.mark.parametrize('mock_model_bundle', ['seq2seq'], indirect=True)
    def test_report_builder_with_mock_seq2seq(self, mock_model_bundle):
        """Test that ReportBuilder produces valid JSON report with mock Seq2Seq results."""
        # Create config
        config = Config()
        config.model.hf_id = "mock-seq2seq"
        config.device.requested = "cpu"
        config.generation.max_new_tokens = 3
        config.generation.seed = 42
        
        # Create collector and inject mock bundle
        collector = InstrumentationCollector(config)
        collector.model_bundle = mock_model_bundle
        
        # Run generation
        prompt = "Translate this"
        results = collector.run(prompt)
        
        # Build report
        builder = ReportBuilder(config)
        report = builder.build(results, prompt)
        
        # Verify report structure
        assert isinstance(report, Report)
        assert report.schema_version == "0.1.0"
        assert report.trace_id is not None
        assert report.model.hf_id == "mock-seq2seq"
        assert report.summary.prompt_tokens > 0
        assert report.summary.generated_tokens > 0
        
        # Seq2Seq reports should have encoder hidden states summaries
        assert report.encoder_hidden_states is not None
        assert len(report.encoder_hidden_states) == mock_model_bundle.num_layers
        
        # Serialize to JSON
        json_str = serialize_report_to_json(report)
        assert json_str is not None
        
        # Parse JSON to verify it's valid
        json_data = json.loads(json_str)
        assert json_data["schema_version"] == "0.1.0"
        assert "encoder_hidden_states" in json_data
        assert len(json_data["encoder_hidden_states"]) == mock_model_bundle.num_layers
    
    @pytest.mark.parametrize('mock_model_bundle', ['seq2seq'], indirect=True)
    def test_mock_seq2seq_output_shapes(self, mock_model_bundle):
        """Test that mock Seq2Seq model returns correctly shaped tensors."""
        config = Config()
        config.model.hf_id = "mock-seq2seq"
        config.device.requested = "cpu"
        config.generation.max_new_tokens = 2
        config.generation.do_sample = False
        
        collector = InstrumentationCollector(config)
        collector.model_bundle = mock_model_bundle
        
        results = collector.run("Test")
        
        # Check encoder outputs
        assert results.encoder_hidden_states is not None
        assert len(results.encoder_hidden_states) == mock_model_bundle.num_layers
        for enc_hidden in results.encoder_hidden_states:
            assert isinstance(enc_hidden, torch.Tensor)
            # Encoder hidden states: (batch_size, encoder_seq_len, hidden_size)
            assert len(enc_hidden.shape) == 3
            assert enc_hidden.shape[0] == 1
            assert enc_hidden.shape[2] == mock_model_bundle.hidden_size
        
        # Check timeline steps
        assert len(results.timeline) > 0
        for step in results.timeline:
            if step.hidden_states is not None:
                # Decoder hidden states: list of (batch_size, 1, hidden_size) per layer
                assert isinstance(step.hidden_states, list)
                assert len(step.hidden_states) == mock_model_bundle.num_layers
            
            if step.cross_attentions is not None:
                # Cross-attentions: list of (batch_size, num_heads, 1, encoder_seq_len) per layer
                assert isinstance(step.cross_attentions, list)
                assert len(step.cross_attentions) == mock_model_bundle.num_layers
                for cross_attn in step.cross_attentions:
                    assert isinstance(cross_attn, torch.Tensor)
                    assert len(cross_attn.shape) == 4
                    assert cross_attn.shape[0] == 1
                    assert cross_attn.shape[1] == mock_model_bundle.num_attention_heads


class TestMockInstrumentationIntegration:
    """Integration tests for mock instrumentation."""
    
    def test_full_pipeline_causal(self, mock_model_bundle, tmp_path):
        """Test full pipeline: collector -> report builder -> JSON serialization."""
        config = Config()
        config.model.hf_id = "mock-causal"
        config.device.requested = "cpu"
        config.generation.max_new_tokens = 3
        config.generation.seed = 42
        config.sink.output_dir = str(tmp_path)
        
        # Run collector
        collector = InstrumentationCollector(config)
        collector.model_bundle = mock_model_bundle
        results = collector.run("Hello world")
        
        # Build report
        builder = ReportBuilder(config)
        report = builder.build(results, "Hello world")
        
        # Serialize to JSON
        json_str = serialize_report_to_json(report)
        
        # Write to file
        output_file = tmp_path / "test_report.json"
        output_file.write_text(json_str)
        
        # Verify file exists and is valid JSON
        assert output_file.exists()
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        # Verify required fields
        required_fields = [
            "schema_version", "trace_id", "created_at_utc",
            "model", "run_config", "prompt", "generated",
            "timeline", "summary", "warnings"
        ]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Verify summaries are valid
        assert len(data["timeline"]) > 0
        assert data["summary"]["prompt_tokens"] > 0
        assert data["summary"]["generated_tokens"] > 0
    
    @pytest.mark.parametrize('mock_model_bundle', ['seq2seq'], indirect=True)
    def test_full_pipeline_seq2seq(self, mock_model_bundle, tmp_path):
        """Test full pipeline with Seq2Seq model."""
        config = Config()
        config.model.hf_id = "mock-seq2seq"
        config.device.requested = "cpu"
        config.generation.max_new_tokens = 3
        config.generation.seed = 42
        config.sink.output_dir = str(tmp_path)
        
        # Run collector
        collector = InstrumentationCollector(config)
        collector.model_bundle = mock_model_bundle
        results = collector.run("Translate: Hello")
        
        # Build report
        builder = ReportBuilder(config)
        report = builder.build(results, "Translate: Hello")
        
        # Serialize to JSON
        json_str = serialize_report_to_json(report)
        
        # Write to file
        output_file = tmp_path / "test_report.json"
        output_file.write_text(json_str)
        
        # Verify file exists and is valid JSON
        assert output_file.exists()
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        # Verify Seq2Seq-specific fields
        assert "encoder_hidden_states" in data
        assert len(data["encoder_hidden_states"]) == mock_model_bundle.num_layers


# ============================================================================
# Phase-0.5 Tests: Extensions and encoder_layers
# ============================================================================

class TestPhase05Features:
    """Test Phase-0.5 features: extensions fields and encoder_layers."""
    
    def test_extensions_in_causal_report(self, mock_model_bundle):
        """Test that extensions field is present in generated CausalLM reports."""
        config = Config()
        config.model.hf_id = "mock-causal"
        config.device.requested = "cpu"
        config.generation.max_new_tokens = 2
        config.generation.seed = 42
        
        collector = InstrumentationCollector(config)
        collector.model_bundle = mock_model_bundle
        results = collector.run("Test")
        
        builder = ReportBuilder(config)
        report = builder.build(results, "Test")
        
        # Verify extensions field exists and defaults to empty dict
        assert hasattr(report, "extensions")
        assert isinstance(report.extensions, dict)
        
        # Verify timeline steps have extensions
        if len(report.timeline) > 0:
            assert hasattr(report.timeline[0], "extensions")
            assert isinstance(report.timeline[0].extensions, dict)
            
            # Verify layer summaries have extensions
            if len(report.timeline[0].layers) > 0:
                assert hasattr(report.timeline[0].layers[0], "extensions")
                assert isinstance(report.timeline[0].layers[0].extensions, dict)
    
    @pytest.mark.parametrize('mock_model_bundle', ['seq2seq'], indirect=True)
    def test_extensions_in_seq2seq_report(self, mock_model_bundle):
        """Test that extensions field is present in generated Seq2Seq reports."""
        config = Config()
        config.model.hf_id = "mock-seq2seq"
        config.device.requested = "cpu"
        config.generation.max_new_tokens = 2
        config.generation.seed = 42
        
        collector = InstrumentationCollector(config)
        collector.model_bundle = mock_model_bundle
        results = collector.run("Translate: Test")
        
        builder = ReportBuilder(config)
        report = builder.build(results, "Translate: Test")
        
        # Verify extensions field exists
        assert hasattr(report, "extensions")
        assert isinstance(report.extensions, dict)
        
        # Verify timeline steps have extensions
        if len(report.timeline) > 0:
            assert hasattr(report.timeline[0], "extensions")
            assert isinstance(report.timeline[0].extensions, dict)
    
    def test_encoder_layers_null_for_causal(self, mock_model_bundle):
        """Test that encoder_layers is null for CausalLM models."""
        config = Config()
        config.model.hf_id = "mock-causal"
        config.device.requested = "cpu"
        config.generation.max_new_tokens = 2
        config.generation.seed = 42
        
        collector = InstrumentationCollector(config)
        collector.model_bundle = mock_model_bundle
        results = collector.run("Test")
        
        builder = ReportBuilder(config)
        report = builder.build(results, "Test")
        
        # CausalLM models should have encoder_layers as None
        assert report.encoder_layers is None
        
        # Serialize to JSON and verify
        json_str = serialize_report_to_json(report)
        json_data = json.loads(json_str)
        
        assert "encoder_layers" in json_data
        assert json_data["encoder_layers"] is None
    
    @pytest.mark.parametrize('mock_model_bundle', ['seq2seq'], indirect=True)
    def test_encoder_layers_populated_for_seq2seq(self, mock_model_bundle):
        """Test that encoder_layers is populated for Seq2Seq models."""
        config = Config()
        config.model.hf_id = "mock-seq2seq"
        config.device.requested = "cpu"
        config.generation.max_new_tokens = 2
        config.generation.seed = 42
        
        collector = InstrumentationCollector(config)
        collector.model_bundle = mock_model_bundle
        results = collector.run("Translate: Test")
        
        builder = ReportBuilder(config)
        report = builder.build(results, "Translate: Test")
        
        # Seq2Seq models should have encoder_layers populated
        assert report.encoder_layers is not None
        assert isinstance(report.encoder_layers, list)
        assert len(report.encoder_layers) == mock_model_bundle.num_layers
        
        # Verify encoder layer structure
        for i, encoder_layer in enumerate(report.encoder_layers):
            assert encoder_layer.layer_index == i
            assert hasattr(encoder_layer, "hidden_summary")
            assert hasattr(encoder_layer, "attention_summary")
            assert hasattr(encoder_layer, "extensions")
            assert isinstance(encoder_layer.extensions, dict)
    
    @pytest.mark.parametrize('mock_model_bundle', ['seq2seq'], indirect=True)
    def test_both_encoder_layers_and_encoder_hidden_states(self, mock_model_bundle):
        """Test that both encoder_layers (new) and encoder_hidden_states (deprecated) work for Seq2Seq."""
        config = Config()
        config.model.hf_id = "mock-seq2seq"
        config.device.requested = "cpu"
        config.generation.max_new_tokens = 2
        config.generation.seed = 42
        
        collector = InstrumentationCollector(config)
        collector.model_bundle = mock_model_bundle
        results = collector.run("Translate: Test")
        
        builder = ReportBuilder(config)
        report = builder.build(results, "Translate: Test")
        
        # Both fields should be present for Seq2Seq
        assert report.encoder_layers is not None
        assert report.encoder_hidden_states is not None
        
        # Verify both are populated
        assert len(report.encoder_layers) == mock_model_bundle.num_layers
        assert len(report.encoder_hidden_states) == mock_model_bundle.num_layers
        
        # Serialize to JSON and verify both are present
        json_str = serialize_report_to_json(report)
        json_data = json.loads(json_str)
        
        assert "encoder_layers" in json_data
        assert "encoder_hidden_states" in json_data
        assert len(json_data["encoder_layers"]) == mock_model_bundle.num_layers
        assert len(json_data["encoder_hidden_states"]) == mock_model_bundle.num_layers
    
    def test_extensions_serialization_in_json(self, mock_model_bundle):
        """Test that extensions fields serialize correctly in JSON output."""
        config = Config()
        config.model.hf_id = "mock-causal"
        config.device.requested = "cpu"
        config.generation.max_new_tokens = 2
        config.generation.seed = 42
        
        collector = InstrumentationCollector(config)
        collector.model_bundle = mock_model_bundle
        results = collector.run("Test")
        
        builder = ReportBuilder(config)
        report = builder.build(results, "Test")
        
        # Serialize to JSON
        json_str = serialize_report_to_json(report)
        json_data = json.loads(json_str)
        
        # Verify extensions field is in JSON at report level
        assert "extensions" in json_data
        assert isinstance(json_data["extensions"], dict)
        
        # Verify extensions in timeline steps
        if len(json_data["timeline"]) > 0:
            assert "extensions" in json_data["timeline"][0]
            assert isinstance(json_data["timeline"][0]["extensions"], dict)
            
            # Verify extensions in layer summaries
            if "layers" in json_data["timeline"][0] and len(json_data["timeline"][0]["layers"]) > 0:
                assert "extensions" in json_data["timeline"][0]["layers"][0]
                assert isinstance(json_data["timeline"][0]["layers"][0]["extensions"], dict)
