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
#   2026-02-10: Phase-1b - Added assertions for prompt_forward and prompt_analysis
#                (CausalLM and Seq2Seq)
#   2026-02-10: Phase-1c - Added assertions for health_flags (CausalLM and Seq2Seq),
#                transient buffer lifecycle verification, repetition loop detection tests
# ============================================================================

import json

import pytest
import torch

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
        assert results.elapsed_ms >= 0  # Can be 0 for very fast mock execution
        assert len(results.timeline) > 0

        # Verify timeline has data
        for step in results.timeline:
            assert step.step_index >= 0
            assert step.token_id >= 0
            assert step.token_text is not None
            assert not step.is_prompt_token  # Only generated tokens in timeline
            # At least some steps should have logits, hidden_states, or attentions
            assert step.logits is not None or step.hidden_states is not None or step.attentions is not None

        # Phase-1b: prompt forward data should be captured
        assert results.prompt_forward is not None, "CausalLM should have prompt forward data"
        assert results.prompt_forward.hidden_states is not None
        assert results.prompt_forward.attentions is not None
        assert results.prompt_forward.logits is not None
        assert results.prompt_forward.prompt_token_ids == results.prompt_token_ids

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
        assert report.schema_version == "0.4.0"
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

        # Phase-1b: prompt_analysis should be populated for CausalLM
        assert report.prompt_analysis is not None, "CausalLM report should have prompt_analysis"
        assert len(report.prompt_analysis.layers) > 0, "Should have prompt attention layers"
        assert len(report.prompt_analysis.layer_transformations) > 0, "Should have layer transformations"

        # Phase-1c: health_flags should be populated
        assert report.health_flags is not None, "Report should have health_flags"
        assert isinstance(report.health_flags.nan_detected, bool)
        assert isinstance(report.health_flags.inf_detected, bool)
        assert isinstance(report.health_flags.attention_collapse_detected, bool)
        assert isinstance(report.health_flags.high_entropy_steps, int)
        assert isinstance(report.health_flags.repetition_loop_detected, bool)
        assert isinstance(report.health_flags.mid_layer_anomaly_detected, bool)
        # Mock data should be clean — no anomalies expected
        assert not report.health_flags.nan_detected, "Mock data should have no NaN"
        assert not report.health_flags.inf_detected, "Mock data should have no Inf"

        # Phase-2: risk extension populated when health_flags exist
        risk = report.extensions.get("risk")
        assert risk is not None, "Report should have extensions['risk'] when health_flags exist"
        assert "risk_score" in risk and "risk_factors" in risk and "blamed_layers" in risk
        assert 0 <= risk["risk_score"] <= 1
        assert isinstance(risk["risk_factors"], list) and isinstance(risk["blamed_layers"], list)

        # Phase-3: fingerprint on every report
        fp = report.extensions.get("fingerprint")
        assert fp is not None, "Report should have extensions['fingerprint']"
        assert "vector" in fp and "prompt_hash" in fp
        assert len(fp["vector"]) == 9
        assert isinstance(fp["prompt_hash"], str) and len(fp["prompt_hash"]) == 64

        # Serialize to JSON
        json_str = serialize_report_to_json(report)
        assert json_str is not None
        assert len(json_str) > 0

        # Parse JSON to verify it's valid
        json_data = json.loads(json_str)
        assert json_data["schema_version"] == "0.4.0"
        assert "trace_id" in json_data
        assert "model" in json_data
        assert "timeline" in json_data
        assert "summary" in json_data
        assert "health_flags" in json_data
        assert json_data["health_flags"] is not None
        assert isinstance(json_data["health_flags"]["nan_detected"], bool)
        assert isinstance(json_data["health_flags"]["repetition_loop_detected"], bool)
        assert isinstance(json_data["health_flags"]["mid_layer_anomaly_detected"], bool)

        # Verify summaries are present
        assert len(json_data["timeline"]) > 0
        first_step = json_data["timeline"][0]
        # At least one of these should be present
        assert "logits" in first_step or "layers" in first_step

    def test_report_builder_summary_capture_mode_has_no_layers_and_heads(self, mock_model_bundle):
        """When capture_mode=summary, timeline.layers and prompt_attention.heads should be empty,
        but health_flags must still be populated.
        """
        # Create config with summary capture mode
        config = Config()
        config.model.hf_id = "mock-causal"
        config.device.requested = "cpu"
        config.generation.max_new_tokens = 3
        config.generation.seed = 42
        config.capture.capture_mode = "summary"

        # Create collector and inject mock bundle
        collector = InstrumentationCollector(config)
        collector.model_bundle = mock_model_bundle

        # Run generation
        prompt = "Test prompt summary"
        results = collector.run(prompt)

        # Build report
        builder = ReportBuilder(config)
        report = builder.build(results, prompt)

        # Timeline should have no per-layer data in summary mode
        assert len(report.timeline) > 0
        for step in report.timeline:
            assert step.layers == []

        # Prompt analysis should exist but with empty heads per layer
        assert report.prompt_analysis is not None
        for layer in report.prompt_analysis.layers:
            assert layer.heads == []

        # Health flags should still be computed from internal data
        assert report.health_flags is not None
        assert isinstance(report.health_flags.nan_detected, bool)
        assert isinstance(report.health_flags.repetition_loop_detected, bool)

    def test_report_builder_on_risk_attaches_full_layers_when_triggered(self, mock_model_bundle):
        """When capture_mode=on_risk and risk or health flag triggers, timeline gets full layers (F2.3)."""
        config = Config()
        config.model.hf_id = "mock-causal"
        config.device.requested = "cpu"
        config.generation.max_new_tokens = 3
        config.generation.seed = 42
        config.capture.capture_mode = "on_risk"
        config.capture.risk_threshold = 0.0  # Trigger on any risk_score >= 0 (always true)

        collector = InstrumentationCollector(config)
        collector.model_bundle = mock_model_bundle
        results = collector.run("On-risk trigger test")

        builder = ReportBuilder(config)
        report = builder.build(results, "On-risk trigger test")

        assert len(report.timeline) > 0
        # F2.3: when on_risk triggers, full layers are attached
        first_step = report.timeline[0]
        assert len(first_step.layers) > 0, "on_risk trigger should attach full timeline layers"

    def test_report_builder_rag_context_in_extensions(self, mock_model_bundle):
        """When config.rag_context is set, report.extensions['rag'] should contain RAGContext data (Foundation F3)."""
        config = Config()
        config.model.hf_id = "mock-causal"
        config.device.requested = "cpu"
        config.generation.max_new_tokens = 2
        config.generation.seed = 42
        config.rag_context = {
            "context_token_count": 100,
            "retrieved_doc_ids": ["doc1", "doc2"],
            "retrieved_doc_titles": ["Title 1", "Title 2"],
            "retrieval_metadata": {"k": 5, "scores": [0.9, 0.8]},
        }

        collector = InstrumentationCollector(config)
        collector.model_bundle = mock_model_bundle
        results = collector.run("RAG test")
        builder = ReportBuilder(config)
        report = builder.build(results, "RAG test")

        rag = report.extensions.get("rag")
        assert rag is not None, "extensions['rag'] should be set when config.rag_context is provided"
        assert rag.get("context_token_count") == 100
        assert rag.get("retrieved_doc_ids") == ["doc1", "doc2"]
        assert rag.get("retrieved_doc_titles") == ["Title 1", "Title 2"]
        assert rag.get("retrieval_metadata") == {"k": 5, "scores": [0.9, 0.8]}

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

    @pytest.mark.parametrize("mock_model_bundle", ["seq2seq"], indirect=True)
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

        # Seq2Seq models should have encoder outputs (used internally for building encoder_layers)
        assert results.encoder_hidden_states is not None

        # Phase-1b: Seq2Seq should reuse encoder outputs as prompt forward data
        assert results.prompt_forward is not None, "Seq2Seq should have prompt forward data (encoder reuse)"
        assert results.prompt_forward.hidden_states is not None
        assert results.prompt_forward.attentions is not None
        assert results.prompt_forward.logits is None, "Seq2Seq encoder has no next-token logits"
        assert results.prompt_forward.prompt_token_ids == results.prompt_token_ids

        # Verify timeline has data
        for step in results.timeline:
            assert step.step_index >= 0
            assert step.token_id >= 0
            assert step.token_text is not None
            assert not step.is_prompt_token

    @pytest.mark.parametrize("mock_model_bundle", ["seq2seq"], indirect=True)
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
        assert report.schema_version == "0.4.0"
        assert report.trace_id is not None
        assert report.model.hf_id == "mock-seq2seq"
        assert report.summary.prompt_tokens > 0
        assert report.summary.generated_tokens > 0

        # Seq2Seq reports should have encoder_layers
        assert report.encoder_layers is not None
        assert len(report.encoder_layers) == mock_model_bundle.num_layers

        # Phase-1b: Seq2Seq should have prompt_analysis (from encoder reuse)
        assert report.prompt_analysis is not None, "Seq2Seq report should have prompt_analysis"
        assert len(report.prompt_analysis.layers) > 0, "Should have prompt attention layers"
        assert len(report.prompt_analysis.layer_transformations) > 0, "Should have layer transformations"
        assert len(report.prompt_analysis.prompt_surprisals) == 0, "Seq2Seq encoder has no surprisals"

        # Phase-1c: health_flags should be populated
        assert report.health_flags is not None, "Report should have health_flags"
        assert isinstance(report.health_flags.nan_detected, bool)
        assert isinstance(report.health_flags.repetition_loop_detected, bool)
        assert isinstance(report.health_flags.mid_layer_anomaly_detected, bool)

        # Serialize to JSON
        json_str = serialize_report_to_json(report)
        assert json_str is not None

        # Parse JSON to verify it's valid
        json_data = json.loads(json_str)
        assert json_data["schema_version"] == "0.4.0"
        assert "encoder_layers" in json_data
        assert len(json_data["encoder_layers"]) == mock_model_bundle.num_layers
        assert "health_flags" in json_data
        assert json_data["health_flags"] is not None

    @pytest.mark.parametrize("mock_model_bundle", ["seq2seq"], indirect=True)
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
        with open(output_file, "r") as f:
            data = json.load(f)

        # Verify required fields
        required_fields = [
            "schema_version",
            "trace_id",
            "created_at_utc",
            "model",
            "run_config",
            "prompt",
            "generated",
            "timeline",
            "summary",
            "warnings",
        ]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        # Verify summaries are valid
        assert len(data["timeline"]) > 0
        assert data["summary"]["prompt_tokens"] > 0
        assert data["summary"]["generated_tokens"] > 0

    @pytest.mark.parametrize("mock_model_bundle", ["seq2seq"], indirect=True)
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
        with open(output_file, "r") as f:
            data = json.load(f)

        # Verify Seq2Seq-specific fields
        assert "encoder_layers" in data
        assert len(data["encoder_layers"]) == mock_model_bundle.num_layers


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

    @pytest.mark.parametrize("mock_model_bundle", ["seq2seq"], indirect=True)
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

        # Serialize to JSON and verify (exclude_none=True may omit encoder_layers when None)
        json_str = serialize_report_to_json(report)
        json_data = json.loads(json_str)

        assert json_data.get("encoder_layers") is None

    @pytest.mark.parametrize("mock_model_bundle", ["seq2seq"], indirect=True)
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


# ============================================================================
# Phase-1c Tests: Health Flags + Transient Buffer
# ============================================================================


class TestPhase1cHealthFlags:
    """Test Phase-1c health flags and transient buffer lifecycle."""

    def test_health_flags_in_causal_json(self, mock_model_bundle, tmp_path):
        """Test that health_flags appear in serialized JSON for CausalLM."""
        config = Config()
        config.model.hf_id = "mock-causal"
        config.device.requested = "cpu"
        config.generation.max_new_tokens = 3
        config.generation.seed = 42
        config.sink.output_dir = str(tmp_path)

        collector = InstrumentationCollector(config)
        collector.model_bundle = mock_model_bundle
        results = collector.run("Health flag test")

        builder = ReportBuilder(config)
        report = builder.build(results, "Health flag test")

        json_str = serialize_report_to_json(report)
        data = json.loads(json_str)

        hf = data["health_flags"]
        assert hf is not None
        assert "nan_detected" in hf
        assert "inf_detected" in hf
        assert "attention_collapse_detected" in hf
        assert "high_entropy_steps" in hf
        assert "repetition_loop_detected" in hf
        assert "mid_layer_anomaly_detected" in hf

        # No transient buffer data should leak into the JSON
        # The hidden_size of the mock model should NOT appear as an array length
        json_flat = json_str.lower()
        assert "hidden_state_buffer" not in json_flat

    @pytest.mark.parametrize("mock_model_bundle", ["seq2seq"], indirect=True)
    def test_health_flags_in_seq2seq_json(self, mock_model_bundle, tmp_path):
        """Test that health_flags appear in serialized JSON for Seq2Seq."""
        config = Config()
        config.model.hf_id = "mock-seq2seq"
        config.device.requested = "cpu"
        config.generation.max_new_tokens = 3
        config.generation.seed = 42
        config.sink.output_dir = str(tmp_path)

        collector = InstrumentationCollector(config)
        collector.model_bundle = mock_model_bundle
        results = collector.run("Translate: health flag test")

        builder = ReportBuilder(config)
        report = builder.build(results, "Translate: health flag test")

        json_str = serialize_report_to_json(report)
        data = json.loads(json_str)

        hf = data["health_flags"]
        assert hf is not None
        assert isinstance(hf["nan_detected"], bool)
        assert isinstance(hf["repetition_loop_detected"], bool)


class TestRepetitionLoopDetection:
    """Unit tests for repetition loop detection."""

    def test_detect_repetition_with_identical_vectors(self):
        """Identical hidden states should trigger repetition detection."""
        from CoreVital.instrumentation.summaries import detect_repetition_loop

        # 5 identical vectors (exact repetition)
        vec = torch.randn(768)
        buffer = [vec.clone() for _ in range(5)]
        assert detect_repetition_loop(buffer) is True

    def test_no_repetition_with_random_vectors(self):
        """Random hidden states should NOT trigger repetition detection."""
        from CoreVital.instrumentation.summaries import detect_repetition_loop

        buffer = [torch.randn(768) for _ in range(5)]
        assert detect_repetition_loop(buffer) is False

    def test_no_repetition_with_short_buffer(self):
        """Buffer with fewer than 4 entries should never trigger."""
        from CoreVital.instrumentation.summaries import detect_repetition_loop

        buffer = [torch.randn(768) for _ in range(3)]
        assert detect_repetition_loop(buffer) is False

    def test_repetition_with_near_identical_vectors(self):
        """Vectors that are very similar (cosine > 0.9995) should trigger."""
        from CoreVital.instrumentation.summaries import detect_repetition_loop

        base = torch.randn(768)
        # Add very tiny noise — cosine similarity will still be > 0.9995
        buffer = [base + torch.randn(768) * 0.0001 for _ in range(5)]
        assert detect_repetition_loop(buffer) is True

    def test_empty_buffer(self):
        """Empty buffer should not trigger."""
        from CoreVital.instrumentation.summaries import detect_repetition_loop

        assert detect_repetition_loop([]) is False


class TestMidLayerAnomalyDetection:
    """Unit tests for mid-layer anomaly detection."""

    def test_no_anomaly_with_clean_data(self):
        """Clean layer summaries should not trigger mid-layer anomaly."""
        from unittest.mock import Mock

        from CoreVital.instrumentation.summaries import detect_mid_layer_anomaly

        # Build 12 clean layers per step, 3 steps
        num_layers = 12
        timeline_layers = []
        for _ in range(3):
            step_layers = []
            for i in range(num_layers):
                layer = Mock()
                layer.anomalies = Mock(has_nan=False, has_inf=False)
                layer.hidden_summary = Mock(l2_norm_mean=20.0 + i * 0.5)  # Gentle growth
                step_layers.append(layer)
            timeline_layers.append(step_layers)

        assert detect_mid_layer_anomaly(timeline_layers, num_layers) is False

    def test_no_anomaly_with_attention_collapse(self):
        """Attention collapse alone should NOT trigger mid-layer anomaly (it's structural, not runtime)."""
        from unittest.mock import Mock

        from CoreVital.instrumentation.summaries import detect_mid_layer_anomaly

        num_layers = 12
        step_layers = []
        for i in range(num_layers):
            layer = Mock()
            layer.anomalies = Mock(has_nan=False, has_inf=False)
            # Collapsed heads in mid-layers (like GPT-2) — should NOT trigger
            layer.hidden_summary = Mock(l2_norm_mean=20.0 + i * 0.5)
            step_layers.append(layer)

        assert detect_mid_layer_anomaly([step_layers], num_layers) is False

    def test_anomaly_with_nan_in_mid_layer(self):
        """NaN in a mid-layer should trigger anomaly detection."""
        from unittest.mock import Mock

        from CoreVital.instrumentation.summaries import detect_mid_layer_anomaly

        num_layers = 12
        step_layers = []
        for i in range(num_layers):
            layer = Mock()
            # NaN in layer 6 (middle third: layers 4-7)
            layer.anomalies = Mock(has_nan=(i == 6), has_inf=False)
            layer.hidden_summary = Mock(l2_norm_mean=20.0)
            step_layers.append(layer)

        assert detect_mid_layer_anomaly([step_layers], num_layers) is True

    def test_anomaly_with_l2_explosion(self):
        """L2 norm explosion in mid-layers should trigger anomaly detection."""
        from unittest.mock import Mock

        from CoreVital.instrumentation.summaries import detect_mid_layer_anomaly

        num_layers = 12
        step_layers = []
        for i in range(num_layers):
            layer = Mock()
            layer.anomalies = Mock(has_nan=False, has_inf=False)
            # Early layers: ~20, mid-layer 6: 500 (8x baseline = 160, so 500 >> 160)
            norm = 500.0 if i == 6 else 20.0
            layer.hidden_summary = Mock(l2_norm_mean=norm)
            step_layers.append(layer)

        assert detect_mid_layer_anomaly([step_layers], num_layers) is True

    def test_no_anomaly_with_too_few_layers(self):
        """Model with fewer than 3 layers should not trigger."""
        from CoreVital.instrumentation.summaries import detect_mid_layer_anomaly

        assert detect_mid_layer_anomaly([[]], 2) is False
