# ============================================================================
# CoreVital - Sink Tests
#
# Purpose: Test DatadogSink, PrometheusSink, and CLI sink routing
# Inputs: Mock Report objects
# Outputs: Test pass/fail
# Dependencies: pytest, unittest.mock, CoreVital
# Usage: pytest tests/test_sinks.py -v
#
# Changelog:
#   2026-02-11: Phase-1d — Initial sink tests
#               - DatadogSink: metric series construction, tag generation, import guard
#               - PrometheusSink: gauge creation, metric update, import guard
#               - CLI: --sink argument routing
# ============================================================================

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from CoreVital.reporting.schema import (
    AttentionConfig,
    AttentionSummary,
    GeneratedInfo,
    GenerationConfig,
    HealthFlags,
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


def _make_report(*, with_health_flags: bool = True, num_steps: int = 3) -> Report:
    """Create a minimal report for sink testing."""
    timeline = []
    for i in range(num_steps):
        timeline.append(
            TimelineStep(
                step_index=i,
                token=TokenInfo(token_id=100 + i, token_text=f"tok{i}", is_prompt_token=False),
                logits_summary=LogitsSummary(
                    entropy=2.5 + i * 0.1,
                    perplexity=5.0 + i * 0.5,
                    surprisal=1.5 + i * 0.2,
                ),
                layers=[
                    LayerSummary(
                        layer_index=0,
                        hidden_summary=HiddenSummary(l2_norm_mean=10.0),
                        attention_summary=AttentionSummary(entropy_mean=0.8),
                    ),
                ],
            )
        )

    return Report(
        schema_version="0.3.0",
        trace_id="sink-test-1234-abcd",
        created_at_utc="2026-02-11T12:00:00Z",
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
            max_new_tokens=num_steps,
            generation=GenerationConfig(do_sample=True, temperature=0.8, top_k=50, top_p=0.95),
            summaries=SummariesConfig(
                hidden=HiddenConfig(
                    enabled=True, stats=["mean"], sketch=SketchConfig(method="randproj", dim=32, seed=0)
                ),
                attention=AttentionConfig(enabled=True, stats=["entropy_mean"]),
                logits=LogitsConfig(enabled=True, stats=["entropy"], topk=5),
            ),
            sink=SinkConfig(type="local_file", target="runs"),
        ),
        prompt=PromptInfo(text="Hello", token_ids=[15496], num_tokens=1),
        generated=GeneratedInfo(output_text=" world foo", token_ids=[995, 123, 456], num_tokens=num_steps),
        timeline=timeline,
        summary=Summary(prompt_tokens=1, generated_tokens=num_steps, total_steps=num_steps + 1, elapsed_ms=200),
        health_flags=HealthFlags(
            nan_detected=False,
            inf_detected=False,
            attention_collapse_detected=True,
            high_entropy_steps=1,
            repetition_loop_detected=False,
            mid_layer_anomaly_detected=False,
        )
        if with_health_flags
        else None,
    )


# ============================================================================
# DatadogSink Tests
# ============================================================================


class TestDatadogSink:
    """Tests for DatadogSink metric construction and write logic."""

    def test_build_tags(self):
        """Tags should include model, device, trace_id, and quantized."""
        from CoreVital.sinks.datadog_sink import DatadogSink

        sink = DatadogSink(api_key="test-key", site="datadoghq.com")
        report = _make_report()
        tags = sink._build_tags(report)

        assert "model:gpt2" in tags
        assert "device:cpu" in tags
        assert "trace_id:sink-tes" in tags
        assert "quantized:false" in tags

    def test_build_tags_quantized(self):
        """Tags should reflect quantization when enabled."""
        from CoreVital.sinks.datadog_sink import DatadogSink

        sink = DatadogSink(api_key="test-key")
        report = _make_report()
        report.model.quantization = QuantizationInfo(enabled=True, method="bitsandbytes-4bit")
        tags = sink._build_tags(report)

        assert "quantized:true" in tags
        assert "quant_method:bitsandbytes-4bit" in tags

    @patch("CoreVital.sinks.datadog_sink._try_import_datadog")
    def test_build_series_count(self, mock_import):
        """Should produce the expected number of metric series."""
        # Mock the datadog classes
        mock_import.return_value = {
            "MetricSeries": MagicMock(side_effect=lambda **kwargs: kwargs),
            "MetricPoint": MagicMock(side_effect=lambda **kwargs: kwargs),
            "MetricIntakeType": MagicMock(GAUGE="gauge"),
            "ApiClient": MagicMock,
            "Configuration": MagicMock,
            "MetricsApi": MagicMock,
            "MetricPayload": MagicMock,
            "MetricResource": MagicMock,
        }
        from CoreVital.sinks.datadog_sink import DatadogSink

        sink = DatadogSink(api_key="test-key")
        report = _make_report(num_steps=3)
        series = sink._build_series(report)

        # Aggregate: total_steps + elapsed_ms + prompt_tokens + 6 health flags = 9
        # Per-step: 3 steps × 3 metrics (entropy, perplexity, surprisal) = 9
        # Total = 18
        assert len(series) == 18, f"Expected 18 series, got {len(series)}"

    @patch("CoreVital.sinks.datadog_sink._try_import_datadog")
    def test_build_series_no_health_flags(self, mock_import):
        """Should handle missing health flags gracefully."""
        mock_import.return_value = {
            "MetricSeries": MagicMock(side_effect=lambda **kwargs: kwargs),
            "MetricPoint": MagicMock(side_effect=lambda **kwargs: kwargs),
            "MetricIntakeType": MagicMock(GAUGE="gauge"),
            "ApiClient": MagicMock,
            "Configuration": MagicMock,
            "MetricsApi": MagicMock,
            "MetricPayload": MagicMock,
            "MetricResource": MagicMock,
        }
        from CoreVital.sinks.datadog_sink import DatadogSink

        sink = DatadogSink(api_key="test-key")
        report = _make_report(with_health_flags=False, num_steps=2)
        series = sink._build_series(report)

        # No health flags: total_steps + elapsed_ms + prompt_tokens = 3
        # Per-step: 2 × 3 = 6
        # Total = 9
        assert len(series) == 9, f"Expected 9 series, got {len(series)}"

    def test_write_creates_local_backup(self, tmp_path):
        """write() should always create a local JSON backup."""
        from CoreVital.sinks.datadog_sink import DatadogSink

        sink = DatadogSink(api_key="test-key", local_output_dir=str(tmp_path))
        report = _make_report()

        # Mock the Datadog API submission
        with patch("CoreVital.sinks.datadog_sink._try_import_datadog") as mock_dd:
            mock_api_instance = MagicMock()
            mock_dd.return_value = {
                "MetricSeries": MagicMock(side_effect=lambda **kwargs: kwargs),
                "MetricPoint": MagicMock(side_effect=lambda **kwargs: kwargs),
                "MetricIntakeType": MagicMock(GAUGE="gauge"),
                "ApiClient": MagicMock(
                    return_value=MagicMock(__enter__=MagicMock(return_value=MagicMock()), __exit__=MagicMock())
                ),
                "Configuration": MagicMock(),
                "MetricsApi": MagicMock(return_value=mock_api_instance),
                "MetricPayload": MagicMock(),
                "MetricResource": MagicMock(),
            }
            result = sink.write(report)

        # Local backup should exist
        json_files = list(tmp_path.glob("trace_*.json"))
        assert len(json_files) == 1, "Should have exactly one local backup"

        # Result should mention both datadog and local
        assert "datadog" in result
        assert str(tmp_path) in result


# ============================================================================
# PrometheusSink Tests
# ============================================================================


class TestPrometheusSink:
    """Tests for PrometheusSink gauge creation and metric exposure."""

    def setup_method(self):
        """Reset class-level state between tests."""
        from CoreVital.sinks.prometheus_sink import PrometheusSink

        PrometheusSink._gauges = {}
        PrometheusSink._server_started = False

    @patch("CoreVital.sinks.prometheus_sink._try_import_prometheus")
    def test_write_creates_gauges(self, mock_import, tmp_path):
        """write() should create and update Prometheus gauges."""
        mock_gauge_instance = MagicMock()
        mock_gauge_class = MagicMock(return_value=mock_gauge_instance)
        mock_import.return_value = {
            "Gauge": mock_gauge_class,
            "Info": MagicMock(),
            "start_http_server": MagicMock(),
        }

        from CoreVital.sinks.prometheus_sink import PrometheusSink

        sink = PrometheusSink.__new__(PrometheusSink)
        sink.port = 9091
        sink.local_output_dir = str(tmp_path)

        report = _make_report()

        with patch.object(PrometheusSink, "_ensure_server"):
            result = sink.write(report)

        # Should have created gauges
        assert mock_gauge_class.call_count > 0, "Should create at least one gauge"
        assert "metrics" in result

    @patch("CoreVital.sinks.prometheus_sink._try_import_prometheus")
    def test_write_starts_server_once(self, mock_import, tmp_path):
        """Server should only start once across multiple writes."""
        mock_start = MagicMock()
        mock_import.return_value = {
            "Gauge": MagicMock(return_value=MagicMock()),
            "Info": MagicMock(),
            "start_http_server": mock_start,
        }

        from CoreVital.sinks.prometheus_sink import PrometheusSink

        sink = PrometheusSink(port=9091, local_output_dir=str(tmp_path))
        report = _make_report()

        sink.write(report)
        sink.write(report)

        # start_http_server should be called exactly once
        assert mock_start.call_count == 1

    def test_write_creates_local_backup(self, tmp_path):
        """write() should always create a local JSON backup."""
        from CoreVital.sinks.prometheus_sink import PrometheusSink

        sink = PrometheusSink(port=9091, local_output_dir=str(tmp_path))
        report = _make_report()

        with patch("CoreVital.sinks.prometheus_sink._try_import_prometheus") as mock_prom:
            mock_prom.return_value = {
                "Gauge": MagicMock(return_value=MagicMock()),
                "Info": MagicMock(),
                "start_http_server": MagicMock(),
            }
            sink.write(report)

        json_files = list(tmp_path.glob("trace_*.json"))
        assert len(json_files) == 1, "Should have exactly one local backup"


# ============================================================================
# CLI Sink Routing Tests
# ============================================================================


class TestCLISinkRouting:
    """Test that --sink flag routes to the correct sink class."""

    def test_cli_parser_has_sink_arg(self):
        """CLI should have --sink with correct choices."""
        from CoreVital.cli import create_parser

        parser = create_parser()
        # Parse with --sink local
        args = parser.parse_args(["run", "--model", "gpt2", "--prompt", "test", "--sink", "local"])
        assert args.sink == "local"

    def test_cli_parser_sink_datadog(self):
        """CLI should accept --sink datadog with --datadog_api_key."""
        from CoreVital.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(
            [
                "run",
                "--model",
                "gpt2",
                "--prompt",
                "test",
                "--sink",
                "datadog",
                "--datadog_api_key",
                "my-key",
            ]
        )
        assert args.sink == "datadog"
        assert args.datadog_api_key == "my-key"

    def test_cli_parser_sink_prometheus(self):
        """CLI should accept --sink prometheus with --prometheus_port."""
        from CoreVital.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(
            [
                "run",
                "--model",
                "gpt2",
                "--prompt",
                "test",
                "--sink",
                "prometheus",
                "--prometheus_port",
                "8888",
            ]
        )
        assert args.sink == "prometheus"
        assert args.prometheus_port == 8888

    def test_cli_parser_sink_sqlite(self):
        """CLI should accept --sink sqlite."""
        from CoreVital.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(
            [
                "run",
                "--model",
                "gpt2",
                "--prompt",
                "test",
                "--sink",
                "sqlite",
            ]
        )
        assert args.sink == "sqlite"

    def test_cli_parser_default_sink(self):
        """Default --sink should be 'sqlite' (database-first)."""
        from CoreVital.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["run", "--model", "gpt2", "--prompt", "test"])
        assert args.sink == "sqlite"

    def test_cli_parser_capture_flag(self):
        """CLI should expose --capture with correct choices."""
        from CoreVital.cli import create_parser

        parser = create_parser()
        # Explicit capture mode
        args = parser.parse_args(["run", "--model", "gpt2", "--prompt", "test", "--capture", "summary"])
        assert args.capture == "summary"

        # Default when omitted
        args_default = parser.parse_args(["run", "--model", "gpt2", "--prompt", "test"])
        assert getattr(args_default, "capture", None) is None

    def test_cli_parser_rag_context(self):
        """CLI should accept --rag-context path."""
        from CoreVital.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(
            [
                "run",
                "--model",
                "gpt2",
                "--prompt",
                "test",
                "--rag-context",
                "/path/to/rag.json",
            ]
        )
        assert args.rag_context == "/path/to/rag.json"

        args_default = parser.parse_args(["run", "--model", "gpt2", "--prompt", "test"])
        assert getattr(args_default, "rag_context", None) is None


# ============================================================================
# SQLite Sink Tests
# ============================================================================


class TestSQLiteSink:
    """Test SQLiteSink write, list_traces, and load_report."""

    def test_write_and_load_report(self, tmp_path):
        """Write a report to SQLite and load it back."""
        from CoreVital.sinks.sqlite_sink import SQLiteSink

        db_path = str(tmp_path / "corevital.db")
        sink = SQLiteSink(db_path=db_path, compress=False)
        report = _make_report(num_steps=2)

        out = sink.write(report)
        assert out == db_path

        loaded = SQLiteSink.load_report(db_path, report.trace_id)
        assert loaded is not None
        assert loaded["trace_id"] == report.trace_id
        assert loaded["model"]["hf_id"] == report.model.hf_id
        assert len(loaded["timeline"]) == 2

    def test_list_traces(self, tmp_path):
        """list_traces returns recent trace metadata without report_json."""
        from CoreVital.sinks.sqlite_sink import SQLiteSink

        db_path = str(tmp_path / "corevital.db")
        sink = SQLiteSink(db_path=db_path, compress=True)
        sink.write(_make_report(num_steps=1))

        traces = SQLiteSink.list_traces(db_path)
        assert len(traces) == 1
        assert "trace_id" in traces[0]
        assert "model_id" in traces[0]
        assert "created_at_utc" in traces[0]

    def test_list_traces_filter_by_model(self, tmp_path):
        """list_traces with model_id filter returns only that model (Phase-6)."""
        from CoreVital.sinks.sqlite_sink import SQLiteSink

        db_path = str(tmp_path / "corevital.db")
        sink = SQLiteSink(db_path=db_path, compress=False)
        r1 = _make_report(num_steps=1)
        sink.write(r1)
        r2 = _make_report(num_steps=2)
        r2.trace_id = "sink-test-5678-efgh"
        r2.model.hf_id = "other-model"
        sink.write(r2)
        traces = SQLiteSink.list_traces(db_path, model_id="gpt2")
        assert len(traces) == 1
        assert traces[0]["model_id"] == "gpt2"

    def test_migrate_command_dry_run(self, tmp_path):
        """migrate --dry-run lists files without writing."""
        from CoreVital.cli import create_parser, migrate_command

        json_dir = tmp_path / "runs"
        json_dir.mkdir()
        minimal_json = (
            '{"schema_version":"0.3.0","trace_id":"abc12345-0000-0000-0000-000000000000",'
            '"created_at_utc":"2026-01-01T00:00:00Z","model":{"hf_id":"gpt2",'
            '"architecture":"GPT2LMHeadModel","num_layers":12,"hidden_size":768,'
            '"num_attention_heads":12,"tokenizer_hf_id":"gpt2","dtype":"float32",'
            '"device":"cpu","quantization":{"enabled":False}},"run_config":{},'
            '"prompt":{"text":"Hi","token_ids":[1],"num_tokens":1},'
            '"generated":{"output_text":"Hi","token_ids":[1],"num_tokens":1},'
            '"timeline":[],"summary":{"prompt_tokens":1,"generated_tokens":1,'
            '"total_steps":2,"elapsed_ms":0},"warnings":[]}'
        )
        (json_dir / "trace_abc12345.json").write_text(minimal_json)
        db_path = str(tmp_path / "corevital.db")

        parser = create_parser()
        args = parser.parse_args(["migrate", "--from-dir", str(json_dir), "--to-db", db_path, "--dry-run"])
        rc = migrate_command(args)
        assert rc == 0
        assert not Path(db_path).exists()

    def test_migrate_command_writes_db(self, tmp_path):
        """migrate writes JSON reports into SQLite."""
        from CoreVital.cli import create_parser, migrate_command
        from CoreVital.utils.serialization import serialize_report_to_json

        json_dir = tmp_path / "runs"
        json_dir.mkdir()
        report = _make_report(num_steps=1)
        json_path = json_dir / "trace_migrate.json"
        json_path.write_text(serialize_report_to_json(report, indent=None), encoding="utf-8")
        db_path = str(tmp_path / "corevital.db")

        parser = create_parser()
        args = parser.parse_args(["migrate", "--from-dir", str(json_dir), "--to-db", db_path])
        rc = migrate_command(args)
        assert rc == 0
        from CoreVital.sinks.sqlite_sink import SQLiteSink

        loaded = SQLiteSink.load_report(db_path, report.trace_id)
        assert loaded is not None
        assert loaded["model"]["hf_id"] == report.model.hf_id

    def test_load_report_by_short_id(self, tmp_path):
        """load_report accepts short trace_id prefix."""
        from CoreVital.sinks.sqlite_sink import SQLiteSink

        db_path = str(tmp_path / "corevital.db")
        sink = SQLiteSink(db_path=db_path, compress=False)
        report = _make_report(num_steps=1)

        sink.write(report)
        short_id = report.trace_id[:8]
        loaded = SQLiteSink.load_report(db_path, short_id)
        assert loaded is not None
        assert loaded["trace_id"] == report.trace_id


# ============================================================================
# Import guard tests
# ============================================================================


class TestImportGuards:
    """Test that sinks give clear errors when optional deps are missing."""

    def test_datadog_sink_init_without_api_call(self):
        """DatadogSink can be constructed without importing datadog-api-client."""
        from CoreVital.sinks.datadog_sink import DatadogSink

        # Construction should succeed — lazy import only happens at write() time
        sink = DatadogSink(api_key="test-key")
        assert sink.api_key == "test-key"

    def test_prometheus_sink_init_without_lib(self):
        """PrometheusSink can be constructed without importing prometheus-client."""
        from CoreVital.sinks.prometheus_sink import PrometheusSink

        # Construction should succeed — lazy import only happens at write() time
        sink = PrometheusSink(port=9999)
        assert sink.port == 9999


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
