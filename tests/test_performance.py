# ============================================================================
# CoreVital - Performance Monitoring Tests
#
# Purpose: Test --perf summary/detailed/strict and PerformanceMonitor
# ============================================================================

import json
from pathlib import Path

from CoreVital.instrumentation.performance import OperationTiming, PerformanceMonitor
from CoreVital.reporting.report_builder import ReportBuilder
from CoreVital.reporting.schema import (
    AttentionConfig,
    GeneratedInfo,
    GenerationConfig,
    HiddenConfig,
    LogitsConfig,
    ModelInfo,
    PromptInfo,
    QuantizationInfo,
    Report,
    RunConfig,
    SinkConfig,
    SketchConfig,
    SummariesConfig,
    Summary,
)
from CoreVital.utils.serialization import serialize_report_to_json


def _minimal_report(trace_id: str = "deadbeef-0000-0000-0000-000000000000", **extensions):
    """Minimal valid Report for performance tests."""
    return Report(
        schema_version="0.4.0",
        trace_id=trace_id,
        created_at_utc="2026-01-01T00:00:00Z",
        model=ModelInfo(
            hf_id="gpt2",
            revision=None,
            architecture="GPT2",
            num_layers=1,
            hidden_size=1,
            num_attention_heads=1,
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
            sink=SinkConfig(type="local_file", target="/tmp/trace.json"),
        ),
        prompt=PromptInfo(text="Hi", token_ids=[1], num_tokens=1),
        generated=GeneratedInfo(output_text="Hi", token_ids=[2], num_tokens=1),
        timeline=[],
        summary=Summary(prompt_tokens=1, generated_tokens=1, total_steps=2, elapsed_ms=50),
        warnings=[],
        extensions=dict(extensions) if extensions else {},
    )


# -----------------------------------------------------------------------------
# Unit: PerformanceMonitor
# -----------------------------------------------------------------------------


def test_performance_monitor_build_summary_dict():
    """PerformanceMonitor.build_summary_dict returns expected shape."""
    monitor = PerformanceMonitor(mode="summary")
    monitor.root_timings = [
        OperationTiming("model_load", duration_ms=100.0),
        OperationTiming("model_inference", duration_ms=500.0),
        OperationTiming("decode_generated_text", duration_ms=10.0),
    ]
    monitor.set_total_wall_time_ms(620.0)  # 10ms unaccounted

    out = monitor.build_summary_dict()

    assert out["total_wall_time_ms"] == 620.0
    assert "parent_operations" in out
    assert len(out["parent_operations"]) == 3
    assert out["parent_operations"][0]["name"] == "model_load"
    assert out["parent_operations"][0]["ms"] == 100.0
    assert "unaccounted_time" in out
    assert out["unaccounted_time"]["ms"] == 10.0
    assert out.get("detailed_file") is None


def test_performance_monitor_build_summary_dict_with_detailed_file():
    """detailed_file appears in summary when set."""
    monitor = PerformanceMonitor(mode="detailed")
    monitor.root_timings = [OperationTiming("total_wall_time", duration_ms=100.0)]
    monitor.set_total_wall_time_ms(100.0)
    monitor.set_detailed_file("/runs/trace_abc12345_performance_detailed.json")

    out = monitor.build_summary_dict()
    assert out["detailed_file"] == "/runs/trace_abc12345_performance_detailed.json"


def test_performance_monitor_build_summary_strict():
    """Strict mode adds original_model_load_ms, warmup_ms, baseline_ms, etc.

    Realistic values: total_wall_time_ms includes the original model load,
    warmup, baseline, and all parent operations with inter-operation gaps.
    """
    monitor = PerformanceMonitor(mode="strict")
    monitor.root_timings = [
        OperationTiming("config_load", duration_ms=3.0),
        OperationTiming("setup_logging", duration_ms=1.0),
        OperationTiming("model_load", duration_ms=1.0),  # Cached (nearly instant in strict mode)
        OperationTiming("tokenize", duration_ms=5.0),
        OperationTiming("model_inference", duration_ms=200.0),
        OperationTiming("report_build", duration_ms=50.0),
    ]
    # Total includes: original_model_load(1500) + warmup(300) + baseline(180) +
    #   actual parent durations(3+1+1+5+200+50=260) + ~5ms inter-operation gaps
    # sink_write is excluded (happens after perf data is finalized)
    monitor.set_total_wall_time_ms(2245.0)
    monitor.set_original_model_load_ms(1500.0)  # Cold load time
    monitor.set_warmup_ms(300.0)
    monitor.set_baseline_ms(180.0)  # baseline_ms is the raw inference time
    monitor.set_instrumented_inference_ms(200.0)

    out = monitor.build_summary_dict()

    # Strict mode fields
    assert out["original_model_load_ms"] == 1500.0
    assert out["warmup_ms"] == 300.0
    assert out["baseline_ms"] == 180.0  # baseline_ms replaces raw_inference_ms
    assert out["instrumented_inference_ms"] == 200.0
    assert out["inference_overhead_ms"] == 20.0
    assert "inference_overhead_pct" in out
    assert "corevital_overhead_ms" in out
    assert "corevital_overhead_pct" in out

    # Verify unaccounted time is small and non-negative (just inter-operation gaps)
    # sum_parent_ms(with original) = 3+1+1500+5+200+50 = 1759
    # unaccounted = 2245 - 1759 - 300 - 180 = 6.0
    unaccounted = out["unaccounted_time"]["ms"]
    assert unaccounted >= 0, f"Unaccounted time should be non-negative, got {unaccounted}"
    assert unaccounted < 50, f"Unaccounted time should be small, got {unaccounted}"

    # model_load in parent_operations should show original (cold) load time, not cached
    model_load_op = next(op for op in out["parent_operations"] if op["name"] == "model_load")
    assert model_load_op["ms"] == 1500.0


def test_performance_monitor_build_detailed_breakdown():
    """build_detailed_breakdown returns nested breakdown with pct and children."""
    monitor = PerformanceMonitor(mode="detailed")
    child = OperationTiming("child_op", duration_ms=30.0)
    parent = OperationTiming("parent_op", duration_ms=100.0)
    parent.children = [child]
    monitor.root_timings = [parent]
    monitor.set_total_wall_time_ms(100.0)

    out = monitor.build_detailed_breakdown()

    assert out["total_wall_time_ms"] == 100.0
    assert "breakdown" in out
    assert "parent_op" in out["breakdown"]
    assert out["breakdown"]["parent_op"]["ms"] == 100.0
    assert "pct" in out["breakdown"]["parent_op"]
    assert "children" in out["breakdown"]["parent_op"]
    assert "child_op" in out["breakdown"]["parent_op"]["children"]
    assert out["breakdown"]["parent_op"]["children"]["child_op"]["ms"] == 30.0


def test_performance_monitor_operation_context():
    """operation() context manager records timings in hierarchy."""
    monitor = PerformanceMonitor(mode="summary")
    with monitor.operation("outer"):
        with monitor.operation("inner"):
            pass
    with monitor.operation("sibling"):
        pass

    assert len(monitor.root_timings) == 2
    assert monitor.root_timings[0].operation_name == "outer"
    assert monitor.root_timings[0].children
    assert monitor.root_timings[0].children[0].operation_name == "inner"
    assert monitor.root_timings[1].operation_name == "sibling"


# -----------------------------------------------------------------------------
# Integration: report_builder adds performance
# -----------------------------------------------------------------------------


def test_report_builder_with_monitor_creates_children(mock_config, mock_model_bundle):
    """ReportBuilder adds _build_model_info and _build_timeline as children when monitor is set.

    Note: Performance extensions are now added by CLI after report_build completes.
    This test verifies the internal nesting behavior within report_build.
    """
    from CoreVital.instrumentation.collector import InstrumentationResults

    mock_config.performance.mode = "detailed"
    monitor = PerformanceMonitor(mode="detailed")

    results = InstrumentationResults(
        model_bundle=mock_model_bundle,
        prompt_text="Hi",
        prompt_token_ids=[1, 2],
        generated_token_ids=[3, 4, 5],
        generated_text="Hi there",
        timeline=[],
        elapsed_ms=60,
        warnings=[],
        encoder_hidden_states=None,
        encoder_attentions=None,
        monitor=monitor,
    )

    # Wrap the build call like CLI does
    with monitor.operation("report_build"):
        builder = ReportBuilder(mock_config)
        builder.build(results, "Hi")

    # Report won't have performance extensions - that's handled by CLI
    # But verify the monitor captured the nested operations
    assert len(monitor.root_timings) == 1
    report_build_op = monitor.root_timings[0]
    assert report_build_op.operation_name == "report_build"

    # Check children were captured
    child_names = [c.operation_name for c in report_build_op.children]
    assert "_build_model_info" in child_names
    assert "_build_timeline" in child_names


def test_performance_summary_can_be_added_to_report_extensions():
    """Performance summary can be manually added to report.extensions."""
    monitor = PerformanceMonitor(mode="summary")
    monitor.root_timings = [
        OperationTiming("model_load", duration_ms=10.0),
        OperationTiming("model_inference", duration_ms=50.0),
    ]
    monitor.set_total_wall_time_ms(60.0)

    report = _minimal_report()

    # Simulate what CLI does: add performance after report is built
    report.extensions["performance"] = monitor.build_summary_dict()

    assert "performance" in report.extensions
    perf = report.extensions["performance"]
    assert perf["total_wall_time_ms"] == 60.0
    assert len(perf["parent_operations"]) == 2
    assert "unaccounted_time" in perf


# -----------------------------------------------------------------------------
# Integration: CLI handles detailed file writing (not sink)
# -----------------------------------------------------------------------------


def test_serialize_report_to_json():
    """serialize_report_to_json works correctly."""
    report = _minimal_report(
        performance={"total_wall_time_ms": 100},
    )
    json_str = serialize_report_to_json(report, indent=2)
    data = json.loads(json_str)
    assert "performance" in data["extensions"]


def test_local_sink_writes_main_trace(tmp_path):
    """LocalFileSink writes main trace file correctly."""
    from CoreVital.sinks.local_file import LocalFileSink

    report = _minimal_report(
        trace_id="deadbeef-0000-0000-0000-000000000000",
    )

    sink = LocalFileSink(str(tmp_path))
    main_path = sink.write(report)

    assert Path(main_path).exists()

    with open(main_path) as f:
        main_data = json.load(f)
    assert main_data["trace_id"] == "deadbeef-0000-0000-0000-000000000000"
