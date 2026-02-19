# ============================================================================
# CoreVital - Full Pipeline Integration Test
#
# Purpose: End-to-end test collector -> report_builder -> schema -> sink -> dashboard load (#16)
# Dependencies: pytest, CoreVital
# Usage: pytest tests/test_integration.py -v
# ============================================================================

import json
from pathlib import Path

from CoreVital.config import Config
from CoreVital.instrumentation.collector import InstrumentationCollector
from CoreVital.reporting.report_builder import ReportBuilder
from CoreVital.reporting.schema import Report
from CoreVital.sinks.local_file import LocalFileSink


def test_full_pipeline_gpt2_cpu(tmp_path: Path) -> None:
    """
    Full pipeline: collector -> report_builder -> schema validation -> sink write.
    Then load JSON and validate every schema field is populated and dashboard can consume it.
    """
    config = Config()
    config.model.hf_id = "gpt2"
    config.device.requested = "cpu"
    config.generation.max_new_tokens = 4
    config.generation.seed = 42
    config.sink.output_dir = str(tmp_path)

    # 1) Collector run
    collector = InstrumentationCollector(config)
    results = collector.run("Hi")
    assert len(results.prompt_token_ids) > 0
    assert len(results.generated_token_ids) > 0
    assert results.elapsed_ms > 0

    # 2) Report build
    builder = ReportBuilder(config)
    report = builder.build(results, "Hi")
    assert report.schema_version == "0.4.0"
    assert report.trace_id
    assert report.model.hf_id == "gpt2"
    assert report.summary.prompt_tokens > 0
    assert report.summary.generated_tokens > 0
    assert report.summary.total_steps > 0
    assert report.summary.elapsed_ms > 0

    # 3) Schema validation (round-trip)
    report_dict = report.model_dump(mode="json")
    Report.model_validate(report_dict)

    # 4) Sink write
    sink = LocalFileSink(str(tmp_path))
    output_path = sink.write(report)
    assert Path(output_path).exists()

    # 5) Dashboard load path: load JSON and assert structure the dashboard expects
    with open(output_path, "r") as f:
        data = json.load(f)

    # Required top-level keys (dashboard and CLI expect these)
    for key in (
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
    ):
        assert key in data, f"Missing key: {key}"

    # Model
    assert data["model"]["hf_id"] == "gpt2"
    assert data["model"]["num_layers"] > 0
    assert "device" in data["model"]

    # Summary
    assert data["summary"]["prompt_tokens"] > 0
    assert data["summary"]["generated_tokens"] > 0
    assert data["summary"]["elapsed_ms"] > 0

    # Timeline (dashboard reads timeline[i]["logits_summary"], etc.)
    timeline = data["timeline"]
    assert isinstance(timeline, list)
    assert len(timeline) >= 1
    step0 = timeline[0]
    assert "step_index" in step0
    assert "token" in step0
    assert "logits_summary" in step0
    ls = step0["logits_summary"]
    # At least one of entropy, perplexity, surprisal present (dashboard charts)
    assert "entropy" in ls or "perplexity" in ls or "surprisal" in ls or "top_k_margin" in ls

    # Prompt analysis (dashboard Prompt Analysis tabs)
    pa = data.get("prompt_analysis")
    assert pa is not None
    assert "layers" in pa
    assert len(pa["layers"]) > 0
    assert "layer_transformations" in pa
    assert "prompt_surprisals" in pa
    layer0 = pa["layers"][0]
    assert "basin_scores" in layer0
    assert "heads" in layer0

    # Health flags (dashboard health section)
    hf = data.get("health_flags")
    assert hf is not None
    assert "nan_detected" in hf and isinstance(hf["nan_detected"], bool)
    assert "inf_detected" in hf and isinstance(hf["inf_detected"], bool)
    assert "attention_collapse_detected" in hf and isinstance(hf["attention_collapse_detected"], bool)
    assert "high_entropy_steps" in hf and isinstance(hf["high_entropy_steps"], int)
    assert "repetition_loop_detected" in hf and isinstance(hf["repetition_loop_detected"], bool)
    assert "mid_layer_anomaly_detected" in hf and isinstance(hf["mid_layer_anomaly_detected"], bool)

    # Extensions.risk (dashboard / compare)
    ext = data.get("extensions", {})
    risk = ext.get("risk")
    assert risk is not None
    assert "risk_score" in risk
    assert 0 <= risk["risk_score"] <= 1.0
