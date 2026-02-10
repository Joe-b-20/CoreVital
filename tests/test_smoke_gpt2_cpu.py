# ============================================================================
# CoreVital - Smoke Test
#
# Purpose: CPU-only smoke test with GPT-2 model
# Inputs: None (uses GPT-2 and simple prompt)
# Outputs: Test pass/fail
# Dependencies: pytest, CoreVital
# Usage: pytest tests/test_smoke_gpt2_cpu.py -v
#
# Changelog:
#   2026-01-13: Initial smoke test for Phase-0
# ============================================================================

import json
from pathlib import Path

from CoreVital.config import Config
from CoreVital.instrumentation.collector import InstrumentationCollector
from CoreVital.reporting.report_builder import ReportBuilder
from CoreVital.sinks.local_file import LocalFileSink


def test_smoke_gpt2_cpu(tmp_path):
    """
    Smoke test: Run monitoring on GPT-2 with a simple prompt on CPU.

    This test verifies:
    1. Model loads successfully
    2. Generation runs without errors
    3. Report is built correctly
    4. JSON file is written with required structure
    """
    # Setup config
    config = Config()
    config.model.hf_id = "gpt2"
    config.device.requested = "cpu"
    config.generation.max_new_tokens = 5
    config.generation.seed = 42
    config.sink.output_dir = str(tmp_path)

    # Run instrumentation
    collector = InstrumentationCollector(config)
    results = collector.run("Hello")

    # Verify results
    assert len(results.prompt_token_ids) > 0, "Prompt should have tokens"
    assert len(results.generated_token_ids) > 0, "Should generate tokens"
    assert results.elapsed_ms > 0, "Should have elapsed time"

    # Build report
    builder = ReportBuilder(config)
    report = builder.build(results, "Hello")

    # Verify report structure
    assert report.schema_version == "0.3.0"
    assert report.trace_id is not None
    assert report.model.hf_id == "gpt2"
    assert report.summary.prompt_tokens > 0
    assert report.summary.generated_tokens > 0

    # Write to sink
    sink = LocalFileSink(str(tmp_path))
    output_path = sink.write(report)

    # Verify file exists
    output_file = Path(output_path)
    assert output_file.exists(), f"Output file should exist: {output_path}"

    # Verify JSON structure
    with open(output_file, "r") as f:
        data = json.load(f)

    # Check required top-level keys
    required_keys = [
        "schema_version",
        "trace_id",
        "created_at_utc",
        "model",
        "run_config",
        "prompt",
        "timeline",
        "summary",
        "warnings",
    ]

    for key in required_keys:
        assert key in data, f"Missing required key: {key}"

    # Check model info
    assert data["model"]["hf_id"] == "gpt2"
    assert data["model"]["num_layers"] > 0
    assert data["model"]["device"] == "cpu"

    # Check prompt
    assert data["prompt"]["text"] == "Hello"
    assert len(data["prompt"]["token_ids"]) > 0

    # Check summary
    assert data["summary"]["prompt_tokens"] > 0
    assert data["summary"]["generated_tokens"] > 0
    assert data["summary"]["elapsed_ms"] > 0

    print("\n✓ Smoke test passed!")
    print(f"  Output: {output_path}")
    print(f"  Prompt tokens: {data['summary']['prompt_tokens']}")
    print(f"  Generated tokens: {data['summary']['generated_tokens']}")
    print(f"  Timeline steps: {len(data['timeline'])}")


if __name__ == "__main__":
    # Allow running directly for manual testing
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        test_smoke_gpt2_cpu(Path(tmpdir))
