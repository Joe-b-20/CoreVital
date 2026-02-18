# ============================================================================
# CoreVital - Production Model Smoke Tests (Foundation F4)
#
# Run with: pytest tests/test_models_production.py -v -m slow
# Skip by default: pytest -m "not slow"
# ============================================================================

import os

import pytest

from CoreVital.config import Config
from CoreVital.instrumentation.collector import InstrumentationCollector
from CoreVital.reporting.report_builder import ReportBuilder

# Production models per execution plan (F4). Use small variants for CI when possible.
PRODUCTION_MODELS = [
    "meta-llama/Llama-3.2-1B",  # Llama 3.2 1B (smallest)
    "mistralai/Mistral-7B-v0.1",  # Mistral 7B
    "mistralai/Mixtral-8x7B-v0.1",  # Mixtral 8x7B (large)
    "Qwen/Qwen2-0.5B",  # Qwen2 0.5B (small for CI)
]

GATED_MODELS = {"meta-llama/Llama-3.2-1B", "mistralai/Mistral-7B-v0.1", "mistralai/Mixtral-8x7B-v0.1"}


def _skip_if_gated(model_id: str) -> None:
    """Skip test when a gated model is requested but HF_TOKEN is not set."""
    if model_id in GATED_MODELS and not os.environ.get("HF_TOKEN"):
        pytest.skip(f"{model_id} is a gated model — set HF_TOKEN to run this test")


@pytest.mark.slow
@pytest.mark.parametrize("model_id", ["meta-llama/Llama-3.2-1B", "Qwen/Qwen2-0.5B"])
def test_production_model_smoke_cpu(model_id: str):
    """Smoke test: run CoreVital on a production model (CPU, 2 tokens). Skipped unless -m slow."""
    _skip_if_gated(model_id)
    config = Config()
    config.model.hf_id = model_id
    config.device.requested = "cpu"
    config.generation.max_new_tokens = 2
    config.generation.seed = 42
    config.generation.do_sample = False
    collector = InstrumentationCollector(config)
    results = collector.run("Hello world")
    assert results is not None
    assert len(results.generated_token_ids) <= 2
    builder = ReportBuilder(config)
    report = builder.build(results, "Hello world")
    assert report.trace_id
    assert report.model.hf_id == model_id
    assert report.extensions.get("risk") is not None
    assert report.extensions.get("fingerprint") is not None


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.parametrize("model_id", PRODUCTION_MODELS)
def test_production_model_smoke_gpu(model_id: str):
    """Smoke test on GPU for all production models. Run with -m 'slow and gpu'."""
    import torch

    _skip_if_gated(model_id)
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    config = Config()
    config.model.hf_id = model_id
    config.device.requested = "cuda"
    config.generation.max_new_tokens = 2
    config.generation.seed = 42
    config.generation.do_sample = False
    collector = InstrumentationCollector(config)
    results = collector.run("Compare model behavior.")
    assert results is not None
    builder = ReportBuilder(config)
    report = builder.build(results, "Compare model behavior.")
    assert report.extensions.get("fingerprint", {}).get("prompt_hash")
