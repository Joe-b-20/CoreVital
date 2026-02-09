# CoreVital

**Pre-Phase-1**: Hugging Face Instrumentation + JSON Trace Artifact + Sink Interface + Performance Monitoring

An open-source Python toolkit for monitoring the internal health of Large Language Models during inference. This implementation provides deep instrumentation of Hugging Face transformers, capturing hidden states, attention patterns, and logit distributions without saving full tensors.

## Features

-  **Deep Instrumentation**: Capture hidden states, attention patterns, and logits at every generation step
-  **Summary Statistics**: Compute lightweight summaries (mean, std, L2 norm, entropy, etc.) instead of full tensors
-  **Performance Monitoring**: Track operation times with `--perf` flag (summary, detailed, or strict mode)
-  **Model Registry**: Single source of truth for model type detection via `ModelCapabilities`
-  **Extensible Persistence**: Pluggable Sink interface (LocalFileSink, HTTPSink stub, DatadogSink/PrometheusSink stubs)
-  **CI/CD**: GitHub Actions workflow with pytest, Ruff linting, and MyPy type checking
-  **Configurable**: YAML configuration with environment variable overrides
-  **CPU/CUDA Support**: Automatic device detection or manual override
-  **Quantization Support**: 4-bit and 8-bit quantization via bitsandbytes for memory-efficient inference
-  **Structured Artifacts**: JSON trace files with schema version `0.2.0` for future compatibility

## Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/Joe-b-20/CoreVital.git
cd CoreVital

# Install in development mode
pip install -e .
```

### Basic Usage
```bash
# Run monitoring on GPT-2 (CausalLM) with a simple prompt
python -m CoreVital.cli run \
  --model gpt2 \
  --prompt "Explain why the sky is blue" \
  --max_new_tokens 50 \
  --device auto

# Run monitoring on T5 (Seq2Seq) model
python -m CoreVital.cli run \
  --model google/flan-t5-small \
  --prompt "My code works and I have no idea why, which is infinitely more terrifying than when it doesn't work and I have no idea why." \
  --max_new_tokens 20 \
  --device auto

# Run with 4-bit quantization (requires CUDA)
python -m CoreVital.cli run \
  --model gpt2 \
  --prompt "Explain why the sky is blue" \
  --max_new_tokens 50 \
  --device cuda \
  --quantize-4

# Run with 8-bit quantization (requires CUDA)
python -m CoreVital.cli run \
  --model gpt2 \
  --prompt "Explain why the sky is blue" \
  --max_new_tokens 50 \
  --device cuda \
  --quantize-8

# Run with performance monitoring (summary mode)
python -m CoreVital.cli run \
  --model gpt2 \
  --prompt "Hello world" \
  --perf

# Run with detailed performance breakdown
python -m CoreVital.cli run \
  --model gpt2 \
  --prompt "Hello world" \
  --perf detailed

# Run with strict mode (includes warmup and baseline measurements)
python -m CoreVital.cli run \
  --model gpt2 \
  --prompt "Hello world" \
  --perf strict

# Output will be saved to ./runs/ directory
```

### CLI Options
```bash
python -m CoreVital.cli run --help

Options:
  --model TEXT              Hugging Face model ID (required)
  --prompt TEXT             Input prompt text (required)
  --max_new_tokens INT      Number of tokens to generate [default: 20]
  --device TEXT             Device: auto|cpu|cuda [default: auto]
  --seed INT                Random seed [default: 42]
  --temperature FLOAT       Sampling temperature [default: 0.8]
  --top_k INT              Top-k sampling [default: 50]
  --top_p FLOAT            Top-p sampling [default: 0.95]
  --quantize-4              Load model with 4-bit quantization (requires CUDA)
  --quantize-8              Load model with 8-bit quantization (requires CUDA)
  --out PATH               Output path (directory or file)
  --remote_sink TEXT       Remote sink: none|http [default: none]
  --remote_url TEXT        Remote sink URL
  --config PATH            Path to custom config YAML file
  --log_level TEXT         Logging level: DEBUG|INFO|WARNING|ERROR [default: INFO]
  --perf [MODE]            Performance monitoring: summary (default), detailed, or strict
```

## Output Format

Each run produces a JSON trace file in `./runs/` with this structure:
```json
{
  "schema_version": "0.2.0",
  "trace_id": "uuid-here",
  "created_at_utc": "2026-01-11T15:22:08Z",
  "model": {
    "hf_id": "gpt2",
    "architecture": "GPT2LMHeadModel",
    "dtype": "float32",
    "device": "cpu",
    "quantization": { "enabled": false, "method": null }
  },
  "run_config": { ... },
  "prompt": {
    "text": "...",
    "token_ids": [...],
    "num_tokens": 10
  },
  "generated": {
    "output_text": "...",
    "token_ids": [...],
    "num_tokens": 50
  },
  "timeline": [
    {
      "step_index": 0,
      "token": { "token_id": 123, "token_text": "hello", "is_prompt_token": false },
      "logits_summary": { "entropy": 8.12, "top1_top2_margin": 0.34, "topk": [...] },
      "layers": [
        {
          "layer_index": 0,
          "hidden_summary": { "mean": 0.001, "std": 0.98, ... },
          "attention_summary": { "entropy_mean": 2.31, ... },
          "cross_attention": { "entropy_mean": 0.92, ... },
          "extensions": {}
        }
      ],
      "extensions": {}
    }
  ],
  "summary": {
    "prompt_tokens": 10,
    "generated_tokens": 50,
    "total_steps": 60,
    "elapsed_ms": 1234
  },
  "warnings": [],
  "encoder_layers": [
    {
      "layer_index": 0,
      "hidden_summary": { "mean": 0.5, "std": 1.2, ... },
      "attention_summary": { "entropy_mean": 2.85, ... },
      "cross_attention": null,
      "extensions": {}
    }
  ],
  "extensions": {}
}
```

### Key Components

- **prompt**: Contains the input prompt text, number of tokens and token IDs
- **generated**: Contains the generated output text, number of tokens and token IDs
- **timeline**: Per-token trace covering generated tokens. Each step contains decoder layer summaries.
- **hidden_summary**: Mean, std, L2 norm, max abs value, and random projection sketch
- **attention_summary**: Entropy statistics (entropy_mean, entropy_min) and concentration metrics (concentration_max). 
  - For decoder layers (in timeline): Contains decoder self-attention
  - For encoder layers (in encoder_layers): Contains encoder self-attention
  - This field ALWAYS contains self-attention, regardless of model type
- **cross_attention**: (Seq2Seq only) Cross-attention statistics showing how the decoder attends to encoder outputs at each generation step. Only used in decoder layers (in timeline). Always null for CausalLM models and for encoder layers.
- **encoder_layers**: (Seq2Seq only) Encoder layer summaries computed once at the start of generation. Each layer includes `hidden_summary` and `attention_summary` (encoder self-attention). Always null for CausalLM models.
- **logits_summary**: Entropy, top-1/top-2 margin, and top-k token probabilities
- **model.dtype**: Model dtype as string. `Optional[str]` — may be `null` or `"quantized_unknown"` when dtype cannot be definitively detected for quantized models.
- **model.revision**: Model commit hash/revision extracted from model config
- **model.quantization**: Quantization information (enabled: bool, method: "4-bit"|"8-bit"|null). The dtype field shows quantized dtypes (int8, uint8) when detectable, or `"quantized_unknown"` otherwise.
- **extensions**: Custom key-value pairs available at Report, TimelineStep, and LayerSummary levels for future metric expansion.

### Performance Monitoring (`--perf`)

The `--perf` flag enables performance monitoring with three modes:

**Summary Mode** (`--perf` or `--perf summary`):
- Adds `performance` extension to the main trace JSON
- Shows total wall time and breakdown by parent operations
- Tracks: config_load, setup_logging, model_load, torch.manual_seed, tokenize, model_inference, report_build

**Detailed Mode** (`--perf detailed`):
- Everything in summary mode, plus:
- Creates a separate `*_performance_detailed.json` file
- Shows nested breakdown with child operations and per-step statistics
- Useful for identifying specific bottlenecks

**Strict Mode** (`--perf strict`):
- Everything in detailed mode, plus:
- Runs warmup before measurements to stabilize GPU timing
- Runs baseline (uninstrumented) inference for comparison
- Reports original model load time (before caching)
- Calculates inference overhead and CoreVital overhead percentages

Example performance output in summary:
```json
{
  "extensions": {
    "performance": {
      "total_wall_time_ms": 2500.0,
      "parent_operations": [
        {"name": "config_load", "ms": 3.0, "pct": 0.12},
        {"name": "model_load", "ms": 1700.0, "pct": 68.0},
        {"name": "model_inference", "ms": 700.0, "pct": 28.0}
      ],
      "unaccounted_time": {"ms": 2.0, "pct": 0.08},
      "detailed_file": "runs/trace_abc123_performance_detailed.json"
    }
  }
}
```

### Model Compatibility Notes

- **Causal Language Models (GPT-2, LLaMA, etc.)**: Fully supported with automatic detection. The tool automatically switches attention implementation from SDPA to 'eager' for Llama models to enable attention weight capture. This may slightly increase inference time but is necessary for attention analysis.
- **Sequence-to-Sequence Models (T5, BART, etc.)**: Fully supported with automatic detection and deep instrumentation. The tool uses manual generation to capture hidden states and attentions, as Seq2Seq models don't return these via the standard `generate()` method. For Seq2Seq models, the tool captures:
  - **Encoder outputs**: Encoder hidden states and encoder self-attention (computed once, fixed for the entire run)
  - **Decoder outputs**: Decoder hidden states and decoder self-attention (computed at each generation step)
  - **Cross-attention**: How the decoder attends to encoder outputs at each generation step, showing how the model "listens" to the encoded input
- **Other Models**: Models using eager attention by default will work without modification. Models that don't support attention output will log warnings.
- **Quantization**: 4-bit and 8-bit quantization via bitsandbytes is supported for models that are compatible. Quantization requires CUDA and will automatically fall back to CPU without quantization if CUDA is unavailable. The quantization status is reflected in the output JSON report.

## Architecture

### Sink Interface

The Sink interface allows pluggable persistence backends:
```python
from CoreVital.sinks.base import Sink
from CoreVital.reporting.schema import Report

class CustomSink(Sink):
    def write(self, report: Report) -> str:
        # Your custom persistence logic
        return "location_identifier"
```

Built-in sinks:
- **LocalFileSink**: Write JSON to local filesystem
- **HTTPSink**: POST JSON to remote endpoint (stub)
- **DatadogSink**: Send metrics to Datadog (stub — planned for phase-2+)
- **PrometheusSink**: Expose metrics for Prometheus scraping (stub — planned for phase-2+)

### Configuration

Override defaults via `configs/default.yaml` or environment variables:
```bash
export COREVITAL_DEVICE=cuda
export COREVITAL_SEED=123
```

## Development

### Running Tests
```bash
# Run all tests (excluding GPU tests)
pytest tests/ -v --tb=short -m "not gpu"

# Run smoke test only
pytest tests/test_smoke_gpt2_cpu.py -v

# Run mock instrumentation tests (fast, no model loading)
pytest tests/test_mock_instrumentation.py -v

# Run performance monitoring tests
pytest tests/test_performance.py -v

# Run with coverage
pytest --cov=CoreVital tests/
```

### Linting & Formatting (Ruff)
```bash
# Check for lint errors
ruff check src/ tests/

# Auto-fix lint errors
ruff check src/ tests/ --fix

# Check formatting
ruff format --check src/ tests/

# Auto-format
ruff format src/ tests/
```

Ruff is configured in `pyproject.toml` with rules: `["E", "F", "I", "W", "B"]` (includes flake8-bugbear).

### Type Checking (MyPy)
```bash
mypy src/CoreVital/ --ignore-missing-imports --warn-return-any --warn-unused-configs
```

### CI/CD

GitHub Actions runs on every push and pull request to `main`:
- **Lint & Format**: Ruff check + format verification
- **Type Check**: MyPy static analysis
- **Test**: pytest suite (Python 3.12)

### Mock Testing Suite

The project includes a comprehensive mock testing suite that allows testing instrumentation logic without loading heavy models. This enables fast, lightweight testing of the instrumentation pipeline.

**Mock Fixtures** (`tests/conftest.py`):
- `mock_model_bundle`: Provides a mock `ModelBundle` with configurable model and tokenizer
- Supports both CausalLM and Seq2Seq architectures via parametrization
- Returns properly shaped tensors for all outputs (hidden states, attentions, cross-attentions)

**Mock Tests** (`tests/test_mock_instrumentation.py`):
- Tests `InstrumentationCollector` with mock models
- Tests `ReportBuilder` produces valid JSON reports
- Verifies tensor shapes for both Causal and Seq2Seq models
- Full pipeline integration tests

Usage:
```bash
# Test CausalLM mocks
pytest tests/test_mock_instrumentation.py::TestMockCausalLMInstrumentation -v

# Test Seq2Seq mocks
pytest tests/test_mock_instrumentation.py::TestMockSeq2SeqInstrumentation -v

# Test full pipeline
pytest tests/test_mock_instrumentation.py::TestMockInstrumentationIntegration -v
```

### Project Structure

- `src/CoreVital/`: Main package
  - `models/`: Model loading, management, and `ModelCapabilities` registry
  - `instrumentation/`: Hooks, collectors, summary computation, and performance monitoring
  - `reporting/`: Schema (v0.2.0), validation, and report building
  - `sinks/`: Persistence backends (LocalFile, HTTP, Datadog, Prometheus)
  - `utils/`: Shared utilities
- `.github/workflows/`: CI/CD pipeline (test.yaml)
- `configs/`: YAML configuration files
- `runs/`: Default output directory for trace artifacts
- `tests/`: Test suite

## Roadmap

**Pre-Phase-1** (Current): Cleanup & Tooling
- ✅ Schema v0.2.0: removed deprecated fields (`encoder_attention`, `encoder_hidden_states`)
- ✅ `ModelCapabilities` registry: centralized model type detection, eliminated scattered hardcoded checks
- ✅ Robust dtype detection: `Optional[str]` with explicit `"quantized_unknown"` fallback
- ✅ CI/CD: GitHub Actions workflow (pytest, Ruff, MyPy) on every push
- ✅ Ruff linter configured with rules `["E", "F", "I", "W", "B"]` (includes flake8-bugbear)
- ✅ MyPy type checking configured
- ✅ Stub sinks: `DatadogSink` and `PrometheusSink` (implementing `Sink` interface)
- ✅ Codebase-wide lint and format cleanup

**Phase-0.75** (Complete): Performance Monitoring
- ✅ `--perf` CLI flag with three modes: summary, detailed, strict
- ✅ Lightweight operation timing via context managers
- ✅ Nested operation hierarchy tracking (parent/child relationships)
- ✅ Per-step statistics for repeated operations (count, min, max, avg)
- ✅ Strict mode: warmup runs, baseline measurements, overhead calculations
- ✅ Separate detailed breakdown JSON file
- ✅ Performance data in `extensions.performance` of main trace

**Phase-0.5** (Complete): Hardening & Future-Proofing
- ✅ Extensions field on Report, TimelineStep, and LayerSummary for future metric expansion
- ✅ Separated encoder_layers (computed once) from decoder timeline for Seq2Seq models
- ✅ Robust Seq2Seq detection with Mock object handling for testing
- ✅ Improved quantization validation and logging
- ✅ Fixed dtype detection for quantized models (now shows int8/uint8 instead of float16)
- ✅ Memory optimizations (decoder self-attention slicing)
- ✅ Standardized logging levels (INFO for model loading, DEBUG for tensor extraction)
- ✅ Comprehensive persistence and validation tests
- ✅ Comprehensive test coverage for Phase-0.5 features (extensions, encoder_layers)

**Phase-0** (Complete): HF instrumentation + JSON trace + Sink interface
- ✅ Capture hidden states, attention, logits
- ✅ Compute lightweight summaries
- ✅ JSON artifact generation
- ✅ LocalFileSink implementation
- ✅ Automatic attention implementation handling (SDPA → eager for attention outputs)
- ✅ Model revision extraction from config
- ✅ Dynamic model loading with automatic Seq2Seq detection (T5, BART, etc.)
- ✅ Manual generation for Seq2Seq models to capture hidden states and attentions
- ✅ Deep Seq2Seq instrumentation: encoder hidden states, encoder attention, and cross-attention metrics
- ✅ 4-bit and 8-bit quantization support via bitsandbytes
- ✅ Mock testing suite for fast instrumentation testing without model loading

**Upcoming Phases**:
- Phase-1: Internal metrics (prompt telemetry, dashboard, integration sinks)
- Phase-2: Risk scores + layer blame
- Phase-3: Prompt fingerprints
- Phase-4: Failure-horizon prediction
- Phase-5: Health-aware decoding
- Phase-6: Cross-model comparison
- Phase-7: Human-readable narratives
- Phase-8: Dashboard + packaging

## Requirements

- Python 3.12+
- PyTorch
- Transformers (Hugging Face)
- PyYAML
- Pydantic
- bitsandbytes (for quantization support)
- accelerate (required by bitsandbytes)

**Dev dependencies** (`pip install -e ".[dev]"`):
- pytest
- ruff
- mypy

## License

Apache 2.0

## Contributing

Contributions welcome! Please open an issue or PR.