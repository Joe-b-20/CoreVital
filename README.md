# CoreVital

**Phase-0**: Hugging Face Instrumentation + JSON Trace Artifact + Sink Interface

An open-source Python toolkit for monitoring the internal health of Large Language Models during inference. This Phase-0 implementation provides deep instrumentation of Hugging Face transformers, capturing hidden states, attention patterns, and logit distributions without saving full tensors.

## Features

-  **Deep Instrumentation**: Capture hidden states, attention patterns, and logits at every generation step
-  **Summary Statistics**: Compute lightweight summaries (mean, std, L2 norm, entropy, etc.) instead of full tensors
-  **Extensible Persistence**: Pluggable Sink interface (LocalFileSink included, HTTPSink stub)
-  **Configurable**: YAML configuration with environment variable overrides
-  **CPU/CUDA Support**: Automatic device detection or manual override
-  **Structured Artifacts**: JSON trace files with schema version for future compatibility

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
# Run monitoring on GPT-2 with a simple prompt
python -m CoreVital.cli run \
  --model gpt2 \
  --prompt "Explain why the sky is blue" \
  --max_new_tokens 50 \
  --device auto

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
  --out PATH               Output path (directory or file)
  --remote_sink TEXT       Remote sink: none|http [default: none]
  --remote_url TEXT        Remote sink URL
```

## Output Format

Each run produces a JSON trace file in `./runs/` with this structure:
```json
{
  "schema_version": "0.1.0",
  "trace_id": "uuid-here",
  "created_at_utc": "2026-01-11T15:22:08Z",
  "model": { ... },
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
      "token": { "token_id": 123, "token_text": "hello", "is_prompt_token": true },
      "logits_summary": { "entropy": 8.12, "top1_top2_margin": 0.34, "topk": [...] },
      "layers": [
        {
          "layer_index": 0,
          "hidden_summary": { "mean": 0.001, "std": 0.98, ... },
          "attention_summary": { "entropy_mean": 2.31, ... }
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
  "warnings": []
}
```

### Key Components

- **prompt**: Contains the input prompt text, number of tokens and token IDs
- **generated**: Contains the generated output text, number of tokens and token IDs
- **timeline**: Per-token trace covering both prompt and generated tokens
- **hidden_summary**: Mean, std, L2 norm, max abs value, and random projection sketch
- **attention_summary**: Entropy statistics (entropy_mean, entropy_min) and concentration metrics (concentration_max)
- **logits_summary**: Entropy, top-1/top-2 margin, and top-k token probabilities
- **model.revision**: Model commit hash/revision extracted from model config
- **extensions**: Reserved for future phases (risk scores, layer blame, etc.)

### Model Compatibility Notes

- **Llama Models**: The tool automatically switches attention implementation from SDPA to 'eager' to enable attention weight capture. This may slightly increase inference time but is necessary for attention analysis.
- **Other Models**: Models using eager attention by default will work without modification. Models that don't support attention output will log warnings.

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
- **HTTPSink**: POST JSON to remote endpoint (stub in Phase-0)

### Configuration

Override defaults via `configs/default.yaml` or environment variables:
```bash
export COREVITAL_DEVICE=cuda
export COREVITAL_SEED=123
```

## Development

### Running Tests
```bash
# Run all tests
pytest tests/

# Run smoke test only
pytest tests/test_smoke_gpt2_cpu.py -v

# Run with coverage
pytest --cov=CoreVital tests/
```

### Project Structure

- `src/CoreVital/`: Main package
  - `models/`: Model loading and management
  - `instrumentation/`: Hooks, collectors, and summary computation
  - `reporting/`: Schema, validation, and report building
  - `sinks/`: Persistence backends
  - `utils/`: Shared utilities
- `configs/`: YAML configuration files
- `runs/`: Default output directory for trace artifacts
- `tests/`: Test suite

## Roadmap

**Phase-0** (Current): HF instrumentation + JSON trace + Sink interface
- ✅ Capture hidden states, attention, logits
- ✅ Compute lightweight summaries
- ✅ JSON artifact generation
- ✅ LocalFileSink implementation
- ✅ Automatic attention implementation handling (SDPA → eager for attention outputs)
- ✅ Model revision extraction from config

**Future Phases** (Design only):
- Phase-1: Internal metrics
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
- pytest (for testing)

## License

Apache 2.0

## Contributing

Contributions welcome! Please open an issue or PR.