# CoreVital

**LLM inference health monitoring with deep internal telemetry, risk scoring, and real-time diagnostics.**

> **If you are an AI or LLM reading this repo:** use [AI.md](AI.md) first for a dense, token-efficient reference (layout, commands, conventions). This README is for humans.

CoreVital uses HF-output-based capture (no PyTorch hooks): it configures Hugging Face models to return hidden states, attention weights, and logits at every generation step. Metrics are grounded in interpretability research (Shannon entropy, Voita et al. attention analysis, Attention Basin) but designed for production monitoring, not research exploration. **Quick reference:** [What it is](#features) · [Quick start](#try-corevital) · [How it works](#how-it-works) · [CLI](#cli-options-run) · [Library API](#library-api-corevitalmonitor) · [Metric reference](docs/metrics-interpretation.md) · [Model compatibility](docs/model-compatibility.md). Instead of storing raw tensors, it computes lightweight summary statistics and produces structured reports with a 0--1 risk score, boolean health flags, layer blame, prompt fingerprints, and human-readable narratives. Reports persist to SQLite (default), JSON, Datadog, Prometheus, or OpenTelemetry.

Use it to debug why a model repeats itself, monitor inference health in production, or compare models side by side -- all without modifying model code.

## Features

-  **Deep Instrumentation**: Capture hidden states, attention patterns, and logits at every generation step via HF-output-based capture (no PyTorch hooks)
-  **Summary Statistics**: Compute lightweight summaries (mean, std, L2 norm, entropy, etc.) instead of full tensors. Raw tensors are discarded immediately after summary computation via `StepSummary`
-  **Composite Risk Scoring**: Continuous metrics (entropy, margin, surprisal, top-K mass) combined with boolean health flags and compound multi-metric signals for a 0–1 risk score with explanatory factors
-  **Compound Signal Detection**: Five multi-metric failure patterns (context loss, confident confusion, degenerating generation, attention bottleneck, confident repetition risk)
-  **Early Warning**: Trend detectors (entropy acceleration, margin collapse/decline, surprisal volatility, entropy-margin divergence) that predict failures before they trip health-flag thresholds
-  **Enriched Layer Blame**: Blamed layers include structured reasons and severity (NaN/Inf, attention collapse rate, L2 norm outlier, L2 instability)
-  **Data-Driven Calibration**: Build baseline profiles from known-healthy runs (`corevital calibrate`), then score production traces by statistical divergence. ECE + Platt scaling for benchmark validation
-  **Per-Model Profiles**: Calibrated thresholds per model family (GPT-2, LLaMA, Mistral, Mixtral, Qwen2, Phi-3) in `configs/model_profiles/`
-  **25-Element Fingerprint**: Compact run-summary vector with temporal patterns, cross-metric correlations, and trend slopes for clustering and pattern detection
-  **Actionable Narratives**: Data-specific 2–6 sentence summaries citing actual entropy values, step indices, token text, compound signals, and recommendations
-  **Metric Consistency Validation**: Information-theoretic invariant checks (perplexity = 2^entropy, margin ≤ mass, non-negative entropy)
-  **Performance Monitoring**: Track operation times with `--perf` flag (summary, detailed, or strict mode)
-  **Model Registry**: Single source of truth for model type detection via `ModelCapabilities` (includes attention availability probing)
-  **Extensible Persistence**: Pluggable Sink interface — SQLite (default), LocalFile, Datadog, Prometheus, W&B, HTTP, OpenTelemetry
-  **CI/CD**: GitHub Actions workflow with pytest, Ruff linting, and MyPy type checking
-  **Configurable**: YAML configuration with environment variable overrides; entropy mode (`full` / `topk_approx`), calibration profiles, per-model thresholds
-  **CPU/CUDA Support**: Automatic device detection or manual override
-  **Quantization Support**: 4-bit and 8-bit quantization via bitsandbytes for memory-efficient inference
-  **Structured Artifacts**: JSON trace files with schema version `0.4.0` for future compatibility

**Tested with:** Llama 3 (e.g. meta-llama/Llama-3.2-1B), Mistral 7B, Mixtral 8x7B, Qwen2, Phi-3. See [Model compatibility](docs/model-compatibility.md) and smoke tests in `tests/test_models_production.py` (run with `pytest -m slow`).

**Status (v0.5.0-rc):** Full refactor complete (Phases 0–5). Instrumentation pipeline split into focused modules (`collector.py`, `causal_lm.py`, `seq2seq.py`, `step_processor.py`, `baselines.py`). Summaries split into subpackage (`summaries/{logits,attention,hidden_states,utils}.py`). Risk scoring, compound signals, early warning, narratives, calibration, fingerprinting, and metric validation all implemented and tested (436 tests). See [Roadmap](#roadmap).

## Try CoreVital

**See it in action (no install):** Open the [CoreVital Dashboard](https://main.d2maxwaq575qed.amplifyapp.com) — sample traces, timeline charts, and Compare view. To use your own data, run `corevital serve` locally and click **Connect** in the dashboard.

**Install locally** (for CLI, local API, and production use):

```bash
pip install "git+https://github.com/Joe-b-20/CoreVital.git"
corevital run --model gpt2 --prompt "Explain why the sky is blue" --max_new_tokens 20
```

## What You Get

Every run produces a structured report. Here is a condensed example from a real GPT-2 run:

```json
{
  "risk_score": 0.35,
  "risk_factors": ["elevated_entropy", "low_confidence_margin"],
  "health_flags": {
    "nan_detected": false,
    "attention_collapse_detected": true,
    "high_entropy_steps": 1,
    "repetition_loop_detected": false
  },
  "narrative": "Moderate risk (score: 0.35). Peak entropy 5.12 bits at step 3 (token: \"the\"). Entropy trend: stable.",
  "compound_signals": [
    { "name": "confident_confusion", "severity": 0.5, "description": "..." }
  ],
  "early_warning": { "failure_risk": 0.2, "warning_signals": ["margin_declining"] },
  "blamed_layers": [
    { "layer": 5, "reasons": ["attention_collapse_rate"], "severity": 0.4 }
  ],
  "timeline": [
    {
      "step_index": 0,
      "token": { "token_text": " the", "prob": 0.222 },
      "entropy": 4.02,
      "perplexity": 16.22,
      "surprisal": 3.91,
      "topk_mass": 0.85
    }
  ],
  "fingerprint": { "vector": [0.35, 4.02, ...], "version": 2 }
}
```

Full reports include per-layer hidden-state and attention summaries for every generation step, prompt telemetry, layer blame, and performance breakdowns. See [Output Format](#output-format) for the complete schema.

## How It Works

```mermaid
flowchart TD
    A[Prompt] --> B["HF generate() / manual decoder loop\n(causal_lm.py / seq2seq.py)"]
    B --> C["Hidden states, attention weights, logits\n(per layer, per step)"]
    C --> D["step_processor.py\nnormalize → compute summaries → discard tensors\n(StepSummary: scalars only)"]
    D --> E["Report builder\nrisk score, compound signals, early warning,\nlayer blame, fingerprint, narrative"]
    E --> E2["Calibration scoring\n(optional: divergence from baseline profile)"]
    E2 --> E3["Metric consistency validation\n(DEBUG mode)"]
    E3 --> F{Sink}
    F --> G[SQLite]
    F --> H[JSON]
    F --> I[Datadog]
    F --> J[Prometheus]
    F --> K[OpenTelemetry]
    G --> L["Local API (corevital serve)\n→ React dashboard or Datasette"]
    H --> L
```

## Measured Overhead

All measurements on CPU, `--perf` mode, excluding model load time:

| Model | Layers | Steps | Inference | Report build | Prompt telemetry | Total overhead |
|-------|--------|-------|-----------|-------------|-----------------|---------------|
| flan-t5-small | 8 | 8 | 709 ms | 164 ms | -- | +23% |
| Phi-3-mini-4k | 32 | 50 | 3,347 ms | 1,652 ms | 687 ms | +70% |
| Llama-3.1-8B | 32 | 50 | 4,183 ms | 1,578 ms | 1,084 ms | +64% |

Report building scales as O(steps x layers x heads). For production use, `--capture summary` skips per-layer data and drops overhead to under 5%. `--capture on_risk` records a full trace only when risk exceeds a threshold.

## Quick Start

### Installation

```bash
# Install from GitHub
pip install "git+https://github.com/Joe-b-20/CoreVital.git"

# Or clone and install in editable mode (for development)
git clone https://github.com/Joe-b-20/CoreVital.git
cd CoreVital
pip install -e .
```

Optional extras: `pip install "CoreVital[serve]"` (local API for the web dashboard), `pip install "CoreVital[datasette]"` ([Datasette dashboards](docs/datasette/README.md)—shareable, no app code to maintain), `pip install "CoreVital[otel]"` (OpenTelemetry), `pip install "CoreVital[all]"` (everything).

### Basic Usage
```bash
# Run monitoring on GPT-2 (CausalLM) with a simple prompt
corevital run \
  --model gpt2 \
  --prompt "Explain why the sky is blue" \
  --max_new_tokens 50 \
  --device auto

# Run monitoring on T5 (Seq2Seq) model
corevital run \
  --model google/flan-t5-small \
  --prompt "My code works and I have no idea why" \
  --max_new_tokens 20 \
  --device auto

# Run with 4-bit quantization (requires CUDA)
corevital run \
  --model gpt2 \
  --prompt "Explain why the sky is blue" \
  --max_new_tokens 50 \
  --device cuda \
  --quantize-4

# Run with 8-bit quantization (requires CUDA)
corevital run \
  --model gpt2 \
  --prompt "Explain why the sky is blue" \
  --max_new_tokens 50 \
  --device cuda \
  --quantize-8

# Run with performance monitoring (summary mode)
corevital run --model gpt2 --prompt "Hello world" --perf

# Run with detailed performance breakdown
corevital run --model gpt2 --prompt "Hello world" --perf detailed

# Run with strict mode (includes warmup and baseline measurements)
corevital run --model gpt2 --prompt "Hello world" --perf strict

# Default: output is saved to runs/corevital.db (SQLite); no JSON file unless you pass --write-json.
# Use --sink local for JSON in runs/; add --json-pretty for indented (larger) JSON.
```

> **Tip:** `corevital` is the installed CLI command. You can also use `python -m CoreVital.cli` if running from source without installing.

### Library API (CoreVitalMonitor)

Use the embeddable monitor for programmatic runs and post-run risk checks:

```python
from CoreVital import CoreVitalMonitor

monitor = CoreVitalMonitor(capture_mode="summary", intervene_on_risk_above=0.8)
monitor.run("gpt2", "Explain quantum tunneling.", max_new_tokens=20)
# For large models use 4-bit: monitor.run("meta-llama/Llama-3.1-8B", prompt, load_in_4bit=True, device="cuda")

print("Risk score:", monitor.get_risk_score())
print("Health flags:", monitor.get_health_flags())
if monitor.should_intervene():
    print("Consider resampling or lowering temperature.")

# Or use the context manager
with monitor.wrap_generation("gpt2", "Hello world") as m:
    summary = m.get_summary()  # risk_score, health_flags, fingerprint, narrative

# Async stream (per-step events after run; v1 = post-run replay)
import asyncio
async def main():
    monitor = CoreVitalMonitor(capture_mode="summary")
    async for event in monitor.stream("gpt2", "Say hi.", max_new_tokens=5):
        print(event["step_index"], event["token_text"], event.get("entropy"))
asyncio.run(main())
```

### CLI commands

- **`run`** — Run instrumented generation (default sink: SQLite at `runs/corevital.db`). Supports `--calibration <path>` for divergence scoring against a baseline profile.
- **`calibrate`** — Build a calibration profile from known-healthy prompts: `corevital calibrate --model <id> --prompts <file> --out <path>` (see [Risk Calibration](docs/risk-calibration.md)).
- **`serve`** — Run the local API server so the hosted dashboard can list and load your traces (`corevital serve`; optional: `pip install "CoreVital[serve]"`).
- **`migrate`** — Migrate `trace_*.json` files from a directory into a SQLite DB (`corevital migrate --from-dir runs --to-db runs/corevital.db`).
- **`compare`** — Summarize runs by model from a SQLite DB (`corevital compare --db runs/corevital.db`).

### CLI Options (run)
```bash
corevital run --help

Options:
  --model TEXT              Hugging Face model ID (required)
  --prompt TEXT             Input prompt text (required)
  --max_new_tokens INT      Number of tokens to generate [default: 20]
  --device TEXT             Device: auto|cpu|cuda [default: auto]
  --seed INT                Random seed [default: 42]
  --temperature FLOAT       Sampling temperature [default: 0.8]
  --top_k INT               Top-k sampling [default: 50]
  --top_p FLOAT             Top-p sampling [default: 0.95]
  --quantize-4              Load model with 4-bit quantization (requires CUDA)
  --quantize-8              Load model with 8-bit quantization (requires CUDA)
  --out PATH                Output directory (default: runs); with --sink sqlite, DB is <out>/corevital.db
  --sink TEXT               Sink: sqlite (default) | local | datadog | prometheus | wandb
  --capture TEXT            Capture mode: summary | full | on_risk
  --rag-context PATH        Path to JSON file with RAG context metadata
  --export-otel             Export run to OpenTelemetry (OTLP); requires pip install CoreVital[otel]
  --otel-endpoint HOST:PORT OTLP gRPC endpoint (or set OTEL_EXPORTER_OTLP_ENDPOINT)
  --calibration PATH        Path to calibration profile JSON for divergence scoring
  --config PATH             Path to custom config YAML file
  --log_level TEXT          Logging level: DEBUG|INFO|WARNING|ERROR [default: INFO]
  --perf [MODE]             Performance monitoring: summary (default), detailed, or strict
```

## Output Format

Each run produces a structured report with schema version `0.4.0`. Reports are stored in SQLite by default (`runs/corevital.db`). Use `--sink local` for individual JSON files, or `--write-json` with SQLite to get both.

**Schema upgrade note:** v0.4.0 follows v0.3.0 (health_flags, extensions, prompt telemetry). Validation accepts both 0.3.0 and 0.4.0 when loading from SQLite or JSON (migration path). Use `corevital migrate` to import legacy JSON traces into the SQLite database.

Report structure:
```json
{
  "schema_version": "0.4.0",
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
      "logits_summary": { "entropy": 8.12, "top_k_margin": 0.34, "topk_mass": 0.72, "topk_probs": [...] },
      "layers": [
        {
          "layer_index": 0,
          "hidden_summary": { "mean": 0.001, "std": 0.98, ... },
          "attention_summary": { "entropy_mean": 2.31, "entropy_mean_normalized": 0.42, ... },
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
  "health_flags": { "nan_detected": false, "high_entropy_steps": 0, ... },
  "extensions": {
    "risk": { "risk_score": 0.2, "risk_factors": [], "blamed_layers": [], "blamed_layers_flat": [] },
    "compound_signals": [],
    "early_warning": { "failure_risk": 0.0, "warning_signals": [] },
    "fingerprint": { "vector": [...], "version": 2, "prompt_hash": "..." },
    "narrative": { "summary": "..." },
    "calibration": { "divergence_score": 0.1, "anomalies": [], "baseline_model_id": "...", "baseline_num_runs": 50 }
  },
  "encoder_layers": [
    {
      "layer_index": 0,
      "hidden_summary": { "mean": 0.5, "std": 1.2, ... },
      "attention_summary": { "entropy_mean": 2.85, ... },
      "cross_attention": null,
      "extensions": {}
    }
  ]
}
```

### Key Components

- **prompt**: Contains the input prompt text, number of tokens and token IDs
- **generated**: Contains the generated output text, number of tokens and token IDs
- **timeline**: Per-token trace covering generated tokens. Each step contains decoder layer summaries.
- **hidden_summary**: Mean, std, L2 norm, max abs value, random projection sketch; `clipped` is true when values were clamped for numerical stability. When clipping occurs, includes `clip_fraction` (fraction of elements clipped) and `clip_max_before` (max abs before clamping)
- **attention_summary**: Entropy statistics (entropy_mean, entropy_mean_normalized, entropy_min), concentration metrics (concentration_max), and per-head max weight (max_weight_per_head) to spot specialist or failing heads. `entropy_mean_normalized` is entropy / log(K) in [0, 1] for consistent cross-sequence-length comparison
  - For decoder layers (in timeline): Contains decoder self-attention
  - For encoder layers (in encoder_layers): Contains encoder self-attention
  - This field ALWAYS contains self-attention, regardless of model type
- **cross_attention**: (Seq2Seq only) Cross-attention statistics showing how the decoder attends to encoder outputs at each generation step. Only used in decoder layers (in timeline). Always null for CausalLM models and for encoder layers.
- **encoder_layers**: (Seq2Seq only) Encoder layer summaries computed once at the start of generation. Each layer includes `hidden_summary` and `attention_summary` (encoder self-attention). Always null for CausalLM models.
- **logits_summary**: Entropy, top-K margin, top-K mass (`topk_mass`), top-K probs (`topk_probs`), perplexity, and surprisal. Supports `entropy_mode: "full"` (default) or `"topk_approx"` for approximate entropy from top-K logits
- **model.dtype**: Model dtype as string. `Optional[str]` — may be `null` or `"quantized_unknown"` when dtype cannot be definitively detected for quantized models.
- **model.revision**: Model commit hash/revision extracted from model config
- **model.quantization**: Quantization information (enabled: bool, method: "4-bit"|"8-bit"|null). The dtype field shows quantized dtypes (int8, uint8) when detectable, or `"quantized_unknown"` otherwise.
- **health_flags**: Aggregated flags (nan_detected, attention_collapse_detected, high_entropy_steps, repetition_loop_detected, mid_layer_anomaly_detected, etc.).
- **extensions**: Risk (`risk_score`, `risk_factors`, `blamed_layers` (enriched List[dict] with layer/reasons/severity), `blamed_layers_flat` (List[int] for backward compat)), compound_signals (List[{name, description, severity, evidence}]), early_warning (`failure_risk`, `warning_signals`), fingerprint (`vector` (25 elements, v2), `version`, `prompt_hash`), narrative, calibration (when profile configured: `divergence_score`, `anomalies`, `baseline_model_id`, `baseline_num_runs`), metric_consistency (when DEBUG logging), RAG, performance.

### Trace File Sizes

Trace files are saved in **compact JSON format** (no indentation, minimal separators) for smaller file sizes. Typical file sizes:

- **Small models (GPT-2, 12 layers, ~15 steps):** 200-600 KB on disk
- **Medium models (Llama-3.1-8B, 32 layers, ~50 steps):** 1.3-1.5 MB on disk (compact JSON)
- **Large models (more layers/steps):** Up to ~5 MB depending on attention sparsity

**Optimizations applied:**
- **Compact JSON:** No indentation, minimal separators (`separators=(",", ":")`) — saves ~63% vs pretty-printed
- **Exclude None fields:** Optional fields set to `None` are omitted from JSON — saves ~19 KB per file
- **Sparse attention storage:** Only attention weights above threshold are stored — saves ~680× vs naive approach

**Note:** If you want to inspect the JSON in a formatted way, use the dashboard's "Raw JSON" section which provides a toggle for pretty-printing. For even smaller files, consider gzip compression (typically achieves 70-80% reduction).

The storage is dominated by **sparse attention profiles** from prompt telemetry (Phase-1b), which store only attention weights above a threshold. Typical storage: **0.5-5 MB** depending on attention patterns (documented in `docs/Phase1 metrics analysis.md`).

### Performance Monitoring (`--perf`)

The `--perf` flag enables performance monitoring with three modes:

**Summary Mode** (`--perf` or `--perf summary`):
- Adds `performance` extension to the main trace JSON
- Shows total wall time and breakdown by parent operations
- Tracks: config_load, setup_logging, model_load, tokenize, model_inference, report_build

**Detailed Mode** (`--perf detailed`):
- Everything in summary mode, plus:
- Embeds a `detailed_breakdown` in the main trace JSON (nested operations, per-step stats)
- Useful for identifying specific bottlenecks

**Strict Mode** (`--perf strict`):
- Everything in detailed mode, plus:
- Runs warmup before measurements to stabilize GPU timing
- Runs baseline (uninstrumented) inference for comparison
- Reports original model load time (before caching)
- Calculates inference overhead and CoreVital overhead percentages

Example performance output in summary (detailed/strict modes add `detailed_breakdown` to the same object):
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
      "unaccounted_time": {"ms": 2.0, "pct": 0.08}
    }
  }
}
```

### Model Compatibility

See [docs/model-compatibility.md](docs/model-compatibility.md) for tested models, attention capture details, quantization notes, and Seq2Seq support.

## Architecture

CoreVital instruments LLM inference via HF-output-based capture: the model is configured to return internal tensors (hidden states, attention weights, logits) on each forward pass; these are then reduced to lightweight summary statistics. The architecture is designed for production use with minimal overhead and storage requirements.

### System Overview

- **Instrumentation Layer**: HF output flags (output_hidden_states, output_attentions) supply tensors during model forward pass
- **Summary Computation**: Lightweight statistics (mean, std, entropy, norms) computed in-memory
- **Report Building**: Structured JSON reports with schema versioning
- **Pluggable Sinks**: Multiple persistence backends (SQLite, local files, Datadog, Prometheus, OTLP)

**Architecture Diagrams:**
- [CoreVital Overview](docs/mermaid/corevital-overview.mmd) — What CoreVital does and why (start here)
- [Module Architecture](docs/mermaid/module-architecture.mmd) — Codebase layout and module dependencies
- [Data Flow](docs/mermaid/metrics-data-flow.mmd) — How data flows from model inference to reports
- [Step Processor Lifecycle](docs/mermaid/step-processor-lifecycle.mmd) — Raw tensors → StepSummary (scalars only)
- [Computation Pipeline](docs/mermaid/phase-1-computation-pipeline.mmd) — Full metrics computation flow
- [Schema Structure](docs/mermaid/schema-v03-structure.mmd) — Report schema v0.4.0 organization
- [Metrics Dependency Chain](docs/mermaid/metrics-dependency-chain.mmd) — How metrics depend on each other
- [Signal Interpretation](docs/mermaid/metrics-signal-interpretation.mmd) — What each metric means and when to act
- [Risk Score Computation](docs/mermaid/risk-score-computation.mmd) — How the composite 0–1 risk score is built
- [Compound Signal Detection](docs/mermaid/compound-signals-detection.mmd) — The 5 multi-metric failure patterns
- [Early Warning Detectors](docs/mermaid/early-warning-detectors.mmd) — The 5 trend detectors
- [Calibration Workflow](docs/mermaid/calibration-workflow.mmd) — Building and using calibration profiles
- [Extensions Computation](docs/mermaid/extensions-computation-flow.mmd) — How all extensions are computed in report_builder
- [Performance Monitoring](docs/mermaid/operations-hierarchy.mmd) — Operation timing hierarchy
- [Execution Flow](docs/mermaid/operations-flow-sequential.mmd) — Sequential execution order

**Production Deployment:** See [Production Deployment Guide](docs/production-deployment.md) for sampling strategies, database setup, metrics export, and alerting.

**Integration Examples:** See [Integration Examples](docs/integration-examples.md) for Flask, FastAPI, and production patterns.

**v0.4.0 launch:** See [Technical launch (v0.4.0)](docs/v0.4.0-launch.md) for an overview of features and design.

**Metrics Interpretation:** See [Metrics Interpretation Guide](docs/metrics-interpretation.md) for per-metric definitions, research citations (Shannon entropy, Voita et al. attention, Attention Basin, etc.), threshold tables, and example scenarios.

**Visual Examples:** See [Visual Examples Guide](docs/visual-examples.md) for interpreting metrics and identifying healthy vs unhealthy runs. The web dashboard (see [Visualizing your Data](#visualizing-your-data-two-viewing-paths)) includes Prompt Analysis (layer transformations, prompt surprisals, sparse attention with a layers×heads basin heatmap, and an Attention Explorer for querying attention to/from tokens), timeline tabs (entropy, perplexity, surprisal, top-K margin, top-K mass), entropy-vs-position chart, and colored output by uncertainty. Timeline charts show missing values as gaps rather than as zero so that absent data is not mistaken for maximum confidence.

### Visualizing your Data (Two Viewing Paths)

CoreVital uses a **decoupled, cloud-native architecture**. How you view your data depends on your use case:

**Path A — Open-Source / Individual Developers**

- View rich trace files (timeline charts, attention heatmaps, Compare view) in the **hosted web dashboard**: [https://main.d2maxwaq575qed.amplifyapp.com](https://main.d2maxwaq575qed.amplifyapp.com). The dashboard is a separate [React app repo](https://github.com/Joe-b-20/corevital-dashboard); it opens in **Demo mode** with sample traces from that repo (`public/demo/`), so you can try it with no install.
- To view your **local SQLite database**, run the local API in your terminal:
  ```bash
  pip install "CoreVital[serve]"
  corevital serve
  ```
  Then open the dashboard in your browser and click **Connect**. The website talks only to the server running on your machine — **your data never leaves your computer**.

**Path B — Enterprise Teams**

- Enterprise users do **not** need the React dashboard. Configure CoreVital’s **native sinks** (Datadog, Prometheus, Weights & Biases) in your `config.yaml` (or via CLI flags) to send metrics and reports directly to the observability tools your company already uses.
- Use `--sink datadog`, `--sink prometheus`, or `--sink wandb` so traces and risk scores flow into your existing dashboards and alerting.

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
- **SQLiteSink** (default): One SQLite DB per run directory; supports `list_traces()`, filters by model_id/prompt_hash. Use `--sink sqlite`.
- **LocalFileSink**: Write JSON to local filesystem (`--sink local`).
- **DatadogSink**: Send metrics/events to Datadog (`--sink datadog`; requires `DD_API_KEY` or `--datadog_api_key`).
- **PrometheusSink**: Expose `/metrics` for scraping (`--sink prometheus`; `--prometheus_port`).
- **WandBSink**: Log metrics, report artifact, and optional basin heatmap to Weights & Biases (`--sink wandb`; `--wandb_project`, `--wandb_entity`, or `WANDB_PROJECT`/`WANDB_ENTITY`).
- **HTTPSink**: POST JSON to remote endpoint.

### Configuration

Override defaults via `configs/default.yaml` or environment variables. Per-model detection thresholds live in `configs/model_profiles/` (see [model compatibility](docs/model-compatibility.md#per-model-threshold-profiles)).
```bash
export COREVITAL_DEVICE=cuda
export COREVITAL_SEED=123
```

## Performance

See [Measured Overhead](#measured-overhead) for real numbers. Key optimization levers:

- **`--capture summary`**: Skips per-layer data; overhead drops to under 5%.
- **`--capture on_risk`**: Summary by default, full trace only when risk exceeds threshold.
- **Sampling**: Instrument a subset of requests (1% random, every N-th).
- **Skip prompt telemetry**: `--no-prompt-telemetry` removes the extra forward pass.

See [Design Journey](docs/design-journey.md) for the performance monitoring design rationale.

## Comparison with Alternatives

CoreVital focuses on **internal inference health monitoring**—instrumenting the model's forward pass to detect issues like repetition loops, attention collapse, and numerical anomalies. Here's how it compares:

| Tool | Focus | Internal Instrumentation | Health Signals | Storage Model |
|------|-------|-------------------------|----------------|---------------|
| **CoreVital** | Internal inference health | Yes (HF-output-based capture) | Entropy, repetition, attention collapse, NaN/Inf | Summary-only (no raw tensors) |
| **LangSmith** | LLM application tracing | No (API-level only) | Output quality scores | Full traces (prompts/responses) |
| **OpenLIT / Langtrace** | LLM observability | No (OpenTelemetry at API level) | Latency, cost, token counts | Request/response traces |
| **Aporia** | AI observability & guardrails | No (application-level) | Output guardrails, drift | Application metrics |
| **Langfuse** | LLM tracing & evals | No (API-level tracing) | Eval scores on outputs | Full traces |

**CoreVital's differentiator:** Only CoreVital instruments **inside** the model's forward pass to capture hidden states, attention patterns, and logits during generation. This enables detection of issues that manifest internally (e.g., attention collapse, repetition loops) before they appear in outputs.

**When to use CoreVital:**
- You're running self-hosted/open-source models (Hugging Face transformers)
- You need to debug why models fail (repetition, confusion, numerical issues)
- You want to monitor model health in production without storing raw activations
- You need to compare models or track degradation over time

**When to use alternatives:**
- You're using API-based LLMs (OpenAI, Anthropic) -- use LangSmith/OpenLIT
- You need application-level tracing -- use Langfuse/Langtrace
- You need output guardrails -- use Aporia

See [Competitive Landscape](docs/competitive-landscape.md) for detailed analysis.

## Use Cases

See [Case Studies](docs/case-studies/) for real-world examples of CoreVital in production.

### 1. Debugging Model Failures

**Problem:** Model produces repetitive or nonsensical output, but you don't know why.

**Solution:** Run CoreVital to see:
- **Repetition loop detected:** Last-layer hidden states became nearly identical -- model is stuck
- **High entropy steps:** Model was confused at specific tokens -- check input context
- **Attention collapse:** Some heads put all weight on one token -- possible training issue
- **NaN/Inf detected:** Numerical instability -- check inputs or model weights

**Example:**
```bash
corevital run --model meta-llama/Llama-3.1-8B \
  --prompt "Explain quantum computing" \
  --max_new_tokens 100 \
  --perf detailed
# View traces via corevital serve and the web dashboard, or corevital compare
```

### 2. Production Monitoring

**Problem:** Monitor model health in production without storing massive tensor dumps.

**Solution:** Use `--capture summary` or `--capture on_risk` to get lightweight health signals:
- Risk score per run
- Health flags (NaN, repetition, attention collapse)
- Time series (entropy, perplexity) for trend analysis

**Example:**
```python
from CoreVital import CoreVitalMonitor

monitor = CoreVitalMonitor(capture_mode="summary")
monitor.run("gpt2", user_prompt, max_new_tokens=50)

if monitor.should_intervene():
    # Resample or fallback to another model
    pass
```

### 3. Model Comparison

**Problem:** Compare how different models or configurations perform on the same prompts.

**Solution:** Run CoreVital on multiple models, then use the dashboard's Compare view or `corevital compare`:
- Risk scores across models
- Health flags comparison
- Entropy/perplexity trends
- Performance overhead

**Example:**
```bash
# Run on model A
corevital run --model gpt2 --prompt "..." --sink sqlite

# Run on model B
corevital run --model meta-llama/Llama-3.2-1B --prompt "..." --sink sqlite

# Compare
corevital compare --db runs/corevital.db
# Or use the web dashboard Compare view (Path A)
```

### 4. Research & Analysis

**Problem:** Analyze model behavior across different prompts or configurations.

**Solution:** Use CoreVital's detailed reports to study:
- Attention patterns (sparse attention profiles)
- Layer transformations (how representations change)
- Entropy profiles (confidence over generation)
- Basin scores (attention focus on prompt middle)

**Example:** See [Phase-1 Metrics Analysis](docs/Phase1%20metrics%20analysis.md) for research-backed metric interpretations.

## Glossary

**Hidden States:** Internal representations computed by each transformer layer. Shape: `(batch, sequence_length, hidden_dim)`. CoreVital summarizes these as mean, std, L2 norm, max absolute value.

**Attention Patterns:** Weights showing which tokens each attention head focuses on. Shape: `(batch, heads, seq_len, seq_len)`. CoreVital computes entropy statistics (how spread out attention is) and detects collapse (heads focusing on one token).

**Logits:** Raw model outputs before softmax, representing scores for each token in the vocabulary. CoreVital computes entropy (uncertainty), perplexity, surprisal, and top-k margins.

**Entropy:** Measure of uncertainty in probability distributions. Low entropy = confident (peaked distribution), high entropy = uncertain (flat distribution). Range: 0 to ~16-17 for typical LLMs.

**Perplexity:** Exponential of entropy. Roughly "how many tokens the model is choosing between." Low perplexity (1-4) = confident, high perplexity (>16) = very uncertain.

**Surprisal:** Negative log probability of the actual token. High surprisal = model was surprised by the token it produced.

**Attention Collapse:** When an attention head puts almost all weight (>0.95) on a single token. Can indicate training issues or model degradation.

**Repetition Loop:** When the model gets stuck repeating the same tokens. Detected by comparing last-layer hidden states across steps (high cosine similarity indicates repetition).

**Risk Score:** Single number (0-1) summarizing overall run health. Computed from boolean health flags (hard ceilings), continuous timeline metrics (entropy, margin, surprisal, top-K mass), and compound signals (multi-metric patterns). Higher = more likely to produce poor output. See [Risk Calibration](docs/risk-calibration.md).

**Health Flags:** Boolean indicators for specific issues: NaN/Inf detected, attention collapse, high entropy steps, repetition loop, mid-layer anomaly.

**Prompt Telemetry:** Extra forward pass over the prompt (before generation) to analyze how the model processes input. Captures layer transformations, attention patterns, and surprisal.

**Cross-Attention:** (Seq2Seq only) How the decoder attends to encoder outputs. Shows which parts of the input the model "listens to" during generation.

**Top-K mass:** Sum of probabilities of the top-K tokens (default K=10). High top-K mass means most probability is concentrated on a small set of candidates; low values indicate spread across many tokens. Formerly exposed as `voter_agreement` (deprecated alias still written for backward compatibility).

**Compound Signals:** Multi-metric failure patterns detected from timeline data. Five patterns: context loss (high entropy + low basin), confident confusion (high entropy + high margin), degenerating generation (rising entropy + declining margin), attention bottleneck (high collapse + elevated entropy), confident repetition risk (low entropy + very high mass).

**Early Warning:** Trend-based predictors that detect degradation patterns before they trip health-flag thresholds. Signals: entropy acceleration, margin collapse/decline, surprisal volatility, entropy-margin divergence.

**Calibration Profile:** Empirical baseline built from known-healthy runs. Contains per-metric distributions (entropy, margin, surprisal per step; L2 norm, attention entropy per layer). Production traces are scored by statistical divergence (z-scores) from the baseline.

**Divergence Score:** 0–1 value measuring how far a trace deviates from a calibration profile baseline (mean |z-score| / 6, capped). Low = similar to baseline, high = anomalous.

**Basin Scores:** Measure of how much each attention head focuses on nearby (local) tokens versus distant ones. High basin score = head attends mostly to neighbors (local pattern). Low basin score = head attends broadly across the sequence (global pattern). Useful for detecting degenerate attention that ignores positional structure.

**Layer Transformations:** Geometric change between consecutive layers' hidden state representations. Computed as the ratio of L2 norms (magnitude change) and cosine similarity (direction change) between layer N and layer N+1. Large magnitude jumps or sharp direction changes can indicate layers where the model "makes decisions."

**Mid-Layer Anomaly:** L2 norm explosion detected specifically in the middle third of model layers. Research (e.g., Meng et al.) suggests middle layers are where factual recall and "truth processing" occur. Anomalies here may indicate the model is struggling with factual content.

**Concentration:** The maximum attention weight any single token receives from a query in an attention head. High concentration (>0.5) means the head is "focused" on specific tokens. Extremely high concentration (>0.95) indicates potential attention collapse.

**Fingerprint:** A compact 25-element numeric vector (v2) summarizing a run's behavior: entropy/margin/surprisal/agreement profiles with temporal slopes, cross-metric correlations, risk score, and health flag booleans. Used for clustering similar runs and detecting patterns. Includes a SHA256 prompt hash for exact duplicate detection. Use `is_legacy_fingerprint()` to detect old 9-element vectors.

**L2 Norm:** The Euclidean length (magnitude) of a vector. In CoreVital, L2 norm of hidden states tracks how "large" representations become across layers and steps. Sudden L2 norm growth ("explosion") can indicate numerical instability.

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
  - `models/`: Model loading, management, and `ModelCapabilities` registry (includes attention availability probing)
  - `instrumentation/`: Modular instrumentation pipeline
    - `collector.py` — orchestrator (~250 lines)
    - `causal_lm.py` — CausalLM generation path
    - `seq2seq.py` — Seq2Seq manual decoder loop
    - `step_processor.py` — per-step tensor → `StepSummary` lifecycle
    - `baselines.py` — warmup, baseline, prompt forward, shared helpers
    - `hooks.py` — unused stub (documented: HF output flags, not PyTorch hooks)
    - `summaries/` — subpackage: `logits.py`, `attention.py`, `hidden_states.py`, `utils.py`
  - `reporting/`: Schema (v0.4.0), validation (including metric consistency), and report building
  - `sinks/`: Persistence backends (SQLite, LocalFile, HTTP, Datadog, Prometheus, W&B)
  - `risk.py`: Risk scoring and enriched layer blame
  - `compound_signals.py`: Multi-metric failure pattern detection
  - `early_warning.py`: Trend-based degradation prediction
  - `narrative.py`: Human-readable report summaries
  - `fingerprint.py`: 25-element fingerprint vectors (v2) and prompt hashing
  - `calibration.py`: Data-driven baseline profiling and divergence scoring
  - `calibration_risk.py`: ECE computation and Platt scaling
  - `config.py`: Configuration with model profiles, entropy mode, calibration options
  - `utils/`: Shared utilities
- `.github/workflows/`: CI/CD pipeline (test.yaml)
- `configs/`: YAML configuration files and `model_profiles/` (gpt2, llama, mistral, mixtral, qwen2, phi3, default)
- `runs/`: Default output directory for trace artifacts
- `tests/`: Test suite (436 tests)

## Roadmap

Phases 0--2 and the local API + dashboard workflow are fully implemented and tested. Phases 3--8 have working implementations that are iterative. See [Design Journey](docs/design-journey.md) for architectural decisions and trade-offs.

| Phase | Focus | Key deliverables |
|-------|-------|-----------------|
| Phase 0 | HF instrumentation | Hidden states, attention, logits capture; summaries; Seq2Seq; quantization |
| Phase 0.5 | Hardening | Extensions model, encoder/decoder separation, memory optimizations |
| Phase 0.75 | Performance monitoring | `--perf` (summary/detailed/strict); operation timing; warmup/baseline |
| Pre-Phase-1 | Cleanup and tooling | Schema v0.4.0, `ModelCapabilities` registry, CI/CD, dtype detection |
| Phase 1 | Metrics and telemetry | Enhanced metrics, prompt telemetry, health flags, dashboard, SQLite default |
| Phase 2 | Risk scoring | `compute_risk_score`, `compute_layer_blame`; `on_risk` capture trigger |
| Phase 3 | Fingerprinting | `compute_fingerprint_vector`, `compute_prompt_hash` |
| Phase 4 | Early warning | `compute_early_warning`; streaming API |
| Phase 5 | Health-aware decoding | `should_intervene()`; configurable risk threshold |
| Phase 6 | Cross-model comparison | Dashboard Compare view; `corevital compare`; SQLite filters |
| Phase 7 | Narratives | Template-based `build_narrative` |
| Phase 8 | Packaging | Dashboard polish; Library API (`CoreVitalMonitor`); OpenTelemetry integration |

## Known Limitations & What's Next

CoreVital v0.4.0 is a working implementation of internal inference monitoring for Hugging Face transformers. The following are known limitations and planned improvements — contributions welcome.

### Serving Framework Support

CoreVital currently instruments models loaded via Hugging Face `transformers` (`AutoModelForCausalLM`, `AutoModelForSeq2SeqLM`). It does **not** yet support optimized serving frameworks:

- **vLLM** — Uses PagedAttention and custom CUDA kernels that bypass standard HF output flags. Integration would require vLLM's `SamplerOutput` hooks or a custom sampler plugin.
- **TGI (Text Generation Inference)** — Rust/Python hybrid server; would need a middleware layer that captures activations before the optimized kernels.
- **llama.cpp / GGUF** — C++ inference with quantized formats outside PyTorch; out of scope for the current HF-output-based approach.

**Path forward:** Abstract the instrumentation interface so backends other than HF `transformers` can plug in. vLLM's `Logprob` output and custom `LogitsProcessor` are likely starting points.

### Real-Time Intervention

The `should_intervene()` API operates **post-run**. For **Seq2Seq models** (T5, BART), you can halt generation mid-stream using a **step callback**: pass `step_callback=(step_index, generated_ids, last_layer_hidden_buffer, last_logits) -> bool` to `InstrumentationCollector.run(prompt, step_callback=...)`. If the callback returns `True`, generation stops. Use the buffer with `detect_repetition_loop()` (from `CoreVital.instrumentation.summaries`) to stop on repetition. CausalLM uses `model.generate()` and does not support per-step callbacks yet.

### Risk Threshold Calibration

Risk scores and thresholds use heuristics overridable per model via [per-model threshold profiles](docs/model-compatibility.md#per-model-threshold-profiles). See [Risk and threshold calibration](docs/risk-calibration.md) for defaults and planned calibration (ECE, benchmark validation).

### Decoding Strategies

The manual decoder loop supports **greedy decoding**, **sampling** (temperature, top-k, top-p), and **beam search** (CausalLM only). Use `--num_beams N` (N > 1) and optionally `--early_stopping` to enable beam search; the timeline reports metrics for the best beam.

### GPU Overhead Benchmarks

The [Measured Overhead](#measured-overhead) table reports numbers for GPT-2 on CPU. For production-scale GPU models and how to measure, see [GPU benchmarks](docs/gpu-benchmarks.md). Overhead is typically dominated by `output_attentions=True` (attention weight materialization) rather than CoreVital's summary computation.

## Requirements

- Python 3.12+
- PyTorch
- Transformers (Hugging Face)
- PyYAML
- Pydantic

**Optional extras** (e.g. `pip install "CoreVital[serve]"`):
- `quantization`: bitsandbytes + accelerate for 4-bit / 8-bit inference (requires CUDA)
- `serve`: FastAPI + uvicorn for `corevital serve` (local API for the hosted React dashboard)
- `datasette`: Datasette + datasette-dashboards for SQLite-based dashboards (see [docs/datasette/README.md](docs/datasette/README.md))
- `datadog`: Datadog API client for `--sink datadog`
- `prometheus`: Prometheus client for `--sink prometheus`
- `wandb`: Weights & Biases for `--sink wandb`
- `otel`: OpenTelemetry SDK + OTLP exporter for `--export-otel`
- `all`: everything above

**Dev dependencies** (`pip install "CoreVital[dev]"`):
- pytest
- ruff
- mypy

**GitHub topics (for repo description):** `llm-observability`, `inference-monitoring`, `model-health`, `ai-safety`.

## License

Apache 2.0

## Contributing

Contributions welcome! Please open an issue or PR.