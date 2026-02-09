# Phase-0.75: Performance Monitoring — Design Journey

## What This Phase Set Out to Do

Phase-0.75 adds a `--perf` flag to CoreVital that answers a simple question: *how much overhead does the instrumentation framework itself add?* When you're wrapping every forward pass, slicing attention tensors, computing norms, building reports — how much of the total wall time is CoreVital, and how much is the model?

This document captures the decisions, discoveries, and fixes that shaped the feature from initial design through production merge.

---

## The Architecture

### Diagram-First Design

Before writing any code, we created two Mermaid diagrams that became the source of truth:

- **`operations-hierarchy.mmd`** — a tree showing every parent and child operation that CoreVital performs, from `config_load` down to individual `compute_hidden_summary` calls within report building.
- **`operations-flow-sequential.mmd`** — a left-to-right flowchart showing the exact execution order through `cli.py → collector.run() → builder.build() → sink.write()`.

These diagrams served a dual purpose: they documented what to time, and they acted as a verification contract. Every node in the hierarchy had to correspond to an actual `with monitor.operation("name")` call in the code. During review, we caught several discrepancies:

- Seq2Seq detection was incorrectly placed as a child of `model_inference` in the hierarchy. In the code, it runs between tokenization and inference — a sibling, not a child. Moved to its correct position under `total_wall_time`.
- Encoder output extraction (`encoder_hidden_states`, `encoder_attentions`) was nested under the wrong parent. Moved to a dedicated `EncExt` group under `model_inference`.
- `json_serialize` appeared as a standalone subgraph in the flow diagram, but in the code, serialization happens *inside* `sink.write()`. Merged it into the `sink_write` subgraph.

**Takeaway**: Diagrams are only useful if they're verified against the code. Treating them as living documentation that must pass review alongside the implementation prevented several monitoring blind spots.

### Three Modes, Progressive Detail

The `--perf` flag supports three modes, each building on the previous:

| Mode | What it adds | Use case |
|------|-------------|----------|
| `summary` | `extensions.performance` in the main trace JSON: total wall time, parent operation breakdown, unaccounted time | Quick overhead check |
| `detailed` | Separate `*_performance_detailed.json` with nested child operations and per-step statistics (min/max/avg across generation steps) | Identifying specific bottlenecks |
| `strict` | Warmup runs, uninstrumented baseline inference, original cold model load time, inference overhead %, CoreVital overhead % | Rigorous benchmarking |

The progressive design means `summary` mode adds negligible overhead itself (a few `time.perf_counter()` calls), while `strict` mode roughly doubles runtime by running warmup + baseline + instrumented inference.

### The `_op()` Helper Pattern

Rather than threading the monitor through every function signature, `cli.py` defines a local helper:

```python
def _op(name: str):
    """Wrap parent operations in monitor.operation() if enabled."""
    return monitor.operation(name) if monitor else nullcontext()
```

This lets every timed block use the same clean pattern:

```python
with _op("config_load"):
    config = Config.from_yaml(args.config) if args.config else Config.from_default()
    # ...
```

When `--perf` is not passed, `monitor` is `None`, `_op()` returns `nullcontext()`, and the code runs with zero overhead — no conditionals, no branches, no timing calls. The same pattern propagates into `collector.py` and `report_builder.py` for child operations.

### The `OperationTiming` Tree

The `PerformanceMonitor` uses a stack-based approach to build a timing tree:

```python
@contextmanager
def operation(self, name, **metadata):
    timing = OperationTiming(operation_name=name, metadata=metadata)
    if self.stack:
        self.stack[-1].children.append(timing)
    else:
        self.root_timings.append(timing)
    self.stack.append(timing)
    try:
        yield timing
    finally:
        self.stack.pop()
        timing.duration_ms = (time.perf_counter() - start) * 1000
```

Nested `with monitor.operation()` calls automatically form parent-child relationships. This gives you a tree like:

```
model_inference (8200ms)
├── generate_seq2seq_manual (7900ms)
│   ├── encoder_forward (2100ms)
│   └── decoder_loop (5800ms)
│       └── decoder_step (per_step: 7 steps, avg 54ms)
└── extract_generated_tokens (0.2ms)
```

---

## The Strict Mode Benchmarking Problem

Strict mode is where the real complexity lives. The goal is to compare instrumented inference against uninstrumented (baseline) inference to isolate CoreVital's overhead. This turns out to be surprisingly hard to do correctly.

### The Negative Overhead Discovery

After the initial implementation, we ran the tool against four real model configurations:

| Trace | Model | Mode | Total |
|-------|-------|------|-------|
| `44ef79d2` | Llama-3.1-8B (CausalLM) | detailed | 21.6s |
| `5c79188a` | Llama-3.1-8B (CausalLM) | strict | 35.6s |
| `e6d5dc65` | flan-t5-small (Seq2Seq) | detailed | 2.7s |
| `b530fe42` | flan-t5-small (Seq2Seq) | strict | 4.6s |

The Llama strict trace was clean — 1.22% inference overhead, plausible and positive. But the flan-t5-small strict trace showed:

```json
"baseline_ms": 570.14,
"instrumented_inference_ms": 318.55,
"inference_overhead_ms": -251.59,
"inference_overhead_pct": -44.13
```

The instrumented run — which does *more work* (extracting hidden states, attentions, computing summaries) — was somehow 44% *faster* than the bare baseline. That's not instrumentation overhead; that's a measurement artifact.

### Root Cause: Three Compounding Bugs

Investigation revealed three issues working together:

**1. Insufficient warmup (CPU cache effects)**

The original implementation ran a single warmup pass before the baseline. On CPU, this isn't enough to fully stabilize memory caches, branch predictors, and any JIT behavior. The baseline ran second (partially warm), while the instrumented run ran third (fully warm), giving it an unfair cache advantage.

*Fix*: Two rounds of warmup before measuring anything.

**2. No seed before baseline (non-deterministic token generation)**

With `do_sample=True` and `temperature=0.8` (the CLI defaults), the baseline and instrumented runs generate different token sequences because no random seed was set before the baseline. Different sequences mean different lengths, different computation paths, and incomparable timings.

*Fix*: Set `torch.manual_seed()` and `torch.cuda.manual_seed_all()` immediately before both the baseline and instrumented runs, ensuring identical token generation.

**3. Inconsistent sampling parameters (Seq2Seq baseline)**

The Seq2Seq baseline's manual sampling loop applied only temperature scaling, omitting the `top_k` and `top_p` filtering that the instrumented path uses. Even with identical seeds, different sampling strategies produce different tokens.

*Fix*: Added `top_k` and `top_p` filtering to `_run_baseline_seq2seq` to match the instrumented path exactly.

### The Lesson

Microbenchmarking on CPU is genuinely difficult. The execution order of code paths can matter more than the code itself due to caching effects. For CoreVital, this meant:

- Warmup must be sufficient (multiple rounds)
- Seeds must be set for both baseline and instrumented to ensure identical work
- Sampling parameters must match exactly between the two paths
- Even with all of this, CPU results should be interpreted with care — GPU benchmarking with proper `torch.cuda.synchronize()` would be more reliable

---

## The HTTP Sink Bug

After merging Phase-0.75, we discovered a silent data loss bug: when `--perf` is used with `--remote_sink http`, the HTTP payload never included `extensions.performance`.

### The Original Flow

```
1. sink.write(report)        ← report has empty extensions
2. monitor.mark_run_end()    ← computes total_wall_time
3. build perf summary
4. IF local_file: read JSON back, patch it, re-write  ← hack
```

The local file path worked only because of a read-patch-write hack that re-opened the file after the fact. The HTTP sink had no equivalent — the POST was already sent.

### The Fix

Restructure to inject performance data *before* the write:

```
1. monitor.mark_run_end()
2. build perf summary
3. report.extensions["performance"] = perf_summary
4. sink.write(report)         ← report is now complete
```

Both sinks now receive identical, complete data. The read-patch-write hack is eliminated entirely.

### The Tradeoff

`sink_write` can no longer be a timed parent operation. You can't include sink_write timing in the performance summary that is *inside* the report that sink_write is trying to serialize — it's a chicken-and-egg problem. In practice, sink_write was 3-63ms (0.07-0.29%) in all traces — pure I/O, not CoreVital processing. The `total_wall_time_ms` now covers `config_load` through `report_build`, which is the processing overhead users actually care about.

---

## What the Traces Revealed About CoreVital Itself

Analyzing real traces wasn't just about validating the monitor — it exposed genuine insights about the tool's own performance characteristics.

### report_build Is the Real Bottleneck (for Large Models)

On Llama-3.1-8B with 50 generation steps:

| Operation | Time | % of total |
|-----------|------|-----------|
| model_load | 13,466ms | 62.4% |
| model_inference | 4,320ms | 20.0% |
| **report_build** | **3,718ms** | **17.2%** |
| config_load + logging + tokenize | ~12ms | ~0.1% |

`report_build` alone took 3.7 seconds — nearly as long as inference itself. The culprit is `_build_timeline`, which iterates over all steps and all layers, computing `compute_hidden_summary` and `compute_attention_summary` for each. At 50 steps × 32 layers × 32 heads, that's 51,200 summary computations in Python.

On flan-t5-small (6 layers, 8 heads), `report_build` was only 101ms — negligible. The scaling is roughly **O(steps × layers × heads)**, which means it would grow significantly with longer generation or deeper models.

### Scaling Projections

Based on the trace data and code architecture, we projected how overhead would scale with real-world prompts (not a few words, not max 50 tokens):

| Concern | Short prompts (our traces) | Real-world (500+ tokens, 7B+) |
|---------|---------------------------|-------------------------------|
| Inference overhead % | ~2-5% | ~0.5-2% (improves — model dominates more) |
| report_build | ~50-100ms | ~1-5s |
| JSON output size | ~100-300KB | ~5-50MB |
| Memory (Python heap) | Negligible | Potentially 100s of MB |

The *percentage* overhead on inference actually gets better with scale, because the model's own forward pass grows much faster than the extraction work. But *post-inference* costs (report building, serialization) and *memory* (holding per-step summaries for hundreds of steps across dozens of layers) are the natural pressure points.

Two mitigations were identified for future phases if needed:
1. **Streaming summaries to disk** — write per-step data incrementally instead of accumulating in memory
2. **Configurable extraction granularity** — skip head-level attention detail, or sample every N-th step

---

## Decisions Log

| Decision | Rationale |
|----------|-----------|
| Context manager pattern (`with monitor.operation()`) | Clean nesting, automatic timing on exit, zero overhead when disabled via `nullcontext()` |
| `time.perf_counter()` over `time.time()` | Monotonic, high resolution, not affected by system clock adjustments |
| Store summaries (norms, entropy) not raw tensors | Memory bounded — raw attention tensors for 500 steps × 32 layers would be gigabytes |
| Three progressive modes, not one | summary for CI pipelines, detailed for debugging, strict for benchmarking — different users need different levels |
| Two warmup rounds in strict mode | One round wasn't sufficient for CPU cache stabilization, as proven by the negative overhead traces |
| Seed before both baseline and instrumented | Without this, `do_sample=True` produces different token sequences, making timing comparison meaningless |
| Remove `sink_write` from timed operations | Chicken-and-egg: can't include sink timing in the data that the sink serializes. Acceptable loss at 0.07-0.29% |
| Inject perf data into Report object, not post-patch files | Clean, sink-agnostic, eliminates read-patch-write hack, works for both local and HTTP |
| `OperationTiming` dataclass with `_samples` list | Supports per-step statistics (min/max/avg) for repeated operations like decoder steps without changing the tree structure |
| Detailed breakdown as separate JSON file | Keeps the main trace lean for summary consumers; detailed data available on demand |

---

## Timeline

| Date | Event |
|------|-------|
| 2026-02-04 | Initial implementation: `PerformanceMonitor`, `OperationTiming`, CLI integration, three modes, test suite |
| 2026-02-05 | Simplified strict mode output: removed origin labeling, made `baseline_ms` the single source for raw inference timing |
| 2026-02-05 | Pre-merge review: fixed diagram inaccuracies, dead code in `build_summary_dict`, unrealistic test values, redundant timing calculations, added changelogs to all modified files |
| 2026-02-05 | Trace analysis (4 real runs): discovered negative overhead in Seq2Seq strict mode |
| 2026-02-05 | Three-part fix for strict mode: double warmup, seed before baseline, align Seq2Seq sampling parameters |
| 2026-02-05 | Merged to main as PR #4 |
| 2026-02-06 | Discovered HTTP sink silent data loss: `extensions.performance` missing from POST payloads |
| 2026-02-06 | Fixed by restructuring injection to happen before `sink.write()`, removing read-patch-write hack |
| 2026-02-06 | Merged to main as PR #5 |

---

## Files Touched

| File | Role in Phase-0.75 |
|------|-------------------|
| `src/CoreVital/instrumentation/performance.py` | **New** — `PerformanceMonitor` class and `OperationTiming` dataclass |
| `src/CoreVital/cli.py` | Orchestration: `--perf` argument parsing, `_op()` helper, perf injection, detailed file writing |
| `src/CoreVital/instrumentation/collector.py` | Parent + child operation wrapping for model_load through model_inference; strict mode warmup/baseline |
| `src/CoreVital/config.py` | `PerformanceConfig` added to root `Config` |
| `src/CoreVital/reporting/report_builder.py` | Child operation timing within report_build |
| `src/CoreVital/instrumentation/__init__.py` | Exported `PerformanceMonitor` and `OperationTiming` |
| `src/CoreVital/models/hf_loader.py` | Optional `monitor` parameter for child operation timing during model loading |
| `src/CoreVital/sinks/local_file.py` | Comment updates reflecting injection flow change |
| `src/CoreVital/sinks/http_sink.py` | Comment updates reflecting injection flow change |
| `src/CoreVital/utils/serialization.py` | Minor updates for extensions serialization |
| `docs/mermaid/operations-hierarchy.mmd` | **New** — verified operation tree |
| `docs/mermaid/operations-flow-sequential.mmd` | **New** — verified sequential flow |
| `tests/test_performance.py` | **New** — 10 tests covering summary, detailed, strict, context manager, integration, serialization |
| `README.md` | Performance monitoring documentation |
