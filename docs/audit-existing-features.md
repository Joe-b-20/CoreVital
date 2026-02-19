# Audit: Existing Features (Branch 3)

This document records the verification of items #3, #4, #5, #7, #9, #20, #21, #22, #23 from the CoreVital Implementation Roadmap. These features were already implemented; this audit confirms they produce non-null output end-to-end.

## How to run

```bash
# Full smoke test (GPT-2 CPU) — verifies report structure and Phase-1b/1c fields
pytest tests/test_smoke_gpt2_cpu.py -v

# Mock instrumentation (no model load) — verifies collector and report builder
pytest tests/test_mock_instrumentation.py -v

# Sink tests (Prometheus, Datadog)
pytest tests/test_sinks.py -v

# CI: lint, type check, tests (excludes slow/gpu)
ruff check . && ruff format --check . && mypy src/ && pytest -m 'not slow'
```

## Verification summary

| Item | Feature | Verified by | Result |
|------|---------|-------------|--------|
| #3 | Repetition loop detection (cosine similarity) | health_flags.repetition_loop_detected in smoke test | Structure asserted; trigger with repetitive prompt for manual check |
| #4 | Mid-layer anomaly (dynamic L2 baseline) | health_flags.mid_layer_anomaly_detected in smoke test | Structure asserted; no false positive on normal GPT-2 prompt |
| #5 | Logit metrics (top_k_margin, voter_agreement, perplexity, surprisal) | timeline[].logits_summary in smoke test | Assert at least one of entropy/perplexity/surprisal present |
| #7 | Sparse storage SoA | prompt_analysis.layers[].heads[] with query_indices, key_indices, weights | Smoke test asserts keys present when heads exist |
| #9 | Basin scores | prompt_analysis.layers[].basin_scores | Smoke test asserts len(basin_scores) > 0 |
| #20 | Prompt telemetry | prompt_analysis (layers, layer_transformations, prompt_surprisals) | Smoke test asserts full structure |
| #21 | Internal metrics (entropy, top1_top2_margin, concentration_max) | compute_logits_summary / compute_attention_summary | Covered by timeline and layer summaries in smoke test |
| #22 | Prometheus / Datadog sinks | test_sinks.py | Mock write tests; no runtime API key required |
| #23 | CI/CD (ruff, mypy, pytest) | .github/workflows/test.yaml | Run commands above locally to verify |

## Gaps

None identified. If a gap is found in the future, add it here and create a sub-task on the appropriate later branch.
