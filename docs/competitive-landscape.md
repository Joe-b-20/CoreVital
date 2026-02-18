# CoreVital: Competitive Landscape

How CoreVital differs from existing LLM observability products and research.

---

## 1. LLM observability products (application layer)

| Player | What they do | What they don't do |
|--------|----------------|-------------------|
| **Langfuse** (open-source; acquired by ClickHouse) | Traces prompts, responses, token usage, latency, cost; OpenTelemetry backend; eval scores on outputs | No instrumentation inside the model forward pass |
| **Openlayer** ($14.5M Series A) | Tracing, cost, latency, hallucination/toxicity scoring, guardrails monitoring | Post-hoc scoring on text; no internal activations |
| **OpenLIT, Langtrace** | OpenTelemetry-based tracing across LLM providers and vector DBs | Request/response and orchestration level only |
| **Elastic, Observe** | LLM observability as part of broader APM | API- and app-level; no model-internal signals |

**Takeaway:** These tools sit on **calls to** the model (or orchestration). They do **not** hook into the model's internal computation (hidden states, attention weights, logits) during generation. CoreVital operates **inside** the inference loop.

## 2. Research (ideas, not shipped products)

- **ERGO, Entropy Sentinel:** Use entropy over next-token distributions for drift/accuracy estimation; scalar summaries per step or per response. Research frameworks, not production observability platforms.
- **SpecRA:** Repetition detection via FFT on token sequences. Different signal (text-level) than CoreVital's hidden-state repetition loop.
- **Attention collapse / attention sinks:** Mechanistic interpretability work; no product that surfaces "collapsed head count" or "attention collapse" as a production health signal.
- **LLMSafeGuard, real-time safeguarding:** Intervene during decoding; not a general "health + risk score + dashboard" observability layer.
- **Entropy-Lens, AVSS, cumulant expansion:** Compute scalar or low-dimensional statistics (entropy, variance-sparsity) **for analysis**. Not designed as the storage model for a production monitoring system.

**Takeaway:** Papers use entropy, repetition, and sometimes attention statistics. No one has turned "internal inference health as summary-only, production-grade observability" into a single, shippable product that cloud providers can bundle.

## 3. CoreVital's differentiation

- **Internal inference instrumentation:** Hooks into the model forward pass (Hugging Face / PyTorch); captures hidden states, attention, and logits **during** generation.
- **Health signals:** Repetition loop (last-layer hidden-state similarity), high-entropy steps (logits entropy > threshold), attention collapse (collapsed/focused head counts), mid-layer anomaly (L2 norms, NaN/Inf), plus NaN/Inf in tensors.
- **Single risk score + flags:** One run-level risk score and health flags for alerting and comparison.
- **Summary-only storage:** No raw tensors in reports; only lightweight summaries. Enables practical storage and long retention.

**Positioning:** "Internal inference health for self-hosted/open-source LLMs -- entropy, repetition, attention collapse, anomalies -- without storing raw activations."

---

## 4. Tensor summarization: design differentiator

### The problem with raw tensors

- **Hidden states:** e.g. `(batch, seq_len, hidden_dim)` -- millions of floats per step; full timeline x layers is prohibitive for production.
- **Attention:** `(batch, heads, seq_len, seq_len)` -- same scaling issue.
- **Logits:** `(batch, seq_len, vocab_size)` -- very large.

Storing raw tensors for every step and layer is not practical for cost, retention, or compliance. Researchers often dump activations for offline analysis; that's not a production observability design.

### What CoreVital does instead

**Never persist raw tensors.** For each step and (when stored) each layer we only compute and store:

- **Hidden states:** Scalar summaries per layer: `mean`, `std`, `l2_norm_mean`, `max_abs`; optional low-dim sketch (e.g. random projection). No full tensor.
- **Attention:** Scalar summaries per layer: `entropy_mean`, `entropy_min`, `entropy_max`, `concentration_max`, `concentration_min`, `collapsed_head_count`, `focused_head_count`. No raw attention matrix.
- **Logits:** Per-step scalars: `entropy`, `perplexity`, `surprisal`, `top_k_margin`, etc.; optional small `topk` list. No full vocabulary distribution.

Health flags (repetition loop, high entropy, attention collapse, mid-layer anomaly, NaN/Inf) are computed from these summaries (and a **transient** in-memory buffer for repetition, which is discarded immediately after the check). The report schema is a fixed, small set of numbers per step/layer -- orders of magnitude smaller than raw tensors.

### Is anyone else doing this?

- **Research:** Entropy-Lens (entropy per layer), Entropy Sentinel (statistics from entropy traces), AVSS (variance-sparsity per layer), cumulant expansion (moments of entropy) all use scalar or low-dimensional summaries instead of full activations. They are used for analysis or offline evaluation, not as the core storage model of a production observability system that runs continuously and serves dashboards/APIs.
- **Products:** Commercial LLM observability tools do not instrument the forward pass at all; they have nothing to "summarize" because they don't see internal tensors.
- **Conclusion:** Using summary-only tensor statistics (fixed schema of scalars per layer/step) as the persisted representation for production inference health monitoring -- with no raw tensor storage -- is a clear design choice. We are not aware of another production system that does this as a first-class, documented design for LLM inference observability.
