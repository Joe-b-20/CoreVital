# CoreVital: Competitive Landscape & IP Positioning

**Purpose:** For partnerships, YC, and patent counsel. Summarizes who does what, CoreVital’s differentiators (including tensor summarization), and a one-pager suitable for a provisional patent.

---

## 1. Competitive landscape

### 1.1 LLM “observability” products (application layer)

| Player | What they do | What they don’t do |
|--------|----------------|-------------------|
| **Langfuse** (open-source; acquired by ClickHouse) | Traces prompts, responses, token usage, latency, cost; OpenTelemetry backend; eval scores on outputs | No instrumentation inside the model forward pass |
| **Openlayer** ($14.5M Series A) | Tracing, cost, latency, hallucination/toxicity scoring, guardrails monitoring | Post-hoc scoring on text; no internal activations |
| **OpenLIT, Langtrace** | OpenTelemetry-based tracing across LLM providers and vector DBs | Request/response and orchestration level only |
| **Elastic, Observe** | LLM observability as part of broader APM | API- and app-level; no model-internal signals |

**Takeaway:** These tools sit on **calls to** the model (or orchestration). They do **not** hook into the model’s internal computation (hidden states, attention weights, logits) during generation. CoreVital operates **inside** the inference loop.

### 1.2 Research (ideas, not shipped products)

- **ERGO, Entropy Sentinel:** Use entropy over next-token distributions for drift/accuracy estimation; scalar summaries per step or per response. Research frameworks, not production observability platforms.
- **SpecRA:** Repetition detection via FFT on token sequences. Different signal (text-level) than CoreVital’s hidden-state repetition loop.
- **Attention collapse / attention sinks:** Mechanistic interpretability work; no product that surfaces “collapsed head count” or “attention collapse” as a production health signal.
- **LLMSafeGuard, real-time safeguarding:** Intervene during decoding; not a general “health + risk score + dashboard” observability layer.
- **Entropy-Lens, AVSS, cumulant expansion:** Compute scalar or low-dimensional statistics (entropy, variance-sparsity) **for analysis**. Not designed as the storage model for a production monitoring system.

**Takeaway:** Papers use entropy, repetition, and sometimes attention statistics. No one has turned “internal inference health as summary-only, production-grade observability” into a single, shippable product that cloud providers can bundle.

### 1.3 CoreVital’s wedge

- **Internal inference instrumentation:** Hooks into the model forward pass (Hugging Face / PyTorch); captures hidden states, attention, and logits **during** generation.
- **Health signals:** Repetition loop (last-layer hidden-state similarity), high-entropy steps (logits entropy > threshold), attention collapse (collapsed/focused head counts), mid-layer anomaly (L2 norms, NaN/Inf), plus NaN/Inf in tensors.
- **Single risk score + flags:** One run-level risk score and health flags for alerting and comparison.
- **Summary-only storage (see Section 2):** No raw tensors in reports; only lightweight summaries. Enables practical storage and long retention.

**Positioning:** “Internal inference health for self-hosted/open-source LLMs—entropy, repetition, attention collapse, anomalies—without storing raw activations.”

---

## 2. Tensor summarization: design differentiator

### 2.1 The problem with raw tensors

- **Hidden states:** e.g. `(batch, seq_len, hidden_dim)` → millions of floats per step; full timeline × layers is prohibitive for production.
- **Attention:** `(batch, heads, seq_len, seq_len)` → same scaling issue.
- **Logits:** `(batch, seq_len, vocab_size)` → very large.

Storing raw tensors for every step and layer is not practical for cost, retention, or compliance. Researchers often dump activations for offline analysis; that’s not a production observability design.

### 2.2 What CoreVital does instead

**Never persist raw tensors.** For each step and (when stored) each layer we only compute and store:

- **Hidden states:** Scalar summaries per layer: `mean`, `std`, `l2_norm_mean`, `max_abs`; optional low-dim sketch (e.g. random projection). No full tensor.
- **Attention:** Scalar summaries per layer: `entropy_mean`, `entropy_min`, `entropy_max`, `concentration_max`, `concentration_min`, `collapsed_head_count`, `focused_head_count`. No raw attention matrix.
- **Logits:** Per-step scalars: `entropy`, `perplexity`, `surprisal`, `top_k_margin`, etc.; optional small `topk` list. No full vocabulary distribution.

Health flags (repetition loop, high entropy, attention collapse, mid-layer anomaly, NaN/Inf) are computed from these summaries (and a **transient** in-memory buffer for repetition, which is discarded immediately after the check). The report schema is a fixed, small set of numbers per step/layer—orders of magnitude smaller than raw tensors.

### 2.3 Is anyone else doing this?

- **Research:** Entropy-Lens (entropy per layer), Entropy Sentinel (statistics from entropy traces), AVSS (variance-sparsity per layer), cumulant expansion (moments of entropy) all use **scalar or low-dimensional summaries** instead of full activations. They are used for **analysis or offline evaluation**, not as the core storage model of a **production observability system** that runs continuously and serves dashboards/APIs.
- **Products:** Commercial LLM observability tools do not instrument the forward pass at all; they have nothing to “summarize” because they don’t see internal tensors. So they don’t face the “store raw vs summarize” choice in the same way.
- **Conclusion:** Using **summary-only tensor statistics** (fixed schema of scalars per layer/step) as the **persisted representation** for production inference health monitoring—with no raw tensor storage—is a clear design choice. We are not aware of another **production** system that does this as a first-class, documented design for LLM inference observability.

This is a strong differentiator to stress in positioning and in any patent: “lightweight summary statistics computed from internal tensors at inference time, with no persistence of raw tensors, for production health and risk monitoring.”

---

## 3. Beating others to market

- **Ship and get in front of inference providers:** Cloud/inference vendors (Together, Replicate, RunPod, Modal, or cloud LLM offerings) are the natural bundlers. Early design partners and pilots matter more than waiting for a patent.
- **Narrow positioning:** “Internal inference health for self-hosted LLMs” is distinct from “we trace your API calls.” Use that consistently.
- **Open-source vs commercial:** Consider open-sourcing core instrumentation (or a community tier) to drive adoption; monetize via enterprise features, SLAs, or bundled/managed offerings.
- **Speed:** Research will keep coming. Own the **product category** and **distribution** (integrations, “bundle with our inference”) rather than relying only on the idea.

---

## 4. Patent strategy (summary)

- **First-to-file:** Priority is set by **filing date**. “Patent first” means **file first**.
- **Provisional:** File a **provisional patent application** to establish an early priority date at low cost (~$65–$325 USPTO + attorney drafting). You have **12 months** to file a non-provisional that claims the benefit of that date.
- **Prior art:** We are not aware of a **product** that does internal inference health monitoring (entropy, repetition loop, attention collapse, mid-layer anomaly) as a production tool, or of a **production** system that uses **summary-only tensor statistics** (no raw storage) for this purpose. A **formal patent search** by counsel is still recommended before relying on “we patented it first.”

---

## 5. One-pager for provisional patent (for counsel)

*Below is a concise technical summary suitable for handing to a patent attorney to draft a provisional. It describes the system and method; claims would be drafted by counsel.*

### Title (draft)

**System and method for monitoring health of large-language-model inference using lightweight tensor summaries.**

### Technical field

Monitoring and observability of large language model (LLM) inference in production; instrumentation of internal model computation (hidden states, attention, logits) without storing raw tensors.

### Problem

- Production LLM inference needs health signals (repetition, confusion, numerical failure, attention collapse) to detect failures and trigger alerts or interventions.
- Raw internal tensors (hidden states, attention weights, logits) are too large to store at scale for every generation step and layer.
- Existing observability tools operate at the application or API level and do not instrument the model’s internal forward pass.

### Solution (CoreVital-style approach)

1. **Instrumentation:** During each generation step, intercept or receive the model’s internal tensors (hidden states, attention, logits) from the forward pass (e.g., via framework hooks or an instrumentation layer).

2. **Summary computation:** For each step and, where applicable, each layer:
   - From **hidden states:** compute and retain only scalar statistics (e.g., mean, standard deviation, L2 norm mean, max absolute value) and optionally a low-dimensional sketch (e.g., random projection); do not persist the full tensor.
   - From **attention:** compute and retain only scalar statistics (e.g., entropy mean/min/max, concentration, counts of collapsed or highly focused heads); do not persist the full attention matrix.
   - From **logits:** compute and retain only scalar statistics (e.g., entropy, perplexity, surprisal) and optionally a small top-k list; do not persist the full vocabulary distribution.

3. **Health and risk signals:** From the summaries (and, if needed, a transient in-memory buffer that is not persisted):
   - Detect repetition (e.g., high similarity of last-layer hidden states across consecutive steps).
   - Detect high uncertainty (e.g., logits entropy above a threshold).
   - Detect attention collapse (e.g., one or more heads with very low entropy or very high concentration).
   - Detect mid-layer anomalies (e.g., L2 norms far from a baseline, or NaN/Inf in layer tensors).
   - Detect numerical failure (NaN/Inf in any monitored tensor).

4. **Output:** Persist only the summary statistics and derived health/risk signals (e.g., risk score, health flags) for each run; do not persist raw tensors. Use the persisted data for dashboards, alerting, comparison across runs, or API access.

### Distinctive aspects to emphasize in claims (for counsel)

- **Summary-only persistence:** No storage of raw hidden-state, attention, or logit tensors for the purpose of observability; only predefined scalar (and optionally low-dimensional) summaries are stored.
- **Inference-time computation:** Summaries and health signals are computed during or immediately after each generation step, as part of the inference pipeline.
- **Health signals derived from summaries:** Repetition, high entropy, attention collapse, and mid-layer anomaly are derived from the summary statistics (and transient buffers where needed), not from full tensor access at query time.
- **Production observability:** The system is designed for continuous production use (e.g., many runs, long retention, integration with alerting and dashboards), not only for offline research or one-off analysis.

### Optional scope (for counsel to consider)

- Application to decoder-only (causal) and encoder-decoder models.
- Optional capture modes (e.g., store full per-layer summaries only when risk exceeds a threshold).
- Integration with external sinks (databases, metrics systems, OpenTelemetry).
- Use of the same summary schema for comparison of multiple runs (e.g., side-by-side comparison of metrics).

---

*Document version: 1.0. Update as competitive and IP landscape changes.*
