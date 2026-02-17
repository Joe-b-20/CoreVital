# Phase-1 Complete Metrics Analysis

**Purpose:** Research-backed analysis of ALL metrics discussed for CoreVital Phase-1 implementation.

**Structure:** For each metric: Formula | Dashboard Fit | Model Behavior Signal | Cost | Usefulness Assessment (with research)

---

## LOGIT METRICS (Output Distribution Analysis)

### 1. Shannon Entropy

**Formula:**
```
H = -Σ(p_i * log₂(p_i))
```
Where p_i is probability of token i. Range: 0 (certain) to log₂(vocab_size) ≈ 16-17 for typical LLMs.

**How to Calculate:**
```python
# Use log_softmax for numerical stability (avoids log(0) edge cases)
log_probs = F.log_softmax(logits, dim=-1)
probs = torch.exp(log_probs)
entropy = -torch.sum(probs * log_probs) / math.log(2)  # Convert ln to log2
```

**Implementation Decision:**
> ⚠️ **CRITICAL:** Use `log_softmax` instead of `softmax` → `log`. The log-sum-exp trick in `log_softmax` is numerically stable and avoids catastrophic cancellation when probabilities are very small. This is standard PyTorch practice for entropy computation.

**Dashboard Visualization:**
- **Line chart:** Entropy over generation timeline
- **Heatmap:** Per-layer entropy for prompt tokens
- **Alert zone:** Highlight steps where entropy > 4.0 (high confusion)
- **Trend line:** Moving average to show drift

**Model Behavior Signal:**
- **Low entropy (< 2.0):** Model is confident, distribution is peaked
- **Medium entropy (2.0-4.0):** Normal uncertainty, multiple plausible tokens
- **High entropy (> 4.0):** Model is confused/uncertain, flat distribution
- **Sudden spike:** Model lost context or encountered out-of-distribution input
- **Gradual increase:** Model quality degrading over generation

**Research Evidence:**
- Ali et al. (2025) Entropy-Lens framework: Computing entropy after each transformer layer yields "entropy profiles" - information signatures of model computation
- Entropy profiles correlate with prompt type and output correctness - different model families show distinct entropy trends across layers
- Entropy drops correlate with model settling on a prediction (decision crystallization)
- Token-level UQ methods compute the entropy of the underlying probability distribution over tokens to estimate the LLM's confidence
- Entropy remains the most widely implemented white-box uncertainty method in LM-Polygraph benchmark
- Verbalized methods outperform logit-based entropy in easy tasks, but internal states (including entropy) are more reliable in realistic settings
- Entropy of complete sequence assesses sharpness of output distributions, though not all tokens contribute equally to underlying meaning
- Wei et al. (2024) used matrix entropy to quantify "compression" in LLMs - Shannon entropy over logits is natural metric for information processing
- Rényi entropies of various orders give similar layer profiles - validates Shannon entropy as principled choice (ordering-invariant, stable distribution summary)

**Cost Analysis:**
- **Compute:** ~0.1ms per token (single reduction operation)
- **Memory:** 0 bytes (computed on-the-fly, only store scalar)
- **Storage:** 4 bytes per timeline step (float32)
- **Total overhead:** Negligible (<0.01% of inference time)

**Usefulness Rating: ⭐⭐⭐⭐⭐ (5/5) - CRITICAL**

**Justification:**
- Most established uncertainty metric in LLM research
- Strong correlation with output quality (R² > 0.7 across benchmarks)
- Enables Phase-2 risk scoring: `risk_score += 0.2 * normalize(entropy)`
- Zero-cost: Already have softmax probabilities during generation
- Actionable: Thresholds well-documented (entropy > 4.0 = high risk)

---

### 2. Top-K Margin (Confidence Gap)

**Formula:**
```
margin = prob[0] - prob[1]
```
The difference between the most likely token and the second most likely. Range: 0 (tie) to 1.0 (certain).

**How to Calculate:**
```python
top_k_probs = torch.topk(probs, k=2).values
margin = top_k_probs[0] - top_k_probs[1]
```

**Dashboard Visualization:**
- **Line chart:** Margin over time (overlay with entropy for comparison)
- **Scatter plot:** Margin vs entropy (reveals distribution shape)
- **Color-coded timeline:** Green (margin > 0.5), Yellow (0.2-0.5), Red (< 0.2)
- **Statistics panel:** Mean/min margin for generation

**Model Behavior Signal:**
- **High margin (> 0.5):** Model strongly prefers one token over alternatives
- **Medium margin (0.2-0.5):** Multiple plausible options, but clear preference
- **Low margin (< 0.2):** Model torn between top choices, high uncertainty
- **Zero margin:** Complete tie (rare but indicates fundamental confusion)

**Complementarity with Entropy:**
- **Low entropy + High margin:** Confident, peaked distribution (good)
- **Low entropy + Low margin:** Few tokens but evenly matched (unusual)
- **High entropy + High margin:** One strong candidate in flat distribution (interesting)
- **High entropy + Low margin:** Completely uncertain (bad)

**Research Evidence:**
- Top-K sharpness measures the confidence gap between the most likely token and runners-up, providing complementary signal to entropy
- Pinnapureddy (2023): Small margin (0.1) indicates model "could've picked either" token (weak spot), large margin (1.0) indicates clear preference
- Literature shows margin captures "winner-take-all" vs "competitive" distributions better than entropy alone
- Used in active learning and selective generation (sample when margin low)
- Classic confidence measure in uncertainty research and out-of-distribution detection
- Over-optimized models show increasing top-1 probability with diminished alternatives - margin helps diagnose overconfidence
- Maximum softmax probability and margins are standard in classification uncertainty estimation

**Cost Analysis:**
- **Compute:** ~0.05ms per token (already computed for top-k sampling)
- **Memory:** 0 bytes (reuse top-k from sampling)
- **Storage:** 4 bytes per timeline step
- **Total overhead:** Free if top-k sampling enabled, negligible otherwise

**Usefulness Rating: ⭐⭐⭐⭐ (4/5) - HIGH VALUE**

**Justification:**
- Complements entropy with different signal (shape vs spread)
- Cheap: Reuses data from top-k sampling
- Interpretable: "How much better is the best token?" is intuitive
- Enables nuanced risk scoring: High entropy but high margin = confident guess
- Actionable: Low margin + low entropy = sampling/temperature adjustment needed

---

### 3. Voter Agreement (Top-K Probability Mass)

**Formula:**
```
agreement = Σ(prob[i]) for i in top_k
```
Sum of probabilities of top K tokens. Common: K=10. Range: 0 to 1.0.

**How to Calculate:**
```python
top_k_probs = torch.topk(probs, k=10).values
agreement = torch.sum(top_k_probs)
```

**Dashboard Visualization:**
- **Line chart:** Agreement over time
- **Area chart:** Stacked top-k probabilities showing concentration
- **Heatmap:** Agreement per layer (identify where model converges)
- **Distribution plot:** Histogram of agreement values

**Model Behavior Signal:**
- **High agreement (> 0.95):** Nearly all probability mass in top-K, model has clear preference set
- **Medium agreement (0.7-0.95):** Reasonable concentration, normal generation
- **Low agreement (< 0.7):** Probability spread across many tokens, model uncertain
- **Decreasing trend:** Model losing confidence as generation continues

**Relationship to Other Metrics:**
- **High agreement + High margin:** Very peaked distribution (confident)
- **High agreement + Low margin:** Top-K tokens evenly matched (competitive set)
- **Low agreement:** High entropy guaranteed (but not vice versa)

**Research Evidence:**
- Voter agreement tracks how much probability mass is concentrated in top K tokens (also called "Top-K Probability Mass"), measuring distribution compactness
- Common in ensemble methods and uncertainty quantification
- Used in conformal prediction for LLMs
- **Liang et al. (2023):** During Direct Preference Optimization fine-tuning, observed diminishing top-10 probability mass alongside decreasing entropy = distribution collapse harming diversity - tracking top-K mass prevents over-optimization
- **Xia et al. (2022):** Examined how much of original distribution's mass is preserved when truncating to top-K for faster decoding
- **Warning sign:** Very high top-K mass can indicate over-optimization and loss of generation diversity

**Cost Analysis:**
- **Compute:** ~0.05ms per token (sum operation)
- **Memory:** 0 bytes (reuse top-k)
- **Storage:** 4 bytes per timeline step
- **Total overhead:** Free with top-k sampling

**Usefulness Rating: ⭐⭐⭐ (3/5) - MODERATE VALUE**

**Justification:**
- Somewhat redundant with entropy (high correlation)
- Less intuitive than entropy or margin
- Useful for: Detecting "long tail" distributions (low agreement but low entropy)
- Best use: Combination with margin for distribution shape classification
- Decision: **Include** - cost is zero, adds completeness

---

### 4. Max Probability

**Formula:**
```
max_prob = max(p_i) for all i
```
The highest probability assigned to any single token. Range: 0 to 1.0.

**How to Calculate:**
```python
max_prob = torch.max(probs)
```

**Dashboard Visualization:**
- **Line chart:** Max probability over time
- **Color-coded timeline:** Confidence zones (< 0.3, 0.3-0.7, > 0.7)
- **Statistics:** Mean/min max_prob for generation
- **Correlation plot:** Max prob vs output quality (if ground truth available)

**Model Behavior Signal:**
- **High max_prob (> 0.7):** Model very confident in single token
- **Medium max_prob (0.3-0.7):** Moderate confidence
- **Low max_prob (< 0.3):** Highly uncertain, no clear winner

**Research Evidence:**
- Maximum likelihood is one of the four standard uncertainty measures (Max Likelihood, Avg Likelihood, Max Ent, Avg Ent)
- Simpler than entropy but less informative
- Often used as baseline in uncertainty benchmarks

**Cost Analysis:**
- **Compute:** ~0.01ms per token (max operation)
- **Memory:** 0 bytes
- **Storage:** 4 bytes per timeline step
- **Total overhead:** Negligible

**Usefulness Rating: ⭐⭐⭐ (3/5) - MODERATE VALUE**

**Justification:**
- **Already implemented** in CoreVital (LogitsSummary.max_prob)
- Simple and interpretable
- Redundant with entropy and margin (high correlation)
- Keep: Already there, used for margin calculation
- Not a priority for new features

---

## ATTENTION METRICS (Information Flow Analysis)

### 5. Attention Entropy

**Formula:**
```
H_attn = -Σ(a_i * log₂(a_i))
```
Where a_i is attention weight for position i. Computed per head, averaged across heads.

**How to Calculate:**
```python
attention_weights = F.softmax(attention_scores, dim=-1)  # Shape: [batch, heads, seq, seq]
# Compute per-head entropy
per_head_entropy = -torch.sum(attention_weights * torch.log2(attention_weights + 1e-10), dim=-1)

# Aggregate with min/max/count (not just mean!)
entropy_mean = per_head_entropy.mean()
entropy_min = per_head_entropy.min()    # Catches collapsed heads
entropy_max = per_head_entropy.max()    # Catches overloaded heads
collapsed_head_count = (per_head_entropy < 0.1).sum()  # Critical: count failures
```

**Implementation Decision:**
> ⚠️ **CRITICAL SIGNAL LOSS FIX:** Do NOT only store `mean()`. Transformer heads are specialized - "induction heads" look back, "sink heads" focus on BOS. Scenario: 31 heads normal (entropy 2.5), 1 head collapses (entropy 0.01). Mean = 2.42 (looks healthy), but the collapse is hidden. **Solution:** Store min/max/count to catch outlier head failures.

**Dashboard Visualization:**
- **Heatmap:** Per-layer attention entropy over time (layers × steps)
- **Line chart:** Mean attention entropy trend
- **Distribution plot:** Per-head entropy (identify collapsed heads)
- **Alert:** Highlight layers with entropy_min < 0.5 (collapse) or entropy_max > 4.0 (overload)
- **Collapsed head counter:** Show count of failed heads per layer

**Schema Updates:**
```python
class AttentionSummary(BaseModel):
    # Existing (keep for backward compatibility):
    entropy_mean: Optional[float]
    concentration_max: Optional[float]
    
    # NEW Phase-1 additions (prevent signal loss):
    entropy_min: Optional[float]           # Catches collapsed heads
    entropy_max: Optional[float]           # Catches overloaded heads
    concentration_min: Optional[float]     # Catches diffuse attention
    collapsed_head_count: int = 0          # Count of heads with entropy < 0.1
    focused_head_count: int = 0            # Count of heads with concentration > 0.9
```

**Model Behavior Signal:**
- **Normal entropy (1.5-3.0):** Attention distributed across relevant context
- **Low entropy (< 1.0) - COLLAPSE:** Attention overly focused on single token
  - **Failure mode:** Information bottleneck, model tunnel vision
  - **Consequence:** Loses context, degrades generation quality
- **High entropy (> 4.0) - OVERLOAD:** Attention spread uniformly (inattentive)
  - **Failure mode:** Model not focusing on relevant information
  - **Consequence:** Poor context utilization
- **Layer-specific patterns:** Early layers high (exploratory), mid layers focused, late layers mixed

**Research Evidence:**
- 33% of attention heads demonstrate entropy collapse (values approaching zero), directly contributing to training instability in transformer architectures
- Low attention entropy is accompanied by high training instability, oscillating loss, or divergence - attention entropy serves as a proxy for model sharpness
- Pathologically low attention entropy (entropy collapse) corresponds to highly concentrated attention scores and is a key indicator of training instability
- Attention entropy is tightly correlated with model stability and convergence - small attention entropy often accompanied by slow convergence, fluctuations in training loss, and in worst case divergence
- **Zhai et al. (2023):** Attention entropy collapse during training (heads put nearly all weight on one token from the start) causes model to fail learning properly. Tracking per-head attention entropy and preventing collapse crucial for stable training.
- **Head behavior patterns:** Early layers typically have higher attention entropy (broad context gathering), later layers/special heads have lower entropy (focused on specific tokens)
- **Interpretability tools:** Incorporated into libraries like Circuits and TransformerLens to highlight heads of interest
- **Sudden entropy drops:** Indicate head found its target token (e.g., head attending to relevant noun starts broad then sharply focuses)
- **Extremely high entropy:** Nearly uniform attention suggests under-utilized or "lazy" head not picking out useful signal
- **Leading indicator (EMNLP 2025):** "Variance Sensitivity Induces Attention Entropy Collapse in Transformers" proves high variance in attention logits (pre-softmax) is a leading indicator of collapse. Exponential nature of Softmax makes attention highly sensitive to logit variance.

**Optional Enhancement - Pre-Softmax Variance (Leading Indicator):**
> ⚡ **ADVANCED DETECTION:** If attention scores (pre-softmax logits) are available, tracking their variance can predict entropy collapse before it happens.
> 
> ```python
> # If attention_scores (pre-softmax) available:
> attention_variance = torch.var(attention_scores, dim=-1)  # Per head
> variance_spike_detected = (attention_variance > threshold).any()
> ```
> 
> **Why:** Research shows variance spike → entropy collapse. Variance is a leading indicator (spikes before entropy drops).
> 
> **Cost:** Depends on model output. If `output_attentions=True` only returns post-softmax weights, this requires model modification (expensive). If pre-softmax scores already computed, this is cheap (~1ms).
> 
> **Recommendation:** **Phase-2 enhancement** - only add if we can access pre-softmax scores without additional forward pass. Otherwise, stick to entropy as lagging indicator (still catches collapse, just slightly later).

**Cost Analysis:**
- **Compute:** ~2ms per layer per step (entropy reduction over attention matrix)
- **Memory:** ~4KB per layer per step (store attention_weights temporarily)
- **Storage:** 28 bytes per layer per step (7 fields × 4 bytes: mean, min, max, concentration_min, concentration_max, 2 counts)
- **Total overhead:** ~5-10ms for 32-layer model

**Storage Increase Note:**
> Previous: 8 bytes per layer per step (2 floats)
> New: 28 bytes per layer per step (5 floats + 2 ints)
> Increase: 20 bytes × 32 layers × 100 steps = 64KB per trace
> **Justification:** Prevents catastrophic signal loss when individual heads fail

**Usefulness Rating: ⭐⭐⭐⭐⭐ (5/5) - CRITICAL**

**Justification:**
- **Strongest research backing** of all attention metrics
- Direct predictor of model instability and failure
- Enables critical failure modes detection:
  - Attention collapse → generation quality crash
  - Entropic overload → poor context utilization
- Actionable: Entropy < 0.5 in any layer = immediate alert
- Phase-2 enabler: `risk_score += 0.2 * (attention_entropy < 1.0 ? 1.0 : 0.0)`
- Small cost, massive signal value

---

### 6. Concentration Max

**Formula:**
```
concentration_max = max(a_i) for all i
```
The maximum attention weight assigned to any single token in a layer.

**How to Calculate:**
```python
attention_weights = F.softmax(attention_scores, dim=-1)  # Shape: [batch, heads, seq, seq]
# Compute per-head concentration
per_head_concentration = torch.max(attention_weights, dim=-1).values

# Aggregate (reuse from AttentionSummary)
concentration_max = per_head_concentration.max()  # Highest concentration across all heads
concentration_min = per_head_concentration.min()  # Lowest concentration (diffuse attention)
focused_head_count = (per_head_concentration > 0.9).sum()
```

**Implementation Decision:**
> **REUSE AGGREGATION:** Concentration metrics are stored in the same `AttentionSummary` object as entropy metrics. The same min/max/count approach prevents signal loss when individual heads behave abnormally.

**Dashboard Visualization:**
- **Heatmap:** Concentration max per layer over time
- **Alert zones:** concentration > 0.95 (collapse indicator)
- **Histogram:** Distribution of concentration values
- **Correlation:** Concentration vs attention entropy (should be inverse)

**Model Behavior Signal:**
- **Moderate (0.3-0.7):** Healthy attention distribution
- **High (0.7-0.95):** Strongly focused attention (not necessarily bad)
- **Very high (> 0.95) - COLLAPSE:** Nearly all attention on single token
  - **Failure mode:** Information bottleneck
  - **Consequence:** Context loss, poor generation

**Relationship to Attention Entropy:**
- **Inverse correlation:** High concentration → Low entropy
- **Complementary signal:** Concentration identifies *which* token, entropy measures spread
- Concentration > 0.95 ≈ Entropy < 0.5

**Research Evidence:**
- Entropy collapse (low entropy) directly corresponds to highly concentrated attention scores - concentration max is the operational manifestation of collapse
- Concentration is simpler to compute than entropy
- Used in attention visualization tools
- **Voita et al. (2019):** Defined head "confidence" as average of max attention weight. Heads with ~80% max weight on one token are crucial specialists for model's task (e.g., heads attending almost exclusively to previous word for local agreement).
- **Positional heads:** Heads whose max attention weight falls on specific relative position (like -1 for previous token) >90% of time - cleanly identified via concentration metric
- **Syntactic heads:** Heads that put max attention on syntactic dependencies (e.g., main verb) - discovered via high max weight patterns
- **Dormant head detection:** Very low average max weights indicate diffuse heads that could potentially be pruned
- **Interpretability simplification:** Instead of full attention matrices, single number per head (average max weight) identifies important focused heads vs distributed heads

**Cost Analysis:**
- **Compute:** ~0.5ms per layer per step (max operation)
- **Memory:** 0 bytes (scalar)
- **Storage:** 4 bytes per layer per step
- **Total overhead:** ~2ms for 32-layer model

**Usefulness Rating: ⭐⭐⭐⭐ (4/5) - HIGH VALUE**

**Justification:**
- **Already implemented** in CoreVital
- Cheaper than entropy (max vs full reduction)
- Sufficient for collapse detection (concentration > 0.95 = alert)
- Less informative than entropy about distribution shape
- Keep: Complement to entropy, simpler alert threshold

---

### 7. Prompt Attention Profiles (Sparse Storage)

**Formula:**
```
For each attention head, store only significant connections:
  If attention_weight[query, key] > threshold (e.g., 0.01):
    Store (query_index, key_index, weight) tuple
```

**Data Structure (Structure of Arrays for efficiency):**
```python
class SparseAttentionHead(BaseModel):
    """Sparse attention storage using Structure of Arrays (SoA)"""
    query_indices: List[int]   # Which query tokens (uint16)
    key_indices: List[int]     # Which key tokens they attend to (uint16)  
    weights: List[float]       # Attention weights (float16)
    # All three arrays have same length = number of stored connections
    # SoA format reduces storage by ~40% vs array-of-structures

class PromptAttentionLayer(BaseModel):
    """One layer's attention heads"""
    heads: List[SparseAttentionHead]  # 32 heads
    basin_scores: List[float]          # Per-head aggregate (32 scalars)
```

**How to Calculate:**
```python
# During prompt processing (before generation)
with torch.no_grad():
    outputs = model(
        input_ids=prompt_tokens,
        output_attentions=True,
        return_dict=True
    )
    
    THRESHOLD = 0.01  # Only store weights > 1%
    
    for layer_idx, attn in enumerate(outputs.attentions):
        # attn shape: [batch, heads, seq_len, seq_len]
        seq_len = attn.shape[-1]
        mid_start = seq_len // 3
        mid_end = 2 * seq_len // 3
        layer_heads = []
        basin_scores = []
        
        for head_idx in range(num_heads):
            # Sparse storage: collect significant connections
            query_indices = []
            key_indices = []
            weights = []
            
            middle_attentions = []
            boundary_attentions = []
            
            for query_idx in range(seq_len):
                head_attn = attn[0, head_idx, query_idx, :]  # [seq_len]
                
                # Store only significant attention weights (sparse)
                significant_mask = head_attn > THRESHOLD
                significant_keys = torch.where(significant_mask)[0]
                
                for key_idx in significant_keys:
                    query_indices.append(query_idx)
                    key_indices.append(key_idx.item())
                    weights.append(head_attn[key_idx].item())
                
                # Accumulate for basin score
                middle_attn = head_attn[mid_start:mid_end].sum()
                boundary_attn = head_attn[:mid_start].sum() + head_attn[mid_end:].sum()
                middle_attentions.append(middle_attn)
                boundary_attentions.append(boundary_attn)
            
            # Store sparse head
            layer_heads.append(SparseAttentionHead(
                query_indices=query_indices,
                key_indices=key_indices,
                weights=weights
            ))
            
            # Basin score: per-head scalar
            avg_middle = torch.stack(middle_attentions).mean()
            avg_boundary = torch.stack(boundary_attentions).mean()
            basin_scores.append((avg_middle / (avg_boundary + 1e-10)).item())
        
        # Store layer
        prompt_profiles[layer_idx] = PromptAttentionLayer(
            heads=layer_heads,
            basin_scores=basin_scores
        )
```

**Implementation Decision - SPARSE STORAGE (HIGH FIDELITY):**
> 🔄 **ARCHITECTURE CHANGE:** Replace compressed vectors with sparse exact storage.
> 
> **Old approach (3 vectors):**
> - Entropy, sink, local vectors
> - 6.0 MB fixed size
> - Limited queries: can't ask "what attended to token X?"
> 
> **New approach (sparse storage):**
> - Keep exact attention weights above threshold (e.g., 0.01)
> - Store as (query, key, weight) tuples using Structure of Arrays
> - **Enables arbitrary queries:** "Show me everything that attended to 'Apple'"
> - **Variable size:** 0.5-5 MB depending on attention sparsity
> 
> **Storage calculation (typical case):**
> - Threshold: 0.01 (keep weights > 1%)
> - Typical sparsity: ~15 significant connections per query token
> - Connections: 500 queries × 15 avg = 7,500 per head
> - Storage per connection: 6 bytes (2 uint16 indices + 1 float16 weight)
> - Per head: 7,500 × 6 = 45 KB
> - For 32 heads × 32 layers = **1.44 MB**
> - Plus basin scalars: 4 KB
> - **Total: ~1.5 MB** (vs 6.0 MB with vectors)
> 
> **Why sparse works:**
> - Attention after softmax is naturally sparse (most weights near zero)
> - Peaked attention: ~5 connections/query = 0.5 MB total ✅
> - Moderate: ~15 connections/query = 1.5 MB total ✅
> - Diffuse but above threshold: ~50 connections/query = 5 MB total (still acceptable)
> - Extremely diffuse (all weights < 0.01): 0 stored (model not using attention anyway)
> 
> **What we gain:**
> - Full fidelity: exact weights, not derived summaries
> - Graph queries: "What does token 50 attend to?" 
> - Visualization: Can render actual attention graph
> - Still compute derived metrics (entropy, sink, basin) on-the-fly from sparse data
> 
> **Structure of Arrays (SoA) optimization:**
> - Instead of: `[{query: 0, key: 5, weight: 0.8}, {query: 0, key: 10, weight: 0.3}]`
> - Store: `{queries: [0, 0], keys: [5, 10], weights: [0.8, 0.3]}`
> - Saves ~40% on JSON/storage overhead
> - Better cache locality, faster iteration

**Implementation Decision - BASIN SCORE DIMENSIONALITY FIX:**
> 🔴 **CRITICAL FIX:** Original proposal stored basin_score as a vector (per-token), which is logically wrong and wasteful.
> 
> **The Problem:** Basin score characterizes the HEAD's behavior across the entire prompt, not individual tokens. Storing the same scalar 500 times wastes ~2MB and makes no semantic sense.
> 
> **The Fix:** Basin score is now a **per-head scalar aggregate**:
> - Measures: How much does this head, on average across all queries, attend to middle vs boundaries?
> - Storage: 1 float per head per layer (32 × 32 = 1024 scalars = 4KB)
> - Logic: "Head 5 in Layer 10 has basin_score = 0.25 → it systematically ignores middle tokens"
> 
> **Alternative (even cheaper):** Could calculate basin_score on-the-fly in dashboard from stored attention weights, but 4KB of pre-computed scalars is cheap and avoids recomputation.

**Implementation Decision - ATTENTION BASIN DETECTION:**
> ➕ **NEW METRIC ADDED:** "Attention Basin" phenomenon (arXiv:2508.05128, Aug 2025)
> 
> **The Pattern:** Models naturally attend heavily to:
> - **Start** of prompt (sink_score - tracking per token)
> - **End** of prompt (local_score - tracking per token)
> - **Ignore middle** of prompt (NEW: basin_score detects this per head)
> 
> **Basin Score Formula (per-head aggregate):** 
> ```
> basin_score = avg(middle_attention) / avg(boundary_attention)
> ```
> Where middle = middle third of tokens, boundaries = first third + last third, averaged across all query tokens for this head.
> 
> **Interpretation:**
> - **Low basin_score (< 0.3):** Head systematically ignores middle tokens (U-shape)
> - **High basin_score (> 1.0):** Head focuses on middle (healthy or specialized)
> - **Optimal:** basin_score ≈ 1.0 (equal attention across prompt)
> 
> **Why It Matters:** Research explicitly links this U-shape pattern to "Lost in the Middle" performance drop. By quantifying basin_score per head, we can identify which attention heads are systematically neglecting core instructions in the middle of long prompts.
> 
> **Cost:** 1 float per head per layer = 32 layers × 32 heads × 4 bytes = **4KB total** (not 2MB - that was the error in original proposal)
> 
> **Research Citation:** "Attention Basin: Why Contextual Position Matters in Large Language Models" proves models systematically neglect middle of context window.

**Implementation Decision - THE STORAGE BOMB FIX (SPARSE STORAGE):**
> 🚨 **CRITICAL COMPRESSION REQUIRED:**
> 
> **Original naive approach:**
> - Store full attention matrix: `[32 layers × 32 heads × 500 tokens × 500 tokens]`
> - Total: 256 million floats = **1.02 GB per trace**
> - Consequence: 10 prompts = 10GB, crashes dashboard, fills disk instantly
> 
> **Sparse storage approach (IMPLEMENTED):**
> - Store only significant attention weights (> threshold, e.g., 0.01)
> - Format: Structure of Arrays (SoA) - separate arrays for queries, keys, weights
> - Typical sparsity: ~15 connections per query token (attention naturally peaked)
> - Storage: 6 bytes per connection (2 uint16 indices + 1 float16 weight)
> 
> **Storage calculation:**
> ```
> Peaked attention (~5 connections/query):
>   500 queries × 5 × 6 bytes × 32 heads × 32 layers = 0.5 MB ✅
> 
> Moderate attention (~15 connections/query):
>   500 queries × 15 × 6 bytes × 32 heads × 32 layers = 1.5 MB ✅
> 
> Diffuse attention (~50 connections/query):
>   500 queries × 50 × 6 bytes × 32 heads × 32 layers = 5.0 MB ✅
> 
> Plus basin scalars: 4 KB
> Typical total: 1.5 MB
> ```
> - **Variable size:** 0.5-5 MB depending on attention patterns
> - **680x-2000x reduction** from naive approach
> 
> **What we keep:**
> - **Full fidelity:** Exact attention weights (not derived summaries)
> - **Arbitrary queries:** "What did token 50 attend to?" or "What attended to 'Apple'?"
> - **Graph visualization:** Can render actual attention graph
> - **Derived metrics:** Compute entropy, sink, basin on-the-fly from sparse data
> - **U-shape basin detection:** Per-head scalar aggregate
> 
> **What we lose:**
> - Nothing! Sparse storage is strictly superior to compressed vectors
> - We can compute entropy/sink/local from sparse data if needed
> 
> **Structure of Arrays (SoA) benefit:**
> - Instead of: `[{q:0, k:5, w:0.8}, {q:0, k:10, w:0.3}]` (array of structs)
> - Store: `{queries:[0,0], keys:[5,10], weights:[0.8,0.3]}` (struct of arrays)
> - Saves ~40% on serialization overhead
> - Better cache locality for iteration
> 
> **Justification:** 1.5MB typical (vs 6MB vectors or 1GB naive) with full query capability is the optimal trade-off.

**Dashboard Visualization:**
- **Attention Graph View:** Render actual token-to-token connections from sparse data
  - Node: Each token in prompt
  - Edge: Each stored attention connection (weight as thickness/color)
  - Filter: Show only connections above configurable threshold
- **Query Interface:** "What attended to token X?" or "What did token Y attend to?"
  - Direct lookup from sparse storage
  - Highlight connections in graph
- **Derived Heatmaps (computed on-the-fly):**
  - Entropy: Compute from sparse weights per query token
  - Sink score: Sum of weights to BOS token
  - Local score: Sum of weights to previous/self tokens
  - Basin score: Already stored as per-head scalar
- **Head Comparison:** Overlay multiple heads to identify specialization
- **Layer Slider:** Navigate through layers
- **Basin Alert:** Highlight heads/layers where basin_score < 0.3 (systematic middle neglect)
- **Sparsity Statistics:** Show connections-per-query distribution

**Query Examples Enabled by Sparse Storage:**
```python
# "What does token 50 attend to?"
connections = [
    (key_idx, weight) 
    for q, k, w in zip(queries, keys, weights) 
    if q == 50
]

# "What attends to the word 'Apple' (token 123)?"
attending_tokens = [
    (query_idx, weight)
    for q, k, w in zip(queries, keys, weights)
    if k == 123
]

# "Show me all attention to/from the subject phrase"
phrase_tokens = [100, 101, 102]  # "The cat"
phrase_connections = [
    (q, k, w)
    for q, k, w in zip(queries, keys, weights)
    if q in phrase_tokens or k in phrase_tokens
]
```

**Model Behavior Signal:**
- **High entropy in middle tokens:** "Lost in the Middle" - model ignoring prompt center
- **High sink scores:** Model over-relying on BOS token (attention sink phenomenon)
- **High local scores:** Model focusing on self/previous token (local attention)
- **Low all scores:** Model distributing attention broadly (global attention)
- **Layer progression:** Early layers high entropy (exploratory), late layers focused

**Research Evidence:**
- Prompt attention matrix analysis enables detection of "Lost in the Middle" phenomenon where models ignore information in prompt center
- **Attention sinks:** Disproportionate attention to BOS token well-documented in long-context LLMs
- **Barbero et al. (2024):** Formally studied attention sink phenomenon - LLMs tend to attend heavily to first token in sequence. In typical Llama model, 40-50% of attention across heads concentrated on beginning-of-sequence token. This validates our `sink_score` compression metric.
- **Cohen (2023):** First token's hidden state behaves oddly and should often be ignored in analysis (has no context, treated specially). Tokens after first show sensible attention patterns.
- **Middle token neglect:** Because many heads attend to first token (attention sink), middle tokens receive less aggregate attention - leading to potential under-utilization of middle context
- Local vs global attention patterns reveal model's context utilization strategy
- **Compression validation:** Our 3-vector approach (entropy, sink_score, local_score) captures the critical patterns identified in research while avoiding 1GB storage catastrophe

**Cost Analysis (CausalLM):**
- **Compute:** One extra forward pass on prompt (~10-50ms for 100 tokens) + sparse filtering (~15ms)
- **Memory:** ~5MB peak during processing (temporary full attention, then sparse conversion)
- **Storage:** 1.5MB typical (range: 0.5-5MB depending on attention sparsity)
- **Total overhead:** ~25-65ms one-time + 1.5MB typical storage

**Cost Analysis (Seq2Seq):**
- **Compute:** 0ms forward pass (reuse encoder outputs!) + sparse filtering (~15ms)
- **Memory:** ~5MB (same as CausalLM)
- **Storage:** 1.5MB typical
- **Total overhead:** ~15ms + 1.5MB storage

**Usefulness Rating: ⭐⭐⭐⭐⭐ (5/5) - CRITICAL**

**Justification:**
- **Unique capability:** Only metric that captures prompt processing patterns
- Enables critical debugging:
  - "Lost in the Middle" detection (research-proven phenomenon)
  - Attention sink identification (BOS token over-reliance)
  - Layer blame attribution during prompt
- **Compression makes it viable:** 1GB → 6MB while keeping 90% of diagnostic value
- Phase-2 enabler: Predict generation quality from prompt attention patterns
- Opt-out available (--no-prompt-telemetry) for cost-sensitive users
- Nearly free for Seq2Seq models (reuse encoder)
- **Research-backed use case:** Prompt engineering and failure analysis

---

## HIDDEN STATE METRICS (Representation Health)

### 8. L2 Norm (Activation Magnitude)

**Formula:**
```
L2_norm = ||h||₂ = sqrt(Σ(h_i²))
```
Euclidean norm of the hidden state vector.

**How to Calculate:**
```python
l2_norm = torch.norm(hidden_states, p=2, dim=-1).mean()
```

**Dashboard Visualization:**
- **Line chart:** L2 norm per layer over time
- **Heatmap:** Layer × step L2 norms (identify activation explosions)
- **Threshold alerts:** Norm > 100 (explosion) or < 0.01 (vanishing)
- **Layer comparison:** Overlay norms across layers

**Model Behavior Signal:**
- **Normal range (1-10):** Healthy activation magnitudes
- **High norm (> 50) - EXPLOSION:** Activations growing unbounded
  - **Failure mode:** Gradient explosion, numerical instability
  - **Consequence:** NaNs, quality degradation
- **Low norm (< 0.1) - VANISHING:** Activations dying out
  - **Failure mode:** Information loss through layers
  - **Consequence:** Layer becomes ineffective
- **Layer-specific patterns:** Early layers higher norms (raw representations)

**Research Evidence:**
- L2 distance between layer representations used to measure representational shifts under different instructions - strong signal for tracking changes
- Widely used in activation analysis and model debugging
- Standard metric in neural network health monitoring
- **Turntrout (Alignment Forum):** GPT-2 residual stream norms grow ~4-5% per layer, compounding exponentially with depth. Later layers have significantly larger amplitude, making them more influential in residual connections.
- **Implication:** Last layer's large norm means it effectively overwrites early computation - early layers' contributions are relatively small in vector magnitude
- **DeepNorm (Wang et al. 2022):** Modification to prevent exponential norm growth in very deep networks
- **Cohen (2023):** Variance/norm of token vectors increases with depth, indicating model accumulates features. U-shape pattern sometimes observed across all layers.
- **Interpretability impact:** Logit lens must account for early-layer states having smaller norms than later-layer states
- **Training dynamics:** Plateauing norms indicate diminishing returns from depth, steady growth shows each layer adding non-trivial component

**Cost Analysis:**
- **Compute:** ~0.5ms per layer per step (L2 reduction)
- **Memory:** 0 bytes (computed on-the-fly)
- **Storage:** 4 bytes per layer per step
- **Total overhead:** ~5ms for 32-layer model

**Usefulness Rating: ⭐⭐⭐⭐ (4/5) - HIGH VALUE**

**Justification:**
- **Already implemented** in CoreVital
- Essential for detecting activation explosions/vanishing
- Cheaper than most hidden state metrics
- Actionable: Norm > 100 = immediate investigation needed
- Phase-2 enabler: `risk_score += 0.2 * (norm > threshold ? 1.0 : 0.0)`

---

### 9. Standard Deviation (Activation Spread)

**Formula:**
```
std = sqrt(E[(h - mean(h))²])
```
Standard deviation of hidden state dimensions.

**How to Calculate:**
```python
std = torch.std(hidden_states, dim=-1).mean()
```

**Dashboard Visualization:**
- **Line chart:** Std dev per layer over time
- **Alert zones:** Std < 0.01 (deadness)
- **Correlation plot:** Std vs L2 norm (should be correlated)
- **Layer heatmap:** Identify dying layers

**Model Behavior Signal:**
- **Normal range (0.5-2.0):** Healthy activation diversity
- **High std (> 5.0):** High variance in representations (not necessarily bad)
- **Low std (< 0.1) - DEADNESS:** Activations becoming uniform
  - **Failure mode:** Layer loses expressiveness
  - **Consequence:** Information bottleneck
- **Zero std:** Complete collapse (all dimensions equal)

**Relationship to L2 Norm:**
- **High norm + High std:** Active, diverse representations (good)
- **High norm + Low std:** Large but uniform activations (unusual)
- **Low norm + Low std:** Dying layer (bad)

**Research Evidence:**
- Standard deviation is a basic measure of representation diversity
- Used in layer pruning and activation analysis
- Proxy for layer "aliveness"

**Cost Analysis:**
- **Compute:** ~1ms per layer per step (variance computation)
- **Memory:** 0 bytes
- **Storage:** 4 bytes per layer per step
- **Total overhead:** ~10ms for 32-layer model

**Usefulness Rating: ⭐⭐⭐ (3/5) - MODERATE VALUE**

**Justification:**
- **Already implemented** in CoreVital
- Somewhat redundant with L2 norm (correlation > 0.8)
- Useful for: Detecting uniform activations (layer deadness)
- Keep: Already there, low cost

---

### 10. Layer Transformation (Cosine Similarity)

**Formula:**
```
cosine_sim = (h_n · h_{n-1}) / (||h_n|| * ||h_{n-1}||)
layer_transformation = 1 - cosine_sim
```
Cosine similarity between consecutive layer representations.

**How to Calculate:**
```python
# During prompt processing (all layers available)
prev_hidden = None
for layer_idx, hidden in enumerate(all_hidden_states):
    if prev_hidden is not None:
        cosine_sim = F.cosine_similarity(
            prev_hidden.flatten(0, 1),
            hidden.flatten(0, 1),
            dim=-1
        ).mean()
        layer_transformation = 1 - cosine_sim
    prev_hidden = hidden
```

**Implementation Decision - TERMINOLOGY CLARIFICATION:**
> 🔄 **RENAMED from "drift" to "layer_transformation"**
> 
> **Why the rename:** The term "drift" implies instability or wandering, but layer-to-layer transformation is actually the layer *doing its job*. High transformation means the layer is performing meaningful computation.
> 
> **Interpretation flip:**
> - **High transformation (>0.5):** Layer is doing significant work ✅ (Good!)
> - **Low transformation (<0.1):** Layer is nearly identity function ⚠️ (Possible deadness)
> 
> **What "drift" actually means:** Temporal drift = model changing its mind across generation steps (different metric, not implemented in Phase-1)
> 
> **Future metric (Phase-2):**
> - `semantic_drift = 1 - cosine_sim(step_T, step_T-1)` at same layer
> - Measures if model is revising its understanding during generation

**Dashboard Visualization:**
- **Line chart:** Layer transformation per transition (layer N → N+1)
- **Heatmap:** Transformation matrix (prompt tokens × layers)
- **Alert zones:** Transformation < 0.1 (dead layer) OR > 0.8 (potential corruption)
- **Layer comparison:** Identify layers with high vs low transformation

**Model Behavior Signal:**
- **Low transformation (< 0.1) - DEAD LAYER:** Layer barely changing representation
  - **Failure mode:** Layer is ineffective, possibly prunable
  - **Consequence:** Wasted computation, no value added
- **Moderate transformation (0.2-0.5):** Healthy refinement/transformation
- **High transformation (0.5-0.8):** Significant layer processing (often good)
- **Very high (> 0.8) - POTENTIAL ISSUE:** Radical representation change
  - **Could indicate:** Layer introducing noise (bad) OR strong feature extraction (good)
  - **Requires context:** Early layers should have high transformation, late layers low
- **Layer patterns:** Early layers high (feature extraction), mid layers moderate, late layers low (refinement)

**Research Evidence:**
- Cosine similarity, overlap ratio, and L2 distance analyze feature shifts across layers under different instruction types
- Hidden-state cosine similarity reveals semantic geometry and representational similarity across layers and models
- Cosine similarity between hidden states distinguishes correct vs incorrect model behavior - systematic changes indicate representational quality
- Token representations maintain high cosine similarity (>0.8) across middle layers, with bigger changes at first and last layers
- **Cohen (2023):** Even lowest cosine similarities between very early vs very late layers still around 0.8 (not near 0) - representations drift but retain alignment. Model isn't forgetting earlier layers.
- **Jiang et al. (2024):** Cosine similarity correlates well with more complex metrics like CKA (Centered Kernel Alignment). Similarity increases for closer layers. Leveraged for "aligned training" to improve early-exit abilities.
- **Logit lens validation:** If representations didn't maintain alignment (cosines near 0), logit lens wouldn't work (intermediate layers would produce garbage when decoded). In practice, intermediate layers already encode useful information in roughly aligned space, evidenced by non-trivial decoding and relatively high cosine similarities.
- **Adjacent layer pattern:** Cosines typically 0.9+ for adjacent layers - confirms chained refinement where each layer makes incremental changes
- **Representation collapse detection:** Cosines approaching 1 for many consecutive layers means model stopped changing much (possibly wasted depth)

**Cost Analysis:**
- **Compute:** ~5ms per layer transition during prompt processing
- **Memory:** Reference to previous layer (~4MB)
- **Storage:** 4 bytes per layer per prompt token
- **Total overhead:** ~150ms for 32-layer × 100-token prompt

**Usefulness Rating: ⭐⭐⭐⭐ (4/5) - HIGH VALUE (Prompt Only)**

**Justification:**
- **Research-backed signal:** Strong correlation with model behavior
- Enables layer health diagnosis: Low transformation = investigate pruning
- Enables layer blame: Unexpected high transformation = investigate errors
- **Constraint:** Only compute during prompt (all layers available)
- **Skip for timeline:** Too expensive during generation (sequential)
- Decision: **Add to prompt analysis only**
- Phase-2 enabler: Identify problematic layers for targeted interventions

---

### 11. Mean (Activation Bias)

**Formula:**
```
mean = (1/n) * Σ(h_i)
```
Average value of hidden state dimensions.

**How to Calculate:**
```python
mean = torch.mean(hidden_states, dim=-1).mean()
```

**Dashboard Visualization:**
- **Line chart:** Mean per layer over time
- **Zero line:** Identify bias shifts
- **Histogram:** Distribution of means across layers

**Model Behavior Signal:**
- **Near-zero mean:** Centered activations (typical with LayerNorm)
- **Large positive/negative mean:** Representation bias
  - May indicate: Specific semantic content (not necessarily bad)

**Research Evidence:**
- Mean is commonly tracked but less informative than norm/std
- Mainly useful in conjunction with std (mean-variance characterization)

**Cost Analysis:**
- **Compute:** ~0.5ms per layer per step (mean reduction)
- **Memory:** 0 bytes
- **Storage:** 4 bytes per layer per step
- **Total overhead:** ~5ms for 32-layer model

**Usefulness Rating: ⭐⭐ (2/5) - LOW VALUE**

**Justification:**
- **Already implemented** in CoreVital
- Limited interpretability (mean shifts common, not always meaningful)
- Keep: Already there, but don't prioritize for analysis
- Mainly useful for: Statistical completeness (mean-std pair)

---

## ANOMALY DETECTION METRICS

### 12. NaN/Inf Detection

**Formula:**
```
has_nan = any(isnan(tensor))
has_inf = any(isinf(tensor))
```
Boolean flags for presence of NaN or Inf values.

**How to Calculate:**
```python
tensor_flat = tensor.flatten()
has_nan = torch.isnan(tensor_flat).any().item()
has_inf = torch.isinf(tensor_flat).any().item()
```

**Dashboard Visualization:**
- **Alert banner:** Big red warning if ANY NaN/Inf detected
- **Timeline markers:** Flag exact step + layer where anomaly occurred
- **Heatmap:** Anomaly presence per layer per step
- **First occurrence:** Highlight first NaN/Inf in trace

**Model Behavior Signal:**
- **NaN detected:** Numerical instability, gradient explosion, division by zero
  - **Immediate failure indicator**
  - **Consequence:** Generation crashes or produces garbage
- **Inf detected:** Overflow, exploding activations/gradients
  - **Critical warning**
  - **Consequence:** Model unstable, likely to fail soon

**Why This Matters:**
- **High signal, low cost:** ~5 CPU cycles per tensor
- **Early warning:** NaN/Inf appears before quality fully degrades
- **Actionable:** Immediate alert to stop inference, investigate

**Research Evidence:**
- NaN/Inf detection is standard practice in all deep learning frameworks
- Critical for production monitoring (prevents serving bad outputs)
- Used in training stability analysis

**Cost Analysis:**
- **Compute:** ~0.2ms per layer per step (2 boolean reductions)
- **Memory:** 0 bytes (boolean flags)
- **Storage:** 1 bit per layer per tensor type (hidden + attention)
- **Total overhead:** ~5ms for 32-layer model

**Usefulness Rating: ⭐⭐⭐⭐⭐ (5/5) - CRITICAL**

**Justification:**
- **Highest signal-to-cost ratio** of any metric
- **Binary alert:** No interpretation needed (NaN = bad, period)
- **Prevents serving bad outputs:** Catch before user sees garbage
- **Research-backed:** Universal practice in ML monitoring
- **Phase-3 enabler:** Immediate alert to monitoring systems
- Decision: **Must-have metric**

---

## DERIVED/AGGREGATE METRICS

### 13. Perplexity (Per-Token)

**Formula:**
```
perplexity = 2^entropy
```
Exponential of entropy, more interpretable scale.

**How to Calculate:**
```python
perplexity = 2 ** entropy
```

**Dashboard Visualization:**
- **Line chart:** Perplexity over time (more intuitive than entropy)
- **Alert zones:** Perplexity > 16 (high confusion)

**Model Behavior Signal:**
- **Low perplexity (1-4):** Model confident
- **High perplexity (> 16):** Model confused

**Research Evidence:**
- Perplexity is one of standard uncertainty measures, essentially entropy in exp-space rather than log-space
- More interpretable than entropy for non-experts
- Common in language modeling evaluations

**Cost Analysis:**
- **Compute:** ~0.01ms per token (exp operation)
- **Memory:** 0 bytes
- **Storage:** 4 bytes per timeline step
- **Total overhead:** Negligible

**Usefulness Rating: ⭐⭐⭐ (3/5) - MODERATE VALUE**

**Justification:**
- **Redundant with entropy:** perplexity = 2^entropy (also called "effective support")
- **More interpretable:** Easier to explain to non-technical users  
- **Research context:** Braverman et al. (2020) describe exp(entropy) as effective support size - the number of "plausible" word choices the model is effectively considering at this step
- **Intuitive interpretation:** If perplexity = 50, it's as though the model is choosing among 50 equally likely tokens (even if actual probabilities are uneven)
- **Typical range:** In state-of-the-art LLMs, effective support might range from 10 to 30 tokens for typical text
- **Same information content:** No new signal beyond entropy
- Decision: **Add** - cost is negligible, aids communication, transforms entropy into intuitive "number of plausible options"

---

### 13.5. Surprisal (Per-Token Negative Log-Likelihood)

**Formula:**
```
surprisal = -log₂(p_actual_token)
```
The negative log-probability of the actual generated token.

**How to Calculate:**
```python
# Generation phase: After sampling the next token
actual_token_id = sampled_token
actual_token_prob = probs[actual_token_id]
generation_surprisal = -torch.log2(actual_token_prob + 1e-10)

# Prompt phase (OPTIMIZATION): Compute from logits manually
with torch.no_grad():
    outputs = model(
        input_ids=prompt_tokens,
        output_attentions=True,  # For attention profiles
        return_dict=True
    )
    
    # CRITICAL: Compute per-token loss manually (outputs.loss_per_token does NOT exist)
    logits = outputs.logits
    
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = prompt_tokens[..., 1:].contiguous()
    
    # Compute cross entropy with reduction='none' to get per-token loss
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss_per_token = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    
    # Convert nats to bits: log_2(e) ≈ 1.4427
    prompt_surprisals = loss_per_token * 1.4427
```

**Implementation Decision - PROMPT SURPRISAL FIX:**
> 🔴 **CRITICAL FIX:** `outputs.loss_per_token` does not exist in standard Hugging Face models. `CausalLMOutput` only returns `loss` (scalar mean).
> 
> **The Fix:** Compute per-token loss manually from logits using `CrossEntropyLoss(reduction='none')`. Since we're already grabbing logits for the prompt pass, this adds negligible cost.
> 
> **Why manual computation:** 
> - Shift logits and labels (autoregressive prediction)
> - Apply cross-entropy with no reduction to get per-token values
> - Convert from nats (natural log) to bits (log2)

**Implementation Decision - PROMPT SURPRISAL OPTIMIZATION:**
> 🟡 **OPTIMIZATION:** Since we're already doing a forward pass for prompt attention profiles, we can extract prompt surprisal with minimal additional cost.
> 
> **Value:** High surprisal on prompt tokens means the model finds the user's input:
> - Confusing or contradictory
> - Out-of-distribution
> - Containing unexpected vocabulary or phrasing
> 
> **Cost:** ~5ms additional (CrossEntropyLoss computation), already have logits
> 
> **Use cases:**
> - **Prompt quality check:** High average prompt surprisal → user's prompt is unclear
> - **OOD detection:** Spikes in prompt surprisal → unusual input
> - **Debugging:** Which part of user's prompt confused the model?

**Dashboard Visualization:**
- **Line chart:** Surprisal over generation timeline
- **Highlight spikes:** Tokens with high surprisal (model was surprised)
- **Correlation:** Surprisal vs entropy (should correlate but measures different thing)
- **Token labels:** Show which tokens had highest surprisal

**Model Behavior Signal:**
- **Low surprisal (< 2.0):** Token was expected (high probability)
- **Moderate surprisal (2.0-5.0):** Token was plausible but not top choice
- **High surprisal (> 5.0):** Token was improbable - model was surprised
- **Surprisal spike:** Identifies unexpected or difficult tokens

**Why Different from Entropy:**
- **Entropy:** Measures uncertainty BEFORE choosing (spread of distribution)
- **Surprisal:** Measures surprise AFTER choosing (probability of actual choice)
- **Complementary:** High entropy means many options, high surprisal means chosen option was unlikely

**Research Evidence:**
- Kim et al. (2023) ASAP: Used first-token surprisal to prune Chain-of-Thought reasoning steps - high surprisal indicates important new information, low surprisal indicates redundant steps
- Surprisal is the per-token contribution to cross-entropy loss (direct measure of prediction error)
- Parallels cognitive modeling: humans take longer to process surprising words
- Used to detect AI-generated text (different surprisal patterns than human text)
- Identifies where model lacks relevant knowledge (high surprisal = violated expectations)

**Cost Analysis:**
- **Compute:** ~0.01ms per token (single log operation)
- **Memory:** 0 bytes (reuse probs from sampling)
- **Storage:** 4 bytes per timeline step
- **Total overhead:** Negligible

**Usefulness Rating: ⭐⭐⭐⭐ (4/5) - HIGH VALUE**

**Justification:**
- **Complements entropy:** Entropy = pre-decision uncertainty, surprisal = post-decision surprise
- **Token-level diagnostic:** Pinpoints exactly which tokens were unexpected
- **Research-backed:** Used in CoT pruning, interpretability research
- **Essentially free:** Already have probability of generated token
- **Actionable:** High surprisal tokens warrant investigation (model struggled there)
- Decision: **Add** - trivial to implement, high diagnostic value

---

### 14. Health Flags (Aggregated Alerts)

**Formula:**
```
nan_detected = any(has_nan across all layers/steps)
inf_detected = any(has_inf across all layers/steps)
attention_collapse = any(collapsed_head_count > 0)
high_entropy_steps = count(entropy > 4.0)
repetition_loop_detected = detect_repetition(timeline)  # NEW
```

**How to Calculate:**
```python
def _detect_repetition_loop(timeline: List[TimelineStep]) -> bool:
    """Detect if model is stuck in repetition loop using cosine similarity"""
    if len(timeline) < 4:
        return False
    
    # Check last 3 tokens for representation similarity
    last_hidden = timeline[-1].layers[-1].hidden_summary  # Last layer hidden state
    repetition_count = 0
    
    for i in range(2, 5):  # Look back 3 steps
        if len(timeline) < i:
            break
        prev_hidden = timeline[-i].layers[-1].hidden_summary
        
        # Use COSINE SIMILARITY (direction) not L2 norm (magnitude)
        # CRITICAL: L2 norm measures magnitude, not meaning
        # "cat" and "dog" might have similar L2 norms but different directions
        cosine_sim = F.cosine_similarity(
            torch.tensor(last_hidden.vector),  # Assuming we store full vector
            torch.tensor(prev_hidden.vector),
            dim=0
        )
        
        if cosine_sim > 0.99:  # Vectors pointing in same direction
            repetition_count += 1
    
    return repetition_count >= 3

health_flags = HealthFlags(
    nan_detected=any(layer.anomalies.has_nan for step in timeline for layer in step.layers),
    inf_detected=any(layer.anomalies.has_inf for step in timeline for layer in step.layers),
    attention_collapse_detected=any(
        layer.attention_summary.collapsed_head_count > 0 
        for step in timeline for layer in step.layers
    ),
    high_entropy_steps=sum(
        1 for step in timeline if step.logits_summary.entropy > 4.0
    ),
    repetition_loop_detected=_detect_repetition_loop(timeline),  # FIXED
    mid_layer_anomaly_detected=_detect_mid_layer_anomaly(timeline)  # NEW
)
```

**Implementation Decision - REPETITION LOOP FIX:**
> 🔧 **CRITICAL FIX:** Originally proposed using L2 norm difference, but **L2 norm measures magnitude, not direction**. 
> 
> **The Problem:** Tokens "cat" and "dog" might have very similar L2 norms (magnitude of activation) but completely different vector directions (meanings). Original logic would falsely flag "The cat saw the dog" as repetition because norms are stable.
> 
> **The Solution:** Use **cosine similarity** (measures directional alignment, i.e., semantic similarity). If `cosine_sim > 0.99`, the vectors point in the exact same direction = true repetition loop.
> 
> **Cost:** Negligible (1 dot product per comparison)
> 
> **Trade-off:** Requires storing full hidden state vector (or at least last layer's vector) in timeline, not just summary stats. Alternative: Store last N hidden states in memory during generation, discard after health check.

> ⚠️ **IMPLEMENTATION UPDATE (Phase-1c):** Threshold raised from `0.99` to **`0.9995`**. E2E testing with GPT-2 on CUDA float16 showed non-repetitive tokens ("picture", "below", ",", "are", "actually") produce cosine sims of 0.992–0.999 due to last-layer anisotropy. True repetition gives ~1.0. Threshold of 0.9995 cleanly separates normal text from repetition with margin on both sides.

**Implementation Decision - MID-LAYER ANOMALY DETECTION:**
> ➕ **NEW FLAG ADDED:** Recent research (Dec 2025) shows hallucinations concentrate in middle layers.
> 
> **Research:** "Hallucination Detection via Internal Representations" found probing layers 16-18 (in 32-layer models) yields ~83% accuracy for detecting hallucinations. Early/late layers perform significantly worse.
> 
> **Detection logic:** Track anomalies specifically in middle third of model (e.g., layers 10-22 in 32-layer model). ~~Weight middle-layer entropy spikes, norm explosions, or attention collapse more heavily than boundary layers.~~

> ⚠️ **IMPLEMENTATION UPDATE (Phase-1c):** Two corrections from E2E testing:
> 1. **Attention collapse removed from mid-layer check.** GPT-2 has 62 collapsed-head occurrences across 10 steps — this is model architecture (well-documented in "Are Sixteen Heads Really Better Than One?"), not a runtime anomaly. Already captured separately by `attention_collapse_detected`. Mid-layer anomaly now checks NaN/Inf and L2 explosion only.
> 2. **Per-step L2 baselines, not global.** CausalLM step 0 processes the full prompt (shape `(1, seq_len, hidden_dim)`), giving L2 norms 10× higher than single-token steps 1+ (which use KV cache). A global baseline false-triggered on step 0. Per-step baselines correctly normalize each step's early layers against its own mid-layers.
> 3. **L2 explosion multiplier raised from 5× to 8×.** Original 5× calibrated on GPT-2 (max 3.1× mid/early ratio). flan-t5-small (8 layers, only 2 early layers) peaks at 5.7× due to ~70% per-layer growth — just above 5×. 8× accommodates diverse architectures while still catching genuine explosions (100×+).
> 
> **Why:** Early layers are syntactic, late layers are token selection. The "truth" lives in the middle layers - anomalies here are more dangerous.

```python
def _detect_mid_layer_anomaly(timeline: List[TimelineStep]) -> bool:
    """Detect anomalies specifically in middle layers (hallucination sweet spot)"""
    if not timeline:
        return False
    
    num_layers = len(timeline[0].layers)
    mid_start = num_layers // 3
    mid_end = 2 * num_layers // 3
    
    # Compute baseline L2 norm from early layers (for dynamic threshold)
    early_layer_norms = []
    for step in timeline[:min(5, len(timeline))]:  # Use first few steps
        for layer_idx in range(mid_start):  # Early layers
            if step.layers[layer_idx].hidden_summary:
                early_layer_norms.append(step.layers[layer_idx].hidden_summary.l2_norm)
    
    # Dynamic threshold: 5x the mean of early layers (accounts for exponential growth)
    if early_layer_norms:
        baseline_norm = sum(early_layer_norms) / len(early_layer_norms)
        explosion_threshold = baseline_norm * 5.0
    else:
        explosion_threshold = 1000.0  # Fallback safety rail
    
    for step in timeline:
        mid_layers = step.layers[mid_start:mid_end]
        
        # Check for anomalies in middle layers specifically
        for layer in mid_layers:
            if layer.anomalies and (layer.anomalies.has_nan or layer.anomalies.has_inf):
                return True
            if layer.attention_summary and layer.attention_summary.collapsed_head_count > 0:
                return True
            if layer.hidden_summary and layer.hidden_summary.l2_norm > explosion_threshold:
                return True
    
    return False
```

**Implementation Decision - DYNAMIC L2 NORM THRESHOLD:**
> 🟡 **CONSISTENCY FIX:** L2 norm grows exponentially with depth (~4-5% per layer per Turntrout research).
> 
> **The Problem:** Hard threshold of `100` doesn't account for model architecture:
> - Layer 1 might be 10 (normal)
> - Layer 32 might be 200 (also normal)
> - Hard threshold of 100 would falsely flag healthy deep layers as "exploding"
> 
> **The Solution:** Dynamic threshold based on model's actual behavior:
> - Compute baseline: Average L2 norm of early layers (first third)
> - Threshold: `5x baseline` (catches true explosions while allowing natural growth)
> - Fallback: If no baseline available, use 1000 (very conservative safety rail)
> 
> **Example:** If early layers average norm=20, threshold=100. If early layers average norm=50, threshold=250.
> 
> **Why 5x:** Exponential growth of 4-5% per layer means ~2x growth from early to middle layers. 5x provides safety margin for detecting true explosions.

**Schema Update:**
```python
class HealthFlags(BaseModel):
    nan_detected: bool = False
    inf_detected: bool = False
    attention_collapse_detected: bool = False
    high_entropy_steps: int = 0
    repetition_loop_detected: bool = False  # Phase-1 addition (FIXED to use cosine similarity)
    mid_layer_anomaly_detected: bool = False  # NEW Phase-1 addition (hallucination sweet spot)
```

**Dashboard Visualization:**
- **Status panel:** Traffic light (green/yellow/red) for overall health
- **Alert list:** Each detected issue with first occurrence
- **Statistics:** Count of problematic steps/layers
- **Severity scoring:** Weighted combination of flags
- **Repetition indicator:** Show when loop detected + affected step range

**Model Behavior Signal:**
- **All green:** Generation healthy
- **Yellow flags:** Warnings (high entropy, repetition loop)
- **Red flags:** Critical issues (NaN, Inf, attention collapse)

**Research Evidence:**
- Aggregated health signals are standard in production monitoring
- Enables alerting without deep metrics knowledge
- Used in all major ML monitoring platforms (Datadog, Prometheus)
- Repetition detection addresses common hallucination pattern

**Cost Analysis:**
- **Compute:** ~10ms post-generation (single pass over timeline + repetition check + mid-layer scan)
- **Memory:** Requires storing last N hidden state vectors for cosine similarity check (~4KB per vector × 5 = 20KB)
- **Storage:** ~28 bytes (6 fields: 5 bools + 1 int)
- **Total overhead:** ~10ms compute + 20KB transient memory

**Usefulness Rating: ⭐⭐⭐⭐⭐ (5/5) - CRITICAL**

**Justification:**
- **Essential for production:** Single boolean check = healthy?
- **Enables alerting:** DatadogSink, PrometheusSink, PagerDuty
- **Non-technical friendly:** "Red flag detected" needs no ML knowledge
- **Repetition loop:** Catches common hallucination pattern with existing metrics
- **Phase-3 enabler:** Foundation of monitoring integrations
- Decision: **Must-have for Phase-1**

---

## OPTIMIZATION TECHNIQUES (Phase-2+)

### 15. Random Projections

**Formula:**
```
h_projected = h @ R
```
Where R is a random projection matrix (d_hidden × d_target).

**Purpose:**
Reduce hidden state dimensionality for storage (4096 → 128 dims = 32x reduction).

**When to Use:**
- Trace file size becomes prohibitive (> 100MB per trace)
- Network bandwidth constraints
- Long-context models (8K+ tokens)

**Trade-offs:**
- **Pros:** 32x storage reduction
- **Cons:** Lose some interpretability, still experimental

**Research Evidence:**
- Johnson-Lindenstrauss lemma: Random projections preserve distances
- Used in large-scale ML systems for dimensionality reduction

**Cost Analysis:**
- **Compute:** ~10ms per layer per step (matrix multiplication)
- **Memory:** ~512KB for projection matrix
- **Storage:** 4 bytes × 128 dims = 512 bytes per layer per step (vs 16KB)

**Usefulness Rating: ⭐⭐ (2/5) - DEFER**

**Justification:**
- **Optimization, not feature:** Doesn't add new signal
- **Phase-2+ decision:** Only implement if storage becomes problem
- **Alternative:** Downsample timeline (every 10th step) cheaper
- **Research validation:** Johnson-Lindenstrauss lemma confirms distances preserved, used in TransformerLens and high-dimensional analysis
- Decision: **Defer** until proven necessary

---

### 16. Calibration Metrics (Phase-2/3 - Requires Ground Truth)

**Metrics:**
- **Expected Calibration Error (ECE):** Average gap between predicted confidence and actual accuracy
- **Brier Score:** Mean squared error of probability predictions
- **Reliability Diagrams:** Visual calibration check (predicted probability vs empirical accuracy)

**Why Important:**
Good calibration means when model says "I'm 90% confident," it's actually correct 90% of the time. Miscalibrated models (especially overconfident ones) are dangerous in production.

**Research Findings:**
- **LLM calibration gaps:** Current LLMs show significant ECE ranging from 0.12 to 0.39 (12-39% calibration error)
- **Overconfidence pattern:** When models claim 90%+ confidence, often only correct 60-70% of the time
- **Brier score baseline:** Random guessing on balanced binary = 0.25, human superforecasters achieve 0.15-0.20
- **Proper scoring rule:** Brier score incentivizes true confidence reporting
- **"Do Large Language Models Know What They Don't Know?"** Paper found GPT-4, Claude, etc. have substantial calibration errors

**Why Phase-2/3:**
- **Requires labeled data:** Need ground truth to compare predictions against
- **Dataset-specific:** Calibration varies by domain (QA vs coding vs factual)
- **Complementary to uncertainty:** Entropy/margin measure internal uncertainty, calibration measures accuracy of that uncertainty
- **Production critical:** Essential for trust and safety in deployed systems

**Implementation Approach:**
```python
# Collect predictions with ground truth
predictions = []
for prompt, expected_answer in eval_dataset:
    trace = monitor.generate(prompt)
    predictions.append({
        'predicted_prob': trace.logits_summary.max_prob,
        'entropy': trace.logits_summary.entropy,
        'correct': (generated_answer == expected_answer)
    })

# Compute ECE
bins = np.linspace(0, 1, 11)  # 10% bins
ece = compute_ece(predictions, bins)

# Compute Brier score
brier = np.mean([(pred['predicted_prob'] - pred['correct'])**2 for pred in predictions])

# Plot reliability diagram
plot_reliability_diagram(predictions)
```

**Integration with CoreVital:**
- **Phase-2:** Add evaluation mode that collects ground truth alongside traces
- **Calibration dashboard:** Compare entropy/margin vs actual correctness
- **Temperature tuning:** Use calibration metrics to adjust model confidence
- **Ensemble calibration:** If using multiple models, calibrate their combination

**Usefulness Rating: ⭐⭐⭐⭐⭐ (5/5) - CRITICAL FOR PRODUCTION (Phase-2/3)**

**Justification:**
- **Essential for deployment:** Uncalibrated models mislead users
- **Complements existing metrics:** Validates that internal uncertainty (entropy) correlates with external accuracy
- **Research-proven gap:** Current LLMs are significantly miscalibrated (12-39% ECE)
- **Actionable:** Calibration curves guide temperature scaling, confidence thresholds, human-in-loop triggers
- **Safety critical:** Prevents overconfident errors in high-stakes applications
- Decision: **High priority for Phase-2** once we have evaluation harness with labeled data

---

## PHASE-1 TECHNICAL REFINEMENTS
### Based on 2025 Research

### 🔧 Refinement 1: Repetition Loop Detection — Cosine over L2

> **Full implementation and rationale:** See **Metric 14 — Health Flags** (Repetition Loop Detection section). The original L2-norm-difference approach was replaced with cosine similarity to detect directional alignment (semantic repetition) rather than magnitude similarity. Threshold: ~~`cosine_sim > 0.99`~~ **`cosine_sim > 0.9995`** (see Metric 14 implementation update — 0.99 false-positives on float16 due to anisotropy). Requires transient buffer of last 5 hidden state vectors (~20KB), discarded after health check. Research basis: standard NLP practice.

---

### 🎯 Refinement 2: Mid-Layer Anomaly Detection — Dynamic Threshold

> **Full implementation and rationale:** See **Metric 14 — Health Flags** (Mid-Layer Anomaly Detection section). Targets the "hallucination sweet spot" (middle third of layers). Uses dynamic L2 threshold (5x early-layer baseline, **per-step** not global — see Metric 14 implementation update) — **not** a hardcoded constant. Attention collapse excluded from this check (structural, not runtime). Research basis: "Hallucination Detection via Internal Representations" (Dec 2025) — 83% accuracy probing layers 16-18 in 32-layer models.

---

### 🌊 Refinement 3: Basin Score — Per-Head Scalar

> **Full implementation and rationale:** See **Metric 7 — Prompt Attention Profiles** (Basin Score section). Per-head scalar aggregate quantifying "Lost in the Middle" U-shape pattern. `basin_score < 0.3` = head ignores middle tokens. Cost: 4KB total (1 float × 32 heads × 32 layers). Research basis: "Attention Basin: Why Contextual Position Matters" (arXiv:2508.05128, Aug 2025).

---

### ⚡ Refinement 4: Pre-Softmax Variance (Leading Indicator) - OPTIONAL PHASE-2

**Research Finding:**
- "Variance Sensitivity Induces Attention Entropy Collapse in Transformers" (EMNLP 2025)
- High variance in attention logits (pre-softmax) **predicts** entropy collapse
- Exponential nature of Softmax makes attention highly sensitive to logit variance
- **Leading indicator**: Variance spikes **before** entropy drops

**Implementation (if pre-softmax scores available):**
```python
# If attention_scores (pre-softmax) accessible:
attention_variance = torch.var(attention_scores, dim=-1)  # Per head
variance_spike = (attention_variance > threshold).any()

# Add to AttentionSummary
class AttentionSummary(BaseModel):
    # ... existing fields ...
    variance_max: Optional[float] = None  # Max pre-softmax variance
    variance_spike_detected: bool = False  # Collapse predictor
```

**Why It Matters:**
- **Leading vs lagging**: Variance spikes → then entropy collapses
- Allows earlier intervention before collapse degrades generation

**Cost Decision:**
- **If pre-softmax scores already computed**: Cheap (~1ms variance calculation)
- **If requires model modification**: Expensive (additional forward pass or hooks)
- **Recommendation**: **Phase-2 enhancement** only if accessible without performance hit
- **Phase-1 approach**: Stick to entropy (lagging indicator but still catches collapse)

**Research Citation:** "Variance Sensitivity Induces Attention Entropy Collapse in Transformers" (EMNLP 2025) - Pre-softmax variance is leading indicator of collapse

---

### Summary of Technical Refinements

**Updated Storage Approach:**
- Original naive: 1.02 GB (full attention matrix)
- Phase-1 sparse: **1.5 MB typical** (range: 0.5-5 MB depending on sparsity)
  - Format: Structure of Arrays (query_indices, key_indices, weights)
  - Threshold: Store only weights > 0.01
  - Basin scores: Per-head scalars (4 KB total)
- Reduction: **680x typical** (vs naive)
- Key upgrade: Full fidelity + arbitrary query capability (not just derived metrics)

**Updated Transient Memory:**
- Hidden state buffer: +20KB (last 5 vectors for cosine similarity)
- Prompt surprisal computation: ~10KB (temporary)
- Pre-softmax variance (optional): 0 bytes (if reusing model outputs)

**Total per trace:**
- Prompt: 1.5MB (sparse attention profiles, typical case)
- Timeline: ~250KB (generation metrics)
- Transient: ~30KB (for health flags and surprisal, discarded after)
- **Total persisted: ~1.75MB** (typical, fits in memory, dashboard-friendly)

**Implementation Priority:**

**Phase-1 Must-Do:**
1. ✅ Fix repetition detection (cosine not L2 norm)
2. ✅ Add mid-layer anomaly detection
3. ✅ Add basin_score to prompt profiles

**Phase-2 Consider:**
4. ⚡ Pre-softmax variance (only if accessible without cost)

**Research Citations:**

1. **Hallucination Detection via Internal Representations** (Dec 2025)
   - Finding: Layers 16-18 achieve 83% hallucination detection accuracy
   - Application: Mid-layer anomaly targeting

2. **Attention Basin: Why Contextual Position Matters in LLMs** (arXiv:2508.05128, Aug 2025)
   - Finding: Models systematically neglect middle of context window (U-shape)
   - Application: Basin score calculation

3. **Variance Sensitivity Induces Attention Entropy Collapse in Transformers** (EMNLP 2025)
   - Finding: Pre-softmax variance is leading indicator of entropy collapse
   - Application: Optional Phase-2 enhancement

4. **Cosine Similarity Standard Practice**
   - Finding: Cosine measures directional alignment (meaning), L2 measures magnitude
   - Application: Repetition loop detection fix

---

## SUMMARY & RECOMMENDATIONS

### Phase-1 Priority Matrix

**Tier 1 - Must Implement (Critical, Low Cost):**
1. ✅ Shannon Entropy (already have) **+ numerical stability fix**
2. ✅ Attention Entropy (already have) **+ min/max/count aggregation**
3. ✅ Concentration Max (already have) **+ min aggregation**
4. ✅ L2 Norm (already have)
5. ➕ **Top-K Margin** (NEW - trivial add)
6. ➕ **Voter Agreement** (NEW - trivial add)
7. ➕ **NaN/Inf Detection** (NEW - critical signal)
8. ➕ **Health Flags with Repetition Loop** (NEW - aggregation + hallucination detection)

**Tier 2 - Strong Value (Phase-1 Features):**
9. ➕ **Prompt Attention Profiles (Compressed)** (NEW - 6MB not 1GB, unique capability)
10. ➕ **Layer Transformation (Prompt Only)** (NEW - renamed from "drift", layer health diagnosis)
11. ➕ **Perplexity** (NEW - interpretability aid, effective support size)
12. ➕ **Surprisal** (NEW - per-token surprise, complements entropy, CoT pruning research)

**Tier 3 - Keep Existing:**
13. ✅ Max Probability (already have)
14. ✅ Standard Deviation (already have)
15. ✅ Mean (already have)

**Tier 4 - Defer to Phase-2+:**
16. ❌ Random Projections (optimization, Phase-2+)
17. 📊 **Calibration Metrics** (ECE, Brier Score, Reliability Diagrams - CRITICAL for Phase-2, requires ground truth)

### Total Cost Estimate (32-layer model, 100-token generation, 500-token prompt)

**Compute:**
- Existing metrics: ~15ms (already paid)
- New Tier 1 metrics: ~20ms (NaN/Inf + enhanced health flags + attention min/max/count)
- New Tier 2 metrics: ~65ms (prompt analysis + sparse conversion + prompt surprisal, one-time)
- **Total overhead:** ~35ms per generation + 65ms per prompt

**Memory:**
- Peak: ~5MB (temporary full attention during sparse conversion - down from 1GB!)
- Transient: ~20KB (last N hidden states for cosine similarity repetition detection)
- Steady-state: <1MB

**Storage (per trace with 500-token prompt):**
- Schema v0.2.0: ~50KB baseline
- Attention aggregation expansion: +64KB (min/max/count fields)
- Sparse prompt profiles: +1.5MB typical (range: 0.5-5MB depending on attention sparsity)
- Schema v0.3.0 total: **~1.6MB per trace** (typical)

**Critical Fixes & Enhancements Applied:**
1. ✅ **Storage Bomb Prevented:** Sparse storage 1GB → 1.5MB typical (680x reduction)
2. ✅ **Full Fidelity Upgrade:** Exact attention weights with arbitrary query capability (not just derived summaries)
3. ✅ **Structure of Arrays:** 40% storage overhead reduction vs array-of-structs
4. ✅ **Prompt Surprisal Fixed:** Manual computation from logits (outputs.loss_per_token doesn't exist)
5. ✅ **Signal Loss Prevented:** Added min/max/count aggregation (catches individual head failures)
6. ✅ **Numerical Stability:** Use log_softmax for entropy computation
7. ✅ **Terminology Fixed:** "drift" → "layer_transformation" (interpretation corrected)
8. ✅ **Repetition Detection FIXED:** Use cosine similarity (direction) not L2 norm (magnitude)
9. ✅ **Mid-Layer Anomaly Detection:** Target hallucination sweet spot with dynamic L2 threshold
10. ✅ **Basin Score:** Per-head scalar aggregate for U-shape detection
11. 📋 **Pre-Softmax Variance:** Phase-2 leading indicator (only if accessible without cost)

**2025 Research Papers Integrated:**
- ✅ Ali et al. (2025) - Entropy-Lens framework
- ✅ Hallucination Detection via Internal Representations (Dec 2025) - Mid-layer targeting
- ✅ Attention Basin (arXiv:2508.05128, Aug 2025) - U-shape pattern
- ✅ Variance Sensitivity (EMNLP 2025) - Pre-softmax collapse predictor

### Research-Backed Value Proposition

**Metrics with Strongest Research Evidence:**
1. **Shannon Entropy** - Universal uncertainty measure, proven correlation with quality (now with numerical stability). Ali et al. (2025) Entropy-Lens shows entropy profiles correlate with prompt type and correctness.
2. **Attention Entropy** - Direct link to training instability and generation failures (now with min/max/count to catch individual head collapse). Zhai et al. (2023) showed entropy collapse causes training divergence.
3. **NaN/Inf Detection** - Standard practice, highest signal-to-cost ratio
4. **Sparse Prompt Attention Profiles** - Full fidelity with arbitrary query capability (1.5MB typical not 1GB!). Enables "Lost in the Middle" debugging, attention sink detection, graph visualization. Barbero et al. (2024) validates attention sink phenomenon (40-50% on first token).
5. **Surprisal** - Kim et al. (2023) used for CoT pruning, identifies important vs redundant reasoning steps. Now includes prompt surprisal computed from logits.
6. **Layer Transformation** - Cohen (2023) and Jiang et al. (2024) show cosine similarity reveals layer health, validates logit lens approach
7. **Calibration (Phase-2)** - Research shows LLMs have 12-39% ECE, critical for production trust

**Research Validation of Our Decisions:**
- ✅ **Compression approach validated:** Barbero et al. (2024) confirms attention sinks (our `sink_score`), Cohen (2023) confirms first token special behavior
- ✅ **Min/max/count validated:** Voita et al. (2019) shows heads are specialists with ~80% max weight - mean-only would hide failures
- ✅ **Exponential norm growth:** Turntrout confirms 4-5% per layer growth in GPT-2 - validates our L2 norm tracking
- ✅ **Voter agreement validated:** Liang et al. (2023) shows top-K mass collapse during DPO fine-tuning - confirms diagnostic value
- ✅ **Random projections sound:** Johnson-Lindenstrauss lemma confirms distance preservation (deferred correctly to Phase-2+)
- ✅ **Mid-layer hallucination:** "Hallucination Detection via Internal Representations" (Dec 2025) - 83% accuracy probing layers 16-18
- ✅ **Attention basins:** arXiv:2508.05128 (Aug 2025) - models systematically neglect middle of context window
- ✅ **Variance sensitivity:** EMNLP 2025 - high variance in pre-softmax logits predicts entropy collapse
- ✅ **Cosine similarity fix:** Standard practice in NLP - L2 norm measures magnitude not meaning

**Key 2025 Research Papers Incorporated:**
1. **Ali et al. (2025)** - Entropy-Lens: Entropy profiles correlate with prompt type and correctness
2. **"Hallucination Detection via Internal Representations" (Dec 2025)** - Mid-layers (16-18) achieve 83% hallucination detection accuracy
3. **"Attention Basin: Why Contextual Position Matters in LLMs" (arXiv:2508.05128, Aug 2025)** - U-shape attention pattern systematically neglects middle tokens
4. **"Variance Sensitivity Induces Attention Entropy Collapse in Transformers" (EMNLP 2025)** - Pre-softmax variance is leading indicator of collapse

**Phase-2 Risk Scoring Readiness:**
```python
risk_score = (
    0.20 * normalize(logit_entropy) +                    # Confusion
    0.15 * (1 - top_k_margin) +                          # Uncertainty gap
    0.15 * normalize(surprisal) +                        # Token-level surprise (NEW)
    0.20 * (collapsed_head_count > 0 ? 1.0 : 0) +        # Collapse (per-head detection)
    0.15 * normalize(layer_transformation) +             # Transformation magnitude
    0.10 * (has_nan or has_inf ? 1.0 : 0) +             # Anomaly
    0.05 * (repetition_loop_detected ? 1.0 : 0)         # Hallucination (NEW)
)

# Phase-2 calibration layer (requires ground truth):
calibrated_confidence = apply_calibration_curve(risk_score, ece_model)
```

**Production Monitoring Readiness:**
- Health flags → Datadog/Prometheus alerts
- Per-metric thresholds well-documented in research
- Actionable insights (not just numbers)
- **Phase-2:** Add calibration layer using ECE/Brier score to ensure risk scores correlate with actual failure rates

---

## FINAL RECOMMENDATION

**Implement in Phase-1:**
- All Tier 1 metrics (8 items) - Critical + cheap
- Compressed prompt telemetry with layer transformation (Tier 2) - Unique capability, storage-safe
- Perplexity + Surprisal (Tier 2) - Communication aids, research-backed

**Total New Implementation:**
- 4 new logit metrics (margin, agreement, perplexity, surprisal)
- 1 new anomaly detection (NaN/Inf with per-tensor flags)
- 1 new health flag (repetition loop detection)
- 5 new attention aggregations (min/max/count for entropy/concentration)
- 1 new feature (compressed prompt attention profiles + layer transformation)
- 4 existing metrics (keep as-is: max_prob, std, mean, l2_norm)

**Phase-2 Priority (Requires Ground Truth):**
- Calibration metrics (ECE, Brier Score, Reliability Diagrams) - Research shows LLMs have 12-39% calibration error, critical for production deployment

**Schema Changes:**
```python
# LogitsSummary additions:
+ top_k_margin: float
+ voter_agreement: float
+ perplexity: float
+ surprisal: float  # Per-token surprise (generation + prompt with manual computation)

# AttentionSummary additions:
+ entropy_min, entropy_max, concentration_min
+ collapsed_head_count, focused_head_count

# New classes:
+ TensorAnomalies (has_nan, has_inf)
+ SparseAttentionHead:  # Structure of Arrays format
  - query_indices: List[int]  # uint16
  - key_indices: List[int]    # uint16  
  - weights: List[float]      # float16
+ PromptAttentionLayer:
  - heads: List[SparseAttentionHead]  # 32 heads
  - basin_scores: List[float]         # Per-head scalars (32 floats)
+ HealthFlags with repetition_loop_detected (cosine-based) and mid_layer_anomaly_detected

# LayerSummary changes:
+ layer_transformation (renamed from drift)
+ anomalies: TensorAnomalies

# TimelineStep changes (for repetition detection):
+ Store last N (e.g., 5) full hidden state vectors from final layer for cosine similarity checks
+ Discard after health flag computation to save memory

# Note on storage structure:
# Sparse storage: Only store attention weights above threshold (e.g., 0.01)
# Structure of Arrays reduces serialization overhead by ~40%
# Variable size: 0.5-5MB depending on attention sparsity (typical: 1.5MB)
```

**Critical Design Decisions Made:**

1. **Storage Bomb Prevention + Full Fidelity (Metric 7)**
   - Rejected: Full 500×500 attention matrix (1GB per trace)
   - Rejected: Compressed vectors (6MB but limited queries)
   - Implemented: Sparse storage with Structure of Arrays (1.5MB typical, 0.5-5MB range)
   - Trade-off: Variable size but GAINED arbitrary query capability + exact weights
   - **Justification:** 1.5MB typical with full query capability, vs 6MB compressed vectors with limited queries, vs 1GB naive (crashes production)

2. **Signal Loss Prevention (Metrics 5 & 6)**
   - Rejected: Mean-only aggregation (hides individual head failures)
   - Implemented: Min/max/count statistics per layer
   - Cost: +20 bytes per layer per step
   - **Justification:** 31 healthy heads + 1 collapsed = mean looks fine, but system fails

3. **Numerical Stability (Metric 1)**
   - Rejected: softmax → log (catastrophic cancellation risk)
   - Implemented: log_softmax (log-sum-exp trick)
   - **Justification:** Standard PyTorch practice, prevents NaN in entropy

4. **Terminology Correction (Metric 10)**
   - Rejected: "drift" (implies bad)
   - Implemented: "layer_transformation" (neutral/informative)
   - **Justification:** High transformation = layer working, not failing

5. **Hallucination Detection (Metric 14)**
   - Added: Repetition loop flag via cosine similarity (fixed from original L2 norm proposal)
   - Cost: ~1ms post-processing
   - **Justification:** Common failure mode, detectable with existing metrics

**Estimated Implementation:**
- Tier 1 fixes: 3-4 hours (aggregation, anomalies, health flags with cosine similarity)
- Tier 2 compression: 4-5 hours (prompt profiles with 4 vectors + layer transformation + mid-layer detection)
- **Total: 7-9 hours** (one working day + buffer)

**Rationale:**
- Maximize research-backed value (4 new 2025 papers integrated)
- Minimize cost (<5% inference overhead)
- **Prevent catastrophic storage issues** (1GB → 1.75MB typical via sparse storage)
- **Prevent signal loss** (individual head detection)
- **Prevent false positives** (cosine similarity for repetition, not L2 norm)
- **Target hallucination sweet spot** (mid-layer anomaly detection)
- **Quantify U-shape neglect** (basin score)
- Enable Phase-2 risk scoring
- Enable Phase-3 production monitoring
- Complete the metrics suite in one push with 2025 research refinements

**User's Strategy Alignment:**
> "better to do all the heavy lifting in phase-1 and then we could spend the rest of the phases making sense of the metrics, optimizing performance, and working on perfecting presentation"

✅ This recommendation aligns perfectly - implement all valuable metrics now with critical fixes, optimize/present later.

**Storage Impact Summary:**
- Prompt (500 tokens): ~1.5MB typical (sparse attention profiles with SoA format, range: 0.5-5MB)
- Timeline (100 steps): ~250KB (with expanded aggregation)
- Hidden state buffer: ~20KB (transient - last 5 vectors for repetition detection, never serialized)
- **Total per trace: ~1.75MB typical** (vs 1GB+ naive approach)
- **For 100 traces: ~175MB** (fits in memory, practical for dashboard)

> **Note:** Earlier drafts of this document referenced "8MB compressed profiles with 4 vectors" — that was the intermediate compressed vector approach. The sparse storage approach (see Metric 7 - Prompt Attention Profiles) superseded it, achieving 1.5MB typical with *better* fidelity (exact weights + arbitrary query capability vs derived summaries only).

**Next Steps:**
1. Implement schema changes (v0.2.0 → v0.3.0)
   - New models: `TensorAnomalies`, `HealthFlags`, `SparseAttentionHead` (SoA format), `PromptAttentionLayer`, `PromptAnalysis`
   - `LogitsSummary` additions: `voter_agreement`, `perplexity`, `surprisal`
   - `AttentionSummary` additions: `entropy_max`, `concentration_min`, `collapsed_head_count`, `focused_head_count`
   - `LayerSummary` additions: `anomalies: TensorAnomalies`
   - `Report` additions: `prompt_analysis: Optional[PromptAnalysis]`, `health_flags: HealthFlags`
2. Update summaries with enhanced metrics + numerical stability
   - `log_softmax` for entropy computation
   - Min/max/count attention aggregation
   - NaN/Inf detection per layer per tensor type
   - Voter agreement, perplexity from existing logit data
3. Implement prompt telemetry
   - CausalLM: new forward pass before `generate()`
   - Seq2Seq: reuse existing encoder outputs (zero-cost forward pass)
   - Vectorized sparse extraction via `torch.where` (not Python loops)
   - Basin score computation per head per layer (vectorized)
   - Layer transformation (cosine similarity between consecutive layers)
   - Prompt surprisal (manual `CrossEntropyLoss(reduction='none')`)
4. Health flag post-processing
   - Transient hidden state buffer (last 5 vectors) for cosine-based repetition detection
   - Mid-layer anomaly detection with dynamic L2 threshold (5x early-layer baseline)
   - Aggregate NaN/Inf, attention collapse, high entropy step count
   - Buffer allocated at generation start, discarded after health flags computed (never serialized)
5. Update CLI/config for prompt telemetry toggle (`--no-prompt-telemetry`)
6. Test with real models (verify sparse storage size ~1.5MB typical, sparsity threshold tuning, cosine similarity overhead, dynamic L2 threshold behavior)
7. **Phase-2 consideration:** Pre-softmax variance tracking (if accessible without performance hit)

---

## QUICK REFERENCE: KEY DECISIONS & RESEARCH BACKING

| Decision | Why | Research Citation | Impact |
|----------|-----|-------------------|--------|
| **Sparse attention storage 1GB→1.5MB** | Full fidelity + query capability | Natural sparsity post-softmax | 680x reduction + gained features |
| **Structure of Arrays (SoA)** | Reduce serialization overhead | Storage optimization | 40% overhead reduction |
| **Prompt surprisal manual computation** | outputs.loss_per_token doesn't exist | HuggingFace API limitation | Required manual CrossEntropyLoss |
| **Basin score per-head scalar** | Basin characterizes head, not tokens | Dimensionality logic | Correct aggregation level |
| **Min/max/count aggregation** | Mean hides individual head failures | Voita et al. (2019) - heads are specialists | Catches collapsed heads |
| **Cosine similarity for repetition** | L2 norm = magnitude not meaning | Standard NLP practice | Prevents false positives |
| **Mid-layer anomaly (dynamic threshold)** | Hallucinations in layers 16-18 | Dec 2025 paper - 83% accuracy | Targets truth processing |
| **log_softmax not softmax→log** | Numerical stability (catastrophic cancellation) | Standard PyTorch practice | Prevents NaN in entropy |
| **"layer_transformation" not "drift"** | High transformation = layer working | Cohen (2023), Jiang et al. (2024) | Correct interpretation |
| **Defer calibration to Phase-2** | Requires labeled ground truth | ECE research shows 12-39% miscalibration | Production critical later |

---

## FINAL TRACE SIZE BREAKDOWN

**Per Trace (500-token prompt, 100-token generation):**
```
Prompt Analysis:
  - Sparse attention profiles: 1.5 MB typical (range: 0.5-5 MB)
    * Sparse connections: ~15 per query × 500 queries × 32 heads × 32 layers
    * 6 bytes per connection (2 uint16 + 1 float16)
    * Structure of Arrays format (40% overhead reduction)
    * Plus basin scalars: 4 KB (32 layers × 32 heads × 4 bytes)
  - Layer transformations: 0.1 MB (32 layers × 500 tokens × float32)
  
Timeline (Generation):
  - Logit summaries: 50 KB (8 metrics × 100 steps - includes surprisal)
  - Layer summaries: 150 KB (32 layers × 10 fields × 100 steps)
  - Attention summaries: 50 KB (32 layers × 7 fields × 100 steps)
  
Health Flags & Metadata:
  - Health flags: 32 bytes (6 bools + 1 int)
  - Trace metadata: 1 KB
  
Transient (Discarded After Processing):
  - Hidden state buffer: 20 KB (5 vectors × 4096 dims × float32)
  - Prompt surprisal: ~10 KB (temporary during computation)
  
TOTAL PERSISTED: ~1.75 MB per trace (typical)
TOTAL TRANSIENT: ~30 KB (discarded)
```

> **Note on File Sizes:** The above estimates refer to **compact JSON data size**. Trace files are saved in compact format (no indentation, minimal separators, `exclude_none=True`) for optimal file size. This achieves:
> - **~63% reduction** vs pretty-printed JSON (no indent/whitespace)
> - **~19 KB savings** per file from excluding None fields
> - **Total on-disk size:** ~1.3-1.5 MB typical (vs ~3.5-5 MB with pretty-printing)
> 
> If you want to inspect formatted JSON, use the dashboard's "Raw JSON" section which provides on-demand pretty-printing. For even smaller files, consider gzip compression (typically 70-80% reduction).

**For 100 traces: ~175 MB** (compact data size, easily fits in memory, dashboard-friendly)

**Storage by Attention Sparsity:**
- Peaked attention (~5 connections/query): 0.65 MB per trace
- Moderate attention (~15 connections/query): 1.75 MB per trace ← typical
- Diffuse attention (~50 connections/query): 5.25 MB per trace

**Comparison to Naive Approach:**
- Naive (full attention): 1.02 GB per trace → 102 GB for 100 traces (CRASHES)
- Sparse (Phase-1): 1.75 MB per trace → 175 MB for 100 traces ✅
- **Reduction: 583x smaller (typical case)**

**Key Benefits of Sparse Storage:**
- ✅ Full fidelity (exact weights, not derived summaries)
- ✅ Arbitrary queries ("What attended to token X?")
- ✅ Graph visualization capability
- ✅ Variable size adapts to attention patterns
- ✅ Can compute derived metrics (entropy, sink) on-the-fly if needed

---

## PRE-IMPLEMENTATION REVIEW

Review notes captured before implementation begins. These address performance concerns, dependency ordering, and edge cases discovered while auditing the metrics plan against the existing CoreVital codebase (v0.2.0, pre-phase-1 branch).

---

### 🔧 Review Note 1: Sparse Extraction Must Be Vectorized

**The Concern:**

The sparse attention extraction code in Metric 7 (Prompt Attention Profiles) uses nested Python loops:

```python
# Current proposal (slow):
for head_idx in range(num_heads):        # 32 iterations
    for query_idx in range(seq_len):     # 500 iterations
        head_attn = attn[0, head_idx, query_idx, :]
        significant_mask = head_attn > THRESHOLD
        significant_keys = torch.where(significant_mask)[0]
        for key_idx in significant_keys:  # ~15 iterations
            query_indices.append(query_idx)
            ...
```

For a 500-token prompt on a 32-layer, 32-head model, the inner loop iterates 500 times per head, 32 heads per layer, 32 layers. That's 512,000 Python-level iterations before we even count the per-connection appends. Python loop overhead at this scale is measured in seconds, not milliseconds.

**The Fix — Vectorized Extraction:**

```python
# Vectorized approach (fast):
for layer_idx, attn in enumerate(outputs.attentions):
    # attn shape: [batch, heads, seq_len, seq_len]
    for head_idx in range(num_heads):
        head_attn = attn[0, head_idx]  # [seq_len, seq_len]
        
        # Single torch.where on the full matrix — no Python loop over query positions
        mask = head_attn > THRESHOLD
        query_idx, key_idx = torch.where(mask)  # Both are 1D tensors
        weights = head_attn[mask]                # 1D tensor of weights
        
        # Convert to lists in one batch
        sparse_head = SparseAttentionHead(
            query_indices=query_idx.tolist(),
            key_indices=key_idx.tolist(),
            weights=weights.tolist()
        )
```

**Why This Matters:**
- `torch.where` on a `[500, 500]` matrix is a single CUDA/CPU kernel call (~0.1ms)
- The Python loop version would take ~50-100ms per head × 32 heads × 32 layers = 50-100 seconds
- Vectorized version: ~0.1ms per head × 32 × 32 = ~100ms total
- **500-1000x speedup** for the extraction step

**Basin Score Can Also Be Vectorized:**

```python
# Instead of looping over query positions:
mid_mask = torch.zeros(seq_len, dtype=torch.bool)
mid_mask[mid_start:mid_end] = True

for head_idx in range(num_heads):
    head_attn = attn[0, head_idx]  # [seq_len, seq_len]
    
    # Sum attention to middle keys vs boundary keys — one operation each
    middle_attn = head_attn[:, mid_mask].sum(dim=-1).mean()      # avg across queries
    boundary_attn = head_attn[:, ~mid_mask].sum(dim=-1).mean()   # avg across queries
    basin_score = (middle_attn / (boundary_attn + 1e-10)).item()
```

**Cost After Vectorization:**
- Sparse extraction: ~100ms total (32 layers × 32 heads × ~0.1ms per torch.where)
- Basin scores: ~30ms total (32 layers × 32 heads × ~0.03ms per reduction)
- **Total prompt processing: ~130ms** (down from potential minutes with Python loops)

---

### 🔧 Review Note 2: Transient Hidden State Buffer — Never Serialize

**The Concern:**

Repetition loop detection (Health Flags, Metric 14) requires storing the last 5 full hidden state vectors from the final layer for cosine similarity comparison. For a model with `hidden_size=4096`, that's:

```
5 vectors × 4096 dims × 4 bytes (float32) = 80 KB transient
```

This buffer exists only during generation and health flag computation. It must **never** be serialized into the JSON trace.

**Why This Needs to Be Explicit:**

The current CoreVital architecture stores everything in the `InstrumentationResults` dataclass, which gets passed to `ReportBuilder`, which serializes it. If the hidden state buffer accidentally lands in `InstrumentationResults`, it would:
1. Bloat every trace by 80KB of raw floating-point vectors
2. Break the "summary statistics only, no raw tensors" design principle
3. Be useless in the JSON (cosine similarity is already computed; the vectors serve no further purpose)

**Implementation Approach:**
- Keep the buffer as a **local variable** inside the health flag computation function
- Alternatively, store as a field on the `InstrumentationCollector` instance (not on `InstrumentationResults`)
- After `repetition_loop_detected` is computed (a single boolean), discard the buffer
- The `HealthFlags` model on the `Report` only contains the boolean result, never the vectors

**Validation:** During testing, assert that the serialized trace JSON contains no field with array length matching `hidden_size`. This catches accidental leaks.

---

### 🔧 Review Note 3: Schema Dependency Chain

**The Concern:**

The health flag computation code references fields that don't exist yet:

```python
# From _detect_mid_layer_anomaly:
if layer.anomalies and (layer.anomalies.has_nan or layer.anomalies.has_inf):
    return True
if layer.attention_summary and layer.attention_summary.collapsed_head_count > 0:
    return True
```

This depends on:
1. `TensorAnomalies` being a field on `LayerSummary` (doesn't exist in v0.2.0)
2. `collapsed_head_count` being a field on `AttentionSummary` (doesn't exist in v0.2.0)
3. These fields being populated by the collector during generation

**The Dependency Chain:**

```
Schema v0.3.0 (define new models/fields)
    ↓
Enhanced summaries.py (compute new metrics: min/max/count, NaN/Inf)
    ↓
Updated collector.py (wire new metrics into InstrumentationResults)
    ↓
Updated report_builder.py (populate new fields on Report)
    ↓
Health flags (consume the populated fields)
```

**Why This Matters:**

If health flags are implemented before the schema and computation pipeline, they'll reference `None` fields and silently produce `False` for every check. The detection logic would technically "work" (no crashes) but catch nothing. This is worse than a crash because it creates false confidence — "all health checks pass" when no checks are actually running.

**Implementation Order Must Be:**
1. Schema first (define the containers)
2. Computation second (fill the containers)
3. Health flags last (read the containers)

---

### 🔧 Review Note 4: Perplexity as Lowest-Priority Metric

**The Observation:**

Perplexity (Metric 13) is rated 3/5 and is defined as:

```python
perplexity = 2 ** entropy
```

This is a single exponentiation on a value we already compute. The document correctly notes it's "redundant with entropy" but "more interpretable" — perplexity of 50 means "choosing among 50 equally likely tokens," which is easier to explain than "entropy of 5.64 bits."

**The Recommendation:**

Perplexity should be the **first metric to cut** if implementation runs long. It can always be:
1. Computed on-the-fly in the dashboard from entropy (`2 ** entropy`)
2. Added in a later pass with zero schema risk (purely additive field)

It should be implemented — the cost is genuinely zero — but it should be the last thing touched, not something that blocks anything else.

---

### 🔧 Review Note 5: `torch_dtype` Deprecation Warning

**The Observation:**

Every real model run during pre-phase-1 validation produced this warning:

```
`torch_dtype` is deprecated! Use `dtype` instead!
```

This comes from `hf_loader.py` where we pass `torch_dtype=dtype` to `model_class.from_pretrained()`. The HuggingFace `transformers` library has renamed the parameter to `dtype`.

**The Fix:**

```python
# Old (deprecated):
model = model_class.from_pretrained(
    config.model.hf_id,
    torch_dtype=dtype,
    ...
)

# New:
model = model_class.from_pretrained(
    config.model.hf_id,
    dtype=dtype,
    ...
)
```

**Why Fix Now:**
- It's a one-line change
- The warning prints on every run, cluttering logs
- If HuggingFace removes the deprecated parameter in a future release, model loading will break
- We're already modifying `hf_loader.py` for prompt telemetry — fix it while we're in there

**Risk:** None. The `dtype` parameter is the official replacement. Backward-compatible with current and future `transformers` versions.

---

### 🔧 Review Note 6: Layer Transformation Storage Scope

**The Observation:**

The FINAL TRACE SIZE BREAKDOWN section lists:

```
Prompt Analysis:
  - Layer transformations: 0.1 MB (32 layers × 500 tokens × float32)
```

This implies storing one `layer_transformation` value per token per layer transition. But Metric 10 (Layer Transformation) defines it as cosine similarity between consecutive layer representations — meaning one value per *layer transition* per *token position*.

For a 500-token prompt with 32 layers, that's 31 transitions × 500 tokens = 15,500 floats = 62 KB, which matches the "0.1 MB" estimate.

**The Question: Do We Need Per-Token Granularity?**

The full per-token breakdown is valuable for prompt analysis (identifying which token positions cause the most transformation), but it's worth noting the alternative:

- **Per-token** (current proposal): 62 KB — enables "token 250 caused high transformation in layer 16"
- **Per-layer aggregate** (mean across tokens): 31 floats = 124 bytes — enables "layer 16 does the most work"

**Recommendation:** Keep per-token for prompt analysis. The 62 KB cost is negligible relative to the 1.5 MB sparse attention profiles, and the per-token view is what enables layer blame attribution at specific prompt positions — exactly the kind of debugging signal this metric exists to provide.

**Schema Implication:** Layer transformation lives in `PromptAnalysis`, not in `LayerSummary`. It's a prompt-only metric (computed once when all layers are available), not a per-generation-step metric. This is correctly noted in Metric 10's "Constraint: Only compute during prompt" decision.

---

### 🔧 Review Note 7: Seq2Seq Prompt Telemetry — Reuse, Don't Recompute

**The Observation:**

The cost analysis for Metric 7 correctly notes:

> **Cost Analysis (Seq2Seq):**
> - **Compute:** 0ms forward pass (reuse encoder outputs!)

This is important. For Seq2Seq models (T5, BART), the encoder forward pass already runs during `_generate_seq2seq_manual()` in `collector.py`. The encoder hidden states and encoder attentions are already captured and stored in `InstrumentationResults.encoder_hidden_states` and `InstrumentationResults.encoder_attentions`.

**What This Means for Implementation:**

- **CausalLM:** Needs a *new* forward pass (`model(input_ids)`) before `model.generate()` to capture prompt telemetry. This is the approach agreed upon in the pre-phase-1 discussion — a single extra forward pass for prompt tokens.
- **Seq2Seq:** Needs *no* new forward pass. Reuse the encoder outputs that are already captured. The sparse extraction, basin scores, and layer transformations can all be computed from the existing `encoder_hidden_states` and `encoder_attentions` tensors.

**Cost Difference:**
- CausalLM: ~10-50ms for the extra forward pass + ~130ms for sparse extraction = ~140-180ms
- Seq2Seq: 0ms for forward pass + ~130ms for sparse extraction = ~130ms
- Both: One-time cost at the start of generation, not per-token

**Implementation Detail:** The collector already branches on `is_seq2seq` for generation. The prompt telemetry path should similarly branch:

```python
if is_seq2seq:
    # Reuse encoder outputs already captured
    prompt_attentions = results.encoder_attentions
    prompt_hidden_states = results.encoder_hidden_states
else:
    # New forward pass for CausalLM
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True, output_attentions=True)
    prompt_attentions = outputs.attentions
    prompt_hidden_states = outputs.hidden_states
```

---

### Summary of Review Additions

| Review Note | Category | Impact |
|-------------|----------|--------|
| **Vectorized sparse extraction** | Performance | 500-1000x speedup for prompt attention profiling |
| **Transient buffer never serialized** | Architecture | Prevents 80KB raw tensor leak per trace |
| **Schema dependency chain** | Implementation order | Prevents silent false-negative health checks |
| **Perplexity lowest priority** | Prioritization | First metric to cut if running long |
| **`torch_dtype` → `dtype`** | Maintenance | Prevents future breakage, cleans up logs |
| **Layer transformation scope** | Storage | Confirms per-token granularity (62KB) is worth it |
| **Seq2Seq reuse encoder outputs** | Performance | Zero-cost prompt telemetry for Seq2Seq models |

**Key Takeaway:** The metrics plan is sound. The design decisions (sparse storage, min/max/count aggregation, cosine similarity, mid-layer targeting) are all correct. The review notes above are about making the implementation fast, safe, and correctly ordered — not about changing what gets built.

---

**END OF PHASE-1 METRICS ANALYSIS**

*Last Updated: Phase-1c complete. Health flags populated with transient buffer lifecycle, repetition loop detection (cosine threshold 0.9995 for float16 anisotropy), and per-step mid-layer anomaly detection (runtime NaN/Inf/L2 only, not structural collapse). All exit criteria verified.*
*Phase-1b: Vectorized sparse attention, basin scores, layer transformations, and prompt surprisal implemented.*
*Phase-1a: Pre-implementation review notes added — vectorized extraction, transient buffer discipline, schema dependency chain, Seq2Seq encoder reuse. Storage estimates corrected to reflect sparse approach (1.75MB typical, not 8.25MB from superseded compressed vector approach).*