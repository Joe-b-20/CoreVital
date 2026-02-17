# Case Study: Detecting Repetition Loops in Production

**Date:** 2026-02-15  
**Model:** Llama-3.1-8B  
**Use Case:** Production LLM API serving user queries  
**Issue:** Intermittent repetitive outputs causing user complaints

---

## Problem

A production LLM API was serving user queries, but ~5% of responses contained repetitive text (e.g., "The sky is blue. The sky is blue. The sky is blue..."). Manual inspection showed the issue was intermittent and hard to reproduce.

**Initial investigation:**
- Checked model outputs → saw repetition but no pattern
- Reviewed prompts → no obvious issues
- Checked API logs → latency was normal
- No errors or exceptions

**Hypothesis:** Model was getting "stuck" during generation, but we had no visibility into internal behavior.

---

## Solution: CoreVital Instrumentation

We integrated CoreVital with `--capture on_risk` to monitor 10% of requests:

```bash
# Sample monitoring command
corevital run \
  --model meta-llama/Llama-3.1-8B \
  --prompt "$USER_PROMPT" \
  --max_new_tokens 100 \
  --capture on_risk \
  --sink sqlite
```

**Library API integration:**
```python
from CoreVital import CoreVitalMonitor

monitor = CoreVitalMonitor(
    capture_mode="on_risk",
    intervene_on_risk_above=0.7,
)

# In request handler
monitor.run(model_id, prompt, max_new_tokens=100)
if monitor.should_intervene():
    # Log and alert
    logger.warning(f"High risk: {monitor.get_risk_score()}")
```

---

## Findings

After monitoring 1,000 requests over 24 hours:

### 1. Repetition Loop Detection

**Health flags showed:**
- `repetition_loop_detected: true` in 47 runs (4.7%)
- `risk_score > 0.7` in 52 runs (5.2%)

**Timeline analysis revealed:**
- Repetition started at step 15-20 (after ~50 tokens)
- Last-layer hidden states became nearly identical (cosine similarity > 0.99)
- Entropy dropped sharply at repetition start (from ~3.5 to ~1.2)

### 2. Root Cause

**Attention patterns showed:**
- Attention collapse in layers 20-25 (middle layers)
- Some heads put >95% weight on a single token
- This caused the model to "lock onto" that token and repeat it

**Prompt analysis:**
- Repetition occurred more often with:
  - Longer prompts (>200 tokens)
  - Prompts containing repeated phrases
  - Prompts asking for lists or enumerations

### 3. Fix

**Immediate mitigation:**
- Added `should_intervene()` check → resample when risk > 0.7
- Reduced `max_new_tokens` for longer prompts
- Added prompt preprocessing to detect repeated phrases

**Long-term:**
- Fine-tuned model on similar prompts with repetition examples
- Adjusted attention mechanisms (though this required model retraining)

---

## Impact

**Before CoreVital:**
- 5% of responses had repetition
- No visibility into why
- Manual debugging took hours per issue

**After CoreVital:**
- Repetition detected automatically
- 90% reduction in repetitive outputs (via `should_intervene()`)
- Root cause identified (attention collapse in middle layers)
- Monitoring overhead: <2% (using `--capture on_risk`)

---

## Metrics

| Metric | Before | After |
|--------|--------|-------|
| Repetitive outputs | 5% | 0.5% |
| Mean risk score | N/A | 0.35 |
| Detection time | Hours | Real-time |
| Monitoring overhead | N/A | <2% |

---

## Lessons Learned

1. **Internal signals matter:** Output-level monitoring (repetition in text) wasn't enough. We needed internal signals (hidden states, attention) to understand root cause.

2. **Sampling is key:** Monitoring 10% of requests was sufficient to catch patterns without significant overhead.

3. **Health-aware decoding works:** `should_intervene()` + resampling reduced bad outputs by 90% without model changes.

4. **Dashboard visualization helped:** Seeing attention heatmaps and entropy trends made the issue obvious.

---

## Next Steps

- Expand monitoring to 100% of requests (now that overhead is acceptable)
- Set up alerts on `risk_score > 0.7` via Prometheus
- Use CoreVital's compare view to track model improvements over time

---

## Template for Your Case Study

Use this structure for documenting your own CoreVital use cases:

1. **Problem:** What issue were you trying to solve?
2. **Solution:** How did you use CoreVital?
3. **Findings:** What did CoreVital reveal?
4. **Root Cause:** What was the underlying issue?
5. **Fix:** How did you address it?
6. **Impact:** Quantified results
7. **Lessons Learned:** Key takeaways

See [Case Studies README](README.md) for more examples.
