# Visual Examples: Good vs Bad Runs

This guide shows what healthy and unhealthy model runs look like in the CoreVital dashboard.

## Healthy Run Indicators

### Entropy Profile
**Good:**
- Entropy stays between 2-4 for most steps
- Gradual, smooth changes (no sudden spikes)
- No sustained periods above 4.0

**Bad:**
- Sudden spikes above 4.0 (model got confused)
- Sustained high entropy (>4) for many steps
- Very low entropy (<1) for extended periods (model may be stuck)

### Attention Patterns
**Good:**
- Attention entropy mean: 1.5-3.0 across layers
- Collapsed head count: 0-2 per layer (normal for smaller models)
- Concentration max: 0.3-0.7 (attention spread across multiple tokens)

**Bad:**
- Many collapsed heads (>5 per layer) -- attention collapse detected
- Concentration max near 1.0 -- heads focusing on single tokens
- Very low attention entropy (<0.5) -- attention collapse

### Health Flags
**Good:**
- All flags `false` or counts at 0
- Risk score: <0.3
- No NaN/Inf detected

**Bad:**
- `nan_detected: true` -- **Critical:** Stop and debug
- `repetition_loop_detected: true` -- Model is stuck repeating
- `attention_collapse_detected: true` -- Many heads collapsed
- `high_entropy_steps > 5` -- Model confused for many steps
- Risk score: >0.7 -- High risk of poor output

## Example Scenarios

### Scenario 1: Repetition Loop

**Symptoms:**
- Entropy drops sharply (from ~3 to ~1)
- Last-layer hidden states show high cosine similarity (>0.99) across steps
- `repetition_loop_detected: true`
- Risk score: 0.8+

**What to do:**
- Check prompt (may contain repeated phrases)
- Reduce `max_new_tokens`
- Use `should_intervene()` to resample

### Scenario 2: Attention Collapse

**Symptoms:**
- Many collapsed heads (>5 per layer)
- Concentration max near 1.0
- `attention_collapse_detected: true`
- Risk score: 0.6-0.8

**What to do:**
- Check if model is trained properly
- May indicate a training issue (heads not learning)
- Consider fine-tuning or using different model

### Scenario 3: High Entropy (Confusion)

**Symptoms:**
- Entropy >4 for many steps
- `high_entropy_steps > 10`
- Risk score: 0.5-0.7

**What to do:**
- Check prompt clarity
- Model may be out of distribution
- Consider prompt engineering

### Scenario 4: NaN/Inf Detected

**Symptoms:**
- `nan_detected: true` or `inf_detected: true`
- Risk score: 1.0 (critical)
- Hidden state norms show NaN/Inf

**What to do:**
- **Stop immediately** — numerical instability
- Check inputs (may contain extreme values)
- Check model weights (may be corrupted)
- Verify quantization settings

## Using the Dashboard

1. **Load a report** (Demo sample, Local file, Database, or Upload)
2. **Check Health Flags** at the top — red badges indicate issues
3. **View Entropy Chart** — look for spikes or sustained high values
4. **Check Attention Heatmaps** — look for collapsed heads (dark red)
5. **Review Risk Score** — >0.7 = high risk

## Comparing Runs

Use the **Compare** view to see differences:
- Load multiple runs from the database
- Select 2+ runs to compare side-by-side
- Differences are highlighted automatically
- Useful for:
  - Comparing models
  - Tracking improvements over time
  - Debugging regressions

## See Also

- [Metrics Guide](Phase1%20metrics%20analysis.md) -- Detailed metric explanations
- [Dashboard](../dashboard.py) — Interactive visualization
- [Case Studies](case-studies/) — Real-world examples
