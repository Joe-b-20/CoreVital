# Case study: [Title]

**Author:** [Name or team]  
**Date:** [YYYY-MM]  
**Context:** [e.g. "Production RAG pipeline", "Debugging repetition in summarization", "A/B test across two models"]

---

## Problem

What went wrong or what you wanted to understand? (e.g. unexplained failures, hard-to-debug behavior, need to compare models.)

## Setup

- **Model(s):** [e.g. meta-llama/Llama-3.2-1B, gpt2]
- **Use case:** [e.g. question answering, code generation]
- **CoreVital usage:** CLI vs Library API; capture mode (summary / full / on_risk); sink (SQLite, local JSON, Prometheus, etc.)

## What you looked at

Which CoreVital outputs did you use?

- Risk score and risk factors
- Health flags (e.g. repetition loop, attention collapse, high entropy steps)
- Early-warning signals
- Narrative summary
- Dashboard (Compare runs, filters, exports)
- Other (fingerprints, RAG context, etc.)

## Results

What did you find? (e.g. "Risk > 0.7 correlated with user reports"; "Repetition loop flag matched manual inspection"; "Model A had lower risk than Model B on the same prompts.")

## Lessons / recommendations

- What would you do again or differently?
- Tips for others in a similar setup?

---

*Optional: link to a redacted report snippet or dashboard screenshot.*
