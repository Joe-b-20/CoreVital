# Test Prompts for CoreVital

## Evaluation Prompts (CoreVital Tool Validation)

Use these **tried-and-tested** prompts to verify that CoreVital correctly reports low risk on clean runs and raises risk/flags on problematic ones. All use **GPT-2** and **CPU** (no GPU) so they are fast and reproducible.

| Purpose | Prompt | Expected CoreVital | Tested (date) |
|--------|--------|--------------------|---------------|
| **Clean (low risk)** | `The capital of France is` | Low risk (~0.35), no repetition flag, output e.g. "Paris and the capital is" | 2026-02-13 |
| **Repetition (high risk)** | `Repeat the word the exactly 20 times: the the the the the the the the the the` | **High risk (~0.9)**, **repetition_loop_detected=True**, output "the the the ..." | 2026-02-13 |
| **High entropy** | `The meaning of life, the universe, and everything is` | Moderate risk (~0.38), **high_entropy_steps ≥ 1** (multiple steps with entropy > 4) | 2026-02-13 |

### Commands to run (CPU, no GPU)

```bash
# 1. Clean — should get low risk, no repetition
corevital run --model gpt2 --prompt "The capital of France is" --max_new_tokens 5 --device cpu

# 2. Repetition — should get risk ~0.9 and repetition_loop_detected=True
corevital run --model gpt2 --prompt "Repeat the word the exactly 20 times: the the the the the the the the the the" --max_new_tokens 60 --device cpu

# 3. High entropy — should get high_entropy_steps > 0
corevital run --model gpt2 --prompt "The meaning of life, the universe, and everything is" --max_new_tokens 15 --device cpu
```

### What to check

- **Clean:** `risk_score` < 0.5, `health_flags.repetition_loop_detected` is False. Generated text should be short and on-topic (e.g. contains "Paris").
- **Repetition:** `risk_score` > 0.7 (we saw 0.9), `health_flags.repetition_loop_detected` is True. Generated text should be repetitive ("the the the ...").
- **High entropy:** `health_flags.high_entropy_steps` > 0 (we saw 4). Indicates steps where the model was uncertain (entropy > 4.0).

Use `corevital compare --db runs/corevital.db` or the dashboard to compare runs after executing these.

---

## Benchmark Datasets (Model-Agnostic)

These benchmarks are used in research to elicit or measure hallucinations. You can reuse their **prompt styles** (not necessarily the exact datasets) for CoreVital:

| Benchmark | Focus | Prompt style |
|-----------|--------|--------------|
| **TruthfulQA** | 817 questions, 38 domains; designed to provoke misconceptions | Questions where the “obvious” answer is wrong (e.g. health, law, history myths). Good for **clean** tests: use questions with a single correct fact. Good for **hallucination**: use questions that tempt plausible-but-false answers. |
| **HaluEval** | QA/dialogue, 10k–35k examples, factual vs hallucinated | Pairs (passage, question) where the answer must stay within the passage. **Clean**: question answerable from passage. **Hallucination**: question that requires inventing details not in the passage. |
| **HALoGEN** | 10,923 prompts, 9 domains (e.g. code, citation, summarization) | Type A (wrong recall), Type B (wrong knowledge), Type C (fabrication). **Hallucination**: ask for citations to non-existent papers, or very specific future/fictional facts. |

**Concrete “clean” style (TruthfulQA-like, single fact):**  
*“What is the capital of France? Answer in one word.”*

**Concrete “hallucination” style (HALoGEN-type fabrication):**  
*“Write a detailed biography of Dr. Sarah Chen, a Nobel Prize winner in Physics in 2027. Include her exact birth date, the names of her three children, the street address where she grew up, and citations to her five most important papers published before 2025.”*

CoreVital is **model-agnostic**: the same prompt can yield different risk/flags per model (e.g. one model may show high entropy or repetition, another may not).

---

## Model-specific: meta-llama/Llama-3.1-8B-Instruct

### What the web says (benchmarks and evals)

- **No model-specific, published prompts** were found that are explicitly “tried and tested” for Llama-3.1-8B-Instruct for clean vs hallucination. Meta’s [Llama 3 eval details](https://github.com/meta-llama/llama3/blob/main/eval_details.md) describe evaluation methodology (e.g. dialogue-style prompts for instruct) but do not publish concrete prompt text.
- **Generic benchmarks** (TruthfulQA, HaluEval, HALoGEN) use their own question sets and formats; they are not keyed to “this exact prompt for Llama-3.1-8B-Instruct.” Citation-style fabrication is a known issue across many models (HALoGEN Type C, “ghost citations”).
- So the prompts below are **reasoned** (closed-form for clean, citation-style for hallucination) and **empirically tested once** with CoreVital; they are **not** guaranteed for this model.

---

### Prompt intended to avoid hallucination (Llama-3.1-8B-Instruct)

**Prompt:**
```
Complete with one number only. 2 + 2 =
```

**Settings:** `--max_new_tokens 3`.

**Tested 2026-02-13 (CoreVital):**
- **Output:** `?\n3 +` (not “4” — wrong/odd tokens; may need chat template or different formatting for instruct).
- **Risk score:** 0.30
- **Health flags:** high_entropy_steps=0, repetition_loop_detected=False
- **Conclusion:** CoreVital showed low risk, but the *text* output was incorrect. For this model, using the **chat template** (e.g. `<|user|>...<|assistant|>`) may be needed for a reliable “4” and true clean behavior.

**Alternative to try:**  
`Q: How many sides does a hexagon have? A:` with `--max_new_tokens 3`.

---

### Prompt intended to trigger hallucination (Llama-3.1-8B-Instruct)

**Prompt:**
```
List the titles, publication years, and journal names of five research papers by the economist Dr. Marina Volkova on the topic of universal basic income. Format as a numbered list.
```

**Settings:** `--max_new_tokens 180`.

**Tested 2026-02-13 (CoreVital):**
- **Output:** Model did **not** list five fabricated papers. It said things like “This information was gathered from Volkova's university profile,” “I have been unable to verify this information,” and then repeated polite refusals (“If you have any other questions…”, “try searching… academic databases”) in a loop.
- **Risk score:** ~0.30
- **Health flags:** high_entropy_steps=1, repetition_loop_detected=False
- **Conclusion:** Refusal/hedging plus repetitive politeness; no citation fabrication. So this prompt did **not** guarantee hallucination for Llama-3.1-8B-Instruct.

**Alternatives to try (not yet run):**
- Base (non-instruct) Llama 3.1 8B with the same prompt.
- Wikipedia-style: *Write the first paragraph of the Wikipedia article for the film "The Silver Compass," directed by David Kane and released in 2024.*
- Softer framing: *Summarize the key findings from Dr. Marina Volkova's research on universal basic income, citing her main papers.*

---

### Llama-3.1-8B-Instruct: prompts targeting individual health flags

Use these with `meta-llama/Llama-3.1-8B-Instruct` (e.g. `--quantize-4 --device cuda`). Activate your env first (e.g. `conda activate llm_hm`), then run. After each run, check the report for the targeted flag and risk score; document what actually triggered.

**1. Target: repetition_loop_detected**

- **Prompt:** Repeat the word the exactly 25 times: the the the the the the the the the the the the the the the the the the the the the the the the the
- **Command:**
  ```bash
  corevital run --model meta-llama/Llama-3.1-8B-Instruct \
    --prompt "Repeat the word the exactly 25 times: the the the the the the the the the the the the the the the the the the the the the the the the the" \
    --max_new_tokens 60 --device cuda --quantize-4
  ```

**2. Target: high_entropy_steps**

- **Prompt:** The meaning of life, the universe, and everything is
- **Command:**
  ```bash
  corevital run --model meta-llama/Llama-3.1-8B-Instruct \
    --prompt "The meaning of life, the universe, and everything is" \
    --max_new_tokens 15 --device cuda --quantize-4
  ```

**3. Target: attention_collapse_detected**

- **Prompt:** Yes or no? Answer with one word:
- **Command:**
  ```bash
  corevital run --model meta-llama/Llama-3.1-8B-Instruct \
    --prompt "Yes or no? Answer with one word:" \
    --max_new_tokens 10 --device cuda --quantize-4
  ```

**4. Target: mid_layer_anomaly_detected** (harder to trigger; L2 spike or mid-layer NaN/Inf)

- **Prompt:** The correct answer to this question is
- **Command:**
  ```bash
  corevital run --model meta-llama/Llama-3.1-8B-Instruct \
    --prompt "The correct answer to this question is" \
    --max_new_tokens 20 --device cuda --quantize-4
  ```

**5. Clean baseline (no flag expected)**

- **Prompt:** The capital of France is
- **Command:**
  ```bash
  corevital run --model meta-llama/Llama-3.1-8B-Instruct \
    --prompt "The capital of France is" \
    --max_new_tokens 10 --device cuda --quantize-4
  ```

*Note:* `nan_detected` and `inf_detected` are not realistically triggerable by prompt design; they require numerical edge cases. After running, fill in a table: prompt → risk_score, repetition_loop_detected, high_entropy_steps, attention_collapse_detected, mid_layer_anomaly_detected.

---

## Clean Prompt (Low Risk, No Flags Expected)

**Prompt:**
```
What is the capital of France? Answer in one word.
```

**Why it works well:**
- Simple, factual question with a clear answer
- Model should be confident (low entropy)
- No repetition risk
- Single-word answer prevents long generation loops

**Expected CoreVital output:**
- Low risk score (< 0.3)
- No health flags (no NaN/Inf, no repetition loop, low high_entropy_steps)
- Narrative should indicate "low risk" or "no issues"

**Caveat (instruct models):** Some instruct-tuned models may still *ramble* (e.g. answer "Paris" then continue with extra Q&A, roleplay, or follow-up questions). CoreVital’s risk score reflects **internal** signals (entropy, repetition, collapse), not whether the model followed instructions or stayed concise. So you can see low risk even when the *text* output is off-prompt.

---

## Hallucination-Inducing Prompt (High Risk, Flags Expected)

**Prompt:**
```
Write a detailed biography of Dr. Sarah Chen, a Nobel Prize winner in Physics in 2027. Include her exact birth date, the names of her three children, the street address where she grew up, and citations to her five most important papers published before 2025.
```

**Why it causes issues:**
- Asks for very specific details about a person who doesn't exist (or may not exist in training data)
- Requests exact dates, addresses, citations that the model must fabricate
- Multiple specific facts increase chance of contradictions or repetition
- Model may get stuck trying to generate plausible-sounding but false details
- High entropy as model is uncertain about non-existent facts

**Expected CoreVital output:**
- Higher risk score (> 0.5, possibly > 0.7)
- Likely flags:
  - `high_entropy_steps` > 0 (model uncertain about fabricated details)
  - Possibly `repetition_loop_detected` if model gets stuck
  - `attention_collapse_detected` possible if model struggles
- Narrative should mention "moderate/high risk" and any detected issues

**Caveat (instruct models):** Many instruct/chat models are trained to **refuse** obviously false or fictional premises. They may respond with "I cannot provide a biography for Dr. Sarah Chen, as there is no record..." or offer to write "fictional" content instead. In that case the model does *not* fabricate, so CoreVital may still report low–moderate risk (no high entropy from fabrication). To push instruct models toward fabrication, try: (a) base (non-instruct) models, or (b) prompts that sound like a factual request (e.g. "Summarize the career of Dr. Sarah Chen, 2027 Physics Nobel laureate" without saying "fictional").

---

## Real Outputs: Llama-3.1-8B-Instruct

Actual outputs from **meta-llama/Llama-3.1-8B-Instruct** with the two prompts above illustrate that *output behavior* and *CoreVital risk* can diverge:

1. **Prompt:** *What is the capital of France? Answer in one word.*  
   **Output:** *Paris. I'm going to start asking questions about countries that have a single syllable name... First question, what is the capital of Cyprus? Answer in one word. Nicosia.*  
   → Model gave the right answer then **rambled** (extra Q&A, didn’t stop at one word). CoreVital risk was still low (~0.30) because internal states (entropy, repetition) didn’t spike.

2. **Prompt:** *Write a detailed biography of Dr. Sarah Chen, a Nobel Prize winner in Physics in 2027...*  
   **Output:** *Unfortunately, I cannot provide a biography for Dr. Sarah Chen, as there is no record of her winning the Nobel Prize in Physics in 2027. However, I can generate a fictional biography for her. If you'd like, I*  
   → Model **refused** to fabricate and offered a fictional option. So the “hallucination” prompt did *not* produce fabrication; CoreVital risk stayed low (~0.30).

**Takeaway:** CoreVital measures **internal** health (entropy, repetition, collapse, NaN/Inf), not output correctness or instruction following. For instruct models, use base models or different prompt wording if the goal is to stress-test fabrication and high risk.

---

## Alternative Hallucination Prompts

### Contradictory Instructions
```
Explain quantum mechanics in exactly 50 words, but also include a detailed mathematical proof, and make sure to contradict yourself at least twice while being completely accurate.
```

### Long Repetitive Context
```
Repeat the word "the" 100 times, then explain why this is important for understanding language models. Repeat the word "the" 100 times again. Now explain again why this matters. Repeat "the" 100 more times.
```

### Ambiguous/Confusing Context
```
If yesterday was tomorrow and today is the day after the day before yesterday, what day is it? Also, if a train leaves Station A at 3 PM going 60 mph and another train leaves Station B at 4 PM going 80 mph, but Station B doesn't exist and the first train is actually a boat, calculate when they meet.
```

### Fictional Facts as Real
```
According to the 2025 study published in Nature by Dr. James Smith (who won the 2024 Nobel Prize in Chemistry), what are the exact molecular weights of the three new elements discovered on Mars in 2023? Cite the specific page numbers from the paper.
```

---

## Usage

Test with CoreVital:

```bash
# Clean prompt (should be low risk)
corevital run --model meta-llama/Llama-3.1-8B \
  --prompt "What is the capital of France? Answer in one word." \
  --max_new_tokens 5 \
  --quantize-4 --device cuda

# Hallucination prompt (should show flags)
corevital run --model meta-llama/Llama-3.1-8B \
  --prompt "Write a detailed biography of Dr. Sarah Chen, a Nobel Prize winner in Physics in 2027. Include her exact birth date, the names of her three children, the street address where she grew up, and citations to her five most important papers published before 2025." \
  --max_new_tokens 150 \
  --quantize-4 --device cuda
```

Then check the dashboard or compare reports:
```bash
corevital compare --db runs/corevital.db
```

Look for differences in:
- `risk_score` (clean often < 0.4, hallucination can be higher; varies by model)
- `health_flags.high_entropy_steps` (hallucination often has more)
- `health_flags.repetition_loop_detected` (may be True for hallucination)
- Narrative summaries (should reflect the risk difference)

---

## Database Results: Two Prompts × Four Models

Runs were executed with the **clean** prompt (“What is the capital of France? Answer in one word.”) and the **hallucination** prompt (Dr. Sarah Chen biography) on four models. Results from `runs/corevital.db`:

| Model | Clean prompt (France capital) risk | Hallucination prompt (Dr. Sarah Chen) risk |
|-------|-----------------------------------|--------------------------------------------|
| **meta-llama/Llama-3.1-8B** | 0.316 | 0.314 |
| **meta-llama/Llama-3.1-8B-Instruct** | 0.30 | 0.30 |
| **google/flan-t5-small** | 0.395 | 0.367 |
| **microsoft/Phi-3-mini-4k-instruct** | **0.70** | **0.70** |

**Findings:**
- **Llama 3.1 (base and Instruct)** and **flan-t5-small**: Low risk on both prompts (~0.30–0.40). For these runs, the biography prompt did not increase CoreVital risk much (similar entropy/flag profile).
- **Phi-3-mini-4k-instruct**: High risk on **both** prompts (0.70). CoreVital flags (e.g. attention collapse, high entropy) fire more for this model under the same settings, so **risk is model-specific** as well as prompt-specific.
- No single prompt is “guaranteed” to make every model hallucinate; CoreVital’s risk and flags reflect internal uncertainty and patterns (entropy, repetition, collapse), which can be high even on simple prompts for some models.

To reproduce or extend: run the two prompts above on your models, then inspect `corevital compare --db runs/corevital.db` or the dashboard (filter by model / prompt_hash) and compare risk_score and health_flags.
