# CoreVital Demo

## Try CoreVital in 5 minutes

1. **Install**
   ```bash
   pip install -e .
   ```

2. **Run one monitored inference** (CPU, no GPU required)
   ```bash
   corevital run --model gpt2 --prompt "Explain why the sky is blue in one sentence." --max_new_tokens 20
   ```
   Output is written to `runs/corevital.db` (SQLite, default) or JSON if you use `--sink local`.

3. **View the report in the dashboard**
   ```bash
   pip install -e ".[dashboard]"
   streamlit run dashboard.py
   ```
   Open the app, choose **Database** as source (if you used the default SQLite sink) or **Local file** and pick a trace from `runs/`.

4. **Try without running a model**
   This folder contains a real Llama-3.1-8B-Instruct report so you can explore the dashboard immediately:
   - **File:** [sample_report.json](sample_report.json)
   - **Model:** meta-llama/Llama-3.1-8B-Instruct (32 layers, 32 attention heads)
   - **Prompt:** "The capital of France is"
   - **Risk score:** 0.39 (attention collapse detected, 3 high-entropy steps)

   In the dashboard, select **Demo sample** in the sidebar.

## About the sample report

The bundled `sample_report.json` is a real CoreVital report from a Llama-3.1-8B-Instruct run (not synthetic). It includes full per-layer summaries for all 32 layers across 10 generation steps, prompt analysis, health flags, risk scoring, fingerprinting, narrative, and performance data.

## Demo database

`corevital_demo.db` contains 4 curated traces from different models, all run with **CUDA**, **4-bit** quantization, **full** capture, and **strict** performance:

| # | Model | Prompt theme |
|---|--------|---------------|
| 1 | **Llama 3.1 Instruct** (meta-llama/Llama-3.1-8B-Instruct) | Historian: compare French vs American Revolution causes |
| 2 | **Mistral Instruct** (mistralai/Mistral-7B-Instruct-v0.2) | Senior engineer: debug race conditions in distributed systems |
| 3 | **Llama 3.2 Instruct** (meta-llama/Llama-3.2-3B-Instruct) | Speculative advances in quantum error correction by 2030 |
| 4 | **FLAN-T5** (google/flan-t5-large) | Summarize Mediterranean diet / causal claims paragraph (seq2seq) |

This is the database used by the hosted Streamlit dashboard so visitors can explore the Compare view and filter by model without running any inference locally.

To regenerate the demo DB (requires GPU, conda env `llm_hm`, and HuggingFace model access):

```bash
conda activate llm_hm
python scripts/gen_demo_db.py
```

The script uses the conda environment `llm_hm` by default (via `conda run -n llm_hm`). Use `--conda-env ""` to use your current Python instead.

Output: `docs/demo/corevital_demo.db` (overwrites existing; previous copy saved as `corevital_demo.db.bak`).
