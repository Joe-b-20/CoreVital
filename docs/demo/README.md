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

`corevital_demo.db` contains 5 curated traces (1 GPT-2, 4 Llama-3.1-8B-Instruct) at varying risk levels (0.30 to 0.48). This is the database used by the hosted Streamlit dashboard so visitors can explore the Compare view and filter by model without running any inference locally.
