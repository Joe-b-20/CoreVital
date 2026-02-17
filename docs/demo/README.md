# CoreVital demo

## Try CoreVital in 5 minutes

1. **Install**
   ```bash
   pip install -e .
   ```

2. **Run one monitored inference** (CPU, no GPU required)
   ```bash
   corevital run --model gpt2 --prompt "Explain why the sky is blue in one sentence." --max_new_tokens 20 --device auto
   ```
   Output is written to `runs/` (SQLite by default: `runs/corevital.db`, or JSON if you use `--sink local`).

3. **View the report in the dashboard**
   ```bash
   pip install -e ".[dashboard]"
   streamlit run dashboard.py
   ```
   Open the app, choose **Database** as source (if you used the default SQLite sink) or **Local file** and pick a trace from `runs/`.

4. **Optional: try without running a model**
   This folder contains a minimal pre-generated report so you can open the dashboard without running a model:
   - **File:** [sample_report.json](sample_report.json)  
   In the dashboard, select **Demo sample** in the sidebar (or use **Upload** and choose `docs/demo/sample_report.json`).

## Regenerating the sample report

From the repo root (with the project installed, e.g. `conda activate llm_hm`):

```bash
python scripts/gen_demo_report.py
```

This overwrites `docs/demo/sample_report.json` with a minimal schema-valid report including health flags and risk.
