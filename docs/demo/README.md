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

3. **View the report in the web dashboard**
   - Open the hosted dashboard: [https://main.d2maxwaq575qed.amplifyapp.com](https://main.d2maxwaq575qed.amplifyapp.com) (source: [corevital-dashboard](https://github.com/Joe-b-20/corevital-dashboard)).
   - In your terminal, run: `pip install "CoreVital[serve]"` then `corevital serve`.
   - In the dashboard, click **Connect** to attach to your local API. Your data stays on your machine; the site talks only to `http://127.0.0.1:8000`.

4. **Try without running a model**
   The hosted dashboard opens in **Demo mode** by default, using sample trace JSON from the [dashboard repo](https://github.com/Joe-b-20/corevital-dashboard) (`public/demo/`). No backend or CoreVital install needed. Alternatively, you can run `corevital serve --db docs/demo/corevital_demo.db` and use the dashboard in **Database** mode to browse the optional demo DB in this repo (see below).

## Demo database (optional)

The file `corevital_demo.db` in this directory is **optional**. It contains 4 curated traces from different models, all run with **CUDA**, **4-bit** quantization, **full** capture, and **strict** performance:

| # | Model | Prompt theme |
|---|--------|---------------|
| 1 | **Llama 3.1 Instruct** (meta-llama/Llama-3.1-8B-Instruct) | Historian: compare French vs American Revolution causes |
| 2 | **Mistral Instruct** (mistralai/Mistral-7B-Instruct-v0.2) | Senior engineer: debug race conditions in distributed systems |
| 3 | **Llama 3.2 Instruct** (meta-llama/Llama-3.2-3B-Instruct) | Speculative advances in quantum error correction by 2030 |
| 4 | **FLAN-T5** (google/flan-t5-large) | Summarize Mediterranean diet / causal claims paragraph (seq2seq) |

Use it with the hosted dashboard in **Database** mode: run `corevital serve --db docs/demo/corevital_demo.db`, then in the dashboard select Database, set the API base URL and DB path, and connect to explore the Compare view without running inference locally.

To regenerate the demo DB (requires GPU and HuggingFace model access):

```bash
python scripts/gen_demo_db.py
```

The script uses the conda environment `llm_hm` by default (via `conda run -n llm_hm`). Use `--conda-env ""` to use your current Python instead.

Output: `docs/demo/corevital_demo.db` (overwrites existing; previous copy saved as `corevital_demo.db.bak`).
