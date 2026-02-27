# CoreVital Datasette dashboards

Out-of-the-box dashboards for CoreVital runs: **Runs list** and **Run detail** (summary, health, entropy/perplexity by step, attention heatmap, prompt/output). One link to share after deploy.

## Quick start (local)

### 1. Install Datasette and the dashboards plugin

```bash
pip install datasette datasette-dashboards
```

### 2. (Optional) Inflate report JSON for run-detail

If your database stores reports only as gzip blob (`report_blob`), run this once so run-detail panels (entropy, heatmap, prompt) work:

```bash
python scripts/inflate_report_json_from_blob.py docs/demo/corevital_demo.db
```

If you use `compress=False` with SQLiteSink, `report_json` is already populated and you can skip this.

### 3. Run Datasette

From the **repo root**:

```bash
datasette docs/demo/corevital_demo.db --metadata docs/datasette/metadata.yml --port 8001
```

### 4. Where the graphs and dashboards are

- **Dashboard list (start here):**  
  Open **http://localhost:8001/-/dashboards** in your browser.  
  If you only see the database and a “reports” table, you’re on the main Datasette page (`/`). Use the **“Dashboards”** link in the top navigation, or go directly to:
  - **http://localhost:8001/-/dashboards**

- On that page you’ll see two dashboards:
  - **CoreVital Runs** — click it to see the table of recent runs.
  - **CoreVital Run Detail** — click it to see the dashboard with **graphs** (entropy line, perplexity line, attention heatmap) and summary/health/prompt panels.

- **To see the graphs:**  
  Click **“CoreVital Run Detail”**, then in the **Trace** dropdown at the top choose a trace (e.g. `f28f1386-...`). The page will load:
  - Summary and health tables
  - **Entropy by step** (line chart)
  - **Perplexity by step** (line chart)
  - **Attention entropy (layer × step)** (heatmap)
  - Prompt and generated text (table)

So: **graphs and dashboard = http://localhost:8001/-/dashboards → CoreVital Run Detail → pick a trace.**

### Using a different database

- **File path:** Pass your DB path instead of `docs/demo/corevital_demo.db`.
- **Database name:** Datasette uses the **filename without extension** as the database name (e.g. `corevital_demo.db` → `corevital_demo`). The metadata is set up for `corevital_demo`. If your file is e.g. `runs/corevital.db`, the DB name is `corevital`. Either:
  - Override the name so it matches metadata:  
    `datasette runs/corevital.db -n corevital_demo --metadata docs/datasette/metadata.yml`
  - Or edit `docs/datasette/metadata.yml` and replace every `db: corevital_demo` with `db: corevital` (or your filename stem).

## Sharing (one link)

Deploy the same app so others can open one URL without logging in.

1. **Build a deployable artifact:** Your Datasette instance is the SQLite file + metadata. For run-detail to work, ensure `report_json` is populated (run the inflate script if you use blob-only storage).

2. **Deploy** to any host that can run Datasette or serve static files + a serverless SQLite API. Common options:

   - **Vercel** — [datasette-publish-vercel](https://datasette.io/plugins/datasette-publish-vercel) or a serverless function that runs Datasette.
   - **Cloud Run** — [datasette-publish-cloudrun](https://datasette.io/plugins/datasette-publish-cloudrun).
   - **Fly.io / Railway / your server** — run `datasette …` and expose the port.

3. **Share** the base URL (e.g. `https://your-app.vercel.app`) or the dashboards path (`https://your-app.vercel.app/-/dashboards`). No auth unless you add it at the host.

See [Datasette’s publishing docs](https://docs.datasette.io/en/stable/publish.html) for step-by-step guides per platform.

## What’s in the metadata

- **`corevital-runs`** — One table: `trace_id`, `created_at_utc`, `model_id`, `risk_score`, `prompt_hash` from `reports`, ordered by time.
- **`corevital-run-detail`** — Filter: trace (dynamic select from `reports`). Panels:
  - Summary (model, total_steps, elapsed_ms, prompt_tokens, risk_score)
  - Health flags (NaN, Inf, attention collapse, high entropy steps, repetition loop, mid-layer anomaly)
  - Narrative (from `extensions.narrative.summary`)
  - Entropy by step (line chart from `timeline[].logits_summary.entropy`)
  - Perplexity by step (line chart from `timeline[].logits_summary.perplexity`)
  - Attention entropy heatmap (layer × step from `timeline[].layers[].attention_summary.entropy_mean`)
  - Prompt and generated text (table; long content is scrollable in the browser)

All run-detail panels use `report_json` and `json_extract` / `json_each`. If `report_json` is empty, run the inflate script or use a DB written with `compress=False`.

## Troubleshooting

- **Where are the graphs?** — Go to **http://localhost:8001/-/dashboards** (not the Datasette home page). Click **CoreVital Run Detail**, then select a **Trace** in the dropdown. The line charts and heatmap appear below the filter.
- **Run detail is empty or “no data”** — Ensure `report_json` is populated. Run `scripts/inflate_report_json_from_blob.py <your_db>` or write new runs with `compress=False`.
- **“No such table: reports”** — You’re pointing at the wrong DB or the DB doesn’t have the CoreVital schema. Use a DB created by CoreVital’s SQLiteSink.
- **Dashboard not found** — Confirm `--metadata docs/datasette/metadata.yml` and that the metadata `db:` name matches your Datasette database name (filename stem or `-n`).
