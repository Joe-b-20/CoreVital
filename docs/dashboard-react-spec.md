# CoreVital Dashboard — React rebuild spec

This doc describes the API and report shape for a **standalone React dashboard** built outside the CoreVital repo (deployed at [https://main.d2maxwaq575qed.amplifyapp.com](https://main.d2maxwaq575qed.amplifyapp.com)). The in-repo Streamlit/Reflex dashboards have been removed; this React app is the canonical dashboard and consumes the same data via `corevital serve`.

**Where is the React app?** [corevital-dashboard](https://github.com/Joe-b-20/corevital-dashboard) — a separate repo, deployable to AWS Amplify. See that repo’s README for run instructions. No dependency on the CoreVital package.

## Data sources

1. **Demo** — Static sample traces in the dashboard repo at `public/demo/` (`index.json` + `trace_*.json`). The app fetches these at load; no backend required. Demo content is maintained in the dashboard repo.
2. **Database** — SQLite at user-provided path via **CoreVital local API** (`corevital serve`): list traces, load report by `trace_id`. The dashboard calls this backend when the user selects "Database" and connects.
3. **Drag-and-drop** — User drops a CoreVital JSON report file in the browser; no upload, FileReader only.

## API (CoreVital backend: `corevital serve`)

When using **Database** mode, the dashboard talks to the CoreVital local API (this repo: `corevital serve` → `src/CoreVital/api.py`). Demo is static in the dashboard repo; no `/api/demo` in CoreVital.

| Method | Path | Purpose |
|--------|------|--------|
| GET | `/api/traces?db_path=...` | List traces. Query: `db_path` (optional; default from `COREVITAL_DB_PATH` or `runs/corevital.db`), optional `limit`, `model_id`, `prompt_hash`, `order_asc`. |
| GET | `/api/traces/{trace_id}?db_path=...` | Load full report JSON for `trace_id`. Short id (e.g. 8 chars) matched as prefix. |

### SQLite schema (read-only)

- **Table** `reports`: `trace_id`, `created_at_utc`, `model_id`, `schema_version`, `report_json` (TEXT or NULL), `report_blob` (BLOB, gzip), `prompt_hash`, `risk_score`.
- **List:** `SELECT trace_id, created_at_utc, model_id, schema_version, prompt_hash, risk_score FROM reports ORDER BY created_at_utc DESC LIMIT ?` (and optional WHERE for model_id / prompt_hash).
- **Load:** `SELECT report_json, report_blob FROM reports WHERE trace_id = ? OR trace_id LIKE ?`. If `report_blob` is not null, decompress with gzip and parse JSON.

## Report JSON shape (summary)

Frontend consumes a single report object. Key top-level keys:

- **model** — `hf_id`, `architecture`, `num_layers`, `hidden_size`, `num_attention_heads`, `device`, `quantization`, etc.
- **summary** — `generated_tokens`, `elapsed_ms`, `total_steps`.
- **prompt** — `text`, `num_tokens`.
- **generated** — `output_text`.
- **run_config** — `generation` (temperature, top_k, top_p), `seed`, `max_new_tokens`.
- **health_flags** — `nan_detected`, `inf_detected`, `attention_collapse_detected`, `high_entropy_steps`, `repetition_loop_detected`, `mid_layer_anomaly_detected` (booleans / int).
- **extensions.risk** — `risk_score`, `risk_factors`, `blamed_layers`.
- **extensions.early_warning** — `failure_risk`, `warning_signals`.
- **extensions.rag** — (optional) `context_token_count`, `retrieved_doc_ids`, `retrieved_doc_titles`.
- **timeline** — Array of steps. Each step:
  - `step_index`, `token` (e.g. `token_text`), `logits_summary` (entropy, perplexity, surprisal, top_k_margin, voter_agreement), `layers` (array of layer objects with `attention_summary`: entropy_mean/min/max, concentration_max/min, collapsed_head_count, focused_head_count).
- **prompt_analysis** — (optional) Layer-wise prompt metrics; basin, surprisals, etc.
- **trace_id**, **schema_version**.

## UI views to replicate

1. **Run detail (single report)**  
   - Header: model name, architecture, layers, generated tokens, elapsed.  
   - Run details expander: trace_id, schema, device, generation params, prompt/output preview.  
   - Health flags (badges: NaN, Inf, attention collapse, high entropy steps, repetition loop, mid-layer anomaly).  
   - Early warning (failure risk, signals).  
   - Risk score (score, factors, blamed layers).  
   - Logits over time: tabs Entropy / Perplexity / Surprisal / Top-K margin / Voter agreement (line or bar charts; entropy with 4.0 threshold line).  
   - Attention heatmaps: layer × step matrix, metric select (entropy_mean, concentration_max, etc.).  
   - Prompt analysis (if present): layer transforms, surprisals, basin.  
   - Performance: total wall time, breakdown by operation (bar/pie), optional nested detailed_breakdown in trace.  
   - Export report as JSON.  

2. **Compare runs (database source only)**  
   - Multiselect traces from list.  
   - Side-by-side metrics table (same keys as Streamlit compare), with “Diff from Run 1” column and highlight for differing cells.  
   - Optional expander: prompts and outputs per run.  

3. **Data source selector**  
   - Demo (static `public/demo/` in dashboard repo) | Database (CoreVital `corevital serve` + DB path) | Drag-and-drop (local JSON file in browser).  

## Tech stack (dashboard repo)

- **React** + **TypeScript** + **Vite**; **Zustand**, **ECharts**, **Tailwind**.  
- **Demo data:** Static files in `public/demo/` (no backend).  
- **Database mode:** Uses CoreVital API from this repo (`corevital serve`, port 8000).  

No backend in the dashboard repo; only contract is this spec and the report JSON shape. See [corevital-dashboard](https://github.com/Joe-b-20/corevital-dashboard) for layout and `public/demo/` for demo content.
