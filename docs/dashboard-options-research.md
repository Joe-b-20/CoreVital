# Dashboard options: one tool for local use + easy sharing

**Goal:** One out-of-the-box dashboard for CoreVital — use it locally (runs, test runs) and share with one link. No maintaining Streamlit/Reflex code; replicate your current layout and features where possible.

**Constraints:** SQLite (and/or JSON) as source; support per-step series, heatmaps, text (narrative, prompts); shareable link; sustainable for future metrics.

---

## Summary table

| Tool | Local use | Share (one link) | Replicate run-detail (per-step, heatmaps, text) | Maintenance | One for life |
|------|-----------|------------------|--------------------------------------------------|-------------|--------------|
| **Datasette + dashboards** | `datasette corevital.db` | Deploy → URL (Vercel, Cloud Run, Netlify) | Yes (SQL + Vega-Lite) | YAML + SQL only | ✅ |
| **Evidence.dev** | Run locally (SQLite in project) | Deploy static site → URL | Yes (SQL + heatmap, line chart) | Markdown + SQL | ✅ |
| **Observable Framework** | `npm run dev` | Deploy static → URL; runtime fetch | Yes (full control, fetch report by ID) | JS/Markdown cells | ✅ |
| **Grafana + Prometheus** | Local or cloud | Public dashboard / snapshot link | Only aggregates (no per-step in Prometheus) | UI only | For ops; need +1 for run detail |
| **Quarto Dashboard** | Local; publish | Quarto Pub / GitHub Pages / Netlify | Yes (Observable JS + fetch) | Qmd + JS | ✅ |
| **Retool / Appsmith** | Self-host or cloud | Public/embed link (plan-dependent) | Possible (widgets + API) | Config, not code | Heavier stack |

---

## Top recommendations

### 1. **Datasette + datasette-dashboards** (best fit for “SQLite is source of truth”)

- **What it is:** Open-source server that turns SQLite into a JSON API and optional dashboards. Dashboards are defined in `metadata.yml`: SQL queries + Vega-Lite charts (line, heatmap, bar, stat), filters, markdown.
- **Local:** `pip install datasette datasette-dashboards` then `datasette docs/demo/corevital_demo.db --load-extension sqlite3-json1` (or your DB path). Open `http://localhost:8001/-/dashboards`.
- **Share:** Deploy the same DB (or a copy) to [Vercel](https://vercel.com/docs/functions/serverless-functions/using-python), [Cloud Run](https://datasette.io/plugins/datasette-publish-cloudrun), or [Netlify](https://docs.datasette.io/en/stable/publish.html). Share that URL — one link, no login.
- **Replicate your dashboard:**  
  - **List runs:** Table from `SELECT trace_id, created_at_utc, model_id, risk_score FROM reports ORDER BY created_at_utc DESC`.  
  - **Run detail:** One dashboard per “report view”. Use SQLite `json_extract()` / `json_each()` on `report_json` (or decompress `report_blob`) to get:
    - Per-step series: e.g. `json_each(report_json, '$.timeline')` → entropy, perplexity, surprisal → Vega-Lite line chart.
    - Layer×step heatmap: extract layer/step and value from timeline → Vega-Lite `mark: rect` heatmap.
    - Narrative, health, risk: `json_extract(report_json, '$.extensions.narrative')` etc. → stat panels or markdown.
  - Filters: dropdown for `trace_id` (from a query), so “pick run → see detail” is one dashboard with one variable.
- **Pros:** No custom app code; SQLite-native; one YAML + SQL config; Vega-Lite is powerful (heatmaps, lines, tooltips).  
- **Cons:** JSON extraction in SQL can be verbose; `datasette-dashboards` is still evolving (check [plugin docs](https://datasette.io/plugins/datasette-dashboards)).  
- **One for life:** Add new SQL queries and Vega-Lite panels to `metadata.yml` when you add metrics; no new framework.

**Docs:** [Datasette](https://docs.datasette.io/) · [datasette-dashboards](https://github.com/rclement/datasette-dashboards) · [Vega-Lite heatmap](https://vega.github.io/vega-lite/examples/rect_heatmap.html)

---

### 2. **Evidence.dev**

- **What it is:** “BI as code” — markdown pages + SQL + chart components (line, heatmap, table, etc.). Connects to SQLite (and others). Renders to static HTML.
- **Local:** Add SQLite as a [data source](https://docs.evidence.dev/core-concepts/data-sources/sqlite); run `evidence dev`; open browser.
- **Share:** [Deploy](https://docs.evidence.dev/deployment/overview) as static site (e.g. Netlify, GitHub Pages). Share the site URL (and optionally page anchors for “run detail” pages).
- **Replicate your dashboard:**  
  - One project; pages like `list_runs.md`, `run_detail.md` (with `trace_id` param).  
  - SQL: same idea as Datasette — `json_extract` / `json_each` on the report column to feed [Line chart](https://docs.evidence.dev/components/charts/line-chart), [Heatmap](https://docs.evidence.dev/components/charts/heatmap), and tables.  
  - [Templated pages](https://docs.evidence.dev/core-concepts/templated-pages) can drive “one page per run” or one page with a run selector.
- **Pros:** SQLite supported; heatmap + line chart; clean “pages as files” model; sub-second interaction.  
- **Cons:** You stay in SQL + markdown (and a bit of Evidence component syntax); nested JSON extraction still in SQL.  
- **One for life:** New metrics → new SQL + new chart blocks in existing pages or new pages.

**Docs:** [Evidence](https://evidence.dev/) · [SQLite](https://docs.evidence.dev/core-concepts/data-sources/sqlite) · [Heatmap](https://docs.evidence.dev/components/charts/heatmap)

---

### 3. **Observable Framework**

- **What it is:** Static site generator for data apps (Markdown + JavaScript cells). Can load data at **runtime** via `fetch()` (e.g. your API or static JSON).
- **Local:** `npx @observablehq/framework create corevital-dashboard` then add a “report viewer” page that takes `?trace_id=...` or `?report=...`, fetches the report (from an API or a static file), and renders with Observable Plot / D3 (entropy line, layer×step heatmap, narrative in markdown).
- **Share:** [Deploy](https://observablehq.com/framework/deploying) to Netlify, GitHub Pages, etc. One URL; optional query params for “view this run”.
- **Replicate your dashboard:**  
  - One “app” with: run list (from API or pre-built JSON) + run detail (fetch one report by ID).  
  - Full control over layout (sidebar, tabs, long prompts in scrollable areas). No long-prompt crashes; you control rendering.  
  - Add metrics later = add new cells or sections.
- **Pros:** Maximum flexibility; runtime fetch so “share this run” = same URL + `?trace_id=xyz`; no backend if you serve JSON from the same static host.  
- **Cons:** You write JS (and a bit of Markdown); need a way to serve report JSON (static files or tiny API).  
- **One for life:** One Framework project; extend with new sections/cells as CoreVital grows.

**Docs:** [Observable Framework](https://observablehq.com/framework) · [Data loaders / fetch](https://observablehq.com/framework/loaders) · [Observable Plot](https://observablehq.com/plot/)

---

## How sharing works in each

- **Datasette:** Deploy instance → share `https://your-deploy-url.com/dashboards/corevital` (or a specific dashboard). Optional: link with query params if the plugin supports filter state in URL.
- **Evidence:** Deploy static site → share `https://your-site.com` or `https://your-site.com/run_detail?trace_id=abc`.
- **Observable Framework:** Deploy static site → share `https://your-site.com` or `https://your-site.com/report?trace_id=abc`. No login; one link.

All three support “one link for others to see” without maintaining a custom Streamlit/Reflex app.

---

## Suggested path

1. **Try Datasette first** (same DB, no new stack):  
   - Install Datasette + datasette-dashboards, point at `docs/demo/corevital_demo.db`.  
   - Add one “runs list” dashboard and one “run detail” dashboard (trace_id filter + SQL pulling from `report_json` / `report_blob`).  
   - If the YAML + SQL + Vega-Lite workflow fits and you can get heatmaps and per-step charts without pain, standardize on it and deploy for sharing.

2. **If you prefer “pages as markdown + SQL” and want heatmaps/line charts out of the box,** try **Evidence** with the same SQLite DB and similar SQL; use templated or param-driven pages for run detail.

3. **If you want maximum control and “share this run” = URL + trace_id,** use **Observable Framework** and a small report API (or static JSON exports). Then you have one static dashboard for the life of CoreVital and you only add cells/sections when you add metrics.

4. **Grafana + Prometheus** remains the right place for operational metrics and alerting; use it in parallel with whichever of the above you choose for run-level detail and sharing.

---

## References

- Datasette: https://datasette.io/  
- datasette-dashboards: https://github.com/rclement/datasette-dashboards  
- Evidence: https://evidence.dev/  
- Observable Framework: https://observablehq.com/framework  
- Vega-Lite (heatmap, line): https://vega.github.io/vega-lite/examples/
