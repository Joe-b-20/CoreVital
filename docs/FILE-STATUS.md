# File status vs main — what changed and what to keep

Generated from branch `feature/dashboard-metrics-benchmarks-docs` (or renamed branch) compared to `origin/main`. Use this to decide what to commit before merging.

---

## 1. Tracked files that differ from main

| Status | File | Explanation | Recommendation |
|--------|------|-------------|----------------|
| **M** | `.devcontainer/devcontainer.json` | Switched from Streamlit (`[dashboard]`, port 8501) to API (`[serve]`, port 8000). | **Keep** — matches React dashboard + corevital serve. |
| **D** | `.streamlit/config.toml` | Streamlit config; Streamlit app removed. | **Keep deleted** — no longer used. |
| **M** | `AI.md` | Optional deps and dashboard/serve references updated. | **Keep** — aligns with current architecture. |
| **M** | `README.md` | Dashboard URL, Path A wording, Try CoreVital dashboard link at top. | **Keep** — better first-time experience. |
| **D** | `dashboard.py` | Old Streamlit dashboard; UI now in separate React repo. | **Keep deleted** — canonical UI is React. |
| **M** | `docs/demo/README.md` | Demo section: dashboard repo link, Demo vs optional DB clarified. | **Keep** — accurate. |
| **M** | `docs/demo/corevital_demo.db` | Binary DB (larger or different content than on main). | **Delete one in main and replace with this one** — if you want a committed demo DB for “Database” mode, keep; else consider not committing (or .gitignore). |
| **M** | `docs/demo/sample_report.json` | Sample report JSON changed (e.g. truncated or different run). | **Delete one in main and replace with this one** — only commit if you want this as the canonical sample; otherwise restore from main or omit. |
| **M** | `docs/gpu-benchmarks.md` | Minor doc tweaks. | **Keep but content needs updateing pull from demo db and populate** if content is correct. |
| **M** | `docs/integration-examples.md` | Links/refs updated. | **Keep**. |
| **M** | `docs/mermaid/schema-v03-structure.mmd` | Diagram update. | **Keep** if schema is accurate. |
| **M** | `docs/model-compatibility.md` | Model/compat notes. | **Keep** if accurate. |
| **M** | `docs/production-deployment.md` | Dashboard URL, Path A. | **Keep**. |
| **M** | `docs/production-model-test-suite.md` | Install: `[dashboard]` → `[serve]`. | **Keep**. |
| **M** | `docs/v0.4.0-launch.md` | Dashboard URL. | **Keep**. |
| **M** | `docs/visual-examples.md` | Dashboard repo link, data sources wording. | **Keep**. |
| **M** | `pyproject.toml` | Optional deps (no `[dashboard]`; `[serve]`, etc.). | **Keep** — matches serve-only. |
| **M** | `requirements.txt` | Simplified (e.g. core + optional note). | **Keep** if it matches pyproject. |
| **A** | `scripts/gen_demo_db.py` | New script to generate demo DB (GPU, conda). | **Keep** — documented in docs/demo/README. |
| **M** | `scripts/try_demo.sh` | Runs `corevital serve` instead of Streamlit; points to React dashboard. | **Keep**. |
| **M** | `src/CoreVital/__init__.py` | Package exports / version. | **Keep** if intentional. |
| **M** | `src/CoreVital/cli.py` | CLI: serve command, run options, etc. | **Keep** — needed for serve. |
| **M** | `src/CoreVital/instrumentation/performance.py` | Perf instrumentation tweaks. | **Keep** if tests pass. |
| **M** | `src/CoreVital/models/hf_loader.py` | Model loading changes. | **Keep** if tests pass. |
| **M** | `src/CoreVital/reporting/validation.py` | Schema validation (e.g. 0.4.0). | **Keep** if schema is correct. |
| **M** | `src/CoreVital/sinks/sqlite_sink.py` | SQLite sink behavior. | **Keep** if tests pass. |
| **M** | `tests/test_mock_instrumentation.py` | Mock tests updated. | **Keep** — run pytest. |
| **M** | `tests/test_performance.py` | Perf tests. | **Keep** — run pytest. |
| **M** | `tests/test_persistence.py` | Persistence tests. | **Keep** — run pytest. |
| **M** | `tests/test_smoke_gpt2_cpu.py` | Smoke test. | **Keep** — run pytest. |

**Summary (tracked):** Keep all **M** and **D** and **A** for the “docs + dashboard cleanup + serve” story. Optionally revert or drop changes to `docs/demo/corevital_demo.db` and `docs/demo/sample_report.json` if you prefer to keep main’s version or stop committing large binaries/JSON.

---

## 2. Untracked files

| File | Explanation | Recommendation |
|------|-------------|----------------|
| `docs/BRANCH-RENAME.md` | Instructions: push new branch, then merge to main. | **Delete** — add and commit. |
| `docs/FILE-STATUS.md` | This file; audit of changes vs main. | **Delete after excuting** — add and commit (or move to a one-off doc). |
| `docs/dashboard-options-research.md` | Research on dashboard options (Datasette, Evidence, etc.). | **Keep** — add and commit; useful reference. |
| `docs/dashboard-react-spec.md` | Spec for React dashboard (data sources, API, repo link). | **Keep** — add and commit. |
| `docs/datasette/README.md` | How to use Datasette with CoreVital DB. | **Keep** — add and commit. |
| `docs/datasette/metadata.yml` | Datasette dashboard metadata. | **Keep** — add and commit. |
| `docs/demo/corevital.db` | Alternate/local demo DB (likely generated). | **Don’t commit** — add `docs/demo/corevital.db` to `.gitignore` if you want to avoid committing. |
| `docs/demo/corevital_demo.db.bak` | Backup of demo DB. | **Don’t commit** — add `*.bak` under docs/demo or globally if desired. |
| `docs/demo/trace_3d992273_performance_detailed.json` | Per-run trace JSON (detailed perf). | **Delete** — generated artifact; add `docs/demo/trace_*_performance_detailed.json` to `.gitignore`. |
| `docs/demo/trace_ac3732b7_performance_detailed.json` | Same. | **Delete** — as above. |
| `docs/demo/trace_b5b0fa32_performance_detailed.json` | Same. | **Delete** — as above. |
| `docs/demo/trace_f28f1386_performance_detailed.json` | Same. | **Delete** — as above. |
| `notebooks/try_corevital.ipynb` | Notebook to try CoreVital (e.g. Colab); updated for dashboard + serve. | **Keep** — add and commit. |
| `scripts/export_report_html.py` | Export a report to Streamlit-like HTML. | **Keep** — add and commit; useful for offline sharing. |
| `scripts/gen_demo_traces.py` | Generate demo trace JSONs (e.g. for dashboard repo). | **Keep** — add and commit if you use it to feed the dashboard repo. |
| `scripts/inflate_report_json_from_blob.py` | Inflate report JSON from SQLite blob (e.g. for Datasette). | **Keep** — add and commit; referenced in datasette docs. |
| `src/CoreVital/api.py` | FastAPI app for `corevital serve` (list/load traces). | **Keep** — add and commit; required for Dashboard “Database” mode. |

**Summary (untracked):**  
- **Add and commit:** `docs/BRANCH-RENAME.md`, `docs/FILE-STATUS.md`, `docs/dashboard-options-research.md`, `docs/dashboard-react-spec.md`, `docs/datasette/`, `notebooks/try_corevital.ipynb`, `scripts/export_report_html.py`, `scripts/gen_demo_traces.py`, `scripts/inflate_report_json_from_blob.py`, `src/CoreVital/api.py`.  
- **Do not commit (and consider ignoring):** `docs/demo/corevital.db`, `docs/demo/corevital_demo.db.bak`, `docs/demo/trace_*_performance_detailed.json`.

---

## 3. Suggested .gitignore additions (optional)

To keep the repo clean and avoid accidentally committing generated/local files:

```gitignore
# Demo artifacts (generated or local copies)
docs/demo/corevital.db
docs/demo/*.bak
docs/demo/trace_*_performance_detailed.json
```

---

## 4. Commands to stage and push (after renaming branch)

```bash
# Rename branch
git branch -m feature/docs-dashboard-cleanup

# Stage only what you want to commit (tracked changes + chosen untracked)
git add -u
git add docs/BRANCH-RENAME.md docs/FILE-STATUS.md
git add docs/dashboard-options-research.md docs/dashboard-react-spec.md
git add docs/datasette/
git add notebooks/
git add scripts/export_report_html.py scripts/gen_demo_traces.py scripts/inflate_report_json_from_blob.py
git add src/CoreVital/api.py

# Optional: add .gitignore entries for demo artifacts
# git add .gitignore

# Review
git status

# Commit and push
git commit -m "docs: React dashboard cleanup, serve-only, demo in dashboard repo; add api.py, datasette, scripts"
git push -u origin feature/docs-dashboard-cleanup
```

Then open a PR from `feature/docs-dashboard-cleanup` → `main` and merge after review.
