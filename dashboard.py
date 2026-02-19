# ============================================================================
# CoreVital - Streamlit Dashboard
#
# Purpose: Interactive visualization of CoreVital JSON reports
# Inputs: JSON report files from ./runs/ or user upload
# Outputs: Interactive charts, heatmaps, health flag indicators
# Dependencies: streamlit, plotly, json, pathlib
# Usage: streamlit run dashboard.py
#
# Changelog:
#   2026-02-11: Phase-1d — Initial dashboard with entropy chart, attention
#               heatmap, health flags, prompt analysis, latency breakdown
# ============================================================================

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import streamlit as st

from CoreVital.reporting.attention_queries import (
    get_attention_from_token,
    get_attention_to_token,
    get_basin_anomalies,
    get_top_connections,
)

# ---------------------------------------------------------------------------
# Plotly is optional — graceful fallback to st.line_chart / st.bar_chart
# ---------------------------------------------------------------------------
try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ============================================================================
# Page config
# ============================================================================
st.set_page_config(
    page_title="CoreVital Dashboard",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Metric keys & explanations (for average users running open-source models)
# Source: docs/Phase1 metrics analysis.md, docs/mermaid/metrics-signal-interpretation.mmd
# ============================================================================

METRIC_GUIDE = """
**What is this report?**
CoreVital records what your model does *inside* each generation step (confidence, attention, hidden states)
so you can spot confusion, repetition, or numerical issues without reading raw logs.

---

**Health Flags** (top-level summary)  
- **NaN/Inf:** Any “not a number” or infinity in the model’s internals → **stop and debug** (bad inputs, precision, or code).  
- **Attention collapse:** Some attention heads put almost all weight on one token. Common in smaller models; only a problem if generation is clearly broken.  
- **High entropy steps:** How many steps had entropy &gt; 4 (model was very uncertain). A few is normal; many suggests confusion.  
- **Repetition loop:** Last-layer hidden states became nearly identical over 3+ steps → model may be stuck repeating.  
- **Mid-layer anomaly:** Unusual values (NaN/Inf or huge norms) in the middle layers → possible numerical or training issue.

---

**Entropy** (per generated token)  
*“How unsure was the model when it picked this token?”*  
- **&lt; 2:** Confident (one clear choice).  
- **2–4:** Normal (several plausible options).  
- **&gt; 4:** Confused (many options, no clear winner). A **red dashed line** at 4.0 marks this. Sudden spikes can mean lost context or weird input.

**Perplexity** (per generated token)  
*“Roughly how many tokens was the model choosing between?”*  
- Same information as entropy, in a different scale: perplexity = 2^entropy.  
- **Low (e.g. 1–4):** Confident. **High (e.g. &gt; 16):** Very uncertain.

**Surprisal** (per generated token)  
*“How surprised was the model by the token it actually produced?”*  
- **&lt; 2:** Token was expected (high probability).  
- **2–5:** Plausible but not the top choice.  
- **&gt; 5:** Model was surprised (unlikely token). Spikes show where the model struggled.

---

**Attention heatmaps** (per layer, per step)  
- **Entropy mean:** How spread out attention is. Very low (e.g. &lt; 0.5) in a head = “collapse” (all weight on one position).  
- **Concentration max:** Max weight on any one position. Near 1.0 = one position ate almost all attention.  
- **Collapsed / focused head count:** How many heads in that layer look collapsed or very focused. Useful to see which layers behave oddly.

**Hidden state L2 norms**  
- Size of the vectors the model passes between layers.  
- **Normal:** Values in a stable range (often model-dependent). **Very high** can mean activations blowing up; **very low** can mean dying activations. Look for sudden jumps or odd patterns by layer/step.

---

**Prompt analysis** (from the extra pass over your prompt)  
- **Layer transformations:** How much each layer changes the representation (cosine similarity between consecutive layers). Healthy models usually show moderate change (e.g. 0.2–0.5); very low or very high can be worth a look.  
- **Prompt surprisals:** How “surprised” the model was by each prompt token. High values = model found that part of the prompt unusual or hard.  
- **Basin score:** Whether attention focuses on the *middle* of the prompt. Low (&lt; 0.3) can indicate “lost in the middle” (model ignoring the middle of long prompts).

---

**Performance**  
- Time spent in each stage (load, tokenize, generate, build report). Use it to see where time goes (e.g. model load vs actual generation).
"""


def load_report(path: str) -> Dict[str, Any]:
    """Load a JSON report file."""
    with open(path, "r") as f:
        return json.load(f)


def health_badge(label: str, value: Any, *, good_when_false: bool = True) -> str:
    """Return a colored markdown badge for a health flag."""
    if isinstance(value, bool):
        is_good = (not value) if good_when_false else value
        icon = "✅" if is_good else "🔴"
        display = str(value)
    elif isinstance(value, int):
        is_good = value == 0
        icon = "✅" if is_good else "⚠️"
        display = str(value)
    else:
        icon = "❓"
        display = str(value)
    return f"{icon} **{label}:** {display}"


def extract_timeline_series(
    report: Dict[str, Any],
    field: str,
    source: str = "logits_summary",
) -> tuple[List[int], List[Optional[float]], List[str]]:
    """Extract a per-step metric series from the timeline.

    Returns (step_indices, values, token_texts).
    """
    steps: List[int] = []
    values: List[Optional[float]] = []
    tokens: List[str] = []
    for step in report.get("timeline", []):
        steps.append(step["step_index"])
        tokens.append(step.get("token", {}).get("token_text", "?"))
        src = step.get(source, {}) or {}
        values.append(src.get(field))
    return steps, values, tokens


def _extract_compare_metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a flat dict of comparable metrics from a report for side-by-side comparison."""
    model = report.get("model") or {}
    summary = report.get("summary") or {}
    run_config = report.get("run_config") or {}
    gen = run_config.get("generation") or {}
    prompt = report.get("prompt") or {}
    generated = report.get("generated") or {}
    health = report.get("health_flags") or {}
    risk = (report.get("extensions") or {}).get("risk") or {}
    quant = model.get("quantization") or {}
    quant_str = quant.get("method") if quant.get("enabled") else "None"
    prompt_text = (prompt.get("text") or "?")[:80]
    if len((prompt.get("text") or "")) > 80:
        prompt_text += "..."
    output_text = (generated.get("output_text") or "?")[:80]
    if len((generated.get("output_text") or "")) > 80:
        output_text += "..."
    return {
        "Risk score": risk.get("risk_score"),
        "NaN detected": health.get("nan_detected"),
        "Inf detected": health.get("inf_detected"),
        "Attention collapse": health.get("attention_collapse_detected"),
        "High entropy steps": health.get("high_entropy_steps"),
        "Repetition loop": health.get("repetition_loop_detected"),
        "Mid-layer anomaly": health.get("mid_layer_anomaly_detected"),
        "Model": model.get("hf_id"),
        "Num layers": model.get("num_layers"),
        "Hidden size": model.get("hidden_size"),
        "Num attention heads": model.get("num_attention_heads"),
        "Quantization": quant_str,
        "Device": model.get("device"),
        "Seed": run_config.get("seed"),
        "Max new tokens": run_config.get("max_new_tokens"),
        "Temperature": gen.get("temperature"),
        "Top-K": gen.get("top_k"),
        "Top-P": gen.get("top_p"),
        "Prompt tokens": prompt.get("num_tokens"),
        "Generated tokens": summary.get("generated_tokens"),
        "Total steps": summary.get("total_steps"),
        "Elapsed (ms)": summary.get("elapsed_ms"),
        "Prompt (preview)": prompt_text,
        "Output (preview)": output_text,
    }


def _format_compare_value(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if isinstance(v, float):
        return f"{v:.3g}" if abs(v) < 1e-3 or abs(v) >= 1e4 else f"{v:.2f}"
    return str(v)


def _render_compare_side_by_side(reports: List[Dict[str, Any]], trace_ids: List[str]) -> None:
    """Render a side-by-side metrics table with differing values highlighted."""
    import pandas as pd

    metrics_list = [_extract_compare_metrics(r) for r in reports]
    keys = list(metrics_list[0].keys()) if metrics_list else []
    col_headers = [f"Run {i + 1} ({tid[:8]})" for i, tid in enumerate(trace_ids)]
    # Build DataFrame: rows = metrics, columns = runs
    data: Dict[str, List[Any]] = {"Metric": keys}
    for i, m in enumerate(metrics_list):
        data[col_headers[i]] = [_format_compare_value(m.get(k)) for k in keys]
    # Add "Diff from Run 1?" column: "Same" or "Different"
    vals_per_row = [[_format_compare_value(metrics_list[i].get(k)) for i in range(len(reports))] for k in keys]
    data["Diff from Run 1?"] = [
        "Same" if all(v == vals_per_row[j][0] for v in vals_per_row[j]) else "Different" for j in range(len(keys))
    ]
    df = pd.DataFrame(data).set_index("Metric")

    # Highlight run columns that differ from Run 1
    def highlight_diff(row: pd.Series) -> List[str]:
        ref = row.iloc[0]
        styles = []
        for j in range(len(row)):
            col_name = row.index[j]
            if col_name == "Diff from Run 1?":
                styles.append("font-weight: bold" if row.iloc[j] == "Different" else "")
            else:
                styles.append("background-color: #fff3cd" if row.iloc[j] != ref else "")
        return styles

    styled = df.style.apply(highlight_diff, axis=1)
    st.markdown(
        "**Metrics comparison** — cells that differ from Run 1 are highlighted. Check **Diff from Run 1?** for a quick scan."
    )
    st.dataframe(styled, use_container_width=True, hide_index=False)
    # Optional: show prompt/output in expanders per run
    with st.expander("Prompts and outputs by run", expanded=False):
        for i, (report, tid) in enumerate(zip(reports, trace_ids, strict=True)):
            prompt = (report.get("prompt") or {}).get("text", "?")
            out = (report.get("generated") or {}).get("output_text", "?")
            st.markdown(f"**Run {i + 1}** (`{tid[:8]}`)")
            st.caption("Prompt")
            st.text(prompt)
            st.caption("Output")
            st.text(out)
            st.divider()


def build_layer_step_matrix(
    report: Dict[str, Any],
    field_path: str,  # e.g. "attention_summary.entropy_mean"
) -> tuple[List[List[Optional[float]]], int, int]:
    """Build a layers×steps matrix from timeline.

    Returns (matrix[layer][step], num_layers, num_steps).
    """
    timeline = report.get("timeline", [])
    if not timeline:
        return [], 0, 0
    num_steps = len(timeline)
    num_layers = len(timeline[0].get("layers", []))
    parts = field_path.split(".")

    matrix: List[List[Optional[float]]] = [[None] * num_steps for _ in range(num_layers)]
    for s_idx, step in enumerate(timeline):
        for l_idx, layer in enumerate(step.get("layers", [])):
            val: Any = layer
            for p in parts:
                val = val.get(p) if isinstance(val, dict) else None
                if val is None:
                    break
            if l_idx < num_layers:
                matrix[l_idx][s_idx] = val
    return matrix, num_layers, num_steps


# ============================================================================
# Sidebar — File selection
# ============================================================================
st.sidebar.title("CoreVital")
st.sidebar.caption("LLM Inference Health Monitor")
st.sidebar.divider()

# File selection
source = st.sidebar.radio(
    "Report source",
    ["Demo sample", "Local files", "Database", "Upload"],
    horizontal=True,
)

report_data: Optional[Dict[str, Any]] = None
report_size_bytes: Optional[int] = None
report_filename: Optional[str] = None
uploaded_bytes: Optional[bytes] = None
selected_path: Optional[Path] = None
db_path: Optional[str] = None

# Max size to load for "Raw JSON" expander without warning (MB)
LARGE_FILE_MB = 50

_REPO_ROOT = Path(__file__).resolve().parent
_DEMO_SAMPLE_PATH = _REPO_ROOT / "docs" / "demo" / "sample_report.json"
_DEMO_DB_PATH = _REPO_ROOT / "docs" / "demo" / "corevital_demo.db"
_DEFAULT_DB_PATH = Path("runs/corevital.db")

if source == "Demo sample":
    if _DEMO_SAMPLE_PATH.exists():
        report_data = load_report(str(_DEMO_SAMPLE_PATH))
        report_filename = "sample_report.json"
        report_size_bytes = _DEMO_SAMPLE_PATH.stat().st_size
        st.sidebar.success("Loaded bundled demo report. No model run required.")
    else:
        st.sidebar.warning(
            "Demo sample not found. Run from repo root or use Upload to load docs/demo/sample_report.json."
        )

elif source == "Local files":
    runs_dir = Path("runs")
    if runs_dir.exists():
        json_files = sorted(runs_dir.glob("trace_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        json_files = [f for f in json_files if "_performance_" not in f.name]
        if json_files:
            selected = st.sidebar.selectbox(
                "Select report",
                json_files,
                format_func=lambda p: f"{p.name} ({p.stat().st_size / (1024 * 1024):.2f} MB)",
            )
            if selected:
                selected_path = selected
                report_filename = selected.name
                report_size_bytes = selected.stat().st_size
                if report_size_bytes > LARGE_FILE_MB * 1024 * 1024:
                    st.sidebar.warning(
                        f"Large file ({report_size_bytes / (1024 * 1024):.1f} MB). "
                        "Consider using --sink sqlite for smaller storage."
                    )
                report_data = load_report(str(selected))
        else:
            st.sidebar.warning("No report files found in ./runs/")
    else:
        st.sidebar.warning("./runs/ directory not found. Run a model first.")

elif source == "Database":
    _initial_db = str(_DEFAULT_DB_PATH) if _DEFAULT_DB_PATH.exists() else str(_DEMO_DB_PATH)
    db_path = st.sidebar.text_input(
        "SQLite path",
        value=_initial_db,
        help="Path to corevital.db. Falls back to bundled demo DB when runs/corevital.db is absent.",
    )
    db_traces: List[Dict[str, Any]] = []
    if db_path:
        try:
            from CoreVital.sinks.sqlite_sink import SQLiteSink

            filter_model = st.sidebar.selectbox(
                "Filter by model",
                options=["(all)"]
                + sorted(
                    {t.get("model_id") or "" for t in SQLiteSink.list_traces(db_path, limit=500) if t.get("model_id")}
                ),
                key="db_filter_model",
            )
            filter_ph = st.sidebar.text_input(
                "Filter by prompt_hash (exact)", key="db_filter_ph", placeholder="optional"
            )
            traces = SQLiteSink.list_traces(
                db_path,
                limit=200,
                model_id=filter_model if filter_model and filter_model != "(all)" else None,
                prompt_hash=filter_ph.strip() or None,
            )
            db_traces = traces
            if traces:
                options = []
                for t in traces:
                    rs = t.get("risk_score")
                    rs_str = f" risk={rs:.2f}" if rs is not None else ""
                    options.append(f"{t['trace_id'][:8]} | {t['model_id']} | {t['created_at_utc']}{rs_str}")
                choice = st.sidebar.selectbox("Select trace", range(len(options)), format_func=lambda i: options[i])
                if choice is not None:
                    trace_id = traces[choice]["trace_id"]
                    report_data = SQLiteSink.load_report(db_path, trace_id)
                    if report_data:
                        report_filename = f"trace_{trace_id[:8]}.json"
                        report_size_bytes = len(json.dumps(report_data, separators=(",", ":")))
                    else:
                        st.sidebar.error("Failed to load report.")
            else:
                st.sidebar.warning("No reports in database (or no matches for filters). Run with --sink sqlite first.")
        except Exception as e:
            st.sidebar.error(f"Database error: {e}")

else:
    uploaded = st.sidebar.file_uploader("Upload JSON report", type=["json"])
    if uploaded:
        report_filename = uploaded.name
        uploaded_bytes = uploaded.getvalue()
        report_size_bytes = len(uploaded_bytes)
        report_data = json.loads(uploaded_bytes.decode("utf-8"))


# ============================================================================
# Main content
# ============================================================================
if source == "Database" and db_path:
    if "db_path" not in st.session_state:
        st.session_state.db_path = db_path
    st.session_state.db_path = db_path
    view_tab = st.radio("View", ["Run detail", "Compare runs"], horizontal=True, key="view_tab")
    if view_tab == "Compare runs":
        from CoreVital.sinks.sqlite_sink import SQLiteSink

        compare_traces = SQLiteSink.list_traces(st.session_state.db_path, limit=300)
        if compare_traces:
            import csv
            import io

            st.subheader("Compare runs")
            st.caption("Select two or more runs below to see metrics side-by-side with differences highlighted.")
            # Build options for multiselect: label -> trace_id
            trace_options = [
                f"{(t.get('trace_id') or '')[:8]} | {t.get('model_id') or '?'} | risk={t.get('risk_score'):.2f}"
                if t.get("risk_score") is not None
                else f"{(t.get('trace_id') or '')[:8]} | {t.get('model_id') or '?'}"
                for t in compare_traces
            ]
            trace_ids = [t.get("trace_id") for t in compare_traces]
            option_to_trace = {trace_options[i]: trace_ids[i] for i in range(len(compare_traces))}
            selected_labels = st.multiselect(
                "Select runs to compare (2 or more)",
                options=trace_options,
                default=[],
                key="compare_multiselect",
                help="Choose two or more runs to view metrics side-by-side.",
            )
            selected_ids = [option_to_trace[label] for label in selected_labels if label in option_to_trace]
            # Summary table (all traces)
            rows = []
            for t in compare_traces:
                rows.append(
                    {
                        "trace_id (short)": (t.get("trace_id") or "")[:8],
                        "model_id": t.get("model_id") or "",
                        "created_at_utc": t.get("created_at_utc") or "",
                        "risk_score": f"{t['risk_score']:.2f}" if t.get("risk_score") is not None else "",
                        "prompt_hash": (t.get("prompt_hash") or "")[:16] + "..."
                        if (t.get("prompt_hash") or "") and len(t.get("prompt_hash", "")) > 16
                        else (t.get("prompt_hash") or ""),
                    }
                )
            with st.expander("All runs in database", expanded=len(selected_labels) == 0):
                st.dataframe(rows, use_container_width=True, hide_index=True)
                buf = io.StringIO()
                w = csv.DictWriter(
                    buf, fieldnames=["trace_id (short)", "model_id", "created_at_utc", "risk_score", "prompt_hash"]
                )
                w.writeheader()
                w.writerows(rows)
                st.download_button(
                    "Export as CSV",
                    data=buf.getvalue(),
                    file_name="corevital_compare.csv",
                    mime="text/csv",
                    key="export_compare_csv",
                )
            # Side-by-side comparison when 2+ selected
            if len(selected_ids) >= 2:
                reports: List[Dict[str, Any]] = []
                loaded_ids: List[str] = []
                for tid in selected_ids:
                    r = SQLiteSink.load_report(st.session_state.db_path, tid)
                    if r:
                        reports.append(r)
                        loaded_ids.append(tid)
                if len(reports) >= 2:
                    _render_compare_side_by_side(reports, loaded_ids)
                else:
                    st.warning("Could not load one or more reports.")
            elif len(selected_labels) == 1:
                st.info("Select at least one more run to compare.")
        else:
            st.info("No traces in database. Run with --sink sqlite first.")
        st.stop()

if report_data is None:
    st.title("CoreVital Dashboard")
    st.info("Select or upload a report file from the sidebar to begin.")
    st.stop()

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
model_info = report_data.get("model", {})
summary = report_data.get("summary", {})
prompt_info = report_data.get("prompt", {})
generated = report_data.get("generated", {})

st.title(f"CoreVital — {model_info.get('hf_id', 'Unknown Model')}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Architecture", model_info.get("architecture", "?"))
col2.metric("Layers", model_info.get("num_layers", "?"))
col3.metric("Generated Tokens", summary.get("generated_tokens", "?"))
col4.metric("Elapsed", f"{summary.get('elapsed_ms', '?')}ms")

run_config = report_data.get("run_config") or {}
gen = run_config.get("generation") or {}
quant = model_info.get("quantization") or {}
quant_method = "?"
if quant.get("enabled"):
    quant_method = quant.get("method") or "enabled"
else:
    quant_method = "None"

with st.expander("Run details", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"**Trace ID:** `{report_data.get('trace_id', '?')}`")
        st.markdown(f"**Schema:** v{report_data.get('schema_version', '?')}")
        st.markdown(f"**Device:** {model_info.get('device', '?')}")
        st.markdown(f"**Dtype:** {model_info.get('dtype', '?')}")
        st.markdown("**Generation**")
        st.markdown(f"Temperature: {gen.get('temperature', '?')}")
        st.markdown(f"Seed: {run_config.get('seed', '?')}")
        st.markdown(f"Top-K: {gen.get('top_k', '?')}")
        st.markdown(f"Top-P: {gen.get('top_p', '?')}")
    with c2:
        st.markdown("**Model**")
        st.markdown(f"Num layers: {model_info.get('num_layers', '?')}")
        st.markdown(f"Hidden size: {model_info.get('hidden_size', '?')}")
        st.markdown(f"Num attention heads: {model_info.get('num_attention_heads', '?')}")
        st.markdown(f"Quantization: {quant_method}")
        st.markdown("**Prompt**")
        st.markdown(f"Prompt tokens: {prompt_info.get('num_tokens', '?')}")
        st.markdown(f"**Prompt:** _{prompt_info.get('text', '?')}_")
    with c3:
        st.markdown(f"**Output:** _{generated.get('output_text', '?')}_")

# ------------------------------------------------------------------
# Export report (Phase-8)
# ------------------------------------------------------------------
export_json = json.dumps(report_data, indent=2, ensure_ascii=False)
st.sidebar.download_button(
    "Export report (JSON)",
    data=export_json,
    file_name=report_filename or "corevital_report.json",
    mime="application/json",
    key="export_report_json",
)

# ------------------------------------------------------------------
# Narrative (Phase-7), when present
# ------------------------------------------------------------------
narrative_data = report_data.get("extensions", {}).get("narrative")
if narrative_data and narrative_data.get("summary"):
    st.info("**Summary:** " + narrative_data["summary"])

with st.expander("📖 How to read these metrics", expanded=False):
    st.markdown(METRIC_GUIDE)

# ------------------------------------------------------------------
# RAG context (Foundation F3), when present
# ------------------------------------------------------------------
rag_context = report_data.get("extensions", {}).get("rag")
if rag_context:
    st.divider()
    st.subheader("RAG Context")
    st.caption(
        "This run was executed with retrieval-augmented context. Use this to correlate behavior with context length or source documents."
    )
    rc1, rc2 = st.columns(2)
    with rc1:
        ctx_tokens = rag_context.get("context_token_count")
        if ctx_tokens is not None:
            st.metric("Context tokens", ctx_tokens)
        doc_ids = rag_context.get("retrieved_doc_ids") or []
        doc_titles = rag_context.get("retrieved_doc_titles") or []
        doc_count = max(len(doc_ids), len(doc_titles)) or (len(doc_ids) if doc_ids else 0)
        if doc_count > 0:
            st.metric("Retrieved documents", doc_count)
    with rc2:
        if doc_titles:
            st.markdown("**Titles:**")
            for t in doc_titles[:10]:
                st.markdown(f"- {t}")
            if len(doc_titles) > 10:
                st.caption(f"... and {len(doc_titles) - 10} more")
        elif doc_ids:
            st.markdown("**Doc IDs:**")
            for d in doc_ids[:10]:
                st.code(d, language=None)
            if len(doc_ids) > 10:
                st.caption(f"... and {len(doc_ids) - 10} more")
    meta = rag_context.get("retrieval_metadata")
    if meta:
        with st.expander("Retrieval metadata", expanded=False):
            st.json(meta)

# ------------------------------------------------------------------
# Health Flags
# ------------------------------------------------------------------
health_flags = report_data.get("health_flags")
if health_flags:
    st.divider()
    st.subheader("Health Flags")
    st.caption(
        "Quick summary of run health. ✅ = OK, 🔴/⚠️ = worth checking. "
        "NaN/Inf = numerical failure; Attention collapse = some heads focus on one token; "
        "High entropy steps = model was very uncertain; Repetition loop = may be stuck; "
        "Mid-layer anomaly = odd values in middle layers."
    )
    hc1, hc2, hc3 = st.columns(3)
    with hc1:
        st.markdown(health_badge("NaN Detected", health_flags.get("nan_detected", False)))
        st.markdown(health_badge("Inf Detected", health_flags.get("inf_detected", False)))
    with hc2:
        st.markdown(health_badge("Attention Collapse", health_flags.get("attention_collapse_detected", False)))
        st.markdown(health_badge("High Entropy Steps", health_flags.get("high_entropy_steps", 0)))
    with hc3:
        st.markdown(health_badge("Repetition Loop", health_flags.get("repetition_loop_detected", False)))
        st.markdown(health_badge("Mid-Layer Anomaly", health_flags.get("mid_layer_anomaly_detected", False)))

# ------------------------------------------------------------------
# Early warning (Phase-4), when present
# ------------------------------------------------------------------
ew_data = report_data.get("extensions", {}).get("early_warning")
if ew_data:
    st.divider()
    st.subheader("Early Warning")
    st.caption("Failure risk and signals derived from timeline (entropy trend, repetition, etc.). Use for triage.")
    ew_risk = ew_data.get("failure_risk")
    ew_signals = ew_data.get("warning_signals") or []
    if ew_risk is not None:
        st.metric("Failure risk", f"{ew_risk:.2f}")
    if ew_signals:
        st.markdown("**Signals:** " + ", ".join(ew_signals))

# ------------------------------------------------------------------
# Risk score and layer blame (Phase-2), when present
# ------------------------------------------------------------------
risk_data = report_data.get("extensions", {}).get("risk")
if risk_data:
    st.divider()
    st.subheader("Risk Score")
    st.caption(
        "Single risk score (0–1) from health flags. High = NaN/Inf, repetition, mid-layer anomaly, or many high-entropy steps. "
        "Blamed layers = layers that had anomalies or attention collapse."
    )
    r_score = risk_data.get("risk_score")
    r_factors = risk_data.get("risk_factors") or []
    r_blamed = risk_data.get("blamed_layers") or []
    if r_score is not None:
        st.metric("Risk score", f"{r_score:.2f}")
    if r_factors:
        st.markdown("**Contributing factors:** " + ", ".join(r_factors))
    if r_blamed:
        st.markdown(f"**Blamed layers:** {r_blamed}")

# ------------------------------------------------------------------
# Entropy & Perplexity over time
# ------------------------------------------------------------------
st.divider()
st.subheader("Logits Metrics Over Time")
st.caption(
    "**Entropy:** &lt;2 confident, 2–4 normal, &gt;4 confused (red line). "
    "**Perplexity:** same idea as entropy (≈ how many tokens the model is choosing between). "
    "**Surprisal:** how surprised the model was by each token; spikes = hard or unexpected tokens."
)
steps, entropies, tokens = extract_timeline_series(report_data, "entropy")
_, perplexities, _ = extract_timeline_series(report_data, "perplexity")
_, surprisals, _ = extract_timeline_series(report_data, "surprisal")
_, topk_margins, _ = extract_timeline_series(report_data, "top_k_margin")
_, voter_agrs, _ = extract_timeline_series(report_data, "voter_agreement")

tab_ent, tab_perp, tab_surp, tab_topk, tab_voter = st.tabs(
    ["Entropy", "Perplexity", "Surprisal", "Top-K Margin", "Voter Agreement"]
)

with tab_ent:
    if any(v is not None for v in entropies):
        if HAS_PLOTLY:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=[v if v is not None else None for v in entropies],
                    mode="lines+markers",
                    name="Entropy (bits)",
                    text=tokens,
                    hovertemplate="Step %{x}<br>Token: %{text}<br>Entropy: %{y:.3f}<extra></extra>",
                )
            )
            # High entropy threshold line
            fig.add_hline(y=4.0, line_dash="dash", line_color="red", annotation_text="High entropy (4.0)")
            fig.update_layout(
                xaxis_title="Generation Step",
                yaxis_title="Entropy (bits)",
                height=350,
                margin=dict(t=30, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            import pandas as pd

            df = pd.DataFrame({"Step": steps, "Entropy": entropies})
            st.line_chart(df, x="Step", y="Entropy")
    else:
        st.info("No entropy data available.")

with tab_perp:
    if any(v is not None for v in perplexities):
        if HAS_PLOTLY:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=[v if v is not None else None for v in perplexities],
                    mode="lines+markers",
                    name="Perplexity",
                    text=tokens,
                    hovertemplate="Step %{x}<br>Token: %{text}<br>Perplexity: %{y:.2f}<extra></extra>",
                )
            )
            fig.update_layout(
                xaxis_title="Generation Step",
                yaxis_title="Perplexity (2^entropy)",
                height=350,
                margin=dict(t=30, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            import pandas as pd

            df = pd.DataFrame({"Step": steps, "Perplexity": perplexities})
            st.line_chart(df, x="Step", y="Perplexity")
    else:
        st.info("No perplexity data available.")

with tab_surp:
    if any(v is not None for v in surprisals):
        if HAS_PLOTLY:
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=steps,
                    y=[v if v is not None else None for v in surprisals],
                    text=tokens,
                    hovertemplate="Step %{x}<br>Token: %{text}<br>Surprisal: %{y:.3f}<extra></extra>",
                )
            )
            fig.update_layout(
                xaxis_title="Generation Step",
                yaxis_title="Surprisal (-log₂ p)",
                height=350,
                margin=dict(t=30, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            import pandas as pd

            df = pd.DataFrame({"Step": steps, "Surprisal": surprisals})
            st.bar_chart(df, x="Step", y="Surprisal")
    else:
        st.info("No surprisal data available.")

with tab_topk:
    if any(v is not None for v in topk_margins):
        if HAS_PLOTLY:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=[v if v is not None else None for v in topk_margins],
                    mode="lines+markers",
                    name="Top-K Margin",
                    text=tokens,
                    hovertemplate="Step %{x}<br>Token: %{text}<br>Top-K Margin: %{y:.3f}<extra></extra>",
                )
            )
            fig.update_layout(
                xaxis_title="Generation Step",
                yaxis_title="Top-K Margin",
                height=350,
                margin=dict(t=30, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            import pandas as pd

            df = pd.DataFrame({"Step": steps, "Top-K Margin": topk_margins})
            st.line_chart(df, x="Step", y="Top-K Margin")
    else:
        st.info("No top-k margin data available.")

with tab_voter:
    if any(v is not None for v in voter_agrs):
        if HAS_PLOTLY:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=[v if v is not None else None for v in voter_agrs],
                    mode="lines+markers",
                    name="Voter Agreement",
                    text=tokens,
                    hovertemplate="Step %{x}<br>Token: %{text}<br>Voter Agreement: %{y:.3f}<extra></extra>",
                )
            )
            fig.update_layout(
                xaxis_title="Generation Step",
                yaxis_title="Voter Agreement",
                height=350,
                margin=dict(t=30, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            import pandas as pd

            df = pd.DataFrame({"Step": steps, "Voter Agreement": voter_agrs})
            st.line_chart(df, x="Step", y="Voter Agreement")
    else:
        st.info("No voter agreement data available.")

# --- Entropy vs Token Position (#25) ---
st.caption("**Entropy vs position:** uncertainty along the generation; vertical line = end of prompt.")
if entropies and HAS_PLOTLY:
    prompt_tokens = (report_data.get("summary") or {}).get("prompt_tokens") or 0
    fig_ep = go.Figure()
    fig_ep.add_trace(
        go.Scatter(
            x=steps,
            y=[v if v is not None else None for v in entropies],
            mode="lines+markers",
            name="Entropy (bits)",
            text=tokens,
            hovertemplate="Position %{x}<br>Token: %{text}<br>Entropy: %{y:.3f}<extra></extra>",
        )
    )
    if prompt_tokens > 0:
        fig_ep.add_vline(
            x=prompt_tokens - 0.5,
            line_dash="dash",
            line_color="gray",
            annotation_text="Prompt end",
        )
    fig_ep.add_hline(y=4.0, line_dash="dot", line_color="red", annotation_text="High (4.0)")
    fig_ep.update_layout(
        xaxis_title="Position (generation step)",
        yaxis_title="Entropy (bits)",
        height=280,
        margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig_ep, use_container_width=True)

# --- Colored Output by Uncertainty (#26) ---
st.subheader("Colored Output")
st.caption("Generated text colored by per-token entropy: green = low uncertainty, yellow = medium, red = high.")
if steps and tokens and entropies:
    parts: List[str] = []
    for tok, ent in zip(tokens, entropies, strict=True):
        if ent is None:
            color = "inherit"
        elif ent > 4.0:
            color = "#e74c3c"
        elif ent >= 2.0:
            color = "#f1c40f"
        else:
            color = "#27ae60"
        escaped = tok.replace("\\", "\\\\").replace("<", "&lt;").replace(">", "&gt;").replace("&", "&amp;")
        parts.append(f'<span style="background-color:{color}; color:black; padding:0 1px;">{escaped}</span>')
    st.markdown(
        "<div style='font-family:monospace; line-height:1.8;'>" + "".join(parts) + "</div>", unsafe_allow_html=True
    )
else:
    st.info("No timeline data for colored output.")

# ------------------------------------------------------------------
# Attention heatmaps
# ------------------------------------------------------------------
st.divider()
st.subheader("Attention Heatmaps")
st.caption(
    "Per layer and step: **Entropy mean** = how spread out attention is (very low = collapse). "
    "**Concentration max** = max weight on one position (near 1 = one token got all attention). "
    "**Collapsed/Focused head count** = number of heads in that layer that look collapsed or very focused."
)
attn_metric = st.selectbox(
    "Metric",
    [
        "attention_summary.entropy_mean",
        "attention_summary.concentration_max",
        "attention_summary.collapsed_head_count",
        "attention_summary.focused_head_count",
    ],
    format_func=lambda s: s.split(".")[-1].replace("_", " ").title(),
)

matrix, n_layers, n_steps = build_layer_step_matrix(report_data, attn_metric)
if matrix and n_layers > 0 and n_steps > 0:
    if HAS_PLOTLY:
        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=[f"Step {i}" for i in range(n_steps)],
                y=[f"Layer {i}" for i in range(n_layers)],
                colorscale="Viridis",
                hovertemplate="Step %{x}<br>%{y}<br>Value: %{z:.4f}<extra></extra>",
            )
        )
        fig.update_layout(
            xaxis_title="Generation Step",
            yaxis_title="Layer",
            height=max(300, n_layers * 30),
            margin=dict(t=30, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Install plotly for heatmap visualization: `pip install plotly`")
        st.json(matrix)
else:
    st.info("No attention data available.")

# ------------------------------------------------------------------
# Hidden state L2 norms heatmap
# ------------------------------------------------------------------
st.divider()
st.subheader("Hidden State L2 Norms")
st.caption(
    "Size of the vectors between layers. Stable values = healthy. "
    "Very high values can mean activations blowing up; very low can mean activations dying out. "
    "Look for sudden jumps or odd patterns by layer or step."
)
l2_matrix, l2_layers, l2_steps = build_layer_step_matrix(report_data, "hidden_summary.l2_norm_mean")
if l2_matrix and l2_layers > 0 and l2_steps > 0:
    if HAS_PLOTLY:
        fig = go.Figure(
            data=go.Heatmap(
                z=l2_matrix,
                x=[f"Step {i}" for i in range(l2_steps)],
                y=[f"Layer {i}" for i in range(l2_layers)],
                colorscale="Hot",
                hovertemplate="Step %{x}<br>%{y}<br>L2 Norm: %{z:.2f}<extra></extra>",
            )
        )
        fig.update_layout(
            xaxis_title="Generation Step",
            yaxis_title="Layer",
            height=max(300, l2_layers * 30),
            margin=dict(t=30, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Install plotly for heatmap visualization.")
else:
    st.info("No hidden state L2 norm data available.")

# ------------------------------------------------------------------
# Prompt Analysis (Phase-1b)
# ------------------------------------------------------------------
prompt_analysis = report_data.get("prompt_analysis")
if prompt_analysis:
    st.divider()
    st.subheader("Prompt Analysis")
    st.caption(
        "From an extra pass over your prompt. **Layer transformations:** how much each layer changes the representation (healthy often 0.2–0.5). "
        "**Prompt surprisals:** how surprised the model was by each prompt token (high = unusual or hard). "
        "**Basin score:** attention on middle vs ends of prompt; low (&lt;0.3) can mean 'lost in the middle'."
    )
    pa_tab1, pa_tab2, pa_tab3, pa_tab4 = st.tabs(
        ["Layer Transformations", "Prompt Surprisals", "Sparse Attention", "Attention Explorer"]
    )

    with pa_tab1:
        transforms = prompt_analysis.get("layer_transformations", [])
        if transforms:
            if HAS_PLOTLY:
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=[f"L{i}→L{i + 1}" for i in range(len(transforms))],
                        y=transforms,
                        marker_color="steelblue",
                        hovertemplate="%{x}<br>Cosine Sim: %{y:.4f}<extra></extra>",
                    )
                )
                fig.update_layout(
                    xaxis_title="Layer Transition",
                    yaxis_title="Cosine Similarity",
                    yaxis_range=[0, 1],
                    height=300,
                    margin=dict(t=20, b=40),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(transforms)
        else:
            st.info("No layer transformation data.")

    with pa_tab2:
        surprisals_pa = prompt_analysis.get("prompt_surprisals", [])
        if surprisals_pa:
            # Get prompt token texts if available
            prompt_tokens = report_data.get("prompt", {}).get("token_ids", [])
            labels = [f"Token {i}" for i in range(len(surprisals_pa))]
            if HAS_PLOTLY:
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=labels,
                        y=surprisals_pa,
                        marker_color="coral",
                        hovertemplate="%{x}<br>Surprisal: %{y:.3f}<extra></extra>",
                    )
                )
                fig.update_layout(
                    xaxis_title="Prompt Token Position",
                    yaxis_title="Surprisal (-log₂ p)",
                    height=300,
                    margin=dict(t=20, b=40),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(surprisals_pa)
        else:
            st.info("No prompt surprisal data (empty for Seq2Seq models).")

    with pa_tab3:
        layers_data = prompt_analysis.get("layers", [])
        if layers_data:
            st.write(f"**{len(layers_data)} layers** with sparse attention data")
            # Basin score heatmap: layers x heads (#11)
            if HAS_PLOTLY:
                num_heads = len(layers_data[0].get("basin_scores", []))
                if num_heads > 0:
                    z = []
                    for ly in layers_data:
                        basins = ly.get("basin_scores", [])
                        z.append([b for b in basins] + [None] * (num_heads - len(basins)))
                    z_arr = np.array(z, dtype=float)
                    fig_heat = go.Figure(
                        data=go.Heatmap(
                            z=z_arr,
                            x=[f"H{i}" for i in range(z_arr.shape[1])],
                            y=[f"L{i}" for i in range(z_arr.shape[0])],
                            colorscale=[
                                [0.0, "#e74c3c"],
                                [0.15, "#e74c3c"],
                                [0.25, "#f1c40f"],
                                [0.5, "#27ae60"],
                                [0.75, "#3498db"],
                                [1.0, "#3498db"],
                            ],
                            zmin=0,
                            zmax=2,
                            hovertemplate="Layer %{y} Head %{x}<br>Basin: %{z:.3f}<extra></extra>",
                        )
                    )
                    fig_heat.update_layout(
                        xaxis_title="Head",
                        yaxis_title="Layer",
                        title="Basin score (red &lt;0.3, green ~1, blue &gt;1.5)",
                        height=min(400, 80 + 20 * len(layers_data)),
                        margin=dict(t=40, b=40),
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)
                st.write("**Detail:** select a layer to see per-head bar chart.")
            selected_layer = st.slider("Layer", 0, len(layers_data) - 1, 0, key="pa_layer")
            layer = layers_data[selected_layer]
            basins = layer.get("basin_scores", [])
            if basins:
                st.write(f"**Basin scores** (layer {selected_layer}) — {len(basins)} heads")
                if HAS_PLOTLY:
                    fig = go.Figure()
                    fig.add_trace(
                        go.Bar(
                            x=[f"Head {i}" for i in range(len(basins))],
                            y=basins,
                            marker_color="mediumseagreen",
                            hovertemplate="Head %{x}<br>Basin Score: %{y:.3f}<extra></extra>",
                        )
                    )
                    fig.update_layout(
                        xaxis_title="Attention Head",
                        yaxis_title="Basin Score",
                        height=250,
                        margin=dict(t=20, b=40),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.bar_chart(basins)
            heads = layer.get("heads", [])
            st.write(
                f"**Sparse attention heads:** {len(heads)} heads, "
                f"total connections: {sum(len(h.get('weights', [])) for h in heads)}"
            )
        else:
            st.info("No sparse attention data.")

    with pa_tab4:
        # Attention Explorer (#10): query sparse attention via helpers
        layers_data = prompt_analysis.get("layers", [])
        if not layers_data:
            st.info("No sparse attention data to query.")
        else:
            num_heads = len(layers_data[0].get("basin_scores", [])) or len(layers_data[0].get("heads", []))
            layer_idx = st.selectbox(
                "Layer", range(len(layers_data)), format_func=lambda i: f"Layer {i}", key="qe_layer"
            )
            head_idx = st.selectbox("Head", range(max(1, num_heads)), format_func=lambda i: f"Head {i}", key="qe_head")
            token_idx = st.number_input("Token index (key or query)", min_value=0, value=0, step=1, key="qe_token")
            layer = layers_data[layer_idx]
            to_token = get_attention_to_token(layer, head_idx, token_idx)
            from_token = get_attention_from_token(layer, head_idx, token_idx)
            top_conn = get_top_connections(layer, head_idx, n=10)
            st.markdown("**Queries attending to this key (token index {}):**".format(token_idx))
            if to_token:
                st.write([f"q{i}: {w:.3f}" for i, w in to_token[:20]])
            else:
                st.write("None")
            st.markdown("**This query (token index {}) attends to:**".format(token_idx))
            if from_token:
                st.write([f"k{i}: {w:.3f}" for i, w in from_token[:20]])
            else:
                st.write("None")
            st.markdown("**Top-10 connections (this head):**")
            if top_conn:
                st.write([f"q{q}→k{k}: {w:.3f}" for q, k, w in top_conn])
            else:
                st.write("None")
            anomalies = get_basin_anomalies(layers_data, threshold=0.3)
            if anomalies:
                st.markdown("**Basin anomalies (score &lt; 0.3):**")
                st.write([f"L{li} H{hi}: {s:.3f}" for li, hi, s in anomalies[:15]])

# ------------------------------------------------------------------
# Performance breakdown (if available)
# ------------------------------------------------------------------
perf_data = report_data.get("extensions", {}).get("performance")
if perf_data:
    st.divider()
    st.subheader("Performance")
    st.caption(
        "Where time was spent: model load, tokenization, generation, and report building. Use this to see what dominates (e.g. load vs actual generation)."
    )
    # Total time (support both total_wall_time_ms and legacy total_ms)
    total_ms = perf_data.get("total_wall_time_ms") or perf_data.get("total_ms")
    unaccounted_raw = perf_data.get("unaccounted_time")
    unaccounted = unaccounted_raw.get("ms") if isinstance(unaccounted_raw, dict) else unaccounted_raw
    st.metric("Total wall time", f"{total_ms:.1f}ms" if isinstance(total_ms, (int, float)) else str(total_ms))
    if unaccounted is not None and isinstance(unaccounted, (int, float)):
        st.caption(f"Unaccounted: {unaccounted:.1f}ms")

    # parent_operations can be list of {name, ms, pct} or dict
    ops_raw = perf_data.get("parent_operations")
    if ops_raw is not None:
        if isinstance(ops_raw, list):
            names = [op.get("name", "?") for op in ops_raw]
            times = [op.get("ms", 0) for op in ops_raw]
            pcts = [op.get("pct") for op in ops_raw]
        else:
            names = list(ops_raw.keys())
            times = [ops_raw[k].get("ms", 0) if isinstance(ops_raw[k], dict) else 0 for k in names]
            pcts = [ops_raw[k].get("pct") if isinstance(ops_raw[k], dict) else None for k in names]

        if names and any(t is not None and t > 0 for t in times):
            if HAS_PLOTLY:
                tab_bar, tab_pie = st.tabs(["By operation (bar)", "Time distribution (pie)"])
                with tab_bar:
                    fig_bar = go.Figure()
                    fig_bar.add_trace(
                        go.Bar(
                            x=names,
                            y=times,
                            marker_color="mediumpurple",
                            text=[
                                f"{t:.1f}ms" + (f" ({p:.1f}%)" if p is not None else "")
                                for t, p in zip(times, pcts or [None] * len(times), strict=True)
                            ],
                            textposition="outside",
                            hovertemplate="%{x}<br>%{y:.1f}ms<extra></extra>",
                        )
                    )
                    fig_bar.update_layout(
                        xaxis_title="Operation",
                        yaxis_title="Time (ms)",
                        height=340,
                        margin=dict(t=20, b=80),
                        xaxis_tickangle=-45,
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                with tab_pie:
                    fig_pie = go.Figure(
                        data=[
                            go.Pie(
                                labels=names,
                                values=times,
                                hole=0.4,
                                hovertemplate="%{label}<br>%{value:.1f}ms (%{percent})<extra></extra>",
                            )
                        ]
                    )
                    fig_pie.update_layout(height=340, margin=dict(t=20, b=20))
                    st.plotly_chart(fig_pie, use_container_width=True)
            else:
                for i, name in enumerate(names):
                    pct_str = f" ({pcts[i]:.1f}%)" if pcts and pcts[i] is not None else ""
                    st.write(f"- **{name}:** {times[i]:.1f}ms{pct_str}")

    # Detailed breakdown from file (nested children)
    detailed_file = perf_data.get("detailed_file")
    if detailed_file and Path(detailed_file).exists():
        with open(detailed_file) as f:
            detailed = json.load(f)
        breakdown = detailed.get("breakdown", {})
        if breakdown and HAS_PLOTLY:
            st.markdown("**Detailed breakdown (top-level)**")
            top_names = list(breakdown.keys())
            top_times = [breakdown[k].get("ms", 0) if isinstance(breakdown[k], dict) else 0 for k in top_names]
            fig_det = go.Figure()
            fig_det.add_trace(
                go.Bar(
                    x=top_names,
                    y=top_times,
                    marker_color="teal",
                    text=[f"{t:.1f}ms" for t in top_times],
                    textposition="outside",
                    hovertemplate="%{x}<br>%{y:.1f}ms<extra></extra>",
                )
            )
            fig_det.update_layout(
                xaxis_title="Operation",
                yaxis_title="Time (ms)",
                height=300,
                margin=dict(t=20, b=80),
                xaxis_tickangle=-45,
            )
            st.plotly_chart(fig_det, use_container_width=True)

        with st.expander("Detailed breakdown (raw JSON)", expanded=False):
            st.json(breakdown)
    elif perf_data.get("detailed_file"):
        st.info("Detailed file path set but file not found (run from project root or adjust path).")

# ------------------------------------------------------------------
# Raw JSON
# ------------------------------------------------------------------
st.divider()
with st.expander("Raw JSON", expanded=False):
    if report_size_bytes is not None:
        size_mb = report_size_bytes / (1024 * 1024)
        st.caption(f"Payload size: {size_mb:.2f} MB")
        if size_mb > LARGE_FILE_MB:
            st.caption("⚠️ Large report — expanding raw JSON may be slow. Prefer Database source for future runs.")
    pretty_print = st.checkbox("Pretty-print JSON (formatted with indentation)", value=False)
    if pretty_print:
        formatted_json = json.dumps(report_data, indent=2, ensure_ascii=False)
        st.code(formatted_json, language="json")
    else:
        compact_json = json.dumps(report_data, separators=(",", ":"), ensure_ascii=False)
        st.code(compact_json, language="json")
