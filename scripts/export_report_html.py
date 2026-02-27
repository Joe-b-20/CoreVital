#!/usr/bin/env python3
# =============================================================================
# Export a single CoreVital report to a self-contained HTML file that looks
# like the Streamlit dashboard (header, health badges, narrative, entropy/
# perplexity/surprisal charts, attention heatmap, prompt & output).
# Share by sending the file or hosting it — one link, no server.
#
# Usage (activate conda env first if you use one, e.g. conda activate llm_hm):
#   python scripts/export_report_html.py --db docs/demo/corevital_demo.db --trace_id f28f1386
#   python scripts/export_report_html.py --json path/to/report.json
#   python scripts/export_report_html.py --db ... --trace_id ... -o run.html
# =============================================================================
# ruff: noqa: E501  (HTML/JS template strings exceed line length)

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_report(path: str) -> dict | None:
    """Load a report from a JSON file (no CoreVital/torch dependency)."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        text = p.read_text(encoding="utf-8")
        return json.loads(text)
    except (json.JSONDecodeError, OSError):
        return None


def load_report_from_db(db_path: str, trace_id: str) -> dict | None:
    """Load one report from SQLite without importing CoreVital (uses sqlite3 + gzip)."""
    import gzip
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT report_json, report_blob FROM reports WHERE trace_id = ? LIMIT 1",
            (trace_id,),
        ).fetchone()
        if not row:
            row = conn.execute(
                "SELECT report_json, report_blob FROM reports WHERE trace_id LIKE ? LIMIT 1",
                (trace_id + "%",),
            ).fetchone()
    if not row:
        return None
    json_str, blob = row[0], row[1]
    if blob:
        json_str = gzip.decompress(blob).decode("utf-8")
    elif not json_str:
        return None
    return json.loads(json_str)


def extract_timeline_series(report: dict, metric: str) -> tuple[list[int], list[float | None], list[str]]:
    """Extract (step_indices, values, token_texts) from report timeline. Metric: 'entropy'|'perplexity'|'surprisal'."""
    timeline = report.get("timeline") or []
    steps: list[int] = []
    values: list[float | None] = []
    tokens: list[str] = []
    for step in timeline:
        steps.append(step.get("step_index", len(steps)))
        tok = step.get("token") or {}
        tokens.append(tok.get("token_text") or "")
        logits = step.get("logits_summary") or {}
        if metric == "entropy":
            v = logits.get("entropy")
        elif metric == "perplexity":
            v = logits.get("perplexity")
        elif metric == "surprisal":
            v = logits.get("surprisal")
        else:
            v = logits.get(metric)
        values.append(float(v) if v is not None else None)
    return steps, values, tokens


def _get_nested(obj: dict, path: str):
    """Get nested key e.g. 'attention_summary.entropy_mean' from a dict."""
    for key in path.split("."):
        obj = (obj or {}).get(key)
    return obj


def build_layer_step_matrix(report: dict, metric_path: str) -> tuple[list[list[float | None]], int, int]:
    """Build layer x step matrix. metric_path e.g. 'attention_summary.entropy_mean'."""
    timeline = report.get("timeline") or []
    if not timeline:
        return [], 0, 0
    n_steps = len(timeline)
    n_layers = 0
    for step in timeline:
        layers = step.get("layers") or []
        n_layers = max(n_layers, len(layers))
    matrix: list[list[float | None]] = [[None] * n_steps for _ in range(n_layers)]
    for step_idx, step in enumerate(timeline):
        for layer_idx, layer in enumerate(step.get("layers") or []):
            v = _get_nested(layer, metric_path)
            if v is not None:
                try:
                    matrix[layer_idx][step_idx] = float(v)
                except (TypeError, ValueError):
                    pass
    return matrix, n_layers, n_steps


def _health_badge(label: str, value, *, good_when_false: bool = True) -> tuple[str, str]:
    """Return (display_value, css_class) for a health flag."""
    if isinstance(value, bool):
        is_good = (not value) if good_when_false else value
        css = "badge-ok" if is_good else "badge-fail"
        display = "Yes" if value else "No"
    elif isinstance(value, int):
        is_good = value == 0
        css = "badge-ok" if is_good else "badge-warn"
        display = str(value)
    else:
        css = "badge-unknown"
        display = str(value) if value is not None else "—"
    return display, css


def build_report_data(report: dict) -> dict:
    """Extract everything needed for the HTML template (charts, text, badges)."""
    model = report.get("model") or {}
    summary = report.get("summary") or {}
    health = report.get("health_flags") or {}
    risk_data = (report.get("extensions") or {}).get("risk") or {}
    narrative_data = (report.get("extensions") or {}).get("narrative") or {}
    prompt_info = report.get("prompt") or {}
    generated = report.get("generated") or {}

    steps, entropies, tokens = extract_timeline_series(report, "entropy")
    _, perplexities, _ = extract_timeline_series(report, "perplexity")
    _, surprisals, _ = extract_timeline_series(report, "surprisal")

    # Clean None for JSON / Plotly
    entropies_plot = [float(x) if x is not None else None for x in entropies]
    perplexities_plot = [float(x) if x is not None else None for x in perplexities]
    surprisals_plot = [float(x) if x is not None else None for x in surprisals]

    attn_matrix, n_layers, n_steps = build_layer_step_matrix(report, "attention_summary.entropy_mean")
    # Flatten for Plotly heatmap: z[row][col] = layer (y) x step (x)
    heatmap_z = attn_matrix
    heatmap_x = list(range(n_steps))
    heatmap_y = list(range(n_layers))

    health_badges = [
        {"label": "NaN Detected", "value": health.get("nan_detected"), "good_when_false": True},
        {"label": "Inf Detected", "value": health.get("inf_detected"), "good_when_false": True},
        {"label": "Attention Collapse", "value": health.get("attention_collapse_detected"), "good_when_false": True},
        {"label": "High Entropy Steps", "value": health.get("high_entropy_steps"), "good_when_false": True},
        {"label": "Repetition Loop", "value": health.get("repetition_loop_detected"), "good_when_false": True},
        {"label": "Mid-Layer Anomaly", "value": health.get("mid_layer_anomaly_detected"), "good_when_false": True},
    ]
    badges_rendered = []
    for b in health_badges:
        display_val, css = _health_badge(b["label"], b["value"], good_when_false=b.get("good_when_false", True))
        badges_rendered.append({"label": b["label"], "display": display_val, "css": css})

    narrative_summary = (narrative_data.get("summary") or "").strip()
    risk_score = risk_data.get("risk_score")
    risk_factors = risk_data.get("risk_factors") or []
    blamed_layers = risk_data.get("blamed_layers") or []

    prompt_text = prompt_info.get("text") or ""
    generated_text = generated.get("output_text") or ""

    # Colored output: list of { "token": str, "entropy": float | null } for span coloring
    colored_tokens = []
    for tok, ent in zip(tokens, entropies, strict=True):
        colored_tokens.append({"token": tok, "entropy": float(ent) if ent is not None else None})

    return {
        "title": f"CoreVital — {model.get('hf_id', 'Unknown Model')}",
        "trace_id": report.get("trace_id", "?"),
        "architecture": model.get("architecture", "?"),
        "num_layers": model.get("num_layers", "?"),
        "generated_tokens": summary.get("generated_tokens", "?"),
        "elapsed_ms": summary.get("elapsed_ms", "?"),
        "risk_score": risk_score,
        "risk_factors": risk_factors,
        "blamed_layers": blamed_layers,
        "narrative_summary": narrative_summary,
        "health_badges": badges_rendered,
        "steps": steps,
        "entropies": entropies_plot,
        "perplexities": perplexities_plot,
        "surprisals": surprisals_plot,
        "tokens": tokens,
        "heatmap_z": heatmap_z,
        "heatmap_x": heatmap_x,
        "heatmap_y": heatmap_y,
        "prompt_text": prompt_text,
        "generated_text": generated_text,
        "colored_tokens": colored_tokens,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Export CoreVital report to Streamlit-like HTML")
    parser.add_argument("--db", type=str, help="Path to SQLite DB (use with --trace_id)")
    parser.add_argument("--trace_id", type=str, help="Trace ID (use with --db)")
    parser.add_argument("--json", type=str, help="Path to report JSON file (instead of --db/--trace_id)")
    parser.add_argument("-o", "--output", type=str, help="Output HTML path (default: report_<trace>.html)")
    args = parser.parse_args()

    report = None
    trace_id = "report"
    if args.json:
        path = Path(args.json)
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            sys.exit(1)
        report = load_report(str(path))
        if report:
            trace_id = (report.get("trace_id") or "report")[:8]
    elif args.db and args.trace_id:
        report = load_report_from_db(args.db, args.trace_id)
        if report:
            trace_id = (report.get("trace_id") or args.trace_id)[:8]
    else:
        parser.error("Use either --json <path> or --db <path> and --trace_id <id>")

    if not report:
        print("Could not load report.", file=sys.stderr)
        sys.exit(1)

    data = build_report_data(report)
    out_path = Path(args.output) if args.output else REPO_ROOT / "report_{}.html".format(trace_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    html = render_html(data)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")


def render_html(data: dict) -> str:
    """Produce a single HTML string with embedded data and Plotly.js."""
    # Embed JSON safely: avoid </script> in the string
    json_str = json.dumps(data, ensure_ascii=False)
    json_str = json_str.replace("</script>", "<\\/script>")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{_esc(data["title"])}</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    :root {{ --bg: #0e1117; --card: #1a1d24; --text: #fafafa; --muted: #9ca3af; --ok: #27ae60; --warn: #f1c40f; --fail: #e74c3c; }}
    * {{ box-sizing: border-box; }}
    body {{ font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 1.5rem; line-height: 1.5; }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    h1 {{ margin: 0 0 1rem; font-size: 1.5rem; }}
    .header-cols {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem; }}
    .metric {{ background: var(--card); padding: 0.75rem 1rem; border-radius: 8px; }}
    .metric label {{ display: block; color: var(--muted); font-size: 0.8rem; }}
    .metric .value {{ font-size: 1.1rem; font-weight: 600; }}
    .section {{ margin-bottom: 1.5rem; }}
    .section h2 {{ font-size: 1.1rem; margin: 0 0 0.5rem; color: var(--muted); }}
    .narrative {{ background: #1e3a5f; padding: 1rem; border-radius: 8px; border-left: 4px solid #3b82f6; }}
    .badges {{ display: flex; flex-wrap: wrap; gap: 0.5rem; }}
    .badge {{ display: inline-flex; align-items: center; gap: 0.35rem; padding: 0.25rem 0.6rem; border-radius: 6px; font-size: 0.85rem; }}
    .badge-ok {{ background: rgba(39,174,96,0.25); color: var(--ok); }}
    .badge-fail {{ background: rgba(231,76,60,0.25); color: var(--fail); }}
    .badge-warn {{ background: rgba(241,196,15,0.2); color: var(--warn); }}
    .badge-unknown {{ background: var(--card); color: var(--muted); }}
    .risk {{ background: var(--card); padding: 0.75rem 1rem; border-radius: 8px; }}
    .chart-container {{ background: var(--card); padding: 1rem; border-radius: 8px; margin-bottom: 1rem; }}
    .chart-container h3 {{ margin: 0 0 0.5rem; font-size: 1rem; }}
    .prompt-output {{ background: var(--card); padding: 1rem; border-radius: 8px; white-space: pre-wrap; max-height: 400px; overflow-y: auto; font-family: ui-monospace, monospace; font-size: 0.9rem; }}
    .colored-output {{ font-family: ui-monospace, monospace; line-height: 1.8; word-break: break-all; }}
    .colored-output span {{ padding: 0 1px; }}
    details {{ margin-bottom: 0.5rem; }}
    details summary {{ cursor: pointer; color: var(--muted); font-size: 0.9rem; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>{_esc(data["title"])}</h1>
    <div class="header-cols">
      <div class="metric"><label>Architecture</label><span class="value">{_esc(str(data["architecture"]))}</span></div>
      <div class="metric"><label>Layers</label><span class="value">{_esc(str(data["num_layers"]))}</span></div>
      <div class="metric"><label>Generated Tokens</label><span class="value">{_esc(str(data["generated_tokens"]))}</span></div>
      <div class="metric"><label>Elapsed</label><span class="value">{data["elapsed_ms"]} ms</span></div>
    </div>

    <div class="section">
      <details><summary>Run details</summary>
        <p>Trace ID: <code>{_esc(data["trace_id"])}</code></p>
      </details>
    </div>

    {f'<div class="section narrative"><strong>Summary:</strong> {_esc(data["narrative_summary"])}</div>' if data.get("narrative_summary") else ""}

    <div class="section">
      <h2>Health Flags</h2>
      <div class="badges">
        {"".join(f'<span class="badge {b["css"]}">{_esc(b["label"])}: {_esc(b["display"])}</span>' for b in data["health_badges"])}
      </div>
    </div>

    {(f'<div class="section risk"><h2>Risk Score</h2><p><strong>{data["risk_score"]:.2f}</strong></p>' + (f"<p>Factors: {', '.join(data['risk_factors'])}</p>" if data["risk_factors"] else "") + (f"<p>Blamed layers: {data['blamed_layers']}</p>" if data["blamed_layers"] else "") + "</div>") if data.get("risk_score") is not None else ""}

    <div class="section">
      <details><summary>How to read these metrics</summary>
        <p>Entropy &lt;2 confident, 2–4 normal, &gt;4 confused (red line at 4). Perplexity = 2^entropy. Surprisal = how surprised the model was by each token. Health: ✅ OK, 🔴/⚠️ worth checking.</p>
      </details>
    </div>

    <div class="section">
      <h2>Logits Over Time</h2>
      <div class="chart-container"><h3>Entropy by step</h3><div id="entropy-chart"></div></div>
      <div class="chart-container"><h3>Perplexity by step</h3><div id="perplexity-chart"></div></div>
      <div class="chart-container"><h3>Surprisal by step</h3><div id="surprisal-chart"></div></div>
    </div>

    <div class="section">
      <h2>Colored Output (by uncertainty)</h2>
      <div class="colored-output" id="colored-output"></div>
    </div>

    <div class="section">
      <h2>Attention Heatmap (layer × step)</h2>
      <div class="chart-container"><div id="heatmap-chart"></div></div>
    </div>

    <div class="section">
      <h2>Prompt</h2>
      <div class="prompt-output">{_esc(data["prompt_text"])}</div>
    </div>
    <div class="section">
      <h2>Generated</h2>
      <div class="prompt-output">{_esc(data["generated_text"])}</div>
    </div>
  </div>

  <script type="application/json" id="report-data">{json_str}</script>
  <script>
    (function() {{
      var el = document.getElementById('report-data');
      var data = JSON.parse(el.textContent);

      // Entropy
      var entropyTrace = {{ x: data.steps, y: data.entropies, type: 'scatter', mode: 'lines+markers', name: 'Entropy', hovertemplate: 'Step %{{x}}<br>Token: %{{text}}<br>Entropy: %{{y:.3f}}<extra></extra>', text: data.tokens }};
      var entropyLayout = {{ xaxis: {{ title: 'Generation Step' }}, yaxis: {{ title: 'Entropy (bits)' }}, height: 350, margin: {{ t: 30, b: 40 }}, shapes: [{{ type: 'line', y0: 4, y1: 4, x0: 0, x1: data.steps.length, line: {{ dash: 'dash', color: 'red' }}, annotation: {{ text: 'High (4.0)' }} }}] }};
      Plotly.newPlot('entropy-chart', [entropyTrace], entropyLayout, {{ responsive: true }});

      // Perplexity
      var perpTrace = {{ x: data.steps, y: data.perplexities, type: 'scatter', mode: 'lines+markers', name: 'Perplexity', text: data.tokens, hovertemplate: 'Step %{{x}}<br>Perplexity: %{{y:.2f}}<extra></extra>' }};
      var perpLayout = {{ xaxis: {{ title: 'Generation Step' }}, yaxis: {{ title: 'Perplexity' }}, height: 350, margin: {{ t: 30, b: 40 }} }};
      Plotly.newPlot('perplexity-chart', [perpTrace], perpLayout, {{ responsive: true }});

      // Surprisal
      var surpTrace = {{ x: data.steps, y: data.surprisals, type: 'bar', name: 'Surprisal', text: data.tokens, hovertemplate: 'Step %{{x}}<br>Surprisal: %{{y:.3f}}<extra></extra>' }};
      var surpLayout = {{ xaxis: {{ title: 'Generation Step' }}, yaxis: {{ title: 'Surprisal' }}, height: 350, margin: {{ t: 30, b: 40 }} }};
      Plotly.newPlot('surprisal-chart', [surpTrace], surpLayout, {{ responsive: true }});

      // Heatmap
      if (data.heatmap_z && data.heatmap_z.length > 0) {{
        var heatTrace = {{ z: data.heatmap_z, x: data.heatmap_x, y: data.heatmap_y, type: 'heatmap', colorscale: 'Viridis', hovertemplate: 'Step %{{x}}<br>Layer %{{y}}<br>Value: %{{z:.4f}}<extra></extra>' }};
        var heatLayout = {{ xaxis: {{ title: 'Step' }}, yaxis: {{ title: 'Layer' }}, height: Math.max(300, data.heatmap_y.length * 25), margin: {{ t: 30, b: 40 }} }};
        Plotly.newPlot('heatmap-chart', [heatTrace], heatLayout, {{ responsive: true }});
      }}

      // Colored output
      var out = document.getElementById('colored-output');
      var spans = [];
      for (var i = 0; i < data.colored_tokens.length; i++) {{
        var t = data.colored_tokens[i];
        var ent = t.entropy;
        var color = ent === null ? 'inherit' : (ent > 4 ? '#e74c3c' : (ent >= 2 ? '#f1c40f' : '#27ae60'));
        var esc = t.token.replace(/\\\\/g, '\\\\\\\\').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/&/g, '&amp;');
        spans.push('<span style="background-color:' + color + ';color:black;">' + esc + '</span>');
      }}
      out.innerHTML = spans.join('');
    }})();
  </script>
</body>
</html>
"""


def _esc(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


if __name__ == "__main__":
    main()
