#!/usr/bin/env python3
"""
Analyze outputs produced by scripts/run_eval_suite.py.

Produces:
- per-run graded table
- model-level accuracy and CoreVital metric summaries
- threshold metrics for risk_score predicting bad outputs on gradable prompts
- basic signal validation for repetition_loop_detected using output repetition heuristic
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from eval_suite_common import EVAL_CASES, check_format_expectation, detect_output_repetition, grade_output

try:
    from CoreVital.config import load_model_profile
except Exception:  # pragma: no cover - optional for analyzer portability
    load_model_profile = None  # type: ignore[assignment]


CASE_BY_ID = {c.case_id: c for c in EVAL_CASES}


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _timeline_stats(report: dict[str, Any]) -> dict[str, Any]:
    ents: list[float] = []
    surps: list[float] = []
    for step in report.get("timeline", []):
        ls = step.get("logits_summary") or {}
        e = ls.get("entropy")
        s = ls.get("surprisal")
        if isinstance(e, (int, float)):
            ents.append(float(e))
        if isinstance(s, (int, float)):
            surps.append(float(s))
    return {
        "entropy_max": max(ents) if ents else None,
        "entropy_avg": (sum(ents) / len(ents)) if ents else None,
        "surprisal_max": max(surps) if surps else None,
        "surprisal_avg": (sum(surps) / len(surps)) if surps else None,
    }


def _profile_entropy_threshold(report: dict[str, Any]) -> float:
    if load_model_profile is None:
        return 4.0
    try:
        architecture = ((report.get("model") or {}).get("architecture")) or ""
        profile = load_model_profile(architecture)
        return float(getattr(profile, "high_entropy_threshold_bits", 4.0))
    except Exception:
        return 4.0


def _recompute_high_entropy_steps(report: dict[str, Any]) -> int:
    threshold = _profile_entropy_threshold(report)
    count = 0
    for step in report.get("timeline", []):
        ls = step.get("logits_summary") or {}
        ent = ls.get("entropy")
        if isinstance(ent, (int, float)) and float(ent) > threshold:
            count += 1
    return count


def _recompute_risk_score(report: dict[str, Any]) -> Optional[float]:
    hf = report.get("health_flags") or {}
    summary = report.get("summary") or {}

    nan_detected = bool(hf.get("nan_detected"))
    inf_detected = bool(hf.get("inf_detected"))
    if nan_detected or inf_detected:
        return 1.0

    score = 0.0
    if bool(hf.get("repetition_loop_detected")):
        score = max(score, 0.9)
    if bool(hf.get("mid_layer_anomaly_detected")):
        score = max(score, 0.7)
    if bool(hf.get("attention_collapse_detected")):
        score = max(score, 0.3)

    total_steps = int(summary.get("total_steps") or 0)
    total_steps = max(1, total_steps)
    high_entropy_steps = int(hf.get("high_entropy_steps") or 0)
    entropy_component = min(1.0, high_entropy_steps / total_steps) * 0.5
    return min(1.0, score + entropy_component)


def _approx_equal(a: Optional[float], b: Optional[float], tol: float = 1e-9) -> Optional[bool]:
    if a is None or b is None:
        return None
    return abs(float(a) - float(b)) <= tol


def _confusion(rows: list[dict[str, Any]], pred_key: str, truth_key: str) -> dict[str, int]:
    tp = fp = tn = fn = 0
    for r in rows:
        truth = r.get(truth_key)
        pred = r.get(pred_key)
        if truth is None or pred is None:
            continue
        pred_b = bool(pred)
        truth_b = bool(truth)
        if pred_b and truth_b:
            tp += 1
        elif pred_b and not truth_b:
            fp += 1
        elif (not pred_b) and truth_b:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _prf(cm: dict[str, int]) -> dict[str, Optional[float]]:
    tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def _risk_threshold_eval(rows: list[dict[str, Any]], threshold: float, truth_key: str) -> dict[str, Any]:
    derived = []
    for r in rows:
        rr = dict(r)
        risk = rr.get("risk_score")
        rr["risk_pred"] = (risk is not None and float(risk) >= threshold)
        derived.append(rr)
    cm = _confusion(derived, "risk_pred", truth_key)
    return {"threshold": threshold, "confusion": cm, **_prf(cm)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze CoreVital eval-suite outputs and auto-grade results.")
    parser.add_argument("--run-dir", type=str, default="evaluation/runs/eval_suite", help="Eval suite output directory")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    run_dir = (project_root / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
    manifest_path = run_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    manifest_rows = _load_manifest(manifest_path)
    graded_rows: list[dict[str, Any]] = []

    for row in manifest_rows:
        out: dict[str, Any] = dict(row)
        case = CASE_BY_ID.get(row["case_id"])
        out["case_category"] = case.category if case else None
        out["case_tags"] = list(case.tags) if case else []
        out["gradable"] = None
        out["output_correct"] = None
        out["output_grade_reason"] = None
        out["output_repetition_detected"] = None
        out["output_repetition_pattern"] = None
        out["output_repetition_count"] = None
        out["format_checked"] = None
        out["format_compliant"] = None
        out["format_reason"] = None
        out["entropy_max"] = None
        out["entropy_avg"] = None
        out["surprisal_max"] = None
        out["surprisal_avg"] = None
        out["corevital_repetition_flag"] = None
        out["corevital_attention_collapse_flag"] = None
        out["corevital_high_entropy_steps"] = None
        out["corevital_high_entropy_steps_recomputed"] = None
        out["corevital_high_entropy_steps_match"] = None
        out["corevital_risk_score_recomputed"] = None
        out["corevital_risk_score_match"] = None
        out["bad_output_label"] = None
        out["quality_risk_proxy"] = None
        out["quality_risk_reasons"] = []

        trace_path = row.get("trace_path")
        if row.get("returncode") == 0 and trace_path and Path(trace_path).exists():
            report = _read_json(Path(trace_path))
            generated_text = (report.get("generated") or {}).get("output_text", "") or ""
            out["generated_text"] = generated_text

            if case is not None:
                g = grade_output(case, generated_text)
                out["gradable"] = g.get("gradable")
                out["output_correct"] = g.get("correct")
                out["output_grade_reason"] = g.get("reason")

            rep = detect_output_repetition(generated_text)
            out["output_repetition_detected"] = rep["repetition_detected"]
            out["output_repetition_pattern"] = rep["pattern"]
            out["output_repetition_count"] = rep["count"]

            if case is not None:
                fmt = check_format_expectation(case, generated_text)
                out["format_checked"] = fmt["checked"]
                out["format_compliant"] = fmt["compliant"]
                out["format_reason"] = fmt["reason"]

            stats = _timeline_stats(report)
            out.update(stats)

            hf = report.get("health_flags") or {}
            out["corevital_repetition_flag"] = bool(hf.get("repetition_loop_detected"))
            out["corevital_attention_collapse_flag"] = bool(hf.get("attention_collapse_detected"))
            out["corevital_high_entropy_steps"] = int(hf.get("high_entropy_steps") or 0)
            out["risk_score"] = ((report.get("extensions") or {}).get("risk") or {}).get("risk_score")
            out["failure_risk"] = ((report.get("extensions") or {}).get("early_warning") or {}).get("failure_risk")
            out["corevital_high_entropy_steps_recomputed"] = _recompute_high_entropy_steps(report)
            out["corevital_high_entropy_steps_match"] = (
                out["corevital_high_entropy_steps"] == out["corevital_high_entropy_steps_recomputed"]
            )
            out["corevital_risk_score_recomputed"] = _recompute_risk_score(report)
            out["corevital_risk_score_match"] = _approx_equal(
                out["risk_score"],
                out["corevital_risk_score_recomputed"],
            )

            # Output-quality label used for threshold evaluation on gradable prompts.
            if out["gradable"] is True and out["output_correct"] is not None:
                out["bad_output_label"] = not bool(out["output_correct"])

            # Output-only quality risk proxy (distinct from CoreVital internal-health signals)
            quality_reasons: list[str] = []
            if out["bad_output_label"] is True:
                quality_reasons.append("bad_output")
            if out.get("format_checked") and out.get("format_compliant") is False:
                quality_reasons.append("format_noncompliant")
            if out["output_repetition_detected"] is True:
                quality_reasons.append("output_repetition")
            if isinstance(out.get("generated_text"), str) and len((out["generated_text"] or "").strip()) == 0:
                quality_reasons.append("empty_output")
            out["quality_risk_reasons"] = quality_reasons
            out["quality_risk_proxy"] = bool(quality_reasons)

        graded_rows.append(out)

    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Write flat CSV for review and external analysis.
    csv_path = analysis_dir / "graded_runs.csv"
    fieldnames = sorted({k for r in graded_rows for k in r.keys()})
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in graded_rows:
            row = dict(r)
            for k, v in list(row.items()):
                if isinstance(v, (list, dict)):
                    row[k] = json.dumps(v, ensure_ascii=False)
            w.writerow(row)

    successful = [r for r in graded_rows if r.get("returncode") == 0]
    gradable = [r for r in successful if r.get("gradable") is True and r.get("output_correct") is not None]
    probe_repetition = [r for r in successful if r.get("case_id") == "repetition_probe"]

    # Model-level rollups
    per_model: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in successful:
        grouped[r["model_id"]].append(r)

    for model_id, rows in grouped.items():
        grad_rows = [r for r in rows if r in gradable]
        correct_n = sum(1 for r in grad_rows if r.get("output_correct") is True)
        entropy_vals = [r["entropy_max"] for r in rows if isinstance(r.get("entropy_max"), (int, float))]
        surprisal_vals = [r["surprisal_avg"] for r in rows if isinstance(r.get("surprisal_avg"), (int, float))]
        risks = [r["risk_score"] for r in rows if isinstance(r.get("risk_score"), (int, float))]
        per_model[model_id] = {
            "runs": len(rows),
            "gradable_runs": len(grad_rows),
            "task_accuracy": (correct_n / len(grad_rows)) if grad_rows else None,
            "avg_risk_score": (sum(risks) / len(risks)) if risks else None,
            "avg_entropy_max": (sum(entropy_vals) / len(entropy_vals)) if entropy_vals else None,
            "avg_surprisal_avg": (sum(surprisal_vals) / len(surprisal_vals)) if surprisal_vals else None,
            "attention_collapse_rate": (
                sum(1 for r in rows if r.get("corevital_attention_collapse_flag")) / len(rows)
            )
            if rows
            else None,
            "high_entropy_run_rate": (
                sum(1 for r in rows if (r.get("corevital_high_entropy_steps") or 0) > 0) / len(rows)
            )
            if rows
            else None,
            "metric_consistency": {
                "high_entropy_steps_match_rate": _rate(
                    [r.get("corevital_high_entropy_steps_match") for r in rows]
                ),
                "risk_score_match_rate": _rate([r.get("corevital_risk_score_match") for r in rows]),
            },
        }

    # Construct-validity oriented summaries
    tag_signal_expectations = _tag_signal_expectations(successful)
    seed_stability = _seed_stability_summary(successful)
    attention_collapse_diagnostics = _attention_collapse_diagnostics(successful)

    # Risk threshold evaluation vs bad outputs (gradable prompts only)
    threshold_metrics = []
    for t in (0.3, 0.5, 0.7):
        threshold_metrics.append(_risk_threshold_eval(gradable, t, "bad_output_label"))

    # Risk threshold evaluation vs output-only quality-risk proxy (all successful runs)
    quality_proxy_rows = [r for r in successful if r.get("quality_risk_proxy") is not None]
    quality_proxy_threshold_metrics = []
    for t in (0.3, 0.5, 0.7):
        quality_proxy_threshold_metrics.append(_risk_threshold_eval(quality_proxy_rows, t, "quality_risk_proxy"))

    # Repetition flag validation: CoreVital repetition_loop vs output repetition heuristic
    repetition_eval_rows = [
        r for r in successful if r.get("output_repetition_detected") is not None and r.get("corevital_repetition_flag") is not None
    ]
    repetition_cm = _confusion(repetition_eval_rows, "corevital_repetition_flag", "output_repetition_detected")
    repetition_eval = {"confusion": repetition_cm, **_prf(repetition_cm)}

    # High entropy signal usefulness on gradable tasks
    entropy_flag_rows = list(gradable)
    for r in entropy_flag_rows:
        r["high_entropy_pred"] = (r.get("corevital_high_entropy_steps") or 0) > 0
    high_entropy_cm = _confusion(entropy_flag_rows, "high_entropy_pred", "bad_output_label")
    high_entropy_eval = {"confusion": high_entropy_cm, **_prf(high_entropy_cm)}

    # CoreVital "health" signal alignment with output-quality proxy (to show mismatch explicitly)
    quality_proxy_entropy_rows = [r for r in quality_proxy_rows]
    for r in quality_proxy_entropy_rows:
        r["high_entropy_pred"] = (r.get("corevital_high_entropy_steps") or 0) > 0
    high_entropy_vs_quality_proxy_cm = _confusion(quality_proxy_entropy_rows, "high_entropy_pred", "quality_risk_proxy")
    high_entropy_vs_quality_proxy_eval = {
        "confusion": high_entropy_vs_quality_proxy_cm,
        **_prf(high_entropy_vs_quality_proxy_cm),
    }

    # Summary object
    summary = {
        "run_dir": str(run_dir),
        "counts": {
            "manifest_rows": len(manifest_rows),
            "successful_runs": len(successful),
            "failed_runs": sum(1 for r in graded_rows if r.get("returncode") != 0),
            "gradable_runs": len(gradable),
            "repetition_probe_runs": len(probe_repetition),
        },
        "per_model": per_model,
        "metric_internal_consistency": {
            "high_entropy_steps_match_rate": _rate([r.get("corevital_high_entropy_steps_match") for r in successful]),
            "risk_score_match_rate": _rate([r.get("corevital_risk_score_match") for r in successful]),
            "notes": [
                "high_entropy_steps is recomputed from timeline entropy using model-profile threshold when available",
                "risk_score is recomputed from health_flags + summary using current risk.py formula",
            ],
        },
        "metric_signal_prevalence": {
            "attention_collapse_rate": _rate([r.get("corevital_attention_collapse_flag") for r in successful]),
            "repetition_loop_rate": _rate([r.get("corevital_repetition_flag") for r in successful]),
            "high_entropy_run_rate": _rate(
                [((r.get("corevital_high_entropy_steps") or 0) > 0) for r in successful if r.get("corevital_high_entropy_steps") is not None]
            ),
        },
        "metric_usefulness_on_stability_baselines": _stability_baseline_eval(successful),
        "metric_construct_validity": {
            "tag_signal_expectations": tag_signal_expectations,
            "seed_stability": seed_stability,
            "attention_collapse_diagnostics": attention_collapse_diagnostics,
        },
        "risk_threshold_metrics_vs_bad_output": threshold_metrics,
        "risk_threshold_metrics_vs_quality_risk_proxy": quality_proxy_threshold_metrics,
        "repetition_flag_eval_vs_output_repetition_heuristic": repetition_eval,
        "high_entropy_flag_eval_vs_bad_output": high_entropy_eval,
        "high_entropy_flag_eval_vs_quality_risk_proxy": high_entropy_vs_quality_proxy_eval,
        "notes": [
            "bad_output_label is only computed for auto-gradable prompts",
            "quality_risk_proxy is output-only and includes bad output, format violations, and repetition",
            "repetition validation uses output text repetition heuristic, not hidden-state ground truth",
            "attention_collapse_detected is reported but not treated as standalone correctness target",
        ],
    }

    summary_path = analysis_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Human-readable markdown report
    md_lines = []
    md_lines.append("# CoreVital Eval Suite Report")
    md_lines.append("")
    md_lines.append(f"- Run dir: `{run_dir}`")
    md_lines.append(f"- Successful runs: `{summary['counts']['successful_runs']}`")
    md_lines.append(f"- Gradable runs: `{summary['counts']['gradable_runs']}`")
    md_lines.append("")
    md_lines.append("## Model Summary")
    md_lines.append("")
    md_lines.append("| Model | Runs | Gradable | Task Accuracy | Avg Risk | High-Entropy Run Rate |")
    md_lines.append("|---|---:|---:|---:|---:|---:|")
    for model_id, stats in per_model.items():
        md_lines.append(
            f"| `{model_id}` | {stats['runs']} | {stats['gradable_runs']} | "
            f"{_fmt(stats['task_accuracy'])} | {_fmt(stats['avg_risk_score'])} | {_fmt(stats['high_entropy_run_rate'])} |"
        )
    md_lines.append("")
    md_lines.append("## CoreVital Metric Internal Consistency")
    md_lines.append("")
    mic = summary["metric_internal_consistency"]
    md_lines.append(f"- `high_entropy_steps` recomputation match rate: `{_fmt(mic['high_entropy_steps_match_rate'])}`")
    md_lines.append(f"- `risk_score` recomputation match rate: `{_fmt(mic['risk_score_match_rate'])}`")
    md_lines.append("")
    md_lines.append("## CoreVital Signal Prevalence (All Runs)")
    md_lines.append("")
    msp = summary["metric_signal_prevalence"]
    md_lines.append(f"- `attention_collapse_detected` rate: `{_fmt(msp['attention_collapse_rate'])}`")
    md_lines.append(f"- `repetition_loop_detected` rate: `{_fmt(msp['repetition_loop_rate'])}`")
    md_lines.append(f"- `high_entropy_steps > 0` run rate: `{_fmt(msp['high_entropy_run_rate'])}`")
    md_lines.append("")
    md_lines.append("## Metric Usefulness on Stability Baselines")
    md_lines.append("")
    sbe = summary["metric_usefulness_on_stability_baselines"]
    md_lines.append(f"- Baseline runs (cases tagged `low_entropy_expected`): `{sbe['count']}`")
    md_lines.append(f"- `high_entropy_steps > 0` on baseline runs (lower is better): `{_fmt(sbe['high_entropy_run_rate'])}`")
    md_lines.append(f"- Avg risk on baseline runs (lower is better): `{_fmt(sbe['avg_risk_score'])}`")
    md_lines.append("")
    md_lines.append("## Metric Construct Validity Checks")
    md_lines.append("")
    md_lines.append("### Tag Signal Expectations")
    md_lines.append("")
    md_lines.append("| Tag | Runs | High-Entropy Run Rate | Avg Risk | Attention Collapse Rate | Repetition Rate |")
    md_lines.append("|---|---:|---:|---:|---:|---:|")
    for tag, item in tag_signal_expectations.items():
        md_lines.append(
            f"| `{tag}` | {item['count']} | {_fmt(item['high_entropy_run_rate'])} | {_fmt(item['avg_risk_score'])} | "
            f"{_fmt(item['attention_collapse_rate'])} | {_fmt(item['repetition_loop_rate'])} |"
        )
    md_lines.append("")
    md_lines.append("### Seed Stability (Per Metric Across Same Model+Case)")
    md_lines.append("")
    md_lines.append(
        f"- Groups with 2+ seeds: `{seed_stability['groups_with_2plus_seeds']}` "
        f"(total groups: `{seed_stability['total_groups']}`)"
    )
    md_lines.append(f"- Avg stddev `risk_score`: `{_fmt(seed_stability['avg_stddev_risk_score'])}`")
    md_lines.append(f"- Avg stddev `entropy_max`: `{_fmt(seed_stability['avg_stddev_entropy_max'])}`")
    md_lines.append(f"- Avg stddev `high_entropy_steps`: `{_fmt(seed_stability['avg_stddev_high_entropy_steps'])}`")
    md_lines.append("")
    md_lines.append("### Attention Collapse Diagnostics")
    md_lines.append("")
    acd = attention_collapse_diagnostics
    md_lines.append(f"- Overall collapse rate: `{_fmt(acd['overall_rate'])}`")
    md_lines.append(f"- Baseline-tag collapse rate (`low_entropy_expected`): `{_fmt(acd['baseline_tag_rate'])}`")
    md_lines.append(f"- Probe-tag collapse rate (`probe`): `{_fmt(acd['probe_tag_rate'])}`")
    md_lines.append(
        f"- Models with collapse rate >= 0.9: `{', '.join(acd['models_ge_0_9']) if acd['models_ge_0_9'] else '(none)'}`"
    )
    md_lines.append("")
    md_lines.append("## Risk Thresholds vs Bad Output (Auto-Graded Only)")
    md_lines.append("")
    md_lines.append("| Threshold | Precision | Recall | F1 | TP | FP | TN | FN |")
    md_lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for item in threshold_metrics:
        cm = item["confusion"]
        md_lines.append(
            f"| {item['threshold']:.1f} | {_fmt(item['precision'])} | {_fmt(item['recall'])} | {_fmt(item['f1'])} | "
            f"{cm['tp']} | {cm['fp']} | {cm['tn']} | {cm['fn']} |"
        )
    md_lines.append("")
    md_lines.append("## Risk Thresholds vs Quality-Risk Proxy (Output-Only)")
    md_lines.append("")
    md_lines.append("| Threshold | Precision | Recall | F1 | TP | FP | TN | FN |")
    md_lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for item in quality_proxy_threshold_metrics:
        cm = item["confusion"]
        md_lines.append(
            f"| {item['threshold']:.1f} | {_fmt(item['precision'])} | {_fmt(item['recall'])} | {_fmt(item['f1'])} | "
            f"{cm['tp']} | {cm['fp']} | {cm['tn']} | {cm['fn']} |"
        )
    md_lines.append("")
    md_lines.append("## Signal Checks")
    md_lines.append("")
    md_lines.append(
        f"- `repetition_loop_detected` vs output repetition heuristic: precision={_fmt(repetition_eval['precision'])}, "
        f"recall={_fmt(repetition_eval['recall'])}, f1={_fmt(repetition_eval['f1'])}"
    )
    md_lines.append(
        f"- `high_entropy_steps > 0` vs bad output (auto-graded prompts): precision={_fmt(high_entropy_eval['precision'])}, "
        f"recall={_fmt(high_entropy_eval['recall'])}, f1={_fmt(high_entropy_eval['f1'])}"
    )
    md_lines.append(
        f"- `high_entropy_steps > 0` vs quality-risk proxy (output-only): precision={_fmt(high_entropy_vs_quality_proxy_eval['precision'])}, "
        f"recall={_fmt(high_entropy_vs_quality_proxy_eval['recall'])}, f1={_fmt(high_entropy_vs_quality_proxy_eval['f1'])}"
    )
    md_lines.append("")
    md_lines.append("## Caveats")
    md_lines.append("")
    for note in summary["notes"]:
        md_lines.append(f"- {note}")
    md_lines.append("")
    md_lines.append("## Outputs")
    md_lines.append("")
    md_lines.append(f"- `graded_runs.csv`: `{csv_path}`")
    md_lines.append(f"- `summary.json`: `{summary_path}`")

    report_path = analysis_dir / "report.md"
    report_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nWrote:\n- {csv_path}\n- {summary_path}\n- {report_path}")
    return 0


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _rate(values: list[Optional[bool]]) -> Optional[float]:
    filtered = [bool(v) for v in values if v is not None]
    if not filtered:
        return None
    return sum(1 for v in filtered if v) / len(filtered)


def _stability_baseline_eval(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = [r for r in rows if "low_entropy_expected" in (r.get("case_tags") or [])]
    risks = [r.get("risk_score") for r in baseline if isinstance(r.get("risk_score"), (int, float))]
    high_entropy_flags = [((r.get("corevital_high_entropy_steps") or 0) > 0) for r in baseline if r.get("corevital_high_entropy_steps") is not None]
    return {
        "count": len(baseline),
        "avg_risk_score": (sum(risks) / len(risks)) if risks else None,
        "high_entropy_run_rate": (sum(1 for v in high_entropy_flags if v) / len(high_entropy_flags)) if high_entropy_flags else None,
    }


def _tag_signal_expectations(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        tags = r.get("case_tags") or []
        for tag in tags:
            buckets[tag].append(r)

    out: dict[str, dict[str, Any]] = {}
    for tag, rs in sorted(buckets.items()):
        risks = [r.get("risk_score") for r in rs if isinstance(r.get("risk_score"), (int, float))]
        out[tag] = {
            "count": len(rs),
            "avg_risk_score": (sum(risks) / len(risks)) if risks else None,
            "high_entropy_run_rate": _rate(
                [((r.get("corevital_high_entropy_steps") or 0) > 0) for r in rs if r.get("corevital_high_entropy_steps") is not None]
            ),
            "attention_collapse_rate": _rate([r.get("corevital_attention_collapse_flag") for r in rs]),
            "repetition_loop_rate": _rate([r.get("corevital_repetition_flag") for r in rs]),
        }
    return out


def _seed_stability_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(str(r.get("model_id")), str(r.get("case_id")))].append(r)

    risk_stds: list[float] = []
    entropy_stds: list[float] = []
    high_entropy_stds: list[float] = []
    groups_with_2plus = 0

    for (_model, _case), rs in groups.items():
        if len(rs) < 2:
            continue
        groups_with_2plus += 1
        risk_vals = [float(r["risk_score"]) for r in rs if isinstance(r.get("risk_score"), (int, float))]
        ent_vals = [float(r["entropy_max"]) for r in rs if isinstance(r.get("entropy_max"), (int, float))]
        he_vals = [float(r.get("corevital_high_entropy_steps") or 0) for r in rs if r.get("corevital_high_entropy_steps") is not None]
        if len(risk_vals) >= 2:
            risk_stds.append(_stddev(risk_vals))
        if len(ent_vals) >= 2:
            entropy_stds.append(_stddev(ent_vals))
        if len(he_vals) >= 2:
            high_entropy_stds.append(_stddev(he_vals))

    return {
        "total_groups": len(groups),
        "groups_with_2plus_seeds": groups_with_2plus,
        "avg_stddev_risk_score": (sum(risk_stds) / len(risk_stds)) if risk_stds else None,
        "avg_stddev_entropy_max": (sum(entropy_stds) / len(entropy_stds)) if entropy_stds else None,
        "avg_stddev_high_entropy_steps": (sum(high_entropy_stds) / len(high_entropy_stds)) if high_entropy_stds else None,
    }


def _attention_collapse_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_model[str(r.get("model_id"))].append(r)

    model_rates: dict[str, Optional[float]] = {}
    for m, rs in by_model.items():
        model_rates[m] = _rate([r.get("corevital_attention_collapse_flag") for r in rs])

    baseline_rows = [r for r in rows if "low_entropy_expected" in (r.get("case_tags") or [])]
    probe_rows = [r for r in rows if "probe" in (r.get("case_tags") or [])]
    return {
        "overall_rate": _rate([r.get("corevital_attention_collapse_flag") for r in rows]),
        "baseline_tag_rate": _rate([r.get("corevital_attention_collapse_flag") for r in baseline_rows]),
        "probe_tag_rate": _rate([r.get("corevital_attention_collapse_flag") for r in probe_rows]) if probe_rows else None,
        "per_model_rate": model_rates,
        "models_ge_0_9": sorted([m for m, v in model_rates.items() if v is not None and v >= 0.9]),
    }


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return var**0.5


if __name__ == "__main__":
    raise SystemExit(main())
