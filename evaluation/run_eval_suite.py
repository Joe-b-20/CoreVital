#!/usr/bin/env python3
"""
Automated CoreVital evaluation runner.

Runs a model x prompt x seed matrix, captures all CoreVital trace JSONs, and writes a
manifest with stdout/stderr/trace metadata so analysis can be fully automated.

Key property: no manual copying from the terminal is required. All outputs are saved.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from eval_suite_common import EvalCase, render_prompt, resolve_models, selected_cases


@dataclass
class RunRecord:
    run_id: str
    model_id: str
    seed: int
    case_id: str
    prompt: str
    rendered_prompt: str
    max_new_tokens: int
    cmd: list[str]
    started_at_utc: str
    duration_ms: int
    returncode: int
    stdout_log: str
    stderr_log: str
    trace_path: Optional[str] = None
    trace_id: Optional[str] = None
    generated_text: Optional[str] = None
    risk_score: Optional[float] = None
    failure_risk: Optional[float] = None
    high_entropy_steps: Optional[int] = None
    attention_collapse_detected: Optional[bool] = None
    repetition_loop_detected: Optional[bool] = None
    nan_or_inf_detected: Optional[bool] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _build_cmd(
    model_id: str,
    prompt_text: str,
    case: EvalCase,
    seed: int,
    *,
    python_bin: str,
    device: str,
    traces_dir: Path,
    timeout_s: int,
    perf_mode: Optional[str],
    capture_mode: Optional[str],
    quantize_4: bool,
    quantize_8: bool,
) -> list[str]:
    cmd = [
        python_bin,
        "-m",
        "CoreVital.cli",
        "run",
        "--model",
        model_id,
        "--prompt",
        prompt_text,
        "--max_new_tokens",
        str(case.max_new_tokens),
        "--seed",
        str(seed),
        "--device",
        device,
        "--sink",
        "local",
        "--out",
        str(traces_dir),
    ]
    if perf_mode:
        cmd.extend(["--perf", perf_mode])
    if capture_mode:
        cmd.extend(["--capture", capture_mode])
    if quantize_4:
        cmd.append("--quantize-4")
    if quantize_8:
        cmd.append("--quantize-8")
    # timeout_s is applied in subprocess.run, not as a CLI arg
    _ = timeout_s
    return cmd


def _read_report_summary(trace_path: Path) -> dict[str, Any]:
    with open(trace_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    ext = d.get("extensions") or {}
    risk = ext.get("risk") or {}
    ew = ext.get("early_warning") or {}
    hf = d.get("health_flags") or {}
    return {
        "trace_id": d.get("trace_id"),
        "generated_text": (d.get("generated") or {}).get("output_text"),
        "risk_score": risk.get("risk_score"),
        "failure_risk": ew.get("failure_risk"),
        "high_entropy_steps": hf.get("high_entropy_steps"),
        "attention_collapse_detected": hf.get("attention_collapse_detected"),
        "repetition_loop_detected": hf.get("repetition_loop_detected"),
        "nan_or_inf_detected": bool(hf.get("nan_detected") or hf.get("inf_detected")),
    }


def _new_trace_file(before: set[str], after: set[str], traces_dir: Path) -> Optional[Path]:
    created = sorted(after - before)
    if not created:
        return None
    # Local sink writes one file per run; use the latest created filename.
    return traces_dir / created[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an automated CoreVital evaluation matrix and capture all outputs.")
    parser.add_argument("--out-dir", type=str, default="evaluation/runs/eval_suite", help="Suite output directory")
    parser.add_argument(
        "--model-preset",
        type=str,
        default="cpu_default",
        help="Model preset from scripts/eval_suite_common.py (default: cpu_default)",
    )
    parser.add_argument("--models", nargs="+", default=None, help="Explicit model IDs (overrides preset)")
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,123",
        help="Comma-separated seeds (default: 42,123)",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--limit-cases", type=int, default=None, help="Limit number of cases for quick runs")
    parser.add_argument("--no-probes", action="store_true", help="Exclude non-graded probe prompts")
    parser.add_argument("--perf", choices=["summary", "detailed", "strict"], default=None)
    parser.add_argument("--capture", choices=["summary", "full", "on_risk"], default="summary")
    parser.add_argument("--timeout", type=int, default=900, help="Per-run timeout seconds")
    parser.add_argument("--quantize-4", action="store_true")
    parser.add_argument("--quantize-8", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    out_dir = (project_root / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    traces_dir = out_dir / "traces"
    logs_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    try:
        models = resolve_models(args.models, args.model_preset)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    cases = selected_cases(include_probes=not args.no_probes, limit=args.limit_cases)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        print("No seeds provided", file=sys.stderr)
        return 2

    suite_meta = {
        "created_at_utc": _utc_now_iso(),
        "project_root": str(project_root),
        "out_dir": str(out_dir),
        "models": models,
        "model_preset": args.model_preset,
        "seeds": seeds,
        "device": args.device,
        "perf": args.perf,
        "capture": args.capture,
        "timeout_seconds": args.timeout,
        "quantize_4": args.quantize_4,
        "quantize_8": args.quantize_8,
        "cases": [c.to_dict() for c in cases],
        "dry_run": bool(args.dry_run),
    }
    (out_dir / "suite_meta.json").write_text(json.dumps(suite_meta, indent=2), encoding="utf-8")

    total_runs = len(models) * len(cases) * len(seeds)
    print(f"Eval suite: {len(models)} models x {len(cases)} cases x {len(seeds)} seeds = {total_runs} runs")
    print(f"Outputs: traces -> {traces_dir}")
    print(f"Manifest: {out_dir / 'manifest.jsonl'}")

    manifest_jsonl = out_dir / "manifest.jsonl"
    if manifest_jsonl.exists() and not args.dry_run:
        # Start a fresh manifest per run directory to keep analysis deterministic.
        manifest_jsonl.unlink()

    env = {**os.environ, "PYTHONPATH": str(project_root / "src")}
    python_bin = sys.executable
    records: list[RunRecord] = []

    run_num = 0
    for model_id in models:
        for case in cases:
            for seed in seeds:
                run_num += 1
                run_id = f"{run_num:04d}_{case.case_id}_{seed}"
                rendered_prompt = render_prompt(case, model_id)
                cmd = _build_cmd(
                    model_id,
                    rendered_prompt,
                    case,
                    seed,
                    python_bin=python_bin,
                    device=args.device,
                    traces_dir=traces_dir,
                    timeout_s=args.timeout,
                    perf_mode=args.perf,
                    capture_mode=args.capture,
                    quantize_4=args.quantize_4,
                    quantize_8=args.quantize_8,
                )
                print(f"[{run_num}/{total_runs}] {model_id} | {case.case_id} | seed={seed}", flush=True)
                if args.dry_run:
                    print("  " + " ".join(cmd))
                    continue

                stdout_log = logs_dir / f"{run_id}.stdout.log"
                stderr_log = logs_dir / f"{run_id}.stderr.log"
                before = {p.name for p in traces_dir.glob("trace_*.json")}
                started = _utc_now_iso()
                t0 = time.perf_counter()
                error: Optional[str] = None
                try:
                    proc = subprocess.run(
                        cmd,
                        cwd=project_root,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=args.timeout,
                    )
                    rc = proc.returncode
                    stdout_text = proc.stdout or ""
                    stderr_text = proc.stderr or ""
                except subprocess.TimeoutExpired as e:
                    rc = 124
                    stdout_text = e.stdout or ""
                    stderr_text = (e.stderr or "") + "\nTIMEOUT"
                    error = "timeout"
                except Exception as e:
                    rc = 1
                    stdout_text = ""
                    stderr_text = f"{type(e).__name__}: {e}"
                    error = f"{type(e).__name__}: {e}"
                duration_ms = int((time.perf_counter() - t0) * 1000)

                stdout_log.write_text(stdout_text, encoding="utf-8")
                stderr_log.write_text(stderr_text, encoding="utf-8")

                after = {p.name for p in traces_dir.glob("trace_*.json")}
                trace_path = _new_trace_file(before, after, traces_dir)

                record = RunRecord(
                    run_id=run_id,
                    model_id=model_id,
                    seed=seed,
                    case_id=case.case_id,
                    prompt=case.prompt,
                    rendered_prompt=rendered_prompt,
                    max_new_tokens=case.max_new_tokens,
                    cmd=cmd,
                    started_at_utc=started,
                    duration_ms=duration_ms,
                    returncode=rc,
                    stdout_log=str(stdout_log),
                    stderr_log=str(stderr_log),
                    trace_path=str(trace_path) if trace_path else None,
                    error=error,
                )

                if rc == 0 and trace_path and trace_path.exists():
                    try:
                        summary = _read_report_summary(trace_path)
                        for k, v in summary.items():
                            setattr(record, k, v)
                    except Exception as e:
                        record.error = f"trace_parse_error: {e}"
                elif rc != 0 and record.error is None:
                    record.error = "run_failed"

                records.append(record)
                with open(manifest_jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

                status = "OK" if record.returncode == 0 else "FAIL"
                trace_short = Path(record.trace_path).name if record.trace_path else "no-trace"
                print(
                    f"  -> {status} ({record.duration_ms} ms), trace={trace_short}, risk={record.risk_score}",
                    flush=True,
                )

    if args.dry_run:
        print("Dry-run complete.")
        return 0

    summary = {
        "total_runs": total_runs,
        "attempted_runs": len(records),
        "successful_runs": sum(1 for r in records if r.returncode == 0),
        "failed_runs": sum(1 for r in records if r.returncode != 0),
        "out_dir": str(out_dir),
        "manifest_jsonl": str(manifest_jsonl),
        "traces_dir": str(traces_dir),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
