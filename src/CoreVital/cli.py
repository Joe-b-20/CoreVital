# ============================================================================
# CoreVital - Command Line Interface
#
# Purpose: CLI entry point for monitoring runs
# Inputs: Command-line arguments
# Outputs: Monitoring results and trace artifacts
# Dependencies: argparse, config, instrumentation, sinks
# Usage: python -m CoreVital.cli run --model gpt2 --prompt "Hello"
#
# Changelog:
#   2026-01-13: Initial CLI with 'run' command for Phase-0
#   2026-01-15: Added --quantize-4 and --quantize-8 flags for quantization support
#   2026-02-04: Phase-0.75 - added --perf flag (summary/detailed/strict modes);
#                performance data injected into report extensions after sink_write;
#                detailed breakdown written as separate JSON file
#   2026-02-06: Fixed HTTP sink missing performance data - inject extensions.performance
#                into Report before sink.write() so both local_file and http sinks
#                receive complete data; removed post-write read-patch-write hack
#   2026-02-10: Phase-1b - added --no-prompt-telemetry flag
#   2026-02-11: Phase-1d - replaced --remote_sink/--remote_url with
#               --sink local|datadog|prometheus and per-sink config flags
# ============================================================================

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict

from CoreVital import __version__
from CoreVital.config import Config
from CoreVital.errors import ConfigurationError, CoreVitalError
from CoreVital.instrumentation.collector import InstrumentationCollector
from CoreVital.logging_utils import get_logger, setup_logging
from CoreVital.reporting.report_builder import ReportBuilder
from CoreVital.sinks.base import Sink
from CoreVital.sinks.local_file import LocalFileSink

logger = get_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """
    Create the argument parser for the CLI.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog="corevital",
        description="Monitor LLM inference health with deep instrumentation",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run monitoring on a model with a prompt",
    )
    run_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Hugging Face model ID (e.g., gpt2, facebook/opt-125m)",
    )
    run_parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Input prompt text",
    )
    run_parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=20,
        help="Maximum number of tokens to generate (default: 20)",
    )
    run_parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use for inference (default: auto)",
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    run_parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)",
    )
    run_parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter (default: 50)",
    )
    run_parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling parameter (default: 0.95)",
    )
    run_parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path (directory or file). Default: ./runs/",
    )
    run_parser.add_argument(
        "--sink",
        type=str,
        choices=["local", "datadog", "prometheus", "sqlite", "wandb"],
        default="sqlite",
        help="Sink: sqlite (DB, default), local (JSON), datadog, prometheus, wandb (Weights & Biases).",
    )
    run_parser.add_argument(
        "--datadog_api_key",
        type=str,
        default=None,
        help="Datadog API key (or set DD_API_KEY env var). Required for --sink datadog.",
    )
    run_parser.add_argument(
        "--datadog_site",
        type=str,
        default="datadoghq.com",
        help="Datadog site (default: datadoghq.com). Also settable via DD_SITE env var.",
    )
    run_parser.add_argument(
        "--prometheus_port",
        type=int,
        default=9091,
        help="Prometheus /metrics endpoint port (default: 9091). For --sink prometheus.",
    )
    run_parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="W&B project name (or set WANDB_PROJECT). For --sink wandb.",
    )
    run_parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity/team (or set WANDB_ENTITY). For --sink wandb.",
    )
    run_parser.add_argument(
        "--write-json",
        action="store_true",
        help="Also write a trace JSON file. With --sink sqlite (default), no JSON is written unless this is set.",
    )
    run_parser.add_argument(
        "--json-pretty",
        action="store_true",
        help="When writing JSON (--sink local or --write-json), use indented format. Larger file size.",
    )
    run_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config YAML file",
    )
    run_parser.add_argument(
        "--quantize-4",
        action="store_true",
        help="Load model with 4-bit quantization using bitsandbytes",
    )
    run_parser.add_argument(
        "--quantize-8",
        action="store_true",
        help="Load model with 8-bit quantization using bitsandbytes",
    )
    run_parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    run_parser.add_argument(
        "--perf",
        nargs="?",
        choices=["summary", "detailed", "strict"],
        const="summary",
        default=None,
        dest="perf_mode",
        metavar="MODE",
        help="Performance monitoring: summary (default), detailed (+ breakdown file), "
        "strict (+ warmup and baseline). Omit to disable.",
    )
    run_parser.add_argument(
        "--capture",
        type=str,
        choices=["summary", "full", "on_risk"],
        default=None,
        help="Capture mode: summary (small payload), full (all internals), or on_risk "
        "(summary unless risk score or health flags trigger a full capture).",
    )
    run_parser.add_argument(
        "--no-prompt-telemetry",
        action="store_true",
        default=False,
        help="Skip prompt forward pass (no sparse attention profiles, basin scores, "
        "layer transformations, or prompt surprisal).",
    )
    run_parser.add_argument(
        "--rag-context",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a JSON file with RAG context metadata (context_token_count, "
        "retrieved_doc_ids, retrieved_doc_titles, retrieval_metadata). Stored in report.extensions['rag'].",
    )
    run_parser.add_argument(
        "--export-otel",
        action="store_true",
        help="Export run to OpenTelemetry (OTLP). Use --otel-endpoint or env. Requires CoreVital[otel].",
    )
    run_parser.add_argument(
        "--otel-endpoint",
        type=str,
        default=None,
        metavar="HOST:PORT",
        help="OTLP gRPC endpoint (e.g. localhost:4317). Optional; env OTEL_EXPORTER_OTLP_ENDPOINT used if unset.",
    )

    # Migrate command: JSON files → SQLite DB
    migrate_parser = subparsers.add_parser(
        "migrate",
        help="Migrate JSON trace files to a SQLite database",
    )
    migrate_parser.add_argument(
        "--from-dir",
        type=str,
        default="runs",
        help="Directory containing trace_*.json files (default: runs)",
    )
    migrate_parser.add_argument(
        "--to-db",
        type=str,
        default="runs/corevital.db",
        help="Output SQLite database path (default: runs/corevital.db)",
    )
    migrate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be migrated without writing",
    )

    # Compare command: summarize risk by model from SQLite DB (Phase-6 optional CLI)
    compare_parser = subparsers.add_parser(
        "compare",
        help="Summarize runs by model from a SQLite database",
    )
    compare_parser.add_argument(
        "--db",
        type=str,
        default="runs/corevital.db",
        help="Path to SQLite database (default: runs/corevital.db)",
    )
    compare_parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximum number of rows to read from the DB (default: 500)",
    )
    compare_parser.add_argument(
        "--prompt-hash",
        type=str,
        default=None,
        help="Optional prompt_hash filter to compare models on the same prompt group.",
    )

    return parser


def run_command(args: argparse.Namespace) -> int:
    """
    Execute the 'run' command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Performance monitoring: Create monitor early to track all parent operations
        perf_mode = getattr(args, "perf_mode", None)
        monitor = None
        if perf_mode:
            from CoreVital.instrumentation.performance import PerformanceMonitor

            monitor = PerformanceMonitor(mode=perf_mode)
            # Start timing now for ALL modes (before config_load)
            monitor.mark_run_start()

        def _op(name: str):
            """Helper to wrap parent operations in monitor.operation() if enabled."""
            return monitor.operation(name) if monitor else nullcontext()

        # === PARENT: config_load ===
        with _op("config_load"):
            config = Config.from_yaml(args.config) if args.config else Config.from_default()

            # Override with CLI arguments
            config.model.hf_id = args.model
            config.device.requested = args.device
            config.generation.max_new_tokens = args.max_new_tokens
            config.generation.seed = args.seed
            config.generation.temperature = args.temperature
            config.generation.top_k = args.top_k
            config.generation.top_p = args.top_p
            config.model.load_in_4bit = args.quantize_4
            config.model.load_in_8bit = args.quantize_8

            if args.out:
                config.sink.output_dir = args.out

            # Sink type from --sink flag
            if args.sink == "local":
                config.sink.type = "local_file"
            elif args.sink == "sqlite":
                config.sink.type = "sqlite"
                if args.out:
                    config.sink.sqlite_path = str(Path(args.out) / "corevital.db")
            elif args.sink == "datadog":
                config.sink.type = "datadog"
            elif args.sink == "prometheus":
                config.sink.type = "prometheus"
            elif args.sink == "wandb":
                config.sink.type = "wandb"

            config.logging.level = args.log_level

            # Store perf mode in config
            if perf_mode:
                config.performance.mode = perf_mode

            # Prompt telemetry (Phase-1b)
            if args.no_prompt_telemetry:
                config.prompt_telemetry.enabled = False

            # Capture mode (Foundation F2)
            if getattr(args, "capture", None):
                config.capture.capture_mode = args.capture

            # RAG context (Foundation F3): load from JSON file if provided
            rag_path = getattr(args, "rag_context", None)
            if rag_path:
                rag_path = Path(rag_path)
                if not rag_path.exists():
                    raise FileNotFoundError(f"RAG context file not found: {rag_path}")
                with open(rag_path, "r") as f:
                    config.rag_context = json.load(f)

            # OpenTelemetry export (optional)
            if getattr(args, "export_otel", False):
                config.otel.export_otel = True
            if getattr(args, "otel_endpoint", None):
                config.otel.otel_endpoint = args.otel_endpoint

        # === PARENT: setup_logging ===
        with _op("setup_logging"):
            setup_logging(config.logging.level, config.logging.format)

        logger.info(f"Starting CoreVital v{__version__}")
        logger.info(f"Model: {args.model}")
        logger.info(f"Device: {args.device}")

        # Run instrumentation (includes model_load, torch.manual_seed, tokenize, model_inference)
        collector = InstrumentationCollector(config)
        raw_results = collector.run(args.prompt, monitor=monitor)

        # === PARENT: report_build ===
        builder = ReportBuilder(config)
        with _op("report_build"):
            report = builder.build(raw_results, args.prompt)

        # === END OF INSTRUMENTED RUN ===
        # Finalize performance data BEFORE sink.write() so both local_file and
        # http sinks receive the complete report including extensions.performance.
        # sink_write is excluded from parent_operations because the performance
        # summary must be finalized before the write that carries it.
        if monitor:
            monitor.mark_run_end()

            # Build performance summary
            perf_summary = monitor.build_summary_dict()

            # For detailed/strict modes, write the detailed breakdown file
            mode = monitor.mode
            trace_id = report.trace_id
            if mode in ("detailed", "strict"):
                output_dir = Path(config.sink.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                detailed_path = output_dir / f"trace_{trace_id[:8]}_performance_detailed.json"
                monitor.set_detailed_file(str(detailed_path))
                perf_summary["detailed_file"] = str(detailed_path)

                detailed_breakdown = monitor.build_detailed_breakdown()
                detailed_breakdown["trace_id"] = trace_id[:8]
                with open(detailed_path, "w") as f:
                    # Compact JSON for smaller file size (dashboard can format on-demand)
                    json.dump(detailed_breakdown, f, separators=(",", ":"), ensure_ascii=False)
                logger.info(f"Performance detailed written to {detailed_path}")

            # Inject into report so sink.write() serializes the complete data
            report.extensions["performance"] = perf_summary

        # === SINK WRITE ===
        # Not wrapped in _op() - happens after perf data is finalized
        sink: Sink
        json_indent = 2 if getattr(args, "json_pretty", False) else None
        if config.sink.type == "local_file":
            sink = LocalFileSink(config.sink.output_dir, indent=json_indent)
        elif config.sink.type == "sqlite":
            from CoreVital.sinks.sqlite_sink import SQLiteSink

            sink = SQLiteSink(
                db_path=config.sink.sqlite_path,
                backup_json=getattr(args, "write_json", False),
                json_indent=json_indent,
            )
        elif config.sink.type == "datadog":
            import os

            from CoreVital.sinks.datadog_sink import DatadogSink

            api_key = args.datadog_api_key or os.environ.get("DD_API_KEY")
            if not api_key:
                raise ConfigurationError("Datadog API key required. Pass --datadog_api_key or set DD_API_KEY env var.")
            site = args.datadog_site
            if os.environ.get("DD_SITE"):
                site = os.environ["DD_SITE"]
            sink = DatadogSink(api_key=api_key, site=site, local_output_dir=config.sink.output_dir)
        elif config.sink.type == "prometheus":
            from CoreVital.sinks.prometheus_sink import PrometheusSink

            sink = PrometheusSink(port=args.prometheus_port, local_output_dir=config.sink.output_dir)
        elif config.sink.type == "wandb":
            import os

            from CoreVital.sinks.wandb_sink import WandBSink

            project = getattr(args, "wandb_project", None) or os.environ.get("WANDB_PROJECT")
            entity = getattr(args, "wandb_entity", None) or os.environ.get("WANDB_ENTITY")
            sink = WandBSink(
                project=project,
                entity=entity,
                local_output_dir=config.sink.output_dir,
            )
        else:
            raise ConfigurationError(f"Unknown sink type: {config.sink.type}")

        output_location = sink.write(report)

        # Optional: export to OpenTelemetry (Langfuse, OpenLIT, OTLP)
        if config.otel.export_otel:
            import os

            from CoreVital.integrations.opentelemetry import export_run_to_otel, get_otel_tracer_meter

            endpoint = config.otel.otel_endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
            tracer, meter = get_otel_tracer_meter(endpoint)
            export_run_to_otel(report, tracer, meter)

        # Print summary
        print("\n" + "=" * 60)
        print("✓ Monitoring Complete")
        print("=" * 60)
        print(f"Model:           {report.model.hf_id}")
        print(f"Total steps:     {report.summary.total_steps}")
        print(f"Elapsed time:    {report.summary.elapsed_ms}ms")
        print(f"Output file:     {output_location}")

        if report.warnings:
            print(f"\nWarnings ({len(report.warnings)}):")
            for warning in report.warnings:
                print(f"  - [{warning.code}] {warning.message}")

        print("=" * 60 + "\n")

        return 0

    except CoreVitalError as e:
        logger.error(f"Monitoring error: {e}")
        print(f"\n✗ Error: {e}\n", file=sys.stderr)
        return 1
    except Exception as e:
        logger.exception("Unexpected error during monitoring")
        print(f"\n✗ Unexpected error: {e}\n", file=sys.stderr)
        return 2


def migrate_command(args: argparse.Namespace) -> int:
    """
    Migrate JSON trace files from a directory into a SQLite database.

    Returns:
        Exit code (0 success, non-zero failure)
    """
    from CoreVital.sinks.sqlite_sink import SQLiteSink
    from CoreVital.utils.serialization import deserialize_report_from_json

    from_dir = Path(args.from_dir)
    to_db = args.to_db
    dry_run = getattr(args, "dry_run", False)

    if not from_dir.is_dir():
        logger.error(f"Directory not found: {from_dir}")
        print(f"\n✗ Error: {from_dir} is not a directory.\n", file=sys.stderr)
        return 1

    json_files = sorted(from_dir.glob("trace_*.json"))
    json_files = [f for f in json_files if "_performance_" not in f.name]

    if not json_files:
        print(f"No trace_*.json files found in {from_dir}")
        return 0

    if dry_run:
        print(f"Dry run: would migrate {len(json_files)} file(s) to {to_db}")
        for f in json_files:
            print(f"  - {f.name}")
        return 0

    sink = SQLiteSink(db_path=to_db, compress=True)
    migrated = 0
    errors = 0
    for path in json_files:
        try:
            report = deserialize_report_from_json(path.read_text(encoding="utf-8"))
            sink.write(report)
            migrated += 1
            logger.debug(f"Migrated {path.name}")
        except Exception as e:
            logger.warning(f"Failed to migrate {path.name}: {e}")
            errors += 1

    print(f"\n✓ Migrated {migrated} report(s) to {to_db}" + (f" ({errors} failed)" if errors else "") + "\n")
    return 0 if errors == 0 else 1


def compare_command(args: argparse.Namespace) -> int:
    """
    Summarize runs by model_id from a SQLite database (Phase-6 CLI compare).

    Prints a small table with count and basic risk statistics per model.
    """
    from CoreVital.sinks.sqlite_sink import SQLiteSink

    db_path = args.db
    limit = getattr(args, "limit", 500)
    prompt_hash = getattr(args, "prompt_hash", None)

    rows = SQLiteSink.list_traces(db_path=db_path, limit=limit, model_id=None, prompt_hash=prompt_hash)
    if not rows:
        print(f"No reports found in {db_path}" + (f" for prompt_hash={prompt_hash}" if prompt_hash else ""))
        return 0

    # Group by model_id and accumulate risk stats
    stats: Dict[str, Dict[str, float]] = {}
    for r in rows:
        model_id = r.get("model_id") or "unknown"
        risk = r.get("risk_score")
        if model_id not in stats:
            stats[model_id] = {
                "count": 0.0,
                "risk_sum": 0.0,
                "risk_min": float("inf"),
                "risk_max": float("-inf"),
                "risk_samples": 0.0,
            }
        s = stats[model_id]
        s["count"] += 1.0
        if risk is not None:
            try:
                rv = float(risk)
            except (TypeError, ValueError):
                rv = None
            if rv is not None:
                s["risk_sum"] += rv
                s["risk_samples"] += 1.0
                if rv < s["risk_min"]:
                    s["risk_min"] = rv
                if rv > s["risk_max"]:
                    s["risk_max"] = rv

    # Print summary table
    print("\nModel comparison from SQLite")
    print(f"DB: {db_path}")
    if prompt_hash:
        print(f"Prompt hash filter: {prompt_hash}")
    print("-" * 72)
    header = f"{'Model':40s} {'Runs':>6s} {'Avg risk':>10s} {'Min':>8s} {'Max':>8s}"
    print(header)
    print("-" * 72)

    for model_id, s in sorted(stats.items(), key=lambda kv: kv[0]):
        count = int(s["count"])
        samples = int(s["risk_samples"])
        if samples > 0:
            avg = s["risk_sum"] / samples
            rmin = s["risk_min"]
            rmax = s["risk_max"]
        else:
            avg = float("nan")
            rmin = float("nan")
            rmax = float("nan")
        print(f"{model_id[:40]:40s} {count:6d} {avg:10.3f} {rmin:8.3f} {rmax:8.3f}")

    print("-" * 72 + "\n")
    return 0


def main() -> int:
    """
    Main CLI entry point.

    Returns:
        Exit code
    """
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "run":
        return run_command(args)
    if args.command == "migrate":
        return migrate_command(args)
    if args.command == "compare":
        return compare_command(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
