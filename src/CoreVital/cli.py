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
# ============================================================================

import argparse
import sys

from CoreVital import __version__
from CoreVital.config import Config
from CoreVital.instrumentation.collector import InstrumentationCollector
from CoreVital.logging_utils import setup_logging, get_logger
from CoreVital.reporting.report_builder import ReportBuilder
from CoreVital.sinks.local_file import LocalFileSink
from CoreVital.sinks.http_sink import HTTPSink
from CoreVital.errors import CoreVitalError


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
        "--remote_sink",
        type=str,
        choices=["none", "http"],
        default="none",
        help="Remote sink type (default: none)",
    )
    run_parser.add_argument(
        "--remote_url",
        type=str,
        default=None,
        help="Remote sink URL (required if remote_sink=http)",
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
        # Load configuration
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
        
        if args.remote_sink != "none":
            config.sink.type = args.remote_sink
            config.sink.remote_url = args.remote_url
        
        config.logging.level = args.log_level
        
        # Setup logging
        setup_logging(config.logging.level, config.logging.format)
        
        logger.info(f"Starting CoreVital v{__version__}")
        logger.info(f"Model: {args.model}")
        logger.info(f"Device: {args.device}")
        
        # Run instrumentation
        collector = InstrumentationCollector(config)
        raw_results = collector.run(args.prompt)
        
        # Build report
        builder = ReportBuilder(config)
        report = builder.build(raw_results, args.prompt)
        
        # Persist via sink
        if config.sink.type == "local_file":
            sink = LocalFileSink(config.sink.output_dir)
        elif config.sink.type == "http":
            if not config.sink.remote_url:
                raise ValueError("remote_url required when sink type is 'http'")
            sink = HTTPSink(config.sink.remote_url)
        else:
            raise ValueError(f"Unknown sink type: {config.sink.type}")
        
        output_location = sink.write(report)
        
        # Print summary
        print("\n" + "="*60)
        print("✓ Monitoring Complete")
        print("="*60)
        print(f"Model:           {report.model.hf_id}")
        print(f"Total steps:     {report.summary.total_steps}")
        print(f"Elapsed time:    {report.summary.elapsed_ms}ms")
        print(f"Output file:     {output_location}")
        
        if report.warnings:
            print(f"\nWarnings ({len(report.warnings)}):")
            for warning in report.warnings:
                print(f"  - [{warning.code}] {warning.message}")
        
        print("="*60 + "\n")
        
        return 0
        
    except CoreVitalError as e:
        logger.error(f"Monitoring error: {e}")
        print(f"\n✗ Error: {e}\n", file=sys.stderr)
        return 1
    except Exception as e:
        logger.exception("Unexpected error during monitoring")
        print(f"\n✗ Unexpected error: {e}\n", file=sys.stderr)
        return 2


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
    
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())