# ============================================================================
# CoreVital - Local File Sink
#
# Purpose: Write reports to local filesystem as JSON files
# Inputs: Report objects
# Outputs: JSON files in specified directory
# Dependencies: pathlib, json, base, utils.serialization
# Usage: sink = LocalFileSink("runs"); sink.write(report)
#
# Changelog:
#   2026-01-13: Initial LocalFileSink for Phase-0
#   2026-02-04: Phase-0.75 - added note: performance data is injected by CLI after write
#   2026-02-06: Performance data now arrives inside report.extensions before write()
#   2026-02-11: JSON size optimization — use compact format (indent=None) for smaller files
# ============================================================================

from pathlib import Path
from typing import Optional

from CoreVital.errors import SinkError
from CoreVital.logging_utils import get_logger
from CoreVital.reporting.schema import Report
from CoreVital.sinks.base import Sink
from CoreVital.utils.serialization import serialize_report_to_json

logger = get_logger(__name__)


class LocalFileSink(Sink):
    """
    Sink that writes reports to local JSON files.
    """

    def __init__(self, output_dir: str = "runs", indent: Optional[int] = None):
        """
        Initialize local file sink.

        Args:
            output_dir: Directory path for output files
            indent: JSON indentation (None=compact, 2=pretty). Pretty produces larger files.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.indent = indent
        logger.info(f"LocalFileSink initialized: {self.output_dir}")

    def write(self, report: Report) -> str:
        """
        Write report to a JSON file.

        Args:
            report: Report to write

        Returns:
            Path to written file

        Raises:
            SinkError: If write fails
        """
        filepath = None
        try:
            # Generate filename from trace_id
            trace_id_str = str(report.trace_id)
            safe_length = min(8, len(trace_id_str))
            filename = f"trace_{trace_id_str[:safe_length]}.json"
            filepath = self.output_dir / filename

            # Serialize report to JSON (compact by default; indent for human-readable when requested)
            json_str = serialize_report_to_json(report, indent=self.indent)

            # Write to file
            with open(filepath, "w") as f:
                f.write(json_str)

            logger.info(f"Report written to {filepath}")

            return str(filepath)

        except Exception as e:
            logger.exception("Failed to write report to file")
            raise SinkError(f"Failed to write report to {filepath if filepath else 'file'}", details=str(e)) from e
