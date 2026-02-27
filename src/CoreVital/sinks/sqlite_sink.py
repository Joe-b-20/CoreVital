# ============================================================================
# CoreVital - SQLite Sink
#
# Purpose: Persist reports to SQLite for lightweight storage and dashboard use
# Inputs: Report objects
# Outputs: SQLite database (reports table)
# Dependencies: pathlib, sqlite3, base, utils.serialization
# Usage: sink = SQLiteSink("runs"); sink.write(report)
#
# Changelog:
#   2026-02-11: Initial SQLite sink — store reports as compact JSON in DB
#   2026-02-18: Schema 0.4.0 — reports stored with schema_version 0.4.0; validation
#               accepts 0.3.0 and 0.4.0 when loading (migration path for existing DBs)
# ============================================================================

import gzip
import json
import sqlite3
from pathlib import Path
from typing import Any, List, Optional

from CoreVital.errors import SinkError
from CoreVital.logging_utils import get_logger
from CoreVital.reporting.schema import Report
from CoreVital.sinks.base import Sink
from CoreVital.utils.serialization import serialize_report_to_json

logger = get_logger(__name__)

# Table schema: one row per report; optional prompt_hash/risk_score for Phase-6 comparison
_INIT_SQL = """
CREATE TABLE IF NOT EXISTS reports (
    trace_id TEXT PRIMARY KEY,
    created_at_utc TEXT NOT NULL,
    model_id TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    report_json TEXT,
    report_blob BLOB,
    prompt_hash TEXT,
    risk_score REAL
);
CREATE INDEX IF NOT EXISTS idx_reports_created ON reports(created_at_utc DESC);
CREATE INDEX IF NOT EXISTS idx_reports_model_id ON reports(model_id);
CREATE INDEX IF NOT EXISTS idx_reports_prompt_hash ON reports(prompt_hash);
"""


class SQLiteSink(Sink):
    """
    Sink that writes reports to a SQLite database.
    Keeps payloads small by storing one report per row (compact JSON).
    """

    def __init__(
        self,
        db_path: str = "runs/corevital.db",
        backup_json: bool = False,
        json_indent: Optional[int] = None,
        compress: bool = True,
    ):
        """
        Initialize SQLite sink.

        Args:
            db_path: Path to SQLite database file (created if missing)
            backup_json: If True, also write a JSON file alongside the DB (same dir as DB parent)
            json_indent: When backup_json is True, indent for JSON file (None=compact, 2=pretty).
            compress: If True, store report as gzip blob (smaller DB, faster for large reports)
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.backup_json = backup_json
        self.json_indent = json_indent
        self.compress = compress
        self._init_db()
        logger.info(f"SQLiteSink initialized: {self.db_path} (compress={compress})")

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(_INIT_SQL)
            # Migration: add report_blob if table existed without it
            cur = conn.execute("SELECT name FROM pragma_table_info('reports') WHERE name = 'report_blob'")
            if cur.fetchone() is None:
                try:
                    conn.execute("ALTER TABLE reports ADD COLUMN report_blob BLOB")
                except sqlite3.OperationalError:
                    pass
            # Migration: add prompt_hash, risk_score for Phase-6 comparison
            for col in ("prompt_hash", "risk_score"):
                cur = conn.execute("SELECT name FROM pragma_table_info('reports') WHERE name = ?", (col,))
                if cur.fetchone() is None:
                    try:
                        conn.execute(
                            f"ALTER TABLE reports ADD COLUMN {col} " + ("TEXT" if col == "prompt_hash" else "REAL")
                        )
                    except sqlite3.OperationalError:
                        pass

    def write(self, report: Report) -> str:
        """
        Write report to SQLite (and optionally a JSON backup).

        Args:
            report: Report to persist

        Returns:
            Database path (e.g. "runs/corevital.db")

        Raises:
            SinkError: If write fails
        """
        try:
            json_str = serialize_report_to_json(report, indent=None)
            ext = getattr(report, "extensions", None) or {}
            fp = ext.get("fingerprint") or {}
            risk = ext.get("risk") or {}
            prompt_hash = fp.get("prompt_hash") if isinstance(fp, dict) else None
            risk_score = risk.get("risk_score") if isinstance(risk, dict) else None
            with sqlite3.connect(self.db_path) as conn:
                if self.compress:
                    blob = gzip.compress(json_str.encode("utf-8"), mtime=0)
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO reports
                        (trace_id, created_at_utc, model_id, schema_version,
                         report_json, report_blob, prompt_hash, risk_score)
                        VALUES (?, ?, ?, ?, NULL, ?, ?, ?)
                        """,
                        (
                            report.trace_id,
                            report.created_at_utc,
                            report.model.hf_id,
                            report.schema_version,
                            blob,
                            prompt_hash,
                            risk_score,
                        ),
                    )
                else:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO reports
                        (trace_id, created_at_utc, model_id, schema_version,
                         report_json, report_blob, prompt_hash, risk_score)
                        VALUES (?, ?, ?, ?, ?, NULL, ?, ?)
                        """,
                        (
                            report.trace_id,
                            report.created_at_utc,
                            report.model.hf_id,
                            report.schema_version,
                            json_str,
                            prompt_hash,
                            risk_score,
                        ),
                    )

            if self.backup_json:
                backup_dir = self.db_path.parent
                safe_id = report.trace_id[:8]
                backup_path = backup_dir / f"trace_{safe_id}.json"
                backup_json_str = serialize_report_to_json(report, indent=self.json_indent)
                backup_path.write_text(backup_json_str, encoding="utf-8")
                logger.debug(f"JSON backup written to {backup_path}")

            logger.info(f"Report written to DB: {self.db_path} (trace_id={report.trace_id[:8]})")
            return str(self.db_path)

        except Exception as e:
            logger.exception("Failed to write report to SQLite")
            raise SinkError(
                f"Failed to write report to {self.db_path}",
                details=str(e),
            ) from e

    @staticmethod
    def list_traces(
        db_path: str,
        limit: int = 100,
        model_id: Optional[str] = None,
        prompt_hash: Optional[str] = None,
        order_asc: bool = False,
    ) -> List[dict]:
        """
        List reports from the database (lightweight: no report body).

        Args:
            db_path: Path to SQLite database
            limit: Max number of rows to return
            model_id: If set, filter by model_id (exact match)
            prompt_hash: If set, filter by prompt_hash (exact match)
            order_asc: If True, order by created_at_utc ASC (oldest first). Default False = newest first.

        Returns:
            List of dicts with keys: trace_id, created_at_utc, model_id, schema_version,
            prompt_hash (if column present), risk_score (if column present).
        """
        path = Path(db_path)
        if not path.exists():
            return []
        order = "ASC" if order_asc else "DESC"
        rows = []
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            params: list = []
            where = []
            if model_id is not None:
                where.append("model_id = ?")
                params.append(model_id)
            if prompt_hash is not None:
                where.append("prompt_hash = ?")
                params.append(prompt_hash)
            params.append(limit)
            sel_full = "SELECT trace_id, created_at_utc, model_id, schema_version, prompt_hash, risk_score FROM reports"
            if where:
                sel_full += " WHERE " + " AND ".join(where)
            sel_full += f" ORDER BY created_at_utc {order} LIMIT ?"
            try:
                cur = conn.execute(sel_full, params)
            except sqlite3.OperationalError:
                # Old DB without prompt_hash/risk_score
                sel_old = "SELECT trace_id, created_at_utc, model_id, schema_version FROM reports"
                if model_id is not None:
                    sel_old += " WHERE model_id = ?"
                sel_old += f" ORDER BY created_at_utc {order} LIMIT ?"
                params_old = [p for p in params if p != prompt_hash] if prompt_hash is not None else params[:-1]
                cur = conn.execute(sel_old, params_old)
            for row in cur:
                rows.append(dict(row))
        return rows

    @staticmethod
    def load_report(db_path: str, trace_id: str) -> Optional[dict]:
        """
        Load a single report by trace_id (deserialize JSON to dict).
        Supports both report_json (text) and report_blob (gzip) columns.

        Args:
            db_path: Path to SQLite database
            trace_id: Full or short trace_id (short matches prefix of stored trace_id)

        Returns:
            Report as dict, or None if not found
        """
        path = Path(db_path)
        if not path.exists():
            return None
        with sqlite3.connect(path) as conn:
            cur = conn.execute(
                "SELECT report_json, report_blob FROM reports WHERE trace_id = ? LIMIT 1",
                (trace_id,),
            )
            row = cur.fetchone()
            if row is None and len(trace_id) < 36:
                cur = conn.execute(
                    "SELECT report_json, report_blob FROM reports WHERE trace_id LIKE ? LIMIT 1",
                    (trace_id + "%",),
                )
                row = cur.fetchone()
        if row is None:
            return None
        json_str, blob = row[0], row[1]
        if blob is not None:
            json_str = gzip.decompress(blob).decode("utf-8")
        elif json_str is None:
            return None
        out: dict[str, Any] = json.loads(json_str)
        return out
