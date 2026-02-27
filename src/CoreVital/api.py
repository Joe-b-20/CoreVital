# ============================================================================
# CoreVital - Local API
#
# Purpose: Lightweight FastAPI server for local SQLite database connections.
#          Used by the hosted React dashboard to list and load traces;
#          data never leaves the user's machine.
# Dependencies: fastapi, uvicorn, sinks.sqlite_sink
# Usage: corevital serve (or uvicorn CoreVital.api:app --host 0.0.0.0 --port 8000)
# ============================================================================

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from CoreVital.sinks.sqlite_sink import SQLiteSink

# Database path: query param (from UI) > env > default relative to cwd
DEFAULT_DB_PATH = "runs/corevital.db"


def _normalize_db_path(path: str) -> str:
    """Use forward slashes so paths from Windows UI work when server runs on Linux/WSL."""
    return path.replace("\\", "/").strip().strip('"')


def _get_db_path(db_path_query: str | None = None) -> str:
    if db_path_query and db_path_query.strip():
        return _normalize_db_path(db_path_query)
    return os.environ.get("COREVITAL_DB_PATH", DEFAULT_DB_PATH)


app = FastAPI(
    title="CoreVital Local API",
    description="List and load trace reports from a local SQLite database for the CoreVital dashboard.",
    version="0.4.0",
)

# Allow all origins so the AWS-hosted React app can query this local server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_report_by_id(db_path: str, trace_id: str) -> dict:
    """Load a single report by trace_id; raises HTTPException 404 if not found or DB missing."""
    if not Path(db_path).exists():
        raise HTTPException(status_code=404, detail="Database not found")
    report = SQLiteSink.load_report(db_path=db_path, trace_id=trace_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"Trace not found: {trace_id}")
    return report


@app.get("/api/traces")
def list_traces(
    limit: int = 100,
    model_id: str | None = None,
    prompt_hash: str | None = None,
    order_asc: bool = False,
    db_path: str | None = None,
) -> list[dict]:
    """
    List runs (trace metadata) from the local SQLite database.
    Returns trace_id, created_at_utc, model_id, schema_version, prompt_hash, risk_score.
    Optional db_path: path to DB (use forward slashes; backslashes are normalized for Windows paths).
    """
    resolved = _get_db_path(db_path)
    if not Path(resolved).exists():
        return []
    return SQLiteSink.list_traces(
        db_path=resolved,
        limit=limit,
        model_id=model_id,
        prompt_hash=prompt_hash,
        order_asc=order_asc,
    )


@app.get("/api/traces/{trace_id}")
def get_trace(trace_id: str, db_path: str | None = None) -> dict:
    """
    Load a single run (full report) by trace_id from the local SQLite database.
    Optional db_path: path to DB (use forward slashes; backslashes are normalized).
    """
    resolved = _get_db_path(db_path)
    return _load_report_by_id(resolved, trace_id)


@app.get("/api/reports/{trace_id}")
def get_report(trace_id: str, db_path: str | None = None) -> dict:
    """
    Load a single run (full report) by trace_id. Same as GET /api/traces/{trace_id};
    provided so the React dashboard can call /api/reports/{trace_id}.
    Optional db_path: path to DB (use forward slashes; backslashes are normalized).
    """
    resolved = _get_db_path(db_path)
    return _load_report_by_id(resolved, trace_id)
