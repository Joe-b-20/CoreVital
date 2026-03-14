#!/usr/bin/env python3
"""
CoreVital Validation Pipeline — Shared Helpers
================================================
Common utilities for extract_features.py and analyze.py:
  - Dark-theme plot configuration
  - Manifest / metadata writers
  - save_table_and_meta(): dual CSV + JSON output
  - Safe statistical primitives (mean, std, slope)
  - Logging configuration
  - Section output directory helpers

Used by both scripts to avoid duplicated logic without
introducing a framework.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Logging ─────────────────────────────────────────────────

def setup_logging(name: str = "corevital", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger with consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S"
        ))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ── Dark-Theme Plot Style ───────────────────────────────────

# Publication-friendly dark palette
DARK_BG = "#1a1a2e"
DARK_FACE = "#16213e"
DARK_TEXT = "#e0e0e0"
DARK_GRID = "#2a2a4a"
ACCENT_COLORS = [
    "#00d4ff",  # cyan
    "#ff6b6b",  # coral
    "#51cf66",  # green
    "#ffd43b",  # gold
    "#cc5de8",  # purple
    "#ff922b",  # orange
    "#20c997",  # teal
    "#e599f7",  # pink
]
CORRECT_COLOR = "#51cf66"
INCORRECT_COLOR = "#ff6b6b"


def apply_dark_theme():
    """Apply a consistent dark theme to all matplotlib plots."""
    plt.style.use("dark_background")
    matplotlib.rcParams.update({
        "figure.facecolor": DARK_BG,
        "axes.facecolor": DARK_FACE,
        "axes.edgecolor": DARK_GRID,
        "axes.labelcolor": DARK_TEXT,
        "axes.grid": True,
        "grid.color": DARK_GRID,
        "grid.alpha": 0.4,
        "text.color": DARK_TEXT,
        "xtick.color": DARK_TEXT,
        "ytick.color": DARK_TEXT,
        "legend.facecolor": DARK_FACE,
        "legend.edgecolor": DARK_GRID,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.facecolor": DARK_BG,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.prop_cycle": matplotlib.cycler(color=ACCENT_COLORS),
    })


# ── Safe Statistics ─────────────────────────────────────────

def safe_mean(vals: Sequence) -> Optional[float]:
    """Mean of finite values, or None."""
    finite = [float(v) for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(finite)) if finite else None


def safe_std(vals: Sequence) -> Optional[float]:
    """Sample std of finite values (ddof=1), or None if < 2 values."""
    finite = [float(v) for v in vals if v is not None and np.isfinite(v)]
    return float(np.std(finite, ddof=1)) if len(finite) >= 2 else None


def safe_min(vals: Sequence) -> Optional[float]:
    """Min of finite values, or None."""
    finite = [float(v) for v in vals if v is not None and np.isfinite(v)]
    return float(min(finite)) if finite else None


def safe_max(vals: Sequence) -> Optional[float]:
    """Max of finite values, or None."""
    finite = [float(v) for v in vals if v is not None and np.isfinite(v)]
    return float(max(finite)) if finite else None


def safe_percentile(vals: Sequence, q: float) -> Optional[float]:
    """Percentile of finite values, or None."""
    finite = [float(v) for v in vals if v is not None and np.isfinite(v)]
    return float(np.percentile(finite, q)) if finite else None


def safe_slope(vals: Sequence) -> Optional[float]:
    """
    Algebraic OLS slope preserving original index positions.
    
    Unlike np.polyfit, this uses cov/var directly: cheaper, no SVD,
    no RankWarning. Preserves the original position of each value
    (gaps from filtered NaN/inf are kept in x-space).
    """
    indexed = [(i, float(v)) for i, v in enumerate(vals)
               if v is not None and np.isfinite(v)]
    if len(indexed) < 3:
        return None
    x = np.array([i for i, _ in indexed], dtype=float)
    y = np.array([v for _, v in indexed], dtype=float)
    x_var = np.var(x, ddof=0)
    if x_var < 1e-12:
        return 0.0
    return float(np.cov(x, y, ddof=0)[0, 1] / x_var)


def safe_diff_std(vals: Sequence) -> Optional[float]:
    """Std of first differences — measures step-to-step volatility."""
    finite = [float(v) for v in vals if v is not None and np.isfinite(v)]
    if len(finite) < 3:
        return None
    diffs = np.diff(finite)
    return float(np.std(diffs, ddof=1)) if len(diffs) >= 2 else None


# ── IO Helpers ──────────────────────────────────────────────

def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it doesn't exist, return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_table(df: pd.DataFrame, path: Path, index: bool = False):
    """Save a DataFrame as CSV."""
    df.to_csv(path, index=index)


def save_json(obj: Any, path: Path):
    """Save a dict/list as formatted JSON, handling numpy/pandas types."""
    class _Encoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return None if np.isnan(o) else float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, pd.Timestamp):
                return o.isoformat()
            if isinstance(o, Path):
                return str(o)
            return super().default(o)

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, cls=_Encoder)


def save_focus_outputs(
    out_dir: Path,
    prefix: str,
    tables: Dict[str, pd.DataFrame],
    summary: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
):
    """
    Standard output writer for each analysis section.
    
    Saves:
      - tables/{prefix}_{name}.csv  for each table
      - summary.json                section summary (AI-oriented)
    """
    tables_dir = ensure_dir(out_dir / "tables")
    for name, df in tables.items():
        csv_path = tables_dir / f"{prefix}_{name}.csv"
        save_table(df, csv_path)
        if logger:
            logger.info(f"  Saved {csv_path.name} ({len(df)} rows)")

    save_json(summary, out_dir / "summary.json")


def save_figure(fig: plt.Figure, path: Path, close: bool = True):
    """Save a matplotlib figure and optionally close it."""
    fig.savefig(path)
    if close:
        plt.close(fig)


# ── Manifest Helpers ────────────────────────────────────────

def build_manifest(
    script_name: str,
    inputs: Dict[str, str],
    outputs: List[str],
    row_counts: Dict[str, int],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a standard manifest dict."""
    manifest = {
        "script": script_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs,
        "outputs": outputs,
        "row_counts": row_counts,
    }
    if extra:
        manifest.update(extra)
    return manifest