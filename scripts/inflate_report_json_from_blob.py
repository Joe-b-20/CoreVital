#!/usr/bin/env python3
# =============================================================================
# Inflate report_blob → report_json for Datasette (and other JSON-based tools)
#
# CoreVital can store reports as gzip blob only (compress=True). Datasette
# run-detail dashboards need report_json for json_extract(). This script
# decompresses report_blob and writes to report_json for affected rows.
#
# Usage:
#   python scripts/inflate_report_json_from_blob.py docs/demo/corevital_demo.db
# =============================================================================

import gzip
import sqlite3
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/inflate_report_json_from_blob.py <db_path>", file=sys.stderr)
        sys.exit(1)
    db_path = Path(sys.argv[1])
    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "SELECT trace_id, report_blob FROM reports WHERE report_blob IS NOT NULL AND (report_json IS NULL OR report_json = '')"
        )
        rows = cur.fetchall()
    if not rows:
        print("No rows to inflate (all reports already have report_json).")
        return

    updated = 0
    with sqlite3.connect(db_path) as conn:
        for trace_id, blob in rows:
            try:
                json_str = gzip.decompress(blob).decode("utf-8")
            except Exception as e:
                print(f"Skip {trace_id[:8]}: decompress failed: {e}", file=sys.stderr)
                continue
            conn.execute(
                "UPDATE reports SET report_json = ? WHERE trace_id = ?",
                (json_str, trace_id),
            )
            updated += 1
        conn.commit()

    print(f"Inflated report_json for {updated} row(s). You can now use Datasette run-detail dashboards.")


if __name__ == "__main__":
    main()
