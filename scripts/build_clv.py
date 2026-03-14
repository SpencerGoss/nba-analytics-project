"""
build_clv.py -- regenerate dashboard/data/clv_summary.json

Reads all rows from the clv_tracking table in database/predictions_history.db
and computes summary statistics for Closing Line Value (CLV) performance.

CLV formula: clv = opening_spread - closing_spread
  positive CLV = we got a better line than where the market closed (good)

Output JSON schema:
  {
    "mean_clv":       float   -- mean of non-NULL clv rows (0.0 if none)
    "pos_rate":       float   -- % of non-NULL clv rows where clv > 0 (0.0 if none)
    "games_tracked":  int     -- total rows in clv_tracking
    "games_with_clv": int     -- rows where clv IS NOT NULL
    "edge_confirmed": bool    -- true if games_with_clv >= 10 AND pos_rate > 50
    "last_updated":   str     -- ISO-8601 UTC timestamp
  }

Run: python scripts/build_clv.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
import logging

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "database" / "predictions_history.db"
OUT_JSON = PROJECT_ROOT / "dashboard" / "data" / "clv_summary.json"

# Minimum closed games required before declaring edge confirmed
MIN_CLOSED_GAMES = 10
# Minimum positive-CLV rate (%) to declare edge confirmed
MIN_POS_RATE = 50.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_clv_rows(db_path: Path) -> list[dict]:
    """Return all rows from clv_tracking as a list of dicts."""
    if not db_path.exists():
        log.warning(f"  WARN: DB not found at {db_path}")
        return []

    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("SELECT * FROM clv_tracking")
        col_names = [desc[0] for desc in cur.description]
        rows = [dict(zip(col_names, row)) for row in cur.fetchall()]
        conn.close()
        return rows
    except sqlite3.OperationalError as exc:
        log.error(f"  WARN: could not query clv_tracking: {exc}")
        return []


# ---------------------------------------------------------------------------
# Stats computation
# ---------------------------------------------------------------------------

def _compute_summary(rows: list[dict]) -> dict:
    """
    Compute CLV summary stats from raw DB rows.
    Guards all closing_spread / clv casts with pd.isna() equivalent checks.
    """
    games_tracked = len(rows)

    # Collect valid (non-NULL) CLV values using explicit isna guard
    clv_values: list[float] = []
    for row in rows:
        raw_clv = row.get("clv")
        # Guard: skip NULL / None / empty string before float cast
        if raw_clv is None:
            continue
        try:
            val = float(raw_clv)
        except (TypeError, ValueError):
            continue
        clv_values.append(val)

    games_with_clv = len(clv_values)

    if games_with_clv == 0:
        mean_clv = 0.0
        pos_rate = 0.0
    else:
        mean_clv = round(sum(clv_values) / games_with_clv, 4)
        pos_count = sum(1 for v in clv_values if v > 0)
        pos_rate = round((pos_count / games_with_clv) * 100, 2)

    edge_confirmed = bool(
        games_with_clv >= MIN_CLOSED_GAMES and pos_rate > MIN_POS_RATE
    )

    return {
        "mean_clv": mean_clv,
        "pos_rate": pos_rate,
        "games_tracked": games_tracked,
        "games_with_clv": games_with_clv,
        "edge_confirmed": edge_confirmed,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_clv(
    db_path: Path = DB_PATH,
    out_path: Path = OUT_JSON,
) -> dict:
    """
    Main entry point. Returns the summary dict (also writes JSON).
    """
    log.info(f"build_clv: reading clv_tracking from {db_path}")
    rows = _load_clv_rows(db_path)
    log.info(f"  {len(rows)} total rows loaded")

    summary = _compute_summary(rows)

    log.info(f"  games_tracked  : {summary['games_tracked']}")
    log.info(f"  games_with_clv : {summary['games_with_clv']}")
    log.info(f"  mean_clv       : {summary['mean_clv']}")
    log.info(f"  pos_rate       : {summary['pos_rate']}%")
    log.info(f"  edge_confirmed : {summary['edge_confirmed']}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    log.info(f"Written -> {out_path}")
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_clv()
