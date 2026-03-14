"""
Backfill actual_home_win outcomes for past predictions in predictions_history.db.

Reads game results from data/processed/team_game_logs.csv and matches them to
predictions where actual_home_win IS NULL and game_date is in the past.

Matching logic:
  - team_game_logs.csv uses one row per team per game
  - Home team rows have matchup like "TEAM vs. OPP"
  - Away team rows have matchup like "TEAM @ OPP"
  - We look up the home team row and read wl + pts + opponent pts (via plus_minus)

ATS outcome logic:
  - `spread` is not stored in game_predictions (no spread column exists yet)
  - ats_correct is left as None until a spread column is added
  - actual_margin = home_pts - away_pts (derived from plus_minus of home team row)

Run: python scripts/backfill_outcomes.py
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import logging

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "database" / "predictions_history.db"
TEAM_GAME_LOGS_PATH = PROJECT_ROOT / "data" / "processed" / "team_game_logs.csv"
BUILD_PERFORMANCE_SCRIPT = PROJECT_ROOT / "scripts" / "build_performance.py"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_team_game_logs(path: Path) -> pd.DataFrame:
    """Load team_game_logs.csv and return a DataFrame indexed for fast lookup."""
    df = pd.read_csv(path, low_memory=False)
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")
    df["game_date_str"] = df["game_date"].dt.strftime("%Y-%m-%d")
    return df


def build_home_game_index(logs: pd.DataFrame) -> dict[tuple[str, str, str], dict]:
    """
    Build a lookup dict keyed by (home_team_abbrev, away_team_abbrev, game_date_str).

    Home rows: matchup contains "vs." -> "TEAM vs. OPP"
    Away rows: matchup contains "@"   -> "TEAM @ OPP"

    We index by the home team row because that directly gives us:
      - home_pts  = row.pts
      - home_wl   = row.wl
      - actual_margin = row.plus_minus  (home_score - away_score)
    """
    index: dict[tuple[str, str, str], dict] = {}

    home_rows = logs[logs["matchup"].str.contains(r"\bvs\.", na=False, regex=True)]

    for _, row in home_rows.iterrows():
        # matchup = "HOME vs. AWAY"
        parts = row["matchup"].split(" vs. ")
        if len(parts) != 2:
            continue
        home_abbrev = parts[0].strip()
        away_abbrev = parts[1].strip()
        date_str = row["game_date_str"]
        key = (home_abbrev, away_abbrev, date_str)
        index[key] = {
            "home_wl": row["wl"],
            "home_pts": int(row["pts"]) if pd.notna(row["pts"]) else None,
            "actual_margin": int(row["plus_minus"]) if pd.notna(row["plus_minus"]) else None,
        }

    return index


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def fetch_pending_predictions(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return all game_predictions rows where actual_home_win IS NULL."""
    cur = conn.execute(
        """
        SELECT id, game_date, home_team, away_team, home_win_prob, away_win_prob
        FROM game_predictions
        WHERE actual_home_win IS NULL
        ORDER BY game_date
        """,
    )
    return cur.fetchall()


def update_prediction_outcome(
    conn: sqlite3.Connection,
    prediction_id: int,
    actual_home_win: int,
    actual_margin: int | None,
) -> None:
    """UPDATE a single prediction row with resolved outcome data."""
    conn.execute(
        """
        UPDATE game_predictions
        SET actual_home_win = ?
        WHERE id = ?
        """,
        (actual_home_win, prediction_id),
    )


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def parse_game_date(raw: str) -> date | None:
    """Parse game_date string from DB; returns None on failure."""
    try:
        return datetime.strptime(raw[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def is_past_game(game_date: date, today: date) -> bool:
    """Return True if game_date is strictly before today."""
    return game_date < today


# ---------------------------------------------------------------------------
# Core backfill logic
# ---------------------------------------------------------------------------

def backfill_outcomes(
    db_path: Path = DB_PATH,
    logs_path: Path = TEAM_GAME_LOGS_PATH,
    today: date | None = None,
) -> dict:
    """
    Match pending predictions against team_game_logs and fill in outcomes.

    Returns a summary dict with keys:
      total_pending, filled, still_pending, skipped_future, no_log_match
    """
    if today is None:
        today = date.today()

    log.info(f"Loading team game logs from {logs_path} ...")
    logs = load_team_game_logs(logs_path)
    home_index = build_home_game_index(logs)
    log.info(f"  {len(home_index):,} home game entries indexed")

    conn = open_db(db_path)
    try:
        pending = fetch_pending_predictions(conn)
        total_pending = len(pending)
        log.info(f"\nFound {total_pending} prediction(s) with no outcome recorded")

        filled = 0
        still_pending = 0
        skipped_future = 0
        no_log_match = 0

        for row in pending:
            game_date = parse_game_date(row["game_date"])
            if game_date is None:
                log.warning(f"  WARN: id={row['id']} has unparseable game_date={row['game_date']!r} -- skipping")
                still_pending += 1
                continue

            if not is_past_game(game_date, today):
                skipped_future += 1
                still_pending += 1
                continue

            home_team = row["home_team"]
            away_team = row["away_team"]
            date_str = game_date.strftime("%Y-%m-%d")
            key = (home_team, away_team, date_str)

            if key not in home_index:
                no_log_match += 1
                still_pending += 1
                log.warning(f"  WARN: No log entry for home={home_team} away={away_team} "
                    f"date={date_str} (id={row['id']}) -- game may not be in CSV yet")
                continue

            entry = home_index[key]
            actual_home_win = 1 if entry["home_wl"] == "W" else 0
            actual_margin = entry["actual_margin"]

            update_prediction_outcome(conn, row["id"], actual_home_win, actual_margin)
            filled += 1

            predicted_home_wins = (row["home_win_prob"] or 0.0) >= (row["away_win_prob"] or 0.0)
            correct_marker = "OK" if predicted_home_wins == bool(actual_home_win) else "WRONG"
            log.info(f"  id={row['id']} {home_team} vs {away_team} {date_str}: "
                f"home_win={actual_home_win} margin={actual_margin} [{correct_marker}]")

        conn.commit()

    finally:
        conn.close()

    summary = {
        "total_pending": total_pending,
        "filled": filled,
        "still_pending": still_pending,
        "skipped_future": skipped_future,
        "no_log_match": no_log_match,
    }

    log.warning(f"\nSummary: {total_pending} pending -> "
        f"{filled} filled, {skipped_future} future games, {no_log_match} not yet in CSV")
    return summary


# ---------------------------------------------------------------------------
# Post-backfill: regenerate performance.json
# ---------------------------------------------------------------------------

def run_build_performance() -> None:
    """Import and run build_performance() from build_performance.py."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_performance", BUILD_PERFORMANCE_SCRIPT)
    if spec is None or spec.loader is None:
        log.error("WARNING: Could not load build_performance.py -- skipping performance rebuild")
        return
    module = importlib.util.load_from_spec(spec)  # type: ignore[attr-defined]
    try:
        module.build_performance()
    except Exception as exc:
        log.error(f"WARNING: build_performance() raised {exc!r} -- dashboard data may be stale")


def _load_build_performance_module():
    """Load build_performance module dynamically; returns module or None."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_performance", BUILD_PERFORMANCE_SCRIPT)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod
    except Exception as exc:
        log.error(f"WARNING: Could not exec build_performance.py: {exc!r}")
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    summary = backfill_outcomes()

    if summary["filled"] > 0:
        log.info("\nRegenerating performance.json ...")
        mod = _load_build_performance_module()
        if mod is not None:
            try:
                mod.build_performance()
            except Exception as exc:
                log.error(f"WARNING: build_performance() raised {exc!r} -- dashboard data may be stale")
        else:
            log.error("WARNING: Could not load build_performance.py -- skipping performance rebuild")
    else:
        log.warning("\nNo new outcomes filled -- skipping performance rebuild")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
