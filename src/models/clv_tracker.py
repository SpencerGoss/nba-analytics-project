"""
CLV (Closing Line Value) Tracker
=================================
Tracks opening and closing spreads to compute CLV for each predicted game.

CLV = opening_spread - closing_spread (home team perspective).
Positive CLV = we got a better line than where the market settled
(e.g., we logged home at -3.5, it closed at -5.5 -> CLV = +2.0, we got the easier cover).

Usage:
    from src.models.clv_tracker import CLVTracker
    tracker = CLVTracker()
    tracker.log_opening_line(game_date, home_team, away_team, opening_spread, home_ml, away_ml)
    tracker.update_closing_line(game_date, home_team, away_team, closing_spread)
    clv = tracker.get_clv_summary()
"""

import sqlite3
import logging
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "database" / "predictions_history.db"

CLV_TABLE = "clv_tracking"


class CLVTracker:
    """Manages opening/closing line logging and CLV computation."""

    def __init__(self, db_path: str | Path = DB_PATH):
        self.db_path = Path(db_path)
        self._ensure_table()

    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_table(self):
        """Create clv_tracking table if it does not exist."""
        with self._connect() as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {CLV_TABLE} (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_date        TEXT NOT NULL,
                    home_team        TEXT NOT NULL,
                    away_team        TEXT NOT NULL,
                    opening_spread   REAL,
                    opening_home_ml  INTEGER,
                    opening_away_ml  INTEGER,
                    closing_spread   REAL,
                    clv              REAL,
                    logged_at        TEXT NOT NULL,
                    updated_at       TEXT,
                    UNIQUE(game_date, home_team, away_team)
                )
            """)
            conn.commit()

    def log_opening_line(
        self,
        game_date: str,
        home_team: str,
        away_team: str,
        opening_spread: float | None,
        opening_home_ml: int | None = None,
        opening_away_ml: int | None = None,
    ) -> bool:
        """Log the opening line for a game. Skips if already logged.

        Args:
            game_date: ISO date string "YYYY-MM-DD".
            home_team: 3-letter abbreviation (e.g. "LAL").
            away_team: 3-letter abbreviation (e.g. "GSW").
            opening_spread: Home team spread (negative = home favored).
            opening_home_ml: Home moneyline (American odds, optional).
            opening_away_ml: Away moneyline (American odds, optional).

        Returns:
            True if a new row was inserted, False if row already existed.
        """
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    f"""
                    INSERT OR IGNORE INTO {CLV_TABLE}
                    (game_date, home_team, away_team,
                     opening_spread, opening_home_ml, opening_away_ml, logged_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (game_date, home_team, away_team,
                     opening_spread, opening_home_ml, opening_away_ml, now),
                )
                inserted = cursor.rowcount > 0
                conn.commit()
            if inserted:
                log.info(
                    f"CLV: logged opening line {home_team} vs {away_team} on {game_date}: "
                    f"spread={opening_spread}"
                )
            return inserted
        except sqlite3.Error as e:
            log.error(f"CLV: failed to log opening line: {e}")
            return False

    def update_closing_line(
        self,
        game_date: str,
        home_team: str,
        away_team: str,
        closing_spread: float,
    ) -> float | None:
        """Update the closing line and compute CLV for a game.

        CLV = opening_spread - closing_spread (positive = we got a better line than closing).
        Only updates rows that already have an opening_spread logged.

        Returns:
            CLV value (float) if successfully computed, else None.
        """
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._connect() as conn:
                row = conn.execute(
                    f"""
                    SELECT opening_spread FROM {CLV_TABLE}
                    WHERE game_date=? AND home_team=? AND away_team=?
                    """,
                    (game_date, home_team, away_team),
                ).fetchone()

                if row is None:
                    log.warning(
                        f"CLV: no opening line found for {home_team} vs {away_team} on {game_date}"
                    )
                    return None

                opening = row["opening_spread"]
                if opening is None:
                    log.warning(
                        f"CLV: opening_spread is NULL for {home_team} vs {away_team} on {game_date}"
                    )
                    return None

                # CLV = opening_spread - closing_spread (home team perspective).
                # Example: opening=-3.5, closing=-5.5 -> CLV=+2.0 (we logged home at -3.5,
                # market settled at -5.5; we got the easier cover -> positive CLV = edge confirmed).
                # Example: opening=-5.5, closing=-3.5 -> CLV=-2.0 (we got the harder line).
                clv = round(float(opening) - float(closing_spread), 2)
                conn.execute(
                    f"""
                    UPDATE {CLV_TABLE}
                    SET closing_spread=?, clv=?, updated_at=?
                    WHERE game_date=? AND home_team=? AND away_team=?
                    """,
                    (closing_spread, clv, now, game_date, home_team, away_team),
                )
                conn.commit()
            log.info(
                f"CLV: {home_team} vs {away_team} on {game_date}: "
                f"opening={opening:+.1f} closing={closing_spread:+.1f} CLV={clv:+.2f}"
            )
            return clv
        except sqlite3.Error as e:
            log.error(f"CLV: failed to update closing line: {e}")
            return None

    def get_clv_summary(self, min_games: int = 10) -> dict:
        """Return CLV statistics for all games with computed CLV.

        Returns dict with:
            n_games: number of games with CLV computed
            mean_clv: average CLV (positive = good)
            positive_clv_rate: fraction of games where CLV > 0
            has_edge: True if n_games >= min_games and mean_clv > 0 and pos_rate > 0.5
            data: list of dicts for each game
        """
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    f"""
                    SELECT game_date, home_team, away_team,
                           opening_spread, closing_spread, clv
                    FROM {CLV_TABLE}
                    WHERE clv IS NOT NULL
                    ORDER BY game_date DESC
                    """
                ).fetchall()

            data = [dict(r) for r in rows]
            n = len(data)
            if n == 0:
                return {
                    "n_games": 0,
                    "mean_clv": None,
                    "positive_clv_rate": None,
                    "has_edge": False,
                    "data": [],
                }

            clvs = [r["clv"] for r in data]
            mean_clv = round(sum(clvs) / n, 3)
            pos_rate = round(sum(1 for c in clvs if c > 0) / n, 3)
            return {
                "n_games": n,
                "mean_clv": mean_clv,
                "positive_clv_rate": pos_rate,
                "has_edge": n >= min_games and mean_clv > 0 and pos_rate > 0.5,
                "data": data,
            }
        except sqlite3.Error as e:
            log.error(f"CLV: failed to get summary: {e}")
            return {
                "n_games": 0,
                "mean_clv": None,
                "positive_clv_rate": None,
                "has_edge": False,
                "data": [],
            }


def log_todays_opening_lines(db_path: str | Path = DB_PATH) -> int:
    """Fetch today's game lines from Pinnacle and log them as opening lines.

    This is called from fetch_odds.py after game_lines.csv is generated.
    Returns the number of new lines logged.
    """
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    try:
        from scripts.fetch_odds import fetch_game_lines
    except ImportError:
        log.error("CLV: could not import fetch_game_lines from scripts.fetch_odds")
        return 0

    lines = fetch_game_lines()
    if lines.empty:
        log.warning("CLV: no game lines returned from Pinnacle")
        return 0

    tracker = CLVTracker(db_path)
    n_logged = 0
    for _, row in lines.iterrows():
        logged = tracker.log_opening_line(
            game_date=str(row.get("date", "")),
            home_team=str(row.get("home_team", "")),
            away_team=str(row.get("away_team", "")),
            opening_spread=row.get("spread"),
            opening_home_ml=row.get("home_moneyline"),
            opening_away_ml=row.get("away_moneyline"),
        )
        if logged:
            n_logged += 1

    log.info(f"CLV: logged {n_logged} new opening lines (of {len(lines)} fetched)")
    return n_logged


def backfill_closing_lines(
    db_path: str | Path = DB_PATH,
    game_lines_path: str | Path | None = None,
    today: date | None = None,
) -> int:
    """Update closing spreads for past games using the last-fetched game_lines.csv.

    The Pinnacle guest API only returns lines for upcoming games. Once a game
    tips off, it disappears from the API. The daily pipeline fetches odds and
    writes game_lines.csv *before* games start, so the spread in that file
    represents the near-closing line (last available before tip-off).

    This function reads game_lines.csv and, for each game whose date is in
    the past (already played) and whose clv_tracking row has no closing_spread,
    writes the spread from game_lines.csv as the closing line and computes CLV.

    Should be called at the START of the daily pipeline, BEFORE fetch_odds.py
    overwrites game_lines.csv with fresh data for today's games.

    Args:
        db_path: Path to predictions_history.db.
        game_lines_path: Path to data/odds/game_lines.csv. Defaults to
            PROJECT_ROOT / "data" / "odds" / "game_lines.csv".
        today: Override for today's date (for testing). Defaults to date.today().

    Returns:
        Number of closing lines updated.
    """
    if game_lines_path is None:
        game_lines_path = PROJECT_ROOT / "data" / "odds" / "game_lines.csv"
    game_lines_path = Path(game_lines_path)

    if today is None:
        today = date.today()

    if not game_lines_path.exists():
        log.warning("CLV backfill: game_lines.csv not found at %s", game_lines_path)
        return 0

    try:
        lines_df = pd.read_csv(game_lines_path)
    except Exception as e:
        log.error("CLV backfill: failed to read game_lines.csv: %s", e)
        return 0

    if lines_df.empty:
        log.info("CLV backfill: game_lines.csv is empty, nothing to backfill")
        return 0

    tracker = CLVTracker(db_path)
    n_updated = 0

    for _, row in lines_df.iterrows():
        game_date_str = str(row.get("date", ""))
        if not game_date_str or len(game_date_str) < 10:
            continue

        # Only update games that have already been played
        try:
            game_date = date.fromisoformat(game_date_str[:10])
        except ValueError:
            continue

        if game_date >= today:
            continue  # game hasn't happened yet

        home_team = str(row.get("home_team", ""))
        away_team = str(row.get("away_team", ""))
        spread_raw = row.get("spread")

        # Guard: skip if spread is null/NaN
        if pd.isna(spread_raw):
            log.debug(
                "CLV backfill: spread is NULL for %s vs %s on %s -- skipping",
                home_team, away_team, game_date_str,
            )
            continue

        closing_spread = float(spread_raw)

        # Check if this game already has a closing line in DB
        try:
            with tracker._connect() as conn:
                existing = conn.execute(
                    f"""
                    SELECT closing_spread FROM {CLV_TABLE}
                    WHERE game_date=? AND home_team=? AND away_team=?
                    """,
                    (game_date_str[:10], home_team, away_team),
                ).fetchone()

            if existing is None:
                # No opening line logged for this game -- skip
                log.debug(
                    "CLV backfill: no opening line for %s vs %s on %s",
                    home_team, away_team, game_date_str,
                )
                continue

            if existing["closing_spread"] is not None:
                # Already has a closing line -- skip
                continue
        except sqlite3.Error as e:
            log.error("CLV backfill: DB error checking existing row: %s", e)
            continue

        clv = tracker.update_closing_line(
            game_date_str[:10], home_team, away_team, closing_spread
        )
        if clv is not None:
            n_updated += 1
            print(
                f"  CLV backfill: {home_team} vs {away_team} {game_date_str[:10]} "
                f"-> closing={closing_spread:+.1f} CLV={clv:+.2f}"
            )

    log.info("CLV backfill: updated %d closing line(s)", n_updated)
    if n_updated > 0:
        print(f"  CLV backfill: {n_updated} closing line(s) updated")
    else:
        print("  CLV backfill: no new closing lines to update")

    return n_updated
