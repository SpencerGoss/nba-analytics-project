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
from datetime import datetime, timezone
from pathlib import Path

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
                conn.execute(
                    f"""
                    INSERT OR IGNORE INTO {CLV_TABLE}
                    (game_date, home_team, away_team,
                     opening_spread, opening_home_ml, opening_away_ml, logged_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (game_date, home_team, away_team,
                     opening_spread, opening_home_ml, opening_away_ml, now),
                )
                inserted = conn.total_changes > 0
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
