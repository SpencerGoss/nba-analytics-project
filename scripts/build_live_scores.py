"""
build_live_scores.py -- produce dashboard/data/live_scores.json

Fetches today's NBA games from the live scoreboard API and enriches each game
with season records, spread info, and game status.

Output shape (wrapped object):
  {
    "games": [
      {
        "id": 1,
        "away": "BOS",
        "home": "NYK",
        "aS": null,
        "hS": null,
        "status": "upcoming",
        "q": "7:30 PM ET",
        "aR": "41-21",
        "hR": "40-22",
        "spread": "",
        "game_date": "2026-03-07"
      }
    ],
    "exported_at": "2026-03-07T..."
  }

Fallback: if the live scoreboard call fails, reads today's games from
database/predictions_history.db and returns them with status="upcoming".

Run: python scripts/build_live_scores.py
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TEAM_LOGS = PROJECT_ROOT / "data" / "processed" / "team_game_logs.csv"
GAME_LINES = PROJECT_ROOT / "data" / "processed" / "game_lines.csv"
PREDICTIONS_DB = PROJECT_ROOT / "database" / "predictions_history.db"
OUT_JSON = PROJECT_ROOT / "dashboard" / "data" / "live_scores.json"

CURRENT_SEASON = 202526


# ---------------------------------------------------------------------------
# Season record helper
# ---------------------------------------------------------------------------

def _build_season_records() -> dict[str, str]:
    """Return {abbreviation: 'W-L'} for all teams this season."""
    try:
        import pandas as pd
        df = pd.read_csv(TEAM_LOGS)
        df = df[df["season"] == CURRENT_SEASON].copy()
        wins = df.groupby("team_abbreviation")["wl"].apply(lambda s: (s == "W").sum())
        losses = df.groupby("team_abbreviation")["wl"].apply(lambda s: (s == "L").sum())
        records: dict[str, str] = {}
        for team in wins.index:
            w = int(wins.get(team, 0))
            lo = int(losses.get(team, 0))
            records[team] = f"{w}-{lo}"
        return records
    except Exception as exc:
        print(f"  WARN: could not build season records: {exc}")
        return {}


# ---------------------------------------------------------------------------
# Spread lookup
# ---------------------------------------------------------------------------

def _build_spread_lookup() -> dict[str, str]:
    """Return {game_id: spread_str} from game_lines.csv if it exists."""
    if not GAME_LINES.exists():
        return {}
    try:
        import pandas as pd
        df = pd.read_csv(GAME_LINES)
        if df.empty or "game_id" not in df.columns:
            return {}
        lookup: dict[str, str] = {}
        for _, row in df.iterrows():
            gid = str(row.get("game_id", ""))
            spread = row.get("home_spread", None)
            home = row.get("home_team", "")
            if gid and spread is not None:
                try:
                    s = float(spread)
                    sign = "+" if s > 0 else ""
                    lookup[gid] = f"{sign}{s} {home}"
                except (TypeError, ValueError):
                    pass
        return lookup
    except Exception as exc:
        print(f"  WARN: could not build spread lookup: {exc}")
        return {}


# ---------------------------------------------------------------------------
# Quarter / time display
# ---------------------------------------------------------------------------

def _period_label(period: int) -> str:
    """Convert period integer to display string."""
    if period == 1:
        return "1st"
    if period == 2:
        return "2nd"
    if period == 3:
        return "3rd"
    if period == 4:
        return "4th"
    if period > 4:
        return f"OT{period - 4}" if period > 5 else "OT"
    return f"Q{period}"


def _format_clock(game_clock: str | None) -> str:
    """Convert 'PT04M22.00S' or '4:22' to '4:22'."""
    if not game_clock:
        return ""
    # NBA live API returns ISO duration format like PT04M22.00S
    if game_clock.startswith("PT"):
        try:
            inner = game_clock[2:].rstrip("S")
            parts = inner.split("M")
            mins = int(parts[0])
            secs = int(float(parts[1])) if len(parts) > 1 else 0
            return f"{mins}:{secs:02d}"
        except Exception:
            pass
    return game_clock


# ---------------------------------------------------------------------------
# Live scoreboard fetch
# ---------------------------------------------------------------------------

def _fetch_live_scoreboard() -> list[dict]:
    """
    Fetch today's games from the NBA live scoreboard.
    Returns a list of game dicts in our target shape.
    """
    from nba_api.live.nba.endpoints.scoreboard import ScoreBoard  # type: ignore

    records = _build_season_records()
    spread_lookup = _build_spread_lookup()

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print("  Fetching live scoreboard from nba_api ...")
    time.sleep(0.6)
    board = ScoreBoard()
    games_raw = board.games.get_dict()

    results: list[dict] = []
    for idx, g in enumerate(games_raw, start=1):
        game_id = str(g.get("gameId", ""))
        game_date = g.get("gameEt", today_str)[:10]

        away_team = g.get("awayTeam", {})
        home_team = g.get("homeTeam", {})

        away_abbr = away_team.get("teamTricode", "")
        home_abbr = home_team.get("teamTricode", "")
        away_score_raw = away_team.get("score", None)
        home_score_raw = home_team.get("score", None)

        game_status_text = g.get("gameStatusText", "").strip()
        game_status = int(g.get("gameStatus", 1))

        # status: 1=upcoming, 2=live, 3=final
        if game_status == 3 or "final" in game_status_text.lower():
            status = "final"
            away_score = int(away_score_raw) if away_score_raw is not None else None
            home_score = int(home_score_raw) if home_score_raw is not None else None
            q_display = "Final"
        elif game_status == 2:
            status = "live"
            away_score = int(away_score_raw) if away_score_raw is not None else None
            home_score = int(home_score_raw) if home_score_raw is not None else None
            period = int(g.get("period", 1))
            game_clock = g.get("gameClock", "")
            clock_str = _format_clock(game_clock)
            q_display = f"{_period_label(period)} {clock_str}".strip()
        else:
            status = "upcoming"
            away_score = None
            home_score = None
            # game_status_text typically has tip-off time like "7:30 pm ET"
            # Normalise to "7:30 PM ET"
            q_display = game_status_text.replace("pm", "PM").replace("et", "ET").replace("PM ET", "PM ET")
            if not q_display:
                q_display = "TBD"

        spread_str = spread_lookup.get(game_id, "")

        results.append({
            "id": idx,
            "away": away_abbr,
            "home": home_abbr,
            "aS": away_score,
            "hS": home_score,
            "status": status,
            "q": q_display,
            "aR": records.get(away_abbr, ""),
            "hR": records.get(home_abbr, ""),
            "spread": spread_str,
            "game_date": game_date,
        })

    print(f"  Found {len(results)} games from live scoreboard")
    return results


# ---------------------------------------------------------------------------
# Fallback: predictions_history.db
# ---------------------------------------------------------------------------

def _fallback_from_db() -> list[dict]:
    """
    Build game list from today's (or most recent) predictions.
    All games returned as upcoming with null scores.
    """
    records = _build_season_records()
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if not PREDICTIONS_DB.exists():
        print("  WARN: predictions_history.db not found -- returning empty game list")
        return []

    conn = sqlite3.connect(str(PREDICTIONS_DB))
    try:
        # Try today first
        rows = conn.execute(
            "SELECT home_team, away_team, game_date FROM game_predictions "
            "WHERE game_date = ? ORDER BY id",
            (today_str,),
        ).fetchall()

        if not rows:
            # Fall back to most recent date in DB
            latest = conn.execute(
                "SELECT MAX(game_date) FROM game_predictions"
            ).fetchone()
            if latest and latest[0]:
                rows = conn.execute(
                    "SELECT home_team, away_team, game_date FROM game_predictions "
                    "WHERE game_date = ? ORDER BY id",
                    (latest[0],),
                ).fetchall()
                print(f"  Fallback: using predictions from {latest[0]}")
    finally:
        conn.close()

    results: list[dict] = []
    for idx, (home, away, gdate) in enumerate(rows, start=1):
        results.append({
            "id": idx,
            "away": away or "",
            "home": home or "",
            "aS": None,
            "hS": None,
            "status": "upcoming",
            "q": "TBD",
            "aR": records.get(away or "", ""),
            "hR": records.get(home or "", ""),
            "spread": "",
            "game_date": gdate or today_str,
        })

    print(f"  Fallback: {len(results)} games from predictions DB")
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_live_scores(out_path: Path = OUT_JSON) -> dict:
    """Fetch live scores and write to JSON. Returns the output dict."""
    try:
        games = _fetch_live_scoreboard()
    except Exception as exc:
        print(f"  ERROR fetching live scoreboard: {exc}")
        print("  Falling back to predictions_history.db ...")
        games = _fallback_from_db()

    exported_at = datetime.now(timezone.utc).isoformat()
    output = {
        "games": games,
        "exported_at": exported_at,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, default=str)

    print(f"  Written -> {out_path}  ({len(games)} games)")
    return output


if __name__ == "__main__":
    print("=== build_live_scores ===")
    build_live_scores()
