"""
fetch_live_scores.py -- write dashboard/data/live_scores.json

Uses nba_api.live.nba.endpoints.scoreboard for live/today's scores.
Falls back to an empty array if the API is unavailable or no games are live.

Output schema per game:
  {
    "game_id": "0022500800",
    "home_team": "LAC", "away_team": "DET",
    "home_score": 87, "away_score": 72,
    "status": "In Progress", "period": 3, "clock": "4:32",
    "status_code": 2
  }

status_code meanings (Scoreboard API convention):
  1 = Pre-Game, 2 = In Progress, 3 = Final

Run: python scripts/fetch_live_scores.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import logging

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUT_JSON = PROJECT_ROOT / "dashboard" / "data" / "live_scores.json"

# NBA API throttle: never call without sleep between requests
NBA_API_SLEEP = 0.6


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

_STATUS_MAP: dict[int, str] = {
    1: "Pre-Game",
    2: "In Progress",
    3: "Final",
}


def _status_label(code: int) -> str:
    return _STATUS_MAP.get(code, "Unknown")


# ---------------------------------------------------------------------------
# Live scoreboard fetch
# ---------------------------------------------------------------------------

def _parse_live_game(game: dict) -> dict:
    """
    Parse a single game dict from the live Scoreboard API.
    Handles missing keys gracefully with safe defaults.
    """
    game_id = str(game.get("gameId", ""))

    home_team_data = game.get("homeTeam", {})
    away_team_data = game.get("awayTeam", {})

    home_abbr = str(home_team_data.get("teamTricode", ""))
    away_abbr = str(away_team_data.get("teamTricode", ""))
    home_score = int(home_team_data.get("score", 0) or 0)
    away_score = int(away_team_data.get("score", 0) or 0)

    status_code = int(game.get("gameStatus", 1) or 1)
    status_text = str(game.get("gameStatusText", _status_label(status_code))).strip()
    period = int(game.get("period", 0) or 0)
    clock = str(game.get("gameClock", "") or "").strip()

    # Normalise clock from "PT04M32.00S" format to "4:32" if needed
    clock = _normalise_clock(clock)

    return {
        "game_id": game_id,
        "home_team": home_abbr,
        "away_team": away_abbr,
        "home_score": home_score,
        "away_score": away_score,
        "status": status_text,
        "period": period,
        "clock": clock,
        "status_code": status_code,
    }


def _normalise_clock(raw: str) -> str:
    """
    Convert ISO 8601 duration clock string "PT04M32.00S" to "4:32".
    If not in that format, return the raw string unchanged.
    """
    if not raw or not raw.startswith("PT"):
        return raw
    try:
        # Strip "PT" prefix and parse minutes/seconds
        inner = raw[2:]  # e.g. "04M32.00S"
        if "M" in inner:
            m_part, s_part = inner.split("M")
            minutes = int(m_part)
            seconds = int(float(s_part.rstrip("S")))
            return f"{minutes}:{seconds:02d}"
    except (ValueError, IndexError):
        pass
    return raw


def fetch_live_scores() -> list[dict]:
    """
    Fetch today's live scores from the NBA live scoreboard API.
    Returns a list of game dicts. Returns [] on any error.
    """
    try:
        from nba_api.live.nba.endpoints import scoreboard  # type: ignore
        board = scoreboard.ScoreBoard()
        time.sleep(NBA_API_SLEEP)
        games_raw = board.games.get_dict()
        games = [_parse_live_game(g) for g in games_raw]
        return games
    except ImportError:
        log.warning("  WARN: nba_api not installed -- returning empty live scores")
        return []
    except Exception as exc:
        log.info(f"  Live scores unavailable: {exc}")
        return []


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_live_scores(out_path: Path = OUT_JSON) -> list[dict]:
    """Fetch live scores and write to JSON. Always writes (even empty list)."""
    log.info("Fetching live scores from NBA API ...")
    games = fetch_live_scores()

    if games:
        live = [g for g in games if g["status_code"] == 2]
        log.info(f"  {len(games)} game(s) today, {len(live)} currently live")
    else:
        log.info("  No games found or API unavailable")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "games": games,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    log.info(f"Written -> {out_path}  ({len(games)} games)")
    return games


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_live_scores()
