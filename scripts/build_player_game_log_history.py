"""
Build on-demand game log history for a specific player + season.

This script fetches PlayerGameLog data for one player/season combination
when a user drills into a season in the comparison tool.  It is intentionally
NOT run as part of the batch pipeline — per-player game logs require one API
call per player-season, which is prohibitively slow for the full historical
corpus (~500k calls).

Usage:
  python scripts/build_player_game_log_history.py --player-id 2544 --season 2003-04
  python scripts/build_player_game_log_history.py --player-id 2544  # all available seasons

Output:
  dashboard/data/game_logs/player_{player_id}_{season_int}.json
  e.g.  dashboard/data/game_logs/player_2544_200304.json

JSON format:
  {
    "player_id": 2544,
    "player_name": "LeBron James",
    "season": "2003-04",
    "season_int": 200304,
    "games": [
      {
        "game_id": "0020300001",
        "game_date": "2003-10-29",
        "matchup": "CLE vs. SAC",
        "wl": "W",
        "min": 42, "pts": 25, "reb": 6, "ast": 9,
        "stl": 2, "blk": 0, "tov": 4,
        "fg_pct": 0.444, "fg3_pct": 0.333, "ft_pct": 0.750,
        "plus_minus": 12
      }
    ]
  }

Rate limit: 0.6s between calls — never removed. All-season mode adds a
2s pause between seasons to stay well within NBA API tolerances.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import logging

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "dashboard" / "data" / "game_logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

THROTTLE_SEC = 0.7
BETWEEN_SEASON_SEC = 2.0

# Seasons for which PlayerGameLog is reliably available via nba_api
GAME_LOG_FIRST_SEASON = "1996-97"
GAME_LOG_LAST_SEASON = "2024-25"


# ---------------------------------------------------------------------------
# Season helpers
# ---------------------------------------------------------------------------

def season_str_to_int(season_str: str) -> int:
    """'2003-04' -> 200304"""
    start, end_yy = season_str.split("-")
    return int(start) * 100 + int(end_yy)


def available_seasons(first: str = GAME_LOG_FIRST_SEASON,
                       last: str = GAME_LOG_LAST_SEASON) -> list[str]:
    """Return all season strings from first to last inclusive."""
    first_year = int(first.split("-")[0])
    last_year = int(last.split("-")[0])
    seasons = []
    for year in range(first_year, last_year + 1):
        end_yy = (year + 1) % 100
        seasons.append(f"{year}-{end_yy:02d}")
    return seasons


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_game_log(player_id: int, season_str: str) -> pd.DataFrame | None:
    """
    Fetch one player's game log for one season.
    Returns None on failure (after one retry).
    """
    from nba_api.stats.endpoints import PlayerGameLog

    kwargs = dict(
        player_id=player_id,
        season=season_str,
        season_type_all_star="Regular Season",
    )

    try:
        df = PlayerGameLog(**kwargs).get_data_frames()[0]
        time.sleep(THROTTLE_SEC)
        return df
    except Exception as exc:
        log.error(f"  WARNING: first attempt failed (player={player_id}, season={season_str}): {exc}")
        time.sleep(2.0)
        try:
            df = PlayerGameLog(**kwargs).get_data_frames()[0]
            time.sleep(THROTTLE_SEC)
            return df
        except Exception as exc2:
            log.error(f"  ERROR: skipping after retry: {exc2}")
            return None


def _resolve_player_name(player_id: int) -> str:
    """Look up player name from nba_api static data."""
    try:
        from nba_api.stats.static import players as nba_players
        results = nba_players.find_player_by_id(player_id)
        if results:
            return results.get("full_name", f"Player {player_id}")
    except Exception:
        pass
    return f"Player {player_id}"


def _normalize_game_row(row: pd.Series) -> dict:
    """Convert a PlayerGameLog row to the output dict format."""
    def g(col, decimals=1):
        val = row.get(col)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        try:
            return round(float(val), decimals)
        except (TypeError, ValueError):
            return val

    return {
        "game_id": str(row.get("Game_ID") or row.get("game_id", "")),
        "game_date": str(row.get("GAME_DATE") or row.get("game_date", "")),
        "matchup": str(row.get("MATCHUP") or row.get("matchup", "")),
        "wl": str(row.get("WL") or row.get("wl", "")),
        "min": g("MIN", 0),
        "pts": g("PTS", 0),
        "reb": g("REB", 0),
        "ast": g("AST", 0),
        "stl": g("STL", 0),
        "blk": g("BLK", 0),
        "tov": g("TOV", 0),
        "fgm": g("FGM", 0),
        "fga": g("FGA", 0),
        "fg_pct": g("FG_PCT", 4),
        "fg3m": g("FG3M", 0),
        "fg3a": g("FG3A", 0),
        "fg3_pct": g("FG3_PCT", 4),
        "ftm": g("FTM", 0),
        "fta": g("FTA", 0),
        "ft_pct": g("FT_PCT", 4),
        "plus_minus": g("PLUS_MINUS", 0),
    }


def build_season_log(player_id: int, season_str: str) -> dict | None:
    """
    Fetch and structure one player-season game log.
    Writes JSON to OUT_DIR.  Returns the output dict or None on failure.
    """
    season_int = season_str_to_int(season_str)
    out_path = OUT_DIR / f"player_{player_id}_{season_int}.json"

    if out_path.exists():
        log.warning(f"  {season_str}: already exists, skipping.")
        return None

    log.info(f"  Fetching game log: player_id={player_id} season={season_str}...", end=" ", flush=True)
    df = fetch_game_log(player_id, season_str)

    if df is None or df.empty:
        log.warning("no data")
        return None

    # Normalise column names for lookup
    df.columns = [c.upper() for c in df.columns]

    player_name = _resolve_player_name(player_id)
    games = [_normalize_game_row(row) for _, row in df.iterrows()]

    output = {
        "player_id": player_id,
        "player_name": player_name,
        "season": season_str,
        "season_int": season_int,
        "games": games,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, separators=(",", ":"))

    log.info(f"done ({len(games)} games) -> {out_path.name}")
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Fetch game log history for a single player (on-demand, not batch)."
    )
    p.add_argument("--player-id", type=int, required=True,
                   help="NBA player ID, e.g. 2544 for LeBron James")
    p.add_argument("--season", default=None, metavar="YYYY-YY",
                   help="Specific season to fetch, e.g. 2003-04. "
                        "Omit to fetch all available seasons.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-fetch even if output file already exists")
    args = p.parse_args(argv)

    player_id = args.player_id

    if args.season:
        seasons = [args.season]
    else:
        seasons = available_seasons()
        log.info(f"build_player_game_log_history: fetching all {len(seasons)} seasons "
              f"for player_id={player_id}")
        log.info("  NOTE: this will take several minutes due to API rate limits.")

    if args.overwrite:
        # Remove existing files so build_season_log won't skip them
        for s in seasons:
            out_path = OUT_DIR / f"player_{player_id}_{season_str_to_int(s)}.json"
            if out_path.exists():
                out_path.unlink()

    fetched = 0
    for i, season_str in enumerate(seasons):
        build_season_log(player_id, season_str)
        fetched += 1
        if len(seasons) > 1 and i < len(seasons) - 1:
            time.sleep(BETWEEN_SEASON_SEC)

    log.info(f"build_player_game_log_history: done. {fetched} seasons processed.")
    log.info(f"  Output directory: {OUT_DIR}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
