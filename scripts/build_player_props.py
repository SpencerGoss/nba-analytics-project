"""
scripts/build_player_props.py

Generates dashboard/data/player_props.json from season averages and game logs.

For each active player with significant minutes (>=20 mpg season avg, >=15 games):
  - Season average: PTS, REB, AST, 3PM, STL, BLK
  - Last-5-game average for same stats
  - Trend = last5_avg - season_avg
  - model_projection = season avg rounded to nearest 0.5
  - Value flag: last5_avg > season_avg by >15% on any stat
  - Opponent: resolved from dashboard/data/todays_picks.json if player's team plays today
  - is_injured: flagged from player_absences.csv for today's date

Run: python scripts/build_player_props.py
Output: dashboard/data/player_props.json
Top 80 players by season minutes, sorted by pts descending.
"""

import json
import logging
import math
import sys
from datetime import date
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLAYER_STATS_PATH = PROJECT_ROOT / "data" / "processed" / "player_stats.csv"
PLAYER_LOGS_PATH = PROJECT_ROOT / "data" / "processed" / "player_game_logs.csv"
ABSENCES_PATH = PROJECT_ROOT / "data" / "processed" / "player_absences.csv"
PICKS_PATH = PROJECT_ROOT / "dashboard" / "data" / "todays_picks.json"
OUTPUT_PATH = PROJECT_ROOT / "dashboard" / "data" / "player_props.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CURRENT_SEASON = 202526
CURRENT_SEASON_ID = 22025

STAT_COLS = ["pts", "reb", "ast", "fg3m", "stl", "blk"]
STAT_LABELS = ["PTS", "REB", "AST", "3PM", "STL", "BLK"]

MIN_MPG = 20.0       # minimum minutes per game (season avg)
MIN_GAMES = 15       # minimum games played
TOP_N = 80           # max players to output
LAST_N = 5           # last-N games for trend
VALUE_PCT = 0.15     # last5 must exceed season avg by this fraction


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def round_to_half(value: float) -> float:
    """Round a float to the nearest 0.5."""
    return math.floor(value * 2 + 0.5) / 2


def compute_trend_str(last5_avg: float, season_avg: float) -> str:
    diff = last5_avg - season_avg
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.1f}"


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_season_stats() -> pd.DataFrame:
    df = pd.read_csv(PLAYER_STATS_PATH)
    df = df[df["season"] == CURRENT_SEASON].copy()
    for col in STAT_COLS + ["gp", "min"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Stats are season totals — divide by gp to get per-game averages
    gp = df["gp"].replace(0, pd.NA)
    for col in STAT_COLS:
        df[col] = df[col] / gp
    # mpg = total minutes / gp
    df["mpg"] = df["min"] / gp
    return df


def load_player_logs() -> pd.DataFrame:
    df = pd.read_csv(PLAYER_LOGS_PATH)
    df = df[df["season_id"] == CURRENT_SEASON_ID].copy()
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")
    for col in STAT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_opponent_map() -> dict[str, str]:
    """Return {team_abbr: opponent_abbr} for today's games from todays_picks.json."""
    if not PICKS_PATH.exists():
        log.warning("todays_picks.json not found — opponent field will be empty")
        return {}
    with open(PICKS_PATH, encoding="utf-8") as fh:
        picks = json.load(fh)
    opponent_map: dict[str, str] = {}
    for game in picks:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        if home and away:
            opponent_map[home] = away
            opponent_map[away] = home
    return opponent_map


def load_absent_player_ids() -> set[int]:
    """Return player_ids flagged as absent for today's date."""
    if not ABSENCES_PATH.exists():
        return set()
    today = pd.Timestamp(date.today())
    df = pd.read_csv(ABSENCES_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")
    mask = (df["was_absent"] == 1) & (df["game_date"] == today)
    return set(df.loc[mask, "player_id"].tolist())


# ---------------------------------------------------------------------------
# Per-player prop builder
# ---------------------------------------------------------------------------

def last_n_values(player_logs: pd.DataFrame, col: str, n: int = LAST_N) -> list[float | None]:
    vals = player_logs.sort_values("game_date")[col].dropna().tail(n).tolist()
    return [round(float(v), 1) for v in vals]


def build_props_for_player(
    player_id: int,
    player_name: str,
    team: str,
    season_row: pd.Series,
    player_logs: pd.DataFrame,
    opponent_map: dict[str, str],
    absent_ids: set[int],
) -> dict | None:
    """Build a player prop entry. Returns None if insufficient data."""
    opponent = opponent_map.get(team, "")
    is_injured = player_id in absent_ids

    props: list[dict] = []
    any_value = False

    for col, label in zip(STAT_COLS, STAT_LABELS):
        season_avg = season_row.get(col)
        if season_avg is None or pd.isna(season_avg):
            continue

        recent = last_n_values(player_logs, col)
        last5_avg = (sum(v for v in recent if v is not None) / len(recent)) if recent else None

        if last5_avg is not None:
            trend_str = compute_trend_str(last5_avg, float(season_avg))
            value = last5_avg > float(season_avg) * (1 + VALUE_PCT)
            model_projection = round(float(last5_avg), 1)
        else:
            trend_str = "0.0"
            value = False
            model_projection = round(float(season_avg), 1)

        if value:
            any_value = True

        book_line = round_to_half(float(season_avg))

        props.append({
            "stat": label,
            "model_projection": model_projection,
            "book_line": book_line,
            "edge": round(model_projection - book_line, 1),
            "recommendation": "OVER" if model_projection > book_line else "UNDER",
            "value": value,
            "trend": trend_str,
            "season_avg": round(float(season_avg), 1),
            "last5": recent,
        })

    if not props:
        return None

    return {
        "player_name": player_name,
        "player_id": player_id,
        "team": team,
        "opponent": opponent,
        "is_injured": is_injured,
        "has_value": any_value,
        "props": props,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("Loading season stats (season=%s)...", CURRENT_SEASON)
    stats = load_season_stats()

    # Apply filters: minimum games and minutes per game
    stats = stats[
        (stats["gp"] >= MIN_GAMES) &
        (stats["mpg"] >= MIN_MPG)
    ].copy()
    log.info("%d players pass filters (gp>=%d, mpg>=%.0f)", len(stats), MIN_GAMES, MIN_MPG)

    if stats.empty:
        log.error("No players passed filters — check CURRENT_SEASON constant (%s)", CURRENT_SEASON)
        sys.exit(1)

    # Sort by pts descending, take top N
    stats = stats.sort_values("pts", ascending=False).head(TOP_N)
    log.info("Top %d players selected (sorted by pts)", len(stats))

    log.info("Loading player game logs (season_id=%s)...", CURRENT_SEASON_ID)
    logs = load_player_logs()
    log.info("Loaded %d player-game rows", len(logs))

    opponent_map = load_opponent_map()
    log.info("Today's games: %d teams with opponents mapped", len(opponent_map))

    absent_ids = load_absent_player_ids()
    log.info("Absent/injured players today: %d", len(absent_ids))

    all_props: list[dict] = []

    for _, row in stats.iterrows():
        player_id = int(row["player_id"])
        player_name = str(row["player_name"])
        team = str(row["team_abbreviation"])

        player_logs = logs[logs["player_id"] == player_id].copy()

        entry = build_props_for_player(
            player_id=player_id,
            player_name=player_name,
            team=team,
            season_row=row,
            player_logs=player_logs,
            opponent_map=opponent_map,
            absent_ids=absent_ids,
        )
        if entry is not None:
            all_props.append(entry)

    log.info("Built prop entries for %d players", len(all_props))

    output = {
        "exported_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "season": CURRENT_SEASON,
        "player_count": len(all_props),
        "players": all_props,
    }

    # The dashboard JS expects a flat array at the top level (window.PLAYER_PROPS = propsJson)
    # so we write the players list directly, and embed metadata as a leading sentinel object
    # Actually: dashboard does Array.isArray(propsJson) check — write flat array only
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(all_props, fh, indent=2, ensure_ascii=False)

    log.info(
        "Wrote %d player prop entries -> %s",
        len(all_props),
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
