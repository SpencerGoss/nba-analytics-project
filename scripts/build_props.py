"""
scripts/build_props.py

Generates dashboard/data/player_props.json.

For each game in dashboard/data/todays_picks.json:
  - Find the top 3-5 players per team (by rolling avg minutes)
  - Compute 10-game rolling averages for PTS, REB, AST, 3PM, STL, BLK
  - Flag players listed in player_absences.csv as injured/absent
  - Book line: null (Pinnacle player-props API not yet integrated)
  - Value flag: model_projection > book_line + 1.5  (no flag when book_line is null)

Run: python scripts/build_props.py
Output: dashboard/data/player_props.json
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PICKS_PATH = PROJECT_ROOT / "dashboard" / "data" / "todays_picks.json"
PLAYER_LOGS_PATH = PROJECT_ROOT / "data" / "processed" / "player_game_logs.csv"
ABSENCES_PATH = PROJECT_ROOT / "data" / "processed" / "player_absences.csv"
OUTPUT_PATH = PROJECT_ROOT / "dashboard" / "data" / "player_props.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STAT_COLS = ["pts", "reb", "ast", "fg3m", "stl", "blk"]
STAT_LABELS = ["PTS", "REB", "AST", "3PM", "STL", "BLK"]
ROLL_WINDOW = 10
LAST_N_GAMES = 5
TOP_N_PER_TEAM = 5
VALUE_THRESHOLD = 1.5  # projection must exceed book line by this to flag value
MIN_GAMES_REQUIRED = 3  # player must have at least this many games to include


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_todays_games() -> list[dict]:
    if not PICKS_PATH.exists():
        log.warning("todays_picks.json not found at %s", PICKS_PATH)
        return []
    with open(PICKS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def load_player_logs() -> pd.DataFrame:
    df = pd.read_csv(PLAYER_LOGS_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")
    df["min_num"] = pd.to_numeric(df["min"], errors="coerce").fillna(0.0)
    for col in STAT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_absent_player_ids(game_date: pd.Timestamp) -> set[int]:
    """Return set of player_ids flagged absent on or before game_date."""
    if not ABSENCES_PATH.exists():
        return set()
    df = pd.read_csv(ABSENCES_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")
    mask = (df["was_absent"] == 1) & (df["game_date"] == game_date)
    return set(df.loc[mask, "player_id"].tolist())


# ---------------------------------------------------------------------------
# Rolling projections
# ---------------------------------------------------------------------------

def compute_player_rolling(
    player_df: pd.DataFrame,
) -> dict[str, float | None]:
    """Compute rolling 10-game average for each stat using shift(1) to prevent leakage."""
    if len(player_df) < MIN_GAMES_REQUIRED:
        return {}
    df = player_df.sort_values("game_date").copy()
    result: dict[str, float | None] = {}
    for col in STAT_COLS:
        series = df[col].shift(1).rolling(ROLL_WINDOW, min_periods=MIN_GAMES_REQUIRED)
        val = series.mean().iloc[-1]
        result[col] = round(float(val), 1) if pd.notna(val) else None
    return result


def last_n_values(player_df: pd.DataFrame, col: str, n: int = LAST_N_GAMES) -> list[float | None]:
    df = player_df.sort_values("game_date").copy()
    vals = df[col].dropna().tail(n).tolist()
    return [round(float(v), 1) for v in vals]


# ---------------------------------------------------------------------------
# Per-game logic
# ---------------------------------------------------------------------------

def build_game_props(
    game: dict,
    current_season_logs: pd.DataFrame,
    absent_player_ids: set[int],
) -> list[dict]:
    """Build prop projections for top players in one game."""
    home_team = game["home_team"]
    away_team = game["away_team"]
    game_date_str = game["game_date"]

    results: list[dict] = []

    for team, opponent in [(home_team, away_team), (away_team, home_team)]:
        team_logs = current_season_logs[
            current_season_logs["team_abbreviation"] == team
        ].copy()

        if team_logs.empty:
            log.warning("No logs found for team %s", team)
            continue

        # Rank players by average minutes played to find starters
        player_avg_min = (
            team_logs.groupby(["player_id", "player_name"])["min_num"]
            .mean()
            .reset_index()
            .sort_values("min_num", ascending=False)
        )
        top_players = player_avg_min.head(TOP_N_PER_TEAM)

        for _, row in top_players.iterrows():
            player_id = int(row["player_id"])
            player_name = str(row["player_name"])
            is_injured = player_id in absent_player_ids

            player_df = team_logs[team_logs["player_id"] == player_id]
            rolling = compute_player_rolling(player_df)

            if not rolling:
                log.debug("Skipping %s — insufficient game history", player_name)
                continue

            props: list[dict] = []
            for col, label in zip(STAT_COLS, STAT_LABELS):
                projection = rolling.get(col)
                if projection is None:
                    continue
                recent = last_n_values(player_df, col)
                book_line: float | None = None  # Pinnacle props not yet available
                edge = (
                    round(projection - book_line, 1)
                    if book_line is not None
                    else None
                )
                value = (
                    edge is not None and edge > VALUE_THRESHOLD
                )
                recommendation = (
                    "OVER" if (value and edge is not None and edge > 0) else None
                )
                props.append({
                    "stat": label,
                    "model_projection": projection,
                    "book_line": book_line,
                    "edge": edge,
                    "recommendation": recommendation,
                    "value": value,
                    "last5": recent,
                })

            results.append({
                "player_name": player_name,
                "player_id": player_id,
                "team": team,
                "opponent": opponent,
                "game_date": game_date_str,
                "is_injured": is_injured,
                "props": props,
            })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("Loading data...")
    games = load_todays_games()
    if not games:
        log.error("No games found in todays_picks.json — aborting")
        sys.exit(1)

    logs = load_player_logs()
    current_season = int(logs["season"].max())
    current_logs = logs[logs["season"] == current_season].copy()
    log.info(
        "Loaded %d player-game rows for season %s", len(current_logs), current_season
    )

    # Use the game_date from the first game for absent lookup
    game_dates = sorted({g["game_date"] for g in games})
    latest_date = pd.Timestamp(game_dates[-1])
    absent_ids = load_absent_player_ids(latest_date)
    log.info("Found %d absent players on %s", len(absent_ids), latest_date.date())

    all_props: list[dict] = []
    for game in games:
        game_props = build_game_props(game, current_logs, absent_ids)
        all_props.extend(game_props)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(all_props, fh, indent=2, ensure_ascii=False)

    player_count = len(all_props)
    game_count = len(games)
    log.info(
        "Wrote %d player projections across %d games -> %s",
        player_count,
        game_count,
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
