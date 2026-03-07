"""
scripts/build_totals.py

Generates dashboard/data/totals.json.

For each game in dashboard/data/todays_picks.json:
  - Model total = home_team rolling-10 avg pts scored
                + away_team rolling-10 avg pts allowed
  - Pace adjustment: if pace_game_roll10 available in matchup features,
    scale by combined pace vs league average
  - Book total: pulled from game_matchup_features if available (no Pinnacle
    totals API yet)
  - Value flag: abs(model_total - book_total) > VALUE_THRESHOLD and
    model_total direction confirmed

Run: python scripts/build_totals.py
Output: dashboard/data/totals.json
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PICKS_PATH = PROJECT_ROOT / "dashboard" / "data" / "todays_picks.json"
TEAM_LOGS_PATH = PROJECT_ROOT / "data" / "processed" / "team_game_logs.csv"
MATCHUP_PATH = PROJECT_ROOT / "data" / "features" / "game_matchup_features.csv"
OUTPUT_PATH = PROJECT_ROOT / "dashboard" / "data" / "totals.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROLL_WINDOW = 10
MIN_GAMES_REQUIRED = 3
VALUE_THRESHOLD = 3.0   # model must differ from book by at least this many pts
LEAGUE_AVG_PACE = 99.5  # approximate NBA average possessions per game (2024-25)

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


def load_team_logs() -> pd.DataFrame:
    df = pd.read_csv(TEAM_LOGS_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")
    df["pts"] = pd.to_numeric(df["pts"], errors="coerce")
    return df


def derive_opp_pts(team_logs: pd.DataFrame) -> pd.DataFrame:
    """
    team_game_logs has no opp_pts column.  Derive it by joining each team
    against its opponent in the same game (via game_id + pts).

    Returns a DataFrame with columns:
        team_id, team_abbreviation, game_id, game_date, pts, opp_pts, season
    """
    slim = team_logs[
        ["team_id", "team_abbreviation", "game_id", "game_date", "pts", "season"]
    ].copy()
    # Self-join on game_id where team_id differs
    opp = slim[["game_id", "team_abbreviation", "pts"]].rename(
        columns={"team_abbreviation": "opp_abbr", "pts": "opp_pts"}
    )
    merged = slim.merge(opp, on="game_id", how="inner")
    merged = merged[merged["team_abbreviation"] != merged["opp_abbr"]].copy()
    return merged


def load_matchup_features() -> pd.DataFrame | None:
    if not MATCHUP_PATH.exists():
        return None
    df = pd.read_csv(MATCHUP_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")
    return df


# ---------------------------------------------------------------------------
# Rolling computations  (shift(1) enforced — no same-game leakage)
# ---------------------------------------------------------------------------

def team_rolling_stats(
    team_abbr: str,
    df: pd.DataFrame,
    season: int,
) -> dict[str, float | None]:
    """
    Returns rolling-10 avg pts scored and pts allowed for `team_abbr`
    in `season`, computed with shift(1).
    """
    mask = (df["team_abbreviation"] == team_abbr) & (df["season"] == season)
    team_df = df[mask].sort_values("game_date").copy()

    if len(team_df) < MIN_GAMES_REQUIRED:
        return {"avg_scored": None, "avg_allowed": None, "avg_pace": None}

    pts_shifted = team_df["pts"].shift(1)
    opp_shifted = team_df["opp_pts"].shift(1)

    avg_scored_series = pts_shifted.rolling(ROLL_WINDOW, min_periods=MIN_GAMES_REQUIRED).mean()
    avg_allowed_series = opp_shifted.rolling(ROLL_WINDOW, min_periods=MIN_GAMES_REQUIRED).mean()

    avg_scored = avg_scored_series.iloc[-1]
    avg_allowed = avg_allowed_series.iloc[-1]

    return {
        "avg_scored": round(float(avg_scored), 1) if pd.notna(avg_scored) else None,
        "avg_allowed": round(float(avg_allowed), 1) if pd.notna(avg_allowed) else None,
        "avg_pace": None,  # populated separately from matchup features if available
    }


def get_pace_from_matchup(
    matchup_df: pd.DataFrame | None,
    home_team: str,
    away_team: str,
) -> tuple[float | None, float | None]:
    """
    Extract the most recent home/away pace_game_roll10 for this matchup.
    Returns (home_pace, away_pace).
    """
    if matchup_df is None:
        return None, None

    mask = (matchup_df["home_team"] == home_team) & (matchup_df["away_team"] == away_team)
    relevant = matchup_df[mask].sort_values("game_date", ascending=False)
    if relevant.empty:
        # Try finding individual team pace from any game
        home_mask = matchup_df["home_team"] == home_team
        away_mask = matchup_df["away_team"] == away_team
        home_rows = matchup_df[home_mask].sort_values("game_date", ascending=False)
        away_rows = matchup_df[away_mask].sort_values("game_date", ascending=False)
        home_pace = (
            home_rows["home_pace_game_roll10"].iloc[0]
            if not home_rows.empty and "home_pace_game_roll10" in home_rows.columns
            else None
        )
        away_pace = (
            away_rows["away_pace_game_roll10"].iloc[0]
            if not away_rows.empty and "away_pace_game_roll10" in away_rows.columns
            else None
        )
        return (
            float(home_pace) if home_pace is not None and pd.notna(home_pace) else None,
            float(away_pace) if away_pace is not None and pd.notna(away_pace) else None,
        )

    row = relevant.iloc[0]
    home_pace = row.get("home_pace_game_roll10")
    away_pace = row.get("away_pace_game_roll10")
    return (
        float(home_pace) if home_pace is not None and pd.notna(home_pace) else None,
        float(away_pace) if away_pace is not None and pd.notna(away_pace) else None,
    )


def apply_pace_adjustment(
    raw_total: float,
    home_pace: float | None,
    away_pace: float | None,
) -> float:
    """
    Scale total by combined game pace relative to league average.
    avg_game_pace = (home_pace + away_pace) / 2
    adjustment_factor = avg_game_pace / LEAGUE_AVG_PACE
    """
    if home_pace is None or away_pace is None:
        return raw_total
    avg_game_pace = (home_pace + away_pace) / 2.0
    if avg_game_pace <= 0:
        return raw_total
    factor = avg_game_pace / LEAGUE_AVG_PACE
    return raw_total * factor


# ---------------------------------------------------------------------------
# Per-game logic
# ---------------------------------------------------------------------------

def build_game_total(
    game: dict,
    enriched_logs: pd.DataFrame,
    matchup_df: pd.DataFrame | None,
    current_season: int,
) -> dict | None:
    home_team = game["home_team"]
    away_team = game["away_team"]
    game_date_str = game["game_date"]

    home_stats = team_rolling_stats(home_team, enriched_logs, current_season)
    away_stats = team_rolling_stats(away_team, enriched_logs, current_season)

    home_scored = home_stats["avg_scored"]
    away_allowed = away_stats["avg_allowed"]
    away_scored = away_stats["avg_scored"]
    home_allowed = home_stats["avg_allowed"]

    # Model projection: home pts + away pts allowed (from away team's perspective)
    if home_scored is None or away_allowed is None:
        log.warning(
            "Insufficient data for %s vs %s — skipping total", home_team, away_team
        )
        return None

    raw_total = home_scored + away_allowed

    # Pace adjustment
    home_pace, away_pace = get_pace_from_matchup(matchup_df, home_team, away_team)
    model_total = apply_pace_adjustment(raw_total, home_pace, away_pace)
    model_total = round(model_total, 1)

    # Book total: not available from current data sources; null for now
    book_total: float | None = None

    edge = round(model_total - book_total, 1) if book_total is not None else None
    value = edge is not None and abs(edge) > VALUE_THRESHOLD

    if value and edge is not None:
        recommendation = "OVER" if edge > 0 else "UNDER"
    else:
        recommendation = None

    return {
        "home_team": home_team,
        "away_team": away_team,
        "game_date": game_date_str,
        "model_total": model_total,
        "book_total": book_total,
        "edge": edge,
        "recommendation": recommendation,
        "value": value,
        "home_avg_scored": home_scored,
        "away_avg_allowed": away_allowed,
        "away_avg_scored": away_scored,
        "home_avg_allowed": home_allowed,
        "home_pace_roll10": round(home_pace, 1) if home_pace is not None else None,
        "away_pace_roll10": round(away_pace, 1) if away_pace is not None else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("Loading data...")
    games = load_todays_games()
    if not games:
        log.error("No games found in todays_picks.json — aborting")
        sys.exit(1)

    raw_logs = load_team_logs()
    enriched = derive_opp_pts(raw_logs)
    current_season = int(raw_logs["season"].max())
    log.info(
        "Loaded %d team-game rows; current season %s", len(enriched), current_season
    )

    matchup_df = load_matchup_features()
    if matchup_df is not None:
        log.info("Loaded matchup features (%d rows)", len(matchup_df))
    else:
        log.warning("Matchup features not found — pace adjustment disabled")

    totals: list[dict] = []
    for game in games:
        result = build_game_total(game, enriched, matchup_df, current_season)
        if result is not None:
            totals.append(result)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(totals, fh, indent=2, ensure_ascii=False)

    log.info(
        "Wrote %d game totals -> %s", len(totals), OUTPUT_PATH
    )


if __name__ == "__main__":
    main()
