"""
build_injuries.py  --  produce dashboard/data/injuries.json

Reads player_absences.csv (current season), cross-references with
dashboard/data/todays_picks.json to surface injury impact per upcoming game.

Player avg pts come from data/processed/player_stats.csv (current season).
Team names resolved via data/processed/teams.csv.

Output schema: list of game objects (one per game in todays_picks),
each with home_injuries / away_injuries lists.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

ABSENCES_CSV = PROJECT_ROOT / "data" / "processed" / "player_absences.csv"
PLAYER_STATS_CSV = PROJECT_ROOT / "data" / "processed" / "player_stats.csv"
TEAMS_CSV = PROJECT_ROOT / "data" / "processed" / "teams.csv"
PICKS_JSON = PROJECT_ROOT / "dashboard" / "data" / "todays_picks.json"
OUT_JSON = PROJECT_ROOT / "dashboard" / "data" / "injuries.json"

from src.config import get_current_season

CURRENT_SEASON = get_current_season()
# How many most-recent game dates to look back for absences
LOOKBACK_DATES = 3

HIGH_IMPACT_PTS = 15.0
MED_IMPACT_PTS = 8.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _impact_label(avg_pts: float) -> str:
    if avg_pts >= HIGH_IMPACT_PTS:
        return "high"
    if avg_pts >= MED_IMPACT_PTS:
        return "medium"
    return "low"


def _spread_note(injuries: list[dict]) -> str:
    high_count = sum(1 for p in injuries if p["impact"] == "high")
    med_count = sum(1 for p in injuries if p["impact"] == "medium")
    if high_count >= 2:
        return f"{high_count} high-impact players out -- significant spread effect expected"
    if high_count == 1:
        name = next(p["player"] for p in injuries if p["impact"] == "high")
        return f"Key player out: {name} -- monitor line movement"
    if med_count > 0:
        return f"{med_count} medium-impact player(s) out -- minor spread effect"
    return "No key injuries affecting this game"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_absences(season: int = CURRENT_SEASON, lookback: int = LOOKBACK_DATES) -> pd.DataFrame:
    """Return rows for absent players in the most recent `lookback` game dates."""
    df = pd.read_csv(ABSENCES_CSV)
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")
    df = df[(df["season"] == season) & (df["was_absent"] == 1)].copy()

    if df.empty:
        return df

    latest_dates = sorted(df["game_date"].unique())[-lookback:]
    return df[df["game_date"].isin(latest_dates)].copy()


def load_player_stats(season: int = CURRENT_SEASON) -> pd.DataFrame:
    """Return player scoring averages for the current season."""
    df = pd.read_csv(PLAYER_STATS_CSV, usecols=["player_id", "player_name", "team_abbreviation", "pts", "gp", "season"])
    df = df[df["season"] == season].copy()
    # pts column is season total; compute per-game average
    df = df.copy()
    df["avg_pts"] = df["pts"] / df["gp"].replace(0, float("nan"))
    return df[["player_id", "player_name", "team_abbreviation", "avg_pts"]].copy()


def load_team_id_map() -> dict[int, str]:
    """Return mapping team_id (int) -> abbreviation."""
    df = pd.read_csv(TEAMS_CSV, usecols=["team_id", "abbreviation"])
    return dict(zip(df["team_id"].astype(int), df["abbreviation"]))


def load_picks() -> list[dict]:
    if not PICKS_JSON.exists():
        return []
    with PICKS_JSON.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, list) else []


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_injuries(
    absences: pd.DataFrame | None = None,
    player_stats: pd.DataFrame | None = None,
    team_id_map: dict[int, str] | None = None,
    picks: list[dict] | None = None,
) -> list[dict]:
    if absences is None:
        absences = load_absences()
    if player_stats is None:
        player_stats = load_player_stats()
    if team_id_map is None:
        team_id_map = load_team_id_map()
    if picks is None:
        picks = load_picks()

    # Map team_id -> abbreviation in absences
    absences = absences.copy()
    absences["team_abbr"] = absences["team_id"].map(team_id_map)

    # Merge avg_pts onto absences using player_id
    if player_stats.empty or "player_id" not in player_stats.columns:
        stats_lookup: dict[int, float] = {}
    else:
        stats_lookup = player_stats.set_index("player_id")["avg_pts"].to_dict()
    absences["avg_pts"] = absences["player_id"].map(stats_lookup).fillna(0.0)

    # Deduplicate: keep the most recent absence per player
    absences = absences.sort_values("game_date")
    absences = absences.drop_duplicates(subset=["player_id", "team_abbr"], keep="last")

    # Build a lookup: team_abbr -> list of injury dicts
    team_injury_map: dict[str, list[dict]] = {}
    for _, row in absences.iterrows():
        abbr = row["team_abbr"]
        if pd.isna(abbr):
            continue
        avg_pts = float(row["avg_pts"])
        entry = {
            "player": str(row["player_name"]),
            "status": "OUT",
            "position": "",          # not available in absences CSV
            "season_avg_pts": round(avg_pts, 1),
            "impact": _impact_label(avg_pts),
        }
        team_injury_map.setdefault(str(abbr), []).append(entry)

    # Sort each team's list by avg_pts descending
    for abbr in team_injury_map:
        team_injury_map[abbr].sort(key=lambda x: x["season_avg_pts"], reverse=True)

    # Assemble per-game output using picks for game context
    results: list[dict] = []
    seen_pairs: set[tuple[str, str, str]] = set()

    for pick in picks:
        home = str(pick.get("home_team", ""))
        away = str(pick.get("away_team", ""))
        game_date = str(pick.get("game_date", ""))
        key = (game_date, home, away)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)

        home_inj = team_injury_map.get(home, [])
        away_inj = team_injury_map.get(away, [])
        all_inj = home_inj + away_inj

        results.append(
            {
                "game_date": game_date,
                "home_team": home,
                "away_team": away,
                "home_injuries": home_inj,
                "away_injuries": away_inj,
                "spread_impact_note": _spread_note(all_inj),
            }
        )

    # If no picks, emit one entry per team that has injuries (for testing)
    if not picks:
        teams_with_injuries = sorted(team_injury_map.keys())
        for team in teams_with_injuries:
            inj = team_injury_map[team]
            results.append(
                {
                    "game_date": "",
                    "home_team": team,
                    "away_team": "",
                    "home_injuries": inj,
                    "away_injuries": [],
                    "spread_impact_note": _spread_note(inj),
                }
            )

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading absences...")
    absences = load_absences()
    print(f"  {len(absences)} recent absent player-game rows")

    player_stats = load_player_stats()
    print(f"  {len(player_stats)} player stat rows (season {CURRENT_SEASON})")

    team_id_map = load_team_id_map()
    picks = load_picks()
    print(f"  {len(picks)} picks in todays_picks.json")

    results = build_injuries(absences, player_stats, team_id_map, picks)
    print(f"  Built injury report for {len(results)} games")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, separators=(",", ":"), ensure_ascii=False)

    print(f"  Written -> {OUT_JSON}")
    for game in results[:3]:
        h_count = len(game["home_injuries"])
        a_count = len(game["away_injuries"])
        print(f"  {game['away_team']} @ {game['home_team']}  "
              f"home_inj={h_count}  away_inj={a_count}  note={game['spread_impact_note'][:60]}")


if __name__ == "__main__":
    main()
