"""
build_standings.py -- produce dashboard/data/standings.json

Builds full NBA conference standings for the current season from
data/processed/team_game_logs.csv.

Per-team stats computed:
  - W, L, Win%
  - Home record (W-L)
  - Away record (W-L)
  - Last 10 SU record
  - Current streak (+N win / -N loss)
  - Games behind conference leader
  - ATS record (null -- no spread data in game logs)
  - O/U record (null -- no totals data yet)

Run: python scripts/build_standings.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TEAM_LOGS = PROJECT_ROOT / "data" / "processed" / "team_game_logs.csv"
TEAMS_CSV = PROJECT_ROOT / "data" / "processed" / "teams.csv"
OUT_JSON = PROJECT_ROOT / "dashboard" / "data" / "standings.json"

from src.config import get_current_season, EAST_DIVISIONS, WEST_DIVISIONS

CURRENT_SEASON = get_current_season()
LAST_N = 10
TOTAL_GAMES = 82  # NBA regular season length

# Playoff structure: seeds 1-6 direct, 7-10 play-in, 11-15 out
PLAYOFF_SEEDS = 6
PLAYIN_SEEDS = 10

# Conference / division mappings from config
EAST = EAST_DIVISIONS
WEST = WEST_DIVISIONS

# Flat lookup: abbreviation -> (conference, division)
_CONF_DIV: dict[str, tuple[str, str]] = {}
for _div, _teams in EAST.items():
    for _t in _teams:
        _CONF_DIV[_t] = ("East", _div)
for _div, _teams in WEST.items():
    for _t in _teams:
        _CONF_DIV[_t] = ("West", _div)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record_str(wins: int, losses: int) -> str:
    from scripts.builder_helpers import record_str
    return record_str(wins, losses)


def _current_streak(wl_series: pd.Series) -> str:
    """
    Return streak string like '+3' or '-2'.
    wl_series is sorted oldest-first; last entry is most recent.
    """
    if wl_series.empty:
        return "+0"
    values = wl_series.tolist()
    last = values[-1]
    count = 0
    for val in reversed(values):
        if val == last:
            count += 1
        else:
            break
    if last == "W":
        return f"+{count}"
    return f"-{count}"


def _games_behind(leader_wins: int, leader_losses: int, team_wins: int, team_losses: int) -> float:
    from scripts.builder_helpers import games_behind
    return games_behind(leader_wins, leader_losses, team_wins, team_losses)


def _compute_clinch_status(rows: list[dict]) -> None:
    """
    Mutate rows in-place to add 'clinch' field based on mathematical
    elimination / clinching logic.

    Clinch values:
      'z'  - clinched #1 seed (conference)
      'x'  - clinched playoff berth (top 6)
      'pi' - clinched play-in spot (top 10)
      'e'  - eliminated from play-in (cannot reach top 10)
      'o'  - eliminated from playoffs (cannot reach top 6, but may reach play-in)
      None - still in contention, nothing clinched
    """
    for row in rows:
        row["clinch"] = None
        row["games_remaining"] = TOTAL_GAMES - row["w"] - row["l"]

    # Max possible wins for each team
    for row in rows:
        row["_max_wins"] = row["w"] + row["games_remaining"]

    # --- Clinch #1 seed: team's current wins > every other team's max wins ---
    first = rows[0]
    if len(rows) > 1:
        second_max = max(r["_max_wins"] for r in rows[1:])
        if first["w"] > second_max:
            first["clinch"] = "z"

    # --- Clinched playoff (top 6): team's wins > 7th place team's max wins ---
    if len(rows) > PLAYOFF_SEEDS:
        seventh_max = rows[PLAYOFF_SEEDS]["_max_wins"]
        for row in rows[:PLAYOFF_SEEDS]:
            if row["clinch"] is None and row["w"] > seventh_max:
                row["clinch"] = "x"

    # --- Clinched play-in (top 10): team's wins > 11th place team's max wins ---
    if len(rows) > PLAYIN_SEEDS:
        eleventh_max = rows[PLAYIN_SEEDS]["_max_wins"]
        for row in rows[:PLAYIN_SEEDS]:
            if row["clinch"] is None and row["w"] > eleventh_max:
                row["clinch"] = "pi"

    # --- Eliminated from play-in: team's max wins < 10th place current wins ---
    if len(rows) >= PLAYIN_SEEDS:
        tenth_wins = rows[PLAYIN_SEEDS - 1]["w"]
        for row in rows[PLAYIN_SEEDS:]:
            if row["_max_wins"] < tenth_wins:
                row["clinch"] = "e"

    # --- Eliminated from direct playoff (top 6): max wins < 6th place wins ---
    if len(rows) >= PLAYOFF_SEEDS:
        sixth_wins = rows[PLAYOFF_SEEDS - 1]["w"]
        for row in rows[PLAYOFF_SEEDS:PLAYIN_SEEDS]:
            if row["clinch"] is None and row["_max_wins"] < sixth_wins:
                row["clinch"] = "o"

    # Clean up temp field
    for row in rows:
        del row["_max_wins"]


def _load_team_names() -> dict[str, str]:
    from scripts.builder_helpers import load_team_names
    return load_team_names(TEAMS_CSV)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_season_logs(season: int = CURRENT_SEASON) -> pd.DataFrame:
    """Load team_game_logs.csv filtered to the given season."""
    df = pd.read_csv(TEAM_LOGS)
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")
    df = df[df["season"] == season].copy()
    df["is_home"] = df["matchup"].str.contains(r" vs\. ", regex=True)
    df["win"] = df["wl"].str.upper() == "W"
    df = df.sort_values(["team_abbreviation", "game_date"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Per-team computation
# ---------------------------------------------------------------------------

def compute_team_record(team_logs: pd.DataFrame) -> dict:
    """
    Given the sorted (oldest-first) game log rows for a single team,
    return a dict of standing stats.
    """
    total_w = int(team_logs["win"].sum())
    total_l = int((~team_logs["win"]).sum())
    total_games = total_w + total_l
    win_pct = round(total_w / total_games, 3) if total_games > 0 else 0.0

    home_rows = team_logs[team_logs["is_home"]]
    away_rows = team_logs[~team_logs["is_home"]]
    home_w = int(home_rows["win"].sum())
    home_l = int((~home_rows["win"]).sum())
    away_w = int(away_rows["win"].sum())
    away_l = int((~away_rows["win"]).sum())

    last10 = team_logs.tail(LAST_N)
    l10_w = int(last10["win"].sum())
    l10_l = len(last10) - l10_w

    streak = _current_streak(team_logs["wl"].reset_index(drop=True))

    return {
        "w": total_w,
        "l": total_l,
        "win_pct": win_pct,
        "home_record": _record_str(home_w, home_l),
        "away_record": _record_str(away_w, away_l),
        "last10": _record_str(l10_w, l10_l),
        "streak": streak,
    }


def build_conference_standings(
    logs: pd.DataFrame,
    team_names: dict[str, str],
    conf: str,
    divisions: dict[str, list[str]],
) -> list[dict]:
    """
    Compute standings rows for all teams in one conference.
    Returns list sorted by wins desc, then losses asc (standard NBA order).
    """
    all_teams = [t for teams in divisions.values() for t in teams]
    rows: list[dict] = []

    for team in all_teams:
        division = next(
            (div for div, tlist in divisions.items() if team in tlist), "Unknown"
        )
        team_logs = logs[logs["team_abbreviation"] == team].sort_values("game_date")

        if team_logs.empty:
            # Team has no games yet -- zero record
            record = {
                "w": 0, "l": 0, "win_pct": 0.0,
                "home_record": "0-0", "away_record": "0-0",
                "last10": "0-0", "streak": "+0",
            }
        else:
            record = compute_team_record(team_logs)

        rows.append({
            "team": team,
            "team_name": team_names.get(team, team),
            "conference": conf,
            "division": division,
            **record,
            "games_behind": 0.0,  # filled in after sorting
            "ats_record": None,
            "ou_record": None,
        })

    # Sort: most wins first; ties broken by fewest losses, then alphabetical
    rows.sort(key=lambda r: (-r["w"], r["l"], r["team"]))

    # Assign rank and games behind the leader (rank-1 team)
    leader_w = rows[0]["w"] if rows else 0
    leader_l = rows[0]["l"] if rows else 0

    for i, row in enumerate(rows):
        row["rank"] = i + 1
        row["games_behind"] = _games_behind(leader_w, leader_l, row["w"], row["l"])

    # Compute clinch / elimination status
    _compute_clinch_status(rows)

    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_standings(
    logs_path: Path = TEAM_LOGS,
    teams_path: Path = TEAMS_CSV,
    out_path: Path = OUT_JSON,
    season: int = CURRENT_SEASON,
) -> dict:
    """Main entry point. Returns the standings dict (also writes JSON)."""
    if not logs_path.exists():
        print(f"WARN: team_game_logs not found: {logs_path} -- skipping standings build")
        return {}

    print(f"Loading game logs from {logs_path} ...")
    logs = load_season_logs(season)
    print(f"  {len(logs)} rows for season {season}")

    team_names: dict[str, str] = {}
    if teams_path.exists():
        team_names = _load_team_names()
    else:
        print(f"  WARN: teams.csv not found at {teams_path} -- using abbreviations only")

    print("Computing East standings ...")
    east = build_conference_standings(logs, team_names, "East", EAST)

    print("Computing West standings ...")
    west = build_conference_standings(logs, team_names, "West", WEST)

    result = {
        "east": east,
        "west": west,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, separators=(",", ":"), default=str)

    print(f"Written -> {out_path}  (East: {len(east)} teams, West: {len(west)} teams)")
    return result


if __name__ == "__main__":
    build_standings()
