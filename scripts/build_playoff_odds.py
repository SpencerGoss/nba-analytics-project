"""
build_playoff_odds.py -- produce dashboard/data/playoff_odds.json

Computes estimated playoff odds and title odds from current season standings.

Algorithm:
  - Ranks 1-6 by conference win%: pct=100 (in playoff)
  - Ranks 7-8 (play-in): pct based on games behind
  - Ranks 9-10: pct = max(0, 50 - games_behind * 8)
  - Ranks below 10: pct = 0
  - Title odds: inverse of (games_behind + 1), normalised so top team gets 20-25%

Output:
  {
    "east": [
      {"team": "Detroit Pistons", "abbr": "DET", "pct": 100, "title": 2.1, "rank": 1}
    ],
    "west": [...],
    "exported_at": "..."
  }

Run: python scripts/build_playoff_odds.py
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
OUT_JSON = PROJECT_ROOT / "dashboard" / "data" / "playoff_odds.json"

CURRENT_SEASON = 202526

# ---------------------------------------------------------------------------
# Conference / division mappings (mirrors build_standings.py)
# ---------------------------------------------------------------------------

EAST: dict[str, list[str]] = {
    "Atlantic":  ["BOS", "BKN", "NYK", "PHI", "TOR"],
    "Central":   ["CHI", "CLE", "DET", "IND", "MIL"],
    "Southeast": ["ATL", "CHA", "MIA", "ORL", "WAS"],
}

WEST: dict[str, list[str]] = {
    "Northwest": ["DEN", "MIN", "OKC", "POR", "UTA"],
    "Pacific":   ["GSW", "LAC", "LAL", "PHX", "SAC"],
    "Southwest": ["DAL", "HOU", "MEM", "NOP", "SAS"],
}

EAST_TEAMS: list[str] = [t for teams in EAST.values() for t in teams]
WEST_TEAMS: list[str] = [t for teams in WEST.values() for t in teams]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _games_behind(leader_w: int, leader_l: int, team_w: int, team_l: int) -> float:
    return ((leader_w - team_w) + (team_l - leader_l)) / 2.0


def _playoff_pct(rank: int, gb: float) -> float:
    """Return estimated playoff probability (0-100) for a given rank/GB."""
    if rank <= 6:
        return 100.0
    if rank <= 8:
        # Play-in zone
        if gb > 10:
            return 15.0
        if gb > 5:
            return 50.0
        return 75.0
    if rank <= 10:
        return max(0.0, 50.0 - gb * 8.0)
    return 0.0


def _load_team_names() -> dict[str, str]:
    try:
        df = pd.read_csv(TEAMS_CSV, usecols=["abbreviation", "full_name"])
        return dict(zip(df["abbreviation"], df["full_name"]))
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Standings computation
# ---------------------------------------------------------------------------

def _compute_conference_standings(logs: pd.DataFrame, conf_teams: list[str]) -> list[dict]:
    """Compute W/L per team for the given conference, return sorted rows."""
    rows: list[dict] = []
    for team in conf_teams:
        tdf = logs[logs["team_abbreviation"] == team]
        w = int((tdf["wl"] == "W").sum())
        lo = int((tdf["wl"] == "L").sum())
        total = w + lo
        win_pct = w / total if total > 0 else 0.0
        rows.append({"abbr": team, "w": w, "l": lo, "win_pct": win_pct})

    # Sort by win%, then W desc, then L asc
    rows.sort(key=lambda r: (-r["win_pct"], -r["w"], r["l"]))

    leader_w = rows[0]["w"] if rows else 0
    leader_l = rows[0]["l"] if rows else 0

    for i, row in enumerate(rows):
        row["rank"] = i + 1
        row["gb"] = _games_behind(leader_w, leader_l, row["w"], row["l"])

    return rows


# ---------------------------------------------------------------------------
# Title odds normalisation
# ---------------------------------------------------------------------------

def _title_odds_map(all_conf_rows: list[list[dict]]) -> dict[str, float]:
    """
    Compute title odds across all 30 teams.
    Uses inverse of (gb_from_overall_leader + 1), normalised so top team gets ~20-25%.
    """
    # Flatten all teams into a single list with their overall GB from the very best team
    flat: list[dict] = []
    for conf_rows in all_conf_rows:
        flat.extend(conf_rows)

    if not flat:
        return {}

    # Find team with best record overall
    best = max(flat, key=lambda r: (r["win_pct"], r["w"]))
    leader_w = best["w"]
    leader_l = best["l"]

    # Compute global GB for each team
    raw_scores: dict[str, float] = {}
    for row in flat:
        global_gb = _games_behind(leader_w, leader_l, row["w"], row["l"])
        raw_scores[row["abbr"]] = 1.0 / (global_gb + 1.0)

    total = sum(raw_scores.values())
    if total == 0:
        return {}

    # Normalise so sum = 100, then scale so top team is at most 25%
    normalised: dict[str, float] = {t: (v / total) * 100.0 for t, v in raw_scores.items()}
    top_val = max(normalised.values())
    if top_val > 25.0:
        scale = 25.0 / top_val
        normalised = {t: v * scale for t, v in normalised.items()}

    return {t: round(v, 1) for t, v in normalised.items()}


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build_playoff_odds(
    logs_path: Path = TEAM_LOGS,
    teams_path: Path = TEAMS_CSV,
    out_path: Path = OUT_JSON,
    season: int = CURRENT_SEASON,
) -> dict:
    if not logs_path.exists():
        print(f"WARN: team_game_logs not found: {logs_path} -- skipping playoff odds build")
        return {}

    print(f"Loading game logs from {logs_path} ...")
    df = pd.read_csv(logs_path)
    df = df[df["season"] == season].copy()
    print(f"  {len(df)} rows for season {season}")

    team_names = _load_team_names()

    print("Computing East standings ...")
    east_rows = _compute_conference_standings(df, EAST_TEAMS)

    print("Computing West standings ...")
    west_rows = _compute_conference_standings(df, WEST_TEAMS)

    title_map = _title_odds_map([east_rows, west_rows])

    def _build_output(rows: list[dict]) -> list[dict]:
        out = []
        for row in rows:
            abbr = row["abbr"]
            rank = row["rank"]
            gb = row["gb"]
            out.append({
                "team": team_names.get(abbr, abbr),
                "abbr": abbr,
                "pct": _playoff_pct(rank, gb),
                "title": title_map.get(abbr, 0.0),
                "rank": rank,
            })
        return out

    east_out = _build_output(east_rows)
    west_out = _build_output(west_rows)

    result = {
        "east": east_out,
        "west": west_out,
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)

    print(f"Written -> {out_path}  (East: {len(east_out)} teams, West: {len(west_out)} teams)")

    print("  Top 5 East:")
    for row in east_out[:5]:
        print(f"    #{row['rank']} {row['abbr']:<4s}  pct={row['pct']:.0f}%  title={row['title']:.1f}%")
    print("  Top 5 West:")
    for row in west_out[:5]:
        print(f"    #{row['rank']} {row['abbr']:<4s}  pct={row['pct']:.0f}%  title={row['title']:.1f}%")

    return result


if __name__ == "__main__":
    print("=== build_playoff_odds ===")
    build_playoff_odds()
