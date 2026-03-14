"""
build_h2h.py  --  produce dashboard/data/head_to_head.json

For each matchup in today's picks, find historical H2H results from
data/processed/team_game_logs.csv (all seasons).

Output is a list of matchup objects:
[
  {
    "home_team": "OKC",
    "away_team": "POR",
    "series_record": "OKC leads 7-3 (last 10 meetings)",
    "avg_total": 228.4,
    "meetings": [
      {
        "date": "2025-12-14",
        "home_team": "OKC",
        "home_score": 118,
        "away_score": 104,
        "winner": "OKC",
        "margin": 14
      }
    ]
  }
]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import logging

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

LOGS_CSV = PROJECT_ROOT / "data" / "processed" / "team_game_logs.csv"
PICKS_JSON = PROJECT_ROOT / "dashboard" / "data" / "todays_picks.json"
OUT_JSON = PROJECT_ROOT / "dashboard" / "data" / "head_to_head.json"

LAST_N_MEETINGS = 10


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_logs() -> pd.DataFrame:
    """Load all team game logs and build a paired-game view (one row per game)."""
    df = pd.read_csv(LOGS_CSV)
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")
    df["is_home"] = df["matchup"].str.contains(" vs. ", regex=False)
    return df


def load_picks() -> list[dict]:
    with open(PICKS_JSON, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# H2H computation
# ---------------------------------------------------------------------------

def _build_game_index(logs: pd.DataFrame) -> pd.DataFrame:
    """Create a single-row-per-game table with home_team, away_team, scores."""
    home = logs[logs["is_home"]].copy()
    away = logs[~logs["is_home"]].copy()

    home = home.rename(columns={
        "team_abbreviation": "home_team",
        "pts": "home_score",
        "wl": "home_wl",
    })[["game_id", "game_date", "home_team", "home_score", "home_wl", "season"]]

    away = away.rename(columns={
        "team_abbreviation": "away_team",
        "pts": "away_score",
    })[["game_id", "away_team", "away_score"]]

    games = home.merge(away, on="game_id", how="inner")
    games["winner"] = games.apply(
        lambda r: r["home_team"] if r["home_wl"] == "W" else r["away_team"],
        axis=1,
    )
    games["margin"] = (games["home_score"] - games["away_score"]).abs()
    games = games.sort_values("game_date").reset_index(drop=True)
    return games


def _get_h2h_meetings(
    games: pd.DataFrame,
    team_a: str,
    team_b: str,
    n: int = LAST_N_MEETINGS,
) -> pd.DataFrame:
    """Return up to n most-recent meetings between two teams (either side)."""
    mask = (
        ((games["home_team"] == team_a) & (games["away_team"] == team_b))
        | ((games["home_team"] == team_b) & (games["away_team"] == team_a))
    )
    meetings = games[mask].sort_values("game_date", ascending=False).head(n)
    return meetings.sort_values("game_date", ascending=False)


def _series_record(meetings: pd.DataFrame, home_team: str, away_team: str) -> str:
    """Return a human-readable series record string."""
    if meetings.empty:
        return "No historical meetings found"
    n = len(meetings)
    home_wins = int((meetings["winner"] == home_team).sum())
    away_wins = int((meetings["winner"] == away_team).sum())
    if home_wins > away_wins:
        leader = home_team
        leader_wins = home_wins
        trailer_wins = away_wins
    elif away_wins > home_wins:
        leader = away_team
        leader_wins = away_wins
        trailer_wins = home_wins
    else:
        return f"Series tied {home_wins}-{away_wins} (last {n} meetings)"
    return f"{leader} leads {leader_wins}-{trailer_wins} (last {n} meetings)"


def compute_h2h(picks: list[dict], games: pd.DataFrame) -> list[dict]:
    results = []
    for pick in picks:
        home_team = pick["home_team"]
        away_team = pick["away_team"]

        meetings = _get_h2h_meetings(games, home_team, away_team)

        series_str = _series_record(meetings, home_team, away_team)

        avg_total = None
        meeting_list: list[dict] = []

        if not meetings.empty:
            totals = (meetings["home_score"] + meetings["away_score"]).dropna()
            avg_total = round(float(totals.mean()), 1) if not totals.empty else None

            for _, row in meetings.iterrows():
                meeting_list.append({
                    "date": row["game_date"].strftime("%Y-%m-%d"),
                    "home_team": row["home_team"],
                    "home_score": int(row["home_score"]),
                    "away_score": int(row["away_score"]),
                    "winner": row["winner"],
                    "margin": int(row["margin"]),
                })

        results.append({
            "home_team": home_team,
            "away_team": away_team,
            "series_record": series_str,
            "avg_total": avg_total,
            "meetings": meeting_list,
        })

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_h2h() -> list[dict]:
    log.info(f"Loading game logs from {LOGS_CSV} ...")
    logs = load_logs()
    log.info(f"  {len(logs)} total rows across all seasons")

    log.info("Building per-game index ...")
    games = _build_game_index(logs)
    log.info(f"  {len(games)} games indexed")

    log.info(f"Loading today's picks from {PICKS_JSON} ...")
    picks = load_picks()
    log.info(f"  {len(picks)} matchups")

    log.info("Computing H2H records ...")
    results = compute_h2h(picks, games)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(results, fh, separators=(",", ":"))
    log.info(f"Written -> {OUT_JSON}")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_h2h()
