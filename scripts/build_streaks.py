"""
build_streaks.py -- produce dashboard/data/streaks.json

Identifies hot/cold players from recent scoring trends, team win/loss streaks,
and home/away split breakdowns for the current season.

Output:
  {
    "hot": [
      {"name": "Shai Gilgeous-Alexander", "team": "OKC", "sub": "Last 5: 38, 31, 42, 28, 35 pts", "stat": "+8.4 PPG"}
    ],
    "cold": [
      {"name": "Zion Williamson", "team": "NOP", "sub": "Last 5: 12, 8, 15, 10, 14 pts", "stat": "36.2% FG"}
    ],
    "team_streaks": [
      {"team": "OKC", "streak": 5},
      {"team": "DET", "streak": -2}
    ],
    "home_away": [
      {"team": "OKC", "home_pct": 84, "away_pct": 72}
    ],
    "exported_at": "..."
  }

Positive streak = win streak length; negative = loss streak length.

Run: python scripts/build_streaks.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PLAYER_LOGS = PROJECT_ROOT / "data" / "processed" / "player_game_logs.csv"
TEAM_LOGS = PROJECT_ROOT / "data" / "processed" / "team_game_logs.csv"
OUT_JSON = PROJECT_ROOT / "dashboard" / "data" / "streaks.json"

from src.config import get_current_season, get_current_season_id
import logging

log = logging.getLogger(__name__)

CURRENT_SEASON_ID = get_current_season_id()  # player_game_logs uses season_id column
CURRENT_SEASON = get_current_season()  # team_game_logs uses season column

MIN_GAMES = 10
LAST_N = 5
HOT_DELTA = 5.0       # PPG above season avg -> hot
COLD_FG_DELTA = -5.0  # FG% drop below season avg -> cold (percentage points)
TOP_PLAYERS = 4
TOP_HOME_AWAY = 10


# ---------------------------------------------------------------------------
# Team streaks
# ---------------------------------------------------------------------------

def _compute_team_streaks(df: pd.DataFrame) -> list[dict]:
    """
    For each team, compute current consecutive W or L streak.
    Positive = wins, negative = losses.
    """
    df = df.sort_values(["team_abbreviation", "game_date"])
    results: list[dict] = []

    for team, tdf in df.groupby("team_abbreviation"):
        wl = tdf["wl"].str.upper().tolist()
        if not wl:
            continue
        last = wl[-1]
        count = 0
        for val in reversed(wl):
            if val == last:
                count += 1
            else:
                break
        streak_val = count if last == "W" else -count
        results.append({"team": str(team), "streak": streak_val})

    # Sort by abs streak desc
    results.sort(key=lambda r: -abs(r["streak"]))
    return results


# ---------------------------------------------------------------------------
# Home/away splits
# ---------------------------------------------------------------------------

def _compute_home_away(df: pd.DataFrame) -> list[dict]:
    """
    Compute home and away win% for all teams.
    Return top TOP_HOME_AWAY teams by overall win%.
    """
    df = df.copy()
    df["is_home"] = df["matchup"].str.contains(r" vs\. ", regex=True)
    df["win"] = df["wl"].str.upper() == "W"

    results: list[dict] = []
    for team, tdf in df.groupby("team_abbreviation"):
        home_df = tdf[tdf["is_home"]]
        away_df = tdf[~tdf["is_home"]]

        home_w = int(home_df["win"].sum())
        home_g = len(home_df)
        away_w = int(away_df["win"].sum())
        away_g = len(away_df)
        total_w = home_w + away_w
        total_g = home_g + away_g

        overall_pct = total_w / total_g if total_g > 0 else 0.0
        home_pct = round(home_w / home_g * 100) if home_g > 0 else 0
        away_pct = round(away_w / away_g * 100) if away_g > 0 else 0

        results.append({
            "team": str(team),
            "home_pct": home_pct,
            "away_pct": away_pct,
            "_overall_pct": overall_pct,
        })

    results.sort(key=lambda r: -r["_overall_pct"])
    # Remove internal sort key
    output = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results[:TOP_HOME_AWAY]]
    return output


# ---------------------------------------------------------------------------
# Hot players
# ---------------------------------------------------------------------------

def _compute_hot_players(df: pd.DataFrame) -> list[dict]:
    """
    Find players whose PPG in last 5 games is >= HOT_DELTA above their season avg.
    Returns top TOP_PLAYERS sorted by delta desc.
    """
    df = df.copy()
    df = df.sort_values(["player_id", "game_date"])

    candidates: list[dict] = []

    for player_id, pdf in df.groupby("player_id"):
        pdf = pdf.sort_values("game_date")
        if len(pdf) < MIN_GAMES:
            continue

        season_avg = pdf["pts"].mean()
        # Require minimum season scoring average (exclude DNP-heavy bench players)
        if season_avg < 8.0:
            continue
        last5 = pdf.tail(LAST_N)
        # Require at least 3 of last 5 are real games (FGA > 0)
        last5_real = last5[last5["fga"].fillna(0) > 0]
        if len(last5_real) < 3:
            continue
        last5_avg = last5["pts"].mean()
        delta = last5_avg - season_avg

        if delta >= HOT_DELTA:
            last5_pts = [int(p) for p in last5["pts"].tolist()]
            player_name = str(pdf["player_name"].iloc[-1])
            team = str(pdf["team_abbreviation"].iloc[-1])
            pts_str = ", ".join(str(p) for p in last5_pts)
            candidates.append({
                "name": player_name,
                "team": team,
                "sub": f"Last 5: {pts_str} pts",
                "stat": f"+{delta:.1f} PPG",
                "_delta": delta,
            })

    candidates.sort(key=lambda r: -r["_delta"])
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in candidates[:TOP_PLAYERS]]


# ---------------------------------------------------------------------------
# Cold players (FG% drop)
# ---------------------------------------------------------------------------

def _compute_cold_players(df: pd.DataFrame) -> list[dict]:
    """
    Find players whose FG% in last 5 games is >= abs(COLD_FG_DELTA) below season avg.
    Returns top TOP_PLAYERS sorted by FG% delta (most negative first).
    """
    df = df.copy()
    df = df.sort_values(["player_id", "game_date"])

    candidates: list[dict] = []

    for player_id, pdf in df.groupby("player_id"):
        pdf = pdf.sort_values("game_date")
        if len(pdf) < MIN_GAMES:
            continue

        # Require minimum season scoring average (exclude DNP-heavy bench players)
        season_avg_pts = pdf["pts"].mean()
        if season_avg_pts < 8.0:
            continue

        # Need FGA data to compute meaningful FG%
        pdf_with_attempts = pdf[pdf["fga"].fillna(0) > 0]
        if len(pdf_with_attempts) < MIN_GAMES:
            continue

        season_fg_pct = pdf_with_attempts["fg_pct"].mean()
        last5 = pdf.tail(LAST_N)
        last5_with_att = last5[last5["fga"].fillna(0) > 0]

        # Require at least 3 of last 5 games with actual attempts (not DNPs)
        if len(last5_with_att) < 3:
            continue

        last5_fg_pct = last5_with_att["fg_pct"].mean()
        delta_pct = (last5_fg_pct - season_fg_pct) * 100.0  # convert to percentage points

        if delta_pct <= COLD_FG_DELTA:
            last5_pts = [int(p) for p in last5["pts"].tolist()]
            player_name = str(pdf["player_name"].iloc[-1])
            team = str(pdf["team_abbreviation"].iloc[-1])
            pts_str = ", ".join(str(p) for p in last5_pts)
            fg_display = last5_fg_pct * 100.0
            candidates.append({
                "name": player_name,
                "team": team,
                "sub": f"Last 5: {pts_str} pts",
                "stat": f"{fg_display:.1f}% FG",
                "_delta": delta_pct,
            })

    candidates.sort(key=lambda r: r["_delta"])  # most negative first
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in candidates[:TOP_PLAYERS]]


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build_streaks(
    player_logs_path: Path = PLAYER_LOGS,
    team_logs_path: Path = TEAM_LOGS,
    out_path: Path = OUT_JSON,
) -> dict:
    # ---- Team logs ----
    if not team_logs_path.exists():
        log.warning(f"WARN: team_game_logs not found: {team_logs_path} -- skipping streaks build")
        return {}

    log.info(f"Loading team game logs from {team_logs_path} ...")
    tdf = pd.read_csv(team_logs_path)
    tdf["game_date"] = pd.to_datetime(tdf["game_date"], format="mixed")
    tdf_season = tdf[tdf["season"] == CURRENT_SEASON].copy()
    log.info(f"  {len(tdf_season)} rows for season {CURRENT_SEASON}")

    log.info("Computing team streaks ...")
    team_streaks = _compute_team_streaks(tdf_season)
    log.info(f"  {len(team_streaks)} team streaks computed")

    log.info("Computing home/away splits ...")
    home_away = _compute_home_away(tdf_season)
    log.info(f"  {len(home_away)} teams in home/away output")

    # ---- Player logs ----
    hot: list[dict] = []
    cold: list[dict] = []

    if not player_logs_path.exists():
        log.warning(f"WARN: player_game_logs not found: {player_logs_path} -- skipping hot/cold players")
    else:
        log.info(f"Loading player game logs from {player_logs_path} ...")
        pdf = pd.read_csv(
            player_logs_path,
            usecols=[
                "season_id", "player_id", "player_name",
                "team_abbreviation", "game_date",
                "pts", "fgm", "fga", "fg_pct",
            ],
        )
        pdf["game_date"] = pd.to_datetime(pdf["game_date"], format="mixed")
        pdf = pdf[pdf["season_id"] == CURRENT_SEASON_ID].copy()
        log.info(f"  {len(pdf)} rows for season_id {CURRENT_SEASON_ID}")

        log.info("Computing hot players ...")
        hot = _compute_hot_players(pdf)
        log.info(f"  {len(hot)} hot players found")

        log.info("Computing cold players ...")
        cold = _compute_cold_players(pdf)
        log.info(f"  {len(cold)} cold players found")

    result = {
        "hot": hot,
        "cold": cold,
        "team_streaks": team_streaks,
        "home_away": home_away,
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, separators=(",", ":"), default=str)

    log.info(f"Written -> {out_path}")
    if hot:
        log.info("  Hot players:")
        for p in hot:
            log.info(f"    {p['name']} ({p['team']}) -- {p['stat']}")
    else:
        log.info("  No hot players meeting threshold")
    if cold:
        log.info("  Cold players:")
        for p in cold:
            log.info(f"    {p['name']} ({p['team']}) -- {p['stat']}")
    else:
        log.info("  No cold players meeting threshold")

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    log.info("=== build_streaks ===")
    build_streaks()
