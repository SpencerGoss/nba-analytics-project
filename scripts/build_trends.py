"""
build_trends.py  --  produce dashboard/data/trends.json

For all 30 NBA teams, compute current-season trend stats from:
  - data/processed/team_game_logs.csv   (box scores, W/L)
  - data/features/team_game_features.csv  (rolling ratings, pace, fg3a)

Output schema per team abbreviation:
{
  "OKC": {
    "last10_su": "8-2",
    "last10_avg_scored": 121.3,
    "last10_avg_allowed": 108.4,
    "home_record": "22-5",
    "away_record": "18-9",
    "streak": 3,          # positive = win streak, negative = loss streak
    "trend_offense": "+4.2",   # last-10 ortg roll10 vs season avg
    "trend_defense": "-2.1",   # last-10 drtg roll10 vs season avg (lower=better)
    "last10_pace": 98.4,
    "last10_fg3_rate": 0.412
  }
}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

LOGS_CSV = PROJECT_ROOT / "data" / "processed" / "team_game_logs.csv"
FEATURES_CSV = PROJECT_ROOT / "data" / "features" / "team_game_features.csv"
OUT_JSON = PROJECT_ROOT / "dashboard" / "data" / "trends.json"

from src.config import get_current_season

CURRENT_SEASON = get_current_season()
LAST_N = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record_str(wins: int, losses: int) -> str:
    return f"{wins}-{losses}"


def _current_streak(wl_series: pd.Series) -> int:
    """Return streak length: positive for wins, negative for losses.

    wl_series should be sorted oldest-first; last entry is most recent.
    """
    if wl_series.empty:
        return 0
    result = wl_series.tolist()
    last = result[-1]
    count = 0
    for val in reversed(result):
        if val == last:
            count += 1
        else:
            break
    return count if last == "W" else -count


def _fmt_diff(value: float) -> str:
    """Format a float difference with explicit sign."""
    return f"+{value:.1f}" if value >= 0 else f"{value:.1f}"


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_logs(season: int = CURRENT_SEASON) -> pd.DataFrame:
    df = pd.read_csv(LOGS_CSV)
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")
    df = df[df["season"] == season].copy()
    # Derive is_home from matchup string: "TEAM vs. OPP" = home, "TEAM @ OPP" = away
    df["is_home"] = df["matchup"].str.contains(" vs. ", regex=False)
    # Derive opponent pts: we need opp_pts per row. Merge same game_id rows.
    pts_map = (
        df.groupby("game_id")[["team_abbreviation", "pts"]]
        .apply(lambda g: g.set_index("team_abbreviation")["pts"].to_dict())
        .to_dict()
    )

    def _opp_pts(row: pd.Series) -> float:
        game_pts = pts_map.get(row["game_id"], {})
        for team, pts_val in game_pts.items():
            if team != row["team_abbreviation"]:
                return float(pts_val)
        return float("nan")

    df["opp_pts"] = df.apply(_opp_pts, axis=1)
    df = df.sort_values(["team_abbreviation", "game_date"]).reset_index(drop=True)
    return df


def load_features(season: int = CURRENT_SEASON) -> pd.DataFrame:
    cols_needed = [
        "team_abbreviation", "game_date", "season",
        "off_rtg_game_roll10", "def_rtg_game_roll10",
        "pace_game_roll10",
        "fg3_pct_roll10",
        "off_rtg_game_roll20", "def_rtg_game_roll20",
    ]
    df = pd.read_csv(FEATURES_CSV, usecols=cols_needed)
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")
    df = df[df["season"] == season].copy()
    df = df.sort_values(["team_abbreviation", "game_date"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Per-team computation
# ---------------------------------------------------------------------------

def compute_team_trends(
    logs: pd.DataFrame,
    features: pd.DataFrame,
) -> dict[str, dict]:
    results: dict[str, dict] = {}

    teams = sorted(logs["team_abbreviation"].dropna().unique())

    for team in teams:
        tlog = logs[logs["team_abbreviation"] == team].sort_values("game_date")
        tfeat = features[features["team_abbreviation"] == team].sort_values("game_date")

        if tlog.empty:
            continue

        # --- Full-season records ---
        home_games = tlog[tlog["is_home"]]
        away_games = tlog[~tlog["is_home"]]
        home_w = int((home_games["wl"] == "W").sum())
        home_l = int((home_games["wl"] == "L").sum())
        away_w = int((away_games["wl"] == "W").sum())
        away_l = int((away_games["wl"] == "L").sum())

        # --- Last 10 ---
        last10_log = tlog.tail(LAST_N)
        last10_wins = int((last10_log["wl"] == "W").sum())
        last10_losses = int((last10_log["wl"] == "L").sum())
        last10_avg_scored = round(float(last10_log["pts"].mean()), 1)
        last10_avg_allowed = round(float(last10_log["opp_pts"].mean()), 1)

        # --- Streak ---
        streak = _current_streak(tlog["wl"].reset_index(drop=True))

        # --- Rating trends (from features) ---
        trend_offense = None
        trend_defense = None
        last10_pace = None
        last10_fg3_rate = None

        if not tfeat.empty:
            last_feat_row = tfeat.iloc[-1]
            roll10_ortg = last_feat_row.get("off_rtg_game_roll10")
            roll20_ortg = last_feat_row.get("off_rtg_game_roll20")
            roll10_drtg = last_feat_row.get("def_rtg_game_roll10")
            roll20_drtg = last_feat_row.get("def_rtg_game_roll20")
            last10_pace_val = last_feat_row.get("pace_game_roll10")
            last10_fg3_val = last_feat_row.get("fg3_pct_roll10")

            if pd.notna(roll10_ortg) and pd.notna(roll20_ortg):
                trend_offense = _fmt_diff(round(float(roll10_ortg) - float(roll20_ortg), 1))
            if pd.notna(roll10_drtg) and pd.notna(roll20_drtg):
                trend_defense = _fmt_diff(round(float(roll10_drtg) - float(roll20_drtg), 1))
            if pd.notna(last10_pace_val):
                last10_pace = round(float(last10_pace_val), 1)
            if pd.notna(last10_fg3_val):
                last10_fg3_rate = round(float(last10_fg3_val), 3)

        results[team] = {
            "last10_su": _record_str(last10_wins, last10_losses),
            "last10_avg_scored": last10_avg_scored,
            "last10_avg_allowed": last10_avg_allowed,
            "home_record": _record_str(home_w, home_l),
            "away_record": _record_str(away_w, away_l),
            "streak": streak,
            "trend_offense": trend_offense,
            "trend_defense": trend_defense,
            "last10_pace": last10_pace,
            "last10_fg3_rate": last10_fg3_rate,
        }

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_trends() -> dict[str, dict]:
    print(f"Loading game logs from {LOGS_CSV} ...")
    logs = load_logs()
    print(f"  {len(logs)} rows for season {CURRENT_SEASON}")

    print(f"Loading team features from {FEATURES_CSV} ...")
    features = load_features()
    print(f"  {len(features)} rows for season {CURRENT_SEASON}")

    print("Computing team trends ...")
    trends = compute_team_trends(logs, features)
    print(f"  {len(trends)} teams processed")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(trends, fh, separators=(",", ":"))
    print(f"Written -> {OUT_JSON}")
    return trends


if __name__ == "__main__":
    build_trends()
