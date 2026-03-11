"""
build_matchup_analysis.py  --  produce dashboard/data/matchup_analysis.json

For each game in today's picks, compute head-to-head metric comparisons
using data/features/team_game_features.csv (rolling ratings, pace, 3PT).

Output is a list of game analysis objects with radar-chart-ready data
(5 dimensions normalized 0-100):
  Offense, Defense, Pace, 3PT Rate, Recent Form

[
  {
    "home_team": "OKC",
    "away_team": "POR",
    "radar": {
      "OKC": [82, 71, 68, 74, 85],
      "POR": [61, 54, 72, 58, 42]
    },
    "dimensions": ["Offense", "Defense", "Pace", "3PT Rate", "Recent Form"],
    "home_ortg": 118.4,
    "home_drtg": 108.1,
    "away_ortg": 112.3,
    "away_drtg": 112.1,
    "home_pace": 98.4,
    "away_pace": 101.2,
    "pace_diff": -2.8,
    "home_fg3_rate": 0.412,
    "away_fg3_rate": 0.381,
    "home_net_rtg": 10.3,
    "away_net_rtg": 0.2,
    "net_rtg_diff": 10.1,
    "home_recent_form": 0.7,   # win_pct_roll10
    "away_recent_form": 0.4
  }
]
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

FEATURES_CSV = PROJECT_ROOT / "data" / "features" / "team_game_features.csv"
PICKS_JSON = PROJECT_ROOT / "dashboard" / "data" / "todays_picks.json"
OUT_JSON = PROJECT_ROOT / "dashboard" / "data" / "matchup_analysis.json"

CURRENT_SEASON = 202526
DIMENSIONS = ["Offense", "Defense", "Pace", "3PT Rate", "Recent Form"]

# Columns to pull from team_game_features
TEAM_COLS = [
    "team_abbreviation", "game_date", "season",
    "off_rtg_game_roll10",
    "def_rtg_game_roll10",
    "net_rtg_game_roll10",
    "pace_game_roll10",
    "fg3_pct_roll10",
    "win_pct_roll10",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_features(season: int = CURRENT_SEASON) -> pd.DataFrame:
    df = pd.read_csv(FEATURES_CSV, usecols=TEAM_COLS)
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")
    df = df[df["season"] == season].copy()
    df = df.sort_values(["team_abbreviation", "game_date"]).reset_index(drop=True)
    return df


def load_picks() -> list[dict]:
    if not PICKS_JSON.exists():
        log.warning("todays_picks.json not found at %s", PICKS_JSON)
        return []
    with open(PICKS_JSON, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _safe_float(val) -> float | None:
    try:
        f = float(val)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def _normalize_0_100(
    home_val: float | None,
    away_val: float | None,
    higher_is_better: bool = True,
) -> tuple[int, int]:
    """Normalize two values to 0-100 scale relative to each other.

    Returns (home_score, away_score). When values are equal both return 50.
    """
    if home_val is None and away_val is None:
        return (50, 50)
    if home_val is None:
        return (40, 60) if higher_is_better else (60, 40)
    if away_val is None:
        return (60, 40) if higher_is_better else (40, 60)

    lo = min(home_val, away_val)
    hi = max(home_val, away_val)
    span = hi - lo

    if span < 1e-9:
        return (50, 50)

    # Scale so that the better team lands ~60-80 range, worse ~30-50
    home_norm = ((home_val - lo) / span) * 60 + 20
    away_norm = ((away_val - lo) / span) * 60 + 20

    if not higher_is_better:
        home_norm, away_norm = away_norm, home_norm

    return (int(round(home_norm)), int(round(away_norm)))


def _get_latest_metrics(
    features: pd.DataFrame, team: str
) -> dict[str, float | None]:
    team_data = features[features["team_abbreviation"] == team]
    if team_data.empty:
        return {
            "ortg": None, "drtg": None, "net_rtg": None,
            "pace": None, "fg3_rate": None, "recent_form": None,
        }
    row = team_data.iloc[-1]
    return {
        "ortg": _safe_float(row.get("off_rtg_game_roll10")),
        "drtg": _safe_float(row.get("def_rtg_game_roll10")),
        "net_rtg": _safe_float(row.get("net_rtg_game_roll10")),
        "pace": _safe_float(row.get("pace_game_roll10")),
        "fg3_rate": _safe_float(row.get("fg3_pct_roll10")),
        "recent_form": _safe_float(row.get("win_pct_roll10")),
    }


# ---------------------------------------------------------------------------
# Per-game analysis
# ---------------------------------------------------------------------------

def compute_matchup_analysis(
    picks: list[dict],
    features: pd.DataFrame,
) -> list[dict]:
    results = []
    for pick in picks:
        home_team = pick["home_team"]
        away_team = pick["away_team"]

        home_m = _get_latest_metrics(features, home_team)
        away_m = _get_latest_metrics(features, away_team)

        # Radar: Offense, Defense, Pace, 3PT Rate, Recent Form
        off_h, off_a = _normalize_0_100(home_m["ortg"], away_m["ortg"], higher_is_better=True)
        # Defense: lower DRtg is better
        def_h, def_a = _normalize_0_100(home_m["drtg"], away_m["drtg"], higher_is_better=False)
        pac_h, pac_a = _normalize_0_100(home_m["pace"], away_m["pace"], higher_is_better=True)
        fg3_h, fg3_a = _normalize_0_100(home_m["fg3_rate"], away_m["fg3_rate"], higher_is_better=True)
        frm_h, frm_a = _normalize_0_100(home_m["recent_form"], away_m["recent_form"], higher_is_better=True)

        pace_diff = None
        if home_m["pace"] is not None and away_m["pace"] is not None:
            pace_diff = round(home_m["pace"] - away_m["pace"], 1)

        net_rtg_diff = None
        if home_m["net_rtg"] is not None and away_m["net_rtg"] is not None:
            net_rtg_diff = round(home_m["net_rtg"] - away_m["net_rtg"], 1)

        results.append({
            "home_team": home_team,
            "away_team": away_team,
            "dimensions": DIMENSIONS,
            "radar": {
                home_team: [off_h, def_h, pac_h, fg3_h, frm_h],
                away_team: [off_a, def_a, pac_a, fg3_a, frm_a],
            },
            "home_ortg": home_m["ortg"],
            "home_drtg": home_m["drtg"],
            "away_ortg": away_m["ortg"],
            "away_drtg": away_m["drtg"],
            "home_pace": home_m["pace"],
            "away_pace": away_m["pace"],
            "pace_diff": pace_diff,
            "home_fg3_rate": home_m["fg3_rate"],
            "away_fg3_rate": away_m["fg3_rate"],
            "home_net_rtg": home_m["net_rtg"],
            "away_net_rtg": away_m["net_rtg"],
            "net_rtg_diff": net_rtg_diff,
            "home_recent_form": home_m["recent_form"],
            "away_recent_form": away_m["recent_form"],
        })

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_matchup_analysis() -> list[dict]:
    print(f"Loading team features from {FEATURES_CSV} ...")
    features = load_features()
    print(f"  {len(features)} rows for season {CURRENT_SEASON}")
    print(f"  Teams with data: {sorted(features['team_abbreviation'].unique())}")

    print(f"Loading today's picks from {PICKS_JSON} ...")
    picks = load_picks()
    if not picks:
        print("No picks found -- skipping matchup analysis.")
        return []
    print(f"  {len(picks)} matchups")

    print("Computing matchup analysis ...")
    results = compute_matchup_analysis(picks, features)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(results, fh, separators=(",", ":"))
    print(f"Written -> {OUT_JSON}")
    return results


if __name__ == "__main__":
    build_matchup_analysis()
