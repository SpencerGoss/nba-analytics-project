"""
build_power_rankings.py  --  produce dashboard/data/power_rankings.json

Ranks all 30 NBA teams using a composite score built from:
  - Net rating last-10 games    (weight 0.40)
  - Net rating season-to-date   (weight 0.20)
  - Pythagorean win%  last-10   (weight 0.25)
  - Recent form win%  last-10   (weight 0.15)

Trend (up/down/same) compares net rating of games[-5:] vs games[-10:-5].
prev_rank is computed from the games[-10:-5] window using the same weights.

Sources:
  data/features/team_game_features.csv
  data/processed/teams.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

FEATURES_CSV = PROJECT_ROOT / "data" / "features" / "team_game_features.csv"
TEAMS_CSV = PROJECT_ROOT / "data" / "processed" / "teams.csv"
OUT_JSON = PROJECT_ROOT / "dashboard" / "data" / "power_rankings.json"

CURRENT_SEASON = 202526
LAST_N = 10
HALF_N = 5

# Composite weights -- must sum to 1.0
W_NET10 = 0.40
W_NET_SEASON = 0.20
W_PYTH10 = 0.25
W_FORM10 = 0.15

# Normalisation range for composite (output 0-100 scale)
COMPOSITE_MIN = -15.0
COMPOSITE_MAX = 15.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        v = float(value)  # type: ignore[arg-type]
        return v if pd.notna(v) else default
    except (TypeError, ValueError):
        return default


def _record_str(wins: int, losses: int) -> str:
    return f"{wins}-{losses}"


def _normalise(value: float, lo: float, hi: float) -> float:
    """Linearly map value from [lo, hi] to [0, 100], clamped."""
    if hi == lo:
        return 50.0
    result = (value - lo) / (hi - lo) * 100.0
    return float(max(0.0, min(100.0, result)))


def _compute_composite(net10: float, net_season: float, pyth10: float, form10: float) -> float:
    """Return a raw composite before final ranking normalisation."""
    # Pythagorean win% is in [0,1]; scale to net-rating-like range (-15..+15)
    pyth_scaled = (pyth10 - 0.5) * 30.0
    # Win% also in [0,1]
    form_scaled = (form10 - 0.5) * 30.0
    return (
        W_NET10 * net10
        + W_NET_SEASON * net_season
        + W_PYTH10 * pyth_scaled
        + W_FORM10 * form_scaled
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_features(season: int = CURRENT_SEASON) -> pd.DataFrame:
    cols = [
        "team_abbreviation",
        "game_date",
        "season",
        "net_rtg_game_roll5",
        "net_rtg_game_roll10",
        "net_rtg_game_roll20",
        "pythagorean_win_pct_roll10",
        "win_pct_roll10",
        "win",
        "cum_wins",
        "cum_losses",
    ]
    df = pd.read_csv(FEATURES_CSV, usecols=cols)
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")
    df = df[df["season"] == season].copy()
    df = df.sort_values(["team_abbreviation", "game_date"]).reset_index(drop=True)
    return df


def load_team_names() -> dict[str, str]:
    """Return mapping abbreviation -> full_name."""
    df = pd.read_csv(TEAMS_CSV, usecols=["abbreviation", "full_name"])
    return dict(zip(df["abbreviation"], df["full_name"]))


# ---------------------------------------------------------------------------
# Per-team metric extraction
# ---------------------------------------------------------------------------

def _team_metrics(tdf: pd.DataFrame) -> dict:
    """Compute metrics for one team from its sorted game log."""
    n = len(tdf)
    last10 = tdf.tail(LAST_N)
    prev5 = tdf.iloc[max(0, n - LAST_N): max(0, n - HALF_N)]
    last5 = tdf.tail(HALF_N)

    last10_wins = int((last10["win"] == 1).sum())
    last10_losses = int((last10["win"] == 0).sum())

    net10 = _safe_float(last10["net_rtg_game_roll10"].iloc[-1]) if not last10.empty else 0.0
    net_season = _safe_float(last10["net_rtg_game_roll20"].iloc[-1]) if not last10.empty else 0.0
    pyth10 = _safe_float(last10["pythagorean_win_pct_roll10"].iloc[-1]) if not last10.empty else 0.5
    form10 = _safe_float(last10["win_pct_roll10"].iloc[-1]) if not last10.empty else 0.5

    # Trend: compare net_rtg_game_roll5 at last-5 boundary vs prev-5 boundary
    avg_pm_last5 = _safe_float(last5["net_rtg_game_roll5"].iloc[-1]) if not last5.empty else 0.0
    avg_pm_prev5 = _safe_float(prev5["net_rtg_game_roll5"].iloc[-1]) if not prev5.empty else 0.0
    diff = avg_pm_last5 - avg_pm_prev5
    if diff > 1.0:
        trend = "up"
    elif diff < -1.0:
        trend = "down"
    else:
        trend = "same"

    # prev_rank metrics: from the games[-10:-5] window
    prev5_net10 = _safe_float(prev5["net_rtg_game_roll10"].iloc[-1]) if not prev5.empty else 0.0
    prev5_net_season = _safe_float(prev5["net_rtg_game_roll20"].iloc[-1]) if not prev5.empty else 0.0
    prev5_pyth10 = _safe_float(prev5["pythagorean_win_pct_roll10"].iloc[-1]) if not prev5.empty else 0.5
    prev5_form10 = _safe_float(prev5["win_pct_roll10"].iloc[-1]) if not prev5.empty else 0.5

    current_net = _safe_float(last10["net_rtg_game_roll10"].iloc[-1]) if not last10.empty else 0.0

    return {
        "last10_wins": last10_wins,
        "last10_losses": last10_losses,
        "net10": net10,
        "net_season": net_season,
        "pyth10": pyth10,
        "form10": form10,
        "trend": trend,
        "composite": _compute_composite(net10, net_season, pyth10, form10),
        "prev_composite": _compute_composite(prev5_net10, prev5_net_season, prev5_pyth10, prev5_form10),
        "current_net": current_net,
    }


# ---------------------------------------------------------------------------
# Main ranking logic
# ---------------------------------------------------------------------------

def build_power_rankings(
    features: pd.DataFrame | None = None,
    team_names: dict[str, str] | None = None,
) -> list[dict]:
    if features is None:
        features = load_features()
    if team_names is None:
        team_names = load_team_names()

    teams = sorted(features["team_abbreviation"].dropna().unique())
    team_metrics: dict[str, dict] = {}

    for team in teams:
        tdf = features[features["team_abbreviation"] == team].sort_values("game_date")
        if tdf.empty:
            continue
        team_metrics[team] = _team_metrics(tdf)

    if not team_metrics:
        return []

    # Sort by composite (current) to assign ranks
    sorted_current = sorted(team_metrics.items(), key=lambda x: x[1]["composite"], reverse=True)
    sorted_prev = sorted(team_metrics.items(), key=lambda x: x[1]["prev_composite"], reverse=True)

    prev_rank_map = {team: idx + 1 for idx, (team, _) in enumerate(sorted_prev)}

    results = []
    for rank, (team, metrics) in enumerate(sorted_current, start=1):
        results.append(
            {
                "rank": rank,
                "team": team,
                "team_name": team_names.get(team, team),
                "composite_score": round(_normalise(metrics["composite"], COMPOSITE_MIN, COMPOSITE_MAX), 1),
                "net_rating": round(metrics["current_net"], 1),
                "last10_record": _record_str(metrics["last10_wins"], metrics["last10_losses"]),
                "trend": metrics["trend"],
                "prev_rank": prev_rank_map.get(team, rank),
            }
        )

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading features...")
    features = load_features()
    print(f"  {len(features)} rows for season {CURRENT_SEASON}")

    team_names = load_team_names()

    rankings = build_power_rankings(features, team_names)
    print(f"  Ranked {len(rankings)} teams")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(rankings, fh, indent=2, ensure_ascii=False)

    print(f"  Written -> {OUT_JSON}")
    for entry in rankings[:5]:
        print(f"  #{entry['rank']:2d}  {entry['team']:<4s}  {entry['team_name']:<32s}  "
              f"score={entry['composite_score']:5.1f}  net={entry['net_rating']:+.1f}  "
              f"L10={entry['last10_record']}  trend={entry['trend']}")


if __name__ == "__main__":
    main()
