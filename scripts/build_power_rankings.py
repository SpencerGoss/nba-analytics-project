"""
build_power_rankings.py  --  produce dashboard/data/power_rankings.json

Ranks all 30 NBA teams using a composite score built from:
  - Net rating last-20 games    (weight 0.35)  -- recent 20-game form
  - Net rating last-10 games    (weight 0.25)  -- hottest recent stretch
  - Pythagorean win%  last-20   (weight 0.20)  -- point-differential efficiency
  - Season win%  (cum_win_pct)  (weight 0.20)  -- full-season record quality

Trend (up/down/same) compares net rating L10 vs the prior 10 games.
prev_rank is computed from the games[-20:-10] window using the same weights.
rank_delta = prev_rank - rank (positive = team improved vs last week).

Sources:
  data/features/team_game_features.csv
  data/processed/teams.csv
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

FEATURES_CSV = PROJECT_ROOT / "data" / "features" / "team_game_features.csv"
TEAMS_CSV = PROJECT_ROOT / "data" / "processed" / "teams.csv"
OUT_JSON = PROJECT_ROOT / "dashboard" / "data" / "power_rankings.json"

from src.config import get_current_season
import logging

log = logging.getLogger(__name__)

CURRENT_SEASON = get_current_season()
LAST_N = 20   # primary recency window (last 20 games)
LAST_10 = 10  # hot-streak window

# Composite weights -- must sum to 1.0
# Season record is the foundation; recent form is an adjustment, not the driver
W_WIN_PCT = 0.45   # full-season win% (anchor)
W_PYTH20 = 0.20    # pythagorean win% last 20 (point-diff quality)
W_NET20 = 0.20     # net rating last 20 (recent form)
W_NET10 = 0.15     # net rating last 10 (hot streak bonus)

# Normalisation range for composite (output 0-100 scale)
COMPOSITE_MIN = -12.0
COMPOSITE_MAX = 12.0

METHOD_DESCRIPTION = (
    "Composite score (0-100) = Season Win% (45%) + Pythagorean Win% L20 (20%) "
    "+ Net Rtg L20 (20%) + Net Rtg L10 (15%). "
    "Anchors on full-season record with recent-form adjustment. "
    "Rank delta = position change vs prior 10-game window."
)


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
    from scripts.builder_helpers import record_str
    return record_str(wins, losses)


def _normalise(value: float, lo: float, hi: float) -> float:
    """Linearly map value from [lo, hi] to [0, 100], clamped."""
    if hi == lo:
        return 50.0
    result = (value - lo) / (hi - lo) * 100.0
    return float(max(0.0, min(100.0, result)))


def _compute_composite(
    net20: float,
    net10: float,
    pyth20: float,
    win_pct_season: float,
) -> float:
    """Return a raw composite before final ranking normalisation."""
    pyth_scaled = (pyth20 - 0.5) * 24.0
    win_pct_scaled = (win_pct_season - 0.5) * 24.0
    return (
        W_NET20 * net20
        + W_NET10 * net10
        + W_PYTH20 * pyth_scaled
        + W_WIN_PCT * win_pct_scaled
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_features(season: int = CURRENT_SEASON) -> pd.DataFrame:
    cols = [
        "team_abbreviation",
        "game_date",
        "season",
        "net_rtg_game_roll10",
        "net_rtg_game_roll20",
        "win_pct_roll20",
        "cum_win_pct",
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
    last20 = tdf.tail(LAST_N)
    last10 = tdf.tail(LAST_10)
    prev10 = tdf.iloc[max(0, n - LAST_N): max(0, n - LAST_10)]
    prev_trend = tdf.iloc[max(0, n - 20): max(0, n - 10)]

    last10_wins = int((last10["win"] == 1).sum())
    last10_losses = int((last10["win"] == 0).sum())
    last20_wins = int((last20["win"] == 1).sum())
    last20_losses = int((last20["win"] == 0).sum())

    net20 = _safe_float(last20["net_rtg_game_roll20"].iloc[-1]) if not last20.empty else 0.0
    net10 = _safe_float(last10["net_rtg_game_roll10"].iloc[-1]) if not last10.empty else 0.0
    pyth20 = _safe_float(last20["win_pct_roll20"].iloc[-1]) if not last20.empty else 0.5
    win_pct_season = _safe_float(tdf["cum_win_pct"].iloc[-1]) if not tdf.empty else 0.5

    net10_prev = _safe_float(prev_trend["net_rtg_game_roll10"].iloc[-1]) if not prev_trend.empty else 0.0
    diff = net10 - net10_prev
    if diff > 1.0:
        trend = "up"
    elif diff < -1.0:
        trend = "down"
    else:
        trend = "same"

    prev_net20 = _safe_float(prev10["net_rtg_game_roll20"].iloc[-1]) if not prev10.empty else 0.0
    prev_net10 = _safe_float(prev10["net_rtg_game_roll10"].iloc[-1]) if not prev10.empty else 0.0
    prev_pyth20 = _safe_float(prev10["win_pct_roll20"].iloc[-1]) if not prev10.empty else 0.5
    prev_win_pct = _safe_float(prev10["cum_win_pct"].iloc[-1]) if not prev10.empty else 0.5

    return {
        "last10_wins": last10_wins,
        "last10_losses": last10_losses,
        "last20_wins": last20_wins,
        "last20_losses": last20_losses,
        "net20": net20,
        "net10": net10,
        "pyth20": pyth20,
        "win_pct_season": win_pct_season,
        "trend": trend,
        "composite": _compute_composite(net20, net10, pyth20, win_pct_season),
        "prev_composite": _compute_composite(prev_net20, prev_net10, prev_pyth20, prev_win_pct),
    }


# ---------------------------------------------------------------------------
# Main ranking logic
# ---------------------------------------------------------------------------

def build_power_rankings(
    features: pd.DataFrame | None = None,
    team_names: dict[str, str] | None = None,
) -> dict:
    if features is None:
        features = load_features()
    if team_names is None:
        team_names = load_team_names()

    teams = sorted(features["team_abbreviation"].dropna().unique())
    team_metrics_map: dict[str, dict] = {}

    for team in teams:
        tdf = features[features["team_abbreviation"] == team].sort_values("game_date")
        if tdf.empty:
            continue
        team_metrics_map[team] = _team_metrics(tdf)

    if not team_metrics_map:
        return {"rankings": [], "method": METHOD_DESCRIPTION, "last_updated": None}

    sorted_current = sorted(team_metrics_map.items(), key=lambda x: x[1]["composite"], reverse=True)
    sorted_prev = sorted(team_metrics_map.items(), key=lambda x: x[1]["prev_composite"], reverse=True)

    prev_rank_map = {team: idx + 1 for idx, (team, _) in enumerate(sorted_prev)}

    results = []
    for rank, (team, metrics) in enumerate(sorted_current, start=1):
        prev_rank = prev_rank_map.get(team, rank)
        rank_delta = prev_rank - rank
        results.append(
            {
                "rank": rank,
                "team": team,
                "team_name": team_names.get(team, team),
                "composite_score": round(_normalise(metrics["composite"], COMPOSITE_MIN, COMPOSITE_MAX), 1),
                "net_rating": round(metrics["net20"], 1),
                "net_rating_l10": round(metrics["net10"], 1),
                "win_pct_season": round(metrics["win_pct_season"], 3),
                "last10_record": _record_str(metrics["last10_wins"], metrics["last10_losses"]),
                "last20_record": _record_str(metrics["last20_wins"], metrics["last20_losses"]),
                "trend": metrics["trend"],
                "prev_rank": prev_rank,
                "rank_delta": rank_delta,
            }
        )

    return {
        "rankings": results,
        "method": METHOD_DESCRIPTION,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("Loading features...")
    features = load_features()
    log.info(f"  {len(features)} rows for season {CURRENT_SEASON}")

    team_names = load_team_names()

    output = build_power_rankings(features, team_names)
    rankings = output["rankings"]
    log.info(f"  Ranked {len(rankings)} teams")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, separators=(",", ":"), ensure_ascii=False)

    log.info(f"  Written -> {OUT_JSON}")
    for entry in rankings[:10]:
        delta_str = f"{entry['rank_delta']:+d}" if entry["rank_delta"] != 0 else " ="
        log.info(f"  #{entry['rank']:2d}  {entry['team']:<4s}  {entry['team_name']:<32s}  "
            f"score={entry['composite_score']:5.1f}  net20={entry['net_rating']:+.1f}  "
            f"net10={entry['net_rating_l10']:+.1f}  "
            f"win%={entry['win_pct_season']:.3f}  "
            f"L10={entry['last10_record']}  trend={entry['trend']}  delta={delta_str}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
