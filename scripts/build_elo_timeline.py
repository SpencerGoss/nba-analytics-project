"""
build_elo_timeline.py -- produce dashboard/data/elo_timeline.json

Reads data/features/elo_ratings.csv and extracts each team's Elo rating
over time for the current season.

Output format:
{
  "teams": {
    "ATL": [{"date": "2025-10-22", "elo": 1512.3, "fast_elo": 1508.1, "elo_momentum": -4.2}, ...],
    ...
  }
}

Samples every 3rd game per team to keep JSON compact.

Run: python scripts/build_elo_timeline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

ELO_CSV = PROJECT_ROOT / "data" / "features" / "elo_ratings.csv"
OUT_JSON = PROJECT_ROOT / "dashboard" / "data" / "elo_timeline.json"

from src.config import get_current_season
import logging

log = logging.getLogger(__name__)

CURRENT_SEASON = get_current_season()
SAMPLE_EVERY_N = 3


def build_elo_timeline() -> dict:
    """Build Elo timeline JSON for the current season."""
    if not ELO_CSV.exists():
        log.warning(f"WARN: Elo ratings CSV not found at {ELO_CSV} -> writing empty elo_timeline.json")
        result: dict = {"teams": {}}
        _write_output(result)
        return result

    log.info(f"Loading Elo ratings from {ELO_CSV} ...")
    df = pd.read_csv(ELO_CSV)
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")

    # Filter to current season
    season_df = df[df["season"] == CURRENT_SEASON].copy()
    if season_df.empty:
        log.warning(f"  No data for season {CURRENT_SEASON} -> writing empty elo_timeline.json")
        result = {"teams": {}}
        _write_output(result)
        return result

    log.info(f"  {len(season_df):,} rows for season {CURRENT_SEASON}")

    # Check which columns are available
    has_fast_elo = "elo_pre_fast" in season_df.columns
    has_momentum = "elo_momentum" in season_df.columns

    if has_fast_elo:
        log.info("  fast_elo column found")
    else:
        log.warning("  fast_elo column not found -> will be null in output")

    if has_momentum:
        log.info("  elo_momentum column found")
    else:
        log.warning("  elo_momentum column not found -> will be null in output")

    # Sort by date for correct ordering
    season_df = season_df.sort_values(["team_abbreviation", "game_date"])

    teams_data: dict[str, list[dict]] = {}
    team_groups = season_df.groupby("team_abbreviation")

    for team, group in team_groups:
        team_str = str(team)
        group = group.reset_index(drop=True)

        # Sample every Nth game, always include first and last
        indices = list(range(0, len(group), SAMPLE_EVERY_N))
        last_idx = len(group) - 1
        if last_idx not in indices:
            indices.append(last_idx)

        sampled = group.iloc[indices]

        entries: list[dict] = []
        for _, row in sampled.iterrows():
            entry: dict = {
                "date": row["game_date"].strftime("%Y-%m-%d"),
                "elo": round(float(row["elo_pre"]), 1),
            }
            if has_fast_elo:
                entry["fast_elo"] = round(float(row["elo_pre_fast"]), 1)
            else:
                entry["fast_elo"] = None

            if has_momentum:
                entry["elo_momentum"] = round(float(row["elo_momentum"]), 1)
            else:
                entry["elo_momentum"] = None

            entries.append(entry)

        teams_data[team_str] = entries

    result = {"teams": teams_data}
    _write_output(result)
    log.info(f"Written -> {OUT_JSON} ({len(teams_data)} teams, "
          f"~{sum(len(v) for v in teams_data.values())} data points)")
    return result


def _write_output(data: dict) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(data, fh, separators=(",", ":"), default=str)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_elo_timeline()
