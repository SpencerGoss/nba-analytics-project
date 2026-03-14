"""
Lineup Efficiency Features
===========================
Aggregates 5-man lineup net rating data into per-team season features
suitable for joining into the game matchup dataset.

Input: data/processed/lineup_data.csv (or raw files if processed not available)
Output: data/features/lineup_team_features.csv

Output columns per team-season:
    season              -- int e.g. 202324
    team_id             -- NBA team ID
    team_abbreviation
    top1_lineup_net_rtg -- best single lineup's net_rating (min 5 gp)
    top3_lineup_net_rtg -- avg net_rating of top 3 lineups by gp
    avg_lineup_net_rtg  -- weighted avg net_rating (weighted by minutes)
    lineup_net_rtg_std  -- std dev of lineup net ratings (depth measure)
    best_off_rating     -- highest offensive rating across lineups
    best_def_rating     -- lowest defensive rating across lineups (best defense)
    n_lineups           -- number of qualifying lineups (gp >= 5)

Leakage note:
    These features use full-season lineup data and are therefore only
    valid for the TRAINING PATH (historical games). They must never be
    used for same-season inference until after the season ends.
    Code path boundary: FR-4.4 (training path only).
"""

import os
import glob
import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────────

PROCESSED_PATH = "data/processed/lineup_data.csv"
RAW_GLOB       = "data/raw/lineups/lineup_data_*.csv"
OUTPUT_PATH    = "data/features/lineup_team_features.csv"
MIN_GP         = 5    # minimum games played for a lineup to qualify


def _load_lineup_data(lineup_csv_path: str = None) -> pd.DataFrame:
    """
    Load lineup data, preferring the processed CSV. Falls back to raw files.

    Returns a DataFrame with columns:
        season, team_id, team_abbreviation, group_name,
        gp, min, net_rating, off_rating, def_rating
    """
    path = lineup_csv_path or PROCESSED_PATH

    if os.path.exists(path):
        df = pd.read_csv(path)
        return df

    # Fall back to raw files
    raw_files = sorted(glob.glob(RAW_GLOB))
    if not raw_files:
        raise FileNotFoundError(
            f"No lineup data found at '{path}' or matching '{RAW_GLOB}'. "
            "Run preprocessing.py first or ensure raw lineup CSVs exist."
        )

    frames = []
    for f in raw_files:
        frames.append(pd.read_csv(f))
    df = pd.concat(frames, ignore_index=True)
    return df


def build_lineup_features(lineup_csv_path: str = None) -> pd.DataFrame:
    """
    Compute team-level lineup efficiency aggregates per season.

    Args:
        lineup_csv_path: Optional path override for lineup data CSV.

    Returns:
        pd.DataFrame indexed by integer position with columns:
            season, team_id, team_abbreviation,
            top1_lineup_net_rtg, top3_lineup_net_rtg, avg_lineup_net_rtg,
            lineup_net_rtg_std, best_off_rating, best_def_rating, n_lineups

        Saves to data/features/lineup_team_features.csv.
    """
    df = _load_lineup_data(lineup_csv_path)

    # Ensure numeric types
    for col in ["gp", "min", "net_rating", "off_rating", "def_rating"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filter to qualifying lineups only
    qualified = df[df["gp"] >= MIN_GP].copy()

    records = []

    for (season, team_id), grp in qualified.groupby(["season", "team_id"]):
        # Sort by games played descending (most-used lineups first)
        grp_sorted = grp.sort_values("gp", ascending=False)

        # top1: best net_rating among any qualifying lineup
        top1 = grp_sorted["net_rating"].max()

        # top3: avg net_rating of the top 3 lineups by games played
        top3_rows = grp_sorted.head(3)
        top3 = top3_rows["net_rating"].mean() if len(top3_rows) > 0 else np.nan

        # avg: weighted average net_rating by minutes (depth-weighted quality)
        total_min = grp_sorted["min"].sum()
        if total_min > 0:
            avg_net = (grp_sorted["net_rating"] * grp_sorted["min"]).sum() / total_min
        else:
            avg_net = grp_sorted["net_rating"].mean()

        # std: dispersion of lineup net ratings (team depth measure)
        std_net = grp_sorted["net_rating"].std() if len(grp_sorted) > 1 else 0.0

        # best offensive rating (highest)
        best_off = grp_sorted["off_rating"].max()

        # best defensive rating (lowest = best defense)
        best_def = grp_sorted["def_rating"].min()

        n_lineups = len(grp_sorted)
        team_abbr = grp_sorted["team_abbreviation"].iloc[0]

        records.append({
            "season":              season,
            "team_id":             team_id,
            "team_abbreviation":   team_abbr,
            "top1_lineup_net_rtg": top1,
            "top3_lineup_net_rtg": top3,
            "avg_lineup_net_rtg":  avg_net,
            "lineup_net_rtg_std":  std_net if not np.isnan(std_net) else 0.0,
            "best_off_rating":     best_off,
            "best_def_rating":     best_def,
            "n_lineups":           n_lineups,
        })

    result = pd.DataFrame(records)

    if result.empty:
        log.warning("  Warning: No qualifying lineups found (gp >= 5). Returning empty DataFrame.")
        return result

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    log.info(f"  Lineup features: {len(result)} team-seasons "
        f"({result['season'].nunique()} seasons, {result['team_id'].nunique()} teams) "
        f"-> {OUTPUT_PATH}")

    return result


if __name__ == "__main__":
    df = build_lineup_features()
    log.debug(df.to_string())
