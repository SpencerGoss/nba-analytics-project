"""
Team Hustle Stats Feature Builder
==================================
Loads team hustle stats from raw CSV files and computes a composite hustle_index
for each team-season. These are season-level aggregates (not game-level), so
shift(1) is not needed — there is no within-game leakage risk.

Features produced (per team per season):
    contested_shots      -- total contested shots
    deflections          -- total deflections
    screen_assists       -- total screen assists
    loose_balls_recovered -- total loose balls recovered
    charges_drawn        -- total charges drawn
    box_outs             -- total box outs
    hustle_index         -- weighted z-score composite (within-season)

Data dependency:
    data/raw/team_hustle_stats/team_hustle_stats_XXXXXX.csv

Usage:
    from src.features.hustle_features import build_hustle_features
    hustle_df = build_hustle_features()
"""

import os
import glob
import re
import warnings

import numpy as np
import pandas as pd
import logging

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

HUSTLE_DATA_DIR = "data/raw/team_hustle_stats/"
OUTPUT_PATH = "data/features/hustle_features.csv"

# Columns to extract from raw CSVs (uppercase) -> snake_case rename
RAW_TO_SNAKE = {
    "CONTESTED_SHOTS": "contested_shots",
    "DEFLECTIONS": "deflections",
    "SCREEN_ASSISTS": "screen_assists",
    "LOOSE_BALLS_RECOVERED": "loose_balls_recovered",
    "CHARGES_DRAWN": "charges_drawn",
    "BOX_OUTS": "box_outs",
}

# Weights for the composite hustle_index (higher = more predictive)
HUSTLE_WEIGHTS = {
    "contested_shots": 0.25,
    "deflections": 0.25,
    "loose_balls_recovered": 0.15,
    "charges_drawn": 0.10,
    "screen_assists": 0.15,
    "box_outs": 0.10,
}

SEASON_CODE_PATTERN = re.compile(r"team_hustle_stats_(\d{6})\.csv$")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_season_code(filepath: str) -> int | None:
    """Extract the 6-digit season code from a hustle stats filename.

    Args:
        filepath: Full or relative path to a hustle stats CSV.

    Returns:
        Integer season code (e.g. 202526), or None if pattern doesn't match.
    """
    basename = os.path.basename(filepath)
    match = SEASON_CODE_PATTERN.search(basename)
    if match:
        return int(match.group(1))
    return None


def _load_all_seasons(data_dir: str) -> pd.DataFrame:
    """Load and concatenate all team hustle stats CSVs, adding season column.

    Args:
        data_dir: Directory containing team_hustle_stats_*.csv files.

    Returns:
        DataFrame with season column and all raw hustle columns.
        Empty DataFrame if no files found.
    """
    pattern = os.path.join(data_dir, "team_hustle_stats_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        return pd.DataFrame()

    dfs = []
    for filepath in files:
        season_code = _extract_season_code(filepath)
        if season_code is None:
            warnings.warn(f"Could not extract season code from {filepath}, skipping.")
            continue

        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            warnings.warn(f"Could not read {filepath}: {e}")
            continue

        # Verify required columns exist
        missing_cols = [c for c in RAW_TO_SNAKE if c not in df.columns]
        if missing_cols:
            warnings.warn(
                f"Season {season_code}: missing columns {missing_cols} in {filepath}, skipping."
            )
            continue

        df["season"] = season_code
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    return combined


def _compute_hustle_index(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a composite hustle_index from z-scored hustle stats within each season.

    Z-scores are computed within each season so that cross-season comparisons
    are relative to that season's league average (accounts for rule changes
    and tracking methodology differences across seasons).

    Args:
        df: DataFrame with season and all snake_case hustle columns.

    Returns:
        Same DataFrame with an added hustle_index column.
    """
    df = df.copy()
    hustle_cols = list(HUSTLE_WEIGHTS.keys())

    # Z-score each hustle stat within its season
    z_cols = []
    for col in hustle_cols:
        z_col = f"_z_{col}"
        z_cols.append(z_col)
        df[z_col] = df.groupby("season")[col].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0
        )

    # Weighted sum of z-scores
    df["hustle_index"] = sum(
        df[f"_z_{col}"] * weight
        for col, weight in HUSTLE_WEIGHTS.items()
    )

    # Drop intermediate z-score columns
    df = df.drop(columns=z_cols)

    return df


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def build_hustle_features(
    data_dir: str = HUSTLE_DATA_DIR,
    output_path: str | None = OUTPUT_PATH,
) -> pd.DataFrame:
    """Build per-team-season hustle features from raw hustle stats CSVs.

    Steps:
    1. Load all season CSVs from data_dir, extracting season codes from filenames
    2. Rename columns to snake_case
    3. Select key hustle columns
    4. Compute within-season z-scored composite hustle_index
    5. Return clean DataFrame

    Args:
        data_dir: Directory containing team_hustle_stats_XXXXXX.csv files.
        output_path: Path to save output CSV. Pass None to skip saving.

    Returns:
        DataFrame with columns: season, team_id, contested_shots, deflections,
        screen_assists, loose_balls_recovered, charges_drawn, box_outs, hustle_index.
        Empty DataFrame with correct schema if no files found.
    """
    output_columns = [
        "season", "team_id", "contested_shots", "deflections",
        "screen_assists", "loose_balls_recovered", "charges_drawn",
        "box_outs", "hustle_index",
    ]

    log.info(f"Loading team hustle stats from {data_dir}...")
    raw_df = _load_all_seasons(data_dir)

    if raw_df.empty:
        log.warning("  No hustle stats CSV files found. Returning empty DataFrame.")
        return pd.DataFrame(columns=output_columns)

    n_seasons = raw_df["season"].nunique()
    n_rows = len(raw_df)
    log.info(f"  Loaded {n_rows:,} team-season rows across {n_seasons} seasons.")

    # Rename raw uppercase columns to snake_case
    raw_df = raw_df.rename(columns=RAW_TO_SNAKE)

    # Select only the columns we need
    keep_cols = ["season", "TEAM_ID"] + list(RAW_TO_SNAKE.values())
    raw_df = raw_df[keep_cols].copy()
    raw_df = raw_df.rename(columns={"TEAM_ID": "team_id"})

    # Drop any duplicate (season, team_id) rows
    before = len(raw_df)
    raw_df = raw_df.drop_duplicates(subset=["season", "team_id"])
    dupes_dropped = before - len(raw_df)
    if dupes_dropped > 0:
        log.info(f"  Dropped {dupes_dropped} duplicate (season, team_id) rows.")

    # Compute composite hustle_index
    log.info("Computing hustle_index (weighted z-score composite)...")
    result = _compute_hustle_index(raw_df)

    # Ensure column order
    result = result[output_columns].reset_index(drop=True)

    # Diagnostics
    for season in sorted(result["season"].unique()):
        n_teams = len(result[result["season"] == season])
        idx_mean = result.loc[result["season"] == season, "hustle_index"].mean()
        log.info(f"  Season {season}: {n_teams} teams, hustle_index mean={idx_mean:.4f}")

    # Save output
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result.to_csv(output_path, index=False)
        log.info(f"Saved {len(result):,} rows -> {output_path}")

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = build_hustle_features()
    if df.empty:
        log.info("No hustle stats data found.")
    else:
        log.info(f"\nShape: {df.shape}")
        log.debug(df.head(10).to_string(index=False))
        log.info(f"\nNaN rates:\n{df.isna().mean()}")
