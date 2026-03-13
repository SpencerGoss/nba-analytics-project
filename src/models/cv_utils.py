"""Shared cross-validation utilities for model training."""
from __future__ import annotations

import pandas as pd


def expanding_season_splits(
    train_df: pd.DataFrame,
    min_train_seasons: int = 6,
    season_col: str = "season",
) -> list[tuple[pd.DataFrame, pd.DataFrame, int | str]]:
    """Create expanding-window season splits for time-series CV.

    Train on seasons 0..i-1, validate on season i.
    Falls back to an 85/15 index split when there are too few seasons.

    Args:
        train_df: Full training DataFrame with a season column.
        min_train_seasons: Minimum number of training seasons before first split.
        season_col: Name of the season column.

    Returns:
        List of (train_subset, valid_subset, valid_season_label) tuples.
    """
    seasons = sorted(train_df[season_col].astype(int).unique())
    splits = []
    for i in range(max(1, min_train_seasons), len(seasons)):
        train_seasons = seasons[:i]
        valid_season = seasons[i]
        tr = train_df[train_df[season_col].astype(int).isin(train_seasons)].copy()
        va = train_df[train_df[season_col].astype(int) == valid_season].copy()
        if not tr.empty and not va.empty:
            splits.append((tr, va, valid_season))

    # fallback when dataset has very few seasons
    if not splits:
        cutoff = int(len(train_df) * 0.85)
        tr = train_df.iloc[:cutoff].copy()
        va = train_df.iloc[cutoff:].copy()
        if not tr.empty and not va.empty:
            splits.append((tr, va, "date_fallback"))
    return splits
