"""
Data integrity smoke tests for pipeline output files.

These tests validate pipeline invariants on the ACTUAL processed/feature CSV
files produced by the pipeline. All tests skip gracefully when the files do
not exist (e.g. on a fresh clone before the pipeline has been run).

Invariants checked:
  - game_matchup_features.csv  : unique game_id, integer season, home_win column
  - player_game_features.csv   : unique (game_id, player_id), integer season
  - player_stats.csv           : player_id non-null, season column exists
  - Rolling feature leakage    : rolling columns must not equal same-row raw stat
"""

import os
import pytest
import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")

MATCHUP_PATH  = os.path.join(PROJECT_ROOT, "data", "features", "game_matchup_features.csv")
PGF_PATH      = os.path.join(PROJECT_ROOT, "data", "features", "player_game_features.csv")
PSTATS_PATH   = os.path.join(PROJECT_ROOT, "data", "processed", "player_stats.csv")

MATCHUP_EXISTS = os.path.exists(MATCHUP_PATH)
PGF_EXISTS     = os.path.exists(PGF_PATH)
PSTATS_EXISTS  = os.path.exists(PSTATS_PATH)


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def matchup_df():
    """Load game_matchup_features.csv once per module."""
    return pd.read_csv(MATCHUP_PATH)


@pytest.fixture(scope="module")
def pgf_df():
    """Load player_game_features.csv once per module."""
    return pd.read_csv(PGF_PATH)


@pytest.fixture(scope="module")
def pstats_df():
    """Load player_stats.csv once per module."""
    return pd.read_csv(PSTATS_PATH)


# ── game_matchup_features.csv ──────────────────────────────────────────────────


@pytest.mark.skipif(not MATCHUP_EXISTS, reason="game_matchup_features.csv not found — run pipeline first")
class TestMatchupFeatures:

    def test_game_id_is_unique(self, matchup_df):
        """Each row should represent exactly one game — no duplicate game_ids."""
        n_total = len(matchup_df)
        n_unique = matchup_df["game_id"].nunique()
        assert n_unique == n_total, (
            f"game_matchup_features.csv has {n_total - n_unique} duplicate game_ids"
        )

    def test_season_is_integer_dtype(self, matchup_df):
        """Season column must be stored as an integer (e.g. 202425, not '2024-25')."""
        dtype = matchup_df["season"].dtype
        assert pd.api.types.is_integer_dtype(dtype), (
            f"season column dtype is {dtype!r}; expected an integer dtype"
        )

    def test_home_win_column_exists(self, matchup_df):
        """home_win must be present — it is the supervised learning target."""
        assert "home_win" in matchup_df.columns, (
            "home_win column missing from game_matchup_features.csv"
        )

    def test_home_win_contains_only_valid_values(self, matchup_df):
        """home_win must be 0, 1, or NaN — no other values allowed."""
        col = matchup_df["home_win"].dropna()
        bad = col[~col.isin([0, 1])]
        assert len(bad) == 0, (
            f"home_win contains {len(bad)} invalid values: {bad.unique().tolist()}"
        )

    def test_no_null_game_id(self, matchup_df):
        """game_id must never be null — it is the primary key."""
        n_null = matchup_df["game_id"].isna().sum()
        assert n_null == 0, f"game_matchup_features.csv has {n_null} null game_ids"

    def test_season_values_are_six_digit_integers(self, matchup_df):
        """Season codes must be six-digit integers like 202425."""
        invalid = matchup_df["season"].dropna()
        invalid = invalid[~invalid.between(190001, 209999)]
        assert len(invalid) == 0, (
            f"game_matchup_features.csv has {len(invalid)} season values outside expected range: "
            f"{invalid.unique().tolist()[:5]}"
        )


# ── player_game_features.csv ───────────────────────────────────────────────────


@pytest.mark.skipif(not PGF_EXISTS, reason="player_game_features.csv not found — run pipeline first")
class TestPlayerGameFeatures:

    def test_game_id_player_id_pair_is_unique(self, pgf_df):
        """Each (game_id, player_id) pair must appear exactly once."""
        n_total = len(pgf_df)
        n_unique = pgf_df.drop_duplicates(subset=["game_id", "player_id"]).shape[0]
        assert n_unique == n_total, (
            f"player_game_features.csv has {n_total - n_unique} duplicate (game_id, player_id) pairs"
        )

    def test_season_is_integer_dtype(self, pgf_df):
        """Season column must be an integer dtype."""
        dtype = pgf_df["season"].dtype
        assert pd.api.types.is_integer_dtype(dtype), (
            f"season column dtype is {dtype!r}; expected an integer dtype"
        )

    def test_no_null_player_id(self, pgf_df):
        """player_id must never be null."""
        n_null = pgf_df["player_id"].isna().sum()
        assert n_null == 0, f"player_game_features.csv has {n_null} null player_ids"

    def test_no_null_game_id(self, pgf_df):
        """game_id must never be null."""
        n_null = pgf_df["game_id"].isna().sum()
        assert n_null == 0, f"player_game_features.csv has {n_null} null game_ids"

    def test_season_values_are_six_digit_integers(self, pgf_df):
        """Season codes must be six-digit integers like 202425."""
        invalid = pgf_df["season"].dropna()
        invalid = invalid[~invalid.between(190001, 209999)]
        assert len(invalid) == 0, (
            f"player_game_features.csv has {len(invalid)} season values outside expected range: "
            f"{invalid.unique().tolist()[:5]}"
        )


# ── player_stats.csv ───────────────────────────────────────────────────────────


@pytest.mark.skipif(not PSTATS_EXISTS, reason="player_stats.csv not found — run pipeline first")
class TestPlayerStats:

    def test_player_id_not_all_null(self, pstats_df):
        """player_id column must have at least one non-null value."""
        assert pstats_df["player_id"].notna().sum() > 0, (
            "player_stats.csv: player_id is entirely null"
        )

    def test_season_column_exists(self, pstats_df):
        """season column must be present in player_stats.csv."""
        assert "season" in pstats_df.columns, (
            "player_stats.csv is missing the 'season' column"
        )

    def test_player_id_column_exists(self, pstats_df):
        """player_id column must be present."""
        assert "player_id" in pstats_df.columns, (
            "player_stats.csv is missing the 'player_id' column"
        )


# ── Rolling feature leakage spot-check ────────────────────────────────────────


@pytest.mark.skipif(not PGF_EXISTS, reason="player_game_features.csv not found — run pipeline first")
class TestRollingFeatureLeakage:
    """
    Verify that rolling window features do not include same-game values.

    The rolling mean over the prior N games must differ from the current
    game's raw stat for the vast majority of rows (same value is only
    expected at sequence boundaries, e.g. the very first game, or when a
    player's rolling average happens to equal their current stat by coincidence).

    We check that at least 50% of rows (with both values present) differ —
    a much looser bound than the true expectation (>95%) but robust to
    short runs, degenerate players, and near-constant stats.
    """

    def test_pts_roll5_differs_from_current_pts(self, pgf_df):
        """pts_roll5 should not simply echo the current game's pts."""
        _assert_rolling_differs_from_raw(pgf_df, raw_col="pts", roll_col="pts_roll5")

    def test_reb_roll5_differs_from_current_reb(self, pgf_df):
        """reb_roll5 should not simply echo the current game's reb."""
        _assert_rolling_differs_from_raw(pgf_df, raw_col="reb", roll_col="reb_roll5")

    def test_ast_roll5_differs_from_current_ast(self, pgf_df):
        """ast_roll5 should not simply echo the current game's ast."""
        _assert_rolling_differs_from_raw(pgf_df, raw_col="ast", roll_col="ast_roll5")


def _assert_rolling_differs_from_raw(
    df: pd.DataFrame, raw_col: str, roll_col: str, min_differ_fraction: float = 0.50
) -> None:
    """
    Assert that ``roll_col`` is NOT identical to ``raw_col`` in the majority of rows.

    If the rolling column equals the raw stat in more than (1 - min_differ_fraction)
    of rows, it is very likely that shift(1) was omitted and current-game data leaked
    into the feature.
    """
    if raw_col not in df.columns or roll_col not in df.columns:
        pytest.skip(f"Columns {raw_col!r} or {roll_col!r} not present — skipping leakage check")

    both_present = df[[raw_col, roll_col]].dropna()
    if len(both_present) == 0:
        pytest.skip(f"No rows with both {raw_col!r} and {roll_col!r} present")

    n_equal = (both_present[raw_col] == both_present[roll_col]).sum()
    fraction_equal = n_equal / len(both_present)
    fraction_differ = 1.0 - fraction_equal

    assert fraction_differ >= min_differ_fraction, (
        f"Rolling leakage detected: {roll_col!r} equals {raw_col!r} in "
        f"{fraction_equal:.1%} of rows (expected < {1 - min_differ_fraction:.0%}). "
        f"Verify that shift(1) is applied before rolling window in player_performance_model.py "
        f"or wherever {roll_col!r} is computed."
    )
