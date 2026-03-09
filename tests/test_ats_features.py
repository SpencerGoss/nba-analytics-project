"""
Tests for src/features/ats_features.py

Covers pure functions:
  - _rolling_ats_record: shift(1) leakage prevention, window sizing, per-team grouping
  - _add_ats_engineered_features: all engineered columns are added correctly
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.ats_features import _rolling_ats_record, _add_ats_engineered_features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_team_games(
    team: str, covers: list[int], start_date: str = "2024-10-01"
) -> pd.DataFrame:
    """Build synthetic rows for a single team, one row per game."""
    dates = pd.date_range(start_date, periods=len(covers), freq="2D")
    return pd.DataFrame({
        "game_date": dates,
        "home_team": team,
        "away_team": "OPP",
        "covers_spread": covers,
    })


def _make_ats_df(n_home_games: int = 12) -> pd.DataFrame:
    """Build a minimal ATS features DataFrame for _add_ats_engineered_features."""
    rows = []
    dates = pd.date_range("2024-10-01", periods=n_home_games, freq="2D")
    for i, d in enumerate(dates):
        rows.append({
            "game_date": d,
            "home_team": "LAL",
            "away_team": "GSW",
            "spread": -3.5 - i * 0.5,
            "covers_spread": 1 if i % 2 == 0 else 0,
            "home_implied_prob": 0.60,
            "away_implied_prob": 0.40,
            "home_days_rest": 2,
            "away_days_rest": 1,
            "home_cum_win_pct": 0.55,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# _rolling_ats_record
# ---------------------------------------------------------------------------

class TestRollingAtsRecord:
    def test_first_game_is_nan(self):
        """With shift(1), the very first game has no prior data -> NaN."""
        df = _make_team_games("LAL", [1, 1, 1, 0, 0])
        result = _rolling_ats_record(df, "home_team", window=3)
        assert np.isnan(result.iloc[0])

    def test_shift1_leakage_prevention(self):
        """
        Games 1-5: covers = [1,1,1,0,0].
        For game 4 (index 3), shift(1) means we only see games 1-3.
        Rolling(3) of [NaN, 1, 1] from the home team's past -> mean of last 3 prior = 1.0.
        """
        df = _make_team_games("LAL", [1, 1, 1, 0, 0])
        result = _rolling_ats_record(df, "home_team", window=3)
        # Game at index 3 should see shift(1) covers = [NaN, 1, 1] -> 1.0
        assert result.iloc[3] == pytest.approx(1.0)

    def test_returns_series_aligned_with_input_index(self):
        df = _make_team_games("LAL", [1, 0, 1, 0, 1])
        result = _rolling_ats_record(df, "home_team", window=5)
        assert isinstance(result, pd.Series)
        assert len(result) == len(df)
        assert list(result.index) == list(df.index)

    def test_multi_team_isolated(self):
        """Each team's rolling record is computed independently."""
        df_lal = _make_team_games("LAL", [1, 1, 1, 1, 1])
        df_gsw = _make_team_games("GSW", [0, 0, 0, 0, 0])
        df = pd.concat([df_lal, df_gsw], ignore_index=True)

        result = _rolling_ats_record(df, "home_team", window=3)

        # LAL's games should have cover rates near 1.0 (after warmup)
        lal_mask = df["home_team"] == "LAL"
        gsw_mask = df["home_team"] == "GSW"

        lal_vals = result[lal_mask].dropna().values
        gsw_vals = result[gsw_mask].dropna().values

        assert np.allclose(lal_vals, 1.0)
        assert np.allclose(gsw_vals, 0.0)

    def test_values_between_0_and_1(self):
        df = _make_team_games("BOS", [1, 0, 1, 1, 0, 1])
        result = _rolling_ats_record(df, "home_team", window=3)
        non_null = result.dropna()
        assert (non_null >= 0.0).all()
        assert (non_null <= 1.0).all()

    def test_window_10_requires_more_warmup(self):
        """With window=10, fewer than 5 games may all be NaN."""
        df = _make_team_games("MIA", [1, 0, 1] * 4, "2024-10-01")  # 12 games
        result = _rolling_ats_record(df, "home_team", window=10)
        # First game is always NaN; after shift(1), min_periods=5 so may have values
        assert np.isnan(result.iloc[0])


# ---------------------------------------------------------------------------
# _add_ats_engineered_features
# ---------------------------------------------------------------------------

class TestAddAtsEngineeredFeatures:

    EXPECTED_COLS = [
        "home_ats_record_5g", "away_ats_record_5g", "diff_ats_record_5g",
        "home_ats_record_10g", "away_ats_record_10g", "diff_ats_record_10g",
        "spread_bucket", "home_dog", "rest_advantage",
        "record_vs_spread_expectation", "spread_x_home_dog", "implied_prob_gap",
    ]

    def test_all_new_columns_added(self):
        df = _make_ats_df()
        result = _add_ats_engineered_features(df)
        for col in self.EXPECTED_COLS:
            assert col in result.columns, f"Missing column: {col}"

    def test_original_rows_preserved(self):
        df = _make_ats_df(n_home_games=8)
        result = _add_ats_engineered_features(df)
        assert len(result) == len(df)

    def test_spread_bucket_small_spread(self):
        """Spread <= 3.0 -> bucket 0."""
        df = _make_ats_df()
        df["spread"] = 2.5
        result = _add_ats_engineered_features(df)
        assert (result["spread_bucket"] == 0).all()

    def test_spread_bucket_medium(self):
        """3 < spread <= 7 -> bucket 1."""
        df = _make_ats_df()
        df["spread"] = 5.0
        result = _add_ats_engineered_features(df)
        assert (result["spread_bucket"] == 1).all()

    def test_spread_bucket_large(self):
        """7 < spread <= 10 -> bucket 2."""
        df = _make_ats_df()
        df["spread"] = 9.0
        result = _add_ats_engineered_features(df)
        assert (result["spread_bucket"] == 2).all()

    def test_spread_bucket_blowout(self):
        """spread > 10 -> bucket 3."""
        df = _make_ats_df()
        df["spread"] = 12.0
        result = _add_ats_engineered_features(df)
        assert (result["spread_bucket"] == 3).all()

    def test_home_dog_flag_when_underdog(self):
        """home_implied_prob < 0.5 -> home_dog = 1.0."""
        df = _make_ats_df()
        df["home_implied_prob"] = 0.40
        df["away_implied_prob"] = 0.60
        result = _add_ats_engineered_features(df)
        assert (result["home_dog"] == 1.0).all()

    def test_home_dog_flag_when_favourite(self):
        """home_implied_prob > 0.5 -> home_dog = 0.0."""
        df = _make_ats_df()
        df["home_implied_prob"] = 0.65
        df["away_implied_prob"] = 0.35
        result = _add_ats_engineered_features(df)
        assert (result["home_dog"] == 0.0).all()

    def test_rest_advantage_computed(self):
        """rest_advantage = home_days_rest - away_days_rest."""
        df = _make_ats_df()
        df["home_days_rest"] = 3
        df["away_days_rest"] = 1
        result = _add_ats_engineered_features(df)
        assert (result["rest_advantage"] == 2).all()

    def test_implied_prob_gap_is_absolute(self):
        """implied_prob_gap = |home_implied_prob - away_implied_prob|."""
        df = _make_ats_df()
        df["home_implied_prob"] = 0.65
        df["away_implied_prob"] = 0.35
        result = _add_ats_engineered_features(df)
        assert np.allclose(result["implied_prob_gap"].dropna(), 0.30)

    def test_diff_ats_record_is_home_minus_away(self):
        """diff_ats_record_5g = home_ats_record_5g - away_ats_record_5g."""
        df = _make_ats_df()
        result = _add_ats_engineered_features(df)
        diff = result["home_ats_record_5g"] - result["away_ats_record_5g"]
        pd.testing.assert_series_equal(
            result["diff_ats_record_5g"], diff, check_names=False
        )

    def test_record_vs_spread_expectation(self):
        """record_vs_spread_expectation = home_cum_win_pct - home_implied_prob."""
        df = _make_ats_df()
        df["home_cum_win_pct"] = 0.60
        df["home_implied_prob"] = 0.55
        result = _add_ats_engineered_features(df)
        assert np.allclose(result["record_vs_spread_expectation"].dropna(), 0.05)

    def test_spread_x_home_dog_interaction(self):
        """When home is underdog (home_dog=1), spread_x_home_dog = spread * 1.0."""
        df = _make_ats_df()
        df["home_implied_prob"] = 0.40
        df["spread"] = 3.5
        result = _add_ats_engineered_features(df)
        expected = 3.5 * 1.0
        assert np.allclose(result["spread_x_home_dog"].dropna(), expected)
