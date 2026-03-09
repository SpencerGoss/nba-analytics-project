"""
Tests for pure helper functions in src/features/player_features.py

Covers:
  - _parse_home_away: is_home flag from matchup string
  - _extract_opponent_abbr: opponent team abbreviation from matchup
  - _compute_player_rolling: shift(1) leakage guard, rolling windows, form deltas
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.player_features import (
    _parse_home_away,
    _extract_opponent_abbr,
    _compute_player_rolling,
    ROLL_WINDOWS,
)


# ---------------------------------------------------------------------------
# _parse_home_away
# ---------------------------------------------------------------------------

class TestParseHomeAway:
    def test_vs_matchup_is_home(self):
        s = pd.Series(["LAL vs. GSW"])
        result = _parse_home_away(s)
        assert result.iloc[0] == 1

    def test_at_matchup_is_away(self):
        s = pd.Series(["LAL @ BOS"])
        result = _parse_home_away(s)
        assert result.iloc[0] == 0

    def test_mixed_series(self):
        s = pd.Series(["OKC vs. MEM", "OKC @ HOU", "OKC vs. LAL"])
        result = _parse_home_away(s)
        assert list(result) == [1, 0, 1]

    def test_returns_integer_dtype(self):
        s = pd.Series(["LAL vs. BOS"])
        result = _parse_home_away(s)
        assert result.dtype == int or np.issubdtype(result.dtype, np.integer)

    def test_empty_series_returns_empty(self):
        s = pd.Series([], dtype=str)
        result = _parse_home_away(s)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# _extract_opponent_abbr
# ---------------------------------------------------------------------------

class TestExtractOpponentAbbr:
    def test_home_game_returns_opponent(self):
        """'LAL vs. GSW' -> opponent is GSW."""
        s = pd.Series(["LAL vs. GSW"])
        result = _extract_opponent_abbr(s)
        assert result.iloc[0] == "GSW"

    def test_away_game_returns_opponent(self):
        """'LAL @ BOS' -> opponent is BOS."""
        s = pd.Series(["LAL @ BOS"])
        result = _extract_opponent_abbr(s)
        assert result.iloc[0] == "BOS"

    def test_mixed_matchups(self):
        s = pd.Series(["OKC vs. MEM", "OKC @ HOU"])
        result = _extract_opponent_abbr(s)
        assert result.iloc[0] == "MEM"
        assert result.iloc[1] == "HOU"

    def test_no_leading_trailing_spaces(self):
        s = pd.Series(["GSW @ LAL"])
        result = _extract_opponent_abbr(s)
        assert result.iloc[0] == "LAL"
        assert result.iloc[0] == result.iloc[0].strip()

    def test_three_letter_abbreviation(self):
        s = pd.Series(["PHX vs. SAC"])
        result = _extract_opponent_abbr(s)
        assert len(result.iloc[0]) == 3


# ---------------------------------------------------------------------------
# _compute_player_rolling
# ---------------------------------------------------------------------------

class TestComputePlayerRolling:
    def _make_player_group(self, n=15, pts_val=20.0) -> pd.DataFrame:
        """Make a sorted player group with constant stats."""
        return pd.DataFrame({
            "game_date": pd.date_range("2026-01-01", periods=n, freq="2D"),
            "pts": [pts_val] * n,
            "reb": [5.0] * n,
            "ast": [7.0] * n,
            "min": [30.0] * n,
            "fg_pct": [0.50] * n,
            "fg3m": [2.0] * n,
            "stl": [1.0] * n,
            "blk": [0.5] * n,
            "tov": [2.0] * n,
            "plus_minus": [5.0] * n,
        })

    def test_first_row_roll5_is_nan(self):
        """With shift(1), the first row has no prior data so roll5 is NaN."""
        df = self._make_player_group()
        result = _compute_player_rolling(df, windows=ROLL_WINDOWS)
        assert pd.isna(result["pts_roll5"].iloc[0])

    def test_shift_prevents_leakage(self):
        """After shift(1), row N's rolling avg must not use row N's value."""
        # Make one game with unusually high pts
        rows = self._make_player_group(n=10, pts_val=20.0)
        # Set game 9 (last row) to 999 — should NOT appear in its own roll5
        rows = rows.copy()
        rows.loc[rows.index[-1], "pts"] = 999.0
        result = _compute_player_rolling(rows, windows=ROLL_WINDOWS)
        # The last row's pts_roll5 should NOT include 999 (which is the current row's pts)
        last_roll5 = result["pts_roll5"].iloc[-1]
        assert last_roll5 != pytest.approx(999.0)
        # In fact it should be ~20.0 (average of prior 5 games at 20)
        assert last_roll5 == pytest.approx(20.0, abs=0.01)

    def test_roll_windows_added(self):
        """All configured windows must produce columns for pts."""
        df = self._make_player_group(n=25)
        result = _compute_player_rolling(df, windows=ROLL_WINDOWS)
        for w in ROLL_WINDOWS:
            assert f"pts_roll{w}" in result.columns, f"Missing pts_roll{w}"

    def test_form_delta_columns_present(self):
        """Form delta columns (pts_form_delta etc.) must be computed."""
        df = self._make_player_group(n=25)
        result = _compute_player_rolling(df, windows=ROLL_WINDOWS)
        for col in ("pts_form_delta", "ast_form_delta", "reb_form_delta", "min_form_delta"):
            assert col in result.columns, f"Missing {col}"

    def test_pts_season_avg_column_present(self):
        df = self._make_player_group(n=10)
        result = _compute_player_rolling(df, windows=ROLL_WINDOWS)
        assert "pts_season_avg" in result.columns

    def test_output_rows_match_input_rows(self):
        df = self._make_player_group(n=12)
        result = _compute_player_rolling(df, windows=ROLL_WINDOWS)
        assert len(result) == 12

    def test_stabilized_roll5_converges_to_value(self):
        """After 6 games of 20 pts, row 6 roll5 should be ~20."""
        df = self._make_player_group(n=10, pts_val=20.0)
        result = _compute_player_rolling(df, windows=ROLL_WINDOWS)
        # Row index 5 (6th game) has 5 prior games all at 20 pts
        assert result["pts_roll5"].iloc[5] == pytest.approx(20.0, abs=0.01)
