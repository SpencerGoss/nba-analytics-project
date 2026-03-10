"""
Tests for pure helper functions in src/features/referee_features.py

Covers:
  - _melt_to_long_format: wide crew format -> one row per (game, referee)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.features.referee_features import _melt_to_long_format, _compute_referee_rolling_stats


# ---------------------------------------------------------------------------
# _melt_to_long_format
# ---------------------------------------------------------------------------

def _make_crew_df(rows: list[dict]) -> pd.DataFrame:
    """Build a crew DataFrame with the expected input schema."""
    return pd.DataFrame(rows)


class TestMeltToLongFormat:
    def _base_row(self, game_id="G001", ref1="Alice Smith",
                  ref2="Bob Jones", ref3="Carol White"):
        return {
            "game_date": "2025-01-01",
            "game_id_bref": game_id,
            "home_team": "LAL",
            "away_team": "GSW",
            "referee_1": ref1,
            "referee_2": ref2,
            "referee_3": ref3,
        }

    def test_three_refs_produce_three_rows(self):
        """One game with 3 referees should produce 3 output rows."""
        df = _make_crew_df([self._base_row()])
        result = _melt_to_long_format(df)
        assert len(result) == 3

    def test_two_games_produce_six_rows(self):
        df = _make_crew_df([
            self._base_row(game_id="G001"),
            self._base_row(game_id="G002"),
        ])
        result = _melt_to_long_format(df)
        assert len(result) == 6

    def test_referee_column_present(self):
        df = _make_crew_df([self._base_row()])
        result = _melt_to_long_format(df)
        assert "referee" in result.columns

    def test_ref_slot_column_dropped(self):
        """The intermediate ref_slot column must not appear in output."""
        df = _make_crew_df([self._base_row()])
        result = _melt_to_long_format(df)
        assert "ref_slot" not in result.columns

    def test_nan_referee_dropped(self):
        """Rows with NaN referee (referee_3=None) are dropped."""
        row = self._base_row(ref3=None)
        df = _make_crew_df([row])
        result = _melt_to_long_format(df)
        assert len(result) == 2  # ref1 and ref2 only

    def test_empty_string_referee_dropped(self):
        """Empty string referee is also dropped."""
        row = self._base_row(ref3="")
        df = _make_crew_df([row])
        result = _melt_to_long_format(df)
        # Only ref1 and ref2 survive
        assert len(result) == 2

    def test_identity_columns_preserved(self):
        """game_date, game_id_bref, home_team, away_team must all be in output."""
        df = _make_crew_df([self._base_row()])
        result = _melt_to_long_format(df)
        for col in ("game_date", "game_id_bref", "home_team", "away_team"):
            assert col in result.columns, f"Missing column: {col}"

    def test_referee_names_all_present(self):
        """All three referee names from a single game must appear in output."""
        row = self._base_row(ref1="Alice Smith", ref2="Bob Jones", ref3="Carol White")
        df = _make_crew_df([row])
        result = _melt_to_long_format(df)
        names = set(result["referee"].tolist())
        assert "Alice Smith" in names
        assert "Bob Jones" in names
        assert "Carol White" in names

    def test_empty_input_returns_empty(self):
        df = pd.DataFrame(columns=["game_date", "game_id_bref", "home_team",
                                    "away_team", "referee_1", "referee_2", "referee_3"])
        result = _melt_to_long_format(df)
        assert result.empty

    def test_whitespace_only_referee_dropped(self):
        """Referee names that are only whitespace should be dropped."""
        row = self._base_row(ref3="   ")
        df = _make_crew_df([row])
        result = _melt_to_long_format(df)
        # ref3="   " stripped to "" -> dropped
        assert len(result) == 2

    def test_index_is_reset(self):
        """Output index should be a clean 0-based RangeIndex."""
        df = _make_crew_df([self._base_row(game_id="G001"),
                            self._base_row(game_id="G002")])
        result = _melt_to_long_format(df)
        assert list(result.index) == list(range(len(result)))


# ---------------------------------------------------------------------------
# _compute_referee_rolling_stats
# ---------------------------------------------------------------------------

def _make_ref_long_df(referee: str, fta_values: list[float],
                      poss_values: list[float] = None) -> pd.DataFrame:
    """Build a long-format referee DataFrame for rolling stats tests."""
    dates = pd.date_range("2024-10-01", periods=len(fta_values), freq="3D")
    poss = poss_values if poss_values is not None else [90.0] * len(fta_values)
    return pd.DataFrame({
        "game_date": dates,
        "game_id_bref": [f"G{i:03d}" for i in range(len(fta_values))],
        "referee": referee,
        "home_team": "LAL",
        "away_team": "GSW",
        "fta_per_game": fta_values,
        "avg_poss": poss,
    })


class TestComputeRefereeRollingStats:
    def test_output_has_rolling_columns(self):
        """Output must contain ref_fta_rate_roll10, ref_fta_rate_roll20, ref_poss_roll10."""
        df = _make_ref_long_df("Alice Smith", [20.0] * 15)
        result = _compute_referee_rolling_stats(df)
        for col in ["ref_fta_rate_roll10", "ref_fta_rate_roll20", "ref_poss_roll10"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_first_game_roll_is_nan_due_to_shift(self):
        """After shift(1), the first game has no prior data so roll must be NaN."""
        df = _make_ref_long_df("Alice Smith", [25.0] * 5)
        result = _compute_referee_rolling_stats(df.copy())
        first = result.sort_values("game_date").iloc[0]
        assert pd.isna(first["ref_fta_rate_roll10"]), (
            "First game should have NaN rolling rate (shift-1 with min_periods=1 "
            "still produces NaN for the very first row after shift)"
        )

    def test_shift1_before_rolling_no_leakage(self):
        """
        Referee works games at 10 FTA for 10 games then 50 FTA on game 11.
        ref_fta_rate_roll10 on game 11 should be ~10.0, NOT influenced by 50.
        """
        fta = [10.0] * 10 + [50.0]
        df = _make_ref_long_df("Bob Jones", fta)
        result = _compute_referee_rolling_stats(df.copy())
        game11 = result.sort_values("game_date").iloc[-1]
        roll = game11["ref_fta_rate_roll10"]
        if not pd.isna(roll):
            assert abs(roll - 10.0) < 1.0, (
                f"Game 11 roll10 should be ~10.0 (prior 10 games at 10 FTA), got {roll}"
            )

    def test_multiple_referees_independent(self):
        """Rolling stats for referee A must not bleed into referee B's calculations."""
        df_a = _make_ref_long_df("Ref A", [30.0] * 12)
        df_b = _make_ref_long_df("Ref B", [10.0] * 12)
        combined = pd.concat([df_a, df_b], ignore_index=True)
        result = _compute_referee_rolling_stats(combined)

        a_rates = result[result["referee"] == "Ref A"]["ref_fta_rate_roll10"].dropna()
        b_rates = result[result["referee"] == "Ref B"]["ref_fta_rate_roll10"].dropna()

        assert (a_rates > 20).all(), "Ref A rolling rates should all be ~30"
        assert (b_rates < 20).all(), "Ref B rolling rates should all be ~10"

    def test_rolling_stats_are_non_negative(self):
        """All rolling rate values should be >= 0."""
        df = _make_ref_long_df("Carol White", [15.0, 20.0, 18.0, 22.0, 10.0] * 3)
        result = _compute_referee_rolling_stats(df.copy())
        for col in ["ref_fta_rate_roll10", "ref_fta_rate_roll20", "ref_poss_roll10"]:
            non_null = result[col].dropna()
            assert (non_null >= 0).all(), f"{col} has negative values"

    def test_row_count_unchanged(self):
        """_compute_referee_rolling_stats must not add or drop rows."""
        df = _make_ref_long_df("Dave Evans", [20.0] * 8)
        result = _compute_referee_rolling_stats(df.copy())
        assert len(result) == len(df)

    def test_output_preserves_referee_column(self):
        """'referee' column must still be present in rolling stats output."""
        df = _make_ref_long_df("Eve Torres", [18.0] * 6)
        result = _compute_referee_rolling_stats(df.copy())
        assert "referee" in result.columns

    def test_rolling_values_are_float(self):
        """ref_fta_rate_roll10 values (non-null) must be float, not int."""
        df = _make_ref_long_df("Frank White", [20.0] * 15)
        result = _compute_referee_rolling_stats(df.copy())
        non_null = result["ref_fta_rate_roll10"].dropna()
        assert non_null.dtype == float or np.issubdtype(non_null.dtype, np.floating)


class TestMeltToLongFormatAdditional:
    def _base_row(self, game_id="G001", ref1="Alice Smith",
                  ref2="Bob Jones", ref3="Carol White"):
        return {
            "game_date": "2025-01-01",
            "game_id_bref": game_id,
            "home_team": "LAL",
            "away_team": "GSW",
            "referee_1": ref1,
            "referee_2": ref2,
            "referee_3": ref3,
        }

    def test_all_three_refs_nan_produces_empty(self):
        """If all three referee slots are NaN, output should be empty."""
        row = self._base_row(ref1=None, ref2=None, ref3=None)
        df = _make_crew_df([row])
        result = _melt_to_long_format(df)
        assert result.empty or len(result) == 0

    def test_referee_column_is_string_type(self):
        """referee column must hold string values (object or StringDtype)."""
        row = self._base_row()
        df = _make_crew_df([row])
        result = _melt_to_long_format(df)
        dtype = result["referee"].dtype
        assert dtype == object or pd.api.types.is_string_dtype(dtype)
