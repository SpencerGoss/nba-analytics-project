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

from src.features.referee_features import _melt_to_long_format


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
