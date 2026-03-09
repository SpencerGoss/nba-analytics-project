"""
Tests for src/features/era_labels.py

Covers:
  - _to_season_int: normalization of various season formats
  - get_era: single-season era lookup
  - label_eras: DataFrame labeling with era columns
  - ERA_DEFINITIONS: completeness and ordering
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.era_labels import (
    ERA_DEFINITIONS,
    _to_season_int,
    get_era,
    label_eras,
)


# ---------------------------------------------------------------------------
# _to_season_int
# ---------------------------------------------------------------------------

class TestToSeasonInt:
    def test_6digit_string(self):
        assert _to_season_int("202425") == 202425

    def test_6digit_integer(self):
        assert _to_season_int(202425) == 202425

    def test_hyphenated_string(self):
        assert _to_season_int("2024-25") == 202425

    def test_hyphenated_historical(self):
        assert _to_season_int("1979-80") == 197980

    def test_old_season_int(self):
        assert _to_season_int(197980) == 197980

    def test_short_form_stripped(self):
        """'24-25' after removing '-' gives '2425' (4 chars) -> prepend '20'."""
        result = _to_season_int("24-25")
        assert result == 202425

    def test_invalid_returns_zero(self):
        assert _to_season_int("notaseason") == 0

    def test_string_with_spaces_stripped(self):
        assert _to_season_int("  202425  ") == 202425


# ---------------------------------------------------------------------------
# get_era
# ---------------------------------------------------------------------------

class TestGetEra:
    def test_current_era_is_3pt_revolution(self):
        era = get_era("202425")
        assert era["era_num"] == 6
        assert "3-Point" in era["era_name"]

    def test_shot_clock_era(self):
        era = get_era("196364")
        assert era["era_num"] == 2

    def test_3pt_introduction_era(self):
        era = get_era("198182")
        assert era["era_num"] == 3

    def test_physical_isolation_era(self):
        era = get_era("199596")
        assert era["era_num"] == 4

    def test_open_court_era(self):
        era = get_era("201011")
        assert era["era_num"] == 5

    def test_foundational_era(self):
        era = get_era("194849")
        assert era["era_num"] == 1

    def test_era_boundary_start(self):
        """First season of 3-point era (1979-80) maps to era 3."""
        era = get_era("197980")
        assert era["era_num"] == 3

    def test_era_boundary_end(self):
        """Last season of Physical era (2003-04) maps to era 4."""
        era = get_era("200304")
        assert era["era_num"] == 4

    def test_unknown_out_of_range(self):
        """Season before any era definition returns Unknown."""
        era = get_era("194546")
        assert era["era_num"] == 0
        assert era["era_name"] == "Unknown"

    def test_returns_dict(self):
        era = get_era("202425")
        assert isinstance(era, dict)

    def test_has_description_key(self):
        era = get_era("202425")
        assert "description" in era

    def test_integer_season_input(self):
        era = get_era(202425)
        assert era["era_num"] == 6


# ---------------------------------------------------------------------------
# label_eras
# ---------------------------------------------------------------------------

class TestLabelEras:
    def _make_df(self, seasons: list) -> pd.DataFrame:
        return pd.DataFrame({"season": seasons})

    def test_adds_four_era_columns(self):
        df = self._make_df(["202425"])
        result = label_eras(df)
        for col in ("era_num", "era_name", "era_start", "era_end"):
            assert col in result.columns, f"Missing column: {col}"

    def test_current_season_labeled_correctly(self):
        df = self._make_df(["202425"])
        result = label_eras(df)
        assert result["era_num"].iloc[0] == 6

    def test_multiple_seasons_different_eras(self):
        df = self._make_df(["197980", "199495", "202425"])
        result = label_eras(df)
        assert result["era_num"].iloc[0] == 3  # 3PT Introduction
        assert result["era_num"].iloc[1] == 4  # Physical/ISO
        assert result["era_num"].iloc[2] == 6  # 3PT Revolution

    def test_original_columns_preserved(self):
        df = pd.DataFrame({"season": ["202425"], "value": [42]})
        result = label_eras(df)
        assert "value" in result.columns
        assert result["value"].iloc[0] == 42

    def test_raises_keyerror_when_season_col_missing(self):
        df = pd.DataFrame({"year": [202425]})
        with pytest.raises(KeyError):
            label_eras(df)

    def test_custom_season_col_name(self):
        df = pd.DataFrame({"my_season": ["202425"]})
        result = label_eras(df, season_col="my_season")
        assert result["era_num"].iloc[0] == 6

    def test_era_name_is_string(self):
        df = self._make_df(["202425"])
        result = label_eras(df)
        assert isinstance(result["era_name"].iloc[0], str)

    def test_era_num_is_integer(self):
        df = self._make_df(["202425"])
        result = label_eras(df)
        assert result["era_num"].iloc[0] == int(result["era_num"].iloc[0])

    def test_unknown_season_gets_zero_era_num(self):
        df = self._make_df(["194546"])
        result = label_eras(df)
        assert result["era_num"].iloc[0] == 0

    def test_integer_season_column_works(self):
        df = pd.DataFrame({"season": [202425, 199495]})
        result = label_eras(df)
        assert result["era_num"].iloc[0] == 6
        assert result["era_num"].iloc[1] == 4


# ---------------------------------------------------------------------------
# ERA_DEFINITIONS integrity
# ---------------------------------------------------------------------------

class TestEraDefinitions:
    def test_has_six_eras(self):
        assert len(ERA_DEFINITIONS) == 6

    def test_era_nums_1_to_6(self):
        nums = sorted(ERA_DEFINITIONS["era_num"].tolist())
        assert nums == [1, 2, 3, 4, 5, 6]

    def test_eras_are_ordered_chronologically(self):
        starts = ERA_DEFINITIONS.sort_values("era_num")["era_start"].tolist()
        assert starts == sorted(starts)

    def test_no_gaps_between_eras(self):
        """Each era's end+1 should equal the next era's start year."""
        sorted_eras = ERA_DEFINITIONS.sort_values("era_start")
        starts = sorted_eras["era_start"].tolist()
        ends = sorted_eras["era_end"].tolist()
        # Check that ends and next starts are adjacent years
        # Era N ends at YYYYYY (e.g., 195354), Era N+1 starts at YYYYYY+101 (e.g., 195455)
        for i in range(len(starts) - 1):
            end_year = int(str(ends[i])[:4])
            next_start_year = int(str(starts[i + 1])[:4])
            assert next_start_year == end_year + 1, (
                f"Gap between era ending at {ends[i]} and next starting at {starts[i+1]}"
            )

    def test_required_columns_present(self):
        for col in ("era_num", "era_name", "era_start", "era_end", "description"):
            assert col in ERA_DEFINITIONS.columns

    def test_last_era_is_open_ended(self):
        """The current era should have era_end=999999 (open-ended)."""
        last = ERA_DEFINITIONS.sort_values("era_num").iloc[-1]
        assert last["era_end"] == 999999
