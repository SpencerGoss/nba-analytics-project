"""
Tests for pure helper functions in src/models/model_explainability.py

Covers:
  - _friendly_feature_name: snake_case -> readable label conversions
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model_explainability import _friendly_feature_name


class TestFriendlyFeatureName:

    def test_home_prefix_replaced(self):
        result = _friendly_feature_name("home_net_rating")
        # "home_" -> "HM " -> title() -> "Hm "
        assert result.lower().startswith("hm ")

    def test_away_prefix_replaced(self):
        result = _friendly_feature_name("away_win_pct")
        # "away_" -> "AW " -> title() -> "Aw "
        assert result.lower().startswith("aw ")

    def test_roll5_suffix_replaced(self):
        result = _friendly_feature_name("net_rtg_roll5")
        assert "(L5)" in result

    def test_roll10_suffix_replaced(self):
        result = _friendly_feature_name("win_pct_roll10")
        assert "(L10)" in result

    def test_roll20_suffix_replaced(self):
        result = _friendly_feature_name("score_roll20")
        assert "(L20)" in result

    def test_pct_suffix_replaced(self):
        result = _friendly_feature_name("home_win_pct")
        assert "%" in result

    def test_underscores_replaced_with_spaces(self):
        result = _friendly_feature_name("net_rating")
        assert "_" not in result

    def test_title_cased(self):
        result = _friendly_feature_name("net_rating")
        # Title case means at least the first letter is uppercase
        assert result[0].isupper()

    def test_returns_string(self):
        result = _friendly_feature_name("any_feature_col")
        assert isinstance(result, str)

    def test_no_home_prefix_when_absent(self):
        result = _friendly_feature_name("net_rating")
        assert "HM" not in result
        assert "AW" not in result

    def test_complex_feature_name(self):
        """home_win_pct_roll10 should contain home prefix, %, and (L10)."""
        result = _friendly_feature_name("home_win_pct_roll10")
        assert result.lower().startswith("hm ")
        assert "%" in result
        assert "(L10)" in result

    def test_diff_feature_no_team_prefix(self):
        """diff_ features have no home/away prefix — no HM or AW in result."""
        result = _friendly_feature_name("diff_net_rtg_roll5")
        assert "Hm" not in result
        assert "Aw" not in result
        assert "(L5)" in result

    def test_away_win_pct_roll20(self):
        """away_win_pct_roll20 should start with aw, contain % and (L20)."""
        result = _friendly_feature_name("away_win_pct_roll20")
        assert result.lower().startswith("aw ")
        assert "%" in result
        assert "(L20)" in result

    def test_no_trailing_underscore_residue(self):
        """Result must not contain trailing or double spaces from underscore replacement."""
        result = _friendly_feature_name("home_net_rating")
        assert "  " not in result

    def test_non_empty_output(self):
        """Output must never be an empty string."""
        for col in ("a", "x_y", "diff_pts"):
            result = _friendly_feature_name(col)
            assert result  # truthy

    def test_home_star_player_out(self):
        """home_star_player_out should start with hm."""
        result = _friendly_feature_name("home_star_player_out")
        assert result.lower().startswith("hm ")

    def test_no_underscores_in_output(self):
        """Output must never contain underscores."""
        for col in ("home_net_rating", "away_win_pct_roll5", "diff_pts_roll10", "days_rest"):
            result = _friendly_feature_name(col)
            assert "_" not in result, f"Underscore found in result for {col!r}: {result!r}"

    def test_days_rest_readable(self):
        """home_days_rest should produce a readable label with no underscores."""
        result = _friendly_feature_name("home_days_rest")
        assert "_" not in result
        assert len(result) > 0

    def test_output_length_reasonable(self):
        """Output should be at most ~50 chars (no runaway expansion)."""
        result = _friendly_feature_name("home_win_pct_roll20")
        assert len(result) <= 50
