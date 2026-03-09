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
