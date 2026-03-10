"""
Tests for pure helper functions in src/models/playoff_odds_model.py

Covers:
  - bt_win_prob: Bradley-Terry win probability with home advantage
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.models.playoff_odds_model import bt_win_prob, HOME_ADVANTAGE, _build_matchup_row


# ---------------------------------------------------------------------------
# bt_win_prob
# ---------------------------------------------------------------------------

class TestBtWinProb:
    def test_returns_float(self):
        result = bt_win_prob(0.6, 0.5)
        assert isinstance(result, float)

    def test_home_advantage_increases_win_prob(self):
        """Home team should have higher prob than if game were neutral."""
        prob_home = bt_win_prob(0.5, 0.5, home=True)
        prob_neutral = bt_win_prob(0.5, 0.5, home=False)
        assert prob_home > prob_neutral

    def test_equal_teams_home_advantage(self):
        """Equal teams with home advantage: prob should equal HOME_ADVANTAGE."""
        prob = bt_win_prob(0.5, 0.5, home=True)
        assert prob == pytest.approx(HOME_ADVANTAGE, abs=0.01)

    def test_equal_teams_no_home_advantage_is_50pct(self):
        """Equal teams, neutral site: prob should be 0.5."""
        prob = bt_win_prob(0.5, 0.5, home=False)
        assert prob == pytest.approx(0.5, abs=0.001)

    def test_stronger_team_has_higher_prob(self):
        """Better win_pct team should have higher probability."""
        prob_strong = bt_win_prob(0.7, 0.3, home=False)
        prob_weak = bt_win_prob(0.3, 0.7, home=False)
        assert prob_strong > prob_weak

    def test_result_in_0_1_range(self):
        """All valid inputs must produce probabilities in (0, 1)."""
        test_cases = [
            (0.8, 0.3, True),
            (0.3, 0.8, True),
            (0.5, 0.5, False),
            (0.9, 0.1, False),
            (0.1, 0.9, True),
        ]
        for wp_a, wp_b, home in test_cases:
            prob = bt_win_prob(wp_a, wp_b, home)
            assert 0.0 < prob < 1.0, f"({wp_a}, {wp_b}, home={home}) -> {prob}"

    def test_extreme_win_pcts_clipped(self):
        """Extreme values (0.0, 1.0) should be clipped to avoid division by zero."""
        prob_high = bt_win_prob(1.0, 0.0, home=False)
        prob_low = bt_win_prob(0.0, 1.0, home=False)
        assert 0.0 < prob_high < 1.0
        assert 0.0 < prob_low < 1.0

    def test_symmetry_without_home_advantage(self):
        """bt_win_prob(a, b) + bt_win_prob(b, a) == 1 at neutral site."""
        prob_ab = bt_win_prob(0.6, 0.4, home=False)
        prob_ba = bt_win_prob(0.4, 0.6, home=False)
        assert prob_ab + prob_ba == pytest.approx(1.0, abs=0.001)

    def test_home_advantage_default_is_true(self):
        """Default home=True should match explicit home=True call."""
        prob_default = bt_win_prob(0.55, 0.45)
        prob_explicit = bt_win_prob(0.55, 0.45, home=True)
        assert prob_default == pytest.approx(prob_explicit, abs=0.001)

    def test_heavy_favourite_home_near_1(self):
        """Very strong team at home should have very high win probability."""
        prob = bt_win_prob(0.9, 0.1, home=True)
        assert prob > 0.95


# ---------------------------------------------------------------------------
# _build_matchup_row
# ---------------------------------------------------------------------------

class TestBuildMatchupRow:

    def _make_feats(self, **kwargs) -> pd.Series:
        return pd.Series(kwargs)

    def test_returns_single_row_dataframe(self):
        home = self._make_feats(pts=110.0)
        away = self._make_feats(pts=105.0)
        result = _build_matchup_row(home, away, feat_cols=["home_pts"])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_output_columns_match_feat_cols(self):
        home = self._make_feats(pts=110.0, reb=45.0)
        away = self._make_feats(pts=105.0, reb=42.0)
        feat_cols = ["home_pts", "away_pts", "diff_pts"]
        result = _build_matchup_row(home, away, feat_cols)
        assert list(result.columns) == feat_cols

    def test_home_prefix_reads_home_feats(self):
        home = self._make_feats(pts=115.0)
        away = self._make_feats(pts=100.0)
        result = _build_matchup_row(home, away, feat_cols=["home_pts"])
        assert result["home_pts"].iloc[0] == pytest.approx(115.0)

    def test_away_prefix_reads_away_feats(self):
        home = self._make_feats(pts=115.0)
        away = self._make_feats(pts=100.0)
        result = _build_matchup_row(home, away, feat_cols=["away_pts"])
        assert result["away_pts"].iloc[0] == pytest.approx(100.0)

    def test_diff_prefix_computes_home_minus_away(self):
        home = self._make_feats(pts=115.0)
        away = self._make_feats(pts=100.0)
        result = _build_matchup_row(home, away, feat_cols=["diff_pts"])
        assert result["diff_pts"].iloc[0] == pytest.approx(15.0)

    def test_no_prefix_prefers_home_feats(self):
        """A column without home_/away_/diff_ prefix should read from home_feats first."""
        home = self._make_feats(season=202425)
        away = self._make_feats(season=202324)  # different value
        result = _build_matchup_row(home, away, feat_cols=["season"])
        assert result["season"].iloc[0] == pytest.approx(202425)

    def test_missing_feature_filled_with_zero(self):
        """A feature column not present in either feats series should be filled with 0."""
        home = self._make_feats(pts=110.0)
        away = self._make_feats(pts=105.0)
        result = _build_matchup_row(home, away, feat_cols=["home_missing_col"])
        assert result["home_missing_col"].iloc[0] == pytest.approx(0.0)

    def test_multiple_feat_cols_all_populated(self):
        home = self._make_feats(pts=110.0, reb=45.0, ast=25.0)
        away = self._make_feats(pts=105.0, reb=42.0, ast=22.0)
        feat_cols = ["home_pts", "away_pts", "diff_pts", "home_reb", "diff_reb"]
        result = _build_matchup_row(home, away, feat_cols)
        assert result["home_pts"].iloc[0] == pytest.approx(110.0)
        assert result["away_pts"].iloc[0] == pytest.approx(105.0)
        assert result["diff_pts"].iloc[0] == pytest.approx(5.0)
        assert result["home_reb"].iloc[0] == pytest.approx(45.0)
        assert result["diff_reb"].iloc[0] == pytest.approx(3.0)
