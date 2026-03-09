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

from src.models.playoff_odds_model import bt_win_prob, HOME_ADVANTAGE


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
