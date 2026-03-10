"""
Tests for pure helper functions in scripts/fetch_odds.py

Covers:
  - american_odds_to_implied_prob: American -> implied probability conversion
  - team_name_to_abb: odds API team name -> NBA abbreviation
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fetch_odds import (
    american_odds_to_implied_prob,
    team_name_to_abb,
    ODDS_TEAM_TO_ABB,
)


# ---------------------------------------------------------------------------
# american_odds_to_implied_prob
# ---------------------------------------------------------------------------

class TestAmericanOddsToImpliedProb:
    def test_even_money_is_50_pct(self):
        """American odds of -100 is even money -> 50% implied prob."""
        # -100: abs(ml)/(abs(ml)+100) = 100/200 = 0.5
        result = american_odds_to_implied_prob(-100)
        assert result == pytest.approx(0.5, abs=0.001)

    def test_heavy_favourite_near_1(self):
        """-500 favourite: 500/600 ≈ 0.8333."""
        result = american_odds_to_implied_prob(-500)
        assert result == pytest.approx(500 / 600, abs=0.001)

    def test_plus_300_underdog(self):
        """+300 underdog: 100/400 = 0.25."""
        result = american_odds_to_implied_prob(300)
        assert result == pytest.approx(0.25, abs=0.001)

    def test_plus_100_is_50_pct(self):
        """+100: 100/200 = 0.5."""
        result = american_odds_to_implied_prob(100)
        assert result == pytest.approx(0.5, abs=0.001)

    def test_none_returns_none(self):
        assert american_odds_to_implied_prob(None) is None

    def test_result_in_0_1_range(self):
        """All valid inputs must produce probabilities in (0, 1)."""
        for ml in [-800, -200, -110, 100, 110, 200, 500]:
            result = american_odds_to_implied_prob(ml)
            assert result is not None
            assert 0.0 < result < 1.0, f"ml={ml} produced {result}"

    def test_negative_larger_prob_than_positive(self):
        """Negative odds (favourite) must have higher implied prob than positive (dog)."""
        fav = american_odds_to_implied_prob(-200)
        dog = american_odds_to_implied_prob(200)
        assert fav > dog

    def test_returns_float(self):
        result = american_odds_to_implied_prob(-110)
        assert isinstance(result, float)

    def test_rounded_to_4_decimals(self):
        result = american_odds_to_implied_prob(-110)
        # Should be rounded to 4 decimal places
        assert result == round(result, 4)

    def test_minus_110_standard_vig(self):
        """-110 is standard vig line: 110/210 ≈ 0.5238."""
        result = american_odds_to_implied_prob(-110)
        assert result == pytest.approx(110 / 210, abs=0.001)


# ---------------------------------------------------------------------------
# team_name_to_abb
# ---------------------------------------------------------------------------

class TestTeamNameToAbb:
    def test_known_team_returns_abbreviation(self):
        """A known team name must return the correct 3-letter abbreviation."""
        # Pick any team from the mapping
        for name, expected_abbr in list(ODDS_TEAM_TO_ABB.items())[:3]:
            result = team_name_to_abb(name)
            assert result == expected_abbr

    def test_unknown_team_returns_original_name(self):
        """Unknown team names are passed through as-is (not raised as errors)."""
        result = team_name_to_abb("Unknown Fantasy Team")
        assert result == "Unknown Fantasy Team"

    def test_result_is_string(self):
        for name in list(ODDS_TEAM_TO_ABB.keys())[:5]:
            assert isinstance(team_name_to_abb(name), str)

    def test_all_mapped_abbreviations_are_3_chars(self):
        """All values in ODDS_TEAM_TO_ABB must be 3-character abbreviations."""
        for name, abbr in ODDS_TEAM_TO_ABB.items():
            assert len(abbr) == 3, f"{name!r} maps to non-3-char abbr {abbr!r}"

    def test_all_mapped_abbreviations_are_uppercase(self):
        for abbr in ODDS_TEAM_TO_ABB.values():
            assert abbr == abbr.upper(), f"Abbreviation {abbr!r} is not all uppercase"

    def test_lakers_maps_to_lal(self):
        assert team_name_to_abb("Los Angeles Lakers") == "LAL"

    def test_celtics_maps_to_bos(self):
        assert team_name_to_abb("Boston Celtics") == "BOS"

    def test_all_30_teams_mapped(self):
        """ODDS_TEAM_TO_ABB must cover all 30 current NBA teams."""
        assert len(ODDS_TEAM_TO_ABB) == 30

    def test_mapping_has_no_duplicate_abbreviations(self):
        """Each abbreviation must be used exactly once (no two teams share an abbr)."""
        abbrs = list(ODDS_TEAM_TO_ABB.values())
        assert len(abbrs) == len(set(abbrs)), "Duplicate abbreviations found"


class TestAmericanOddsAdditional:
    def test_minus_200_formula(self):
        """-200: 200/(200+100) = 200/300 ≈ 0.6667."""
        result = american_odds_to_implied_prob(-200)
        assert result == pytest.approx(200 / 300, abs=0.001)

    def test_vig_means_pair_sums_above_one(self):
        """Fair line pairs (-110/-110) sum > 1.0 due to bookmaker vig."""
        fav = american_odds_to_implied_prob(-110)
        dog = american_odds_to_implied_prob(-110)
        total = fav + dog
        assert total > 1.0, f"Expected vig sum > 1.0, got {total}"

    def test_plus_200_formula(self):
        """+200: 100/(200+100) = 1/3 ≈ 0.3333."""
        result = american_odds_to_implied_prob(200)
        assert result == pytest.approx(100 / 300, abs=0.001)
