"""Tests for shared odds utility functions."""
import pytest
from src.models.odds_utils import (
    american_to_decimal,
    american_to_implied_prob,
    no_vig_odds_ratio,
    expected_value,
)


class TestAmericanToDecimal:
    def test_positive_odds(self):
        assert american_to_decimal(150) == 2.5

    def test_negative_odds(self):
        assert american_to_decimal(-150) == pytest.approx(1.6667, abs=0.001)

    def test_even_money(self):
        assert american_to_decimal(100) == 2.0

    def test_large_favorite(self):
        assert american_to_decimal(-300) == pytest.approx(1.3333, abs=0.001)

    def test_large_underdog(self):
        assert american_to_decimal(300) == 4.0


class TestAmericanToImpliedProb:
    def test_favorite(self):
        assert american_to_implied_prob(-150) == pytest.approx(0.6, abs=0.001)

    def test_underdog(self):
        assert american_to_implied_prob(150) == pytest.approx(0.4, abs=0.001)

    def test_even_money(self):
        assert american_to_implied_prob(100) == pytest.approx(0.5, abs=0.001)

    def test_none_input(self):
        assert american_to_implied_prob(None) is None

    def test_heavy_favorite(self):
        assert american_to_implied_prob(-200) == pytest.approx(0.6667, abs=0.001)


class TestNoVigOddsRatio:
    def test_balanced_market(self):
        home, away = no_vig_odds_ratio(-110, -110)
        assert home == pytest.approx(0.5, abs=0.001)
        assert away == pytest.approx(0.5, abs=0.001)

    def test_favorite_underdog(self):
        home, away = no_vig_odds_ratio(-200, 170)
        assert home + away == pytest.approx(1.0, abs=0.001)
        assert home > 0.5

    def test_none_home(self):
        assert no_vig_odds_ratio(None, -110) == (None, None)

    def test_none_away(self):
        assert no_vig_odds_ratio(-110, None) == (None, None)

    def test_both_none(self):
        assert no_vig_odds_ratio(None, None) == (None, None)

    def test_symmetry(self):
        home, away = no_vig_odds_ratio(-150, 130)
        away2, home2 = no_vig_odds_ratio(130, -150)
        assert home == pytest.approx(home2, abs=0.001)
        assert away == pytest.approx(away2, abs=0.001)


class TestExpectedValue:
    def test_positive_ev(self):
        assert expected_value(0.6, 0.5) == pytest.approx(0.2, abs=0.001)

    def test_negative_ev(self):
        assert expected_value(0.4, 0.5) == pytest.approx(-0.2, abs=0.001)

    def test_zero_ev(self):
        assert expected_value(0.5, 0.5) == pytest.approx(0.0, abs=0.001)

    def test_zero_market(self):
        assert expected_value(0.5, 0.0) is None

    def test_negative_market(self):
        assert expected_value(0.5, -0.1) is None
