"""
Tests for pure helper functions in src/models/ats_backtest.py

Covers:
  - compute_roi_flat_110: ROI arithmetic at -110 standard vig
  - compute_clv_spread: CLV formula for home vs away bet side
  - _compute_edge: edge between model prob and market prob
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ats_backtest import (
    MIN_BACKTEST_GAMES,
    compute_roi_flat_110,
    compute_clv_spread,
    _compute_edge,
)


# ---------------------------------------------------------------------------
# compute_roi_flat_110
# ---------------------------------------------------------------------------

def _make_covers(n_wins: int, n_losses: int) -> pd.Series:
    """Build a covers Series with enough games to meet MIN_BACKTEST_GAMES."""
    total = n_wins + n_losses
    pad = max(0, MIN_BACKTEST_GAMES - total)
    # Pad with 50% win rate to keep the rest neutral
    pad_wins = pad // 2
    pad_losses = pad - pad_wins
    data = [1] * (n_wins + pad_wins) + [0] * (n_losses + pad_losses)
    return pd.Series(data)


class TestComputeRoiFlat110:
    def test_required_fields_in_output(self):
        covers = _make_covers(300, 200)
        result = compute_roi_flat_110(covers)
        for key in ("n_bets", "wins", "losses", "hit_rate", "net_units", "roi"):
            assert key in result, f"Missing key: {key}"

    def test_perfect_record_positive_roi(self):
        covers = _make_covers(MIN_BACKTEST_GAMES, 0)
        result = compute_roi_flat_110(covers)
        assert result["roi"] > 0.0
        assert result["wins"] == MIN_BACKTEST_GAMES
        assert result["losses"] == 0

    def test_zero_wins_negative_roi(self):
        covers = _make_covers(0, MIN_BACKTEST_GAMES)
        result = compute_roi_flat_110(covers)
        assert result["roi"] < 0.0
        assert result["wins"] == 0
        assert result["losses"] == MIN_BACKTEST_GAMES

    def test_hit_rate_50pct_is_negative_roi(self):
        """At 50% hit rate with -110 vig, ROI should be slightly negative."""
        n = MIN_BACKTEST_GAMES
        covers = _make_covers(n // 2, n - n // 2)
        result = compute_roi_flat_110(covers)
        assert result["hit_rate"] == pytest.approx(0.5, abs=0.01)
        assert result["roi"] < 0.0  # vig makes 50% a loser

    def test_nan_values_excluded(self):
        data = [1.0] * 300 + [0.0] * 200 + [float("nan")] * 100
        data.extend([1] * 200 + [0] * 200)  # pad to >= MIN_BACKTEST_GAMES
        covers = pd.Series(data)
        result = compute_roi_flat_110(covers)
        # NaN rows should not be counted in n_bets
        assert result["n_bets"] == covers.dropna().astype(int).count()

    def test_raises_when_too_few_games(self):
        covers = pd.Series([1, 0, 1])
        with pytest.raises(ValueError, match=str(MIN_BACKTEST_GAMES)):
            compute_roi_flat_110(covers)

    def test_roi_arithmetic_110_vig(self):
        """With 550 wins / 450 losses, check exact -110 vig arithmetic."""
        n_wins, n_losses = 550, 450
        pad = max(0, MIN_BACKTEST_GAMES - 1000)
        pad_wins = pad // 2
        covers = pd.Series([1] * (n_wins + pad_wins) + [0] * (n_losses + pad - pad_wins))
        result = compute_roi_flat_110(covers)
        w = result["wins"]
        l = result["losses"]
        expected_net = w * (100 / 110) - l * 1.0
        assert result["net_units"] == pytest.approx(expected_net, abs=0.01)

    def test_n_bets_equals_wins_plus_losses(self):
        covers = _make_covers(300, 200)
        result = compute_roi_flat_110(covers)
        assert result["n_bets"] == result["wins"] + result["losses"]


# ---------------------------------------------------------------------------
# compute_clv_spread
# ---------------------------------------------------------------------------

class TestComputeClvSpread:
    def test_home_positive_clv(self):
        """Home side: bet at -3.5, closed at -5.5 -> CLV = bet - close = +2.0."""
        clv = compute_clv_spread(-3.5, -5.5, "home")
        assert clv == pytest.approx(2.0)

    def test_home_negative_clv(self):
        """Home side: bet at -5.5, closed at -3.5 -> CLV = -2.0."""
        clv = compute_clv_spread(-5.5, -3.5, "home")
        assert clv == pytest.approx(-2.0)

    def test_home_zero_clv_no_movement(self):
        clv = compute_clv_spread(-3.5, -3.5, "home")
        assert clv == pytest.approx(0.0)

    def test_away_positive_clv(self):
        """Away side: bet at +3.5, closed at +5.5 -> CLV = close - bet = +2.0."""
        clv = compute_clv_spread(3.5, 5.5, "away")
        assert clv == pytest.approx(2.0)

    def test_away_negative_clv(self):
        """Away side: bet at +5.5, closed at +3.5 -> CLV = close - bet = -2.0."""
        clv = compute_clv_spread(5.5, 3.5, "away")
        assert clv == pytest.approx(-2.0)

    def test_nan_spread_returns_zero(self):
        import numpy as np
        clv = compute_clv_spread(float("nan"), -3.5, "home")
        assert clv == pytest.approx(0.0)

    def test_none_returns_zero(self):
        clv = compute_clv_spread(None, -3.5, "home")
        assert clv == pytest.approx(0.0)

    def test_invalid_type_returns_zero(self):
        clv = compute_clv_spread("bad", -3.5, "home")
        assert clv == pytest.approx(0.0)

    def test_returns_float(self):
        result = compute_clv_spread(-3.5, -5.5, "home")
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# _compute_edge
# ---------------------------------------------------------------------------

class TestComputeEdge:
    def test_positive_edge_when_model_higher(self):
        row = {"covers_spread_prob": 0.60, "home_implied_prob": 0.50}
        assert _compute_edge(row) == pytest.approx(0.10)

    def test_negative_edge_when_model_lower(self):
        row = {"covers_spread_prob": 0.40, "home_implied_prob": 0.55}
        assert _compute_edge(row) == pytest.approx(-0.15)

    def test_zero_edge_when_equal(self):
        row = {"covers_spread_prob": 0.55, "home_implied_prob": 0.55}
        assert _compute_edge(row) == pytest.approx(0.0)

    def test_nan_when_model_prob_missing(self):
        row = {"covers_spread_prob": float("nan"), "home_implied_prob": 0.55}
        assert math.isnan(_compute_edge(row))

    def test_nan_when_market_prob_missing(self):
        row = {"covers_spread_prob": 0.60, "home_implied_prob": float("nan")}
        assert math.isnan(_compute_edge(row))

    def test_nan_when_both_missing(self):
        row = {"covers_spread_prob": None, "home_implied_prob": None}
        # None is treated as NaN by pd.isna
        assert math.isnan(_compute_edge(row))
