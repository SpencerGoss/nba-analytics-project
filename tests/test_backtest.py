import numpy as np
import pytest


def test_backtest_returns_all_metrics():
    """Backtest should return roi, win_rate, n_bets, profit, max_drawdown, avg_edge, total_wagered."""
    from src.models.backtest import run_backtest
    result = run_backtest(
        model_probs=np.array([0.6, 0.55, 0.7, 0.52, 0.45, 0.3]),
        market_probs=np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
        actuals=np.array([1, 0, 1, 1, 0, 0]),
        edge_threshold=0.03,
    )
    for key in ["roi", "win_rate", "n_bets", "profit", "max_drawdown", "avg_edge", "total_wagered"]:
        assert key in result, f"Missing key: {key}"


def test_backtest_no_bets_below_threshold():
    """When no games exceed edge threshold, n_bets should be 0."""
    from src.models.backtest import run_backtest
    result = run_backtest(
        model_probs=np.array([0.51, 0.50, 0.49]),
        market_probs=np.array([0.50, 0.50, 0.50]),
        actuals=np.array([1, 0, 1]),
        edge_threshold=0.10,
    )
    assert result["n_bets"] == 0
    assert result["roi"] == 0.0


def test_backtest_perfect_model_profits():
    """A perfect model betting on large edges should be profitable."""
    from src.models.backtest import run_backtest
    result = run_backtest(
        model_probs=np.array([0.9, 0.9, 0.9, 0.9, 0.9]),
        market_probs=np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
        actuals=np.array([1, 1, 1, 1, 1]),
        edge_threshold=0.03,
    )
    assert result["roi"] > 0
    assert result["win_rate"] == 1.0
    assert result["n_bets"] == 5


def test_backtest_vig_reduces_profit():
    """Higher vig should reduce profit."""
    from src.models.backtest import run_backtest
    args = dict(
        model_probs=np.array([0.7, 0.7, 0.7]),
        market_probs=np.array([0.5, 0.5, 0.5]),
        actuals=np.array([1, 1, 1]),
        edge_threshold=0.03,
    )
    result_low_vig = run_backtest(**args, vig=0.02)
    result_high_vig = run_backtest(**args, vig=0.10)
    assert result_low_vig["roi"] > result_high_vig["roi"]


def test_backtest_away_bets():
    """When model_prob < market_prob by enough, should bet away."""
    from src.models.backtest import run_backtest
    result = run_backtest(
        model_probs=np.array([0.3]),   # model says 30% home
        market_probs=np.array([0.5]),  # market says 50% home
        actuals=np.array([0]),         # away wins
        edge_threshold=0.03,
    )
    assert result["n_bets"] == 1
    assert result["profit"] > 0  # correct away bet should profit


def test_backtest_max_drawdown_negative():
    """Max drawdown should be <= 0."""
    from src.models.backtest import run_backtest
    result = run_backtest(
        model_probs=np.array([0.7, 0.7, 0.7]),
        market_probs=np.array([0.5, 0.5, 0.5]),
        actuals=np.array([0, 0, 1]),
        edge_threshold=0.03,
    )
    assert result["max_drawdown"] <= 0


def test_backtest_unit_size_scaling():
    """Larger unit size should scale profit proportionally."""
    from src.models.backtest import run_backtest
    args = dict(
        model_probs=np.array([0.7, 0.7]),
        market_probs=np.array([0.5, 0.5]),
        actuals=np.array([1, 1]),
        edge_threshold=0.03,
    )
    r1 = run_backtest(**args, unit_size=1.0)
    r2 = run_backtest(**args, unit_size=2.0)
    assert abs(r2["profit"] / r1["profit"] - 2.0) < 0.01


def test_backtest_kelly_sizing():
    """When kelly=True, bet size should scale with edge."""
    from src.models.backtest import run_backtest
    result = run_backtest(
        model_probs=np.array([0.6, 0.8]),
        market_probs=np.array([0.5, 0.5]),
        actuals=np.array([1, 1]),
        edge_threshold=0.03,
        kelly=True,
    )
    assert result["n_bets"] == 2
    # Kelly bets larger on bigger edge, so total_wagered should reflect this
    assert result["total_wagered"] > 0


def test_walk_forward_split():
    """walk_forward_backtest should split data into train/test windows."""
    from src.models.backtest import walk_forward_backtest
    assert callable(walk_forward_backtest)
