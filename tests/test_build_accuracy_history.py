"""Tests for scripts/build_accuracy_history.py"""
import sys
from pathlib import Path
import pandas as pd
import pytest
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_accuracy_history import build_history


def _make_live_df(records: list[tuple]) -> pd.DataFrame:
    """Helper: list of (game_date, home_win_prob, actual_home_win)."""
    rows = [
        {"game_date": d, "home_win_prob": p, "actual_home_win": a}
        for d, p, a in records
    ]
    return pd.DataFrame(rows)


# ─── build_history with live predictions ────────────────────────────────────

def test_build_history_empty_returns_list():
    with patch("scripts.build_accuracy_history.load_live_predictions", return_value=pd.DataFrame()), \
         patch("scripts.build_accuracy_history.load_backtest_seasons", return_value=pd.DataFrame()):
        result = build_history()
    assert isinstance(result, list)
    assert result == []


def test_build_history_single_day():
    live = _make_live_df([
        ("2026-01-01", 0.7, 1),   # correct
        ("2026-01-01", 0.6, 1),   # correct
        ("2026-01-01", 0.4, 0),   # correct (predicted away, away won)
        ("2026-01-01", 0.8, 0),   # wrong
    ])
    with patch("scripts.build_accuracy_history.load_live_predictions", return_value=live), \
         patch("scripts.build_accuracy_history.load_backtest_seasons", return_value=pd.DataFrame()):
        result = build_history()
    assert len(result) == 1
    row = result[0]
    assert row["date"] == "2026-01-01"
    assert row["games"] == 4
    assert row["correct"] == 3
    assert row["daily_accuracy"] == pytest.approx(0.75)


def test_build_history_multi_day_rolling_accuracy():
    live = _make_live_df([
        ("2026-01-01", 0.7, 1),  # correct
        ("2026-01-01", 0.3, 1),  # wrong
        ("2026-01-02", 0.8, 1),  # correct
        ("2026-01-02", 0.6, 1),  # correct
    ])
    with patch("scripts.build_accuracy_history.load_live_predictions", return_value=live), \
         patch("scripts.build_accuracy_history.load_backtest_seasons", return_value=pd.DataFrame()):
        result = build_history()
    assert len(result) == 2
    assert result[0]["date"] == "2026-01-01"
    assert result[1]["date"] == "2026-01-02"
    # Cumulative after day 2: 3 correct / 4 games = 0.75
    assert result[1]["rolling_accuracy"] == pytest.approx(0.75)


def test_build_history_schema():
    live = _make_live_df([("2026-02-10", 0.65, 1)])
    with patch("scripts.build_accuracy_history.load_live_predictions", return_value=live), \
         patch("scripts.build_accuracy_history.load_backtest_seasons", return_value=pd.DataFrame()):
        result = build_history()
    assert len(result) == 1
    row = result[0]
    for key in ("date", "daily_accuracy", "rolling_accuracy", "games", "correct", "cumulative_games"):
        assert key in row, f"Missing key: {key}"


def test_build_history_accuracy_range():
    live = _make_live_df([
        ("2026-01-01", 0.7, 1),
        ("2026-01-02", 0.6, 0),
    ])
    with patch("scripts.build_accuracy_history.load_live_predictions", return_value=live), \
         patch("scripts.build_accuracy_history.load_backtest_seasons", return_value=pd.DataFrame()):
        result = build_history()
    for row in result:
        assert 0.0 <= row["daily_accuracy"] <= 1.0
        assert 0.0 <= row["rolling_accuracy"] <= 1.0


def test_build_history_sorted_by_date():
    live = _make_live_df([
        ("2026-01-03", 0.7, 1),
        ("2026-01-01", 0.6, 1),
        ("2026-01-02", 0.8, 0),
    ])
    with patch("scripts.build_accuracy_history.load_live_predictions", return_value=live), \
         patch("scripts.build_accuracy_history.load_backtest_seasons", return_value=pd.DataFrame()):
        result = build_history()
    dates = [r["date"] for r in result]
    assert dates == sorted(dates)


def test_build_history_cumulative_games_increases():
    live = _make_live_df([
        ("2026-01-01", 0.7, 1),
        ("2026-01-02", 0.6, 0),
        ("2026-01-03", 0.8, 1),
    ])
    with patch("scripts.build_accuracy_history.load_live_predictions", return_value=live), \
         patch("scripts.build_accuracy_history.load_backtest_seasons", return_value=pd.DataFrame()):
        result = build_history()
    cg = [r["cumulative_games"] for r in result]
    for i in range(1, len(cg)):
        assert cg[i] > cg[i-1]


def test_build_history_backtest_supplement():
    """When fewer than 5 live rows, backtest seasons are added."""
    live = _make_live_df([("2026-03-01", 0.7, 1)])
    backtest = pd.DataFrame([
        {"test_season": "202324", "accuracy": 0.67},
        {"test_season": "202425", "accuracy": 0.68},
    ])
    with patch("scripts.build_accuracy_history.load_live_predictions", return_value=live), \
         patch("scripts.build_accuracy_history.load_backtest_seasons", return_value=backtest):
        result = build_history()
    # Should have synthetic rows prepended
    assert len(result) > 1
    synthetic = [r for r in result if r.get("backtest")]
    assert len(synthetic) > 0
