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


def test_build_history_5_plus_live_rows_no_backtest_supplement():
    """When 5+ distinct live days exist, no backtest supplement is added."""
    live = _make_live_df([
        (f"2026-01-0{i+1}", 0.65, 1) for i in range(5)
    ])
    backtest = pd.DataFrame([
        {"test_season": "202324", "accuracy": 0.67},
    ])
    with patch("scripts.build_accuracy_history.load_live_predictions", return_value=live), \
         patch("scripts.build_accuracy_history.load_backtest_seasons", return_value=backtest):
        result = build_history()
    synthetic = [r for r in result if r.get("backtest")]
    assert len(synthetic) == 0


def test_build_history_perfect_day_accuracy_is_1():
    """A day where every prediction is correct must have daily_accuracy=1.0."""
    live = _make_live_df([
        ("2026-01-05", 0.9, 1),  # home prob > 0.5, home won
        ("2026-01-05", 0.8, 1),  # home prob > 0.5, home won
        ("2026-01-05", 0.3, 0),  # home prob < 0.5, away won
    ])
    with patch("scripts.build_accuracy_history.load_live_predictions", return_value=live), \
         patch("scripts.build_accuracy_history.load_backtest_seasons", return_value=pd.DataFrame()):
        result = build_history()
    assert len(result) == 1
    assert result[0]["daily_accuracy"] == pytest.approx(1.0)


def test_build_history_zero_pct_day_accuracy_is_0():
    """A day where every prediction is wrong must have daily_accuracy=0.0."""
    live = _make_live_df([
        ("2026-01-06", 0.9, 0),  # predicted home win, away won
        ("2026-01-06", 0.8, 0),
    ])
    with patch("scripts.build_accuracy_history.load_live_predictions", return_value=live), \
         patch("scripts.build_accuracy_history.load_backtest_seasons", return_value=pd.DataFrame()):
        result = build_history()
    assert result[0]["daily_accuracy"] == pytest.approx(0.0)


def test_build_history_correct_field_is_int():
    """The 'correct' field in each row should be an integer, not a float."""
    live = _make_live_df([("2026-02-01", 0.7, 1), ("2026-02-01", 0.4, 0)])
    with patch("scripts.build_accuracy_history.load_live_predictions", return_value=live), \
         patch("scripts.build_accuracy_history.load_backtest_seasons", return_value=pd.DataFrame()):
        result = build_history()
    assert isinstance(result[0]["correct"], int)
    assert isinstance(result[0]["games"], int)


def test_build_history_boundary_exactly_half_prob():
    """home_win_prob=0.5 is treated as predicting home win (>= 0.5 threshold)."""
    live = _make_live_df([("2026-02-15", 0.5, 1)])  # prob=0.5, home wins -> correct
    with patch("scripts.build_accuracy_history.load_live_predictions", return_value=live), \
         patch("scripts.build_accuracy_history.load_backtest_seasons", return_value=pd.DataFrame()):
        result = build_history()
    assert result[0]["correct"] == 1
    assert result[0]["daily_accuracy"] == pytest.approx(1.0)


def test_build_history_cumulative_games_equals_running_sum():
    """cumulative_games must equal the running total of all prior days' games."""
    live = _make_live_df([
        ("2026-01-01", 0.7, 1),
        ("2026-01-01", 0.6, 1),
        ("2026-01-02", 0.8, 0),
        ("2026-01-03", 0.55, 1),
        ("2026-01-03", 0.45, 0),
        ("2026-01-03", 0.65, 1),
    ])
    with patch("scripts.build_accuracy_history.load_live_predictions", return_value=live), \
         patch("scripts.build_accuracy_history.load_backtest_seasons", return_value=pd.DataFrame()):
        result = build_history()
    running = 0
    for row in result:
        running += row["games"]
        assert row["cumulative_games"] == running


def test_build_history_empty_backtest_live_under_5():
    """When live < 5 rows but backtest is empty, only live rows are returned (no crash)."""
    live = _make_live_df([("2026-03-01", 0.7, 1), ("2026-03-02", 0.6, 0)])
    with patch("scripts.build_accuracy_history.load_live_predictions", return_value=live), \
         patch("scripts.build_accuracy_history.load_backtest_seasons", return_value=pd.DataFrame()):
        result = build_history()
    assert len(result) == 2
    assert all(not r.get("backtest") for r in result)


def test_build_history_backtest_rows_before_first_live_date():
    """All synthetic backtest rows must be dated strictly before the first live date."""
    live = _make_live_df([("2026-03-01", 0.7, 1)])
    backtest = pd.DataFrame([
        {"test_season": "202223", "accuracy": 0.66},
        {"test_season": "202324", "accuracy": 0.67},
        {"test_season": "202425", "accuracy": 0.68},
    ])
    with patch("scripts.build_accuracy_history.load_live_predictions", return_value=live), \
         patch("scripts.build_accuracy_history.load_backtest_seasons", return_value=backtest):
        result = build_history()
    first_live = "2026-03-01"
    for row in result:
        if row.get("backtest"):
            assert row["date"] < first_live, (
                f"Synthetic row dated {row['date']} is not before first live date {first_live}"
            )


def test_build_history_rolling_accuracy_three_days():
    """rolling_accuracy after 3 days is cumulative correct / cumulative games."""
    live = _make_live_df([
        ("2026-01-01", 0.7, 1),   # correct
        ("2026-01-02", 0.3, 1),   # wrong
        ("2026-01-03", 0.8, 1),   # correct
        ("2026-01-03", 0.6, 0),   # wrong
    ])
    with patch("scripts.build_accuracy_history.load_live_predictions", return_value=live), \
         patch("scripts.build_accuracy_history.load_backtest_seasons", return_value=pd.DataFrame()):
        result = build_history()
    # After day 3: 2 correct / 4 total = 0.5
    assert result[-1]["rolling_accuracy"] == pytest.approx(0.5)
    assert result[-1]["cumulative_games"] == 4


def test_build_history_backtest_synthetic_accuracy_matches_row():
    """Each synthetic backtest row's daily_accuracy must match the source backtest accuracy."""
    live = _make_live_df([("2026-03-10", 0.7, 1)])
    backtest = pd.DataFrame([
        {"test_season": "202324", "accuracy": 0.671},
    ])
    with patch("scripts.build_accuracy_history.load_live_predictions", return_value=live), \
         patch("scripts.build_accuracy_history.load_backtest_seasons", return_value=backtest):
        result = build_history()
    synthetic = [r for r in result if r.get("backtest")]
    assert len(synthetic) == 1
    assert synthetic[0]["daily_accuracy"] == pytest.approx(0.671, abs=1e-3)


def test_build_history_daily_accuracy_is_float():
    """daily_accuracy in each row must be a Python float, not int or str."""
    live = _make_live_df([("2026-01-01", 0.7, 1), ("2026-01-01", 0.6, 0)])
    with patch("scripts.build_accuracy_history.load_live_predictions", return_value=live), \
         patch("scripts.build_accuracy_history.load_backtest_seasons", return_value=pd.DataFrame()):
        result = build_history()
    assert isinstance(result[0]["daily_accuracy"], float)


def test_build_history_rolling_accuracy_present_in_first_row():
    """rolling_accuracy must be present in the very first row."""
    live = _make_live_df([("2026-01-01", 0.8, 1)])
    with patch("scripts.build_accuracy_history.load_live_predictions", return_value=live), \
         patch("scripts.build_accuracy_history.load_backtest_seasons", return_value=pd.DataFrame()):
        result = build_history()
    assert "rolling_accuracy" in result[0]


def test_build_history_games_always_at_least_one():
    """Each date row must have at least one game (no zero-game rows)."""
    live = _make_live_df([
        ("2026-01-01", 0.7, 1),
        ("2026-01-02", 0.6, 0),
        ("2026-01-03", 0.8, 1),
    ])
    with patch("scripts.build_accuracy_history.load_live_predictions", return_value=live), \
         patch("scripts.build_accuracy_history.load_backtest_seasons", return_value=pd.DataFrame()):
        result = build_history()
    for row in result:
        assert row["games"] >= 1
