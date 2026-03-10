"""Tests for scripts/build_player_props.py

Covers pure helper functions:
  - round_to_half: rounds to nearest 0.5
  - compute_trend_str: formats last5 vs season delta
  - last_n_values: tail N values from a player log column
  - build_props_for_player: full prop entry assembly
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_player_props import (
    round_to_half,
    compute_trend_str,
    last_n_values,
    build_props_for_player,
)


# ---------------------------------------------------------------------------
# round_to_half
# ---------------------------------------------------------------------------

class TestRoundToHalf:
    def test_exact_half_unchanged(self):
        assert round_to_half(0.5) == 0.5
        assert round_to_half(1.5) == 1.5
        assert round_to_half(22.5) == 22.5

    def test_rounds_up_to_nearest_half(self):
        assert round_to_half(22.3) == 22.5

    def test_rounds_down_to_nearest_half(self):
        assert round_to_half(22.1) == 22.0

    def test_integer_input(self):
        assert round_to_half(20.0) == 20.0

    def test_midpoint_rounds_up(self):
        # 22.25 -> floor(22.25*2 + 0.5)/2 = floor(45.0)/2 = 22.5
        assert round_to_half(22.25) == 22.5

    def test_zero(self):
        assert round_to_half(0.0) == 0.0


# ---------------------------------------------------------------------------
# compute_trend_str
# ---------------------------------------------------------------------------

class TestComputeTrendStr:
    def test_positive_trend_has_plus_sign(self):
        result = compute_trend_str(25.0, 20.0)
        assert result.startswith("+")

    def test_negative_trend_has_minus_sign(self):
        result = compute_trend_str(15.0, 20.0)
        assert result.startswith("-")

    def test_zero_diff_has_plus_sign(self):
        # diff=0 -> sign = "+" (>= 0 branch)
        result = compute_trend_str(20.0, 20.0)
        assert result.startswith("+")
        assert "0.0" in result

    def test_one_decimal_place(self):
        result = compute_trend_str(23.3, 20.0)
        # diff = 3.3 -> "+3.3"
        assert result == "+3.3"

    def test_negative_one_decimal(self):
        result = compute_trend_str(17.0, 20.0)
        assert result == "-3.0"


# ---------------------------------------------------------------------------
# last_n_values
# ---------------------------------------------------------------------------

def _make_log(dates, pts_values) -> pd.DataFrame:
    return pd.DataFrame({
        "game_date": pd.to_datetime(dates),
        "pts": pts_values,
        "reb": [5.0] * len(dates),
    })


class TestLastNValues:
    def test_returns_last_n_games(self):
        dates = pd.date_range("2025-01-01", periods=10, freq="D")
        pts = list(range(1, 11))  # 1..10
        df = _make_log(dates, pts)
        result = last_n_values(df, "pts", n=5)
        assert result == [6.0, 7.0, 8.0, 9.0, 10.0]

    def test_returns_fewer_when_not_enough_games(self):
        dates = pd.date_range("2025-01-01", periods=3, freq="D")
        df = _make_log(dates, [10.0, 15.0, 20.0])
        result = last_n_values(df, "pts", n=5)
        assert len(result) == 3

    def test_empty_df_returns_empty_list(self):
        df = pd.DataFrame({"game_date": pd.Series(dtype="datetime64[ns]"), "pts": []})
        result = last_n_values(df, "pts", n=5)
        assert result == []

    def test_values_rounded_to_one_decimal(self):
        dates = pd.date_range("2025-01-01", periods=2, freq="D")
        df = _make_log(dates, [10.333, 20.666])
        result = last_n_values(df, "pts", n=5)
        assert result == [10.3, 20.7]

    def test_sorted_by_game_date(self):
        # Rows are in reverse date order — output should still be chronological
        dates = pd.to_datetime(["2025-01-05", "2025-01-01", "2025-01-03"])
        df = pd.DataFrame({"game_date": dates, "pts": [30.0, 10.0, 20.0]})
        result = last_n_values(df, "pts", n=3)
        assert result == [10.0, 20.0, 30.0]

    def test_nans_dropped(self):
        dates = pd.date_range("2025-01-01", periods=4, freq="D")
        df = _make_log(dates, [10.0, float("nan"), 20.0, 15.0])
        result = last_n_values(df, "pts", n=5)
        # NaN is dropped — only 3 values remain
        assert float("nan") not in result
        assert len(result) == 3


# ---------------------------------------------------------------------------
# build_props_for_player
# ---------------------------------------------------------------------------

def _make_season_row(**kwargs) -> pd.Series:
    defaults = {"pts": 20.0, "reb": 8.0, "ast": 5.0}
    defaults.update(kwargs)
    return pd.Series(defaults)


def _make_player_logs(pts_values: list, n_days: int = None) -> pd.DataFrame:
    n = n_days or len(pts_values)
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "game_date": dates,
        "pts": pts_values + [float("nan")] * (n - len(pts_values)),
        "reb": [5.0] * n,
        "ast": [3.0] * n,
    })


class TestBuildPropsForPlayer:
    def test_returns_dict_with_required_keys(self):
        season_row = _make_season_row()
        logs = _make_player_logs([22.0, 24.0, 25.0, 23.0, 26.0])
        result = build_props_for_player(
            player_id=1, player_name="Test Player", team="LAL",
            season_row=season_row, player_logs=logs,
            opponent_map={}, absent_ids=set(),
        )
        assert result is not None
        for key in ["player_name", "player_id", "team", "opponent",
                    "is_injured", "has_value", "props"]:
            assert key in result, f"Missing key: {key}"

    def test_returns_none_when_no_stats(self):
        # season_row with all NaN stats
        season_row = pd.Series({"pts": float("nan"), "reb": float("nan"), "ast": float("nan")})
        logs = _make_player_logs([])
        result = build_props_for_player(
            player_id=1, player_name="No Stats", team="LAL",
            season_row=season_row, player_logs=logs,
            opponent_map={}, absent_ids=set(),
        )
        assert result is None

    def test_is_injured_set_when_in_absent_ids(self):
        season_row = _make_season_row()
        logs = _make_player_logs([20.0] * 5)
        result = build_props_for_player(
            player_id=42, player_name="Hurt Player", team="BOS",
            season_row=season_row, player_logs=logs,
            opponent_map={}, absent_ids={42},
        )
        assert result is not None
        assert result["is_injured"] is True

    def test_not_injured_when_not_in_absent_ids(self):
        season_row = _make_season_row()
        logs = _make_player_logs([20.0] * 5)
        result = build_props_for_player(
            player_id=7, player_name="Healthy Player", team="GSW",
            season_row=season_row, player_logs=logs,
            opponent_map={}, absent_ids=set(),
        )
        assert result["is_injured"] is False

    def test_opponent_wired_from_opponent_map(self):
        season_row = _make_season_row()
        logs = _make_player_logs([20.0] * 5)
        result = build_props_for_player(
            player_id=1, player_name="X", team="LAL",
            season_row=season_row, player_logs=logs,
            opponent_map={"LAL": "GSW"}, absent_ids=set(),
        )
        assert result["opponent"] == "GSW"

    def test_has_value_true_when_last5_exceeds_threshold(self):
        from scripts.build_player_props import VALUE_PCT
        season_avg = 20.0
        # last5 > season_avg * (1 + VALUE_PCT)
        high_last5 = season_avg * (1 + VALUE_PCT) + 2.0
        season_row = _make_season_row(pts=season_avg)
        logs = _make_player_logs([high_last5] * 5)
        result = build_props_for_player(
            player_id=1, player_name="Hot Player", team="LAL",
            season_row=season_row, player_logs=logs,
            opponent_map={}, absent_ids=set(),
        )
        assert result["has_value"] is True

    def test_props_list_nonempty(self):
        season_row = _make_season_row()
        logs = _make_player_logs([22.0] * 5)
        result = build_props_for_player(
            player_id=1, player_name="P", team="LAL",
            season_row=season_row, player_logs=logs,
            opponent_map={}, absent_ids=set(),
        )
        assert len(result["props"]) > 0

    def test_prop_has_required_keys(self):
        season_row = _make_season_row()
        logs = _make_player_logs([22.0] * 5)
        result = build_props_for_player(
            player_id=1, player_name="P", team="LAL",
            season_row=season_row, player_logs=logs,
            opponent_map={}, absent_ids=set(),
        )
        required = {"stat", "model_projection", "book_line", "edge",
                    "recommendation", "value", "trend", "season_avg", "last5"}
        for prop in result["props"]:
            assert required.issubset(prop.keys())
