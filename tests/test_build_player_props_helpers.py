"""
Tests for pure helper functions in scripts/build_player_props.py

Covers:
  - round_to_half: rounding to nearest 0.5
  - compute_trend_str: +/- trend string formatting
  - last_n_values: sorted last-N values from player log
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_player_props import (
    round_to_half,
    compute_trend_str,
    last_n_values,
    LAST_N,
)


# ---------------------------------------------------------------------------
# round_to_half
# ---------------------------------------------------------------------------

class TestRoundToHalf:
    def test_integer_unchanged(self):
        assert round_to_half(10.0) == 10.0

    def test_rounds_up_to_half(self):
        # 10.3 -> nearest 0.5 is 10.5
        assert round_to_half(10.3) == 10.5

    def test_rounds_down_to_integer(self):
        # 10.1 -> nearest 0.5 is 10.0
        assert round_to_half(10.1) == 10.0

    def test_exact_half_unchanged(self):
        assert round_to_half(10.5) == 10.5

    def test_zero(self):
        assert round_to_half(0.0) == 0.0

    def test_negative_value(self):
        # -2.3 -> floor((-2.3)*2 + 0.5) / 2 = floor(-4.1) / 2 = -5/2 = -2.5
        assert round_to_half(-2.3) == pytest.approx(-2.5)

    def test_returns_float(self):
        result = round_to_half(7.7)
        assert isinstance(result, float)

    def test_large_value(self):
        # 99.7 is below 99.75 midpoint, so rounds DOWN to 99.5
        assert round_to_half(99.7) == pytest.approx(99.5)
        # 99.8 is above midpoint, rounds UP to 100.0
        assert round_to_half(99.8) == pytest.approx(100.0)

    def test_step_0_75_rounds_to_1(self):
        # 0.75 -> nearest 0.5 is 1.0
        assert round_to_half(0.75) == 1.0

    def test_step_0_25_rounds_to_half(self):
        # 0.25 -> nearest 0.5 is 0.5
        assert round_to_half(0.25) == 0.5


# ---------------------------------------------------------------------------
# compute_trend_str
# ---------------------------------------------------------------------------

class TestComputeTrendStr:
    def test_positive_trend_has_plus(self):
        result = compute_trend_str(25.0, 20.0)
        assert result.startswith("+")

    def test_negative_trend_no_plus(self):
        result = compute_trend_str(15.0, 20.0)
        assert result.startswith("-")

    def test_zero_diff_has_plus(self):
        """Exactly equal: diff=0.0, sign is '+' by convention."""
        result = compute_trend_str(20.0, 20.0)
        assert result.startswith("+")
        assert "0.0" in result

    def test_one_decimal_place(self):
        result = compute_trend_str(22.5, 20.0)
        # diff = 2.5 -> "+2.5"
        assert result == "+2.5"

    def test_negative_one_decimal(self):
        result = compute_trend_str(17.5, 20.0)
        assert result == "-2.5"

    def test_returns_string(self):
        assert isinstance(compute_trend_str(10.0, 10.0), str)

    def test_large_delta(self):
        result = compute_trend_str(40.0, 15.0)
        assert result == "+25.0"


# ---------------------------------------------------------------------------
# last_n_values
# ---------------------------------------------------------------------------

class TestLastNValues:
    def _make_logs(self, pts_dates: list[tuple]) -> pd.DataFrame:
        """Build player logs with (pts, game_date) tuples."""
        rows = [{"pts": p, "game_date": pd.Timestamp(d)} for p, d in pts_dates]
        return pd.DataFrame(rows)

    def test_returns_last_n_sorted_by_date(self):
        df = self._make_logs([
            (20.0, "2026-01-05"),
            (15.0, "2026-01-01"),
            (18.0, "2026-01-03"),
            (22.0, "2026-01-07"),
            (10.0, "2025-12-28"),
            (12.0, "2025-12-30"),
        ])
        result = last_n_values(df, "pts", n=3)
        # Last 3 by date: Jan 3 (18), Jan 5 (20), Jan 7 (22) -> sorted ascending -> [18,20,22]
        assert result == [18.0, 20.0, 22.0]

    def test_returns_at_most_n_values(self):
        df = self._make_logs([(float(i), f"2026-01-{i+1:02d}") for i in range(10)])
        result = last_n_values(df, "pts", n=LAST_N)
        assert len(result) <= LAST_N

    def test_returns_empty_when_all_nan(self):
        df = pd.DataFrame({"pts": [float("nan"), float("nan")],
                           "game_date": [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")]})
        result = last_n_values(df, "pts")
        assert result == []

    def test_values_rounded_to_one_decimal(self):
        df = self._make_logs([(10.555, "2026-01-01")])
        result = last_n_values(df, "pts", n=1)
        assert result == [10.6]

    def test_fewer_games_than_n(self):
        """When only 3 games exist and n=5, return all 3."""
        df = self._make_logs([(20.0, "2026-01-01"), (22.0, "2026-01-02"), (18.0, "2026-01-03")])
        result = last_n_values(df, "pts", n=5)
        assert len(result) == 3

    def test_returns_list(self):
        df = self._make_logs([(10.0, "2026-01-01")])
        assert isinstance(last_n_values(df, "pts"), list)

    def test_skips_nan_in_middle(self):
        """NaN values are dropped before taking the last N."""
        df = self._make_logs([
            (float("nan"), "2026-01-01"),
            (20.0, "2026-01-02"),
            (25.0, "2026-01-03"),
        ])
        result = last_n_values(df, "pts", n=3)
        # NaN dropped -> [20.0, 25.0]
        assert float("nan") not in result
        assert len(result) == 2
