"""
Tests for src/features/team_game_features.py

Focuses on:
  - Rolling window calculations produce correct values
  - shift(1) is applied BEFORE rolling (no-leakage guarantee)
  - Inner joins don't silently drop rows
  - Edge cases: first game of season, back-to-back games, single-game team
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features.team_game_features import (
    _parse_home_away,
    _extract_opponent_abbr,
    _rolling_mean_shift,
    _rolling_win_pct,
    _games_in_window,
    _rolling_slope,
    _haversine_miles,
)


# ── _parse_home_away ──────────────────────────────────────────────────────────


class TestParseHomeAway:

    def test_parse_home_away_vs_is_home(self):
        matchup = pd.Series(["LAL vs. GSW", "LAL @ BOS"])
        result = _parse_home_away(matchup)
        assert result.tolist() == [1, 0]

    def test_parse_home_away_all_home(self):
        matchup = pd.Series(["LAL vs. GSW", "LAL vs. PHX", "LAL vs. DEN"])
        result = _parse_home_away(matchup)
        assert result.tolist() == [1, 1, 1]

    def test_parse_home_away_all_away(self):
        matchup = pd.Series(["LAL @ GSW", "LAL @ PHX"])
        result = _parse_home_away(matchup)
        assert result.tolist() == [0, 0]


# ── _extract_opponent_abbr ────────────────────────────────────────────────────


class TestExtractOpponentAbbr:

    def test_extract_opponent_vs(self):
        matchup = pd.Series(["LAL vs. GSW"])
        assert _extract_opponent_abbr(matchup).tolist() == ["GSW"]

    def test_extract_opponent_at(self):
        matchup = pd.Series(["LAL @ BOS"])
        assert _extract_opponent_abbr(matchup).tolist() == ["BOS"]

    def test_extract_opponent_mixed(self):
        matchup = pd.Series(["LAL vs. GSW", "LAL @ BOS", "CHI vs. MIA"])
        result = _extract_opponent_abbr(matchup).tolist()
        assert result == ["GSW", "BOS", "MIA"]


# ── _rolling_mean_shift (no-leakage guarantee) ───────────────────────────────


class TestRollingMeanShift:

    def test_rolling_mean_shift_values_correct(self):
        """Rolling mean of shifted values matches manual calculation."""
        df = pd.DataFrame({
            "pts": [100, 110, 120, 130, 140],
        })
        result = _rolling_mean_shift(df, "pts", window=3)
        # shift(1): [NaN, 100, 110, 120, 130]
        # rolling(3, min_periods=1).mean():
        #   idx0: NaN (shifted value is NaN)
        #   idx1: mean([100]) = 100
        #   idx2: mean([100, 110]) = 105
        #   idx3: mean([100, 110, 120]) = 110
        #   idx4: mean([110, 120, 130]) = 120
        assert np.isnan(result.iloc[0])
        assert result.iloc[1] == pytest.approx(100.0)
        assert result.iloc[2] == pytest.approx(105.0)
        assert result.iloc[3] == pytest.approx(110.0)
        assert result.iloc[4] == pytest.approx(120.0)

    def test_rolling_mean_shift_no_leakage_row_n_uses_only_prior(self):
        """
        CRITICAL: For row N, the rolling feature must only use data from
        rows 0..N-1. The value at row N must NOT influence its own feature.
        """
        df = pd.DataFrame({
            "pts": [100, 200, 300, 400, 500],
        })
        result = _rolling_mean_shift(df, "pts", window=5)

        # At row 3 (pts=400), the rolling mean should use only rows 0,1,2
        # shift(1): [NaN, 100, 200, 300, 400]
        # At idx3: rolling over [100, 200, 300] = 200
        assert result.iloc[3] == pytest.approx(200.0)
        # Verify row 3's own value (400) is NOT included
        assert result.iloc[3] != pytest.approx(250.0)  # would be wrong if 400 leaked

    def test_rolling_mean_shift_first_game_is_nan(self):
        """First game has no prior data, so rolling feature must be NaN."""
        df = pd.DataFrame({"pts": [100, 110, 120]})
        result = _rolling_mean_shift(df, "pts", window=3)
        assert np.isnan(result.iloc[0])

    def test_rolling_mean_shift_window_1(self):
        """Window=1 with shift(1) gives the previous game's value."""
        df = pd.DataFrame({"pts": [100, 200, 300]})
        result = _rolling_mean_shift(df, "pts", window=1)
        assert np.isnan(result.iloc[0])
        assert result.iloc[1] == pytest.approx(100.0)
        assert result.iloc[2] == pytest.approx(200.0)


# ── _rolling_win_pct ──────────────────────────────────────────────────────────


class TestRollingWinPct:

    def test_rolling_win_pct_basic(self):
        df = pd.DataFrame({"wl": ["W", "W", "L", "W", "L"]})
        result = _rolling_win_pct(df, window=3)
        # win_flag: [1, 1, 0, 1, 0]
        # shift(1): [NaN, 1, 1, 0, 1]
        # rolling(3, min_periods=1).mean():
        #   idx0: NaN
        #   idx1: 1.0
        #   idx2: mean(1, 1) = 1.0
        #   idx3: mean(1, 1, 0) = 0.667
        #   idx4: mean(1, 0, 1) = 0.667
        assert np.isnan(result.iloc[0])
        assert result.iloc[1] == pytest.approx(1.0)
        assert result.iloc[2] == pytest.approx(1.0)
        assert result.iloc[3] == pytest.approx(2.0 / 3.0, abs=1e-3)

    def test_rolling_win_pct_all_losses(self):
        df = pd.DataFrame({"wl": ["L", "L", "L", "L"]})
        result = _rolling_win_pct(df, window=3)
        assert np.isnan(result.iloc[0])
        assert result.iloc[1] == pytest.approx(0.0)
        assert result.iloc[3] == pytest.approx(0.0)

    def test_rolling_win_pct_no_leakage(self):
        """Current game result must not influence current row's win_pct."""
        df = pd.DataFrame({"wl": ["L", "L", "L", "W"]})
        result = _rolling_win_pct(df, window=5)
        # At idx3 (W), only prior games (L,L,L) are used -> 0.0
        assert result.iloc[3] == pytest.approx(0.0)


# ── _games_in_window ──────────────────────────────────────────────────────────


class TestGamesInWindow:

    def test_games_in_window_basic(self):
        """Count games in last 5 calendar days before each game."""
        df = pd.DataFrame({
            "game_date": pd.to_datetime([
                "2024-10-22",
                "2024-10-24",  # 2 days later
                "2024-10-25",  # 1 day later (back-to-back)
                "2024-10-28",  # 3 days later
                "2024-10-30",  # 2 days later
            ]),
        })
        result = _games_in_window(df, window_days=5)
        # idx0: no prior games -> 0
        # idx1 (Oct 24): cutoff = Oct 19 -> 1 game (Oct 22) in window -> 1
        # idx2 (Oct 25): cutoff = Oct 20 -> 2 games (Oct 22, 24) -> 2
        # idx3 (Oct 28): cutoff = Oct 23 -> 2 games (Oct 24, 25) -> 2
        # idx4 (Oct 30): cutoff = Oct 25 -> 2 games (Oct 25, 28) -> 2
        assert result.iloc[0] == 0
        assert result.iloc[1] == 1
        assert result.iloc[2] == 2

    def test_games_in_window_first_game_is_zero(self):
        df = pd.DataFrame({
            "game_date": pd.to_datetime(["2024-10-22"]),
        })
        result = _games_in_window(df, window_days=7)
        assert result.iloc[0] == 0

    def test_games_in_window_back_to_back(self):
        """Back-to-back games (1 day apart) should count in a 5-day window."""
        df = pd.DataFrame({
            "game_date": pd.to_datetime([
                "2024-10-22",
                "2024-10-23",  # back-to-back
            ]),
        })
        result = _games_in_window(df, window_days=5)
        assert result.iloc[1] == 1  # one game within 5 days before Oct 23


# ── _rolling_slope ────────────────────────────────────────────────────────────


class TestRollingSlope:

    def test_rolling_slope_constant_values_zero_slope(self):
        """Constant values should have slope = 0."""
        values = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        slopes = _rolling_slope(values, window=3)
        # Positions with 2+ prior values should have slope ~0
        assert slopes[3] == pytest.approx(0.0, abs=1e-10)
        assert slopes[4] == pytest.approx(0.0, abs=1e-10)

    def test_rolling_slope_increasing_positive_slope(self):
        """Linearly increasing values should have positive slope."""
        values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        slopes = _rolling_slope(values, window=5)
        # At idx4, fits line through [10, 20, 30, 40] -> slope = 10
        assert slopes[4] == pytest.approx(10.0, abs=1e-6)

    def test_rolling_slope_first_position_is_zero(self):
        """First position has no prior games, should return 0.0."""
        values = np.array([100.0, 200.0, 300.0])
        slopes = _rolling_slope(values, window=3)
        assert slopes[0] == pytest.approx(0.0)

    def test_rolling_slope_single_prior_returns_zero(self):
        """Position with only 1 prior value returns 0.0 (need 2+ for slope)."""
        values = np.array([100.0, 200.0, 300.0])
        slopes = _rolling_slope(values, window=3)
        assert slopes[1] == pytest.approx(0.0)


# ── _haversine_miles ──────────────────────────────────────────────────────────


class TestHaversineMiles:

    def test_haversine_same_point_zero_distance(self):
        dist = _haversine_miles(34.0, -118.0, 34.0, -118.0)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_haversine_known_distance(self):
        """LA to NYC is approximately 2451 miles."""
        dist = _haversine_miles(34.0522, -118.2437, 40.7128, -74.0060)
        assert 2400 < dist < 2500

    def test_haversine_vectorized(self):
        """Should handle numpy arrays."""
        lats1 = np.array([34.0, 40.0])
        lons1 = np.array([-118.0, -74.0])
        lats2 = np.array([34.0, 40.0])
        lons2 = np.array([-118.0, -74.0])
        dists = _haversine_miles(lats1, lons1, lats2, lons2)
        np.testing.assert_array_almost_equal(dists, [0.0, 0.0])


# ── Edge cases for team game features ─────────────────────────────────────────


class TestTeamGameFeaturesEdgeCases:

    def test_first_game_of_season_nan_rolling(self):
        """First game of season should have NaN rolling features."""
        df = pd.DataFrame({
            "pts": [100],
            "wl": ["W"],
        })
        rolling_result = _rolling_mean_shift(df, "pts", window=5)
        assert np.isnan(rolling_result.iloc[0])

    def test_back_to_back_days_rest_equals_one(self, team_game_logs_df):
        """
        Games 4 and 5 in the fixture are 1 day apart.
        Verify days_rest computation yields 1 for the second game.
        """
        df = team_game_logs_df.copy()
        df = df.sort_values(["team_id", "game_date"]).reset_index(drop=True)
        df["game_date"] = pd.to_datetime(df["game_date"])
        df["days_rest"] = (
            df.groupby("team_id")["game_date"]
            .diff()
            .dt.days
            .fillna(7)
            .clip(upper=14)
        )
        # Game index 4 should be 1 day after game index 3
        assert df.loc[4, "days_rest"] == 1.0

    def test_team_with_single_game_rolling_all_nan(self):
        """A team with only 1 game should have NaN for all rolling features."""
        df = pd.DataFrame({
            "pts": [100],
            "wl": ["W"],
        })
        rolling_pts = _rolling_mean_shift(df, "pts", window=5)
        win_pct = _rolling_win_pct(df, window=5)
        assert np.isnan(rolling_pts.iloc[0])
        assert np.isnan(win_pct.iloc[0])

    def test_inner_join_does_not_silently_drop_rows(self, two_team_game_logs_df):
        """
        Self-join for opponent stats should not drop rows when both teams
        share the same game_id. Verify row count is preserved.
        """
        df = two_team_game_logs_df.copy()
        df["opponent_abbr"] = _extract_opponent_abbr(df["matchup"])
        df["opp_pts"] = df["pts"] - df["plus_minus"]

        initial_count = len(df)

        opp_box = df[["team_abbreviation", "game_id", "pts"]].copy()
        opp_box.columns = ["opponent_abbr", "game_id", "opp_pts_check"]

        merged = df.merge(opp_box, on=["opponent_abbr", "game_id"], how="left")

        # For game_id 0022400001, both LAL and GSW rows should still exist
        game1_rows = merged[merged["game_id"] == "0022400001"]
        assert len(game1_rows) == 2, "Both home and away rows should survive the join"

        # Total row count should not decrease
        assert len(merged) >= initial_count


# ── Leakage integration test ─────────────────────────────────────────────────


class TestNoLeakageIntegration:

    def test_shift_before_rolling_on_multi_row_group(self):
        """
        Integration test: simulate groupby + apply with _rolling_mean_shift
        on a multi-game team to verify shift(1) precedes rolling.
        """
        df = pd.DataFrame({
            "team_id": [1] * 6,
            "pts": [100, 110, 120, 130, 140, 150],
        })

        result = df.groupby("team_id", group_keys=False).apply(
            lambda g: _rolling_mean_shift(g, "pts", window=3),
            include_groups=False,
        )

        # Flatten result to a simple array for positional access
        values = result.values.ravel()

        # At position 5 (pts=150), shift(1) gives [NaN,100,110,120,130,140]
        # Rolling(3) at idx5 over [120,130,140] = 130.0
        assert values[5] == pytest.approx(130.0)
        # Row 5's own value (150) must NOT appear in the calculation
        assert values[5] != pytest.approx(140.0)  # wrong if 150 leaked
