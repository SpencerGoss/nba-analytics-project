"""
Tests for src/features/injury_proxy.py

Focuses on:
  - Absent rotation player detection (player misses games)
  - missing_minutes and star_player_out flags
  - merge_asof with MAX_STALE_DAYS=25 staleness cutoff
  - Rolling baseline minutes use shift(1) to avoid leakage
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features.injury_proxy import (
    MIN_ROTATION_MINUTES,
    MIN_ROTATION_GAMES,
    STAR_USG_THRESHOLD,
    ROLL_WINDOW,
    build_injury_proxy_features,
    _aggregate_from_absences_df,
)


# ── Helper to write synthetic data to temp files ──────────────────────────────


def _write_test_data(tmp_path, player_game_logs_df, adv_stats_df):
    """Write synthetic DataFrames to CSV files for build_injury_proxy_features."""
    game_log_path = str(tmp_path / "player_game_logs.csv")
    adv_path = str(tmp_path / "adv_stats.csv")
    output_path = str(tmp_path / "output" / "injury_proxy.csv")

    player_game_logs_df.to_csv(game_log_path, index=False)
    adv_stats_df.to_csv(adv_path, index=False)

    return game_log_path, adv_path, output_path


# ── Absent rotation player detection ─────────────────────────────────────────


class TestAbsentRotationDetection:

    def test_player_missing_games_flagged_as_absent(
        self, tmp_path, player_game_logs_df, adv_stats_df
    ):
        """
        Player A plays games 1-5, misses 6-8, plays 9-10.
        After establishing a rotation baseline (games 1-5 at 30 min avg),
        Player A should be flagged as absent for games 6-8.
        """
        game_log_path, adv_path, output_path = _write_test_data(
            tmp_path, player_game_logs_df, adv_stats_df
        )

        result = build_injury_proxy_features(
            game_log_path=game_log_path,
            adv_stats_path=adv_path,
            output_path=output_path,
        )

        # Player A misses game_ids for indices 5, 6, 7
        team_id = 1610612747

        rows_with_missing = result[
            (result["team_id"] == team_id)
            & (result["missing_minutes"] > 0)
        ]

        # Player A (30 min avg) should be flagged as absent for at least some
        # of the games they missed. The missing_minutes should be ~30.
        assert len(rows_with_missing) > 0, (
            "Expected some games to have missing_minutes > 0 when Player A is absent"
        )
        # Verify the missing minutes are from Player A (~30 min)
        assert rows_with_missing["missing_minutes"].max() == pytest.approx(30.0, abs=1.0)

    def test_bench_player_not_flagged_as_rotation(
        self, tmp_path, player_game_logs_df, adv_stats_df
    ):
        """
        Player B plays 10 min per game (below MIN_ROTATION_MINUTES=15).
        Even if Player B somehow missed a game, they should NOT be flagged
        as a missing rotation player.
        """
        # Modify: remove Player B from game 6 to test
        logs = player_game_logs_df.copy()
        game_6_id = "0022400006"
        logs = logs[~((logs["player_id"] == 102) & (logs["game_id"] == game_6_id))]

        game_log_path, adv_path, output_path = _write_test_data(
            tmp_path, logs, adv_stats_df
        )

        result = build_injury_proxy_features(
            game_log_path=game_log_path,
            adv_stats_path=adv_path,
            output_path=output_path,
        )

        # Player B at 10 min avg should not be in rotation, so missing_minutes
        # for game 6 should NOT include Player B's 10 minutes
        game_6_row = result[
            (result["game_id"].astype(str) == game_6_id)
            & (result["team_id"] == 1610612747)
        ]
        if not game_6_row.empty:
            # If there are missing minutes, they should be from Player A (30 min), not B (10 min)
            mm = game_6_row["missing_minutes"].iloc[0]
            # Player B's 10 min should NOT be included
            assert mm < 15, (
                f"Bench player (10 min avg) should not add to missing_minutes, got {mm}"
            )


# ── Star player out flag ──────────────────────────────────────────────────────


class TestStarPlayerOut:

    def test_star_player_out_when_high_usage_player_absent(
        self, tmp_path, adv_stats_df
    ):
        """
        When Player C (usg_pct=0.28 >= 0.25 threshold) misses a game,
        star_player_out should be 1.
        """
        dates = pd.date_range("2024-10-22", periods=10, freq="3D")
        game_ids = [f"002240{i:04d}" for i in range(1, 11)]

        rows = []
        for i in range(10):
            # Player A: plays all games
            rows.append({
                "season": 202425, "player_id": 101, "player_name": "Player A",
                "team_id": 1610612747, "team_abbreviation": "LAL",
                "game_id": game_ids[i], "game_date": dates[i], "min": 30,
            })
            # Player C (star): plays games 0-6, misses 7-8, plays 9
            if i not in [7, 8]:
                rows.append({
                    "season": 202425, "player_id": 103, "player_name": "Player C",
                    "team_id": 1610612747, "team_abbreviation": "LAL",
                    "game_id": game_ids[i], "game_date": dates[i], "min": 35,
                })

        logs = pd.DataFrame(rows)
        game_log_path, adv_path, output_path = _write_test_data(
            tmp_path, logs, adv_stats_df
        )

        result = build_injury_proxy_features(
            game_log_path=game_log_path,
            adv_stats_path=adv_path,
            output_path=output_path,
        )

        # When Player C (usg_pct=0.28) is absent, star_player_out should be 1
        star_games = result[
            (result["star_player_out"] == 1)
            & (result["team_id"] == 1610612747)
        ]

        assert len(star_games) > 0, (
            "Expected at least one game with star_player_out=1 when Player C (usg=0.28) is absent"
        )
        # Missing minutes should reflect Player C's ~35 min baseline
        assert star_games["missing_minutes"].max() >= 30.0


# ── merge_asof staleness (MAX_STALE_DAYS) ────────────────────────────────────


class TestMergeAsofStaleness:

    def test_player_absent_within_max_stale_days_is_flagged(
        self, tmp_path, adv_stats_df
    ):
        """
        Player absent for 10 days (within MAX_STALE_DAYS=25) should be flagged.
        """
        # Player plays games 1-5 (days 0-12), then game on day 22 (10 days later)
        dates = list(pd.date_range("2024-10-22", periods=5, freq="3D"))
        dates.append(pd.Timestamp("2024-11-11"))  # 10 days after last game

        game_ids = [f"002240{i:04d}" for i in range(1, 7)]

        rows = []
        for i in range(6):
            # Player A plays games 0-4, misses game 5
            if i < 5:
                rows.append({
                    "season": 202425, "player_id": 101, "player_name": "Player A",
                    "team_id": 1610612747, "team_abbreviation": "LAL",
                    "game_id": game_ids[i], "game_date": dates[i], "min": 30,
                })
            # Player C plays all games (so team has games on those dates)
            rows.append({
                "season": 202425, "player_id": 103, "player_name": "Player C",
                "team_id": 1610612747, "team_abbreviation": "LAL",
                "game_id": game_ids[i], "game_date": dates[i], "min": 35,
            })

        logs = pd.DataFrame(rows)
        game_log_path, adv_path, output_path = _write_test_data(
            tmp_path, logs, adv_stats_df
        )

        result = build_injury_proxy_features(
            game_log_path=game_log_path,
            adv_stats_path=adv_path,
            output_path=output_path,
        )

        # Game 6 (0022400006): Player A last played ~10 days ago -> within 25-day window
        game_6 = result[
            (result["game_id"].astype(str) == "0022400006")
            & (result["team_id"] == 1610612747)
        ]
        if not game_6.empty:
            assert game_6["missing_minutes"].iloc[0] > 0, (
                "Player absent for 10 days (within MAX_STALE_DAYS=25) should be flagged"
            )

    def test_player_absent_beyond_max_stale_days_not_flagged(
        self, tmp_path, adv_stats_df
    ):
        """
        Player absent for 30 days (beyond MAX_STALE_DAYS=25) should NOT be flagged.
        The player may have been traded, waived, or suffered a season-ending injury.
        """
        # Player plays games on Oct 22-31, then team has game Dec 1 (>30 days later)
        dates = list(pd.date_range("2024-10-22", periods=5, freq="2D"))
        dates.append(pd.Timestamp("2024-12-01"))  # 32 days after last game (Oct 30)

        game_ids = [f"002240{i:04d}" for i in range(1, 7)]

        rows = []
        for i in range(6):
            # Player A plays games 0-4, misses game 5 (32+ days later)
            if i < 5:
                rows.append({
                    "season": 202425, "player_id": 101, "player_name": "Player A",
                    "team_id": 1610612747, "team_abbreviation": "LAL",
                    "game_id": game_ids[i], "game_date": dates[i], "min": 30,
                })
            # Player C plays all games
            rows.append({
                "season": 202425, "player_id": 103, "player_name": "Player C",
                "team_id": 1610612747, "team_abbreviation": "LAL",
                "game_id": game_ids[i], "game_date": dates[i], "min": 35,
            })

        logs = pd.DataFrame(rows)
        game_log_path, adv_path, output_path = _write_test_data(
            tmp_path, logs, adv_stats_df
        )

        result = build_injury_proxy_features(
            game_log_path=game_log_path,
            adv_stats_path=adv_path,
            output_path=output_path,
        )

        # Game 6: Player A last played >30 days ago -> beyond MAX_STALE_DAYS=25
        game_6 = result[
            (result["game_id"].astype(str) == "0022400006")
            & (result["team_id"] == 1610612747)
        ]
        if not game_6.empty:
            # Player A's missing minutes should NOT be counted (stale)
            # Only Player C might contribute, but C played, so missing=0
            mm = game_6["missing_minutes"].iloc[0]
            # Should not include Player A's 30 min contribution
            assert mm < 25, (
                f"Player absent >30 days (beyond MAX_STALE_DAYS=25) should NOT be flagged. "
                f"Got missing_minutes={mm}"
            )


# ── Rotation availability ────────────────────────────────────────────────────


class TestRotationAvailability:

    def test_rotation_availability_full_strength(
        self, tmp_path, player_game_logs_df, adv_stats_df
    ):
        """
        When no rotation players are absent, rotation_availability should be 1.0.
        """
        game_log_path, adv_path, output_path = _write_test_data(
            tmp_path, player_game_logs_df, adv_stats_df
        )

        result = build_injury_proxy_features(
            game_log_path=game_log_path,
            adv_stats_path=adv_path,
            output_path=output_path,
        )

        # Early games (before Player A misses) should have availability = 1.0
        early_games = result[
            (result["team_id"] == 1610612747)
            & (result["missing_minutes"] == 0)
        ]
        if not early_games.empty:
            assert all(early_games["rotation_availability"] == 1.0)

    def test_rotation_availability_clipped_0_to_1(
        self, tmp_path, player_game_logs_df, adv_stats_df
    ):
        """rotation_availability should always be between 0.0 and 1.0."""
        game_log_path, adv_path, output_path = _write_test_data(
            tmp_path, player_game_logs_df, adv_stats_df
        )

        result = build_injury_proxy_features(
            game_log_path=game_log_path,
            adv_stats_path=adv_path,
            output_path=output_path,
        )

        assert result["rotation_availability"].min() >= 0.0
        assert result["rotation_availability"].max() <= 1.0


# ── Output schema ─────────────────────────────────────────────────────────────


class TestInjuryProxyOutputSchema:

    def test_output_has_required_columns(
        self, tmp_path, player_game_logs_df, adv_stats_df
    ):
        """Output must contain all expected columns for downstream joins."""
        game_log_path, adv_path, output_path = _write_test_data(
            tmp_path, player_game_logs_df, adv_stats_df
        )

        result = build_injury_proxy_features(
            game_log_path=game_log_path,
            adv_stats_path=adv_path,
            output_path=output_path,
        )

        required_cols = [
            "team_id", "game_id",
            "missing_minutes", "missing_usg_pct",
            "rotation_availability", "star_player_out",
            "n_missing_rotation",
        ]
        for col in required_cols:
            assert col in result.columns, f"Missing required column: {col}"

    def test_game_id_is_string_team_id_is_int(
        self, tmp_path, player_game_logs_df, adv_stats_df
    ):
        """
        Join keys must be normalized: game_id as str, team_id as int.
        This prevents silent left-join key mismatches downstream.
        """
        game_log_path, adv_path, output_path = _write_test_data(
            tmp_path, player_game_logs_df, adv_stats_df
        )

        result = build_injury_proxy_features(
            game_log_path=game_log_path,
            adv_stats_path=adv_path,
            output_path=output_path,
        )

        assert pd.api.types.is_string_dtype(result["game_id"]), "game_id should be string dtype"
        assert np.issubdtype(result["team_id"].dtype, np.integer), "team_id should be int"


# ── _aggregate_from_absences_df ───────────────────────────────────────────────


def _make_absences_df(rows: list[dict]) -> pd.DataFrame:
    """Create a minimal player_absences-style DataFrame."""
    return pd.DataFrame(rows)


class TestAggregateFromAbsencesDf:

    def test_output_schema(self):
        """Output must contain all required aggregate columns."""
        df = _make_absences_df([
            {"player_id": 1, "team_id": 100, "game_id": "G1",
             "min_roll5": 30.0, "usg_pct": 0.20, "was_absent": 0},
        ])
        result = _aggregate_from_absences_df(df)
        for col in ["game_id", "team_id", "total_expected_minutes"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_absences_missing_minutes_is_zero_or_nan(self):
        """When no player is absent, missing_minutes should be 0 or absent."""
        df = _make_absences_df([
            {"player_id": 1, "team_id": 100, "game_id": "G1",
             "min_roll5": 30.0, "usg_pct": 0.20, "was_absent": 0},
            {"player_id": 2, "team_id": 100, "game_id": "G1",
             "min_roll5": 25.0, "usg_pct": 0.18, "was_absent": 0},
        ])
        result = _aggregate_from_absences_df(df)
        row = result[result["game_id"] == "G1"]
        assert len(row) == 1
        mm = row["missing_minutes"].iloc[0] if "missing_minutes" in row.columns else 0
        assert pd.isna(mm) or mm == 0.0

    def test_missing_minutes_sums_absent_players(self):
        """missing_minutes should be sum of min_roll5 for absent players only."""
        df = _make_absences_df([
            {"player_id": 1, "team_id": 100, "game_id": "G1",
             "min_roll5": 30.0, "usg_pct": 0.20, "was_absent": 1},
            {"player_id": 2, "team_id": 100, "game_id": "G1",
             "min_roll5": 28.0, "usg_pct": 0.18, "was_absent": 0},
        ])
        result = _aggregate_from_absences_df(df)
        row = result[result["game_id"] == "G1"]
        assert row["missing_minutes"].iloc[0] == pytest.approx(30.0)

    def test_n_missing_rotation_counts_absent_players(self):
        """n_missing_rotation should be the count of absent player-rows."""
        df = _make_absences_df([
            {"player_id": 1, "team_id": 100, "game_id": "G1",
             "min_roll5": 30.0, "usg_pct": 0.20, "was_absent": 1},
            {"player_id": 2, "team_id": 100, "game_id": "G1",
             "min_roll5": 25.0, "usg_pct": 0.15, "was_absent": 1},
            {"player_id": 3, "team_id": 100, "game_id": "G1",
             "min_roll5": 20.0, "usg_pct": 0.12, "was_absent": 0},
        ])
        result = _aggregate_from_absences_df(df)
        row = result[result["game_id"] == "G1"]
        assert row["n_missing_rotation"].iloc[0] == 2

    def test_star_player_out_flagged_when_high_usg_absent(self):
        """star_player_out should be 1 when absent player has usg_pct >= STAR_USG_THRESHOLD."""
        df = _make_absences_df([
            {"player_id": 1, "team_id": 100, "game_id": "G1",
             "min_roll5": 35.0, "usg_pct": STAR_USG_THRESHOLD, "was_absent": 1},
        ])
        result = _aggregate_from_absences_df(df)
        row = result[result["game_id"] == "G1"]
        assert row["star_player_out"].iloc[0] == 1

    def test_star_player_out_zero_when_low_usg_absent(self):
        """star_player_out should be 0 when absent player has usg_pct < STAR_USG_THRESHOLD."""
        df = _make_absences_df([
            {"player_id": 1, "team_id": 100, "game_id": "G1",
             "min_roll5": 20.0, "usg_pct": STAR_USG_THRESHOLD - 0.01, "was_absent": 1},
        ])
        result = _aggregate_from_absences_df(df)
        row = result[result["game_id"] == "G1"]
        assert row["star_player_out"].iloc[0] == 0

    def test_total_expected_minutes_includes_all_rotation(self):
        """total_expected_minutes sums min_roll5 for ALL players (absent + present)."""
        df = _make_absences_df([
            {"player_id": 1, "team_id": 100, "game_id": "G1",
             "min_roll5": 30.0, "usg_pct": 0.20, "was_absent": 1},
            {"player_id": 2, "team_id": 100, "game_id": "G1",
             "min_roll5": 25.0, "usg_pct": 0.18, "was_absent": 0},
        ])
        result = _aggregate_from_absences_df(df)
        row = result[result["game_id"] == "G1"]
        assert row["total_expected_minutes"].iloc[0] == pytest.approx(55.0)

    def test_multiple_teams_same_game(self):
        """Two teams in the same game should produce separate rows."""
        df = _make_absences_df([
            {"player_id": 1, "team_id": 100, "game_id": "G1",
             "min_roll5": 30.0, "usg_pct": 0.20, "was_absent": 0},
            {"player_id": 2, "team_id": 200, "game_id": "G1",
             "min_roll5": 28.0, "usg_pct": 0.18, "was_absent": 0},
        ])
        result = _aggregate_from_absences_df(df)
        assert len(result) == 2
        assert set(result["team_id"].tolist()) == {100, 200}

    def test_missing_usg_pct_column_present(self):
        """missing_usg_pct must appear in output schema."""
        df = _make_absences_df([
            {"player_id": 1, "team_id": 100, "game_id": "G1",
             "min_roll5": 30.0, "usg_pct": 0.25, "was_absent": 1},
        ])
        result = _aggregate_from_absences_df(df)
        assert "missing_usg_pct" in result.columns

    def test_empty_input_returns_empty_dataframe(self):
        """Empty absences DataFrame must produce empty output without error."""
        df = _make_absences_df([])
        df["player_id"] = pd.Series(dtype=int)
        df["team_id"] = pd.Series(dtype=int)
        df["game_id"] = pd.Series(dtype=str)
        df["min_roll5"] = pd.Series(dtype=float)
        df["usg_pct"] = pd.Series(dtype=float)
        df["was_absent"] = pd.Series(dtype=int)
        result = _aggregate_from_absences_df(df)
        assert isinstance(result, pd.DataFrame)
        assert result.empty


class TestInjuryProxyConstants:
    def test_min_rotation_minutes_positive(self):
        assert MIN_ROTATION_MINUTES > 0

    def test_min_rotation_games_positive(self):
        assert MIN_ROTATION_GAMES > 0

    def test_star_usg_threshold_between_0_and_1(self):
        assert 0.0 < STAR_USG_THRESHOLD < 1.0

    def test_roll_window_positive_integer(self):
        assert isinstance(ROLL_WINDOW, int) and ROLL_WINDOW > 0
