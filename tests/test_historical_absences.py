"""
Tests for src/data/get_historical_absences.py

Verifies:
  1. Shape contract: output DataFrame has required columns and >0 rows
  2. Leakage: was_absent=1 players have no game log entry for that game_id
  3. Season format: season column is integer dtype
  4. No future data: min_roll5 uses shift(1) before rolling — value matches expected
  5. Output file: CSV is written to tmp_path with >0 rows
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.get_historical_absences import build_player_absences


# ── Fixtures ─────────────────────────────────────────────────────────────────

REQUIRED_COLS = [
    "player_id", "player_name", "team_id", "game_id",
    "game_date", "season", "min_roll5", "usg_pct", "was_absent",
]


def _make_game_log(tmp_path, rows_df):
    """Write synthetic game log and advanced stats CSVs; return paths."""
    log_path = str(tmp_path / "player_game_logs.csv")
    adv_path = str(tmp_path / "player_stats_advanced.csv")
    out_path = str(tmp_path / "injuries" / "player_absences.csv")

    rows_df.to_csv(log_path, index=False)

    # Build a matching advanced stats stub (one row per player per season)
    player_seasons = rows_df[["player_id", "season"]].drop_duplicates()
    usg_map = {101: 0.22, 102: 0.10, 103: 0.28}
    player_seasons = player_seasons.copy()
    player_seasons["usg_pct"] = player_seasons["player_id"].map(usg_map).fillna(0.18)
    player_seasons.to_csv(adv_path, index=False)

    return log_path, adv_path, out_path


def _make_synthetic_log():
    """
    3 players, 10 games each (all on same team/season).

    Player A (id=101): plays games 1-5, misses 6-8, plays 9-10. 30 min each.
    Player B (id=102): plays all 10 games. 10 min each (bench — below rotation).
    Player C (id=103): plays all 10 games. 35 min each (star).
    """
    dates = pd.date_range("2024-10-22", periods=10, freq="3D")
    game_ids = [f"002240{i:04d}" for i in range(1, 11)]
    rows = []

    for i in range(10):
        # Player A: plays games 0-4 and 8-9 (misses 5, 6, 7)
        if i not in [5, 6, 7]:
            rows.append({
                "season": 202425, "player_id": 101, "player_name": "Player A",
                "team_id": 1610612747, "team_abbreviation": "LAL",
                "game_id": game_ids[i], "game_date": dates[i].strftime("%Y-%m-%d"),
                "min": 30,
            })
        # Player B: all games, low minutes
        rows.append({
            "season": 202425, "player_id": 102, "player_name": "Player B",
            "team_id": 1610612747, "team_abbreviation": "LAL",
            "game_id": game_ids[i], "game_date": dates[i].strftime("%Y-%m-%d"),
            "min": 10,
        })
        # Player C: all games, high minutes
        rows.append({
            "season": 202425, "player_id": 103, "player_name": "Player C",
            "team_id": 1610612747, "team_abbreviation": "LAL",
            "game_id": game_ids[i], "game_date": dates[i].strftime("%Y-%m-%d"),
            "min": 35,
        })

    return pd.DataFrame(rows)


# ── Test 1: Shape contract ────────────────────────────────────────────────────


class TestShape:

    def test_required_columns_present_and_rows_nonzero(self, tmp_path):
        """
        build_player_absences() must return a DataFrame with all required columns
        and at least one row given a non-trivial synthetic game log.
        """
        df = _make_synthetic_log()
        log_path, adv_path, out_path = _make_game_log(tmp_path, df)

        result = build_player_absences(
            game_log_path=log_path,
            adv_stats_path=adv_path,
            output_path=out_path,
        )

        assert isinstance(result, pd.DataFrame), "Must return a DataFrame"
        assert len(result) > 0, "Output must have at least one row"

        for col in REQUIRED_COLS:
            assert col in result.columns, f"Missing required column: {col}"


# ── Test 2: Leakage — was_absent=1 players not in game log ───────────────────


class TestLeakage:

    def test_absent_players_not_in_game_log(self, tmp_path):
        """
        Every row where was_absent=1 must correspond to a player-game pair that
        does NOT appear in the input game log. If a player is marked absent but
        actually played, that's a false positive indicating a logic bug.
        """
        df = _make_synthetic_log()
        log_path, adv_path, out_path = _make_game_log(tmp_path, df)

        result = build_player_absences(
            game_log_path=log_path,
            adv_stats_path=adv_path,
            output_path=out_path,
        )

        absent_rows = result[result["was_absent"] == 1]
        if absent_rows.empty:
            pytest.skip("No absent rows found — cannot verify leakage contract")

        # Build set of (player_id, game_id) that actually appeared in the log
        actual_played = set(
            zip(
                df["player_id"].astype(str),
                df["game_id"].astype(str),
            )
        )

        for _, row in absent_rows.iterrows():
            key = (str(int(row["player_id"])), str(row["game_id"]).strip())
            assert key not in actual_played, (
                f"Player {row['player_id']} marked absent for game {row['game_id']} "
                f"but they appear in the game log. Leakage detected."
            )


# ── Test 3: Season format ─────────────────────────────────────────────────────


class TestSeasonFormat:

    def test_season_column_is_integer(self, tmp_path):
        """
        All values in the 'season' column must be integers (e.g. 202425),
        not strings like '2024-25'. Project convention: season as int.
        """
        df = _make_synthetic_log()
        log_path, adv_path, out_path = _make_game_log(tmp_path, df)

        result = build_player_absences(
            game_log_path=log_path,
            adv_stats_path=adv_path,
            output_path=out_path,
        )

        assert np.issubdtype(result["season"].dtype, np.integer), (
            f"season column must be integer dtype, got {result['season'].dtype}"
        )

        # Also verify no values look like string seasons
        unique_seasons = result["season"].unique()
        for s in unique_seasons:
            assert int(s) == s, f"Season value {s!r} is not a plain integer"


# ── Test 4: No future data (shift-1 verified) ─────────────────────────────────


class TestNoFutureData:

    def test_min_roll5_matches_prior_5_game_average(self, tmp_path):
        """
        For a player who plays exactly 20 min for games 1-5 and then 30 min for
        game 6, min_roll5 on game 6 should be 20.0 (average of games 1-5, NOT
        including game 6's 30 min). This verifies shift(1) is applied before
        rolling, so the current game's minutes never contaminate the rolling avg.
        """
        dates = pd.date_range("2024-10-22", periods=7, freq="3D")
        game_ids = [f"002240{i:04d}" for i in range(1, 8)]

        rows = []
        # Player A: 20 min for games 1-5, 30 min for game 6, 25 min for game 7
        for i in range(7):
            minutes = 20 if i < 5 else (30 if i == 5 else 25)
            rows.append({
                "season": 202425, "player_id": 101, "player_name": "Player A",
                "team_id": 1610612747, "team_abbreviation": "LAL",
                "game_id": game_ids[i], "game_date": dates[i].strftime("%Y-%m-%d"),
                "min": minutes,
            })
            # Player B: always present to keep team_games populated
            rows.append({
                "season": 202425, "player_id": 102, "player_name": "Player B",
                "team_id": 1610612747, "team_abbreviation": "LAL",
                "game_id": game_ids[i], "game_date": dates[i].strftime("%Y-%m-%d"),
                "min": 25,
            })

        df = pd.DataFrame(rows)
        log_path, adv_path, out_path = _make_game_log(tmp_path, df)

        result = build_player_absences(
            game_log_path=log_path,
            adv_stats_path=adv_path,
            output_path=out_path,
        )

        # Get Player A's row for game 6 (index 5, game_id "0022400006")
        # Note: game_id may have leading zeros stripped when read from CSV
        # as integer. Match against both forms.
        target_game_id = game_ids[5]
        target_game_id_int = str(int(target_game_id))  # "22400006"
        game6_player_a = result[
            (result["player_id"].astype(str) == "101")
            & (
                result["game_id"].astype(str).str.strip().isin(
                    [target_game_id, target_game_id_int]
                )
            )
        ]

        assert not game6_player_a.empty, (
            "No row found for Player A in game 6 — check output logic"
        )

        roll5 = game6_player_a["min_roll5"].iloc[0]
        assert abs(roll5 - 20.0) < 0.01, (
            f"Expected min_roll5=20.0 for game 6 (avg of games 1-5 at 20 min each), "
            f"got {roll5}. shift(1) may not be applied correctly."
        )


# ── Test 5: Output file written ───────────────────────────────────────────────


class TestOutputFile:

    def test_output_csv_exists_with_data(self, tmp_path):
        """
        After calling build_player_absences(output_path=...), the output CSV
        must exist on disk and be a valid CSV with at least one data row.
        """
        df = _make_synthetic_log()
        log_path, adv_path, out_path = _make_game_log(tmp_path, df)

        build_player_absences(
            game_log_path=log_path,
            adv_stats_path=adv_path,
            output_path=out_path,
        )

        assert os.path.exists(out_path), (
            f"Expected output CSV at {out_path} but file does not exist"
        )

        loaded = pd.read_csv(out_path)
        assert len(loaded) > 0, "Output CSV must contain at least one data row"

        for col in REQUIRED_COLS:
            assert col in loaded.columns, (
                f"Output CSV missing required column: {col}"
            )
