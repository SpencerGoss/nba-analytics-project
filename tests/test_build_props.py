"""
tests/test_build_props.py

Unit tests for scripts/build_props.py
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# Ensure project root on sys.path so scripts package can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_props import (
    ROLL_WINDOW,
    STAT_COLS,
    VALUE_THRESHOLD,
    MIN_GAMES_REQUIRED,
    build_game_props,
    compute_player_rolling,
    last_n_values,
    load_absent_player_ids,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_player_df(
    player_id: int = 1,
    player_name: str = "Test Player",
    team: str = "OKC",
    n_games: int = 15,
    pts_base: float = 25.0,
) -> pd.DataFrame:
    """Create a synthetic player game log with predictable values."""
    dates = pd.date_range("2026-01-01", periods=n_games, freq="2D")
    rows = []
    for i, d in enumerate(dates):
        rows.append({
            "player_id": player_id,
            "player_name": player_name,
            "team_abbreviation": team,
            "game_id": 10000 + i,
            "game_date": d,
            "min_num": 32.0,
            "pts": pts_base + i,
            "reb": 5.0,
            "ast": 4.0,
            "fg3m": 2.0,
            "stl": 1.0,
            "blk": 0.5,
            "season": 202526,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# compute_player_rolling
# ---------------------------------------------------------------------------

class TestComputePlayerRolling:
    def test_returns_empty_dict_when_too_few_games(self):
        df = _make_player_df(n_games=MIN_GAMES_REQUIRED - 1)
        result = compute_player_rolling(df)
        assert result == {}

    def test_returns_projection_for_each_stat(self):
        df = _make_player_df(n_games=15)
        result = compute_player_rolling(df)
        assert set(result.keys()) == set(STAT_COLS)

    def test_projections_are_floats(self):
        df = _make_player_df(n_games=15)
        result = compute_player_rolling(df)
        for col in STAT_COLS:
            assert isinstance(result[col], float), f"{col} should be float"

    def test_shift_prevents_leakage(self):
        """
        shift(1) means the final row is excluded from its own projection.
        With n_games=12 and constant pts=25, rolling mean of prior 10 = 25.
        Last actual value should NOT inflate the average.
        """
        n = 12
        df = _make_player_df(n_games=n, pts_base=25.0)
        # Make the last row have a very large pts value
        df_copy = df.copy()
        df_copy.loc[df_copy.index[-1], "pts"] = 999.0
        result = compute_player_rolling(df_copy)
        # If shift(1) is working, the last 999 value is not included
        assert result["pts"] < 100.0, "Last-row value leaked into projection"

    def test_rolling_window_size(self):
        """Projection uses at most ROLL_WINDOW prior games."""
        n_games = ROLL_WINDOW + 5
        # First ROLL_WINDOW games have pts=10, last 5 have pts=100
        rows = []
        for i in range(n_games):
            rows.append({
                "player_id": 1,
                "player_name": "P",
                "team_abbreviation": "OKC",
                "game_date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                "min_num": 30.0,
                "pts": 10.0 if i < ROLL_WINDOW else 100.0,
                "reb": 5.0, "ast": 4.0, "fg3m": 2.0, "stl": 1.0, "blk": 0.5,
                "season": 202526,
            })
        df = pd.DataFrame(rows)
        result = compute_player_rolling(df)
        # shift(1): last row (pts=100) is excluded; window covers rows [6..15] (after shift)
        # The most recent 10 shifted values are rows index 5..14 (pts: 10,10,10,10,10,100,100,100,100,100)
        assert result["pts"] is not None
        # Should be somewhere between 10 and 100 — not all-10 or all-100
        assert 10.0 < result["pts"] < 100.0


# ---------------------------------------------------------------------------
# last_n_values
# ---------------------------------------------------------------------------

class TestLastNValues:
    def test_returns_last_5_games(self):
        df = _make_player_df(n_games=10, pts_base=20.0)
        vals = last_n_values(df, "pts", n=5)
        assert len(vals) == 5

    def test_returns_fewer_when_not_enough_data(self):
        df = _make_player_df(n_games=3, pts_base=20.0)
        vals = last_n_values(df, "pts", n=5)
        assert len(vals) == 3

    def test_values_are_rounded_floats(self):
        df = _make_player_df(n_games=8)
        vals = last_n_values(df, "pts")
        assert all(isinstance(v, float) for v in vals)

    def test_drops_nan(self):
        # 8 rows, 2 NaN -> 6 valid; tail(5) -> 5 values; none should be NaN
        df = _make_player_df(n_games=8)
        df.loc[df.index[1], "pts"] = float("nan")
        df.loc[df.index[4], "pts"] = float("nan")
        vals = last_n_values(df, "pts", n=5)
        assert len(vals) == 5
        assert all(v == v for v in vals), "NaN values should not appear in output"


# ---------------------------------------------------------------------------
# load_absent_player_ids
# ---------------------------------------------------------------------------

class TestLoadAbsentPlayerIds:
    def test_returns_set_of_absent_ids(self, tmp_path):
        csv = tmp_path / "player_absences.csv"
        csv.write_text(
            "player_id,player_name,team_id,game_id,game_date,season,min_roll5,usg_pct,was_absent\n"
            "1,Alice,100,9001,2026-03-06,202526,30.0,0.25,1\n"
            "2,Bob,101,9002,2026-03-06,202526,28.0,0.22,0\n"
            "3,Carol,102,9003,2026-03-06,202526,25.0,0.20,1\n"
        )
        with patch("scripts.build_props.ABSENCES_PATH", csv):
            result = load_absent_player_ids(pd.Timestamp("2026-03-06"))
        assert result == {1, 3}

    def test_returns_empty_set_when_file_missing(self, tmp_path):
        missing = tmp_path / "does_not_exist.csv"
        with patch("scripts.build_props.ABSENCES_PATH", missing):
            result = load_absent_player_ids(pd.Timestamp("2026-03-06"))
        assert result == set()

    def test_filters_by_date(self, tmp_path):
        csv = tmp_path / "player_absences.csv"
        csv.write_text(
            "player_id,player_name,team_id,game_id,game_date,season,min_roll5,usg_pct,was_absent\n"
            "1,Alice,100,9001,2026-03-05,202526,30.0,0.25,1\n"
            "2,Bob,101,9002,2026-03-06,202526,28.0,0.22,1\n"
        )
        with patch("scripts.build_props.ABSENCES_PATH", csv):
            result = load_absent_player_ids(pd.Timestamp("2026-03-06"))
        assert result == {2}
        assert 1 not in result


# ---------------------------------------------------------------------------
# build_game_props
# ---------------------------------------------------------------------------

class TestBuildGameProps:
    def _make_logs(self) -> pd.DataFrame:
        p1 = _make_player_df(player_id=1, player_name="Star One", team="OKC", n_games=15)
        p2 = _make_player_df(player_id=2, player_name="Star Two", team="OKC", n_games=15, pts_base=20.0)
        p3 = _make_player_df(player_id=3, player_name="Opponent Star", team="POR", n_games=15, pts_base=18.0)
        return pd.concat([p1, p2, p3], ignore_index=True)

    def _game(self) -> dict:
        return {
            "game_date": "2026-03-06",
            "home_team": "OKC",
            "away_team": "POR",
        }

    def test_returns_list_of_player_dicts(self):
        logs = self._make_logs()
        result = build_game_props(self._game(), logs, absent_player_ids=set())
        assert isinstance(result, list)
        assert len(result) > 0

    def test_player_dict_has_required_keys(self):
        logs = self._make_logs()
        result = build_game_props(self._game(), logs, absent_player_ids=set())
        required = {"player_name", "player_id", "team", "opponent", "game_date", "is_injured", "props"}
        for player in result:
            assert required.issubset(player.keys()), f"Missing keys in {player}"

    def test_props_have_required_fields(self):
        logs = self._make_logs()
        result = build_game_props(self._game(), logs, absent_player_ids=set())
        required = {"stat", "model_projection", "book_line", "edge", "recommendation", "value", "last5"}
        for player in result:
            for prop in player["props"]:
                assert required.issubset(prop.keys()), f"Missing prop keys: {prop}"

    def test_injured_player_flagged(self):
        logs = self._make_logs()
        result = build_game_props(self._game(), logs, absent_player_ids={1})
        injured_players = [p for p in result if p["player_id"] == 1]
        assert len(injured_players) == 1
        assert injured_players[0]["is_injured"] is True

    def test_healthy_player_not_flagged(self):
        logs = self._make_logs()
        result = build_game_props(self._game(), logs, absent_player_ids=set())
        for player in result:
            assert player["is_injured"] is False

    def test_book_line_is_null(self):
        """Pinnacle props not yet available — all book_lines should be null."""
        logs = self._make_logs()
        result = build_game_props(self._game(), logs, absent_player_ids=set())
        for player in result:
            for prop in player["props"]:
                assert prop["book_line"] is None

    def test_value_false_when_book_line_null(self):
        logs = self._make_logs()
        result = build_game_props(self._game(), logs, absent_player_ids=set())
        for player in result:
            for prop in player["props"]:
                assert prop["value"] is False

    def test_covers_both_teams(self):
        logs = self._make_logs()
        result = build_game_props(self._game(), logs, absent_player_ids=set())
        teams = {p["team"] for p in result}
        assert "OKC" in teams
        assert "POR" in teams

    def test_skips_players_with_insufficient_history(self):
        logs = self._make_logs()
        # Add a player with only 1 game
        tiny = _make_player_df(player_id=99, player_name="Rookie", team="OKC", n_games=1)
        all_logs = pd.concat([logs, tiny], ignore_index=True)
        result = build_game_props(self._game(), all_logs, absent_player_ids=set())
        names = [p["player_name"] for p in result]
        assert "Rookie" not in names

    def test_projections_are_numbers(self):
        logs = self._make_logs()
        result = build_game_props(self._game(), logs, absent_player_ids=set())
        for player in result:
            for prop in player["props"]:
                assert isinstance(prop["model_projection"], (int, float))

    def test_last5_contains_floats(self):
        logs = self._make_logs()
        result = build_game_props(self._game(), logs, absent_player_ids=set())
        for player in result:
            for prop in player["props"]:
                for v in prop["last5"]:
                    assert isinstance(v, float)
