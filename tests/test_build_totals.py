"""
tests/test_build_totals.py

Unit tests for scripts/build_totals.py
"""
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_totals import (
    LEAGUE_AVG_PACE,
    MIN_GAMES_REQUIRED,
    ROLL_WINDOW,
    VALUE_THRESHOLD,
    apply_pace_adjustment,
    build_game_total,
    derive_opp_pts,
    get_pace_from_matchup,
    team_rolling_stats,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_team_logs(
    team_abbr: str = "OKC",
    n_games: int = 15,
    pts: float = 115.0,
    season: int = 202526,
) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=n_games, freq="2D")
    rows = []
    for i, d in enumerate(dates):
        rows.append({
            "team_id": 1,
            "team_abbreviation": team_abbr,
            "game_id": hash(team_abbr) % 10000 + i,
            "game_date": d,
            "pts": pts,
            "season": season,
        })
    return pd.DataFrame(rows)


def _make_enriched_logs(
    home_abbr: str = "OKC",
    away_abbr: str = "POR",
    n_games: int = 15,
    home_pts: float = 118.0,
    away_pts: float = 108.0,
    season: int = 202526,
) -> pd.DataFrame:
    """
    Simulates the output of derive_opp_pts(): each team has opp_pts filled.
    """
    dates = pd.date_range("2026-01-01", periods=n_games, freq="2D")
    rows = []
    for i, d in enumerate(dates):
        game_id = i + 1
        rows.append({
            "team_id": 1, "team_abbreviation": home_abbr,
            "opp_abbr": away_abbr, "game_id": game_id,
            "game_date": d, "pts": home_pts, "opp_pts": away_pts, "season": season,
        })
        rows.append({
            "team_id": 2, "team_abbreviation": away_abbr,
            "opp_abbr": home_abbr, "game_id": game_id,
            "game_date": d, "pts": away_pts, "opp_pts": home_pts, "season": season,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# derive_opp_pts
# ---------------------------------------------------------------------------

class TestDeriveOppPts:
    def test_opp_pts_column_created(self):
        home = _make_team_logs("BOS", pts=120.0)
        away = _make_team_logs("NYK", pts=110.0)
        # Give them matching game_ids
        home["game_id"] = range(1, len(home) + 1)
        away["game_id"] = range(1, len(away) + 1)
        combined = pd.concat([home, away], ignore_index=True)
        result = derive_opp_pts(combined)
        assert "opp_pts" in result.columns

    def test_opp_pts_values_are_correct(self):
        home = _make_team_logs("BOS", pts=120.0, n_games=3)
        away = _make_team_logs("NYK", pts=110.0, n_games=3)
        home["game_id"] = [101, 102, 103]
        away["game_id"] = [101, 102, 103]
        combined = pd.concat([home, away], ignore_index=True)
        result = derive_opp_pts(combined)
        bos_rows = result[result["team_abbreviation"] == "BOS"]
        assert (bos_rows["opp_pts"] == 110.0).all()

    def test_no_self_join(self):
        team = _make_team_logs("OKC", n_games=5)
        team["game_id"] = range(1, 6)
        result = derive_opp_pts(team)
        # No opponent -> empty (team can't play itself)
        assert len(result) == 0

    def test_shape_doubles_rows(self):
        home = _make_team_logs("OKC", n_games=5)
        away = _make_team_logs("POR", n_games=5)
        home["game_id"] = range(1, 6)
        away["game_id"] = range(1, 6)
        combined = pd.concat([home, away], ignore_index=True)
        result = derive_opp_pts(combined)
        # Each game produces 2 rows (one per team)
        assert len(result) == 10


# ---------------------------------------------------------------------------
# team_rolling_stats
# ---------------------------------------------------------------------------

class TestTeamRollingStats:
    def test_returns_avg_scored_and_allowed(self):
        df = _make_enriched_logs("OKC", "POR", n_games=15)
        stats = team_rolling_stats("OKC", df, season=202526)
        assert "avg_scored" in stats
        assert "avg_allowed" in stats

    def test_returns_none_when_too_few_games(self):
        df = _make_enriched_logs("OKC", "POR", n_games=MIN_GAMES_REQUIRED - 1)
        stats = team_rolling_stats("OKC", df, season=202526)
        assert stats["avg_scored"] is None
        assert stats["avg_allowed"] is None

    def test_returns_none_for_unknown_team(self):
        df = _make_enriched_logs("OKC", "POR", n_games=15)
        stats = team_rolling_stats("LAL", df, season=202526)
        assert stats["avg_scored"] is None

    def test_shift_prevents_leakage(self):
        """
        Last game should not contribute to its own projection.
        Set last game pts to extreme value; projection must not be at that extreme.
        """
        df = _make_enriched_logs("OKC", "POR", n_games=15, home_pts=100.0)
        df_copy = df.copy()
        # Inflate last OKC game to extreme value
        okc_idx = df_copy[df_copy["team_abbreviation"] == "OKC"].index[-1]
        df_copy.loc[okc_idx, "pts"] = 999.0
        stats = team_rolling_stats("OKC", df_copy, season=202526)
        assert stats["avg_scored"] is not None
        assert stats["avg_scored"] < 200.0, "Last-row pts leaked into rolling average"

    def test_rolling_values_are_floats(self):
        df = _make_enriched_logs("OKC", "POR", n_games=15)
        stats = team_rolling_stats("OKC", df, season=202526)
        assert isinstance(stats["avg_scored"], float)
        assert isinstance(stats["avg_allowed"], float)


# ---------------------------------------------------------------------------
# apply_pace_adjustment
# ---------------------------------------------------------------------------

class TestApplyPaceAdjustment:
    def test_no_adjustment_when_pace_none(self):
        raw = 220.0
        result = apply_pace_adjustment(raw, None, None)
        assert result == raw

    def test_faster_pace_increases_total(self):
        raw = 220.0
        result = apply_pace_adjustment(raw, LEAGUE_AVG_PACE + 5, LEAGUE_AVG_PACE + 5)
        assert result > raw

    def test_slower_pace_decreases_total(self):
        raw = 220.0
        result = apply_pace_adjustment(raw, LEAGUE_AVG_PACE - 5, LEAGUE_AVG_PACE - 5)
        assert result < raw

    def test_league_average_pace_unchanged(self):
        raw = 220.0
        result = apply_pace_adjustment(raw, LEAGUE_AVG_PACE, LEAGUE_AVG_PACE)
        assert abs(result - raw) < 0.01

    def test_no_adjustment_with_one_none(self):
        raw = 220.0
        result = apply_pace_adjustment(raw, LEAGUE_AVG_PACE, None)
        assert result == raw

    def test_zero_pace_returns_raw(self):
        raw = 220.0
        result = apply_pace_adjustment(raw, 0.0, 0.0)
        assert result == raw


# ---------------------------------------------------------------------------
# get_pace_from_matchup
# ---------------------------------------------------------------------------

class TestGetPaceFromMatchup:
    def _make_matchup_df(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "home_team": "OKC",
            "away_team": "POR",
            "game_date": pd.Timestamp("2026-02-01"),
            "home_pace_game_roll10": 101.5,
            "away_pace_game_roll10": 98.3,
        }])

    def test_returns_pace_for_known_matchup(self):
        df = self._make_matchup_df()
        home_pace, away_pace = get_pace_from_matchup(df, "OKC", "POR")
        assert home_pace == pytest.approx(101.5)
        assert away_pace == pytest.approx(98.3)

    def test_returns_none_none_when_matchup_df_is_none(self):
        home_pace, away_pace = get_pace_from_matchup(None, "OKC", "POR")
        assert home_pace is None
        assert away_pace is None

    def test_falls_back_to_team_level_when_matchup_not_found(self):
        df = self._make_matchup_df()
        # Ask for a different matchup — OKC was home so we fall back
        home_pace, away_pace = get_pace_from_matchup(df, "OKC", "LAL")
        # Should still find OKC's home pace from the available row
        assert home_pace == pytest.approx(101.5)


# ---------------------------------------------------------------------------
# build_game_total
# ---------------------------------------------------------------------------

class TestBuildGameTotal:
    def _game(self) -> dict:
        return {
            "game_date": "2026-03-06",
            "home_team": "OKC",
            "away_team": "POR",
        }

    def test_returns_dict_with_required_keys(self):
        df = _make_enriched_logs("OKC", "POR", n_games=15)
        result = build_game_total(self._game(), df, None, 202526)
        assert result is not None
        required = {
            "home_team", "away_team", "game_date",
            "model_total", "book_total", "edge",
            "recommendation", "value",
            "home_avg_scored", "away_avg_allowed",
        }
        assert required.issubset(result.keys())

    def test_model_total_is_float(self):
        df = _make_enriched_logs("OKC", "POR", n_games=15, home_pts=118.0, away_pts=108.0)
        result = build_game_total(self._game(), df, None, 202526)
        assert isinstance(result["model_total"], float)

    def test_model_total_roughly_sums_scores(self):
        """home_avg_scored + away_avg_allowed should be close to model_total (no pace adj)."""
        df = _make_enriched_logs("OKC", "POR", n_games=15, home_pts=118.0, away_pts=108.0)
        result = build_game_total(self._game(), df, None, 202526)
        expected = result["home_avg_scored"] + result["away_avg_allowed"]
        assert abs(result["model_total"] - expected) < 1.0

    def test_book_total_is_null(self):
        df = _make_enriched_logs("OKC", "POR", n_games=15)
        result = build_game_total(self._game(), df, None, 202526)
        assert result["book_total"] is None

    def test_edge_is_null_when_book_total_null(self):
        df = _make_enriched_logs("OKC", "POR", n_games=15)
        result = build_game_total(self._game(), df, None, 202526)
        assert result["edge"] is None

    def test_value_false_when_no_book_total(self):
        df = _make_enriched_logs("OKC", "POR", n_games=15)
        result = build_game_total(self._game(), df, None, 202526)
        assert result["value"] is False

    def test_returns_none_for_team_with_no_data(self):
        df = _make_enriched_logs("OKC", "POR", n_games=15)
        game = {"game_date": "2026-03-06", "home_team": "LAL", "away_team": "GSW"}
        result = build_game_total(game, df, None, 202526)
        assert result is None

    def test_returns_none_when_insufficient_games(self):
        df = _make_enriched_logs("OKC", "POR", n_games=MIN_GAMES_REQUIRED - 1)
        result = build_game_total(self._game(), df, None, 202526)
        assert result is None

    def test_pace_adjustment_applied_when_matchup_available(self):
        df = _make_enriched_logs("OKC", "POR", n_games=15, home_pts=118.0, away_pts=108.0)
        matchup_df = pd.DataFrame([{
            "home_team": "OKC",
            "away_team": "POR",
            "game_date": pd.Timestamp("2026-02-28"),
            "home_pace_game_roll10": LEAGUE_AVG_PACE + 5.0,
            "away_pace_game_roll10": LEAGUE_AVG_PACE + 5.0,
        }])
        result_no_pace = build_game_total(self._game(), df, None, 202526)
        result_with_pace = build_game_total(self._game(), df, matchup_df, 202526)
        # Faster pace should push total higher
        assert result_with_pace["model_total"] > result_no_pace["model_total"]

    def test_recommendation_is_none_when_no_book_total(self):
        df = _make_enriched_logs("OKC", "POR", n_games=15)
        result = build_game_total(self._game(), df, None, 202526)
        assert result["recommendation"] is None
