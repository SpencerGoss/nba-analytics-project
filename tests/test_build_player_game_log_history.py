"""Tests for scripts/build_player_game_log_history.py pure utility functions."""
import sys
from pathlib import Path
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_player_game_log_history import (
    season_str_to_int,
    available_seasons,
    _normalize_game_row,
)


# ─── season_str_to_int ────────────────────────────────────────────────────────

def test_season_str_to_int_standard():
    assert season_str_to_int("2003-04") == 200304


def test_season_str_to_int_current():
    assert season_str_to_int("2024-25") == 202425


def test_season_str_to_int_turn_of_century():
    assert season_str_to_int("1999-00") == 199900


def test_season_str_to_int_result_is_int():
    result = season_str_to_int("2020-21")
    assert isinstance(result, int)


# ─── available_seasons ────────────────────────────────────────────────────────

def test_available_seasons_includes_first_and_last():
    seasons = available_seasons("2020-21", "2022-23")
    assert "2020-21" in seasons
    assert "2022-23" in seasons


def test_available_seasons_count():
    seasons = available_seasons("2020-21", "2022-23")
    assert len(seasons) == 3


def test_available_seasons_ordered():
    seasons = available_seasons("2018-19", "2021-22")
    years = [int(s.split("-")[0]) for s in seasons]
    assert years == sorted(years)


def test_available_seasons_format():
    seasons = available_seasons("2019-20", "2020-21")
    for s in seasons:
        assert "-" in s
        start, end_yy = s.split("-")
        assert len(start) == 4
        assert len(end_yy) == 2


def test_available_seasons_single():
    seasons = available_seasons("2023-24", "2023-24")
    assert seasons == ["2023-24"]


# ─── _normalize_game_row ─────────────────────────────────────────────────────

def _make_row(data: dict) -> pd.Series:
    return pd.Series(data)


def test_normalize_row_uppercase_columns():
    """PlayerGameLog returns uppercase column names."""
    row = _make_row({
        "Game_ID": "0022400001",
        "GAME_DATE": "2024-10-22",
        "MATCHUP": "OKC vs. MEM",
        "WL": "W",
        "MIN": 34,
        "PTS": 32,
        "REB": 5,
        "AST": 7,
        "STL": 2,
        "BLK": 1,
        "TOV": 3,
        "FGM": 12,
        "FGA": 22,
        "FG_PCT": 0.545,
        "FG3M": 3,
        "FG3A": 8,
        "FG3_PCT": 0.375,
        "FTM": 5,
        "FTA": 6,
        "FT_PCT": 0.833,
        "PLUS_MINUS": 12,
    })
    result = _normalize_game_row(row)
    assert result["pts"] == 32
    assert result["ast"] == 7
    assert result["game_date"] == "2024-10-22"
    assert result["wl"] == "W"


def test_normalize_row_schema():
    row = _make_row({
        "Game_ID": "g001", "GAME_DATE": "2025-01-01", "MATCHUP": "OKC vs. LAL",
        "WL": "W", "MIN": 30, "PTS": 25, "REB": 5, "AST": 4, "STL": 1,
        "BLK": 0, "TOV": 2, "FGM": 9, "FGA": 18, "FG_PCT": 0.5,
        "FG3M": 2, "FG3A": 5, "FG3_PCT": 0.4, "FTM": 5, "FTA": 6,
        "FT_PCT": 0.833, "PLUS_MINUS": 8,
    })
    result = _normalize_game_row(row)
    for key in ("game_id", "game_date", "matchup", "wl", "pts", "reb", "ast",
                "fgm", "fga", "fg_pct", "plus_minus"):
        assert key in result, f"Missing key: {key}"


def test_normalize_row_null_stat():
    """Missing stats should return None, not crash."""
    row = _make_row({
        "Game_ID": "g001", "GAME_DATE": "2025-01-01", "MATCHUP": "OKC vs. LAL",
        "WL": "W",
        # Intentionally omit numeric stats
    })
    result = _normalize_game_row(row)
    assert result["pts"] is None
    assert result["reb"] is None


def test_normalize_row_nan_stat():
    """NaN stats should return None."""
    row = _make_row({
        "Game_ID": "g001", "GAME_DATE": "2025-01-01", "MATCHUP": "OKC vs. LAL",
        "WL": "W", "PTS": float("nan"),
    })
    result = _normalize_game_row(row)
    assert result["pts"] is None


def test_normalize_row_fg_pct_rounded():
    row = _make_row({
        "Game_ID": "g001", "GAME_DATE": "2025-01-01", "MATCHUP": "OKC @ MIA",
        "WL": "L", "MIN": 28, "PTS": 18, "REB": 4, "AST": 3, "STL": 0,
        "BLK": 0, "TOV": 2, "FGM": 7, "FGA": 15, "FG_PCT": 0.46666,
        "FG3M": 1, "FG3A": 4, "FG3_PCT": 0.25, "FTM": 3, "FTA": 4,
        "FT_PCT": 0.75, "PLUS_MINUS": -5,
    })
    result = _normalize_game_row(row)
    # FG_PCT is stored with 4 decimal places
    assert result["fg_pct"] == pytest.approx(0.4667)


def test_normalize_row_three_point_stats():
    """fg3m, fg3a, fg3_pct must all be present and correctly extracted."""
    row = _make_row({
        "Game_ID": "g001", "GAME_DATE": "2025-01-01", "MATCHUP": "LAL vs. GSW",
        "WL": "W", "MIN": 32, "PTS": 20, "REB": 4, "AST": 5, "STL": 1,
        "BLK": 1, "TOV": 2, "FGM": 8, "FGA": 16, "FG_PCT": 0.5,
        "FG3M": 3, "FG3A": 7, "FG3_PCT": 0.4286, "FTM": 1, "FTA": 2,
        "FT_PCT": 0.5, "PLUS_MINUS": 4,
    })
    result = _normalize_game_row(row)
    assert result["fg3m"] == 3
    assert result["fg3a"] == 7
    assert result["fg3_pct"] == pytest.approx(0.4286, abs=0.0001)


def test_normalize_row_free_throw_stats():
    """ftm, fta, ft_pct must all be present."""
    row = _make_row({
        "Game_ID": "g001", "GAME_DATE": "2025-01-01", "MATCHUP": "BOS vs. PHI",
        "WL": "W", "MIN": 35, "PTS": 28, "REB": 8, "AST": 3, "STL": 2,
        "BLK": 0, "TOV": 3, "FGM": 10, "FGA": 20, "FG_PCT": 0.5,
        "FG3M": 2, "FG3A": 6, "FG3_PCT": 0.333, "FTM": 6, "FTA": 8,
        "FT_PCT": 0.75, "PLUS_MINUS": 10,
    })
    result = _normalize_game_row(row)
    assert result["ftm"] == 6
    assert result["fta"] == 8
    assert result["ft_pct"] == pytest.approx(0.75, abs=0.0001)


def test_normalize_row_pts_rounded_to_integer():
    """PTS (integer stat) must be rounded to 0 decimals."""
    row = _make_row({
        "Game_ID": "g001", "GAME_DATE": "2025-01-01", "MATCHUP": "OKC vs. LAL",
        "WL": "W", "PTS": 25.9,  # should round to 26
    })
    result = _normalize_game_row(row)
    assert result["pts"] == 26


def test_normalize_row_game_id_fallback():
    """When Game_ID is absent, game_id from lowercase fallback must be used."""
    row = _make_row({
        "game_id": "fallback_id", "GAME_DATE": "2025-01-01", "MATCHUP": "OKC vs. LAL",
        "WL": "W",
    })
    result = _normalize_game_row(row)
    assert result["game_id"] == "fallback_id"


def test_available_seasons_includes_lockout_season():
    """2011-12 lockout season must be included as available."""
    seasons = available_seasons("2010-11", "2012-13")
    assert "2011-12" in seasons


def test_season_str_to_int_post_century():
    """2000-01 cross-century season must encode correctly."""
    assert season_str_to_int("2000-01") == 200001


def test_season_str_to_int_ancient_season():
    """1996-97 season (Jordan era) must encode correctly."""
    assert season_str_to_int("1996-97") == 199697


def test_available_seasons_two_consecutive():
    seasons = available_seasons("2021-22", "2022-23")
    assert len(seasons) == 2
    assert "2021-22" in seasons
    assert "2022-23" in seasons


def test_normalize_row_stl_blk_tov():
    """stl, blk, tov must all appear in normalized output."""
    row = _make_row({
        "Game_ID": "g001", "GAME_DATE": "2025-01-01", "MATCHUP": "OKC vs. LAL",
        "WL": "W", "MIN": 30, "PTS": 20, "REB": 5, "AST": 4,
        "STL": 2, "BLK": 1, "TOV": 3,
        "FGM": 8, "FGA": 16, "FG_PCT": 0.5,
        "FG3M": 2, "FG3A": 5, "FG3_PCT": 0.4, "FTM": 2, "FTA": 3,
        "FT_PCT": 0.667, "PLUS_MINUS": 5,
    })
    result = _normalize_game_row(row)
    assert result["stl"] == 2
    assert result["blk"] == 1
    assert result["tov"] == 3
