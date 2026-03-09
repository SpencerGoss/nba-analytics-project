"""Tests for scripts/build_player_detail.py"""
import sys
from pathlib import Path
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_player_detail import (
    _safe_float,
    _safe_int,
    _per_game,
    _trend,
    _parse_opponent,
    _load_season_stats,
    _load_advanced,
    _load_clutch,
    _compute_prop_trends,
    CURRENT_SEASON,
)


# ─── _safe_float ──────────────────────────────────────────────────────────────

def test_safe_float_normal():
    assert _safe_float(0.535) == pytest.approx(0.535)


def test_safe_float_rounds():
    assert _safe_float(0.5357, decimals=2) == pytest.approx(0.54)


def test_safe_float_none():
    assert _safe_float(None) is None


def test_safe_float_nan():
    assert _safe_float(float("nan")) is None


def test_safe_float_string_number():
    assert _safe_float("32.1") == pytest.approx(32.1)


def test_safe_float_invalid_string():
    assert _safe_float("N/A") is None


# ─── _safe_int ────────────────────────────────────────────────────────────────

def test_safe_int_normal():
    assert _safe_int(55) == 55


def test_safe_int_float_input():
    assert _safe_int(55.9) == 55


def test_safe_int_none():
    assert _safe_int(None) is None


def test_safe_int_nan():
    assert _safe_int(float("nan")) is None


# ─── _per_game ────────────────────────────────────────────────────────────────

def test_per_game_basic():
    # 300 points / 10 games = 30.0
    result = _per_game(300, 10)
    assert result == pytest.approx(30.0)


def test_per_game_zero_games():
    assert _per_game(300, 0) is None


def test_per_game_none_total():
    assert _per_game(None, 10) is None


# ─── _trend ───────────────────────────────────────────────────────────────────

def test_trend_up():
    # Last5 >10% above season avg
    assert _trend(35.0, 30.0) == "up"


def test_trend_down():
    # Last5 >10% below season avg
    assert _trend(25.0, 30.0) == "down"


def test_trend_flat():
    # Within 10%
    assert _trend(30.5, 30.0) == "flat"


def test_trend_none_inputs():
    assert _trend(None, 30.0) == "flat"
    assert _trend(30.0, None) == "flat"
    assert _trend(None, None) == "flat"


def test_trend_zero_season_avg():
    # Avoid division by zero
    result = _trend(1.0, 0.0)
    assert result in ("up", "flat")


# ─── _parse_opponent ──────────────────────────────────────────────────────────

def test_parse_opponent_vs():
    # "BOS vs. NYK" — home game, opponent is NYK
    assert _parse_opponent("BOS vs. NYK", "BOS") == "NYK"


def test_parse_opponent_at():
    # "BOS @ MIA" — away game, opponent is MIA
    assert _parse_opponent("BOS @ MIA", "BOS") == "MIA"


def test_parse_opponent_unknown():
    result = _parse_opponent("malformed string", "BOS")
    assert isinstance(result, str)


# ─── _load_season_stats ───────────────────────────────────────────────────────

def _make_player_stats(records: list[tuple]) -> pd.DataFrame:
    """(player_id, player_name, team_abbr, season, gp, pts, reb, ast, stl, blk, tov, fg_pct, fg3_pct, ft_pct, min)"""
    rows = []
    for pid, name, team, season, gp, pts, reb, ast, stl, blk, tov, fg_pct, fg3_pct, ft_pct, minutes in records:
        rows.append({
            "player_id": pid, "player_name": name, "team_abbreviation": team,
            "season": season, "gp": gp, "pts": pts, "reb": reb, "ast": ast,
            "stl": stl, "blk": blk, "tov": tov,
            "fg_pct": fg_pct, "fg3_pct": fg3_pct, "ft_pct": ft_pct,
            "min": minutes,
        })
    return pd.DataFrame(rows)


def test_load_season_stats_per_game():
    df = _make_player_stats([
        (1, "Test Player", "OKC", CURRENT_SEASON, 10, 300, 50, 60, 20, 10, 25, 0.5, 0.38, 0.87, 340),
    ])
    result = _load_season_stats(df)
    assert 1 in result
    stats = result[1]
    assert stats["ppg"] == pytest.approx(30.0)  # 300 / 10
    assert stats["rpg"] == pytest.approx(5.0)   # 50 / 10
    assert stats["apg"] == pytest.approx(6.0)   # 60 / 10


def test_load_season_stats_schema():
    df = _make_player_stats([
        (2, "Player B", "LAL", CURRENT_SEASON, 20, 400, 80, 100, 30, 20, 40, 0.5, 0.36, 0.82, 680),
    ])
    result = _load_season_stats(df)
    row = result[2]
    for key in ("ppg", "rpg", "apg", "spg", "bpg", "topg", "fg_pct", "gp"):
        assert key in row, f"Missing key: {key}"


def test_load_season_stats_fg_pct():
    df = _make_player_stats([
        (3, "Player C", "GSW", CURRENT_SEASON, 10, 200, 40, 50, 15, 10, 20, 0.5, 0.38, 0.87, 340),
    ])
    result = _load_season_stats(df)
    assert result[3]["fg_pct"] == pytest.approx(0.5)


# ─── _load_advanced ───────────────────────────────────────────────────────────

def _make_advanced(records: list[tuple]) -> pd.DataFrame:
    """(player_id, season, ts_pct, usg_pct, pie, efg_pct, off_rating, def_rating, net_rating)"""
    rows = []
    for pid, season, ts, usg, pie, efg, off_r, def_r, net_r in records:
        rows.append({
            "player_id": pid, "season": season,
            "ts_pct": ts, "usg_pct": usg, "pie": pie, "efg_pct": efg,
            "off_rating": off_r, "def_rating": def_r, "net_rating": net_r,
            "ast_pct": 0.3, "oreb_pct": 0.05, "dreb_pct": 0.15, "gp": 50,
        })
    return pd.DataFrame(rows)


def test_load_advanced_schema():
    df = _make_advanced([(1, CURRENT_SEASON, 0.64, 0.32, 0.18, 0.56, 125.3, 107.2, 18.1)])
    result = _load_advanced(df)
    row = result[1]
    for key in ("ts_pct", "usg_pct", "pie", "off_rating", "def_rating", "net_rating"):
        assert key in row, f"Missing key: {key}"


def test_load_advanced_empty():
    df = pd.DataFrame(columns=["player_id", "season", "gp", "ts_pct", "usg_pct", "pie",
                                "efg_pct", "off_rating", "def_rating", "net_rating",
                                "ast_pct", "oreb_pct", "dreb_pct"])
    result = _load_advanced(df)
    assert isinstance(result, dict)
    assert len(result) == 0


# ─── _load_clutch ─────────────────────────────────────────────────────────────

def _make_clutch(records: list[tuple]) -> pd.DataFrame:
    """(player_id, season, gp, pts, reb, ast, fg_pct, ft_pct)"""
    rows = []
    for pid, season, gp, pts, reb, ast, fg_pct, ft_pct in records:
        rows.append({
            "player_id": pid, "season": season,
            "gp": gp, "pts": pts, "reb": reb, "ast": ast,
            "fg_pct": fg_pct, "ft_pct": ft_pct, "plus_minus": 0,
        })
    return pd.DataFrame(rows)


def test_load_clutch_schema():
    df = _make_clutch([(1, CURRENT_SEASON, 30, 180, 30, 50, 0.51, 0.88)])
    result = _load_clutch(df)
    row = result[1]
    for key in ("ppg", "rpg", "apg", "fg_pct", "gp"):
        assert key in row, f"Missing key: {key}"


def test_load_clutch_per_game():
    df = _make_clutch([(2, CURRENT_SEASON, 10, 60, 10, 20, 0.5, 0.85)])
    result = _load_clutch(df)
    assert result[2]["ppg"] == pytest.approx(6.0)  # 60 / 10


# ─── _compute_prop_trends ─────────────────────────────────────────────────────

def test_compute_prop_trends_up():
    # Last 5 avg 25pt, season avg 20pt => up
    last5 = [{"pts": 25, "reb": 5, "ast": 4} for _ in range(5)]
    season_stats = {"ppg": 20.0, "rpg": 5.0, "apg": 4.0}
    result = _compute_prop_trends(last5, season_stats)
    assert result["pts_trend"] == "up"


def test_compute_prop_trends_down():
    last5 = [{"pts": 10, "reb": 3, "ast": 2} for _ in range(5)]
    season_stats = {"ppg": 25.0, "rpg": 6.0, "apg": 5.0}
    result = _compute_prop_trends(last5, season_stats)
    assert result["pts_trend"] == "down"


def test_compute_prop_trends_flat():
    last5 = [{"pts": 20, "reb": 5, "ast": 4} for _ in range(5)]
    season_stats = {"ppg": 20.0, "rpg": 5.0, "apg": 4.0}
    result = _compute_prop_trends(last5, season_stats)
    assert result["pts_trend"] == "flat"


def test_compute_prop_trends_empty_last5():
    result = _compute_prop_trends([], {"ppg": 20.0, "rpg": 5.0, "apg": 4.0})
    assert result == {}


def test_compute_prop_trends_schema():
    last5 = [{"pts": 20, "reb": 5, "ast": 4} for _ in range(5)]
    season_stats = {"ppg": 20.0, "rpg": 5.0, "apg": 4.0}
    result = _compute_prop_trends(last5, season_stats)
    for key in ("pts_last5_avg", "pts_season_avg", "pts_trend",
                "reb_last5_avg", "reb_season_avg", "reb_trend",
                "ast_last5_avg", "ast_season_avg", "ast_trend"):
        assert key in result, f"Missing key: {key}"
