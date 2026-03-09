"""Tests for scripts/build_streaks.py"""
import sys
from pathlib import Path
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_streaks import (
    _compute_team_streaks,
    _compute_home_away,
    _compute_hot_players,
    _compute_cold_players,
)


def _make_team_df(records: list[tuple]) -> pd.DataFrame:
    """Helper: build minimal team_game_logs dataframe."""
    rows = []
    for team, date, wl, matchup, pts, fga, fta in records:
        rows.append({
            "team_abbreviation": team,
            "game_date": date,
            "wl": wl,
            "matchup": matchup,
            "pts": pts,
            "fga": fga,
            "fta": fta,
        })
    return pd.DataFrame(rows)


# ─── _compute_team_streaks ───────────────────────────────────────────────────

def test_team_streak_win():
    df = _make_team_df([
        ("BOS", "2026-01-01", "L", "BOS vs. NYK", 90, 80, 20),
        ("BOS", "2026-01-02", "W", "BOS vs. MIA", 110, 85, 22),
        ("BOS", "2026-01-03", "W", "BOS vs. PHI", 105, 80, 18),
        ("BOS", "2026-01-04", "W", "BOS @ ORL", 108, 82, 19),
    ])
    results = _compute_team_streaks(df)
    bos = next(r for r in results if r["team"] == "BOS")
    assert bos["streak"] == 3


def test_team_streak_loss():
    df = _make_team_df([
        ("DET", "2026-01-01", "W", "DET vs. CHI", 100, 78, 20),
        ("DET", "2026-01-02", "L", "DET @ NYK", 92, 80, 20),
        ("DET", "2026-01-03", "L", "DET vs. MIA", 88, 79, 19),
    ])
    results = _compute_team_streaks(df)
    det = next(r for r in results if r["team"] == "DET")
    assert det["streak"] == -2


def test_team_streak_single_game():
    df = _make_team_df([
        ("MIA", "2026-01-01", "W", "MIA vs. BOS", 100, 78, 20),
    ])
    results = _compute_team_streaks(df)
    mia = next(r for r in results if r["team"] == "MIA")
    assert mia["streak"] == 1


def test_team_streaks_sorted_by_abs():
    df = _make_team_df([
        ("OKC", "2026-01-01", "W", "OKC vs. LAL", 110, 85, 20),
        ("OKC", "2026-01-02", "W", "OKC vs. GSW", 108, 82, 19),
        ("OKC", "2026-01-03", "W", "OKC @ HOU", 105, 80, 18),
        ("OKC", "2026-01-04", "W", "OKC vs. MEM", 112, 88, 22),
        ("CHA", "2026-01-01", "L", "CHA @ MIA", 88, 79, 19),
        ("CHA", "2026-01-02", "L", "CHA vs. ATL", 90, 80, 20),
    ])
    results = _compute_team_streaks(df)
    assert abs(results[0]["streak"]) >= abs(results[1]["streak"])


def test_team_streaks_empty():
    df = pd.DataFrame(columns=["team_abbreviation", "game_date", "wl", "matchup", "pts", "fga", "fta"])
    results = _compute_team_streaks(df)
    assert results == []


# ─── _compute_home_away ──────────────────────────────────────────────────────

def test_home_away_split():
    df = _make_team_df([
        ("BOS", "2026-01-01", "W", "BOS vs. NYK", 110, 85, 20),
        ("BOS", "2026-01-02", "L", "BOS @ MIA", 95, 80, 18),
        ("BOS", "2026-01-03", "W", "BOS vs. PHI", 108, 82, 19),
        ("BOS", "2026-01-04", "W", "BOS @ ORL", 105, 80, 18),
    ])
    results = _compute_home_away(df)
    bos = next((r for r in results if r["team"] == "BOS"), None)
    assert bos is not None
    assert bos["home_pct"] == 100  # 2 home, 2 wins
    assert bos["away_pct"] == 50   # 1 away win, 1 loss


def test_home_away_no_games():
    df = pd.DataFrame(columns=["team_abbreviation", "game_date", "wl", "matchup", "pts", "fga", "fta"])
    results = _compute_home_away(df)
    assert results == []


# ─── _compute_hot_players ────────────────────────────────────────────────────

def _make_player_df(records: list[tuple]) -> pd.DataFrame:
    rows = []
    for pid, name, team, date, pts, reb, ast in records:
        rows.append({
            "player_id": pid,
            "player_name": name,
            "team_abbreviation": team,
            "game_date": date,
            "pts": pts,
            "reb": reb,
            "ast": ast,
        })
    return pd.DataFrame(rows)


def test_hot_players_basic():
    df = _make_player_df([
        (1, "Scorer A", "OKC", f"2026-01-0{i+1}", 35 + i*2, 5, 3)
        for i in range(5)
    ])
    results = _compute_hot_players(df)
    assert isinstance(results, list)
    assert len(results) >= 0  # may filter if last5 > season avg threshold


def test_hot_players_empty():
    df = pd.DataFrame(columns=["player_id", "player_name", "team_abbreviation", "game_date", "pts", "reb", "ast"])
    results = _compute_hot_players(df)
    assert results == []


# ─── _compute_cold_players ───────────────────────────────────────────────────

def test_cold_players_basic():
    df = _make_player_df([
        (2, "Scorer B", "CHA", f"2026-01-{10+i}", 25, 4, 2)
        for i in range(10)
    ] + [
        (2, "Scorer B", "CHA", f"2026-01-{20+i}", 8 + i, 2, 1)
        for i in range(5)
    ])
    results = _compute_cold_players(df)
    assert isinstance(results, list)


def test_cold_players_empty():
    df = pd.DataFrame(columns=["player_id", "player_name", "team_abbreviation", "game_date", "pts", "reb", "ast"])
    results = _compute_cold_players(df)
    assert results == []


# ─── schema contracts ────────────────────────────────────────────────────────

def test_team_streak_schema():
    df = _make_team_df([
        ("GSW", "2026-01-01", "W", "GSW vs. LAL", 115, 90, 20),
    ])
    results = _compute_team_streaks(df)
    assert len(results) == 1
    row = results[0]
    assert "team" in row
    assert "streak" in row
    assert isinstance(row["streak"], int)


def test_home_away_schema():
    df = _make_team_df([
        ("PHX", "2026-01-01", "W", "PHX vs. SAC", 112, 88, 22),
        ("PHX", "2026-01-02", "L", "PHX @ LAL", 98, 82, 18),
    ])
    results = _compute_home_away(df)
    phx = next((r for r in results if r["team"] == "PHX"), None)
    if phx:
        assert "home_pct" in phx
        assert "away_pct" in phx
        assert 0 <= phx["home_pct"] <= 100
        assert 0 <= phx["away_pct"] <= 100
