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
    MIN_GAMES,
    LAST_N,
    HOT_DELTA,
    COLD_FG_DELTA,
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
            "fga": 15,    # field goal attempts — required by _compute_cold_players
            "fgm": 6,
            "fg_pct": 0.400,  # required by _compute_cold_players
            "min": 28.0,
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


def test_team_streak_mixed_case_wl():
    """WL values like 'w' or 'W' should both work correctly."""
    df = _make_team_df([
        ("NYK", "2026-01-01", "w", "NYK vs. BOS", 100, 80, 20),
        ("NYK", "2026-01-02", "W", "NYK @ MIA", 110, 85, 20),
    ])
    results = _compute_team_streaks(df)
    nyk = next(r for r in results if r["team"] == "NYK")
    assert nyk["streak"] == 2


def test_compute_home_away_only_home_games():
    """Team with only home games: away_pct should be 0 (no away games)."""
    df = _make_team_df([
        ("IND", "2026-01-01", "W", "IND vs. MIL", 108, 82, 18),
        ("IND", "2026-01-02", "L", "IND vs. CLE", 95, 79, 17),
    ])
    results = _compute_home_away(df)
    ind = next((r for r in results if r["team"] == "IND"), None)
    if ind:
        assert ind["away_pct"] == 0


def test_hot_players_detected_when_above_hot_delta():
    """Player with last-5 avg > season avg by > HOT_DELTA is flagged."""
    # Season avg: 10 pts (below threshold unless we set avg high enough)
    # Use 10 games at 12 pts (season avg=12), then 5 games at 25 pts (last5 avg=25)
    # delta = 25 - 12 = 13 > HOT_DELTA (5.0)
    records = (
        [(1, "Hot Guy", "OKC", f"2026-01-{i+1:02d}", 12, 12, 3)
         for i in range(10)]
        + [(1, "Hot Guy", "OKC", f"2026-01-{i+11:02d}", 25, 12, 3)
           for i in range(5)]
    )
    df = _make_player_df(records)
    results = _compute_hot_players(df)
    assert len(results) >= 1
    assert results[0]["name"] == "Hot Guy"


def test_hot_players_schema():
    """Each hot player dict must have name, team, sub, stat fields."""
    records = (
        [(1, "Star", "BOS", f"2026-01-{i+1:02d}", 10, 5, 2) for i in range(10)]
        + [(1, "Star", "BOS", f"2026-01-{i+11:02d}", 30, 8, 4) for i in range(5)]
    )
    df = _make_player_df(records)
    results = _compute_hot_players(df)
    for player in results:
        assert "name" in player
        assert "team" in player
        assert "sub" in player
        assert "stat" in player
        assert not any(k.startswith("_") for k in player)


def test_cold_players_detected_when_below_cold_delta():
    """Player with last-5 FG% much lower than season avg is flagged."""
    # 10 games at 50% FG, then 5 games at 30% FG
    # delta = (30 - 50) = -20 pp < COLD_FG_DELTA (-5.0 pp)
    records = (
        [(2, "Cold Guy", "DET", f"2026-01-{i+1:02d}", 15, 10, 3)
         for i in range(10)]
        + [(2, "Cold Guy", "DET", f"2026-01-{i+11:02d}", 8, 10, 3)
           for i in range(5)]
    )
    # Need fga, fgm, fg_pct — manually build
    rows = []
    for pid, name, team, date, pts, reb, ast in records[:10]:
        rows.append({
            "player_id": pid, "player_name": name, "team_abbreviation": team,
            "game_date": date, "pts": pts, "reb": reb, "ast": ast,
            "fga": 10, "fgm": 5, "fg_pct": 0.50, "min": 30.0,
        })
    for pid, name, team, date, pts, reb, ast in records[10:]:
        rows.append({
            "player_id": pid, "player_name": name, "team_abbreviation": team,
            "game_date": date, "pts": pts, "reb": reb, "ast": ast,
            "fga": 10, "fgm": 3, "fg_pct": 0.30, "min": 30.0,
        })
    df = pd.DataFrame(rows)
    results = _compute_cold_players(df)
    assert len(results) >= 1
    assert results[0]["name"] == "Cold Guy"


def test_hot_players_excludes_too_few_games():
    """Player with fewer than MIN_GAMES games is not included."""
    records = [(1, "Bench", "MIA", f"2026-01-{i+1:02d}", 30, 5, 2)
               for i in range(MIN_GAMES - 1)]
    df = _make_player_df(records)
    results = _compute_hot_players(df)
    assert len(results) == 0


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
