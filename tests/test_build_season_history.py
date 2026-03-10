"""Tests for scripts/build_season_history.py"""
import sys
from pathlib import Path
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_season_history import (
    season_label,
    filter_seasons,
    build_standings,
    build_games,
    build_output,
    SEASON_CODES,
    SEASON_LABELS,
)


# ─── season_label ─────────────────────────────────────────────────────────────

def test_season_label_known():
    assert season_label(202425) == "2024-25"
    assert season_label(202021) == "2020-21"


def test_season_label_unknown():
    # Unknown code falls back to string representation
    result = season_label(201920)
    assert isinstance(result, str)
    assert "201920" in result


# ─── filter_seasons ───────────────────────────────────────────────────────────

def _make_logs(records: list[tuple]) -> pd.DataFrame:
    """Helper: (team, season, game_date, wl, matchup, pts, game_id)."""
    rows = []
    for team, season, date, wl, matchup, pts, game_id in records:
        rows.append({
            "team_abbreviation": team,
            "team_name": f"{team} Team",
            "season": season,
            "game_date": pd.to_datetime(date),
            "wl": wl,
            "matchup": matchup,
            "pts": pts,
            "game_id": game_id,
        })
    return pd.DataFrame(rows)


def test_filter_seasons_keeps_valid():
    df = _make_logs([
        ("BOS", 202425, "2025-01-01", "W", "BOS vs. NYK", 110, "g001"),
        ("BOS", 201920, "2020-01-01", "L", "BOS vs. MIA", 100, "g002"),  # out of range
    ])
    result = filter_seasons(df)
    assert len(result) == 1
    assert result["season"].iloc[0] == 202425


def test_filter_seasons_empty_input():
    df = pd.DataFrame(columns=["season"])
    result = filter_seasons(df)
    assert result.empty


def test_filter_seasons_all_seasons():
    rows = [
        ("BOS", code, "2025-01-01", "W", "BOS vs. NYK", 110, f"g{i:03d}")
        for i, code in enumerate(SEASON_CODES)
    ]
    df = _make_logs(rows)
    result = filter_seasons(df)
    assert len(result) == len(SEASON_CODES)


# ─── build_standings ──────────────────────────────────────────────────────────

def test_standings_win_pct():
    df = _make_logs([
        ("BOS", 202425, "2025-01-01", "W", "BOS vs. NYK", 110, "g001"),
        ("BOS", 202425, "2025-01-02", "W", "BOS vs. MIA", 108, "g002"),
        ("BOS", 202425, "2025-01-03", "L", "BOS vs. PHI", 95, "g003"),
    ])
    standings = build_standings(df)
    bos = next(s for s in standings if s["abbr"] == "BOS")
    assert bos["w"] == 2
    assert bos["l"] == 1
    assert bos["pct"] == pytest.approx(0.667, abs=0.001)


def test_standings_sorted_by_pct():
    df = _make_logs([
        ("OKC", 202425, "2025-01-01", "W", "OKC vs. LAL", 110, "g001"),
        ("OKC", 202425, "2025-01-02", "W", "OKC vs. GSW", 108, "g002"),
        ("CHA", 202425, "2025-01-01", "L", "CHA vs. MIA", 88, "g003"),
    ])
    standings = build_standings(df)
    pcts = [s["pct"] for s in standings]
    assert pcts == sorted(pcts, reverse=True)


def test_standings_schema():
    df = _make_logs([("GSW", 202425, "2025-01-01", "W", "GSW vs. LAL", 115, "g001")])
    standings = build_standings(df)
    row = standings[0]
    for key in ("abbr", "name", "w", "l", "pct"):
        assert key in row, f"Missing key: {key}"


def test_standings_no_negative_pct():
    df = _make_logs([("DET", 202425, "2025-01-01", "L", "DET vs. CHI", 90, "g001")])
    standings = build_standings(df)
    assert standings[0]["pct"] >= 0.0


# ─── build_games ──────────────────────────────────────────────────────────────

def test_games_home_only():
    """build_games should only include home rows (containing 'vs.')."""
    df = _make_logs([
        ("BOS", 202425, "2025-01-01", "W", "BOS vs. NYK", 110, "g001"),
        ("NYK", 202425, "2025-01-01", "L", "NYK @ BOS", 100, "g001"),
    ])
    games = build_games(df)
    assert len(games) == 1
    assert games[0]["home_abbr"] == "BOS"
    assert games[0]["away_abbr"] == "NYK"


def test_games_winner_field():
    df = _make_logs([
        ("BOS", 202425, "2025-01-01", "W", "BOS vs. NYK", 110, "g001"),
        ("NYK", 202425, "2025-01-01", "L", "NYK @ BOS", 100, "g001"),
    ])
    games = build_games(df)
    assert "Boston Celtics" in games[0]["winner"]


def test_games_sorted_by_date_desc():
    df = _make_logs([
        ("BOS", 202425, "2025-01-01", "W", "BOS vs. NYK", 110, "g001"),
        ("BOS", 202425, "2025-01-03", "W", "BOS vs. MIA", 108, "g002"),
        ("NYK", 202425, "2025-01-01", "L", "NYK @ BOS", 100, "g001"),
        ("MIA", 202425, "2025-01-03", "L", "MIA @ BOS", 100, "g002"),
    ])
    games = build_games(df)
    dates = [g["date"] for g in games]
    assert dates == sorted(dates, reverse=True)


def test_games_schema():
    df = _make_logs([
        ("BOS", 202425, "2025-01-01", "W", "BOS vs. NYK", 110, "g001"),
    ])
    games = build_games(df)
    if games:
        for key in ("date", "home", "away", "home_abbr", "away_abbr", "home_pts", "winner"):
            assert key in games[0], f"Missing key: {key}"


def test_games_empty():
    df = _make_logs([("NYK", 202425, "2025-01-01", "L", "NYK @ BOS", 100, "g001")])
    games = build_games(df)
    assert games == []


# ─── build_output ─────────────────────────────────────────────────────────────

def test_build_output_structure():
    df = _make_logs([
        ("BOS", 202425, "2025-01-01", "W", "BOS vs. NYK", 110, "g001"),
        ("NYK", 202425, "2025-01-01", "L", "NYK @ BOS", 100, "g001"),
    ])
    output = build_output(df)
    assert "seasons" in output
    assert "data" in output
    assert isinstance(output["seasons"], list)
    assert isinstance(output["data"], dict)


def test_build_output_seasons_newest_first():
    df = _make_logs([
        ("BOS", code, "2025-01-01", "W", "BOS vs. NYK", 110, f"g{i:03d}")
        for i, code in enumerate(SEASON_CODES)
    ])
    output = build_output(df)
    seasons = output["seasons"]
    # Seasons list should be newest first (2024-25 first, 2020-21 last)
    assert seasons[0] == "2024-25"
    assert seasons[-1] == "2020-21"


def test_season_label_all_known_codes():
    """Every code in SEASON_CODES must have a proper label (not just str(code))."""
    for code in SEASON_CODES:
        label = season_label(code)
        assert isinstance(label, str) and len(label) > 0
        # Labels should be like "2024-25", not just "202425"
        assert "-" in label, f"Expected formatted label for {code}, got {label!r}"


def test_standings_wins_plus_losses_equals_games_played():
    """w + l must equal the total number of games played for each team."""
    df = _make_logs([
        ("BOS", 202425, "2025-01-01", "W", "BOS vs. NYK", 110, "g001"),
        ("BOS", 202425, "2025-01-02", "L", "BOS vs. MIA", 95, "g002"),
        ("BOS", 202425, "2025-01-03", "W", "BOS vs. PHI", 112, "g003"),
    ])
    standings = build_standings(df)
    bos = next(s for s in standings if s["abbr"] == "BOS")
    assert bos["w"] + bos["l"] == 3


def test_standings_multiple_teams():
    """Three teams with different records produce three separate standings entries."""
    df = _make_logs([
        ("BOS", 202425, "2025-01-01", "W", "BOS vs. NYK", 110, "g001"),
        ("LAL", 202425, "2025-01-01", "W", "LAL vs. GSW", 112, "g002"),
        ("CHI", 202425, "2025-01-01", "L", "CHI vs. MIL", 88, "g003"),
    ])
    standings = build_standings(df)
    abbrs = {s["abbr"] for s in standings}
    assert "BOS" in abbrs
    assert "LAL" in abbrs
    assert "CHI" in abbrs


def test_games_home_pts_correct():
    """home_pts in the game entry must match the home team's pts from the log."""
    df = _make_logs([
        ("OKC", 202425, "2025-01-05", "W", "OKC vs. UTA", 127, "g010"),
        ("UTA", 202425, "2025-01-05", "L", "UTA @ OKC", 112, "g010"),
    ])
    games = build_games(df)
    assert len(games) == 1
    assert games[0]["home_pts"] == 127


def test_build_output_skips_empty_seasons():
    # Only 202425 data — other seasons should be absent from data dict
    df = _make_logs([
        ("BOS", 202425, "2025-01-01", "W", "BOS vs. NYK", 110, "g001"),
    ])
    output = build_output(df)
    assert "2024-25" in output["data"]
    assert "2020-21" not in output["data"]
