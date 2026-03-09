"""Tests for scripts/build_playoff_odds.py"""
import sys
from pathlib import Path
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_playoff_odds import (
    _games_behind,
    _playoff_pct,
    _compute_conference_standings,
    _title_odds_map,
)


# ─── _games_behind ───────────────────────────────────────────────────────────

def test_games_behind_leader():
    assert _games_behind(40, 10, 40, 10) == 0.0


def test_games_behind_two_games():
    # Leader: 40W-10L; Team: 39W-12L
    gb = _games_behind(40, 10, 39, 12)
    assert gb == pytest.approx(1.5)


def test_games_behind_positive():
    gb = _games_behind(50, 5, 40, 15)
    assert gb > 0


# ─── _playoff_pct ────────────────────────────────────────────────────────────

def test_playoff_pct_top6_always_100():
    for rank in range(1, 7):
        assert _playoff_pct(rank, 0) == 100.0
    assert _playoff_pct(6, 8) == 100.0  # Even with GB, top 6 = 100%


def test_playoff_pct_playin_zone():
    # Rank 7-8 play-in
    assert _playoff_pct(7, 3) == 75.0
    assert _playoff_pct(8, 7) == 50.0
    assert _playoff_pct(7, 12) == 15.0


def test_playoff_pct_bubble_10():
    assert _playoff_pct(10, 0) == 50.0
    assert _playoff_pct(10, 6) == max(0.0, 50.0 - 6*8.0)


def test_playoff_pct_eliminated():
    assert _playoff_pct(11, 10) == 0.0
    assert _playoff_pct(15, 20) == 0.0


def test_playoff_pct_never_negative():
    # Even with huge GB, pct should be >= 0
    assert _playoff_pct(10, 100) >= 0.0


# ─── _compute_conference_standings ───────────────────────────────────────────

def _make_logs(records: list[tuple]) -> pd.DataFrame:
    """Helper: list of (team, wl) pairs."""
    rows = [{"team_abbreviation": t, "wl": w} for t, w in records]
    return pd.DataFrame(rows)


def test_standings_ordering():
    logs = _make_logs([
        ("BOS", "W"), ("BOS", "W"), ("BOS", "W"),  # 3-0
        ("NYK", "W"), ("NYK", "L"),                  # 1-1
        ("MIA", "L"), ("MIA", "L"),                  # 0-2
    ])
    results = _compute_conference_standings(logs, ["BOS", "NYK", "MIA"])
    assert results[0]["abbr"] == "BOS"
    assert results[-1]["abbr"] == "MIA"


def test_standings_rank_assigned():
    logs = _make_logs([("OKC", "W"), ("LAL", "L")])
    results = _compute_conference_standings(logs, ["OKC", "LAL"])
    ranks = [r["rank"] for r in results]
    assert sorted(ranks) == list(range(1, len(results)+1))


def test_standings_leader_gb_zero():
    logs = _make_logs([("BOS", "W"), ("BOS", "W")])
    results = _compute_conference_standings(logs, ["BOS"])
    assert results[0]["gb"] == 0.0


def test_standings_required_fields():
    logs = _make_logs([("DET", "W"), ("DET", "L")])
    results = _compute_conference_standings(logs, ["DET"])
    row = results[0]
    for key in ("abbr", "w", "l", "win_pct", "rank", "gb"):
        assert key in row, f"Missing field: {key}"


def test_standings_empty_conf():
    logs = _make_logs([("BOS", "W")])
    results = _compute_conference_standings(logs, [])
    assert results == []


def test_standings_win_pct_range():
    logs = _make_logs([("GSW", "W"), ("GSW", "W"), ("GSW", "L")])
    results = _compute_conference_standings(logs, ["GSW"])
    assert 0.0 <= results[0]["win_pct"] <= 1.0


# ─── _title_odds_map ─────────────────────────────────────────────────────────

def test_title_odds_sum_near_100():
    logs = _make_logs([
        ("BOS", "W"), ("BOS", "W"),
        ("NYK", "W"), ("NYK", "L"),
        ("MIA", "L"), ("MIA", "L"),
    ])
    east = _compute_conference_standings(logs, ["BOS", "NYK", "MIA"])
    west_logs = _make_logs([("OKC", "W"), ("LAL", "W"), ("GSW", "L")])
    west = _compute_conference_standings(west_logs, ["OKC", "LAL", "GSW"])
    odds = _title_odds_map([east, west])
    total = sum(odds.values())
    assert abs(total - 100.0) < 1.0  # Should sum ~100%


def test_title_odds_best_team_highest():
    logs = _make_logs([
        ("BOS", "W"), ("BOS", "W"), ("BOS", "W"),
        ("MIA", "L"), ("MIA", "L"),
    ])
    east = _compute_conference_standings(logs, ["BOS", "MIA"])
    odds = _title_odds_map([east])
    assert odds.get("BOS", 0) > odds.get("MIA", 0)


def test_title_odds_no_teams():
    assert _title_odds_map([]) == {}
    assert _title_odds_map([[]]) == {}


def test_title_odds_all_teams_have_entry():
    logs = _make_logs([
        ("BOS", "W"), ("NYK", "W"), ("MIA", "L"),
    ])
    conf = _compute_conference_standings(logs, ["BOS", "NYK", "MIA"])
    odds = _title_odds_map([conf])
    for team in ["BOS", "NYK", "MIA"]:
        assert team in odds
