"""
Tests for scripts/build_h2h.py
"""

from __future__ import annotations

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_games(rows: list[dict]) -> pd.DataFrame:
    """Create a per-game indexed DataFrame matching _build_game_index output."""
    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")
    return df


SAMPLE_GAMES = _make_games([
    {"game_id": 1, "game_date": "2024-01-01", "home_team": "OKC", "away_team": "POR",
     "home_score": 118, "away_score": 104, "home_wl": "W", "winner": "OKC", "margin": 14, "season": 202324},
    {"game_id": 2, "game_date": "2024-02-15", "home_team": "POR", "away_team": "OKC",
     "home_score": 112, "away_score": 110, "home_wl": "W", "winner": "POR", "margin": 2, "season": 202324},
    {"game_id": 3, "game_date": "2024-03-20", "home_team": "OKC", "away_team": "POR",
     "home_score": 125, "away_score": 105, "home_wl": "W", "winner": "OKC", "margin": 20, "season": 202324},
    # Unrelated game - should not appear
    {"game_id": 4, "game_date": "2024-03-21", "home_team": "LAL", "away_team": "GSW",
     "home_score": 110, "away_score": 100, "home_wl": "W", "winner": "LAL", "margin": 10, "season": 202324},
])

SAMPLE_PICKS = [
    {"home_team": "OKC", "away_team": "POR"},
    {"home_team": "LAL", "away_team": "BOS"},  # no H2H data in sample
]


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

def test_import():
    from scripts.build_h2h import (
        _get_h2h_meetings,
        _series_record,
        compute_h2h,
    )
    assert callable(compute_h2h)


# ---------------------------------------------------------------------------
# _get_h2h_meetings
# ---------------------------------------------------------------------------

def test_get_h2h_meetings_count():
    from scripts.build_h2h import _get_h2h_meetings
    meetings = _get_h2h_meetings(SAMPLE_GAMES, "OKC", "POR", n=10)
    # OKC vs POR appears in game_ids 1, 2, 3
    assert len(meetings) == 3


def test_get_h2h_meetings_limit():
    from scripts.build_h2h import _get_h2h_meetings
    meetings = _get_h2h_meetings(SAMPLE_GAMES, "OKC", "POR", n=2)
    assert len(meetings) == 2


def test_get_h2h_meetings_bidirectional():
    """Should find games regardless of which team is home/away."""
    from scripts.build_h2h import _get_h2h_meetings
    meetings_ab = _get_h2h_meetings(SAMPLE_GAMES, "OKC", "POR")
    meetings_ba = _get_h2h_meetings(SAMPLE_GAMES, "POR", "OKC")
    assert len(meetings_ab) == len(meetings_ba)


def test_get_h2h_meetings_no_history():
    from scripts.build_h2h import _get_h2h_meetings
    meetings = _get_h2h_meetings(SAMPLE_GAMES, "LAL", "BOS")
    assert meetings.empty


def test_get_h2h_meetings_sorted_desc():
    """Most recent meeting should be first."""
    from scripts.build_h2h import _get_h2h_meetings
    meetings = _get_h2h_meetings(SAMPLE_GAMES, "OKC", "POR")
    dates = meetings["game_date"].tolist()
    assert dates == sorted(dates, reverse=True)


# ---------------------------------------------------------------------------
# _series_record
# ---------------------------------------------------------------------------

def test_series_record_leader():
    from scripts.build_h2h import _series_record
    meetings = _get_h2h_meetings_helper()
    record = _series_record(meetings, "OKC", "POR")
    # OKC wins 2, POR wins 1
    assert "OKC leads 2-1" in record


def test_series_record_tied():
    from scripts.build_h2h import _series_record
    tied_games = _make_games([
        {"game_id": 1, "game_date": "2024-01-01", "home_team": "OKC", "away_team": "POR",
         "home_score": 110, "away_score": 100, "home_wl": "W", "winner": "OKC", "margin": 10, "season": 202324},
        {"game_id": 2, "game_date": "2024-02-01", "home_team": "POR", "away_team": "OKC",
         "home_score": 115, "away_score": 108, "home_wl": "W", "winner": "POR", "margin": 7, "season": 202324},
    ])
    record = _series_record(tied_games, "OKC", "POR")
    assert "tied" in record.lower()


def test_series_record_empty():
    from scripts.build_h2h import _series_record
    record = _series_record(pd.DataFrame(), "OKC", "POR")
    assert "no" in record.lower() or "found" in record.lower()


def _get_h2h_meetings_helper():
    from scripts.build_h2h import _get_h2h_meetings
    return _get_h2h_meetings(SAMPLE_GAMES, "OKC", "POR")


# ---------------------------------------------------------------------------
# compute_h2h
# ---------------------------------------------------------------------------

def test_compute_h2h_output_length():
    from scripts.build_h2h import compute_h2h
    results = compute_h2h(SAMPLE_PICKS, SAMPLE_GAMES)
    assert len(results) == 2


def test_compute_h2h_keys():
    from scripts.build_h2h import compute_h2h
    results = compute_h2h(SAMPLE_PICKS, SAMPLE_GAMES)
    okc_por = results[0]
    expected_keys = {"home_team", "away_team", "series_record", "avg_total", "meetings"}
    assert expected_keys == set(okc_por.keys())


def test_compute_h2h_avg_total():
    from scripts.build_h2h import compute_h2h
    results = compute_h2h(SAMPLE_PICKS, SAMPLE_GAMES)
    okc_por = results[0]
    # Game totals: 222, 222, 230
    expected = round((222 + 222 + 230) / 3, 1)
    assert okc_por["avg_total"] == expected


def test_compute_h2h_meetings_structure():
    from scripts.build_h2h import compute_h2h
    results = compute_h2h(SAMPLE_PICKS, SAMPLE_GAMES)
    okc_por = results[0]
    assert len(okc_por["meetings"]) == 3
    meeting = okc_por["meetings"][0]
    assert "date" in meeting
    assert "home_team" in meeting
    assert "home_score" in meeting
    assert "away_score" in meeting
    assert "winner" in meeting
    assert "margin" in meeting


def test_compute_h2h_no_history_team():
    """A matchup with no history should still appear with empty meetings."""
    from scripts.build_h2h import compute_h2h
    results = compute_h2h(SAMPLE_PICKS, SAMPLE_GAMES)
    lal_bos = results[1]
    assert lal_bos["meetings"] == []
    assert lal_bos["avg_total"] is None


def test_compute_h2h_home_team_field():
    from scripts.build_h2h import compute_h2h
    results = compute_h2h(SAMPLE_PICKS, SAMPLE_GAMES)
    assert results[0]["home_team"] == "OKC"
    assert results[0]["away_team"] == "POR"


# ---------------------------------------------------------------------------
# _build_game_index integration smoke test
# ---------------------------------------------------------------------------

def test_build_game_index_from_logs():
    """_build_game_index should produce one row per game."""
    from scripts.build_h2h import _build_game_index
    import pandas as pd

    raw_logs = pd.DataFrame([
        {"game_id": 99, "team_abbreviation": "LAL", "matchup": "LAL vs. BOS",
         "wl": "W", "pts": 110, "game_date": "2025-01-01", "season": 202526},
        {"game_id": 99, "team_abbreviation": "BOS", "matchup": "BOS @ LAL",
         "wl": "L", "pts": 100, "game_date": "2025-01-01", "season": 202526},
    ])
    raw_logs["game_date"] = pd.to_datetime(raw_logs["game_date"], format="mixed")
    raw_logs["is_home"] = raw_logs["matchup"].str.contains(" vs. ", regex=False)

    games = _build_game_index(raw_logs)
    assert len(games) == 1
    assert games.iloc[0]["home_team"] == "LAL"
    assert games.iloc[0]["away_team"] == "BOS"
    assert games.iloc[0]["winner"] == "LAL"
    assert games.iloc[0]["home_score"] == 110
    assert games.iloc[0]["away_score"] == 100
