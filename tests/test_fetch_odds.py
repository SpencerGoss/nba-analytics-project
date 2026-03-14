"""Tests for scripts/fetch_odds.py — Pinnacle odds fetcher.

Tests the pure helper functions and data assembly logic.
API calls are mocked to avoid hitting the live Pinnacle API.
"""
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

import scripts.fetch_odds as fo


# ---------------------------------------------------------------------------
# american_odds_to_implied_prob
# ---------------------------------------------------------------------------

def test_implied_prob_favorite():
    # -200 => 200/(200+100) = 0.6667
    result = fo.american_odds_to_implied_prob(-200)
    assert abs(result - 0.6667) < 0.001


def test_implied_prob_underdog():
    # +200 => 100/(200+100) = 0.3333
    result = fo.american_odds_to_implied_prob(200)
    assert abs(result - 0.3333) < 0.001


def test_implied_prob_even():
    # +100 => 100/(100+100) = 0.5
    result = fo.american_odds_to_implied_prob(100)
    assert result == 0.5


def test_implied_prob_none():
    assert fo.american_odds_to_implied_prob(None) is None


def test_implied_prob_big_favorite():
    # -1000 => 1000/1100 = 0.9091
    result = fo.american_odds_to_implied_prob(-1000)
    assert abs(result - 0.9091) < 0.001


def test_implied_prob_big_underdog():
    # +1000 => 100/1100 = 0.0909
    result = fo.american_odds_to_implied_prob(1000)
    assert abs(result - 0.0909) < 0.001


# ---------------------------------------------------------------------------
# team_name_to_abb
# ---------------------------------------------------------------------------

def test_known_team():
    assert fo.team_name_to_abb("Boston Celtics") == "BOS"
    assert fo.team_name_to_abb("Los Angeles Lakers") == "LAL"
    assert fo.team_name_to_abb("Oklahoma City Thunder") == "OKC"


def test_unknown_team():
    result = fo.team_name_to_abb("Seattle SuperSonics")
    assert result == "Seattle SuperSonics"  # pass through


def test_all_30_teams_mapped():
    assert len(fo.ODDS_TEAM_TO_ABB) == 30
    for name, abb in fo.ODDS_TEAM_TO_ABB.items():
        assert len(abb) == 3
        assert abb == abb.upper()


# ---------------------------------------------------------------------------
# fetch_game_lines (mocked)
# ---------------------------------------------------------------------------

MOCK_MATCHUPS = [
    {
        "id": 1001,
        "parentId": None,
        "participants": [
            {"alignment": "home", "name": "Boston Celtics"},
            {"alignment": "away", "name": "Los Angeles Lakers"},
        ],
        "startTime": "2026-03-15T00:30:00Z",
    },
    {
        "id": 1002,
        "parentId": None,
        "participants": [
            {"alignment": "home", "name": "Golden State Warriors"},
            {"alignment": "away", "name": "Miami Heat"},
        ],
        "startTime": "2026-03-15T03:00:00Z",
    },
    # Alt line (parentId set) — should be filtered out
    {
        "id": 1003,
        "parentId": 1001,
        "participants": [
            {"alignment": "home", "name": "Boston Celtics"},
            {"alignment": "away", "name": "Los Angeles Lakers"},
        ],
        "startTime": "2026-03-15T00:30:00Z",
    },
    # Futures (neutral) — should be filtered out
    {
        "id": 1004,
        "parentId": None,
        "participants": [
            {"alignment": "neutral", "name": "NBA Championship Winner"},
        ],
        "startTime": "2026-06-01T00:00:00Z",
    },
]

MOCK_MARKETS = [
    # Moneyline for game 1001
    {
        "matchupId": 1001,
        "type": "moneyline",
        "period": 0,
        "prices": [
            {"designation": "home", "price": -180},
            {"designation": "away", "price": 155},
        ],
    },
    # Spread for game 1001
    {
        "matchupId": 1001,
        "type": "spread",
        "period": 0,
        "prices": [
            {"designation": "home", "points": -4.5, "price": -110},
            {"designation": "away", "points": 4.5},
        ],
    },
    # Alt spread for game 1001 (should be ignored — mid already in spreads)
    {
        "matchupId": 1001,
        "type": "spread",
        "period": 0,
        "prices": [
            {"designation": "home", "points": -7.5, "price": 120},
            {"designation": "away", "points": 7.5},
        ],
    },
    # Total for game 1001
    {
        "matchupId": 1001,
        "type": "total",
        "period": 0,
        "prices": [
            {"designation": "over", "points": 220.5, "price": -110},
            {"designation": "under", "points": 220.5, "price": -110},
        ],
    },
    # Period 1 market — should be filtered out
    {
        "matchupId": 1001,
        "type": "moneyline",
        "period": 1,
        "prices": [
            {"designation": "home", "price": -130},
        ],
    },
    # Moneyline for game 1002
    {
        "matchupId": 1002,
        "type": "moneyline",
        "period": 0,
        "prices": [
            {"designation": "home", "price": 110},
            {"designation": "away", "price": -130},
        ],
    },
]


@patch("scripts.fetch_odds.get_pinnacle")
def test_fetch_game_lines_basic(mock_api):
    mock_api.side_effect = [MOCK_MATCHUPS, MOCK_MARKETS]
    df = fo.fetch_game_lines()

    assert len(df) == 2
    assert set(df.columns) >= {"date", "home_team", "away_team", "home_moneyline", "spread", "total"}

    bos_row = df[df["home_team"] == "BOS"].iloc[0]
    assert bos_row["away_team"] == "LAL"
    assert bos_row["home_moneyline"] == -180
    assert bos_row["spread"] == -4.5
    assert bos_row["total"] == 220.5
    # UTC 2026-03-15T00:30:00Z = 7:30 PM ET on 2026-03-14
    assert bos_row["date"] == "2026-03-14"


@patch("scripts.fetch_odds.get_pinnacle")
def test_fetch_game_lines_alt_spread_ignored(mock_api):
    """Alt spreads should not overwrite the primary spread."""
    mock_api.side_effect = [MOCK_MATCHUPS, MOCK_MARKETS]
    df = fo.fetch_game_lines()
    bos_row = df[df["home_team"] == "BOS"].iloc[0]
    assert bos_row["spread"] == -4.5  # Primary, not -7.5 alt


@patch("scripts.fetch_odds.get_pinnacle")
def test_fetch_game_lines_filters_futures(mock_api):
    """Futures (neutral alignment) should be excluded."""
    mock_api.side_effect = [MOCK_MATCHUPS, MOCK_MARKETS]
    df = fo.fetch_game_lines()
    assert len(df) == 2  # Only 2 real games, not the future


@patch("scripts.fetch_odds.get_pinnacle")
def test_fetch_game_lines_empty_matchups(mock_api):
    mock_api.side_effect = [[], MOCK_MARKETS]
    df = fo.fetch_game_lines()
    assert len(df) == 0


@patch("scripts.fetch_odds.get_pinnacle")
def test_fetch_game_lines_none_response(mock_api):
    mock_api.return_value = None
    df = fo.fetch_game_lines()
    assert len(df) == 0


# ---------------------------------------------------------------------------
# build_model_vs_odds
# ---------------------------------------------------------------------------

def test_build_model_vs_odds_game_rows():
    game_lines = pd.DataFrame([{
        "date": "2026-03-15",
        "home_team": "BOS",
        "away_team": "LAL",
        "home_moneyline": -180,
        "away_moneyline": 155,
        "spread": -4.5,
        "total": 220.5,
    }])
    game_proj = pd.DataFrame([{
        "date": "2026-03-15",
        "home_team": "BOS",
        "away_team": "LAL",
        "model_win_prob": 0.72,
    }])
    player_props = pd.DataFrame()
    player_proj = pd.DataFrame()

    result = fo.build_model_vs_odds(game_lines, player_props, game_proj, player_proj)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["stat"] == "win_prob"
    assert row["model_projection"] == 0.72
    # Implied prob of -180 is ~0.6429
    assert abs(row["sportsbook_line"] - 0.6429) < 0.01
    assert row["gap"] is not None


def test_build_model_vs_odds_flagging():
    """Gap > 5pp should be flagged."""
    game_lines = pd.DataFrame([{
        "date": "2026-03-15",
        "home_team": "BOS",
        "away_team": "LAL",
        "home_moneyline": 100,  # implied 0.5
        "away_moneyline": -120,
        "spread": 2.0,
        "total": 215.0,
    }])
    game_proj = pd.DataFrame([{
        "date": "2026-03-15",
        "home_team": "BOS",
        "away_team": "LAL",
        "model_win_prob": 0.65,  # gap = 0.15 > 0.05 threshold
    }])
    result = fo.build_model_vs_odds(game_lines, pd.DataFrame(), game_proj, pd.DataFrame())
    assert bool(result.iloc[0]["flagged"]) is True


def test_build_model_vs_odds_no_model():
    """Missing model projection should produce None gap."""
    game_lines = pd.DataFrame([{
        "date": "2026-03-15",
        "home_team": "BOS",
        "away_team": "LAL",
        "home_moneyline": -150,
        "away_moneyline": 130,
        "spread": -3.0,
        "total": 218.0,
    }])
    game_proj = pd.DataFrame(columns=["date", "home_team", "away_team", "model_win_prob"])
    result = fo.build_model_vs_odds(game_lines, pd.DataFrame(), game_proj, pd.DataFrame())
    assert len(result) == 1
    assert result.iloc[0]["gap"] is None
