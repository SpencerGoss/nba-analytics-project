"""
Tests for scripts/fetch_live_scores.py
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

def test_import():
    from scripts.fetch_live_scores import (
        _parse_live_game,
        _normalise_clock,
        fetch_live_scores,
        build_live_scores,
    )
    assert callable(fetch_live_scores)
    assert callable(build_live_scores)


# ---------------------------------------------------------------------------
# _normalise_clock
# ---------------------------------------------------------------------------

def test_normalise_clock_iso_format():
    from scripts.fetch_live_scores import _normalise_clock
    assert _normalise_clock("PT04M32.00S") == "4:32"


def test_normalise_clock_zero_seconds():
    from scripts.fetch_live_scores import _normalise_clock
    assert _normalise_clock("PT10M00.00S") == "10:00"


def test_normalise_clock_passthrough_plain():
    from scripts.fetch_live_scores import _normalise_clock
    assert _normalise_clock("4:32") == "4:32"


def test_normalise_clock_empty():
    from scripts.fetch_live_scores import _normalise_clock
    assert _normalise_clock("") == ""


def test_normalise_clock_half_minute():
    from scripts.fetch_live_scores import _normalise_clock
    # PT00M30.00S -> 0:30
    assert _normalise_clock("PT00M30.00S") == "0:30"


# ---------------------------------------------------------------------------
# _parse_live_game
# ---------------------------------------------------------------------------

def test_parse_live_game_full():
    from scripts.fetch_live_scores import _parse_live_game
    game = {
        "gameId": "0022500800",
        "homeTeam": {"teamTricode": "LAC", "score": 87},
        "awayTeam": {"teamTricode": "DET", "score": 72},
        "gameStatus": 2,
        "gameStatusText": "In Progress",
        "period": 3,
        "gameClock": "PT04M32.00S",
    }
    result = _parse_live_game(game)
    assert result["game_id"] == "0022500800"
    assert result["home_team"] == "LAC"
    assert result["away_team"] == "DET"
    assert result["home_score"] == 87
    assert result["away_score"] == 72
    assert result["status"] == "In Progress"
    assert result["period"] == 3
    assert result["clock"] == "4:32"
    assert result["status_code"] == 2


def test_parse_live_game_pregame():
    from scripts.fetch_live_scores import _parse_live_game
    game = {
        "gameId": "0022500801",
        "homeTeam": {"teamTricode": "BOS", "score": 0},
        "awayTeam": {"teamTricode": "NYK", "score": 0},
        "gameStatus": 1,
        "gameStatusText": "7:30 pm ET",
        "period": 0,
        "gameClock": "",
    }
    result = _parse_live_game(game)
    assert result["status_code"] == 1
    assert result["home_score"] == 0
    assert result["away_score"] == 0
    assert result["period"] == 0


def test_parse_live_game_final():
    from scripts.fetch_live_scores import _parse_live_game
    game = {
        "gameId": "0022500802",
        "homeTeam": {"teamTricode": "OKC", "score": 114},
        "awayTeam": {"teamTricode": "POR", "score": 98},
        "gameStatus": 3,
        "gameStatusText": "Final",
        "period": 4,
        "gameClock": "",
    }
    result = _parse_live_game(game)
    assert result["status"] == "Final"
    assert result["status_code"] == 3
    assert result["home_score"] == 114


def test_parse_live_game_missing_keys():
    """_parse_live_game must not crash on minimal/empty input."""
    from scripts.fetch_live_scores import _parse_live_game
    result = _parse_live_game({})
    assert result["game_id"] == ""
    assert result["home_team"] == ""
    assert result["away_team"] == ""
    assert result["home_score"] == 0
    assert result["away_score"] == 0


def test_parse_live_game_schema():
    from scripts.fetch_live_scores import _parse_live_game
    game = {
        "gameId": "0022500800",
        "homeTeam": {"teamTricode": "LAC", "score": 87},
        "awayTeam": {"teamTricode": "DET", "score": 72},
        "gameStatus": 2,
        "gameStatusText": "In Progress",
        "period": 3,
        "gameClock": "PT04M32.00S",
    }
    result = _parse_live_game(game)
    required_keys = {
        "game_id", "home_team", "away_team", "home_score", "away_score",
        "status", "period", "clock", "status_code",
    }
    assert required_keys == set(result.keys())


# ---------------------------------------------------------------------------
# fetch_live_scores -- mocked nba_api
# ---------------------------------------------------------------------------

def test_fetch_live_scores_returns_list_on_api_success():
    """When nba_api returns games, fetch_live_scores returns a list of dicts."""
    import scripts.fetch_live_scores as fls

    mock_game = {
        "gameId": "0022500800",
        "homeTeam": {"teamTricode": "LAC", "score": 87},
        "awayTeam": {"teamTricode": "DET", "score": 72},
        "gameStatus": 2,
        "gameStatusText": "In Progress",
        "period": 3,
        "gameClock": "PT04M32.00S",
    }

    parsed_game = fls._parse_live_game(mock_game)

    # Patch at the module boundary: _parse_live_game is already tested;
    # here we verify fetch_live_scores correctly passes API output through it.
    mock_board = MagicMock()
    mock_board.games.get_dict.return_value = [mock_game]
    mock_scoreboard_cls = MagicMock(return_value=mock_board)

    with patch("nba_api.live.nba.endpoints.scoreboard.ScoreBoard", mock_scoreboard_cls):
        result = fls.fetch_live_scores()

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["home_team"] == "LAC"
    assert result[0]["clock"] == "4:32"


def test_fetch_live_scores_returns_empty_on_exception():
    """Any exception from nba_api -> returns [] without crashing."""
    from scripts.fetch_live_scores import fetch_live_scores

    mock_scoreboard_mod = MagicMock()
    mock_scoreboard_mod.ScoreBoard.side_effect = Exception("API error")

    with patch.dict(
        "sys.modules",
        {
            "nba_api": MagicMock(),
            "nba_api.live": MagicMock(),
            "nba_api.live.nba": MagicMock(),
            "nba_api.live.nba.endpoints": MagicMock(),
            "nba_api.live.nba.endpoints.scoreboard": mock_scoreboard_mod,
        },
    ):
        import importlib
        import scripts.fetch_live_scores as fls
        importlib.reload(fls)
        result = fls.fetch_live_scores()
        assert result == []


# ---------------------------------------------------------------------------
# build_live_scores (integration)
# ---------------------------------------------------------------------------

def test_build_live_scores_writes_json(tmp_path):
    """build_live_scores writes valid JSON payload even when no games returned."""
    out_path = tmp_path / "live_scores.json"

    # Patch fetch_live_scores to return empty list (no API call)
    with patch("scripts.fetch_live_scores.fetch_live_scores", return_value=[]):
        import scripts.fetch_live_scores as fls
        fls.build_live_scores(out_path=out_path)

    assert out_path.exists()
    with open(out_path) as f:
        data = json.load(f)
    assert "games" in data
    assert "last_updated" in data
    assert isinstance(data["games"], list)


def test_build_live_scores_empty_produces_valid_json(tmp_path):
    """Empty game list -> writes payload with empty games array."""
    out_path = tmp_path / "live_scores.json"
    with patch("scripts.fetch_live_scores.fetch_live_scores", return_value=[]):
        import scripts.fetch_live_scores as fls
        result = fls.build_live_scores(out_path=out_path)
    assert result == []
    with open(out_path) as f:
        data = json.load(f)
    assert data["games"] == []


def test_build_live_scores_with_games(tmp_path):
    """When fetch returns games, payload contains them."""
    out_path = tmp_path / "live_scores.json"
    mock_games = [
        {
            "game_id": "0022500800",
            "home_team": "LAC", "away_team": "DET",
            "home_score": 87, "away_score": 72,
            "status": "In Progress", "period": 3, "clock": "4:32",
            "status_code": 2,
        }
    ]
    with patch("scripts.fetch_live_scores.fetch_live_scores", return_value=mock_games):
        import scripts.fetch_live_scores as fls
        result = fls.build_live_scores(out_path=out_path)

    assert len(result) == 1
    with open(out_path) as f:
        data = json.load(f)
    assert len(data["games"]) == 1
    assert data["games"][0]["home_team"] == "LAC"


# ---------------------------------------------------------------------------
# _status_label
# ---------------------------------------------------------------------------

def test_status_label_pregame():
    from scripts.fetch_live_scores import _status_label
    assert _status_label(1) == "Pre-Game"


def test_status_label_in_progress():
    from scripts.fetch_live_scores import _status_label
    assert _status_label(2) == "In Progress"


def test_status_label_final():
    from scripts.fetch_live_scores import _status_label
    assert _status_label(3) == "Final"


def test_status_label_unknown_code():
    from scripts.fetch_live_scores import _status_label
    assert _status_label(99) == "Unknown"


def test_status_label_zero_is_unknown():
    from scripts.fetch_live_scores import _status_label
    assert _status_label(0) == "Unknown"


def test_status_label_returns_string():
    from scripts.fetch_live_scores import _status_label
    result = _status_label(1)
    assert isinstance(result, str)
