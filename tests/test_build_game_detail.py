"""Tests for scripts/build_game_detail.py."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def data_dir(tmp_path):
    """Create minimal JSON fixtures for game detail builder."""
    picks = [
        {
            "home_team": "OKC",
            "away_team": "LAL",
            "game_date": "2026-03-13",
            "predicted_winner": "OKC",
            "home_win_prob": 0.72,
            "away_win_prob": 0.28,
            "model_confidence": 72,
            "confidence_tier": "Solid Pick",
            "spread": -5.5,
            "projected_margin": 6.2,
            "ats_pick": "OKC",
            "value_bet": True,
            "edge_pct": 0.08,
            "kelly_fraction": 0.02,
        }
    ]
    picks_path = tmp_path / "todays_picks.json"
    picks_path.write_text(json.dumps(picks))

    standings = {
        "east": [],
        "west": [
            {"team": "OKC", "w": 50, "l": 12, "win_pct": 0.806,
             "home_record": "28-4", "away_record": "22-8",
             "conference": "West", "rank": 1},
            {"team": "LAL", "w": 38, "l": 24, "win_pct": 0.613,
             "home_record": "22-9", "away_record": "16-15",
             "conference": "West", "rank": 5},
        ],
    }
    standings_path = tmp_path / "standings.json"
    standings_path.write_text(json.dumps(standings))

    context = [
        {
            "home_team": "OKC",
            "away_team": "LAL",
            "home_b2b": False,
            "away_b2b": True,
            "home_rest_days": 2,
            "away_rest_days": 0,
            "home_last10": "8-2",
            "away_last10": "5-5",
            "home_streak": 5,
            "away_streak": -2,
            "situational_flags": ["AWAY_B2B", "REST_ADV_HOME", "HOME_HOT"],
            "context_summary": "OKC has rest advantage",
        }
    ]
    context_path = tmp_path / "game_context.json"
    context_path.write_text(json.dumps(context))

    injuries_path = tmp_path / "injuries.json"
    injuries_path.write_text("[]")

    h2h_path = tmp_path / "head_to_head.json"
    h2h_path.write_text("[]")

    return tmp_path, picks_path, standings_path, context_path, injuries_path, h2h_path


@pytest.fixture
def out_json(tmp_path):
    return tmp_path / "output" / "game_detail.json"


def _patch_and_build(monkeypatch, data_dir_tuple, out_json_path):
    tmp_path, picks, standings, context, injuries, h2h = data_dir_tuple
    import scripts.build_game_detail as mod

    monkeypatch.setattr(mod, "PICKS_JSON", picks)
    monkeypatch.setattr(mod, "STANDINGS_JSON", standings)
    monkeypatch.setattr(mod, "CONTEXT_JSON", context)
    monkeypatch.setattr(mod, "INJURIES_JSON", injuries)
    monkeypatch.setattr(mod, "H2H_JSON", h2h)
    monkeypatch.setattr(mod, "OUT_JSON", out_json_path)
    # Prevent CSV fallback for H2H
    monkeypatch.setattr(mod, "TEAM_LOGS_CSV", tmp_path / "nonexistent.csv")
    return mod.build_game_detail()


def test_output_has_games(monkeypatch, data_dir, out_json):
    result = _patch_and_build(monkeypatch, data_dir, out_json)
    assert "games" in result
    assert len(result["games"]) == 1


def test_game_structure(monkeypatch, data_dir, out_json):
    result = _patch_and_build(monkeypatch, data_dir, out_json)
    game = result["games"][0]
    assert game["game_id"] == "OKC_LAL_2026-03-13"
    assert game["home"]["team"] == "OKC"
    assert game["away"]["team"] == "LAL"
    assert "prediction" in game
    assert "context" in game
    assert "h2h" in game
    assert "injuries" in game


def test_prediction_fields(monkeypatch, data_dir, out_json):
    result = _patch_and_build(monkeypatch, data_dir, out_json)
    pred = result["games"][0]["prediction"]
    assert pred["predicted_winner"] == "OKC"
    assert pred["home_win_prob"] == 0.72
    assert pred["value_bet"] is True


def test_context_factors(monkeypatch, data_dir, out_json):
    result = _patch_and_build(monkeypatch, data_dir, out_json)
    ctx = result["games"][0]["context"]
    factors = ctx["factors"]
    # Should have: moderate confidence, away B2B, rest advantage, home hot, record gap, value bet
    assert any("confidence" in f.lower() for f in factors)
    assert any("back-to-back" in f.lower() for f in factors)
    assert any("rest" in f.lower() for f in factors)


def test_standings_populated(monkeypatch, data_dir, out_json):
    result = _patch_and_build(monkeypatch, data_dir, out_json)
    home = result["games"][0]["home"]
    assert home["record"] == "50-12"
    assert home["rank"] == 1


def test_empty_picks(monkeypatch, tmp_path, out_json):
    empty_picks = tmp_path / "empty_picks.json"
    empty_picks.write_text("[]")
    import scripts.build_game_detail as mod

    monkeypatch.setattr(mod, "PICKS_JSON", empty_picks)
    monkeypatch.setattr(mod, "OUT_JSON", out_json)
    result = mod.build_game_detail()
    assert result == {"games": []}


def test_writes_json_file(monkeypatch, data_dir, out_json):
    _patch_and_build(monkeypatch, data_dir, out_json)
    assert out_json.exists()
    with open(out_json) as f:
        data = json.load(f)
    assert len(data["games"]) == 1


def test_generate_factors_decisive_margin():
    from scripts.build_game_detail import _generate_factors
    pick = {"home_team": "OKC", "away_team": "LAL",
            "predicted_winner": "OKC", "model_confidence": 85,
            "projected_margin": 12.5}
    factors = _generate_factors(pick, None, None, None)
    assert any("decisive" in f.lower() for f in factors)


def test_generate_factors_tight_game():
    from scripts.build_game_detail import _generate_factors
    pick = {"home_team": "OKC", "away_team": "LAL",
            "predicted_winner": "OKC", "model_confidence": 55,
            "projected_margin": 1.5}
    factors = _generate_factors(pick, None, None, None)
    assert any("tight" in f.lower() for f in factors)
