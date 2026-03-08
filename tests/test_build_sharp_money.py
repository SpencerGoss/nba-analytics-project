"""
Tests for scripts/build_sharp_money.py

Covers:
  - _sharp_rating: STRONG/MODERATE/LEAN thresholds
  - _edge_component: normalisation and None handling
  - _confidence_component: |prob - 0.5| score and sharp side
  - _situational_component: flag-based advantage score
  - build_sharp_money: graceful empty-input handling, required fields
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_sharp_money import (
    STRONG_THRESHOLD,
    MODERATE_THRESHOLD,
    EDGE_CAP,
    _sharp_rating,
    _edge_component,
    _confidence_component,
    _situational_component,
    build_sharp_money,
)

REQUIRED_FIELDS = {
    "home_team", "away_team", "game_date",
    "sharp_score", "sharp_side", "sharp_rating",
    "edge_pct", "model_confidence", "situational_advantage",
    "key_factors", "kelly_fraction",
}


# ---------------------------------------------------------------------------
# _sharp_rating
# ---------------------------------------------------------------------------

class TestSharpRating:
    def test_strong_at_threshold(self):
        assert _sharp_rating(STRONG_THRESHOLD) == "STRONG"

    def test_strong_above_threshold(self):
        assert _sharp_rating(100) == "STRONG"

    def test_moderate_at_threshold(self):
        assert _sharp_rating(MODERATE_THRESHOLD) == "MODERATE"

    def test_moderate_between_thresholds(self):
        mid = (STRONG_THRESHOLD + MODERATE_THRESHOLD) // 2
        assert _sharp_rating(mid) == "MODERATE"

    def test_lean_below_moderate(self):
        assert _sharp_rating(MODERATE_THRESHOLD - 1) == "LEAN"

    def test_lean_at_zero(self):
        assert _sharp_rating(0) == "LEAN"


# ---------------------------------------------------------------------------
# _edge_component
# ---------------------------------------------------------------------------

class TestEdgeComponent:
    def test_none_vbet_returns_zero(self):
        score, edge, kelly = _edge_component(None)
        assert score == pytest.approx(0.0)
        assert edge is None
        assert kelly is None

    def test_zero_edge_returns_zero(self):
        score, edge, kelly = _edge_component({"edge_pct": 0.0})
        assert score == pytest.approx(0.0)

    def test_full_cap_edge_returns_one(self):
        score, edge, kelly = _edge_component({"edge_pct": EDGE_CAP})
        assert score == pytest.approx(1.0)

    def test_above_cap_clamped_to_one(self):
        score, edge, kelly = _edge_component({"edge_pct": EDGE_CAP * 2})
        assert score == pytest.approx(1.0)

    def test_half_cap_returns_half(self):
        score, edge, kelly = _edge_component({"edge_pct": EDGE_CAP / 2})
        assert score == pytest.approx(0.5)

    def test_kelly_passed_through(self):
        _, _, kelly = _edge_component({"edge_pct": 0.1, "kelly_fraction": 0.05})
        assert kelly == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# _confidence_component
# ---------------------------------------------------------------------------

class TestConfidenceComponent:
    def test_none_pick_returns_defaults(self):
        score, conf, side = _confidence_component(None)
        assert score == pytest.approx(0.0)
        assert conf == pytest.approx(50.0)
        assert side == ""

    def test_high_home_prob_picks_home(self):
        pick = {"home_win_prob": 0.75, "away_win_prob": 0.25,
                "home_team": "BOS", "away_team": "MIA"}
        score, conf, side = _confidence_component(pick)
        assert side == "BOS"
        assert conf == pytest.approx(75.0)
        assert score > 0

    def test_high_away_prob_picks_away(self):
        pick = {"home_win_prob": 0.30, "away_win_prob": 0.70,
                "home_team": "BOS", "away_team": "MIA"}
        _, _, side = _confidence_component(pick)
        assert side == "MIA"

    def test_fifty_fifty_score_is_zero(self):
        pick = {"home_win_prob": 0.50, "away_win_prob": 0.50,
                "home_team": "BOS", "away_team": "MIA"}
        score, _, _ = _confidence_component(pick)
        assert score == pytest.approx(0.0)

    def test_certain_win_score_is_one(self):
        pick = {"home_win_prob": 1.0, "away_win_prob": 0.0,
                "home_team": "BOS", "away_team": "MIA"}
        score, _, _ = _confidence_component(pick)
        assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _situational_component
# ---------------------------------------------------------------------------

class TestSituationalComponent:
    def test_none_ctx_returns_zero(self):
        score, flag, factors = _situational_component(None)
        assert score == pytest.approx(0.0)
        assert flag == ""
        assert factors == []

    def test_empty_flags_returns_zero(self):
        ctx = {"home_team": "BOS", "away_team": "MIA", "situational_flags": []}
        score, _, factors = _situational_component(ctx)
        assert score == pytest.approx(0.0)
        assert factors == []

    def test_home_hot_flag_adds_to_score(self):
        ctx = {"home_team": "BOS", "away_team": "MIA",
               "situational_flags": ["HOME_HOT"]}
        score, _, _ = _situational_component(ctx)
        assert score > 0

    def test_away_cold_flag_adds_to_score(self):
        ctx = {"home_team": "BOS", "away_team": "MIA",
               "situational_flags": ["AWAY_COLD"]}
        score, _, _ = _situational_component(ctx)
        assert score > 0

    def test_opposing_flags_cancel_out(self):
        ctx = {"home_team": "BOS", "away_team": "MIA",
               "situational_flags": ["HOME_HOT", "AWAY_HOT"]}
        score, _, _ = _situational_component(ctx)
        assert score == pytest.approx(0.0)

    def test_key_factors_nonempty_when_flags_present(self):
        ctx = {"home_team": "BOS", "away_team": "MIA",
               "situational_flags": ["HOME_HOT"]}
        _, _, factors = _situational_component(ctx)
        assert len(factors) > 0


# ---------------------------------------------------------------------------
# build_sharp_money (empty inputs)
# ---------------------------------------------------------------------------

class TestBuildSharpMoneyEmpty:
    def test_returns_list(self, monkeypatch, tmp_path):
        import scripts.build_sharp_money as mod
        monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
        monkeypatch.setattr(mod, "OUT_JSON", tmp_path / "sharp_money.json")

        result = build_sharp_money()
        assert isinstance(result, list)

    def test_empty_when_no_data(self, monkeypatch, tmp_path):
        import scripts.build_sharp_money as mod
        monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
        monkeypatch.setattr(mod, "OUT_JSON", tmp_path / "sharp_money.json")

        result = build_sharp_money()
        assert result == []
