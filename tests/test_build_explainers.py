"""
tests/test_build_explainers.py
------------------------------
Tests for scripts/build_explainers.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# Ensure project root is on sys.path so scripts.build_explainers imports cleanly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import build_explainers as be


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_pick(**overrides) -> dict:
    base = {
        "game_date": "2026-03-06",
        "home_team": "OKC",
        "away_team": "POR",
        "home_team_name": "Oklahoma City Thunder",
        "away_team_name": "Portland Trail Blazers",
        "home_win_prob": 0.63,
        "away_win_prob": 0.37,
        "predicted_winner": "OKC",
        "ats_pick": "OKC",
        "spread": 5.0,
        "value_bet": True,
        "edge_pct": 0.131,
        "model_name": "gradient_boosting_v2",
    }
    base.update(overrides)
    return base


def _make_features(**overrides) -> pd.Series:
    """Return a minimal feature Series mirroring real CSV column names."""
    base = {
        "game_date": pd.Timestamp("2026-03-04"),
        "home_team": "OKC",
        "away_team": "POR",
        "home_win_pct_roll10": 0.8,       # 8 wins
        "away_win_pct_roll10": 0.3,       # 3 wins
        "home_days_rest": 2.0,
        "away_days_rest": 1.0,
        "home_is_back_to_back": 0,
        "away_is_back_to_back": 0,
        "diff_net_rtg_game_roll10": 6.5,
        "diff_pythagorean_win_pct_roll10": 0.15,
        "home_pythagorean_win_pct_roll10": 0.72,
        "away_pythagorean_win_pct_roll10": 0.57,
    }
    base.update(overrides)
    return pd.Series(base)


def _make_value_bet(**overrides) -> dict:
    base = {
        "game_date": "2026-03-06",
        "home_team": "OKC",
        "away_team": "POR",
        "home_team_name": "Oklahoma City Thunder",
        "away_team_name": "Portland Trail Blazers",
        "model_prob": 0.63,
        "market_prob": 0.4952,
        "edge_pct": 0.1368,
        "recommended_side": "OKC",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Test 1: generate_bullets returns 2-4 bullets
# ---------------------------------------------------------------------------

class TestBulletCount:
    def test_bullet_count_between_2_and_4(self):
        pick = _make_pick()
        features = _make_features()
        vb = _make_value_bet()
        bullets = be.generate_bullets(pick, features, vb, context=None)
        assert 2 <= len(bullets) <= 4, f"Expected 2-4 bullets, got {len(bullets)}: {bullets}"

    def test_fallback_bullet_when_no_conditions_met(self):
        """With neutral features no rule fires -> fallback bullet added."""
        pick = _make_pick(home_win_prob=0.51, away_win_prob=0.49, edge_pct=None)
        features = _make_features(
            home_win_pct_roll10=0.5,
            away_win_pct_roll10=0.5,
            home_days_rest=2.0,
            away_days_rest=2.0,
            diff_net_rtg_game_roll10=1.0,
            diff_pythagorean_win_pct_roll10=0.02,
        )
        bullets = be.generate_bullets(pick, features, value_bet=None, context=None)
        assert len(bullets) >= 1
        assert any("Model projects" in b for b in bullets), bullets

    def test_no_features_still_produces_bullets(self):
        """Script must not crash when features row is None."""
        pick = _make_pick()
        bullets = be.generate_bullets(pick, features=None, value_bet=None, context=None)
        assert len(bullets) >= 1


# ---------------------------------------------------------------------------
# Test 2: Away B2B bullet
# ---------------------------------------------------------------------------

class TestB2BBullet:
    def test_b2b_from_features_column(self):
        pick = _make_pick()
        features = _make_features(away_is_back_to_back=1)
        bullets = be.generate_bullets(pick, features, value_bet=None, context=None)
        b2b_bullets = [b for b in bullets if "2nd game in 2 nights" in b]
        assert len(b2b_bullets) == 1

    def test_b2b_from_context(self):
        pick = _make_pick()
        features = _make_features(away_is_back_to_back=0)
        context = {"home_team": "OKC", "away_team": "POR", "away_is_back_to_back": True}
        bullets = be.generate_bullets(pick, features, value_bet=None, context=context)
        b2b_bullets = [b for b in bullets if "2nd game in 2 nights" in b]
        assert len(b2b_bullets) == 1

    def test_no_b2b_bullet_when_neither_b2b(self):
        pick = _make_pick()
        features = _make_features(away_is_back_to_back=0, home_is_back_to_back=0)
        bullets = be.generate_bullets(pick, features, value_bet=None, context=None)
        b2b_bullets = [b for b in bullets if "2nd game in 2 nights" in b]
        assert len(b2b_bullets) == 0


# ---------------------------------------------------------------------------
# Test 3: Hot/cold streak bullets
# ---------------------------------------------------------------------------

class TestStreakBullets:
    def test_hot_home_team_bullet(self):
        pick = _make_pick(home_win_prob=0.55)  # below HIGH_WIN_PROB, so rule 2 skipped
        features = _make_features(
            home_win_pct_roll10=0.8,     # 8 wins
            away_win_pct_roll10=0.5,
            diff_net_rtg_game_roll10=1.0,
            diff_pythagorean_win_pct_roll10=0.02,
        )
        bullets = be.generate_bullets(pick, features, value_bet=None, context=None)
        hot_bullets = [b for b in bullets if "last 10 games" in b and "Oklahoma" in b]
        assert len(hot_bullets) == 1

    def test_struggling_away_team_bullet(self):
        pick = _make_pick(home_win_prob=0.55, edge_pct=None)
        features = _make_features(
            home_win_pct_roll10=0.5,
            away_win_pct_roll10=0.2,     # 2 wins -- below threshold
            diff_net_rtg_game_roll10=1.0,
            diff_pythagorean_win_pct_roll10=0.02,
        )
        bullets = be.generate_bullets(pick, features, value_bet=None, context=None)
        struggle_bullets = [b for b in bullets if "struggled recently" in b]
        assert len(struggle_bullets) == 1


# ---------------------------------------------------------------------------
# Test 4: High edge bullet
# ---------------------------------------------------------------------------

class TestEdgeBullet:
    def test_high_edge_triggers_bullet(self):
        pick = _make_pick(home_win_prob=0.55, edge_pct=0.15)
        features = _make_features(
            home_win_pct_roll10=0.5,
            away_win_pct_roll10=0.5,
            home_days_rest=2.0,
            away_days_rest=2.0,
            diff_net_rtg_game_roll10=1.0,
            diff_pythagorean_win_pct_roll10=0.02,
        )
        bullets = be.generate_bullets(pick, features, value_bet=None, context=None)
        edge_bullets = [b for b in bullets if "Model edge" in b]
        assert len(edge_bullets) == 1
        assert "15.0%" in edge_bullets[0]

    def test_low_edge_no_bullet(self):
        pick = _make_pick(home_win_prob=0.55, edge_pct=0.05)
        features = _make_features(
            home_win_pct_roll10=0.5,
            away_win_pct_roll10=0.5,
            home_days_rest=2.0,
            away_days_rest=2.0,
            diff_net_rtg_game_roll10=1.0,
            diff_pythagorean_win_pct_roll10=0.02,
        )
        bullets = be.generate_bullets(pick, features, value_bet=None, context=None)
        edge_bullets = [b for b in bullets if "Model edge" in b]
        assert len(edge_bullets) == 0


# ---------------------------------------------------------------------------
# Test 5: build_explainers produces correct JSON structure
# ---------------------------------------------------------------------------

class TestBuildExplainers:
    def _make_minimal_matchup_df(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "game_date": pd.Timestamp("2026-03-04"),
            "home_team": "OKC",
            "away_team": "POR",
            "home_win_pct_roll10": 0.8,
            "away_win_pct_roll10": 0.3,
            "home_days_rest": 2.0,
            "away_days_rest": 1.0,
            "home_is_back_to_back": 0,
            "away_is_back_to_back": 0,
            "diff_net_rtg_game_roll10": 6.5,
            "diff_pythagorean_win_pct_roll10": 0.15,
            "home_pythagorean_win_pct_roll10": 0.72,
            "away_pythagorean_win_pct_roll10": 0.57,
        }])

    def test_output_has_required_keys(self):
        picks = [_make_pick()]
        vb = [_make_value_bet()]
        df = self._make_minimal_matchup_df()
        result = be.build_explainers(picks, vb, df, game_context=[])
        assert len(result) == 1
        rec = result[0]
        for key in ("home_team", "away_team", "game_date", "bullets", "recommended_side", "one_liner"):
            assert key in rec, f"Missing key: {key}"

    def test_bullets_are_strings(self):
        picks = [_make_pick()]
        vb = [_make_value_bet()]
        df = self._make_minimal_matchup_df()
        result = be.build_explainers(picks, vb, df, game_context=[])
        for bullet in result[0]["bullets"]:
            assert isinstance(bullet, str)
            assert len(bullet) > 10

    def test_multiple_games_all_processed(self):
        picks = [
            _make_pick(),
            _make_pick(home_team="BOS", away_team="NYK",
                       home_team_name="Boston Celtics", away_team_name="New York Knicks",
                       home_win_prob=0.73, away_win_prob=0.27),
        ]
        vb: list = []
        df = pd.DataFrame([
            {
                "game_date": pd.Timestamp("2026-03-04"),
                "home_team": "OKC", "away_team": "POR",
                "home_win_pct_roll10": 0.8, "away_win_pct_roll10": 0.3,
                "home_days_rest": 2.0, "away_days_rest": 1.0,
                "home_is_back_to_back": 0, "away_is_back_to_back": 0,
                "diff_net_rtg_game_roll10": 6.5,
                "diff_pythagorean_win_pct_roll10": 0.15,
                "home_pythagorean_win_pct_roll10": 0.72,
                "away_pythagorean_win_pct_roll10": 0.57,
            },
            {
                "game_date": pd.Timestamp("2026-03-03"),
                "home_team": "BOS", "away_team": "NYK",
                "home_win_pct_roll10": 0.9, "away_win_pct_roll10": 0.7,
                "home_days_rest": 3.0, "away_days_rest": 1.0,
                "home_is_back_to_back": 0, "away_is_back_to_back": 0,
                "diff_net_rtg_game_roll10": 7.2,
                "diff_pythagorean_win_pct_roll10": 0.20,
                "home_pythagorean_win_pct_roll10": 0.80,
                "away_pythagorean_win_pct_roll10": 0.60,
            },
        ])
        result = be.build_explainers(picks, vb, df, game_context=[])
        assert len(result) == 2
        teams = {(r["home_team"], r["away_team"]) for r in result}
        assert ("OKC", "POR") in teams
        assert ("BOS", "NYK") in teams

    def test_recommended_side_is_valid_team(self):
        picks = [_make_pick()]
        vb = [_make_value_bet()]
        df = self._make_minimal_matchup_df()
        result = be.build_explainers(picks, vb, df, game_context=[])
        rec = result[0]
        assert rec["recommended_side"] in (rec["home_team"], rec["away_team"])

    def test_missing_game_context_file_handled(self):
        """load_game_context must return [] for a nonexistent path."""
        fake_path = Path("/nonexistent/game_context.json")
        ctx = be.load_game_context(fake_path)
        assert ctx == []

    def test_no_unicode_arrow_in_bullets(self):
        """Ensure no Unicode right-arrow in output (Windows cp1252 safety)."""
        picks = [_make_pick()]
        vb = [_make_value_bet()]
        df = self._make_minimal_matchup_df()
        result = be.build_explainers(picks, vb, df, game_context=[])
        for rec in result:
            for bullet in rec["bullets"]:
                assert "\u2192" not in bullet, f"Unicode arrow found in: {bullet}"
            assert "\u2192" not in rec["one_liner"]


# ---------------------------------------------------------------------------
# Test 6: one_liner generator
# ---------------------------------------------------------------------------

class TestOneLiner:
    def test_one_liner_mentions_recommended_side(self):
        pick = _make_pick()
        vb = _make_value_bet(recommended_side="OKC")
        one_liner = be.generate_one_liner(pick, recommended_side="OKC", value_bet=vb)
        assert "Oklahoma City Thunder" in one_liner or "OKC" in one_liner

    def test_one_liner_fallback_no_value_bet(self):
        pick = _make_pick(spread=None)
        one_liner = be.generate_one_liner(pick, recommended_side="OKC", value_bet=None)
        assert isinstance(one_liner, str)
        assert len(one_liner) > 10


# ---------------------------------------------------------------------------
# Test 7: build_matchup_index uses most-recent row per team pair
# ---------------------------------------------------------------------------

class TestMatchupIndex:
    def test_most_recent_row_selected(self):
        df = pd.DataFrame([
            {"game_date": pd.Timestamp("2025-11-01"), "home_team": "OKC", "away_team": "POR",
             "home_win_pct_roll10": 0.5},
            {"game_date": pd.Timestamp("2026-03-04"), "home_team": "OKC", "away_team": "POR",
             "home_win_pct_roll10": 0.8},
        ])
        index = be.build_matchup_index(df)
        assert ("OKC", "POR") in index
        assert index[("OKC", "POR")]["home_win_pct_roll10"] == 0.8

    def test_index_keys_are_tuples(self):
        df = pd.DataFrame([
            {"game_date": pd.Timestamp("2026-03-04"), "home_team": "BOS", "away_team": "NYK",
             "home_win_pct_roll10": 0.7},
        ])
        index = be.build_matchup_index(df)
        assert all(isinstance(k, tuple) for k in index.keys())
