"""
Tests for scripts/build_bet_tracker.py

Covers:
  - _result_label: WIN/LOSS/None classification
  - _actual_winner: home/away winner resolution
  - build_bet_tracker: graceful handling when DB is absent (via monkeypatch)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_bet_tracker import _result_label, _actual_winner


# ---------------------------------------------------------------------------
# _result_label
# ---------------------------------------------------------------------------

class TestResultLabel:
    def test_win_when_home_predicted_and_home_won(self):
        assert _result_label(1, "BOS", "BOS") == "WIN"

    def test_loss_when_home_predicted_and_away_won(self):
        assert _result_label(0, "BOS", "BOS") == "LOSS"

    def test_win_when_away_predicted_and_away_won(self):
        assert _result_label(0, "MIA", "BOS") == "WIN"

    def test_loss_when_away_predicted_and_home_won(self):
        assert _result_label(1, "MIA", "BOS") == "LOSS"

    def test_none_when_actual_is_none(self):
        assert _result_label(None, "BOS", "BOS") is None

    def test_none_when_actual_is_nan(self):
        import math
        assert _result_label(float("nan"), "BOS", "BOS") is None

    def test_correct_string_values(self):
        assert _result_label(1, "BOS", "BOS") in ("WIN", "LOSS")
        assert _result_label(0, "BOS", "BOS") in ("WIN", "LOSS")


# ---------------------------------------------------------------------------
# _actual_winner
# ---------------------------------------------------------------------------

class TestActualWinner:
    def test_home_team_when_home_won(self):
        assert _actual_winner(1, "BOS", "MIA") == "BOS"

    def test_away_team_when_away_won(self):
        assert _actual_winner(0, "BOS", "MIA") == "MIA"

    def test_none_when_not_played(self):
        assert _actual_winner(None, "BOS", "MIA") is None

    def test_none_when_nan(self):
        assert _actual_winner(float("nan"), "BOS", "MIA") is None


# ---------------------------------------------------------------------------
# build_bet_tracker (DB absent via monkeypatch)
# ---------------------------------------------------------------------------

class TestBuildBetTrackerNoDB:
    def test_returns_list_when_db_absent(self, monkeypatch, tmp_path):
        import scripts.build_bet_tracker as mod
        monkeypatch.setattr(mod, "DB_PATH", tmp_path / "nonexistent.db")
        monkeypatch.setattr(mod, "OUT_JSON", tmp_path / "bet_tracker.json")

        from scripts.build_bet_tracker import build_bet_tracker
        result = build_bet_tracker()
        assert isinstance(result, list)

    def test_returns_empty_when_db_absent(self, monkeypatch, tmp_path):
        import scripts.build_bet_tracker as mod
        monkeypatch.setattr(mod, "DB_PATH", tmp_path / "nonexistent.db")
        monkeypatch.setattr(mod, "OUT_JSON", tmp_path / "bet_tracker.json")

        from scripts.build_bet_tracker import build_bet_tracker
        result = build_bet_tracker()
        assert result == []
