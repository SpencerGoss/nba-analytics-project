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


# ---------------------------------------------------------------------------
# Helpers for integration tests
# ---------------------------------------------------------------------------

def _make_predictions_db(tmp_path: Path, rows: list[dict]) -> Path:
    import sqlite3
    db = tmp_path / "predictions_history.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        """CREATE TABLE game_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL DEFAULT '2026-03-06T00:00:00',
            game_date TEXT,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            home_win_prob REAL NOT NULL,
            away_win_prob REAL NOT NULL,
            actual_home_win INTEGER,
            notes TEXT
        )"""
    )
    for r in rows:
        conn.execute(
            """INSERT INTO game_predictions
               (created_at, game_date, home_team, away_team, home_win_prob, away_win_prob, actual_home_win)
               VALUES (?,?,?,?,?,?,?)""",
            (
                r.get("created_at", "2026-03-06T00:00:00"),
                r.get("game_date", "2026-03-06"),
                r["home_team"],
                r["away_team"],
                r["home_win_prob"],
                r["away_win_prob"],
                r.get("actual_home_win"),
            ),
        )
    conn.commit()
    conn.close()
    return db


SAMPLE_ROWS = [
    {"game_date": "2026-03-06", "home_team": "BOS", "away_team": "NYK",
     "home_win_prob": 0.72, "away_win_prob": 0.28, "actual_home_win": 1},
    {"game_date": "2026-03-06", "home_team": "LAL", "away_team": "GSW",
     "home_win_prob": 0.40, "away_win_prob": 0.60, "actual_home_win": 0},
]


# ---------------------------------------------------------------------------
# build_bet_tracker (integration with real DB)
# ---------------------------------------------------------------------------

class TestBuildBetTrackerWithDB:
    def test_returns_correct_count(self, monkeypatch, tmp_path):
        db = _make_predictions_db(tmp_path, SAMPLE_ROWS)
        import scripts.build_bet_tracker as mod
        monkeypatch.setattr(mod, "DB_PATH", db)
        monkeypatch.setattr(mod, "OUT_JSON", tmp_path / "bet_tracker.json")
        monkeypatch.setattr(mod, "VALUE_BETS_JSON", tmp_path / "value_bets.json")

        from scripts.build_bet_tracker import build_bet_tracker
        result = build_bet_tracker()
        assert len(result) == 2

    def test_schema_keys_present(self, monkeypatch, tmp_path):
        db = _make_predictions_db(tmp_path, SAMPLE_ROWS[:1])
        import scripts.build_bet_tracker as mod
        monkeypatch.setattr(mod, "DB_PATH", db)
        monkeypatch.setattr(mod, "OUT_JSON", tmp_path / "bet_tracker.json")
        monkeypatch.setattr(mod, "VALUE_BETS_JSON", tmp_path / "value_bets.json")

        from scripts.build_bet_tracker import build_bet_tracker
        result = build_bet_tracker()
        assert len(result) == 1
        required = {
            "game_date", "home_team", "away_team", "predicted_winner",
            "home_win_prob", "away_win_prob", "ats_pick", "spread",
            "value_bet", "actual_winner", "result", "created_at",
        }
        assert required.issubset(result[0].keys())

    def test_predicted_winner_home_when_home_favoured(self, monkeypatch, tmp_path):
        db = _make_predictions_db(tmp_path, SAMPLE_ROWS[:1])
        import scripts.build_bet_tracker as mod
        monkeypatch.setattr(mod, "DB_PATH", db)
        monkeypatch.setattr(mod, "OUT_JSON", tmp_path / "bt.json")
        monkeypatch.setattr(mod, "VALUE_BETS_JSON", tmp_path / "vb.json")

        from scripts.build_bet_tracker import build_bet_tracker
        result = build_bet_tracker()
        assert result[0]["predicted_winner"] == "BOS"

    def test_result_win_when_correct_prediction(self, monkeypatch, tmp_path):
        # BOS home_win_prob=0.72, actual_home_win=1 -> WIN
        db = _make_predictions_db(tmp_path, SAMPLE_ROWS[:1])
        import scripts.build_bet_tracker as mod
        monkeypatch.setattr(mod, "DB_PATH", db)
        monkeypatch.setattr(mod, "OUT_JSON", tmp_path / "bt.json")
        monkeypatch.setattr(mod, "VALUE_BETS_JSON", tmp_path / "vb.json")

        from scripts.build_bet_tracker import build_bet_tracker
        result = build_bet_tracker()
        assert result[0]["result"] == "WIN"

    def test_result_win_away_correct_prediction(self, monkeypatch, tmp_path):
        # LAL home_win_prob=0.40 -> predicted GSW; actual_home_win=0 (away won) -> WIN
        db = _make_predictions_db(tmp_path, SAMPLE_ROWS[1:])
        import scripts.build_bet_tracker as mod
        monkeypatch.setattr(mod, "DB_PATH", db)
        monkeypatch.setattr(mod, "OUT_JSON", tmp_path / "bt.json")
        monkeypatch.setattr(mod, "VALUE_BETS_JSON", tmp_path / "vb.json")

        from scripts.build_bet_tracker import build_bet_tracker
        result = build_bet_tracker()
        assert result[0]["result"] == "WIN"

    def test_result_none_when_not_played(self, monkeypatch, tmp_path):
        rows = [{"game_date": "2026-03-07", "home_team": "MIA", "away_team": "ORL",
                 "home_win_prob": 0.55, "away_win_prob": 0.45}]
        db = _make_predictions_db(tmp_path, rows)
        import scripts.build_bet_tracker as mod
        monkeypatch.setattr(mod, "DB_PATH", db)
        monkeypatch.setattr(mod, "OUT_JSON", tmp_path / "bt.json")
        monkeypatch.setattr(mod, "VALUE_BETS_JSON", tmp_path / "vb.json")

        from scripts.build_bet_tracker import build_bet_tracker
        result = build_bet_tracker()
        assert result[0]["result"] is None

    def test_deduplication_keeps_latest(self, monkeypatch, tmp_path):
        # Two rows for same game with different probs -- dedup keeps first (highest id)
        rows = [
            {"game_date": "2026-03-06", "home_team": "BOS", "away_team": "NYK",
             "home_win_prob": 0.60, "away_win_prob": 0.40, "created_at": "2026-03-06T01:00:00"},
            {"game_date": "2026-03-06", "home_team": "BOS", "away_team": "NYK",
             "home_win_prob": 0.70, "away_win_prob": 0.30, "created_at": "2026-03-06T02:00:00"},
        ]
        db = _make_predictions_db(tmp_path, rows)
        import scripts.build_bet_tracker as mod
        monkeypatch.setattr(mod, "DB_PATH", db)
        monkeypatch.setattr(mod, "OUT_JSON", tmp_path / "bt.json")
        monkeypatch.setattr(mod, "VALUE_BETS_JSON", tmp_path / "vb.json")

        from scripts.build_bet_tracker import build_bet_tracker
        result = build_bet_tracker()
        # Should have exactly one record for BOS vs NYK
        bos_rows = [r for r in result if r["home_team"] == "BOS"]
        assert len(bos_rows) == 1
