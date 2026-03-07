"""
Tests for scripts/backfill_outcomes.py.

Uses a temporary SQLite DB and an in-memory DataFrame so no real files are
touched during the test run.
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup — allow importing from scripts/
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backfill_outcomes import (
    backfill_outcomes,
    build_home_game_index,
    fetch_pending_predictions,
    is_past_game,
    load_team_game_logs,
    open_db,
    parse_game_date,
    update_prediction_outcome,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _create_predictions_db(path: Path) -> None:
    """Create game_predictions table and seed test rows."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE game_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            game_date TEXT,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            home_win_prob REAL NOT NULL,
            away_win_prob REAL NOT NULL,
            model_name TEXT,
            model_artifact TEXT,
            decision_threshold REAL,
            feature_count INTEGER,
            actual_home_win INTEGER,
            notes TEXT
        )
        """
    )
    rows = [
        # Past game — home team won (ORL beat DAL on 2026-03-05)
        ("2026-03-06T00:00:00", "2026-03-05", "ORL", "DAL", 0.70, 0.30, None),
        # Past game — away team won (home team lost)
        ("2026-03-06T00:00:01", "2026-03-05", "WAS", "UTA", 0.61, 0.39, None),
        # Future game — should be skipped
        ("2026-03-06T00:00:02", "2099-01-01", "MIA", "BKN", 0.90, 0.10, None),
        # Already has outcome — should not appear in pending query
        ("2026-03-06T00:00:03", "2026-03-05", "HOU", "GSW", 0.60, 0.40, 1),
    ]
    conn.executemany(
        """
        INSERT INTO game_predictions
            (created_at, game_date, home_team, away_team, home_win_prob, away_win_prob, actual_home_win)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()


def _make_logs_df() -> pd.DataFrame:
    """Return a minimal team_game_logs-style DataFrame for testing."""
    rows = [
        # ORL home win vs DAL on 2026-03-05
        {
            "team_abbreviation": "ORL",
            "game_date": "2026-03-05 00:00:00",
            "matchup": "ORL vs. DAL",
            "wl": "W",
            "pts": 120,
            "plus_minus": 8,
        },
        {
            "team_abbreviation": "DAL",
            "game_date": "2026-03-05 00:00:00",
            "matchup": "DAL @ ORL",
            "wl": "L",
            "pts": 112,
            "plus_minus": -8,
        },
        # WAS home loss vs UTA on 2026-03-05 (away team UTA won)
        {
            "team_abbreviation": "WAS",
            "game_date": "2026-03-05 00:00:00",
            "matchup": "WAS vs. UTA",
            "wl": "L",
            "pts": 98,
            "plus_minus": -6,
        },
        {
            "team_abbreviation": "UTA",
            "game_date": "2026-03-05 00:00:00",
            "matchup": "UTA @ WAS",
            "wl": "W",
            "pts": 104,
            "plus_minus": 6,
        },
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests: parse_game_date
# ---------------------------------------------------------------------------

class TestParseGameDate:
    def test_plain_date_string(self):
        result = parse_game_date("2026-03-05")
        assert result == date(2026, 3, 5)

    def test_datetime_string_with_time(self):
        result = parse_game_date("2026-03-05T12:34:56")
        assert result == date(2026, 3, 5)

    def test_none_input(self):
        assert parse_game_date(None) is None  # type: ignore[arg-type]

    def test_invalid_string(self):
        assert parse_game_date("not-a-date") is None


# ---------------------------------------------------------------------------
# Tests: is_past_game
# ---------------------------------------------------------------------------

class TestIsPastGame:
    def test_yesterday_is_past(self):
        assert is_past_game(date(2026, 3, 4), date(2026, 3, 5)) is True

    def test_today_is_not_past(self):
        assert is_past_game(date(2026, 3, 5), date(2026, 3, 5)) is False

    def test_tomorrow_is_not_past(self):
        assert is_past_game(date(2026, 3, 6), date(2026, 3, 5)) is False


# ---------------------------------------------------------------------------
# Tests: build_home_game_index
# ---------------------------------------------------------------------------

class TestBuildHomeGameIndex:
    def test_home_rows_indexed(self):
        df = _make_logs_df()
        df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")
        df["game_date_str"] = df["game_date"].dt.strftime("%Y-%m-%d")
        index = build_home_game_index(df)
        assert ("ORL", "DAL", "2026-03-05") in index
        assert ("WAS", "UTA", "2026-03-05") in index

    def test_away_rows_not_in_index(self):
        df = _make_logs_df()
        df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")
        df["game_date_str"] = df["game_date"].dt.strftime("%Y-%m-%d")
        index = build_home_game_index(df)
        # Away rows should not appear as keys
        assert ("DAL", "ORL", "2026-03-05") not in index

    def test_home_win_entry_correct(self):
        df = _make_logs_df()
        df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")
        df["game_date_str"] = df["game_date"].dt.strftime("%Y-%m-%d")
        index = build_home_game_index(df)
        entry = index[("ORL", "DAL", "2026-03-05")]
        assert entry["home_wl"] == "W"
        assert entry["actual_margin"] == 8

    def test_home_loss_entry_correct(self):
        df = _make_logs_df()
        df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")
        df["game_date_str"] = df["game_date"].dt.strftime("%Y-%m-%d")
        index = build_home_game_index(df)
        entry = index[("WAS", "UTA", "2026-03-05")]
        assert entry["home_wl"] == "L"
        assert entry["actual_margin"] == -6


# ---------------------------------------------------------------------------
# Tests: fetch_pending_predictions
# ---------------------------------------------------------------------------

class TestFetchPendingPredictions:
    def test_only_returns_null_outcome_rows(self, tmp_path):
        db_path = tmp_path / "test.db"
        _create_predictions_db(db_path)
        conn = open_db(db_path)
        pending = fetch_pending_predictions(conn)
        conn.close()
        # Row 4 (HOU vs GSW) already has actual_home_win=1 so only 3 rows pending
        assert len(pending) == 3

    def test_pending_rows_have_expected_teams(self, tmp_path):
        db_path = tmp_path / "test.db"
        _create_predictions_db(db_path)
        conn = open_db(db_path)
        pending = fetch_pending_predictions(conn)
        conn.close()
        teams = {(r["home_team"], r["away_team"]) for r in pending}
        assert ("ORL", "DAL") in teams
        assert ("WAS", "UTA") in teams
        assert ("MIA", "BKN") in teams
        # Already-resolved row must not appear
        assert ("HOU", "GSW") not in teams


# ---------------------------------------------------------------------------
# Tests: update_prediction_outcome
# ---------------------------------------------------------------------------

class TestUpdatePredictionOutcome:
    def test_updates_actual_home_win(self, tmp_path):
        db_path = tmp_path / "test.db"
        _create_predictions_db(db_path)
        conn = open_db(db_path)
        # id=1 is ORL vs DAL
        update_prediction_outcome(conn, 1, 1, 8)
        conn.commit()
        row = conn.execute("SELECT actual_home_win FROM game_predictions WHERE id=1").fetchone()
        conn.close()
        assert row["actual_home_win"] == 1

    def test_updates_away_team_win(self, tmp_path):
        db_path = tmp_path / "test.db"
        _create_predictions_db(db_path)
        conn = open_db(db_path)
        update_prediction_outcome(conn, 2, 0, -6)
        conn.commit()
        row = conn.execute("SELECT actual_home_win FROM game_predictions WHERE id=2").fetchone()
        conn.close()
        assert row["actual_home_win"] == 0


# ---------------------------------------------------------------------------
# Tests: backfill_outcomes (integration)
# ---------------------------------------------------------------------------

class TestBackfillOutcomes:
    def _run(self, tmp_path: Path) -> tuple[dict, sqlite3.Connection]:
        db_path = tmp_path / "test.db"
        _create_predictions_db(db_path)

        # Write logs CSV so backfill_outcomes can load it
        logs_df = _make_logs_df()
        logs_csv = tmp_path / "team_game_logs.csv"
        logs_df.to_csv(logs_csv, index=False)

        summary = backfill_outcomes(
            db_path=db_path,
            logs_path=logs_csv,
            today=date(2026, 3, 6),  # yesterday's games are "past"
        )
        conn = open_db(db_path)
        return summary, conn

    def test_fills_expected_count(self, tmp_path):
        summary, conn = self._run(tmp_path)
        conn.close()
        # 2 past games with log entries (ORL/DAL and WAS/UTA); MIA/BKN is future
        assert summary["filled"] == 2

    def test_future_game_not_filled(self, tmp_path):
        summary, conn = self._run(tmp_path)
        row = conn.execute(
            "SELECT actual_home_win FROM game_predictions WHERE home_team='MIA'"
        ).fetchone()
        conn.close()
        assert row["actual_home_win"] is None
        assert summary["skipped_future"] == 1

    def test_home_win_set_correctly(self, tmp_path):
        _, conn = self._run(tmp_path)
        row = conn.execute(
            "SELECT actual_home_win FROM game_predictions WHERE home_team='ORL'"
        ).fetchone()
        conn.close()
        assert row["actual_home_win"] == 1

    def test_home_loss_set_correctly(self, tmp_path):
        _, conn = self._run(tmp_path)
        row = conn.execute(
            "SELECT actual_home_win FROM game_predictions WHERE home_team='WAS'"
        ).fetchone()
        conn.close()
        assert row["actual_home_win"] == 0

    def test_already_resolved_row_untouched(self, tmp_path):
        _, conn = self._run(tmp_path)
        row = conn.execute(
            "SELECT actual_home_win FROM game_predictions WHERE home_team='HOU'"
        ).fetchone()
        conn.close()
        # Was pre-seeded as 1 and should still be 1 (not re-processed)
        assert row["actual_home_win"] == 1

    def test_no_log_match_counted(self, tmp_path):
        """Prediction for a game not in the CSV gets counted in no_log_match."""
        db_path = tmp_path / "test2.db"
        # Single past prediction with no matching log entry
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE game_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                game_date TEXT,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                home_win_prob REAL NOT NULL,
                away_win_prob REAL NOT NULL,
                model_name TEXT,
                model_artifact TEXT,
                decision_threshold REAL,
                feature_count INTEGER,
                actual_home_win INTEGER,
                notes TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO game_predictions (created_at, game_date, home_team, away_team, home_win_prob, away_win_prob) "
            "VALUES ('2026-03-06T00:00:00', '2026-03-05', 'PHX', 'CHI', 0.70, 0.30)"
        )
        conn.commit()
        conn.close()

        # Empty logs CSV (no PHX vs CHI row)
        empty_logs = pd.DataFrame(columns=_make_logs_df().columns)
        logs_csv = tmp_path / "empty_logs.csv"
        empty_logs.to_csv(logs_csv, index=False)

        summary = backfill_outcomes(
            db_path=db_path,
            logs_path=logs_csv,
            today=date(2026, 3, 6),
        )
        assert summary["no_log_match"] == 1
        assert summary["filled"] == 0

    def test_summary_keys_present(self, tmp_path):
        summary, conn = self._run(tmp_path)
        conn.close()
        expected_keys = {"total_pending", "filled", "still_pending", "skipped_future", "no_log_match"}
        assert expected_keys == set(summary.keys())

    def test_total_pending_reflects_null_rows(self, tmp_path):
        summary, conn = self._run(tmp_path)
        conn.close()
        # 3 rows had NULL actual_home_win in seed data
        assert summary["total_pending"] == 3
