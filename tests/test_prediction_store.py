"""Tests for src/outputs/prediction_store.py."""
import os
import sqlite3
import tempfile
import pytest

from src.outputs.prediction_store import (
    init_store,
    write_game_prediction,
    STORE_PATH,
)


@pytest.fixture
def tmp_db(tmp_path):
    """Return path to a temporary SQLite database."""
    return str(tmp_path / "test_predictions.db")


class TestInitStore:
    def test_creates_database_file(self, tmp_db):
        init_store(tmp_db)
        assert os.path.exists(tmp_db)

    def test_creates_table(self, tmp_db):
        init_store(tmp_db)
        con = sqlite3.connect(tmp_db)
        tables = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        con.close()
        assert ("game_predictions",) in tables

    def test_creates_indexes(self, tmp_db):
        init_store(tmp_db)
        con = sqlite3.connect(tmp_db)
        indexes = {
            row[0]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        con.close()
        assert "idx_game_date" in indexes
        assert "idx_created_at" in indexes
        assert "idx_teams" in indexes

    def test_wal_mode_enabled(self, tmp_db):
        init_store(tmp_db)
        con = sqlite3.connect(tmp_db)
        mode = con.execute("PRAGMA journal_mode;").fetchone()[0]
        con.close()
        assert mode == "wal"

    def test_idempotent(self, tmp_db):
        """Calling init_store twice doesn't raise."""
        init_store(tmp_db)
        init_store(tmp_db)

    def test_creates_parent_directory(self, tmp_path):
        nested = str(tmp_path / "deep" / "nested" / "test.db")
        init_store(nested)
        assert os.path.exists(nested)


class TestWriteGamePrediction:
    @pytest.fixture
    def basic_prediction(self):
        return {
            "game_date": "2026-03-13",
            "home_team": "BOS",
            "away_team": "LAL",
            "home_win_prob": 0.62,
            "away_win_prob": 0.38,
        }

    def test_inserts_and_returns_rowid(self, tmp_db, basic_prediction):
        rowid = write_game_prediction(basic_prediction, tmp_db)
        assert isinstance(rowid, int)
        assert rowid >= 1

    def test_data_persisted(self, tmp_db, basic_prediction):
        write_game_prediction(basic_prediction, tmp_db)
        con = sqlite3.connect(tmp_db)
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT * FROM game_predictions").fetchone()
        con.close()
        assert row is not None
        assert row["game_date"] == "2026-03-13"
        assert row["home_team"] == "BOS"
        assert row["away_team"] == "LAL"

    def test_probabilities_stored_as_floats(self, tmp_db, basic_prediction):
        write_game_prediction(basic_prediction, tmp_db)
        con = sqlite3.connect(tmp_db)
        row = con.execute(
            "SELECT home_win_prob, away_win_prob FROM game_predictions"
        ).fetchone()
        con.close()
        assert abs(row[0] - 0.62) < 1e-6
        assert abs(row[1] - 0.38) < 1e-6

    def test_optional_fields(self, tmp_db):
        pred = {
            "game_date": "2026-03-13",
            "home_team": "BOS",
            "away_team": "LAL",
            "home_win_prob": 0.62,
            "away_win_prob": 0.38,
            "model_name": "game_outcome_v2",
            "model_artifact": "game_outcome_model_calibrated.pkl",
            "decision_threshold": 0.55,
            "feature_count": 352,
        }
        write_game_prediction(pred, tmp_db)
        con = sqlite3.connect(tmp_db)
        row = con.execute(
            "SELECT model_name, model_artifact, decision_threshold, feature_count "
            "FROM game_predictions"
        ).fetchone()
        con.close()
        assert row[0] == "game_outcome_v2"
        assert row[1] == "game_outcome_model_calibrated.pkl"
        assert abs(row[2] - 0.55) < 1e-6
        assert row[3] == 352

    def test_notes_as_string(self, tmp_db, basic_prediction):
        basic_prediction["notes"] = "some note"
        write_game_prediction(basic_prediction, tmp_db)
        con = sqlite3.connect(tmp_db)
        row = con.execute("SELECT notes FROM game_predictions").fetchone()
        con.close()
        assert row[0] == "some note"

    def test_notes_as_dict_serialized(self, tmp_db, basic_prediction):
        basic_prediction["notes"] = {"key": "value", "num": 42}
        write_game_prediction(basic_prediction, tmp_db)
        con = sqlite3.connect(tmp_db)
        row = con.execute("SELECT notes FROM game_predictions").fetchone()
        con.close()
        import json
        parsed = json.loads(row[0])
        assert parsed["key"] == "value"
        assert parsed["num"] == 42

    def test_duplicate_prevention_same_game(self, tmp_db, basic_prediction):
        """Writing the same game_date+home+away twice returns same rowid."""
        rowid1 = write_game_prediction(basic_prediction, tmp_db)
        rowid2 = write_game_prediction(basic_prediction, tmp_db)
        assert rowid1 == rowid2
        # Verify only one row in the table
        con = sqlite3.connect(tmp_db)
        count = con.execute("SELECT COUNT(*) FROM game_predictions").fetchone()[0]
        con.close()
        assert count == 1

    def test_different_games_both_inserted(self, tmp_db):
        pred1 = {
            "game_date": "2026-03-13",
            "home_team": "BOS",
            "away_team": "LAL",
            "home_win_prob": 0.62,
            "away_win_prob": 0.38,
        }
        pred2 = {
            "game_date": "2026-03-13",
            "home_team": "GSW",
            "away_team": "MIA",
            "home_win_prob": 0.55,
            "away_win_prob": 0.45,
        }
        rowid1 = write_game_prediction(pred1, tmp_db)
        rowid2 = write_game_prediction(pred2, tmp_db)
        assert rowid1 != rowid2
        con = sqlite3.connect(tmp_db)
        count = con.execute("SELECT COUNT(*) FROM game_predictions").fetchone()[0]
        con.close()
        assert count == 2

    def test_same_teams_different_date(self, tmp_db):
        pred1 = {
            "game_date": "2026-03-13",
            "home_team": "BOS",
            "away_team": "LAL",
            "home_win_prob": 0.62,
            "away_win_prob": 0.38,
        }
        pred2 = {
            "game_date": "2026-03-14",
            "home_team": "BOS",
            "away_team": "LAL",
            "home_win_prob": 0.60,
            "away_win_prob": 0.40,
        }
        rowid1 = write_game_prediction(pred1, tmp_db)
        rowid2 = write_game_prediction(pred2, tmp_db)
        assert rowid1 != rowid2

    def test_no_game_date_allows_multiple(self, tmp_db):
        """Predictions without game_date should still deduplicate on (NULL, home, away)."""
        pred = {
            "home_team": "BOS",
            "away_team": "LAL",
            "home_win_prob": 0.62,
            "away_win_prob": 0.38,
        }
        rowid1 = write_game_prediction(pred, tmp_db)
        rowid2 = write_game_prediction(pred, tmp_db)
        # NULL = NULL is false in SQL, so both insert (SQLite: NULL != NULL)
        # This tests current behavior -- both rows are inserted
        con = sqlite3.connect(tmp_db)
        count = con.execute("SELECT COUNT(*) FROM game_predictions").fetchone()[0]
        con.close()
        # NULL comparisons in SQL return NULL (not true), so no dedup for NULL dates
        assert count == 2

    def test_created_at_populated(self, tmp_db, basic_prediction):
        write_game_prediction(basic_prediction, tmp_db)
        con = sqlite3.connect(tmp_db)
        row = con.execute("SELECT created_at FROM game_predictions").fetchone()
        con.close()
        assert row[0] is not None
        assert "T" in row[0]  # ISO format has a T separator
