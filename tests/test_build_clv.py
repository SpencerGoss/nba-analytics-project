"""
Tests for scripts/build_clv.py

Covers:
  - _compute_summary: mean_clv, pos_rate, edge_confirmed logic
  - build_clv: graceful handling when DB absent
  - Output JSON required fields
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_clv import _compute_summary, build_clv

REQUIRED_FIELDS = {
    "mean_clv", "pos_rate", "games_tracked",
    "games_with_clv", "edge_confirmed", "last_updated",
}


# ---------------------------------------------------------------------------
# _compute_summary
# ---------------------------------------------------------------------------

class TestComputeSummary:
    def test_empty_rows_returns_zeros(self):
        result = _compute_summary([])
        assert result["games_tracked"] == 0
        assert result["games_with_clv"] == 0
        assert result["mean_clv"] == pytest.approx(0.0)
        assert result["pos_rate"] == pytest.approx(0.0)
        assert result["edge_confirmed"] is False

    def test_all_null_clv_returns_zeros(self):
        rows = [{"clv": None}, {"clv": None}]
        result = _compute_summary(rows)
        assert result["games_tracked"] == 2
        assert result["games_with_clv"] == 0
        assert result["mean_clv"] == pytest.approx(0.0)

    def test_positive_clv_values_compute_mean(self):
        rows = [{"clv": 1.0}, {"clv": 3.0}]
        result = _compute_summary(rows)
        assert result["mean_clv"] == pytest.approx(2.0)
        assert result["games_with_clv"] == 2

    def test_mixed_positive_negative_pos_rate(self):
        rows = [{"clv": 1.0}, {"clv": -1.0}, {"clv": 2.0}, {"clv": -2.0}]
        result = _compute_summary(rows)
        assert result["pos_rate"] == pytest.approx(50.0)

    def test_all_positive_clv_full_pos_rate(self):
        rows = [{"clv": 0.5}, {"clv": 1.5}]
        result = _compute_summary(rows)
        assert result["pos_rate"] == pytest.approx(100.0)

    def test_required_fields_present(self):
        result = _compute_summary([])
        missing = REQUIRED_FIELDS - set(result.keys())
        assert not missing, f"Missing fields: {missing}"

    def test_edge_confirmed_false_when_insufficient_data(self):
        # With 0 closed games, edge_confirmed must be False
        result = _compute_summary([{"clv": None}])
        assert result["edge_confirmed"] is False


# ---------------------------------------------------------------------------
# build_clv (DB absent)
# ---------------------------------------------------------------------------

class TestBuildClvNoDB:
    def test_returns_dict_when_db_absent(self, tmp_path):
        result = build_clv(
            db_path=tmp_path / "nonexistent.db",
            out_path=tmp_path / "clv_summary.json",
        )
        assert isinstance(result, dict)

    def test_output_file_written_when_db_absent(self, tmp_path):
        out = tmp_path / "clv_summary.json"
        build_clv(db_path=tmp_path / "nonexistent.db", out_path=out)
        assert out.exists()

    def test_output_has_required_fields_when_db_absent(self, tmp_path):
        out = tmp_path / "clv_summary.json"
        result = build_clv(db_path=tmp_path / "nonexistent.db", out_path=out)
        missing = REQUIRED_FIELDS - set(result.keys())
        assert not missing, f"Missing fields: {missing}"

    def test_games_tracked_zero_when_db_absent(self, tmp_path):
        result = build_clv(
            db_path=tmp_path / "nonexistent.db",
            out_path=tmp_path / "clv_summary.json",
        )
        assert result["games_tracked"] == 0


# ---------------------------------------------------------------------------
# _compute_summary -- extended edge cases
# ---------------------------------------------------------------------------

class TestComputeSummaryExtended:
    def test_edge_confirmed_true_when_sufficient_positive_clv(self):
        # 10 games with clv > 0 gives pos_rate=100 -> edge_confirmed
        rows = [{"clv": float(i)} for i in range(1, 11)]
        result = _compute_summary(rows)
        assert result["edge_confirmed"] is True

    def test_edge_confirmed_false_when_pos_rate_exactly_50(self):
        # pos_rate = 50.0 is NOT > 50, so edge_confirmed = False
        rows = [{"clv": 1.0}] * 5 + [{"clv": -1.0}] * 5
        result = _compute_summary(rows)
        assert result["pos_rate"] == pytest.approx(50.0)
        assert result["edge_confirmed"] is False

    def test_edge_confirmed_false_when_too_few_games(self):
        # 9 games with high pos_rate still not edge_confirmed (need >= 10)
        rows = [{"clv": 1.0}] * 9
        result = _compute_summary(rows)
        assert result["games_with_clv"] == 9
        assert result["edge_confirmed"] is False

    def test_negative_clv_values_included_in_mean(self):
        rows = [{"clv": -2.0}, {"clv": -4.0}]
        result = _compute_summary(rows)
        assert result["mean_clv"] == pytest.approx(-3.0)

    def test_zero_pos_rate_when_all_negative_clv(self):
        rows = [{"clv": -1.0}, {"clv": -2.0}]
        result = _compute_summary(rows)
        assert result["pos_rate"] == pytest.approx(0.0)

    def test_string_clv_skipped_gracefully(self):
        rows = [{"clv": "bad"}, {"clv": 1.0}]
        result = _compute_summary(rows)
        assert result["games_with_clv"] == 1
        assert result["mean_clv"] == pytest.approx(1.0)

    def test_games_tracked_counts_all_rows_including_null(self):
        rows = [{"clv": None}, {"clv": 1.0}, {"clv": 2.0}]
        result = _compute_summary(rows)
        assert result["games_tracked"] == 3
        assert result["games_with_clv"] == 2


# ---------------------------------------------------------------------------
# build_clv (real SQLite DB)
# ---------------------------------------------------------------------------

def _make_clv_db(tmp_path: Path, rows: list[dict]) -> Path:
    import sqlite3
    db = tmp_path / "predictions_history.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        """CREATE TABLE clv_tracking (
            id INTEGER PRIMARY KEY,
            game_date TEXT,
            home_team TEXT,
            away_team TEXT,
            opening_spread REAL,
            closing_spread REAL,
            clv REAL
        )"""
    )
    for r in rows:
        conn.execute(
            "INSERT INTO clv_tracking (game_date, home_team, away_team, clv) VALUES (?,?,?,?)",
            (r.get("game_date", "2026-01-01"), r.get("home_team", "BOS"),
             r.get("away_team", "NYK"), r.get("clv")),
        )
    conn.commit()
    conn.close()
    return db


class TestBuildClvWithDB:
    def test_games_tracked_matches_rows(self, tmp_path):
        db = _make_clv_db(tmp_path, [{"clv": 1.0}, {"clv": -0.5}])
        result = build_clv(db_path=db, out_path=tmp_path / "clv.json")
        assert result["games_tracked"] == 2

    def test_mean_clv_computed_correctly(self, tmp_path):
        db = _make_clv_db(tmp_path, [{"clv": 2.0}, {"clv": 4.0}])
        result = build_clv(db_path=db, out_path=tmp_path / "clv.json")
        assert result["mean_clv"] == pytest.approx(3.0)

    def test_null_clv_rows_excluded_from_mean(self, tmp_path):
        db = _make_clv_db(tmp_path, [{"clv": None}, {"clv": 4.0}])
        result = build_clv(db_path=db, out_path=tmp_path / "clv.json")
        assert result["games_with_clv"] == 1
        assert result["mean_clv"] == pytest.approx(4.0)

    def test_output_json_written(self, tmp_path):
        db = _make_clv_db(tmp_path, [{"clv": 1.0}])
        out = tmp_path / "clv.json"
        build_clv(db_path=db, out_path=out)
        assert out.exists()
        import json
        data = json.loads(out.read_text())
        assert "mean_clv" in data

    def test_no_clv_table_returns_zeros(self, tmp_path):
        import sqlite3
        db = tmp_path / "empty.db"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE game_predictions (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()
        result = build_clv(db_path=db, out_path=tmp_path / "clv.json")
        assert result["games_tracked"] == 0
