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
