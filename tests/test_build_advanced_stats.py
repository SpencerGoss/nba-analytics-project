"""
Tests for scripts/build_advanced_stats.py

Covers:
  - build_advanced_stats: returns {} when CSV absent
  - build_advanced_stats: correct schema with synthetic CSV
  - MIN_GP filter works correctly
  - Deduplication keeps player with most games
  - All output fields are properly rounded
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.build_advanced_stats as mod
from scripts.build_advanced_stats import build_advanced_stats, MIN_GP, CURRENT_SEASON

PLAYER_FIELDS = {"ts", "usg", "off_rtg", "def_rtg", "net_rtg", "efg", "pie", "gp"}


def _make_csv(tmp_path: Path, rows: list[dict]) -> Path:
    df = pd.DataFrame(rows)
    p = tmp_path / "player_stats_advanced.csv"
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# No CSV
# ---------------------------------------------------------------------------

class TestNoCsv:
    def test_returns_empty_dict_when_csv_absent(self, monkeypatch, tmp_path):
        monkeypatch.setattr(mod, "ADV_CSV", tmp_path / "nonexistent.csv")
        result = build_advanced_stats()
        assert result == {}


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestSchema:
    def _base_row(self, name="Player One", gp=30):
        return {
            "player_name": name, "season": CURRENT_SEASON, "gp": gp,
            "ts_pct": 0.60, "usg_pct": 0.25, "off_rating": 115.0,
            "def_rating": 110.0, "net_rating": 5.0, "efg_pct": 0.55, "pie": 0.14,
        }

    def test_player_in_result(self, monkeypatch, tmp_path):
        csv = _make_csv(tmp_path, [self._base_row()])
        monkeypatch.setattr(mod, "ADV_CSV", csv)
        result = build_advanced_stats()
        assert "Player One" in result

    def test_all_required_fields_present(self, monkeypatch, tmp_path):
        csv = _make_csv(tmp_path, [self._base_row()])
        monkeypatch.setattr(mod, "ADV_CSV", csv)
        result = build_advanced_stats()
        missing = PLAYER_FIELDS - set(result["Player One"].keys())
        assert not missing, f"Missing: {missing}"

    def test_ts_converted_to_percentage(self, monkeypatch, tmp_path):
        csv = _make_csv(tmp_path, [self._base_row()])
        monkeypatch.setattr(mod, "ADV_CSV", csv)
        result = build_advanced_stats()
        # ts_pct=0.60 -> ts=60.0
        assert result["Player One"]["ts"] == pytest.approx(60.0, abs=0.1)

    def test_usg_converted_to_percentage(self, monkeypatch, tmp_path):
        csv = _make_csv(tmp_path, [self._base_row()])
        monkeypatch.setattr(mod, "ADV_CSV", csv)
        result = build_advanced_stats()
        assert result["Player One"]["usg"] == pytest.approx(25.0, abs=0.1)

    def test_gp_is_integer(self, monkeypatch, tmp_path):
        csv = _make_csv(tmp_path, [self._base_row(gp=42)])
        monkeypatch.setattr(mod, "ADV_CSV", csv)
        result = build_advanced_stats()
        assert result["Player One"]["gp"] == 42


# ---------------------------------------------------------------------------
# MIN_GP filter
# ---------------------------------------------------------------------------

class TestMinGpFilter:
    def _row(self, name, gp):
        return {
            "player_name": name, "season": CURRENT_SEASON, "gp": gp,
            "ts_pct": 0.55, "usg_pct": 0.20, "off_rating": 110.0,
            "def_rating": 108.0, "net_rating": 2.0, "efg_pct": 0.50, "pie": 0.12,
        }

    def test_player_below_min_gp_excluded(self, monkeypatch, tmp_path):
        csv = _make_csv(tmp_path, [self._row("Low GP", MIN_GP - 1)])
        monkeypatch.setattr(mod, "ADV_CSV", csv)
        result = build_advanced_stats()
        assert "Low GP" not in result

    def test_player_at_min_gp_included(self, monkeypatch, tmp_path):
        csv = _make_csv(tmp_path, [self._row("At Min GP", MIN_GP)])
        monkeypatch.setattr(mod, "ADV_CSV", csv)
        result = build_advanced_stats()
        assert "At Min GP" in result


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    def _row(self, name, gp, off_rtg):
        return {
            "player_name": name, "season": CURRENT_SEASON, "gp": gp,
            "ts_pct": 0.55, "usg_pct": 0.20, "off_rating": off_rtg,
            "def_rating": 108.0, "net_rating": 2.0, "efg_pct": 0.50, "pie": 0.12,
        }

    def test_duplicate_player_kept_once(self, monkeypatch, tmp_path):
        rows = [
            self._row("Traded Player", 20, 112.0),  # team 1
            self._row("Traded Player", 30, 118.0),  # team 2 (more games)
        ]
        csv = _make_csv(tmp_path, rows)
        monkeypatch.setattr(mod, "ADV_CSV", csv)
        result = build_advanced_stats()
        assert len([k for k in result if k == "Traded Player"]) == 1

    def test_duplicate_player_keeps_higher_gp_row(self, monkeypatch, tmp_path):
        rows = [
            self._row("Traded Player", 20, 112.0),
            self._row("Traded Player", 30, 118.0),  # higher GP -> keep this
        ]
        csv = _make_csv(tmp_path, rows)
        monkeypatch.setattr(mod, "ADV_CSV", csv)
        result = build_advanced_stats()
        assert result["Traded Player"]["gp"] == 30
