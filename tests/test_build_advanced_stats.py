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


# ---------------------------------------------------------------------------
# Rounding
# ---------------------------------------------------------------------------

class TestRounding:
    def _row(self, name="Rounding Player", gp=30, **overrides):
        base = {
            "player_name": name, "season": CURRENT_SEASON, "gp": gp,
            "ts_pct": 0.5555, "usg_pct": 0.2345, "off_rating": 115.123,
            "def_rating": 109.678, "net_rating": 5.445, "efg_pct": 0.5015, "pie": 0.1405,
        }
        base.update(overrides)
        return base

    def test_ts_rounded_to_one_decimal(self, monkeypatch, tmp_path):
        csv = _make_csv(tmp_path, [self._row()])
        monkeypatch.setattr(mod, "ADV_CSV", csv)
        result = build_advanced_stats()
        val = result["Rounding Player"]["ts"]
        assert val == round(val, 1)

    def test_off_rtg_rounded_to_one_decimal(self, monkeypatch, tmp_path):
        csv = _make_csv(tmp_path, [self._row()])
        monkeypatch.setattr(mod, "ADV_CSV", csv)
        result = build_advanced_stats()
        val = result["Rounding Player"]["off_rtg"]
        assert val == round(val, 1)

    def test_net_rtg_rounded_to_one_decimal(self, monkeypatch, tmp_path):
        csv = _make_csv(tmp_path, [self._row()])
        monkeypatch.setattr(mod, "ADV_CSV", csv)
        result = build_advanced_stats()
        val = result["Rounding Player"]["net_rtg"]
        assert val == round(val, 1)


# ---------------------------------------------------------------------------
# Season filter
# ---------------------------------------------------------------------------

class TestEfgConversion:
    def _row(self, name="EFG Player", gp=30, efg_pct=0.55):
        return {
            "player_name": name, "season": CURRENT_SEASON, "gp": gp,
            "ts_pct": 0.60, "usg_pct": 0.25, "off_rating": 115.0,
            "def_rating": 110.0, "net_rating": 5.0, "efg_pct": efg_pct, "pie": 0.14,
        }

    def test_efg_converted_to_percentage(self, monkeypatch, tmp_path):
        """efg_pct=0.55 -> efg=55.0 in output."""
        csv = _make_csv(tmp_path, [self._row(efg_pct=0.55)])
        monkeypatch.setattr(mod, "ADV_CSV", csv)
        result = build_advanced_stats()
        assert result["EFG Player"]["efg"] == pytest.approx(55.0, abs=0.1)

    def test_efg_rounded_to_one_decimal(self, monkeypatch, tmp_path):
        csv = _make_csv(tmp_path, [self._row(efg_pct=0.5015)])
        monkeypatch.setattr(mod, "ADV_CSV", csv)
        result = build_advanced_stats()
        val = result["EFG Player"]["efg"]
        assert val == round(val, 1)


class TestMultiplePlayers:
    def _row(self, name, gp=30):
        return {
            "player_name": name, "season": CURRENT_SEASON, "gp": gp,
            "ts_pct": 0.55, "usg_pct": 0.20, "off_rating": 110.0,
            "def_rating": 108.0, "net_rating": 2.0, "efg_pct": 0.50, "pie": 0.12,
        }

    def test_all_three_players_in_result(self, monkeypatch, tmp_path):
        rows = [self._row("Alpha"), self._row("Beta"), self._row("Gamma")]
        csv = _make_csv(tmp_path, rows)
        monkeypatch.setattr(mod, "ADV_CSV", csv)
        result = build_advanced_stats()
        assert "Alpha" in result
        assert "Beta" in result
        assert "Gamma" in result

    def test_result_values_are_dicts(self, monkeypatch, tmp_path):
        rows = [self._row("PlayerX")]
        csv = _make_csv(tmp_path, rows)
        monkeypatch.setattr(mod, "ADV_CSV", csv)
        result = build_advanced_stats()
        assert isinstance(result["PlayerX"], dict)


class TestSeasonFilter:
    def _row(self, name, season, gp=30):
        return {
            "player_name": name, "season": season, "gp": gp,
            "ts_pct": 0.55, "usg_pct": 0.20, "off_rating": 110.0,
            "def_rating": 108.0, "net_rating": 2.0, "efg_pct": 0.50, "pie": 0.12,
        }

    def test_prior_season_player_excluded(self, monkeypatch, tmp_path):
        """Players from a prior season (not CURRENT_SEASON) must not appear."""
        rows = [
            self._row("Current Season Player", CURRENT_SEASON),
            self._row("Old Season Player", CURRENT_SEASON - 1),
        ]
        csv = _make_csv(tmp_path, rows)
        monkeypatch.setattr(mod, "ADV_CSV", csv)
        result = build_advanced_stats()
        assert "Current Season Player" in result
        assert "Old Season Player" not in result

    def test_only_current_season_returned(self, monkeypatch, tmp_path):
        """All returned players belong to the current season."""
        rows = [
            self._row(f"Player{i}", CURRENT_SEASON, gp=30) for i in range(3)
        ] + [
            self._row(f"OldPlayer{i}", CURRENT_SEASON - 1, gp=40) for i in range(2)
        ]
        csv = _make_csv(tmp_path, rows)
        monkeypatch.setattr(mod, "ADV_CSV", csv)
        result = build_advanced_stats()
        assert all(k.startswith("Player") for k in result)
