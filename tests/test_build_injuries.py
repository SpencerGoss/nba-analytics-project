"""
Tests for scripts/build_injuries.py

Covers:
  - Required JSON fields present on every game entry
  - impact labels computed correctly (high/medium/low thresholds)
  - spread_impact_note generated correctly
  - Games from picks are represented in output
  - Teams with no injuries produce empty lists
  - Deduplication: same player appears only once per team
  - Graceful handling of empty inputs
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_injuries import (
    CURRENT_SEASON,
    HIGH_IMPACT_PTS,
    MED_IMPACT_PTS,
    _impact_label,
    _spread_note,
    build_injuries,
)

REQUIRED_GAME_FIELDS = {
    "game_date",
    "home_team",
    "away_team",
    "home_injuries",
    "away_injuries",
    "spread_impact_note",
}

REQUIRED_PLAYER_FIELDS = {"player", "status", "season_avg_pts", "impact"}
VALID_IMPACTS = {"high", "medium", "low"}
VALID_STATUSES = {"OUT"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_absences(records: list[dict]) -> pd.DataFrame:
    """Build a minimal absences DataFrame."""
    base = {
        "player_id": 999,
        "player_name": "Test Player",
        "team_id": 1610612737,
        "game_id": 12345,
        "game_date": pd.Timestamp("2026-03-04"),
        "season": CURRENT_SEASON,
        "min_roll5": 30.0,
        "usg_pct": 0.25,
        "was_absent": 1,
    }
    rows = []
    for rec in records:
        row = {**base, **rec}
        rows.append(row)
    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")
    return df


def _make_player_stats(records: list[dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        rows.append(
            {
                "player_id": rec.get("player_id", 999),
                "player_name": rec.get("player_name", "Test Player"),
                "team_abbreviation": rec.get("team_abbreviation", "OKC"),
                "avg_pts": rec.get("avg_pts", 10.0),
            }
        )
    return pd.DataFrame(rows)


def _make_picks(games: list[tuple[str, str]]) -> list[dict]:
    return [
        {
            "game_date": "2026-03-06",
            "home_team": h,
            "away_team": a,
        }
        for h, a in games
    ]


# Team ID -> abbreviation map used in tests
TEAM_ID_MAP = {
    1610612737: "ATL",
    1610612738: "BOS",
    1610612747: "LAL",
}


# ---------------------------------------------------------------------------
# Unit tests: _impact_label
# ---------------------------------------------------------------------------

class TestImpactLabel:
    def test_high_at_threshold(self):
        assert _impact_label(HIGH_IMPACT_PTS) == "high"

    def test_high_above_threshold(self):
        assert _impact_label(HIGH_IMPACT_PTS + 5) == "high"

    def test_medium_at_threshold(self):
        assert _impact_label(MED_IMPACT_PTS) == "medium"

    def test_medium_just_below_high(self):
        assert _impact_label(HIGH_IMPACT_PTS - 0.1) == "medium"

    def test_low_below_medium(self):
        assert _impact_label(MED_IMPACT_PTS - 0.1) == "low"

    def test_low_at_zero(self):
        assert _impact_label(0.0) == "low"


# ---------------------------------------------------------------------------
# Unit tests: _spread_note
# ---------------------------------------------------------------------------

class TestSpreadNote:
    def test_no_injuries(self):
        note = _spread_note([])
        assert "No key injuries" in note

    def test_single_high(self):
        injuries = [{"player": "Star Player", "impact": "high", "season_avg_pts": 25.0}]
        note = _spread_note(injuries)
        assert "Star Player" in note

    def test_two_high_impacts(self):
        injuries = [
            {"player": "A", "impact": "high", "season_avg_pts": 25.0},
            {"player": "B", "impact": "high", "season_avg_pts": 20.0},
        ]
        note = _spread_note(injuries)
        assert "significant" in note.lower() or "2" in note

    def test_medium_only(self):
        injuries = [{"player": "X", "impact": "medium", "season_avg_pts": 10.0}]
        note = _spread_note(injuries)
        assert "medium" in note.lower() or "minor" in note.lower()


# ---------------------------------------------------------------------------
# Unit tests: build_injuries
# ---------------------------------------------------------------------------

class TestBuildInjuries:
    def test_returns_list(self):
        absences = _make_absences([{"player_id": 1, "player_name": "A", "team_id": 1610612737}])
        stats = _make_player_stats([{"player_id": 1, "player_name": "A", "team_abbreviation": "ATL", "avg_pts": 20.0}])
        picks = _make_picks([("ATL", "BOS")])
        result = build_injuries(absences, stats, TEAM_ID_MAP, picks)
        assert isinstance(result, list)

    def test_required_fields_on_game_entries(self):
        absences = _make_absences([{"player_id": 1, "player_name": "A", "team_id": 1610612737}])
        stats = _make_player_stats([{"player_id": 1, "avg_pts": 20.0, "team_abbreviation": "ATL"}])
        picks = _make_picks([("ATL", "BOS")])
        result = build_injuries(absences, stats, TEAM_ID_MAP, picks)
        for game in result:
            missing = REQUIRED_GAME_FIELDS - set(game.keys())
            assert not missing, f"Missing fields: {missing}"

    def test_required_player_fields(self):
        absences = _make_absences([{"player_id": 1, "player_name": "Star", "team_id": 1610612737}])
        stats = _make_player_stats([{"player_id": 1, "player_name": "Star", "avg_pts": 25.0, "team_abbreviation": "ATL"}])
        picks = _make_picks([("ATL", "BOS")])
        result = build_injuries(absences, stats, TEAM_ID_MAP, picks)
        home_inj = result[0]["home_injuries"]
        assert len(home_inj) >= 1
        for p in home_inj:
            missing = REQUIRED_PLAYER_FIELDS - set(p.keys())
            assert not missing, f"Missing player fields: {missing}"

    def test_high_impact_assigned_correctly(self):
        absences = _make_absences([{"player_id": 1, "player_name": "Star", "team_id": 1610612737}])
        stats = _make_player_stats([{"player_id": 1, "avg_pts": 20.0, "team_abbreviation": "ATL"}])
        picks = _make_picks([("ATL", "BOS")])
        result = build_injuries(absences, stats, TEAM_ID_MAP, picks)
        player_entry = result[0]["home_injuries"][0]
        assert player_entry["impact"] == "high"

    def test_low_impact_assigned_correctly(self):
        absences = _make_absences([{"player_id": 2, "player_name": "Bench", "team_id": 1610612737}])
        stats = _make_player_stats([{"player_id": 2, "avg_pts": 3.0, "team_abbreviation": "ATL"}])
        picks = _make_picks([("ATL", "BOS")])
        result = build_injuries(absences, stats, TEAM_ID_MAP, picks)
        player_entry = result[0]["home_injuries"][0]
        assert player_entry["impact"] == "low"

    def test_team_with_no_injuries_has_empty_list(self):
        absences = _make_absences([{"player_id": 1, "player_name": "A", "team_id": 1610612737}])
        stats = _make_player_stats([{"player_id": 1, "avg_pts": 5.0, "team_abbreviation": "ATL"}])
        picks = _make_picks([("ATL", "BOS")])
        result = build_injuries(absences, stats, TEAM_ID_MAP, picks)
        # BOS (team_id 1610612738) has no absences
        game = result[0]
        assert game["away_injuries"] == []

    def test_deduplication_same_player_once(self):
        absences = _make_absences([
            {"player_id": 1, "player_name": "A", "team_id": 1610612737, "game_date": "2026-03-03"},
            {"player_id": 1, "player_name": "A", "team_id": 1610612737, "game_date": "2026-03-04"},
        ])
        stats = _make_player_stats([{"player_id": 1, "avg_pts": 10.0, "team_abbreviation": "ATL"}])
        picks = _make_picks([("ATL", "BOS")])
        result = build_injuries(absences, stats, TEAM_ID_MAP, picks)
        player_ids = [p["player"] for p in result[0]["home_injuries"]]
        assert len(player_ids) == len(set(player_ids)), "Player appears more than once"

    def test_status_is_out(self):
        absences = _make_absences([{"player_id": 1, "team_id": 1610612737}])
        stats = _make_player_stats([{"player_id": 1, "avg_pts": 12.0, "team_abbreviation": "ATL"}])
        picks = _make_picks([("ATL", "BOS")])
        result = build_injuries(absences, stats, TEAM_ID_MAP, picks)
        for p in result[0]["home_injuries"]:
            assert p["status"] in VALID_STATUSES

    def test_empty_absences_returns_games_with_empty_lists(self):
        empty_abs = pd.DataFrame(columns=[
            "player_id", "player_name", "team_id", "game_id",
            "game_date", "season", "min_roll5", "usg_pct", "was_absent",
        ])
        empty_abs["game_date"] = pd.to_datetime(empty_abs["game_date"], format="mixed")
        stats = _make_player_stats([])
        picks = _make_picks([("ATL", "BOS")])
        result = build_injuries(empty_abs, stats, TEAM_ID_MAP, picks)
        assert len(result) == 1
        assert result[0]["home_injuries"] == []
        assert result[0]["away_injuries"] == []

    def test_no_picks_returns_team_entries(self):
        absences = _make_absences([{"player_id": 1, "player_name": "A", "team_id": 1610612737}])
        stats = _make_player_stats([{"player_id": 1, "avg_pts": 10.0, "team_abbreviation": "ATL"}])
        result = build_injuries(absences, stats, TEAM_ID_MAP, picks=[])
        assert len(result) >= 1

    def test_injuries_sorted_by_avg_pts_desc(self):
        absences = _make_absences([
            {"player_id": 1, "player_name": "Bench", "team_id": 1610612737},
            {"player_id": 2, "player_name": "Star", "team_id": 1610612737},
        ])
        stats = _make_player_stats([
            {"player_id": 1, "avg_pts": 5.0, "team_abbreviation": "ATL"},
            {"player_id": 2, "avg_pts": 25.0, "team_abbreviation": "ATL"},
        ])
        picks = _make_picks([("ATL", "BOS")])
        result = build_injuries(absences, stats, TEAM_ID_MAP, picks)
        pts_list = [p["season_avg_pts"] for p in result[0]["home_injuries"]]
        assert pts_list == sorted(pts_list, reverse=True), "Players not sorted by avg_pts desc"

    def test_spread_impact_note_present_and_nonempty(self):
        absences = _make_absences([{"player_id": 1, "team_id": 1610612737}])
        stats = _make_player_stats([{"player_id": 1, "avg_pts": 5.0, "team_abbreviation": "ATL"}])
        picks = _make_picks([("ATL", "BOS")])
        result = build_injuries(absences, stats, TEAM_ID_MAP, picks)
        assert result[0]["spread_impact_note"]


# ---------------------------------------------------------------------------
# Integration smoke test (requires real data files)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestInjuriesIntegration:
    def test_loads_real_data(self):
        abs_path = PROJECT_ROOT / "data" / "processed" / "player_absences.csv"
        stats_path = PROJECT_ROOT / "data" / "processed" / "player_stats.csv"
        teams_path = PROJECT_ROOT / "data" / "processed" / "teams.csv"
        picks_path = PROJECT_ROOT / "dashboard" / "data" / "todays_picks.json"

        for p in [abs_path, stats_path, teams_path]:
            if not p.exists():
                pytest.skip(f"Missing {p}")

        from scripts.build_injuries import (
            load_absences,
            load_player_stats,
            load_team_id_map,
            load_picks,
        )

        absences = load_absences()
        stats = load_player_stats()
        team_id_map = load_team_id_map()
        picks = load_picks()

        result = build_injuries(absences, stats, team_id_map, picks)
        assert isinstance(result, list)
        for game in result:
            missing = REQUIRED_GAME_FIELDS - set(game.keys())
            assert not missing
