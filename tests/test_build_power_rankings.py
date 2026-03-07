"""
Tests for scripts/build_power_rankings.py

Covers:
  - All 30 teams ranked 1-30 (no duplicates, no gaps)
  - Required JSON fields present on every entry
  - composite_score in [0, 100]
  - trend is one of up/down/same
  - prev_rank is a valid integer
  - Rank 1 has highest composite_score
  - Graceful handling of empty input
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_power_rankings import (
    CURRENT_SEASON,
    build_power_rankings,
    load_features,
    load_team_names,
)

REQUIRED_FIELDS = {
    "rank",
    "team",
    "team_name",
    "composite_score",
    "net_rating",
    "last10_record",
    "trend",
    "prev_rank",
}

VALID_TRENDS = {"up", "down", "same"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_features(n_games: int = 20, teams: list[str] | None = None) -> pd.DataFrame:
    """Build a minimal features DataFrame suitable for build_power_rankings."""
    if teams is None:
        teams = ["OKC", "BOS", "MIL"]

    rows = []
    for team in teams:
        for i in range(n_games):
            rows.append(
                {
                    "team_abbreviation": team,
                    "game_date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                    "season": CURRENT_SEASON,
                    "net_rtg_game_roll5": 5.0 if (team == "OKC" and i >= n_games - 5) else 0.0,
                    "net_rtg_game_roll10": float(i) * 0.5 if team == "OKC" else float(i) * 0.2,
                    "net_rtg_game_roll20": float(i) * 0.3,
                    "pythagorean_win_pct_roll10": 0.55 if team == "OKC" else 0.50,
                    "win_pct_roll10": 0.60 if team == "OKC" else 0.48,
                    "win": 1 if i % 2 == 0 else 0,
                    "cum_wins": i // 2,
                    "cum_losses": i - i // 2,
                }
            )
    return pd.DataFrame(rows)


def _make_team_names(teams: list[str]) -> dict[str, str]:
    return {t: f"{t} Full Name" for t in teams}


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestBuildPowerRankings:
    def test_returns_list(self):
        features = _make_features()
        names = _make_team_names(["OKC", "BOS", "MIL"])
        result = build_power_rankings(features, names)
        assert isinstance(result, list)

    def test_team_count(self):
        teams = ["OKC", "BOS", "MIL"]
        features = _make_features(teams=teams)
        names = _make_team_names(teams)
        result = build_power_rankings(features, names)
        assert len(result) == len(teams)

    def test_required_fields_present(self):
        features = _make_features()
        names = _make_team_names(["OKC", "BOS", "MIL"])
        result = build_power_rankings(features, names)
        for entry in result:
            missing = REQUIRED_FIELDS - set(entry.keys())
            assert not missing, f"Missing fields: {missing}"

    def test_ranks_are_sequential_no_gaps(self):
        teams = ["OKC", "BOS", "MIL", "LAL", "GSW"]
        features = _make_features(teams=teams)
        names = _make_team_names(teams)
        result = build_power_rankings(features, names)
        ranks = [e["rank"] for e in result]
        assert ranks == list(range(1, len(teams) + 1)), f"Non-sequential ranks: {ranks}"

    def test_no_duplicate_teams(self):
        teams = ["OKC", "BOS", "MIL"]
        features = _make_features(teams=teams)
        names = _make_team_names(teams)
        result = build_power_rankings(features, names)
        team_abbrs = [e["team"] for e in result]
        assert len(team_abbrs) == len(set(team_abbrs)), "Duplicate team abbreviations in output"

    def test_composite_score_in_range(self):
        features = _make_features()
        names = _make_team_names(["OKC", "BOS", "MIL"])
        result = build_power_rankings(features, names)
        for entry in result:
            score = entry["composite_score"]
            assert 0.0 <= score <= 100.0, f"composite_score={score} out of [0,100]"

    def test_trend_values_valid(self):
        features = _make_features()
        names = _make_team_names(["OKC", "BOS", "MIL"])
        result = build_power_rankings(features, names)
        for entry in result:
            assert entry["trend"] in VALID_TRENDS, f"Invalid trend: {entry['trend']}"

    def test_prev_rank_is_int(self):
        features = _make_features()
        names = _make_team_names(["OKC", "BOS", "MIL"])
        result = build_power_rankings(features, names)
        for entry in result:
            assert isinstance(entry["prev_rank"], int), f"prev_rank type: {type(entry['prev_rank'])}"

    def test_best_team_ranked_first(self):
        """OKC is consistently set to higher metrics -- should rank #1."""
        features = _make_features()
        names = _make_team_names(["OKC", "BOS", "MIL"])
        result = build_power_rankings(features, names)
        assert result[0]["team"] == "OKC"

    def test_last10_record_format(self):
        features = _make_features()
        names = _make_team_names(["OKC", "BOS", "MIL"])
        result = build_power_rankings(features, names)
        for entry in result:
            rec = entry["last10_record"]
            parts = rec.split("-")
            assert len(parts) == 2, f"Bad record format: {rec}"
            assert parts[0].isdigit() and parts[1].isdigit()

    def test_empty_features_returns_empty(self):
        empty = pd.DataFrame(
            columns=[
                "team_abbreviation", "game_date", "season",
                "net_rtg_game_roll5", "net_rtg_game_roll10", "net_rtg_game_roll20",
                "pythagorean_win_pct_roll10", "win_pct_roll10",
                "win", "cum_wins", "cum_losses",
            ]
        )
        result = build_power_rankings(empty, {})
        assert result == []

    def test_team_name_populated(self):
        features = _make_features(teams=["OKC"])
        names = {"OKC": "Oklahoma City Thunder"}
        result = build_power_rankings(features, names)
        assert result[0]["team_name"] == "Oklahoma City Thunder"

    def test_team_name_fallback_to_abbr(self):
        """If team not in names dict, should fall back to abbreviation."""
        features = _make_features(teams=["OKC"])
        result = build_power_rankings(features, {})
        assert result[0]["team_name"] == "OKC"


# ---------------------------------------------------------------------------
# Integration smoke test (requires real data files)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestPowerRankingsIntegration:
    def test_loads_real_data_and_ranks_30_teams(self):
        features_path = PROJECT_ROOT / "data" / "features" / "team_game_features.csv"
        teams_path = PROJECT_ROOT / "data" / "processed" / "teams.csv"
        if not features_path.exists() or not teams_path.exists():
            pytest.skip("Real data files not available")

        features = load_features()
        team_names = load_team_names()
        result = build_power_rankings(features, team_names)

        assert len(result) == 30, f"Expected 30 teams, got {len(result)}"
        ranks = [e["rank"] for e in result]
        assert ranks == list(range(1, 31))

        for entry in result:
            missing = REQUIRED_FIELDS - set(entry.keys())
            assert not missing, f"Missing fields on {entry['team']}: {missing}"
            assert 0.0 <= entry["composite_score"] <= 100.0
            assert entry["trend"] in VALID_TRENDS
