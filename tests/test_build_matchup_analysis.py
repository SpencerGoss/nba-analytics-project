"""
Tests for scripts/build_matchup_analysis.py
"""

from __future__ import annotations

import math

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_features(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")
    return df


SAMPLE_FEATURES = _make_features([
    {
        "team_abbreviation": "OKC", "game_date": "2025-03-01", "season": 202526,
        "off_rtg_game_roll10": 120.0, "def_rtg_game_roll10": 105.0,
        "net_rtg_game_roll10": 15.0, "pace_game_roll10": 100.0,
        "fg3_pct_roll10": 0.40, "win_pct_roll10": 0.80,
    },
    {
        "team_abbreviation": "POR", "game_date": "2025-03-01", "season": 202526,
        "off_rtg_game_roll10": 108.0, "def_rtg_game_roll10": 115.0,
        "net_rtg_game_roll10": -7.0, "pace_game_roll10": 98.0,
        "fg3_pct_roll10": 0.35, "win_pct_roll10": 0.30,
    },
])

SAMPLE_PICKS = [
    {"home_team": "OKC", "away_team": "POR"},
    {"home_team": "LAL", "away_team": "BOS"},  # no feature data
]


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

def test_import():
    from scripts.build_matchup_analysis import (
        compute_matchup_analysis,
        _normalize_0_100,
        _get_latest_metrics,
    )
    assert callable(compute_matchup_analysis)


# ---------------------------------------------------------------------------
# _normalize_0_100
# ---------------------------------------------------------------------------

def test_normalize_equal_values():
    from scripts.build_matchup_analysis import _normalize_0_100
    h, a = _normalize_0_100(100.0, 100.0, higher_is_better=True)
    assert h == 50
    assert a == 50


def test_normalize_higher_is_better_home_wins():
    from scripts.build_matchup_analysis import _normalize_0_100
    h, a = _normalize_0_100(120.0, 108.0, higher_is_better=True)
    assert h > a


def test_normalize_lower_is_better_away_wins():
    """Lower DRtg is better; team with lower value should score higher."""
    from scripts.build_matchup_analysis import _normalize_0_100
    # OKC drtg=105 (better), POR drtg=115 (worse) -> home should get higher score
    h, a = _normalize_0_100(105.0, 115.0, higher_is_better=False)
    assert h > a


def test_normalize_both_none():
    from scripts.build_matchup_analysis import _normalize_0_100
    h, a = _normalize_0_100(None, None)
    assert h == 50
    assert a == 50


def test_normalize_home_none():
    from scripts.build_matchup_analysis import _normalize_0_100
    h, a = _normalize_0_100(None, 100.0, higher_is_better=True)
    assert h < a


def test_normalize_away_none():
    from scripts.build_matchup_analysis import _normalize_0_100
    h, a = _normalize_0_100(100.0, None, higher_is_better=True)
    assert h > a


def test_normalize_scores_in_0_100():
    from scripts.build_matchup_analysis import _normalize_0_100
    h, a = _normalize_0_100(110.0, 90.0, higher_is_better=True)
    assert 0 <= h <= 100
    assert 0 <= a <= 100


# ---------------------------------------------------------------------------
# _get_latest_metrics
# ---------------------------------------------------------------------------

def test_get_latest_metrics_known_team():
    from scripts.build_matchup_analysis import _get_latest_metrics
    metrics = _get_latest_metrics(SAMPLE_FEATURES, "OKC")
    assert metrics["ortg"] == pytest.approx(120.0)
    assert metrics["drtg"] == pytest.approx(105.0)
    assert metrics["net_rtg"] == pytest.approx(15.0)
    assert metrics["pace"] == pytest.approx(100.0)
    assert metrics["fg3_rate"] == pytest.approx(0.40)
    assert metrics["recent_form"] == pytest.approx(0.80)


def test_get_latest_metrics_unknown_team():
    from scripts.build_matchup_analysis import _get_latest_metrics
    metrics = _get_latest_metrics(SAMPLE_FEATURES, "UNKN")
    # All values should be None for unknown team
    for val in metrics.values():
        assert val is None


# ---------------------------------------------------------------------------
# compute_matchup_analysis
# ---------------------------------------------------------------------------

def test_compute_matchup_analysis_length():
    from scripts.build_matchup_analysis import compute_matchup_analysis
    results = compute_matchup_analysis(SAMPLE_PICKS, SAMPLE_FEATURES)
    assert len(results) == 2


def test_compute_matchup_analysis_keys():
    from scripts.build_matchup_analysis import compute_matchup_analysis
    results = compute_matchup_analysis(SAMPLE_PICKS, SAMPLE_FEATURES)
    item = results[0]
    expected_keys = {
        "home_team", "away_team", "dimensions", "radar",
        "home_ortg", "home_drtg", "away_ortg", "away_drtg",
        "home_pace", "away_pace", "pace_diff",
        "home_fg3_rate", "away_fg3_rate",
        "home_net_rtg", "away_net_rtg", "net_rtg_diff",
        "home_recent_form", "away_recent_form",
    }
    assert expected_keys == set(item.keys())


def test_compute_matchup_analysis_radar_shape():
    from scripts.build_matchup_analysis import compute_matchup_analysis, DIMENSIONS
    results = compute_matchup_analysis(SAMPLE_PICKS, SAMPLE_FEATURES)
    item = results[0]
    assert "OKC" in item["radar"]
    assert "POR" in item["radar"]
    assert len(item["radar"]["OKC"]) == len(DIMENSIONS)
    assert len(item["radar"]["POR"]) == len(DIMENSIONS)


def test_compute_matchup_analysis_okc_better_offense():
    """OKC has higher ortg, so its Offense radar score should be higher."""
    from scripts.build_matchup_analysis import compute_matchup_analysis, DIMENSIONS
    results = compute_matchup_analysis(SAMPLE_PICKS, SAMPLE_FEATURES)
    item = results[0]
    off_idx = DIMENSIONS.index("Offense")
    assert item["radar"]["OKC"][off_idx] > item["radar"]["POR"][off_idx]


def test_compute_matchup_analysis_okc_better_defense():
    """OKC has lower drtg (105 vs 115), so its Defense radar score should be higher."""
    from scripts.build_matchup_analysis import compute_matchup_analysis, DIMENSIONS
    results = compute_matchup_analysis(SAMPLE_PICKS, SAMPLE_FEATURES)
    item = results[0]
    def_idx = DIMENSIONS.index("Defense")
    assert item["radar"]["OKC"][def_idx] > item["radar"]["POR"][def_idx]


def test_compute_matchup_analysis_pace_diff():
    from scripts.build_matchup_analysis import compute_matchup_analysis
    results = compute_matchup_analysis(SAMPLE_PICKS, SAMPLE_FEATURES)
    item = results[0]
    # OKC pace=100, POR pace=98 -> diff = +2.0
    assert item["pace_diff"] == pytest.approx(2.0)


def test_compute_matchup_analysis_net_rtg_diff():
    from scripts.build_matchup_analysis import compute_matchup_analysis
    results = compute_matchup_analysis(SAMPLE_PICKS, SAMPLE_FEATURES)
    item = results[0]
    # OKC net=15, POR net=-7 -> diff = 22.0
    assert item["net_rtg_diff"] == pytest.approx(22.0)


def test_compute_matchup_analysis_no_data_team():
    """LAL vs BOS with no feature data should still produce output with Nones."""
    from scripts.build_matchup_analysis import compute_matchup_analysis
    results = compute_matchup_analysis(SAMPLE_PICKS, SAMPLE_FEATURES)
    item = results[1]
    assert item["home_team"] == "LAL"
    assert item["away_team"] == "BOS"
    assert item["home_ortg"] is None
    assert item["away_ortg"] is None
    assert item["pace_diff"] is None
    assert item["net_rtg_diff"] is None


def test_compute_matchup_analysis_radar_0_100_bounds():
    """All radar scores should be within 0-100."""
    from scripts.build_matchup_analysis import compute_matchup_analysis
    results = compute_matchup_analysis(SAMPLE_PICKS, SAMPLE_FEATURES)
    for item in results:
        for team_scores in item["radar"].values():
            for score in team_scores:
                assert 0 <= score <= 100, f"Score {score} out of 0-100 range"


def test_compute_matchup_analysis_dimensions_field():
    from scripts.build_matchup_analysis import compute_matchup_analysis, DIMENSIONS
    results = compute_matchup_analysis(SAMPLE_PICKS, SAMPLE_FEATURES)
    for item in results:
        assert item["dimensions"] == DIMENSIONS
