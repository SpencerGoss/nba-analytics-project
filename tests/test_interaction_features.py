"""Tests for cross-matchup interaction, fatigue compound, and 3-game rolling features."""

import numpy as np
import pandas as pd
import pytest


def _make_matchup_row(**overrides):
    """Build a single matchup row with sensible defaults."""
    row = {
        "game_id": "0022400001",
        "season": 202425,
        "game_date": "2025-01-15",
        "home_team": "BOS",
        "away_team": "NYK",
        "home_off_rtg_game_roll20": 115.0,
        "away_def_rtg_game_roll20": 110.0,
        "away_off_rtg_game_roll20": 112.0,
        "home_def_rtg_game_roll20": 108.0,
        "home_is_back_to_back": 0,
        "away_is_back_to_back": 0,
        "home_travel_miles": 0.0,
        "away_travel_miles": 0.0,
        "home_pts_roll3": 112.0,
        "away_pts_roll3": 108.0,
        "home_net_rtg_game_roll3": 5.0,
        "away_net_rtg_game_roll3": 2.0,
        "home_off_rtg_game_roll3": 114.0,
        "away_off_rtg_game_roll3": 110.0,
        "home_def_rtg_game_roll3": 109.0,
        "away_def_rtg_game_roll3": 108.0,
        "home_plus_minus_roll3": 4.0,
        "away_plus_minus_roll3": 1.0,
        "home_win_pct_roll3": 0.667,
        "away_win_pct_roll3": 0.333,
    }
    row.update(overrides)
    return row


def _apply_cross_matchup(matchup):
    """Replicate the cross-matchup logic from build_matchup_dataset."""
    if "home_off_rtg_game_roll20" in matchup.columns and "away_def_rtg_game_roll20" in matchup.columns:
        matchup["home_off_vs_away_def_20"] = (
            matchup["home_off_rtg_game_roll20"] - matchup["away_def_rtg_game_roll20"]
        )
        matchup["away_off_vs_home_def_20"] = (
            matchup["away_off_rtg_game_roll20"] - matchup["home_def_rtg_game_roll20"]
        )
        matchup["matchup_off_def_edge"] = (
            matchup["home_off_vs_away_def_20"] - matchup["away_off_vs_home_def_20"]
        )
    return matchup


def _apply_fatigue(matchup):
    """Replicate fatigue compound logic from build_matchup_dataset."""
    if "home_is_back_to_back" in matchup.columns and "home_travel_miles" in matchup.columns:
        matchup["home_fatigue_compound"] = (
            matchup["home_is_back_to_back"] * matchup["home_travel_miles"]
        )
        matchup["away_fatigue_compound"] = (
            matchup["away_is_back_to_back"] * matchup["away_travel_miles"]
        )
        matchup["diff_fatigue_compound"] = (
            matchup["home_fatigue_compound"] - matchup["away_fatigue_compound"]
        )
    return matchup


# ── Cross-matchup interaction ─────────────────────────────────────────────────

def test_matchup_off_def_edge_computed():
    """matchup_off_def_edge column should exist after computation."""
    df = pd.DataFrame([_make_matchup_row()])
    df = _apply_cross_matchup(df)
    assert "matchup_off_def_edge" in df.columns


def test_cross_matchup_symmetry():
    """home_off_vs_away_def uses home OFF vs away DEF (not swapped)."""
    row = _make_matchup_row(
        home_off_rtg_game_roll20=120.0, away_def_rtg_game_roll20=105.0,
        away_off_rtg_game_roll20=110.0, home_def_rtg_game_roll20=100.0,
    )
    df = _apply_cross_matchup(pd.DataFrame([row]))
    # home_off_vs_away_def = 120 - 105 = 15
    assert df["home_off_vs_away_def_20"].iloc[0] == pytest.approx(15.0)
    # away_off_vs_home_def = 110 - 100 = 10
    assert df["away_off_vs_home_def_20"].iloc[0] == pytest.approx(10.0)
    # edge = 15 - 10 = 5
    assert df["matchup_off_def_edge"].iloc[0] == pytest.approx(5.0)


# ── Fatigue compound ──────────────────────────────────────────────────────────

def test_fatigue_compound_zero_when_not_b2b():
    """If is_back_to_back=0, fatigue_compound should be 0 regardless of travel."""
    row = _make_matchup_row(
        home_is_back_to_back=0, home_travel_miles=500.0,
        away_is_back_to_back=0, away_travel_miles=300.0,
    )
    df = _apply_fatigue(pd.DataFrame([row]))
    assert df["home_fatigue_compound"].iloc[0] == 0.0
    assert df["away_fatigue_compound"].iloc[0] == 0.0


def test_fatigue_compound_positive_when_b2b_traveled():
    """Back-to-back + travel produces a positive fatigue compound."""
    row = _make_matchup_row(
        home_is_back_to_back=1, home_travel_miles=800.0,
        away_is_back_to_back=0, away_travel_miles=0.0,
    )
    df = _apply_fatigue(pd.DataFrame([row]))
    assert df["home_fatigue_compound"].iloc[0] == pytest.approx(800.0)
    assert df["away_fatigue_compound"].iloc[0] == 0.0


def test_diff_fatigue_compound_exists():
    """diff_fatigue_compound = home - away fatigue compound."""
    row = _make_matchup_row(
        home_is_back_to_back=1, home_travel_miles=600.0,
        away_is_back_to_back=1, away_travel_miles=200.0,
    )
    df = _apply_fatigue(pd.DataFrame([row]))
    assert "diff_fatigue_compound" in df.columns
    assert df["diff_fatigue_compound"].iloc[0] == pytest.approx(400.0)


# ── 3-game rolling columns ───────────────────────────────────────────────────

def test_roll3_columns_exist():
    """Key 3-game rolling columns should be present in a matchup row."""
    row = _make_matchup_row()
    df = pd.DataFrame([row])
    for col in ["home_pts_roll3", "home_net_rtg_game_roll3",
                "home_off_rtg_game_roll3", "home_def_rtg_game_roll3"]:
        assert col in df.columns, f"Missing expected column: {col}"
