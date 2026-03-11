"""Tests for win/loss streak feature (_compute_streak)."""

import numpy as np
import pandas as pd
import pytest

from src.features.team_game_features import _compute_streak


def _make_team_games(results, team_id=1, season=202425):
    """Build a minimal DataFrame from a list of 'W'/'L' strings."""
    n = len(results)
    return pd.DataFrame({
        "team_id": team_id,
        "season": season,
        "game_date": pd.date_range("2025-10-20", periods=n, freq="2D"),
        "wl": results,
    })


# ── First game ────────────────────────────────────────────────────────────────

def test_streak_starts_at_zero():
    """First game of a season has streak 0 (no prior games)."""
    df = _make_team_games(["W", "W", "L"])
    streak = _compute_streak(df)
    assert streak.iloc[0] == 0


# ── Positive (win) streaks ────────────────────────────────────────────────────

def test_streak_positive_after_wins():
    """After 3 consecutive wins, streak entering game 4 should be 3."""
    df = _make_team_games(["W", "W", "W", "W"])
    streak = _compute_streak(df)
    # Game 0: 0, Game 1: 1 (1 prior W), Game 2: 2, Game 3: 3
    assert streak.iloc[3] == 3


# ── Negative (loss) streaks ───────────────────────────────────────────────────

def test_streak_negative_after_losses():
    """After 2 consecutive losses, streak entering game 3 should be -2."""
    df = _make_team_games(["L", "L", "L"])
    streak = _compute_streak(df)
    assert streak.iloc[2] == -2


# ── Reset on result change ────────────────────────────────────────────────────

def test_streak_resets_on_result_change():
    """A win after a loss streak resets; pre-game value is the loss streak."""
    df = _make_team_games(["L", "L", "W", "W"])
    streak = _compute_streak(df)
    # Game 2 enters with -2 loss streak (2 prior losses)
    assert streak.iloc[2] == -2
    # Game 3 enters with +1 win streak (the W at game 2)
    assert streak.iloc[3] == 1


# ── Shift / no leakage ───────────────────────────────────────────────────────

def test_streak_shift_no_leakage():
    """Streak at game N reflects results through game N-1 only."""
    df = _make_team_games(["W", "L", "W", "W", "W"])
    streak = _compute_streak(df)
    # Game 1: prior is W -> streak=1
    assert streak.iloc[1] == 1
    # Game 2: prior is L -> streak=-1 (just the one loss)
    assert streak.iloc[2] == -1


# ── Per-team independence ─────────────────────────────────────────────────────

def test_streak_per_team_independent():
    """Different teams have independent streaks when run separately."""
    df_a = _make_team_games(["W", "W", "W"], team_id=1)
    df_b = _make_team_games(["L", "L", "L"], team_id=2)
    streak_a = _compute_streak(df_a)
    streak_b = _compute_streak(df_b)
    # Team A at game 2: +2, Team B at game 2: -2
    assert streak_a.iloc[2] == 2
    assert streak_b.iloc[2] == -2


# ── Season boundary ──────────────────────────────────────────────────────────

def test_streak_resets_at_season_boundary():
    """New season starts fresh (streak computed per team+season group)."""
    # Simulate: season 1 ends with 3 wins, season 2 starts
    s1 = _make_team_games(["W", "W", "W"], season=202324)
    s2 = _make_team_games(["L", "L"], season=202425)
    # _compute_streak is called per (team_id, season) group in production.
    # Verify each group independently resets:
    streak_s1 = _compute_streak(s1)
    streak_s2 = _compute_streak(s2)
    assert streak_s1.iloc[2] == 2   # entering game 3 with 2 prior wins
    assert streak_s2.iloc[0] == 0   # new season, no prior games


# ── Column presence in matchup diff ───────────────────────────────────────────

def test_streak_column_in_diff_stats():
    """'streak' is listed in the diff_stats for matchup differentials."""
    # Rather than running the full pipeline, verify the constant list
    from src.features.team_game_features import build_matchup_dataset
    import inspect
    source = inspect.getsource(build_matchup_dataset)
    assert '"streak"' in source, "streak should be in diff_stats list"
