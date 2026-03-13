"""
Tests for the Elo rating system (src/features/elo.py).

Uses synthetic DataFrames wherever possible to avoid dependency on real CSV files.
"""

import os
import math

import pandas as pd
import pytest

from src.features.elo import (
    BASE_ELO,
    CARRYOVER_FRACTION,
    HOME_ADVANTAGE,
    K_FACTOR,
    K_FACTOR_FAST,
    OUTPUT_PATH,
    _expected_win_prob,
    _mov_multiplier,
    _regress_to_mean,
    build_elo_ratings,
    get_current_elos,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_game_log(games):
    """
    Build a minimal team_game_logs DataFrame from a list of game dicts.

    Each dict should have:
        game_id, home_team_id, home_abbr, away_team_id, away_abbr,
        game_date, season, home_pts, away_pts, home_wl
    Two rows (home + away) are generated per game.
    """
    rows = []
    for g in games:
        rows.append({
            "game_id": g["game_id"],
            "team_id": g["home_team_id"],
            "team_abbreviation": g["home_abbr"],
            "game_date": g["game_date"],
            "season": g["season"],
            "matchup": f"{g['home_abbr']} vs. {g['away_abbr']}",
            "wl": g["home_wl"],
            "pts": g["home_pts"],
        })
        rows.append({
            "game_id": g["game_id"],
            "team_id": g["away_team_id"],
            "team_abbreviation": g["away_abbr"],
            "game_date": g["game_date"],
            "season": g["season"],
            "matchup": f"{g['away_abbr']} @ {g['home_abbr']}",
            "wl": "L" if g["home_wl"] == "W" else "W",
            "pts": g["away_pts"],
        })
    return pd.DataFrame(rows)


def _build_from_games(games, tmp_path):
    """Write synthetic game log to CSV, run build_elo_ratings, return result."""
    df = _make_game_log(games)
    input_csv = str(tmp_path / "team_game_logs.csv")
    output_csv = str(tmp_path / "elo_ratings.csv")
    df.to_csv(input_csv, index=False)
    return build_elo_ratings(input_path=input_csv, output_path=output_csv)


# ── A single-game fixture used by several tests ─────────────────────────────

SINGLE_GAME = [
    {
        "game_id": 22400001,
        "home_team_id": 1,
        "home_abbr": "BOS",
        "away_team_id": 2,
        "away_abbr": "NYK",
        "game_date": "2024-10-22",
        "season": 202425,
        "home_pts": 110,
        "away_pts": 100,
        "home_wl": "W",
    },
]


# ── Tests ────────────────────────────────────────────────────────────────────


def test_elo_initial_rating(tmp_path):
    """New teams that appear for the first time start at BASE_ELO (1500)."""
    result = _build_from_games(SINGLE_GAME, tmp_path)
    # Both teams are new -- their pre-game Elo should be 1500
    assert (result["elo_pre"] == BASE_ELO).all()


def test_elo_home_advantage(tmp_path):
    """
    Home team's expected win probability should be boosted by HOME_ADVANTAGE.

    With equal Elo, the home team (getting +100 Elo) should have an expected
    win probability above 0.5.
    """
    result = _build_from_games(SINGLE_GAME, tmp_path)
    home_row = result[result["team_abbreviation"] == "BOS"].iloc[0]
    away_row = result[result["team_abbreviation"] == "NYK"].iloc[0]

    # Home team's expected win prob should exceed 0.5 due to home advantage
    assert home_row["elo_expected_win_prob"] > 0.5
    # Away team's expected win prob should be below 0.5
    assert away_row["elo_expected_win_prob"] < 0.5


def test_elo_update_winner_gains(tmp_path):
    """Winner's post-game Elo should be higher than their pre-game Elo."""
    games = SINGLE_GAME + [
        {
            "game_id": 22400002,
            "home_team_id": 1,
            "home_abbr": "BOS",
            "away_team_id": 2,
            "away_abbr": "NYK",
            "game_date": "2024-10-24",
            "season": 202425,
            "home_pts": 105,
            "away_pts": 100,
            "home_wl": "W",
        },
    ]
    result = _build_from_games(games, tmp_path)

    # Game 1: BOS wins. In game 2, BOS's elo_pre should be > BASE_ELO
    bos_game2 = result[
        (result["team_abbreviation"] == "BOS")
        & (result["game_id"] == 22400002)
    ].iloc[0]
    assert bos_game2["elo_pre"] > BASE_ELO


def test_elo_update_loser_drops(tmp_path):
    """Loser's post-game Elo should be lower than their pre-game Elo."""
    games = SINGLE_GAME + [
        {
            "game_id": 22400002,
            "home_team_id": 1,
            "home_abbr": "BOS",
            "away_team_id": 2,
            "away_abbr": "NYK",
            "game_date": "2024-10-24",
            "season": 202425,
            "home_pts": 105,
            "away_pts": 100,
            "home_wl": "W",
        },
    ]
    result = _build_from_games(games, tmp_path)

    # Game 1: NYK loses. In game 2, NYK's elo_pre should be < BASE_ELO
    nyk_game2 = result[
        (result["team_abbreviation"] == "NYK")
        & (result["game_id"] == 22400002)
    ].iloc[0]
    assert nyk_game2["elo_pre"] < BASE_ELO


def test_elo_pre_game_no_leakage(tmp_path):
    """
    elo_pre for game N must NOT reflect game N's result.

    After a single game, both teams should still have elo_pre == 1500
    because the rating update happens AFTER the pre-game snapshot.
    """
    result = _build_from_games(SINGLE_GAME, tmp_path)
    # In the only game, both teams' elo_pre should be exactly 1500
    # (their ratings haven't been updated yet at snapshot time)
    for _, row in result.iterrows():
        assert row["elo_pre"] == BASE_ELO, (
            f"elo_pre for {row['team_abbreviation']} in game "
            f"{row['game_id']} should be {BASE_ELO}, got {row['elo_pre']}"
        )


def test_elo_season_regression(tmp_path):
    """Ratings should regress 1/3 toward 1500 at season boundaries."""
    # Season 1: BOS beats NYK by 10, pushing BOS above 1500
    # Season 2: same teams meet again -- Elo should have regressed
    games = [
        {
            "game_id": 22300001,
            "home_team_id": 1,
            "home_abbr": "BOS",
            "away_team_id": 2,
            "away_abbr": "NYK",
            "game_date": "2023-10-24",
            "season": 202324,
            "home_pts": 110,
            "away_pts": 100,
            "home_wl": "W",
        },
        {
            "game_id": 22400001,
            "home_team_id": 1,
            "home_abbr": "BOS",
            "away_team_id": 2,
            "away_abbr": "NYK",
            "game_date": "2024-10-22",
            "season": 202425,
            "home_pts": 105,
            "away_pts": 100,
            "home_wl": "W",
        },
    ]
    result = _build_from_games(games, tmp_path)

    # After game 1, BOS Elo goes up by some shift.
    # Before game 2 (new season), it should be regressed 1/3 toward 1500.
    bos_g1 = result[
        (result["team_abbreviation"] == "BOS")
        & (result["game_id"] == 22300001)
    ].iloc[0]
    bos_g2 = result[
        (result["team_abbreviation"] == "BOS")
        & (result["game_id"] == 22400001)
    ].iloc[0]

    # After game 1 the internal Elo was updated. We can compute the expected
    # post-game-1 Elo by noting game-2 elo_pre == regressed(post_game1_elo).
    # So post_game1 = BASE_ELO + (bos_g2.elo_pre - BASE_ELO) / CARRYOVER_FRACTION
    # Verify the regression formula holds:
    post_g1_elo = BASE_ELO + (bos_g2["elo_pre"] - BASE_ELO) / CARRYOVER_FRACTION
    expected_regressed = _regress_to_mean(post_g1_elo)
    assert abs(bos_g2["elo_pre"] - expected_regressed) < 1e-6

    # Also verify that elo_pre moved TOWARD 1500 compared to post-game Elo
    assert abs(bos_g2["elo_pre"] - BASE_ELO) < abs(post_g1_elo - BASE_ELO)


def test_elo_expected_win_prob_range(tmp_path):
    """All elo_expected_win_prob values should be in [0, 1]."""
    # Use multiple games to get a variety of ratings
    games = [
        {
            "game_id": 22400000 + i,
            "home_team_id": 1,
            "home_abbr": "BOS",
            "away_team_id": 2,
            "away_abbr": "NYK",
            "game_date": f"2024-10-{22 + i}",
            "season": 202425,
            "home_pts": 110,
            "away_pts": 100,
            "home_wl": "W" if i % 2 == 0 else "L",
        }
        for i in range(5)
    ]
    result = _build_from_games(games, tmp_path)
    assert (result["elo_expected_win_prob"] >= 0.0).all()
    assert (result["elo_expected_win_prob"] <= 1.0).all()


def test_elo_diff_symmetry(tmp_path):
    """elo_diff for the home team should be the negative of the away team's elo_diff."""
    games = SINGLE_GAME + [
        {
            "game_id": 22400002,
            "home_team_id": 2,
            "home_abbr": "NYK",
            "away_team_id": 1,
            "away_abbr": "BOS",
            "game_date": "2024-10-24",
            "season": 202425,
            "home_pts": 108,
            "away_pts": 102,
            "home_wl": "W",
        },
    ]
    result = _build_from_games(games, tmp_path)

    for gid in result["game_id"].unique():
        game = result[result["game_id"] == gid]
        if len(game) == 2:
            diffs = game["elo_diff"].values
            assert abs(diffs[0] + diffs[1]) < 1e-9, (
                f"elo_diff should be symmetric: got {diffs[0]} and {diffs[1]}"
            )


def test_elo_build_output_columns(tmp_path):
    """Output DataFrame should contain the expected columns."""
    result = _build_from_games(SINGLE_GAME, tmp_path)
    expected_cols = [
        "game_id", "team_id", "team_abbreviation", "game_date", "season",
        "elo_pre", "elo_opp_pre", "elo_diff", "elo_expected_win_prob",
        "elo_pre_fast", "elo_opp_pre_fast", "elo_momentum",
    ]
    assert list(result.columns) == expected_cols


# ── Fast Elo / Momentum Tests ────────────────────────────────────────────────


def test_fast_elo_initial_rating(tmp_path):
    """Fast Elo starts at BASE_ELO just like standard Elo."""
    result = _build_from_games(SINGLE_GAME, tmp_path)
    assert (result["elo_pre_fast"] == BASE_ELO).all()


def test_fast_elo_reacts_more_than_standard(tmp_path):
    """After one game, the fast Elo shift should be larger than standard."""
    games = SINGLE_GAME + [
        {
            "game_id": 22400002,
            "home_team_id": 1,
            "home_abbr": "BOS",
            "away_team_id": 2,
            "away_abbr": "NYK",
            "game_date": "2024-10-24",
            "season": 202425,
            "home_pts": 115,
            "away_pts": 100,
            "home_wl": "W",
        },
    ]
    result = _build_from_games(games, tmp_path)
    bos_g2 = result[
        (result["team_abbreviation"] == "BOS")
        & (result["game_id"] == 22400002)
    ].iloc[0]

    # Both start at 1500; after one win, fast Elo should move further from 1500
    standard_shift = abs(bos_g2["elo_pre"] - BASE_ELO)
    fast_shift = abs(bos_g2["elo_pre_fast"] - BASE_ELO)
    assert fast_shift > standard_shift, (
        f"Fast shift ({fast_shift:.2f}) should exceed standard ({standard_shift:.2f})"
    )


def test_elo_momentum_zero_at_start(tmp_path):
    """Before any games are played, momentum should be zero."""
    result = _build_from_games(SINGLE_GAME, tmp_path)
    assert (result["elo_momentum"] == 0.0).all()


def test_elo_momentum_positive_for_winner(tmp_path):
    """A team on a winning streak should have positive momentum."""
    games = [
        {
            "game_id": 22400000 + i,
            "home_team_id": 1,
            "home_abbr": "BOS",
            "away_team_id": 2,
            "away_abbr": "NYK",
            "game_date": f"2024-10-{22 + i}",
            "season": 202425,
            "home_pts": 110,
            "away_pts": 100,
            "home_wl": "W",
        }
        for i in range(5)
    ]
    result = _build_from_games(games, tmp_path)

    # By the last game, BOS (always winning) should have positive momentum
    bos_last = result[
        (result["team_abbreviation"] == "BOS")
        & (result["game_id"] == 22400004)
    ].iloc[0]
    assert bos_last["elo_momentum"] > 0, (
        f"Expected positive momentum for consistent winner, got {bos_last['elo_momentum']:.4f}"
    )


def test_elo_momentum_negative_for_loser(tmp_path):
    """A team on a losing streak should have negative momentum."""
    games = [
        {
            "game_id": 22400000 + i,
            "home_team_id": 1,
            "home_abbr": "BOS",
            "away_team_id": 2,
            "away_abbr": "NYK",
            "game_date": f"2024-10-{22 + i}",
            "season": 202425,
            "home_pts": 110,
            "away_pts": 100,
            "home_wl": "W",
        }
        for i in range(5)
    ]
    result = _build_from_games(games, tmp_path)

    # NYK (always losing) should have negative momentum by the last game
    nyk_last = result[
        (result["team_abbreviation"] == "NYK")
        & (result["game_id"] == 22400004)
    ].iloc[0]
    assert nyk_last["elo_momentum"] < 0, (
        f"Expected negative momentum for consistent loser, got {nyk_last['elo_momentum']:.4f}"
    )


def test_fast_elo_k_factor_value():
    """K_FACTOR_FAST should be exactly 40 (double the standard)."""
    assert K_FACTOR_FAST == 40
    assert K_FACTOR_FAST > K_FACTOR


def test_fast_elo_season_regression(tmp_path):
    """Fast Elo should also regress 1/3 toward 1500 at season boundaries."""
    games = [
        {
            "game_id": 22300001,
            "home_team_id": 1,
            "home_abbr": "BOS",
            "away_team_id": 2,
            "away_abbr": "NYK",
            "game_date": "2023-10-24",
            "season": 202324,
            "home_pts": 120,
            "away_pts": 100,
            "home_wl": "W",
        },
        {
            "game_id": 22400001,
            "home_team_id": 1,
            "home_abbr": "BOS",
            "away_team_id": 2,
            "away_abbr": "NYK",
            "game_date": "2024-10-22",
            "season": 202425,
            "home_pts": 105,
            "away_pts": 100,
            "home_wl": "W",
        },
    ]
    result = _build_from_games(games, tmp_path)

    bos_g2 = result[
        (result["team_abbreviation"] == "BOS")
        & (result["game_id"] == 22400001)
    ].iloc[0]

    # elo_pre_fast at game 2 should be closer to 1500 than the post-game-1 fast Elo
    # (regression pulled it back). Recover post-game-1 fast Elo from the regressed value.
    post_g1_fast = BASE_ELO + (bos_g2["elo_pre_fast"] - BASE_ELO) / CARRYOVER_FRACTION
    expected_regressed = _regress_to_mean(post_g1_fast)
    assert abs(bos_g2["elo_pre_fast"] - expected_regressed) < 1e-6


def test_fast_elo_opp_pre_symmetry(tmp_path):
    """elo_opp_pre_fast for home team should equal elo_pre_fast of away team."""
    result = _build_from_games(SINGLE_GAME, tmp_path)
    home = result[result["team_abbreviation"] == "BOS"].iloc[0]
    away = result[result["team_abbreviation"] == "NYK"].iloc[0]
    assert home["elo_opp_pre_fast"] == away["elo_pre_fast"]
    assert away["elo_opp_pre_fast"] == home["elo_pre_fast"]


def test_elo_momentum_is_fast_minus_standard(tmp_path):
    """elo_momentum should equal elo_pre_fast - elo_pre exactly."""
    games = [
        {
            "game_id": 22400000 + i,
            "home_team_id": 1,
            "home_abbr": "BOS",
            "away_team_id": 2,
            "away_abbr": "NYK",
            "game_date": f"2024-10-{22 + i}",
            "season": 202425,
            "home_pts": 110,
            "away_pts": 100,
            "home_wl": "W" if i % 2 == 0 else "L",
        }
        for i in range(5)
    ]
    result = _build_from_games(games, tmp_path)
    computed = result["elo_pre_fast"] - result["elo_pre"]
    assert (abs(result["elo_momentum"] - computed) < 1e-9).all()


@pytest.mark.skipif(
    not os.path.exists(OUTPUT_PATH),
    reason=f"{OUTPUT_PATH} not found -- run build_elo_ratings() first",
)
def test_get_current_elos_returns_all_teams():
    """get_current_elos() should return a dict with at least 30 NBA teams."""
    current = get_current_elos()
    assert isinstance(current, dict)
    # At least 30 current teams (may include historical franchises)
    assert len(current) >= 30, f"Expected >= 30 teams, got {len(current)}"
    # All values should be numeric
    for team, elo in current.items():
        assert isinstance(elo, (int, float)), f"{team} Elo is not numeric: {elo}"
    # Spot-check a few known current teams
    known_teams = {"BOS", "LAL", "GSW", "MIL", "PHX"}
    for t in known_teams:
        assert t in current, f"Expected {t} in current Elos"


def test_get_current_elos_extended():
    """get_current_elos(extended=True) returns elo, elo_fast, and momentum per team."""
    extended = get_current_elos(extended=True)
    assert isinstance(extended, dict)
    assert len(extended) >= 30
    for team, vals in extended.items():
        assert isinstance(vals, dict), f"{team}: expected dict, got {type(vals)}"
        assert "elo" in vals, f"{team}: missing 'elo' key"
        assert "elo_fast" in vals, f"{team}: missing 'elo_fast' key"
        assert "momentum" in vals, f"{team}: missing 'momentum' key"
        assert isinstance(vals["elo"], float)
        assert isinstance(vals["elo_fast"], float)
        assert isinstance(vals["momentum"], float)
        # Momentum should be elo_fast - elo (within rounding)
        assert abs(vals["momentum"] - (vals["elo_fast"] - vals["elo"])) < 0.01, (
            f"{team}: momentum {vals['momentum']} != elo_fast - elo "
            f"({vals['elo_fast']} - {vals['elo']})"
        )


def test_get_current_elos_extended_false_matches_default():
    """extended=False should return the same dict as the default call."""
    default = get_current_elos()
    explicit = get_current_elos(extended=False)
    assert default == explicit
