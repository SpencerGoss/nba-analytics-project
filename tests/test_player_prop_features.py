"""Tests for build_player_prop_features() — player prop feature engineering."""

import pandas as pd
import numpy as np
import pytest


def _make_fake_game_logs(n_players=3, n_games=20):
    """Create synthetic player game logs for testing."""
    rng = np.random.RandomState(42)
    rows = []
    for pid in range(1, n_players + 1):
        for g in range(n_games):
            rows.append({
                "player_id": pid,
                "player_name": f"Player_{pid}",
                "team_abbreviation": "LAL",
                "game_id": f"002260{g:04d}",
                "game_date": f"2025-{11 + g // 15:02d}-{1 + g % 28:02d}",
                "matchup": "LAL vs. BOS" if g % 2 == 0 else "LAL @ BOS",
                "min": float(rng.randint(20, 38)),
                "fgm": float(rng.randint(3, 12)),
                "fga": float(rng.randint(8, 22)),
                "fg3m": float(rng.randint(0, 5)),
                "fg3a": float(rng.randint(2, 10)),
                "ftm": float(rng.randint(0, 8)),
                "fta": float(rng.randint(0, 10)),
                "oreb": float(rng.randint(0, 3)),
                "dreb": float(rng.randint(1, 8)),
                "reb": float(rng.randint(2, 12)),
                "ast": float(rng.randint(1, 10)),
                "pts": float(rng.randint(8, 35)),
                "pf": float(rng.randint(0, 5)),
                "tov": float(rng.randint(0, 5)),
                "plus_minus": float(rng.randint(-15, 15)),
                "season": 202526,
                "season_id": 22025,
                "wl": "W" if rng.random() > 0.5 else "L",
            })
    return pd.DataFrame(rows)


def test_build_player_prop_features_cols():
    """Output must have all required prop feature columns."""
    from src.features.player_features import build_player_prop_features

    fake_logs = _make_fake_game_logs()
    df = build_player_prop_features(game_logs=fake_logs)
    required = [
        "player_id", "game_date", "minutes",
        "pts_per36", "reb_per36", "ast_per36", "fg3m_per36",
        "minutes_ewma", "usage_rate_ewma",
        "is_home", "is_b2b",
    ]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"


def test_features_shifted():
    """EWMA features must be NaN on first game (shift prevents leakage)."""
    from src.features.player_features import build_player_prop_features

    fake_logs = _make_fake_game_logs(n_players=2, n_games=10)
    df = build_player_prop_features(game_logs=fake_logs)
    # Use nth(0) to get the actual first row (not first non-NaN like .first())
    first_games = df.groupby("player_id").nth(0)
    # First game should have NaN for EWMA features (shifted)
    assert first_games["minutes_ewma"].isna().all(), "EWMA not shifted -- data leakage!"


def test_per36_rates_reasonable():
    """Per-36 rates should be positive and finite."""
    from src.features.player_features import build_player_prop_features

    fake_logs = _make_fake_game_logs()
    df = build_player_prop_features(game_logs=fake_logs)
    # Filter to rows with actual minutes played
    valid = df[df["minutes"] > 0]
    for col in ["pts_per36", "reb_per36", "ast_per36", "fg3m_per36"]:
        assert (valid[col] >= 0).all(), f"{col} has negative values"
        assert valid[col].isna().sum() == 0, f"{col} has NaN for non-zero minutes"


def test_is_b2b_flag():
    """is_b2b should be 0 or 1."""
    from src.features.player_features import build_player_prop_features

    fake_logs = _make_fake_game_logs()
    df = build_player_prop_features(game_logs=fake_logs)
    assert df["is_b2b"].isin([0, 1, 0.0, 1.0]).all()


def test_ewma_columns_present():
    """EWMA columns for pts, reb, ast, fg3m should exist."""
    from src.features.player_features import build_player_prop_features

    fake_logs = _make_fake_game_logs()
    df = build_player_prop_features(game_logs=fake_logs)
    for stat in ["pts_ewma", "reb_ewma", "ast_ewma", "fg3m_ewma"]:
        assert stat in df.columns, f"Missing EWMA column: {stat}"


def test_season_game_num():
    """season_game_num should start at 1 and increase."""
    from src.features.player_features import build_player_prop_features

    fake_logs = _make_fake_game_logs(n_players=1, n_games=15)
    df = build_player_prop_features(game_logs=fake_logs)
    assert df["season_game_num"].min() == 1
    assert df["season_game_num"].max() == 15


def test_zero_minutes_per36_safe():
    """Per-36 rates should be 0 when minutes are 0 (no division error)."""
    from src.features.player_features import build_player_prop_features

    fake_logs = _make_fake_game_logs(n_players=1, n_games=5)
    fake_logs.loc[fake_logs.index[2], "min"] = 0.0
    df = build_player_prop_features(game_logs=fake_logs)
    zero_min_row = df[df["minutes"] == 0.0]
    for col in ["pts_per36", "reb_per36", "ast_per36", "fg3m_per36"]:
        assert (zero_min_row[col] == 0.0).all(), f"{col} not 0 when minutes=0"


def test_does_not_mutate_input():
    """Input DataFrame should not be modified."""
    from src.features.player_features import build_player_prop_features

    fake_logs = _make_fake_game_logs(n_players=1, n_games=10)
    original_cols = list(fake_logs.columns)
    original_len = len(fake_logs)
    build_player_prop_features(game_logs=fake_logs)
    assert list(fake_logs.columns) == original_cols
    assert len(fake_logs) == original_len
