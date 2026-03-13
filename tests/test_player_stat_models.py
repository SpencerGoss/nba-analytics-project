"""Tests for Stage 2: Per-stat prediction models (PTS, REB, AST, 3PM)."""
import numpy as np
import pandas as pd
import pytest


def test_stat_targets():
    from src.models.player_stat_models import STAT_TARGETS
    assert set(STAT_TARGETS) == {"pts", "reb", "ast", "fg3m"}


def test_train_callable():
    from src.models.player_stat_models import train_stat_models
    assert callable(train_stat_models)


def test_predict_callable():
    from src.models.player_stat_models import predict_player_stat
    assert callable(predict_player_stat)


def _make_synthetic_df(n=200, seed=42):
    """Create synthetic player data for testing."""
    np.random.seed(seed)
    return pd.DataFrame({
        "player_id": np.repeat(range(1, 11), 20),
        "game_date": pd.date_range("2025-01-01", periods=20).tolist() * 10,
        "minutes": np.random.normal(28, 5, n).clip(5, 48),
        "pts_per36": np.random.normal(18, 5, n).clip(0),
        "reb_per36": np.random.normal(7, 3, n).clip(0),
        "ast_per36": np.random.normal(4, 2, n).clip(0),
        "fg3m_per36": np.random.normal(2, 1, n).clip(0),
        "minutes_ewma": np.random.normal(28, 3, n),
        "usage_rate_ewma": np.random.normal(20, 5, n),
        "pts_ewma": np.random.normal(15, 5, n),
        "reb_ewma": np.random.normal(5, 2, n),
        "ast_ewma": np.random.normal(3, 2, n),
        "fg3m_ewma": np.random.normal(1.5, 1, n),
        "is_home": np.random.randint(0, 2, n),
        "is_b2b": np.random.randint(0, 2, n),
        "season_game_num": np.tile(range(1, 21), 10),
    })


def test_train_and_predict_synthetic():
    """Train on synthetic data, predict, verify output is reasonable."""
    from src.models.player_stat_models import train_stat_models, predict_player_stat
    import tempfile

    df = _make_synthetic_df()

    with tempfile.TemporaryDirectory() as tmpdir:
        result = train_stat_models(df, artifacts_dir=tmpdir)
        assert "pts" in result
        assert "reb" in result
        assert "ast" in result
        assert "fg3m" in result

        # Predict PTS for a player playing 30 minutes
        features = df.iloc[0:1]
        pred = predict_player_stat("pts", features, predicted_minutes=30.0, artifacts_dir=tmpdir)
        assert isinstance(pred, float)
        assert 0 < pred < 60  # Reasonable PTS range


def test_scaling_by_minutes():
    """More minutes should generally mean more stats."""
    from src.models.player_stat_models import train_stat_models, predict_player_stat
    import tempfile

    df = _make_synthetic_df()

    with tempfile.TemporaryDirectory() as tmpdir:
        train_stat_models(df, artifacts_dir=tmpdir)
        features = df.iloc[0:1]
        pts_20 = predict_player_stat("pts", features, predicted_minutes=20.0, artifacts_dir=tmpdir)
        pts_36 = predict_player_stat("pts", features, predicted_minutes=36.0, artifacts_dir=tmpdir)
        # More minutes = more points (same per-36 rate scaled differently)
        assert pts_36 > pts_20


def test_predict_unknown_stat_raises():
    """Requesting an unknown stat should raise ValueError."""
    from src.models.player_stat_models import train_stat_models, predict_player_stat
    import tempfile

    df = _make_synthetic_df()

    with tempfile.TemporaryDirectory() as tmpdir:
        train_stat_models(df, artifacts_dir=tmpdir)
        features = df.iloc[0:1]
        with pytest.raises(ValueError, match="Unknown stat"):
            predict_player_stat("stl", features, predicted_minutes=30.0, artifacts_dir=tmpdir)


def test_no_negative_predictions():
    """Predictions should never be negative."""
    from src.models.player_stat_models import train_stat_models, predict_player_stat
    import tempfile

    df = _make_synthetic_df()

    with tempfile.TemporaryDirectory() as tmpdir:
        train_stat_models(df, artifacts_dir=tmpdir)
        features = df.iloc[0:1]
        for stat in ["pts", "reb", "ast", "fg3m"]:
            pred = predict_player_stat(stat, features, predicted_minutes=10.0, artifacts_dir=tmpdir)
            assert pred >= 0, f"{stat} prediction was negative: {pred}"


def test_metadata_saved():
    """Training should save metadata JSON."""
    from src.models.player_stat_models import train_stat_models
    import tempfile
    import json
    from pathlib import Path

    df = _make_synthetic_df()

    with tempfile.TemporaryDirectory() as tmpdir:
        train_stat_models(df, artifacts_dir=tmpdir)
        meta_path = Path(tmpdir) / "player_stat_metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        for stat in ["pts", "reb", "ast", "fg3m"]:
            assert stat in meta
            assert "cv_mae" in meta[stat]
            assert "n_samples" in meta[stat]


def test_quantile_predict_callable():
    from src.models.player_stat_models import predict_player_stat_quantiles
    assert callable(predict_player_stat_quantiles)


def test_quantile_ordering():
    """p25 <= p50 <= p75 for each stat."""
    from src.models.player_stat_models import (
        train_stat_models,
        train_quantile_models,
        predict_player_stat_quantiles,
    )
    import tempfile

    df = _make_synthetic_df()

    with tempfile.TemporaryDirectory() as tmpdir:
        train_stat_models(df, artifacts_dir=tmpdir)
        train_quantile_models(df, artifacts_dir=tmpdir)

        features = df.iloc[0:1]
        q = predict_player_stat_quantiles(
            "pts", features, predicted_minutes=30.0, artifacts_dir=tmpdir
        )
        assert "p25" in q
        assert "p50" in q
        assert "p75" in q
        assert q["p25"] <= q["p50"] <= q["p75"]


def test_quantile_all_stats():
    """Quantile models should work for all stat targets."""
    from src.models.player_stat_models import (
        STAT_TARGETS,
        train_stat_models,
        train_quantile_models,
        predict_player_stat_quantiles,
    )
    import tempfile

    df = _make_synthetic_df()

    with tempfile.TemporaryDirectory() as tmpdir:
        train_stat_models(df, artifacts_dir=tmpdir)
        train_quantile_models(df, artifacts_dir=tmpdir)
        features = df.iloc[0:1]
        for stat in STAT_TARGETS:
            q = predict_player_stat_quantiles(
                stat, features, predicted_minutes=30.0, artifacts_dir=tmpdir
            )
            assert q["p25"] >= 0
            assert q["p75"] >= q["p25"]
