"""Tests for Stage 1: Minutes prediction model with blowout adjustment."""
import numpy as np
import pandas as pd
import pytest


def test_blowout_adjustment():
    """Blowout adjustment reduces minutes for large spreads."""
    from src.models.player_minutes_model import apply_blowout_adjustment
    base = 32.0
    adj_close = apply_blowout_adjustment(base, spread=2.0)
    adj_7 = apply_blowout_adjustment(base, spread=7.0)
    adj_14 = apply_blowout_adjustment(base, spread=14.0)
    # Close game: minimal reduction
    assert adj_close > 30
    # 7pt spread: moderate reduction
    assert 28 < adj_7 < 32
    # 14pt spread: larger reduction
    assert adj_14 < adj_7
    # Never absurdly low
    assert adj_14 > 18


def test_blowout_zero_spread():
    """Zero spread should return close to base minutes."""
    from src.models.player_minutes_model import apply_blowout_adjustment
    result = apply_blowout_adjustment(32.0, spread=0.0)
    assert abs(result - 32.0) < 2.0


def test_blowout_negative_spread():
    """Negative spread uses absolute value -- same result as positive."""
    from src.models.player_minutes_model import apply_blowout_adjustment
    pos = apply_blowout_adjustment(30.0, spread=10.0)
    neg = apply_blowout_adjustment(30.0, spread=-10.0)
    assert pos == neg


def test_blowout_monotonic():
    """Larger spreads should always reduce minutes more."""
    from src.models.player_minutes_model import apply_blowout_adjustment
    base = 34.0
    prev = base
    for sp in [0, 3, 6, 9, 12, 15, 20]:
        adj = apply_blowout_adjustment(base, spread=float(sp))
        assert adj <= prev
        prev = adj


def test_train_minutes_model_callable():
    from src.models.player_minutes_model import train_minutes_model
    assert callable(train_minutes_model)


def test_predict_minutes_callable():
    from src.models.player_minutes_model import predict_minutes
    assert callable(predict_minutes)


def test_train_and_predict_synthetic():
    """Train on synthetic data, predict, verify output is reasonable."""
    from src.models.player_minutes_model import train_minutes_model, predict_minutes
    import tempfile
    import os

    # Create synthetic training features
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "player_id": np.repeat(range(1, 11), 20),
        "game_date": pd.date_range("2025-01-01", periods=20).tolist() * 10,
        "minutes": np.random.normal(28, 5, n).clip(0, 48),
        "minutes_ewma": np.random.normal(28, 3, n),
        "usage_rate_ewma": np.random.normal(20, 5, n),
        "pts_ewma": np.random.normal(15, 5, n),
        "is_home": np.random.randint(0, 2, n),
        "is_b2b": np.random.randint(0, 2, n),
        "season_game_num": np.tile(range(1, 21), 10),
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        # Train
        result = train_minutes_model(df, artifacts_dir=tmpdir)
        assert result is not None
        assert os.path.exists(os.path.join(tmpdir, "player_minutes_model.pkl"))
        assert os.path.exists(os.path.join(tmpdir, "player_minutes_features.pkl"))
        assert os.path.exists(os.path.join(tmpdir, "player_minutes_metadata.json"))
        assert "cv_mae" in result
        assert "n_samples" in result
        assert result["n_samples"] == 200

        # Predict
        features = df.iloc[0:1].drop(columns=["minutes"])
        pred = predict_minutes(features, spread=3.0, artifacts_dir=tmpdir)
        assert isinstance(pred, float)
        assert 10 < pred < 48  # Reasonable range


def test_predict_with_blowout_spread():
    """Prediction with large spread should be less than with zero spread."""
    from src.models.player_minutes_model import train_minutes_model, predict_minutes
    import tempfile

    np.random.seed(99)
    n = 200
    df = pd.DataFrame({
        "player_id": np.repeat(range(1, 11), 20),
        "game_date": pd.date_range("2025-01-01", periods=20).tolist() * 10,
        "minutes": np.random.normal(30, 4, n).clip(0, 48),
        "minutes_ewma": np.random.normal(30, 3, n),
        "usage_rate_ewma": np.random.normal(22, 4, n),
        "pts_ewma": np.random.normal(18, 5, n),
        "is_home": np.random.randint(0, 2, n),
        "is_b2b": np.random.randint(0, 2, n),
        "season_game_num": np.tile(range(1, 21), 10),
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        train_minutes_model(df, artifacts_dir=tmpdir)
        features = df.iloc[0:1].drop(columns=["minutes"])
        pred_close = predict_minutes(features, spread=1.0, artifacts_dir=tmpdir)
        pred_blowout = predict_minutes(features, spread=15.0, artifacts_dir=tmpdir)
        assert pred_blowout < pred_close


def test_predict_clipped_to_valid_range():
    """Predicted minutes should be clipped to [0, 48]."""
    from src.models.player_minutes_model import train_minutes_model, predict_minutes
    import tempfile

    np.random.seed(7)
    n = 200
    df = pd.DataFrame({
        "player_id": np.repeat(range(1, 11), 20),
        "game_date": pd.date_range("2025-01-01", periods=20).tolist() * 10,
        "minutes": np.random.normal(28, 5, n).clip(0, 48),
        "minutes_ewma": np.random.normal(28, 3, n),
        "usage_rate_ewma": np.random.normal(20, 5, n),
        "pts_ewma": np.random.normal(15, 5, n),
        "is_home": np.random.randint(0, 2, n),
        "is_b2b": np.random.randint(0, 2, n),
        "season_game_num": np.tile(range(1, 21), 10),
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        train_minutes_model(df, artifacts_dir=tmpdir)
        features = df.iloc[0:1].drop(columns=["minutes"])
        pred = predict_minutes(features, spread=0.0, artifacts_dir=tmpdir)
        assert 0 <= pred <= 48


def test_train_drops_nan_rows():
    """Training should silently drop rows with NaN in features or target."""
    from src.models.player_minutes_model import train_minutes_model
    import tempfile

    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "player_id": range(n),
        "game_date": pd.date_range("2025-01-01", periods=n),
        "minutes": np.random.normal(28, 5, n).clip(0, 48),
        "minutes_ewma": np.random.normal(28, 3, n),
        "usage_rate_ewma": np.random.normal(20, 5, n),
        "pts_ewma": np.random.normal(15, 5, n),
        "is_home": np.random.randint(0, 2, n),
        "is_b2b": np.random.randint(0, 2, n),
        "season_game_num": range(1, n + 1),
    })
    # Inject NaNs
    df.loc[0, "minutes"] = np.nan
    df.loc[1, "minutes_ewma"] = np.nan

    with tempfile.TemporaryDirectory() as tmpdir:
        result = train_minutes_model(df, artifacts_dir=tmpdir)
        assert result["n_samples"] == 98
