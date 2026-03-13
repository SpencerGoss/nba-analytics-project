"""Stage 1: Minutes prediction model with blowout adjustment.

Predicts how many minutes a player will play using their recent EWMA features.
Applies a blowout adjustment based on the expected game spread -- starters in
blowouts play fewer minutes.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score


# Blowout adjustment: logistic curve parameters
# At spread=0, blowout_prob ~0.12 (minimal). At spread=15, blowout_prob ~0.85.
BLOWOUT_STEEPNESS = 0.3
BLOWOUT_MIDPOINT = 10.0
BLOWOUT_REDUCTION = 0.25  # Max 25% minutes reduction in a blowout

MINUTES_FEATURES = [
    "minutes_ewma",
    "usage_rate_ewma",
    "pts_ewma",
    "is_home",
    "is_b2b",
    "season_game_num",
]


def apply_blowout_adjustment(base_minutes: float, spread: float) -> float:
    """Reduce predicted minutes based on blowout probability.

    Uses a logistic curve: blowout_prob = 1 / (1 + exp(-k*(|spread| - midpoint)))
    Minutes reduced by: blowout_prob * BLOWOUT_REDUCTION * base_minutes

    Args:
        base_minutes: Predicted minutes without blowout adjustment.
        spread: Expected game spread (absolute value used).

    Returns:
        Adjusted minutes (float).
    """
    abs_spread = abs(spread)
    blowout_prob = 1 / (1 + np.exp(-BLOWOUT_STEEPNESS * (abs_spread - BLOWOUT_MIDPOINT)))
    reduction = blowout_prob * BLOWOUT_REDUCTION
    return base_minutes * (1 - reduction)


def train_minutes_model(
    features_df: pd.DataFrame,
    artifacts_dir: str = "models/artifacts",
) -> dict:
    """Train minutes prediction model.

    Args:
        features_df: DataFrame with MINUTES_FEATURES columns + 'minutes' target.
        artifacts_dir: Where to save model artifacts.

    Returns:
        Dict with training metrics.
    """
    artifacts = Path(artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    # Drop rows with NaN in features or target
    df = features_df.dropna(subset=MINUTES_FEATURES + ["minutes"]).copy()

    X = df[MINUTES_FEATURES]
    y = df["minutes"]

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=20,
        learning_rate=0.05,
        loss="huber",
        random_state=42,
    )

    # Cross-validation for MAE estimate
    cv_scores = cross_val_score(model, X, y, cv=3, scoring="neg_mean_absolute_error")
    cv_mae = -cv_scores.mean()

    # Train on full data
    model.fit(X, y)

    # Save artifacts (pickle required for scikit-learn models -- trusted local files only)
    with open(artifacts / "player_minutes_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(artifacts / "player_minutes_features.pkl", "wb") as f:
        pickle.dump(MINUTES_FEATURES, f)

    metadata = {
        "cv_mae": round(cv_mae, 3),
        "n_samples": len(df),
        "features": MINUTES_FEATURES,
    }
    with open(artifacts / "player_minutes_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def predict_minutes(
    features: pd.DataFrame,
    spread: float = 0.0,
    artifacts_dir: str = "models/artifacts",
) -> float:
    """Predict minutes for a player with blowout adjustment.

    Args:
        features: Single-row DataFrame with MINUTES_FEATURES columns.
        spread: Expected game spread (absolute value used).
        artifacts_dir: Where model artifacts are stored.

    Returns:
        Predicted minutes (float), adjusted for blowout risk.
    """
    artifacts = Path(artifacts_dir)

    # Load trusted local model artifacts (pickle required for scikit-learn)
    with open(artifacts / "player_minutes_model.pkl", "rb") as f:
        model = pickle.load(f)  # noqa: S301 - trusted local artifact
    with open(artifacts / "player_minutes_features.pkl", "rb") as f:
        feat_cols = pickle.load(f)  # noqa: S301 - trusted local artifact

    X = features[feat_cols].iloc[:1]
    base_pred = float(model.predict(X)[0])
    base_pred = np.clip(base_pred, 0, 48)

    return apply_blowout_adjustment(base_pred, spread)
