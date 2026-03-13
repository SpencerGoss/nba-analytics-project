"""Stage 2: Per-stat prediction models (PTS, REB, AST, 3PM).

Each stat has its own GBM model that predicts per-36 rates. The prediction
is then scaled by predicted minutes: stat = per36_rate * (predicted_minutes / 36).

This two-stage approach (minutes -> per-36 rates) improves accuracy because
minutes explain ~65% of stat variance.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

STAT_TARGETS = ["pts", "reb", "ast", "fg3m"]

STAT_FEATURES = [
    "minutes_ewma",
    "usage_rate_ewma",
    "pts_ewma",
    "reb_ewma",
    "ast_ewma",
    "fg3m_ewma",
    "is_home",
    "is_b2b",
    "season_game_num",
]


def train_stat_models(
    features_df: pd.DataFrame,
    artifacts_dir: str = "models/artifacts",
) -> dict:
    """Train one GBM per stat target on per-36 rates.

    Args:
        features_df: DataFrame with STAT_FEATURES columns + per-36 target columns.
        artifacts_dir: Where to save model artifacts.

    Returns:
        Dict of {stat: {cv_mae, n_samples}}.
    """
    artifacts = Path(artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    results = {}

    for stat in STAT_TARGETS:
        target_col = f"{stat}_per36"
        df = features_df.dropna(subset=STAT_FEATURES + [target_col]).copy()

        X = df[STAT_FEATURES]
        y = df[target_col]

        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=20,
            learning_rate=0.05,
            loss="huber",
            random_state=42,
        )

        cv_scores = cross_val_score(
            model, X, y, cv=3, scoring="neg_mean_absolute_error"
        )
        cv_mae = -cv_scores.mean()

        model.fit(X, y)

        # Pickle is required for scikit-learn model serialization (trusted local files)
        with open(artifacts / f"player_{stat}_model.pkl", "wb") as f:
            pickle.dump(model, f)

        results[stat] = {"cv_mae": round(cv_mae, 3), "n_samples": len(df)}

    # Pickle is required for scikit-learn feature list (trusted local files)
    with open(artifacts / "player_stat_features.pkl", "wb") as f:
        pickle.dump(STAT_FEATURES, f)
    with open(artifacts / "player_stat_metadata.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def predict_player_stat(
    stat: str,
    features: pd.DataFrame,
    predicted_minutes: float,
    artifacts_dir: str = "models/artifacts",
) -> float:
    """Predict a stat for a player, scaled by predicted minutes.

    Args:
        stat: One of STAT_TARGETS (pts, reb, ast, fg3m).
        features: Single-row DataFrame with STAT_FEATURES columns.
        predicted_minutes: Output from Stage 1 minutes model.
        artifacts_dir: Where model artifacts are stored.

    Returns:
        Predicted stat value (e.g., 22.5 points).
    """
    if stat not in STAT_TARGETS:
        raise ValueError(f"Unknown stat: {stat}. Must be one of {STAT_TARGETS}")

    artifacts = Path(artifacts_dir)

    # Pickle required for scikit-learn models (trusted local artifacts only)
    with open(artifacts / f"player_{stat}_model.pkl", "rb") as f:
        model = pickle.load(f)  # noqa: S301
    with open(artifacts / "player_stat_features.pkl", "rb") as f:
        feat_cols = pickle.load(f)  # noqa: S301

    X = features[feat_cols].iloc[:1]
    per36_rate = float(model.predict(X)[0])
    per36_rate = max(per36_rate, 0.0)  # No negative stats

    return per36_rate * (predicted_minutes / 36.0)
