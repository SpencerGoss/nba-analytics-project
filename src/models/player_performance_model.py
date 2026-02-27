"""
Player Performance Prediction Model
=====================================
Trains regression models to forecast individual player stats for the next game
(points, rebounds, assists) using rolling pre-game features.

Workflow:
  1. Load player_game_features.csv (built by player_features.py)
  2. Time-based train/test split
  3. Train a GradientBoostingRegressor per stat target
  4. Evaluate with MAE and RMSE
  5. Save model artifacts to models/artifacts/

Usage:
    python src/models/player_performance_model.py

    Or import:
        from src.models.player_performance_model import train_player_models
        models, metrics = train_player_models()
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


# ── Config ─────────────────────────────────────────────────────────────────────

FEATURES_PATH  = "data/features/player_game_features.csv"
ARTIFACTS_DIR  = "models/artifacts"

# Stat targets to predict
TARGETS = ["pts", "reb", "ast"]

# Most recent seasons held out for evaluation
TEST_SEASONS = ["202324", "202425"]

# Minimum games played in training set for a player to be included
MIN_TRAIN_GAMES = 20


# ── Feature selection ──────────────────────────────────────────────────────────

def get_feature_cols(df: pd.DataFrame, target: str) -> list:
    """
    Return feature columns for predicting `target`.
    Excludes the raw current-game stat (that's the label),
    plus meta/identifier columns.
    """
    # Raw current-game stats — all of these would leak the result
    raw_game_stats = [
        "pts", "reb", "ast", "stl", "blk", "tov", "pf",
        "min", "fgm", "fga", "fg_pct",
        "fg3m", "fg3a", "fg3_pct",
        "ftm", "fta", "ft_pct",
        "plus_minus", "win", "wl",
    ]

    exclude = set(raw_game_stats) | {
        "season", "player_id", "player_name", "team_id",
        "team_abbreviation", "game_id", "game_date", "matchup",
    }

    return [
        c for c in df.columns
        if c not in exclude
        and df[c].dtype in [np.float64, np.int64, float, int]
    ]


# ── Train one model ────────────────────────────────────────────────────────────

def _train_one(
    X_train, y_train, X_test, y_test,
    target: str,
) -> tuple:
    """Train gradient boosting + ridge baseline for a single stat target."""

    # Baseline: Ridge regression
    ridge = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
        ("model",   Ridge(alpha=1.0)),
    ])
    ridge.fit(X_train, y_train)
    ridge_pred  = ridge.predict(X_test)
    ridge_mae   = mean_absolute_error(y_test, ridge_pred)
    ridge_rmse  = root_mean_squared_error(y_test, ridge_pred)

    # Main: Gradient Boosting
    gb = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("model",   GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            loss="squared_error",
        )),
    ])
    gb.fit(X_train, y_train)
    gb_pred  = gb.predict(X_test)
    gb_mae   = mean_absolute_error(y_test, gb_pred)
    gb_rmse  = root_mean_squared_error(y_test, gb_pred)

    print(f"\n  {target.upper():>4}  Ridge → MAE: {ridge_mae:.3f} | RMSE: {ridge_rmse:.3f}")
    print(f"  {target.upper():>4}  GBM   → MAE: {gb_mae:.3f} | RMSE: {gb_rmse:.3f}")

    return gb, {"mae": gb_mae, "rmse": gb_rmse, "ridge_mae": ridge_mae, "n_test": len(y_test)}


# ── Main trainer ───────────────────────────────────────────────────────────────

def train_player_models(
    features_path: str = FEATURES_PATH,
    artifacts_dir: str = ARTIFACTS_DIR,
    test_seasons: list = TEST_SEASONS,
    targets: list      = TARGETS,
) -> tuple:
    """
    Train one regression model per stat target (pts, reb, ast).

    Returns:
        (models_dict, metrics_dict)
        models_dict: {stat: trained_pipeline}
        metrics_dict: {stat: {mae, rmse, ...}}
    """
    print("=" * 60)
    print("PLAYER PERFORMANCE PREDICTION MODELS")
    print("=" * 60)

    print("\nLoading player game features...")
    df = pd.read_csv(features_path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)
    print(f"  Total rows: {len(df):,} | Players: {df.player_id.nunique():,} | Seasons: {df.season.nunique()}")

    train_df = df[~df["season"].astype(str).isin(test_seasons)].copy()
    test_df  = df[ df["season"].astype(str).isin(test_seasons)].copy()
    print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")

    # Filter out players with too few training games
    train_counts = train_df.groupby("player_id")["game_id"].transform("count")
    train_df = train_df[train_counts >= MIN_TRAIN_GAMES]
    print(f"  Train after min-games filter ({MIN_TRAIN_GAMES}): {len(train_df):,}")

    models  = {}
    metrics = {}

    for target in targets:
        print(f"\n{'─'*40}")
        print(f"Target: {target.upper()}")

        feat_cols = get_feature_cols(df, target)

        # Drop rows where target is missing
        t_train = train_df.dropna(subset=[target])
        t_test  = test_df.dropna(subset=[target])

        X_train = t_train[feat_cols]
        y_train = t_train[target]
        X_test  = t_test[feat_cols]
        y_test  = t_test[target]

        model, m = _train_one(X_train, y_train, X_test, y_test, target)
        models[target]  = model
        metrics[target] = m

        # Feature importances
        imp = pd.Series(
            model.named_steps["model"].feature_importances_,
            index=feat_cols,
        ).sort_values(ascending=False)
        print(f"\n  Top 10 features for {target}:")
        print(imp.head(10).to_string())

        # Save model + feature list
        os.makedirs(artifacts_dir, exist_ok=True)
        with open(os.path.join(artifacts_dir, f"player_{target}_model.pkl"), "wb") as f:
            pickle.dump(model, f)
        with open(os.path.join(artifacts_dir, f"player_{target}_features.pkl"), "wb") as f:
            pickle.dump(feat_cols, f)
        imp.reset_index().rename(
            columns={"index": "feature", 0: "importance"}
        ).to_csv(os.path.join(artifacts_dir, f"player_{target}_importances.csv"), index=False)

    print(f"\n{'='*60}")
    print("SUMMARY")
    for target, m in metrics.items():
        print(f"  {target.upper():>4}: MAE = {m['mae']:.3f}  |  RMSE = {m['rmse']:.3f}  "
              f"(baseline MAE = {m['ridge_mae']:.3f})")

    return models, metrics


def predict_player_next_game(
    player_name: str,
    features_path: str = FEATURES_PATH,
    artifacts_dir: str = ARTIFACTS_DIR,
    targets: list      = TARGETS,
) -> dict:
    """
    Predict a player's next game stats using their most recent rolling features.

    Args:
        player_name: e.g. "LeBron James" (case-sensitive)

    Returns:
        dict with predicted pts, reb, ast
    """
    df = pd.read_csv(features_path)
    df["game_date"] = pd.to_datetime(df["game_date"])

    player_df = df[df["player_name"] == player_name].sort_values("game_date")
    if player_df.empty:
        return {"error": f"No data found for player: {player_name}"}

    latest = player_df.iloc[-1]
    results = {"player": player_name, "last_game": str(latest["game_date"].date())}

    for target in targets:
        feat_path  = os.path.join(artifacts_dir, f"player_{target}_features.pkl")
        model_path = os.path.join(artifacts_dir, f"player_{target}_model.pkl")

        with open(feat_path,  "rb") as f:
            feat_cols = pickle.load(f)
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        X = latest[feat_cols].to_frame().T.fillna(0)
        pred = model.predict(X)[0]
        results[f"pred_{target}"] = round(float(pred), 1)

    return results


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    models, metrics = train_player_models()
