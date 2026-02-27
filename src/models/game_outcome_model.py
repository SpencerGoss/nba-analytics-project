"""
Game Outcome Prediction Model
==============================
Trains and evaluates a gradient-boosted classifier to predict NBA game outcomes
(home team win/loss) using pre-game rolling team statistics.

Workflow:
  1. Load game_matchup_features.csv (built by team_game_features.py)
  2. Train/test split by season (time-based — no future leakage)
  3. Train a GradientBoostingClassifier (also tries Logistic Regression baseline)
  4. Evaluate and print accuracy, ROC-AUC, and a classification report
  5. Save model artifacts to models/artifacts/

Usage:
    python src/models/game_outcome_model.py

    Or import:
        from src.models.game_outcome_model import train_game_outcome_model
        model, metrics = train_game_outcome_model()
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    classification_report, confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# ── Config ─────────────────────────────────────────────────────────────────────

MATCHUP_PATH   = "data/features/game_matchup_features.csv"
ARTIFACTS_DIR  = "models/artifacts"
TARGET         = "home_win"

# Use the most recent N seasons as test set
TEST_SEASONS   = ["202324", "202425"]

# Only train on the modern era — pre-2000 basketball has different statistical
# distributions (pace, scoring, no 3-point emphasis) that add noise when
# predicting today's game. 2000-01 onward gives ~25 seasons of clean data.
TRAIN_START_SEASON = "200001"


# ── Feature selection ──────────────────────────────────────────────────────────

def get_feature_cols(df: pd.DataFrame) -> list:
    """Return all numeric home_/away_ feature columns (excludes meta and target)."""
    exclude = {TARGET, "game_id", "season", "game_date", "home_team", "away_team"}
    return [
        c for c in df.columns
        if c not in exclude
        and df[c].dtype in [np.float64, np.int64, float, int]
    ]


# ── Train / evaluate ───────────────────────────────────────────────────────────

def train_game_outcome_model(
    matchup_path: str = MATCHUP_PATH,
    artifacts_dir: str = ARTIFACTS_DIR,
    test_seasons: list = TEST_SEASONS,
) -> tuple:
    """
    Train a game outcome prediction model.

    Returns:
        (trained_pipeline, metrics_dict)
    """
    print("=" * 60)
    print("GAME OUTCOME PREDICTION MODEL")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading matchup features...")
    df = pd.read_csv(matchup_path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)
    print(f"  Total games: {len(df):,} | Seasons: {df.season.nunique()}")

    # Drop rows where the target is missing (e.g. unplayed future games)
    n_before = len(df)
    df = df.dropna(subset=[TARGET])
    if len(df) < n_before:
        print(f"  Dropped {n_before - len(df):,} rows with missing target")

    # Restrict to modern era for training — pre-2000 stats have different
    # distributions that add noise rather than signal for predicting modern games
    df = df[df["season"].astype(str) >= TRAIN_START_SEASON].copy()
    print(f"  Modern era ({TRAIN_START_SEASON}+): {len(df):,} games")

    # ── Train / test split (time-based) ───────────────────────────────────────
    train = df[~df["season"].astype(str).isin(test_seasons)].copy()
    test  = df[ df["season"].astype(str).isin(test_seasons)].copy()
    print(f"  Train: {len(train):,} games | Test: {len(test):,} games")
    print(f"  Test seasons: {test_seasons}")

    feat_cols = get_feature_cols(df)
    X_train = train[feat_cols]
    y_train = train[TARGET]
    X_test  = test[feat_cols]
    y_test  = test[TARGET]

    # ── Baseline: Logistic Regression ─────────────────────────────────────────
    print("\n--- Baseline: Logistic Regression ---")
    lr_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
        ("clf",     LogisticRegression(max_iter=1000, random_state=42)),
    ])
    lr_pipe.fit(X_train, y_train)
    lr_pred  = lr_pipe.predict(X_test)
    lr_proba = lr_pipe.predict_proba(X_test)[:, 1]
    lr_acc   = accuracy_score(y_test, lr_pred)
    lr_auc   = roc_auc_score(y_test, lr_proba)
    print(f"  Accuracy : {lr_acc:.4f}")
    print(f"  ROC-AUC  : {lr_auc:.4f}")

    # ── Main model: Gradient Boosting ─────────────────────────────────────────
    print("\n--- Main Model: Gradient Boosting Classifier ---")
    gb_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("clf",     GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )),
    ])
    gb_pipe.fit(X_train, y_train)
    gb_pred  = gb_pipe.predict(X_test)
    gb_proba = gb_pipe.predict_proba(X_test)[:, 1]
    gb_acc   = accuracy_score(y_test, gb_pred)
    gb_auc   = roc_auc_score(y_test, gb_proba)
    print(f"  Accuracy : {gb_acc:.4f}")
    print(f"  ROC-AUC  : {gb_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, gb_pred, target_names=["Away Win", "Home Win"]))

    # ── Feature importance ────────────────────────────────────────────────────
    importances = pd.Series(
        gb_pipe.named_steps["clf"].feature_importances_,
        index=feat_cols,
    ).sort_values(ascending=False)
    print("\nTop 15 Most Important Features:")
    print(importances.head(15).to_string())

    # ── Save artifacts ────────────────────────────────────────────────────────
    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(artifacts_dir, "game_outcome_model.pkl")
    feat_path  = os.path.join(artifacts_dir, "game_outcome_features.pkl")
    imp_path   = os.path.join(artifacts_dir, "game_outcome_importances.csv")

    with open(model_path, "wb") as f:
        pickle.dump(gb_pipe, f)
    with open(feat_path, "wb") as f:
        pickle.dump(feat_cols, f)
    importances.reset_index().rename(
        columns={"index": "feature", 0: "importance"}
    ).to_csv(imp_path, index=False)

    print(f"\nModel saved → {model_path}")

    metrics = {
        "lr_accuracy": lr_acc, "lr_auc": lr_auc,
        "gb_accuracy": gb_acc, "gb_auc": gb_auc,
        "n_train": len(train), "n_test": len(test),
        "test_seasons": test_seasons,
    }
    return gb_pipe, metrics


def predict_game(
    home_team_abbr: str,
    away_team_abbr: str,
    features_path: str = MATCHUP_PATH,
    artifacts_dir: str = ARTIFACTS_DIR,
) -> dict:
    """
    Quick helper: estimate win probability for a specific matchup using
    each team's most recent rolling stats.

    Args:
        home_team_abbr: e.g. "BOS"
        away_team_abbr: e.g. "LAL"

    Returns:
        dict with home_win_prob and away_win_prob
    """
    model_path = os.path.join(artifacts_dir, "game_outcome_model.pkl")
    feat_path  = os.path.join(artifacts_dir, "game_outcome_features.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(feat_path, "rb") as f:
        feat_cols = pickle.load(f)

    df = pd.read_csv(features_path)
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Get each team's latest row
    home_row = (
        df[df["home_team"] == home_team_abbr]
        .sort_values("game_date")
        .iloc[-1]
    )
    away_row = (
        df[df["away_team"] == away_team_abbr]
        .sort_values("game_date")
        .iloc[-1]
    )

    row = pd.concat([home_row, away_row], axis=0)
    row_df = row.to_frame().T[feat_cols].fillna(0)

    prob = model.predict_proba(row_df)[0]
    return {
        "home_team": home_team_abbr,
        "away_team": away_team_abbr,
        "home_win_prob": round(float(prob[1]), 4),
        "away_win_prob": round(float(prob[0]), 4),
    }


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model, metrics = train_game_outcome_model()
    print(f"\nFinal model accuracy: {metrics['gb_accuracy']:.1%}")
    print(f"Final model ROC-AUC:  {metrics['gb_auc']:.4f}")
