"""
Game Outcome Prediction Model  (v2 — improved, numpy-only)
==========================================================
Trains and evaluates a gradient-boosted classifier to predict NBA game
outcomes (home team win/loss) using pre-game rolling team statistics.

Key improvements over v1:
  • Collinearity removal  — drops features whose |Pearson r| with another
    feature exceeds 0.92, keeping the one more correlated with the target.
  • Importance pruning    — two-pass training; prune near-zero-importance
    features before the final 600-tree pass.
  • Better hyperparameters — shallower trees (max_depth=3) with more
    estimators and stronger leaf regularisation (min_samples_leaf=20).
  • Calibrated probabilities — Platt scaling so win-probability outputs
    are well-calibrated, not just rank-ordered.
  • Pure numpy — no scikit-learn dependency.

Usage:
    python src/models/game_outcome_model.py
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, os.path.dirname(__file__))

from numpy_gbm import (
    FastGBMClassifier, CalibratedGBMClassifier,
    remove_collinear_features, prune_by_importance,
    accuracy_score, roc_auc_score, brier_score_loss, log_loss,
)


# ── Config ─────────────────────────────────────────────────────────────────────

MATCHUP_PATH   = "data/features/game_matchup_features.csv"
ARTIFACTS_DIR  = "models/artifacts"
TARGET         = "home_win"

TEST_SEASONS        = ["202324", "202425"]
TRAIN_START_SEASON  = "200001"

CORR_THRESHOLD      = 0.92
IMP_CUMULATIVE_FRAC = 0.999


# ── Feature utilities ──────────────────────────────────────────────────────────

def get_feature_cols(df):
    """Return all numeric non-meta feature columns."""
    exclude = {
        TARGET, "game_id", "season", "game_date",
        "home_team", "away_team", "home_win",
    }
    return [
        c for c in df.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(df[c])
    ]


# ── Logistic regression baseline (pure numpy) ─────────────────────────────────

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


class _SimpleLogisticRegression:
    """Mini logistic regression via gradient descent (no sklearn needed)."""
    def __init__(self, C=0.1, max_iter=500, lr=0.01):
        self.C, self.max_iter, self.lr = C, max_iter, lr
        self.w_, self.b_ = None, None

    def _impute_scale(self, X, fit=False):
        if fit:
            self._med = np.nanmedian(X, axis=0)
            Xc = np.where(np.isnan(X), self._med, X)
            self._std = np.std(Xc, axis=0)
            self._std[self._std == 0] = 1.0
            self._mn  = np.mean(Xc, axis=0)
        Xc = np.where(np.isnan(X), self._med, X)
        return (Xc - self._mn) / self._std

    def fit(self, X, y):
        Xs = self._impute_scale(X, fit=True)
        n, p = Xs.shape
        self.w_ = np.zeros(p)
        self.b_ = 0.0
        for _ in range(self.max_iter):
            logits = Xs @ self.w_ + self.b_
            proba  = _sigmoid(logits)
            err    = proba - y
            grad_w = Xs.T @ err / n + self.w_ / (self.C * n)
            grad_b = err.mean()
            self.w_ -= self.lr * grad_w
            self.b_ -= self.lr * grad_b
        return self

    def predict_proba(self, X):
        Xs = self._impute_scale(X)
        p  = _sigmoid(Xs @ self.w_ + self.b_)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ── Train / evaluate ───────────────────────────────────────────────────────────

def train_game_outcome_model(
    matchup_path=MATCHUP_PATH,
    artifacts_dir=ARTIFACTS_DIR,
    test_seasons=TEST_SEASONS,
):
    """
    Train an improved game outcome prediction model.

    Returns:
        (calibrated_model, metrics_dict)
    """
    print("=" * 60)
    print("GAME OUTCOME PREDICTION MODEL  (v2 — improved, numpy-only)")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading matchup features...")
    df = pd.read_csv(matchup_path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)
    print(f"  Total games: {len(df):,} | Seasons: {df.season.nunique()}")

    n_before = len(df)
    df = df.dropna(subset=[TARGET])
    if len(df) < n_before:
        print(f"  Dropped {n_before - len(df):,} rows with missing target")

    df = df[df["season"].astype(str) >= TRAIN_START_SEASON].copy()
    print(f"  Modern era ({TRAIN_START_SEASON}+): {len(df):,} games")

    # ── Time-based split ──────────────────────────────────────────────────────
    train = df[~df["season"].astype(str).isin(test_seasons)].copy()
    test  = df[ df["season"].astype(str).isin(test_seasons)].copy()
    print(f"  Train: {len(train):,} games | Test: {len(test):,} games")
    print(f"  Test seasons: {test_seasons}")

    all_feat_cols = get_feature_cols(df)
    print(f"\n  Starting features: {len(all_feat_cols)}")

    X_train_full = train[all_feat_cols].values.astype(float)
    y_train      = train[TARGET].values.astype(float)
    X_test_full  = test[all_feat_cols].values.astype(float)
    y_test       = test[TARGET].values.astype(float)

    # ── Step 1: Remove collinear features ─────────────────────────────────────
    print("\n--- Step 1: Collinearity Removal ---")
    kept_after_corr = remove_collinear_features(
        train[all_feat_cols], y_train, threshold=CORR_THRESHOLD
    )
    dropped_corr = len(all_feat_cols) - len(kept_after_corr)
    print(f"  Dropped {dropped_corr} collinear features (|r| > {CORR_THRESHOLD})")
    print(f"  Features remaining: {len(kept_after_corr)}")

    X_train_corr = train[kept_after_corr].values.astype(float)
    X_test_corr  = test[kept_after_corr].values.astype(float)

    # ── Baseline: Logistic Regression ─────────────────────────────────────────
    print("\n--- Baseline: Logistic Regression ---")
    lr = _SimpleLogisticRegression(C=0.1, max_iter=500, lr=0.01)
    lr.fit(X_train_corr, y_train)
    lr_proba = lr.predict_proba(X_test_corr)[:, 1]
    lr_pred  = lr.predict(X_test_corr)
    lr_acc   = accuracy_score(y_test, lr_pred)
    lr_auc   = roc_auc_score(y_test, lr_proba)
    lr_brier = brier_score_loss(y_test, lr_proba)
    print(f"  Accuracy : {lr_acc:.4f}")
    print(f"  ROC-AUC  : {lr_auc:.4f}")
    print(f"  Brier    : {lr_brier:.4f}")

    # ── Step 2: Pass 1 — initial GBM fit for importances ──────────────────────
    print("\n--- Step 2: Initial GBM Fit (200 trees, for importance pruning) ---")
    gb_pass1 = FastGBMClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3,
        min_samples_leaf=20, subsample=0.75, max_features=0.5,
        random_state=42,
    )
    gb_pass1.fit(X_train_corr, y_train)
    print("  Pass-1 done.")

    # ── Step 3: Prune low-importance features ─────────────────────────────────
    print("\n--- Step 3: Importance Pruning ---")
    kept_after_imp = prune_by_importance(
        gb_pass1.feature_importances_,
        kept_after_corr,
        cumulative_frac=IMP_CUMULATIVE_FRAC,
        min_features=10,
    )
    dropped_imp = len(kept_after_corr) - len(kept_after_imp)
    print(f"  Dropped {dropped_imp} near-zero-importance features")
    print(f"  Features remaining: {len(kept_after_imp)}")

    feat_idx_final = [kept_after_corr.index(f) for f in kept_after_imp]
    X_train_final = X_train_corr[:, feat_idx_final]
    X_test_final  = X_test_corr[:,  feat_idx_final]

    # ── Step 4: Pass 2 — full GBM fit on pruned features ─────────────────────
    print("\n--- Step 4: Final GBM Fit (600 trees, pruned features) ---")
    model = CalibratedGBMClassifier(
        n_estimators=600, learning_rate=0.04, max_depth=3,
        min_samples_leaf=20, subsample=0.75, max_features=0.5,
        random_state=42,
    )
    # Use last 20% of training data as calibration set
    n_cal  = max(500, int(len(X_train_final) * 0.20))
    X_cal  = X_train_final[-n_cal:]
    y_cal  = y_train[-n_cal:]
    X_tr2  = X_train_final[:-n_cal]
    y_tr2  = y_train[:-n_cal]
    print(f"  GBM train: {len(X_tr2):,}  |  calibration: {len(X_cal):,}")

    model.fit(X_tr2, y_tr2, X_cal=X_cal, y_cal=y_cal)
    print("  Pass-2 + calibration done.")

    # Evaluate
    cal_proba = model.predict_proba(X_test_final)[:, 1]
    cal_pred  = model.predict(X_test_final)
    cal_acc   = accuracy_score(y_test, cal_pred)
    cal_auc   = roc_auc_score(y_test, cal_proba)
    cal_brier = brier_score_loss(y_test, cal_proba)
    cal_ll    = log_loss(y_test, cal_proba)

    # Uncalibrated scores for comparison
    raw_proba = model.estimator.predict_proba(X_test_final)[:, 1]
    raw_pred  = model.estimator.predict(X_test_final)
    gb_acc    = accuracy_score(y_test, raw_pred)
    gb_auc    = roc_auc_score(y_test, raw_proba)
    gb_brier  = brier_score_loss(y_test, raw_proba)
    gb_ll     = log_loss(y_test, raw_proba)

    print(f"\n  Raw GBM  → Accuracy: {gb_acc:.4f}  AUC: {gb_auc:.4f}  "
          f"Brier: {gb_brier:.4f}  LogLoss: {gb_ll:.4f}")
    print(f"  Calibrated → Accuracy: {cal_acc:.4f}  AUC: {cal_auc:.4f}  "
          f"Brier: {cal_brier:.4f}  LogLoss: {cal_ll:.4f}")

    # Confusion matrix summary
    tp = int(((cal_pred == 1) & (y_test == 1)).sum())
    tn = int(((cal_pred == 0) & (y_test == 0)).sum())
    fp = int(((cal_pred == 1) & (y_test == 0)).sum())
    fn = int(((cal_pred == 0) & (y_test == 1)).sum())
    print(f"\n  Confusion Matrix (Calibrated):")
    print(f"    TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    print(f"    Precision={prec:.4f}  Recall={rec:.4f}")

    # Feature importances
    importances = pd.Series(
        model.feature_importances_,
        index=kept_after_imp,
    ).sort_values(ascending=False)

    print("\nTop 20 Most Important Features:")
    print(importances.head(20).to_string())

    # ── Save artifacts ────────────────────────────────────────────────────────
    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(artifacts_dir, "game_outcome_model.pkl")
    feat_path  = os.path.join(artifacts_dir, "game_outcome_features.pkl")
    imp_path   = os.path.join(artifacts_dir, "game_outcome_importances.csv")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(feat_path, "wb") as f:
        pickle.dump(kept_after_imp, f)

    importances.reset_index().rename(
        columns={"index": "feature", 0: "importance"}
    ).to_csv(imp_path, index=False)

    print(f"\nModel saved → {model_path}")
    print(f"Features used: {len(kept_after_imp)}  "
          f"(of {len(all_feat_cols)} original, "
          f"{len(all_feat_cols) - len(kept_after_imp)} removed)")

    metrics = {
        "lr_accuracy"  : lr_acc,    "lr_auc"     : lr_auc,
        "gb_accuracy"  : gb_acc,    "gb_auc"     : gb_auc,
        "gb_brier"     : gb_brier,  "gb_logloss" : gb_ll,
        "cal_accuracy" : cal_acc,   "cal_auc"    : cal_auc,
        "cal_brier"    : cal_brier, "cal_logloss": cal_ll,
        "n_train"      : len(train), "n_test"    : len(test),
        "n_features_original" : len(all_feat_cols),
        "n_features_final"    : len(kept_after_imp),
        "test_seasons" : test_seasons,
    }
    return model, metrics


def predict_game(
    home_team_abbr,
    away_team_abbr,
    features_path=MATCHUP_PATH,
    artifacts_dir=ARTIFACTS_DIR,
):
    """
    Estimate win probability for a specific matchup using each team's
    most recent rolling stats.
    """
    model_path = os.path.join(artifacts_dir, "game_outcome_model.pkl")
    feat_path  = os.path.join(artifacts_dir, "game_outcome_features.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(feat_path, "rb") as f:
        feat_cols = pickle.load(f)

    df = pd.read_csv(features_path)
    df["game_date"] = pd.to_datetime(df["game_date"])

    home_rows = df[df["home_team"] == home_team_abbr].sort_values("game_date")
    away_rows = df[df["away_team"] == away_team_abbr].sort_values("game_date")

    if home_rows.empty or away_rows.empty:
        return {"error": "Team not found in features"}

    latest_row = home_rows.iloc[[-1]].copy()
    for col in feat_cols:
        if col.startswith("away_") and col in away_rows.columns:
            latest_row[col] = away_rows.iloc[-1][col]

    X = latest_row[feat_cols].values.astype(float)
    proba = model.predict_proba(X)[0, 1]

    return {
        "home_team"     : home_team_abbr,
        "away_team"     : away_team_abbr,
        "home_win_prob" : round(float(proba), 4),
        "away_win_prob" : round(1.0 - float(proba), 4),
    }


if __name__ == "__main__":
    _, metrics = train_game_outcome_model()
    print("\n" + "=" * 60)
    print("FINAL METRICS SUMMARY")
    print("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4f}")
        else:
            print(f"  {k:30s}: {v}")
