"""
Game Outcome Prediction Model
Train and evaluate a classifier to predict NBA game outcomes (home win/loss)
from pre-game matchup features.

Key behavior:
  1. Time-based split by season (no leakage)
  2. Candidate model selection with season-based expanding validation splits
  3. Decision-threshold tuning to maximize validation accuracy
  4. Retrain selected model on full train set, evaluate on holdout test seasons
  5. Save model artifacts + feature importances
Game Outcome Prediction Model  (v2 — improved, numpy-only)
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

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, os.path.dirname(__file__))

from numpy_gbm import (
    FastGBMClassifier, CalibratedGBMClassifier,
    remove_collinear_features, prune_by_importance,
    accuracy_score, roc_auc_score, brier_score_loss, log_loss,
)


warnings.filterwarnings("ignore")


# ── Config ─────────────────────────────────────────────────────────────────────

MATCHUP_PATH = "data/features/game_matchup_features.csv"
ARTIFACTS_DIR = "models/artifacts"
TARGET = "home_win"
TEST_SEASONS = ["202324", "202425"]
TRAIN_START_SEASON = "200001"
MIN_TRAIN_SEASONS_FOR_TUNING = 6
TEST_SEASONS        = ["202324", "202425"]
TRAIN_START_SEASON  = "200001"

CORR_THRESHOLD      = 0.92
IMP_CUMULATIVE_FRAC = 0.999


# ── Feature utilities ──────────────────────────────────────────────────────────

def get_feature_cols(df: pd.DataFrame) -> list:
    """Prefer compact matchup differential features with critical context."""
    exclude = {TARGET, "game_id", "season", "game_date", "home_team", "away_team"}
    numeric_cols = [
        c for c in df.columns
        if c not in exclude and df[c].dtype in [np.float64, np.int64, float, int]
    ]

    diff_cols = [c for c in numeric_cols if c.startswith("diff_")]
    context_cols = [
        c for c in numeric_cols
        if c in {
            "home_days_rest", "away_days_rest",
            "home_is_back_to_back", "away_is_back_to_back",
        }
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

    if diff_cols:
        return sorted(set(diff_cols + context_cols))
    return numeric_cols


# ── Helpers ────────────────────────────────────────────────────────────────────

def _best_threshold(y_true: pd.Series, proba: np.ndarray) -> tuple:
    """Find probability threshold that maximizes accuracy."""
    best_t, best_acc = 0.50, -1.0
    for t in np.arange(0.35, 0.66, 0.01):
        pred = (proba >= t).astype(int)
        acc = accuracy_score(y_true, pred)
        if acc > best_acc:
            best_acc = acc
            best_t = float(round(t, 2))
    return best_t, best_acc


def _season_splits(train_df: pd.DataFrame) -> list:
    """
    Create expanding season splits:
      train up to season i-1, validate on season i.
    """
    seasons = sorted(train_df["season"].astype(str).unique())
    splits = []
    for i in range(max(1, MIN_TRAIN_SEASONS_FOR_TUNING - 1), len(seasons)):
        train_seasons = seasons[:i]
        valid_season = seasons[i]
        tr = train_df[train_df["season"].astype(str).isin(train_seasons)].copy()
        va = train_df[train_df["season"].astype(str) == valid_season].copy()
        if not tr.empty and not va.empty:
            splits.append((tr, va, valid_season))

    # fallback when dataset has very few seasons
    if not splits:
        cutoff = int(len(train_df) * 0.85)
        tr = train_df.iloc[:cutoff].copy()
        va = train_df.iloc[cutoff:].copy()
        if not tr.empty and not va.empty:
            splits.append((tr, va, "date_fallback"))
    return splits


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
    matchup_path: str = MATCHUP_PATH,
    artifacts_dir: str = ARTIFACTS_DIR,
    test_seasons: list = TEST_SEASONS,
) -> tuple:
    """Train a game outcome model and return (pipeline, metrics)."""
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
    test = df[df["season"].astype(str).isin(test_seasons)].copy()
    print(f"  Train: {len(train):,} games | Test: {len(test):,} games")
    print(f"  Test seasons: {test_seasons}")

    feat_cols = get_feature_cols(df)
    X_train = train[feat_cols]
    y_train = train[TARGET]
    X_test = test[feat_cols]
    y_test = test[TARGET]

    candidates = {
        "logistic": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=42)),
        ]),
        "gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("clf", GradientBoostingClassifier(
                n_estimators=350,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.9,
                random_state=42,
            )),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("clf", RandomForestClassifier(
                n_estimators=700,
                max_depth=9,
                min_samples_leaf=8,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    }

    splits = _season_splits(train)
    print(f"\n--- Model selection across {len(splits)} validation split(s) ---")

    model_scores = {}
    for name, pipe in candidates.items():
        split_accs, split_aucs, split_thresholds = [], [], []

        for tr, va, split_name in splits:
            X_sub, y_sub = tr[feat_cols], tr[TARGET]
            X_val, y_val = va[feat_cols], va[TARGET]

            pipe.fit(X_sub, y_sub)
            val_proba = pipe.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_proba)
            best_t, val_acc = _best_threshold(y_val, val_proba)

            split_accs.append(val_acc)
            split_aucs.append(val_auc)
            split_thresholds.append(best_t)
            print(f"  {name:>17} | split={split_name} | acc={val_acc:.4f} | auc={val_auc:.4f} | th={best_t:.2f}")

        model_scores[name] = {
            "mean_val_acc": float(np.mean(split_accs)),
            "mean_val_auc": float(np.mean(split_aucs)),
            "threshold": float(round(np.mean(split_thresholds), 2)),
        }

    best_name = max(model_scores, key=lambda k: model_scores[k]["mean_val_acc"])
    best_threshold = model_scores[best_name]["threshold"]
    print(
        f"\nSelected model: {best_name} "
        f"(mean val acc={model_scores[best_name]['mean_val_acc']:.4f}, "
        f"mean val auc={model_scores[best_name]['mean_val_auc']:.4f}, "
        f"threshold={best_threshold:.2f})"
    )

    best_pipe = candidates[best_name]
    best_pipe.fit(X_train, y_train)
    test_proba = best_pipe.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= best_threshold).astype(int)

    test_acc = accuracy_score(y_test, test_pred)
    test_auc = roc_auc_score(y_test, test_proba)

    print(f"  Test Accuracy : {test_acc:.4f}")
    print(f"  Test ROC-AUC  : {test_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, test_pred, target_names=["Away Win", "Home Win"]))

    clf = best_pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = pd.Series(clf.feature_importances_, index=feat_cols).sort_values(ascending=False)
    else:
        importances = pd.Series(np.abs(clf.coef_[0]), index=feat_cols).sort_values(ascending=False)

    print("\nTop 15 Most Important Features:")
    print(importances.head(15).to_string())
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

    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(artifacts_dir, "game_outcome_model.pkl")
    feat_path = os.path.join(artifacts_dir, "game_outcome_features.pkl")
    imp_path = os.path.join(artifacts_dir, "game_outcome_importances.csv")

    with open(model_path, "wb") as f:
        pickle.dump(best_pipe, f)
    with open(feat_path, "wb") as f:
        pickle.dump(feat_cols, f)
        pickle.dump(model, f)
    with open(feat_path, "wb") as f:
        pickle.dump(kept_after_imp, f)

    importances.reset_index().rename(
        columns={"index": "feature", 0: "importance"}
    ).to_csv(imp_path, index=False)

    importances.reset_index().rename(columns={"index": "feature", 0: "importance"}).to_csv(imp_path, index=False)
    print(f"\nModel saved → {model_path}")
    print(f"Features used: {len(kept_after_imp)}  "
          f"(of {len(all_feat_cols)} original, "
          f"{len(all_feat_cols) - len(kept_after_imp)} removed)")

    metrics = {
        "selected_model": best_name,
        "decision_threshold": best_threshold,
        "validation_mean_accuracy": model_scores[best_name]["mean_val_acc"],
        "validation_mean_auc": model_scores[best_name]["mean_val_auc"],
        "test_accuracy": float(test_acc),
        "test_auc": float(test_auc),
        "n_train": len(train),
        "n_test": len(test),
        "test_seasons": test_seasons,
        "n_features": len(feat_cols),
    }
    return best_pipe, metrics


def predict_game(
    home_team_abbr: str,
    away_team_abbr: str,
    features_path: str = MATCHUP_PATH,
    artifacts_dir: str = ARTIFACTS_DIR,
) -> dict:
    """Estimate win probability for a matchup.

    Strategy:
      1) Use most recent exact historical pairing if available.
      2) Fallback: synthesize a matchup row using each team's most recent
         home/away context and differential columns.
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
    feat_path = os.path.join(artifacts_dir, "game_outcome_features.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(feat_path, "rb") as f:
        feat_cols = pickle.load(f)

    df = pd.read_csv(features_path)
    df["game_date"] = pd.to_datetime(df["game_date"])

    exact = df[(df["home_team"] == home_team_abbr) & (df["away_team"] == away_team_abbr)]
    if not exact.empty:
        row = exact.sort_values("game_date").iloc[-1].copy()
    else:
        home_rows = df[df["home_team"] == home_team_abbr].sort_values("game_date")
        away_rows = df[df["away_team"] == away_team_abbr].sort_values("game_date")
        if home_rows.empty or away_rows.empty:
            return {"error": f"Not enough history to build features for {home_team_abbr} vs {away_team_abbr}."}

        row = home_rows.iloc[-1].copy()
        away_source = away_rows.iloc[-1]

        for c in df.columns:
            if c.startswith("away_"):
                row[c] = away_source.get(c, row.get(c, np.nan))

        for c in feat_cols:
            if c.startswith("diff_"):
                base = c.replace("diff_", "")
                h_col, a_col = f"home_{base}", f"away_{base}"
                if h_col in row.index and a_col in row.index:
                    row[c] = row[h_col] - row[a_col]

    row_df = row.to_frame().T.reindex(columns=feat_cols).fillna(0)
    prob = model.predict_proba(row_df)[0]
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
    model, metrics = train_game_outcome_model()
    print(f"\nFinal model accuracy: {metrics['test_accuracy']:.1%}")
    print(f"Final model ROC-AUC:  {metrics['test_auc']:.4f}")
    _, metrics = train_game_outcome_model()
    print("\n" + "=" * 60)
    print("FINAL METRICS SUMMARY")
    print("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4f}")
        else:
            print(f"  {k:30s}: {v}")
