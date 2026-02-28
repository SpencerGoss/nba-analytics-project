"""
Game Outcome Prediction Model
=============================
Train and evaluate a classifier to predict NBA game outcomes (home win/loss)
from pre-game matchup features.

Key behavior:
  1. Time-based split by season (no leakage)
  2. Candidate model selection with season-based expanding validation splits
  3. Decision-threshold tuning to maximize validation accuracy
  4. Retrain selected model on full train set, evaluate on holdout test seasons
  5. Save model artifacts + feature importances
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

warnings.filterwarnings("ignore")


# ── Config ─────────────────────────────────────────────────────────────────────

MATCHUP_PATH = "data/features/game_matchup_features.csv"
ARTIFACTS_DIR = "models/artifacts"
TARGET = "home_win"
TEST_SEASONS = ["202324", "202425"]
TRAIN_START_SEASON = "200001"
MIN_TRAIN_SEASONS_FOR_TUNING = 6


# ── Feature selection ──────────────────────────────────────────────────────────

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


# ── Train / evaluate ───────────────────────────────────────────────────────────

def train_game_outcome_model(
    matchup_path: str = MATCHUP_PATH,
    artifacts_dir: str = ARTIFACTS_DIR,
    test_seasons: list = TEST_SEASONS,
) -> tuple:
    """Train a game outcome model and return (pipeline, metrics)."""
    print("=" * 60)
    print("GAME OUTCOME PREDICTION MODEL")
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

    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(artifacts_dir, "game_outcome_model.pkl")
    feat_path = os.path.join(artifacts_dir, "game_outcome_features.pkl")
    imp_path = os.path.join(artifacts_dir, "game_outcome_importances.csv")

    with open(model_path, "wb") as f:
        pickle.dump(best_pipe, f)
    with open(feat_path, "wb") as f:
        pickle.dump(feat_cols, f)

    importances.reset_index().rename(columns={"index": "feature", 0: "importance"}).to_csv(imp_path, index=False)
    print(f"\nModel saved → {model_path}")

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
    return {
        "home_team": home_team_abbr,
        "away_team": away_team_abbr,
        "home_win_prob": round(float(prob[1]), 4),
        "away_win_prob": round(float(prob[0]), 4),
    }


if __name__ == "__main__":
    model, metrics = train_game_outcome_model()
    print(f"\nFinal model accuracy: {metrics['test_accuracy']:.1%}")
    print(f"Final model ROC-AUC:  {metrics['test_auc']:.4f}")
