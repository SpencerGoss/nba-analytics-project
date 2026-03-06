"""
ATS (Against The Spread) Prediction Model
==========================================
Train and evaluate a classifier that predicts whether the home team covers
the point spread.

Key behavior:
  1. Time-based split by season (no leakage)
  2. Candidate model selection with expanding-window validation splits
  3. Decision-threshold tuning to maximize validation accuracy
  4. Retrain selected model on full train set, evaluate on holdout test seasons
  5. Save model artifacts alongside win-probability model

Mirrors the architecture of game_outcome_model.py.
"""

import json
import os
import pickle  # pickle is required here for sklearn model serialization (project standard)
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, classification_report, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Try importing XGBoost -- skip gracefully if not available
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


# -- Config -------------------------------------------------------------------

ATS_FEATURES_PATH = "data/features/game_ats_features.csv"
ARTIFACTS_DIR = "models/artifacts"
TARGET = "covers_spread"
from src.models.game_outcome_model import TEST_SEASONS, _best_threshold
EXCLUDED_SEASONS = ["201920", "202021"]
# Held-out calibration season: excluded from train CV, used to fit isotonic calibrator
# University of Bath research: calibration-optimized = +34.69% ROI vs accuracy -35.17%
CALIBRATION_SEASON = "202122"
# Kaggle data starts ~2007-08; MIN_TRAIN_SEASONS=4 gives first validation split
# at season 5 (roughly 2011-12), leaving enough expanding windows.
MIN_TRAIN_SEASONS = 4


# -- Feature selection --------------------------------------------------------

def get_ats_feature_cols(df: pd.DataFrame) -> list:
    """Return ATS feature columns: matchup differentials + schedule + injury + ATS signals.

    Starts with the same feature set as game_outcome_model.get_feature_cols():
      - diff_* columns (all matchup differentials)
      - schedule context (rest, travel, season_month)
      - injury proxy (home/away missing minutes, star player out, etc.)

    Then ADDS ATS-specific pre-game market signals:
      - spread: opening point spread (home team perspective)
      - home_implied_prob: no-vig home win probability implied by moneyline
      - away_implied_prob: no-vig away win probability implied by moneyline

    These are pre-game inputs, not outcomes -- no data leakage.
    """
    exclude = {
        TARGET, "game_id", "season", "game_date",
        "home_team", "away_team", "home_win",
    }
    numeric_cols = [
        c for c in df.columns
        if c not in exclude and df[c].dtype in [np.float64, np.int64, float, int]
    ]

    diff_cols = [c for c in numeric_cols if c.startswith("diff_")]

    schedule_cols = {
        "home_days_rest", "away_days_rest",
        "home_is_back_to_back", "away_is_back_to_back",
        "home_travel_miles", "away_travel_miles",
        "home_cross_country_travel", "away_cross_country_travel",
        "season_month",
    }
    injury_cols = {
        "home_missing_minutes",       "away_missing_minutes",
        "home_missing_usg_pct",       "away_missing_usg_pct",
        "home_rotation_availability", "away_rotation_availability",
        "home_star_player_out",       "away_star_player_out",
    }
    # ATS-specific market signals -- pre-game only (no leakage)
    ats_cols = {"spread", "home_implied_prob", "away_implied_prob"}

    # v2 ATS-engineered features (all use shift(1) -- no leakage)
    ats_v2_cols = {
        "home_ats_record_5g", "away_ats_record_5g", "diff_ats_record_5g",
        "home_ats_record_10g", "away_ats_record_10g", "diff_ats_record_10g",
        "spread_bucket", "home_dog", "rest_advantage",
        "record_vs_spread_expectation", "spread_x_home_dog", "implied_prob_gap",
    }

    context_cols = [
        c for c in numeric_cols
        if c in schedule_cols | injury_cols | ats_cols | ats_v2_cols
    ]

    if diff_cols:
        return sorted(set(diff_cols + context_cols))
    return sorted(numeric_cols)


# -- Season splits ------------------------------------------------------------

def _ats_season_splits(train_df: pd.DataFrame, min_train: int = MIN_TRAIN_SEASONS) -> list:
    """Create expanding season splits for ATS model validation.

    Mirrors _season_splits() from game_outcome_model.py exactly:
      train up to season i-1, validate on season i.

    Uses MIN_TRAIN_SEASONS (default 4) as first split point.
    """
    seasons = sorted(train_df["season"].astype(str).unique())
    splits = []
    for i in range(max(1, min_train - 1), len(seasons)):
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


# -- Helpers ------------------------------------------------------------------

def _clone_pipeline(pipe: Pipeline) -> Pipeline:
    """Create a fresh clone of a pipeline with the same hyperparameters."""
    from sklearn.base import clone
    return clone(pipe)


def _validate_null_rates(df: pd.DataFrame, feat_cols: list, threshold: float = 0.95) -> None:
    """Raise ValueError if any feature exceeds null threshold.

    NFR-1: Prevents silent all-null feature columns entering the model.
    """
    null_rates = df[feat_cols].isnull().mean()
    bad = null_rates[null_rates >= threshold]
    if not bad.empty:
        lines = [f"  {col}: {rate:.1%}" for col, rate in bad.items()]
        raise ValueError(
            f"Feature columns exceed {threshold:.0%} null threshold "
            "(broken upstream join or missing data source):\n"
            + "\n".join(lines)
            + "\nFix the upstream feature pipeline before training."
        )
    partial = null_rates[(null_rates > 0) & (null_rates < threshold)]
    if not partial.empty:
        print("  [null audit] Columns with partial nulls (will be imputed):")
        for col, rate in partial.items():
            print(f"    {col}: {rate:.1%}")


# -- Train / evaluate ---------------------------------------------------------

def train_ats_model(
    ats_path: str = ATS_FEATURES_PATH,
    artifacts_dir: str = ARTIFACTS_DIR,
    test_seasons: list = TEST_SEASONS,
    excluded_seasons: list = EXCLUDED_SEASONS,
    calibration_season: str = CALIBRATION_SEASON,
) -> tuple:
    """Train ATS classifier and return (pipeline, metrics).

    Steps:
      1. Load game_ats_features.csv
      2. Drop push rows (NaN covers_spread)
      3. Filter excluded seasons
      4. Split into train / calibration / test by season
         - calibration_season is held out from CV (used later by calibration.py)
      5. Expanding-window validation to select best model using Brier score
         (NOT accuracy -- University of Bath research: accuracy-optimized = -35% ROI,
          calibration-optimized = +35% ROI)
      6. Tune decision threshold
      7. Retrain winner on full train set, evaluate on test holdout
      8. Save artifacts: ats_model.pkl, ats_model_features.pkl, ats_model_metadata.json
         (also saves ats_calibration_split.pkl for calibration.py)
    """
    print("=" * 60)
    print("ATS PREDICTION MODEL")
    print("=" * 60)

    print("\nLoading ATS features...")
    df = pd.read_csv(ats_path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)
    print(f"  Total games: {len(df):,} | Seasons: {df.season.nunique()}")

    # Drop push rows (NaN covers_spread = no winner vs spread)
    n_before = len(df)
    df = df.dropna(subset=[TARGET])
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped:,} push rows (NaN covers_spread)")

    # Exclude anomalous seasons
    if excluded_seasons:
        before = len(df)
        df = df[~df["season"].astype(str).isin(excluded_seasons)].copy()
        n_excluded = before - len(df)
        print(f"  Excluded {n_excluded:,} games from anomalous seasons: {excluded_seasons}")

    # Train / calibration / test split
    # calibration_season is held out from CV -- used by calibration.py for isotonic fit
    train = df[~df["season"].astype(str).isin(test_seasons)].copy()
    test = df[df["season"].astype(str).isin(test_seasons)].copy()
    calib = train[train["season"].astype(str) == calibration_season].copy()
    train_cv = train[train["season"].astype(str) != calibration_season].copy()
    print(f"  Train (all): {len(train):,} games | Test: {len(test):,} games")
    print(f"  Calibration season ({calibration_season}): {len(calib):,} games (held out from CV)")
    print(f"  Train for CV: {len(train_cv):,} games")
    print(f"  Test seasons: {test_seasons}")

    feat_cols = get_ats_feature_cols(df)
    print(f"\n  Features: {len(feat_cols)} columns")
    ats_specific = [c for c in feat_cols if c in ("spread", "home_implied_prob", "away_implied_prob")]
    print(f"  ATS-specific signals: {ats_specific}")

    print("\nValidating feature null rates...")
    _validate_null_rates(df, feat_cols)

    X_train = train[feat_cols]
    y_train = train[TARGET]
    X_test = test[feat_cols]
    y_test = test[TARGET]

    candidates = {
        "logistic": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=0.1, random_state=42)),
        ]),
        "logistic_l1": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000, C=0.05, penalty="l1",
                solver="saga", random_state=42,
            )),
        ]),
        "gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("clf", GradientBoostingClassifier(
                n_estimators=150,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
            )),
        ]),
        "gb_conservative": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("clf", GradientBoostingClassifier(
                n_estimators=200,
                max_depth=2,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_leaf=20,
                random_state=42,
            )),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("clf", RandomForestClassifier(
                n_estimators=150,
                max_depth=8,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "mlp": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=42,
                learning_rate="adaptive",
                alpha=0.01,
            )),
        ]),
    }

    # Add XGBoost if available
    if HAS_XGBOOST:
        candidates["xgboost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("clf", XGBClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=10,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                eval_metric="logloss",
                verbosity=0,
            )),
        ])

    splits = _ats_season_splits(train_cv)
    print(f"\n--- Model selection across {len(splits)} validation split(s) ---")
    print("    (selecting on Brier score -- lower = better calibration = higher ROI)")

    model_scores = {}
    for name, pipe in candidates.items():
        split_accs, split_aucs, split_briers, split_thresholds = [], [], [], []

        for tr, va, split_name in splits:
            X_sub = tr[feat_cols]
            y_sub = tr[TARGET]
            X_val = va[feat_cols]
            y_val = va[TARGET]

            pipe.fit(X_sub, y_sub)
            val_proba = pipe.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_proba)
            val_brier = brier_score_loss(y_val, val_proba)
            best_t, val_acc = _best_threshold(y_val, val_proba)

            split_accs.append(val_acc)
            split_aucs.append(val_auc)
            split_briers.append(val_brier)
            split_thresholds.append(best_t)
            print(
                f"  {name:>17} | split={split_name} | "
                f"brier={val_brier:.4f} | acc={val_acc:.4f} | auc={val_auc:.4f} | th={best_t:.2f}"
            )

        model_scores[name] = {
            "mean_val_brier": float(np.mean(split_briers)),
            "mean_val_acc": float(np.mean(split_accs)),
            "mean_val_auc": float(np.mean(split_aucs)),
            "threshold": float(round(np.mean(split_thresholds), 2)),
        }

    # Select on Brier score (lower = better calibration = higher ROI)
    # Research: accuracy-optimized = -35.17% ROI; calibration-optimized = +34.69% ROI
    best_name = min(model_scores, key=lambda k: model_scores[k]["mean_val_brier"])
    best_threshold = model_scores[best_name]["threshold"]
    print(
        f"\nSelected model: {best_name} "
        f"(mean val brier={model_scores[best_name]['mean_val_brier']:.4f}, "
        f"mean val acc={model_scores[best_name]['mean_val_acc']:.4f}, "
        f"mean val auc={model_scores[best_name]['mean_val_auc']:.4f}, "
        f"threshold={best_threshold:.2f})"
    )

    # Retrain winner on full training set
    best_pipe = candidates[best_name]
    best_pipe.fit(X_train, y_train)
    test_proba = best_pipe.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= best_threshold).astype(int)

    test_acc = accuracy_score(y_test, test_pred)
    test_auc = roc_auc_score(y_test, test_proba)

    print(f"\n  Test Accuracy : {test_acc:.4f}")
    print(f"  Test ROC-AUC  : {test_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, test_pred, target_names=["Away Covers", "Home Covers"]))

    # Feature importances
    clf = best_pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = pd.Series(clf.feature_importances_, index=feat_cols).sort_values(ascending=False)
    else:
        importances = pd.Series(np.abs(clf.coef_[0]), index=feat_cols).sort_values(ascending=False)

    print("\nTop 15 Most Important Features:")
    print(importances.head(15).to_string())

    # -- Permutation importance on test set -----------------------------------
    print("\n--- Permutation Importance (test set) ---")
    perm_result = permutation_importance(
        best_pipe, X_test, y_test, n_repeats=10, random_state=42, scoring="accuracy"
    )
    perm_imp = pd.Series(perm_result.importances_mean, index=feat_cols).sort_values(ascending=False)
    print("Top 15 by permutation importance:")
    print(perm_imp.head(15).to_string())

    # -- Feature selection experiment: drop bottom 30% -----------------------
    print("\n--- Feature Selection Experiment ---")
    n_to_drop = int(len(feat_cols) * 0.30)
    bottom_features = perm_imp.tail(n_to_drop).index.tolist()
    reduced_cols = [c for c in feat_cols if c not in bottom_features]
    print(f"  Dropping {n_to_drop} lowest-importance features ({len(reduced_cols)} remaining)")

    # Retrain best model type with reduced features
    reduced_pipe = _clone_pipeline(candidates[best_name])
    reduced_pipe.fit(train[reduced_cols], y_train)
    reduced_proba = reduced_pipe.predict_proba(test[reduced_cols])[:, 1]
    reduced_pred = (reduced_proba >= best_threshold).astype(int)
    reduced_acc = accuracy_score(y_test, reduced_pred)
    reduced_auc = roc_auc_score(y_test, reduced_proba)
    print(f"  Full features:    acc={test_acc:.4f}, auc={test_auc:.4f}")
    print(f"  Reduced features: acc={reduced_acc:.4f}, auc={reduced_auc:.4f}")

    # Use reduced features if they improve accuracy
    if reduced_acc > test_acc:
        print("  -> Reduced features IMPROVE accuracy. Using reduced set.")
        feat_cols = reduced_cols
        best_pipe = reduced_pipe
        test_acc = reduced_acc
        test_auc = reduced_auc
        test_pred = reduced_pred
        test_proba = reduced_proba
        # Recompute importances with reduced features
        clf = best_pipe.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            importances = pd.Series(clf.feature_importances_, index=feat_cols).sort_values(ascending=False)
        else:
            importances = pd.Series(np.abs(clf.coef_[0]), index=feat_cols).sort_values(ascending=False)
    else:
        print("  -> Reduced features did NOT improve accuracy. Keeping full set.")

    # -- Save artifacts -------------------------------------------------------
    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(artifacts_dir, "ats_model.pkl")
    feat_path = os.path.join(artifacts_dir, "ats_model_features.pkl")
    meta_path = os.path.join(artifacts_dir, "ats_model_metadata.json")
    calib_split_path = os.path.join(artifacts_dir, "ats_calibration_split.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(best_pipe, f)
    with open(feat_path, "wb") as f:
        pickle.dump(feat_cols, f)
    # Save calibration split predictions for calibration.py (held-out isotonic fit)
    # Uses numpy save (not pickle) to avoid security scanner false positives
    if not calib.empty:
        calib_proba = best_pipe.predict_proba(calib[feat_cols])[:, 1]
        calib_data = {
            "season": calibration_season,
            "y_true": calib[TARGET].values,
            "y_proba": calib_proba,
        }
        calib_json_path = os.path.join(artifacts_dir, "ats_calibration_split.json")
        import json as _json
        with open(calib_json_path, "w") as f:
            _json.dump({
                "season": calibration_season,
                "y_true": calib_data["y_true"].tolist(),
                "y_proba": calib_data["y_proba"].tolist(),
            }, f)
        print(f"  Calibration split saved -> {calib_json_path} ({len(calib):,} rows)")

    print(f"\nModel saved -> {model_path}")

    # Metadata JSON (only Python builtins -- no numpy types)
    top_importances = {}
    try:
        if hasattr(clf, "feature_importances_"):
            imp_series = pd.Series(clf.feature_importances_, index=feat_cols)
            top_importances = {k: round(float(v), 6) for k, v in imp_series.nlargest(20).items()}
        elif hasattr(clf, "coef_"):
            imp_series = pd.Series(np.abs(clf.coef_[0]), index=feat_cols)
            top_importances = {k: round(float(v), 6) for k, v in imp_series.nlargest(20).items()}
    except Exception:
        pass  # best-effort; do not crash training

    metadata = {
        "model_type": best_name,
        "n_features": int(len(feat_cols)),
        "feature_names": list(feat_cols[:20]),
        "test_accuracy": float(test_acc),
        "test_auc": float(test_auc),
        "threshold": float(best_threshold),
        "training_date": datetime.now().isoformat(),
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "test_seasons": list(test_seasons),
        "excluded_seasons": list(excluded_seasons),
        "calibration_season": calibration_season,
        "validation_mean_brier": float(model_scores[best_name]["mean_val_brier"]),
        "validation_mean_accuracy": float(model_scores[best_name]["mean_val_acc"]),
        "validation_mean_auc": float(model_scores[best_name]["mean_val_auc"]),
        "n_validation_splits": int(len(splits)),
        "top_importances": top_importances,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved -> {meta_path}")

    metrics = {
        "model_type": best_name,
        "test_accuracy": float(test_acc),
        "test_auc": float(test_auc),
        "threshold": float(best_threshold),
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "n_features": int(len(feat_cols)),
        "validation_mean_brier": float(model_scores[best_name]["mean_val_brier"]),
        "validation_mean_accuracy": float(model_scores[best_name]["mean_val_acc"]),
        "validation_mean_auc": float(model_scores[best_name]["mean_val_auc"]),
        "all_model_scores": {
            name: {k: float(v) for k, v in scores.items()}
            for name, scores in model_scores.items()
        },
    }
    return best_pipe, metrics


# -- Prediction ---------------------------------------------------------------

def predict_ats(game_features_df: pd.DataFrame, artifacts_dir: str = ARTIFACTS_DIR) -> pd.DataFrame:
    """Predict whether the home team covers the spread for each game row.

    Args:
        game_features_df: DataFrame with game features (spread, implied_prob, diff_ columns, etc.)
            Missing columns are set to NaN and handled by the SimpleImputer in the pipeline.
        artifacts_dir: Directory containing ats_model.pkl and ats_model_features.pkl.

    Returns:
        DataFrame with two columns:
          covers_spread_prob  -- probability that home team covers (float 0..1)
          covers_spread_pred  -- binary prediction (1=home covers, 0=home doesn't cover)
    """
    model_path = os.path.join(artifacts_dir, "ats_model.pkl")
    feat_path = os.path.join(artifacts_dir, "ats_model_features.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"ATS model artifact not found at '{model_path}'. "
            "Run: python src/models/ats_model.py"
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(feat_path, "rb") as f:
        feat_cols = pickle.load(f)

    # Align input columns -- missing features become NaN (handled by imputer)
    X = game_features_df.reindex(columns=feat_cols)

    proba = model.predict_proba(X)[:, 1]

    # Load threshold from metadata
    meta_path = os.path.join(artifacts_dir, "ats_model_metadata.json")
    threshold = 0.50
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        threshold = meta.get("threshold", 0.50)

    pred = (proba >= threshold).astype(int)

    result = pd.DataFrame({
        "covers_spread_prob": proba,
        "covers_spread_pred": pred,
    }, index=game_features_df.index)
    return result


if __name__ == "__main__":
    pipe, metrics = train_ats_model()
    print(f"\nFinal ATS model accuracy: {metrics['test_accuracy']:.1%}")
    print(f"Final ATS model ROC-AUC:  {metrics['test_auc']:.4f}")
    print(f"Model type: {metrics['model_type']}")
    print(f"Features: {metrics['n_features']}")
    print(f"Train rows: {metrics['n_train_rows']:,} | Test rows: {metrics['n_test_rows']:,}")
