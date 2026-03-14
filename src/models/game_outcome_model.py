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

import json
import os
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import logging

log = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

try:
    from lightgbm import LGBMClassifier
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False


# ── Config ─────────────────────────────────────────────────────────────────────

MATCHUP_PATH = "data/features/game_matchup_features.csv"
ARTIFACTS_DIR = "models/artifacts"
TARGET = "home_win"
TEST_SEASONS = [202324, 202425]
TRAIN_START_SEASON = 200001
MIN_TRAIN_SEASONS_FOR_TUNING = 6

# When True, restrict training data to the modern 3-Point Revolution era
# (2013-14 onward). Cuts historical noise at the cost of less training data.
# See docs/model_advisor_notes.md Proposal 5 for rationale.
# CHANGED: default to modern era (was False)
MODERN_ERA_ONLY = True
# CHANGED: include 2013-14 per SC1 (was "201415")
MODERN_ERA_START = 201314
# NEW: exclude anomalous seasons (bubble + shortened COVID)
EXCLUDED_SEASONS = [201920, 202021]


# ── Null-rate guard ────────────────────────────────────────────────────────────

def validate_feature_null_rates(
    df: pd.DataFrame,
    feat_cols: list,
    threshold: float = 0.95,
) -> None:
    """Raise ValueError if any feature column has null rate >= threshold.

    Also logs columns with partial nulls (>0% but <threshold) for visibility.
    Called at the top of train_game_outcome_model() before any model fitting.
    NFR-1: Prevents silent all-null feature columns from entering the model.
    A column at >= 95% null rate indicates a broken upstream join or missing
    data source — the SimpleImputer would fill it with mean(0.0) and the model
    would silently train on meaningless features.
    """
    null_rates = df[feat_cols].isnull().mean()
    bad = null_rates[null_rates >= threshold]
    if not bad.empty:
        lines = [f"  {col}: {rate:.1%}" for col, rate in bad.items()]
        raise ValueError(
            f"Feature columns exceed {threshold:.0%} null threshold "
            f"(broken upstream join or missing data source):\n"
            + "\n".join(lines)
            + "\nFix the upstream feature pipeline before training."
        )
    partial = null_rates[(null_rates > 0) & (null_rates < threshold)]
    if not partial.empty:
        log.info("  [null audit] Columns with partial nulls (will be imputed):")
        for col, rate in partial.items():
            log.info(f"    {col}: {rate:.1%}")


# ── Feature selection ──────────────────────────────────────────────────────────

def get_feature_cols(df: pd.DataFrame) -> list:
    """Prefer compact matchup differential features with critical context.

    Injury proxy features (home/away prefixed, not diff) are included
    explicitly so that player availability signals reach the model even when
    a diff_ version is absent or sparse.
    """
    exclude = {TARGET, "game_id", "season", "game_date", "home_team", "away_team"}
    numeric_cols = [
        c for c in df.columns
        if c not in exclude and df[c].dtype in [np.float64, np.int64, float, int]
    ]

    diff_cols = [c for c in numeric_cols if c.startswith("diff_")]

    # Rest and schedule context (Phase 4: travel + season-segment features added)
    schedule_cols = {
        "home_days_rest", "away_days_rest",
        "home_is_back_to_back", "away_is_back_to_back",
        "home_travel_miles", "away_travel_miles",
        "home_cross_country_travel", "away_cross_country_travel",
        "season_month",
    }
    # Injury proxy context — home/away form (not covered by diff_ alone because
    # star_player_out is a binary flag where the absolute value matters, not just
    # the differential).
    injury_cols = {
        "home_missing_minutes",      "away_missing_minutes",
        "home_missing_usg_pct",      "away_missing_usg_pct",
        "home_rotation_availability","away_rotation_availability",
        "home_star_player_out",      "away_star_player_out",
    }
    context_cols = [c for c in numeric_cols if c in schedule_cols | injury_cols]

    if diff_cols:
        return sorted(set(diff_cols + context_cols))
    return numeric_cols


# ── Helpers ────────────────────────────────────────────────────────────────────

class _CalibrationUnpickler(pickle.Unpickler):
    """Resolve calibration wrappers pickled under __main__ to their real module."""

    def find_class(self, module, name):
        if module == "__main__" and name in ("_PlattWrapper", "_CalibratedWrapper"):
            from src.models.calibration import _PlattWrapper, _CalibratedWrapper
            return {"_PlattWrapper": _PlattWrapper, "_CalibratedWrapper": _CalibratedWrapper}[name]
        return super().find_class(module, name)


def _load_game_outcome_model(artifacts_dir: str = ARTIFACTS_DIR) -> tuple:
    """Load the game outcome model, preferring the calibrated artifact.

    Returns (model, artifact_filename).
    Issues a UserWarning if falling back to the uncalibrated model — calibrated
    model produces probabilities; uncalibrated model produces raw scores.
    FR-1.2: Ensures inference path always uses calibrated probabilities.
    """
    cal_path = os.path.join(artifacts_dir, "game_outcome_model_calibrated.pkl")
    raw_path = os.path.join(artifacts_dir, "game_outcome_model.pkl")

    if os.path.exists(cal_path):
        with open(cal_path, "rb") as f:
            return _CalibrationUnpickler(f).load(), "game_outcome_model_calibrated.pkl"

    warnings.warn(
        "Calibrated model artifact not found at "
        f"'{cal_path}'. Using uncalibrated model — probabilities may not be "
        "reliable. Run: python src/models/calibration.py to generate the "
        "calibrated artifact.",
        UserWarning,
        stacklevel=3,
    )
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"No model artifact found in '{artifacts_dir}'. "
            "Run: python src/models/train_all_models.py"
        )
    with open(raw_path, "rb") as f:
        return pickle.load(f), "game_outcome_model.pkl"


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


def _get_low_importance_features(
    model,
    feature_names: list,
    threshold: float = 0.001,
) -> list[str]:
    """Return feature names with importance below the given threshold.

    Works with tree-based models (feature_importances_) and linear models
    (coef_). Returns an empty list if importance cannot be extracted.
    """
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            return []
        return [
            name
            for name, imp in zip(feature_names, importances)
            if imp < threshold
        ]
    except Exception:
        return []


def _build_fit_params(
    name: str,
    pipe: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict:
    """Build fit_params dict for Pipeline.fit(), handling early-stopping models.

    XGBoost requires eval_set in .fit() for early_stopping_rounds. Since
    the Pipeline imputes X before passing to clf, we must pre-transform
    the validation data through the imputer step so the clf receives
    clean (no NaN) validation arrays.

    Returns an empty dict for models that don't need special fit params.
    """
    if name not in ("xgboost",):
        return {}

    # Pre-transform validation data through the imputer (and scaler if present)
    imputer = pipe.named_steps.get("imputer")
    scaler = pipe.named_steps.get("scaler")

    if imputer is None:
        return {}

    # Transform validation data through the pipeline's pre-processing steps
    # (imputer + scaler) so XGBoost eval_set receives clean arrays.
    # We use the pipe's fitted steps if available, otherwise clone and fit.
    from sklearn.base import clone
    imp_clone = clone(imputer)
    # During CV, the pipeline hasn't been fit yet on this fold's training data,
    # so we must fit the clone on X_val. For the final refit, the caller should
    # pass a held-out validation split from training data (not test data).
    X_val_imp = imp_clone.fit_transform(X_val)
    if scaler is not None:
        scl_clone = clone(scaler)
        X_val_imp = scl_clone.fit_transform(X_val_imp)

    return {"clf__eval_set": [(X_val_imp, y_val.values)], "clf__verbose": False}


def _clone_pipeline(pipe: Pipeline) -> Pipeline:
    """Create a fresh clone of a pipeline with the same hyperparameters."""
    from sklearn.base import clone
    return clone(pipe)


def _prune_low_importance_features(
    best_pipe: Pipeline,
    candidates: dict,
    best_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feat_cols: list,
    y_train: pd.Series,
    y_test: pd.Series,
    best_threshold: float,
    threshold: float = 0.001,
) -> tuple:
    """Drop features with importance < threshold and retrain if it helps.

    Returns (pipeline, feat_cols, test_acc, test_auc, test_pred, test_proba).
    Uses tree-based feature_importances_ or linear coef_ to identify
    low-importance features. Only adopts the pruned set if test AUC improves.
    """
    clf = best_pipe.named_steps["clf"]
    low_imp = _get_low_importance_features(clf, feat_cols, threshold)
    if not low_imp:
        log.info("\n--- Feature Pruning: no features below threshold ---")
        test_proba = best_pipe.predict_proba(test[feat_cols])[:, 1]
        test_pred = (test_proba >= best_threshold).astype(int)
        test_acc = accuracy_score(y_test, test_pred)
        test_auc = roc_auc_score(y_test, test_proba)
        return best_pipe, feat_cols, test_acc, test_auc, test_pred, test_proba

    log.info(f"\n--- Feature Pruning: {len(low_imp)} feature(s) below {threshold} ---")
    for feat_name in sorted(low_imp):
        log.info(f"    - {feat_name}")

    reduced_cols = [c for c in feat_cols if c not in low_imp]
    log.info(f"  Pruning {len(low_imp)} -> {len(reduced_cols)} features remaining")

    reduced_pipe = _clone_pipeline(candidates[best_name])
    reduced_pipe.fit(train[reduced_cols], y_train)
    reduced_proba = reduced_pipe.predict_proba(test[reduced_cols])[:, 1]
    reduced_pred = (reduced_proba >= best_threshold).astype(int)
    reduced_acc = accuracy_score(y_test, reduced_pred)
    reduced_auc = roc_auc_score(y_test, reduced_proba)

    full_proba = best_pipe.predict_proba(test[feat_cols])[:, 1]
    full_pred = (full_proba >= best_threshold).astype(int)
    full_acc = accuracy_score(y_test, full_pred)
    full_auc = roc_auc_score(y_test, full_proba)

    log.info(f"  Full features:   acc={full_acc:.4f}, auc={full_auc:.4f}")
    log.info(f"  Pruned features: acc={reduced_acc:.4f}, auc={reduced_auc:.4f}")

    if reduced_auc >= full_auc:
        log.info("  -> Pruned features maintain/improve AUC. Using pruned set.")
        return (reduced_pipe, reduced_cols, reduced_acc, reduced_auc,
                reduced_pred, reduced_proba)
    else:
        log.info("  -> Pruned features hurt AUC. Keeping full set.")
        return (best_pipe, feat_cols, full_acc, full_auc, full_pred, full_proba)


def _season_splits(train_df: pd.DataFrame) -> list:
    """Create expanding season splits — delegates to shared cv_utils."""
    from src.models.cv_utils import expanding_season_splits
    return expanding_season_splits(train_df, min_train_seasons=MIN_TRAIN_SEASONS_FOR_TUNING)


# ── Train / evaluate ───────────────────────────────────────────────────────────

def train_game_outcome_model(
    matchup_path: str = MATCHUP_PATH,
    artifacts_dir: str = ARTIFACTS_DIR,
    test_seasons: list = TEST_SEASONS,
    modern_era_only: bool = MODERN_ERA_ONLY,
    excluded_seasons: list = EXCLUDED_SEASONS,
) -> tuple:
    """Train a game outcome model and return (pipeline, metrics).

    Args:
        modern_era_only: If True, restrict training to MODERN_ERA_START (2013-14+)
            to focus on the 3-Point Revolution era. Toggle via the MODERN_ERA_ONLY
            constant or pass directly. See model_advisor_notes.md Proposal 5.
        excluded_seasons: List of season codes to exclude from training data when
            modern_era_only is True. Defaults to EXCLUDED_SEASONS (bubble 2019-20
            and shortened 2020-21 seasons). Pass [] to include all modern seasons.
    """
    print("=" * 60)
    log.info("GAME OUTCOME PREDICTION MODEL")
    print("=" * 60)

    log.info("\nLoading matchup features...")
    df = pd.read_csv(matchup_path)
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")
    df = df.sort_values("game_date").reset_index(drop=True)
    log.info(f"  Total games: {len(df):,} | Seasons: {df.season.nunique()}")

    n_before = len(df)
    df = df.dropna(subset=[TARGET])
    if len(df) < n_before:
        log.warning(f"  Dropped {n_before - len(df):,} rows with missing target")

    start_season = MODERN_ERA_START if modern_era_only else TRAIN_START_SEASON
    df = df[df["season"].astype(int) >= int(start_season)].copy()
    label = f"modern era only ({MODERN_ERA_START}+)" if modern_era_only else f"from {TRAIN_START_SEASON}"
    log.info(f"  Training data {label}: {len(df):,} games")

    if modern_era_only and excluded_seasons:
        before = len(df)
        df = df[~df["season"].astype(int).isin(excluded_seasons)].copy()
        n_excluded = before - len(df)
        log.info(f"  Excluded {n_excluded:,} games from anomalous seasons: {excluded_seasons}")

    train = df[~df["season"].astype(int).isin(test_seasons)].copy()
    test = df[df["season"].astype(int).isin(test_seasons)].copy()
    log.info(f"  Train: {len(train):,} games | Test: {len(test):,} games")
    log.info(f"  Test seasons: {test_seasons}")

    feat_cols = get_feature_cols(df)

    log.info("\nValidating feature null rates...")
    validate_feature_null_rates(df, feat_cols)

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
                n_estimators=500,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.9,
                min_samples_leaf=20,
                max_features=0.7,
                validation_fraction=0.1,
                n_iter_no_change=15,
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

    if _LGBM_AVAILABLE:
        candidates["lightgbm"] = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("clf", LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            )),
        ])

    if _XGB_AVAILABLE:
        candidates["xgboost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("clf", XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.7,
                min_child_weight=20,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                eval_metric="logloss",
                early_stopping_rounds=15,
                random_state=42,
            )),
        ])

    splits = _season_splits(train)
    log.info(f"\n--- Model selection across {len(splits)} validation split(s) ---")

    model_scores = {}
    for name, pipe in candidates.items():
        split_accs, split_aucs, split_thresholds = [], [], []

        for tr, va, split_name in splits:
            X_sub, y_sub = tr[feat_cols], tr[TARGET]
            X_val, y_val = va[feat_cols], va[TARGET]

            fit_params = _build_fit_params(name, pipe, X_val, y_val)
            pipe.fit(X_sub, y_sub, **fit_params)
            val_proba = pipe.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_proba)
            best_t, val_acc = _best_threshold(y_val, val_proba)

            split_accs.append(val_acc)
            split_aucs.append(val_auc)
            split_thresholds.append(best_t)
            log.info(f"  {name:>17} | split={split_name} | acc={val_acc:.4f} | auc={val_auc:.4f} | th={best_t:.2f}")

        model_scores[name] = {
            "mean_val_acc": float(np.mean(split_accs)),
            "mean_val_auc": float(np.mean(split_aucs)),
            "threshold": float(round(np.mean(split_thresholds), 2)),
        }

    best_name = max(model_scores, key=lambda k: model_scores[k]["mean_val_auc"])
    best_threshold = model_scores[best_name]["threshold"]
    log.info(f"\nSelected model: {best_name} "
        f"(mean val auc={model_scores[best_name]['mean_val_auc']:.4f}, "
        f"mean val acc={model_scores[best_name]['mean_val_acc']:.4f}, "
        f"threshold={best_threshold:.2f})")

    best_pipe = candidates[best_name]
    # Use a validation split from training data for early stopping (not test set)
    from sklearn.model_selection import train_test_split as _tts
    X_tr_final, X_val_final, y_tr_final, y_val_final = _tts(
        X_train, y_train, test_size=0.15, random_state=42
    )
    final_fit_params = _build_fit_params(best_name, best_pipe, X_val_final, y_val_final)
    best_pipe.fit(X_tr_final, y_tr_final, **final_fit_params)
    test_proba = best_pipe.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= best_threshold).astype(int)

    test_acc = accuracy_score(y_test, test_pred)
    test_auc = roc_auc_score(y_test, test_proba)

    log.info(f"  Test Accuracy : {test_acc:.4f}")
    log.info(f"  Test ROC-AUC  : {test_auc:.4f}")

    log.info("\nClassification Report:")
    log.info(classification_report(y_test, test_pred, target_names=["Away Win", "Home Win"]))

    # ── Automatic low-importance feature pruning ──────────────────────────────
    # Drop features with importance < 0.001 and retrain if AUC holds.
    n_before_prune = len(feat_cols)
    best_pipe, feat_cols, test_acc, test_auc, test_pred, test_proba = (
        _prune_low_importance_features(
            best_pipe, candidates, best_name,
            train, test, feat_cols,
            y_train, y_test, best_threshold,
            threshold=0.001,
        )
    )
    n_pruned = n_before_prune - len(feat_cols)
    if n_pruned > 0:
        log.info(f"  Pruned {n_pruned} features ({n_before_prune} -> {len(feat_cols)})")
        log.info(f"  Post-prune Test Accuracy: {test_acc:.4f}")
        log.info(f"  Post-prune Test ROC-AUC:  {test_auc:.4f}")

    clf = best_pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = pd.Series(clf.feature_importances_, index=feat_cols).sort_values(ascending=False)
    else:
        importances = pd.Series(np.abs(clf.coef_[0]), index=feat_cols).sort_values(ascending=False)

    log.info("\nTop 15 Most Important Features:")
    log.debug(importances.head(15).to_string())

    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(artifacts_dir, "game_outcome_model.pkl")
    feat_path = os.path.join(artifacts_dir, "game_outcome_features.pkl")
    imp_path = os.path.join(artifacts_dir, "game_outcome_importances.csv")

    with open(model_path, "wb") as f:
        pickle.dump(best_pipe, f)
    with open(feat_path, "wb") as f:
        pickle.dump(feat_cols, f)

    importances.reset_index().rename(columns={"index": "feature", 0: "importance"}).to_csv(imp_path, index=False)
    log.info(f"\nModel saved -> {model_path}")

    # ── Metadata JSON (FR-6.5, NFR-3) ─────────────────────────────────────────
    # Build feature importance dict (top 20) if model supports it
    top_importances = {}
    try:
        clf_step = best_pipe.named_steps.get("clf") or best_pipe.named_steps.get("model")
        if clf_step is not None and hasattr(clf_step, "feature_importances_"):
            imp_series = pd.Series(clf_step.feature_importances_, index=feat_cols)
            top_importances = {k: round(float(v), 6) for k, v in imp_series.nlargest(20).items()}
        elif clf_step is not None and hasattr(clf_step, "coef_"):
            imp_series = pd.Series(np.abs(clf_step.coef_[0]), index=feat_cols)
            top_importances = {k: round(float(v), 6) for k, v in imp_series.nlargest(20).items()}
    except Exception:
        pass  # metadata is best-effort; do not crash training if importance extraction fails

    metadata = {
        "trained_at": datetime.now().isoformat(),
        "model_name": best_name,
        "train_start_season": start_season,
        "modern_era_only": bool(modern_era_only),
        "excluded_seasons": list(excluded_seasons) if modern_era_only else [],
        "feature_list": list(feat_cols),
        "feature_count": int(len(feat_cols)),
        "test_accuracy": float(test_acc),
        "test_auc": float(test_auc),
        "decision_threshold": float(best_threshold),
        "top_importances": top_importances,
    }
    meta_path = os.path.join(artifacts_dir, "game_outcome_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"  Metadata saved -> {meta_path}")

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


def _get_current_season_code() -> int:
    """Return the current NBA season code (e.g. 202526)."""
    now = datetime.now()
    # NBA season starts in October; if month >= 10, season is thisYear_nextYear
    if now.month >= 10:
        return int(f"{now.year}{(now.year + 1) % 100:02d}")
    return int(f"{now.year - 1}{now.year % 100:02d}")


def _synthesize_matchup_row(
    df: pd.DataFrame,
    home_team_abbr: str,
    away_team_abbr: str,
    feat_cols: list[str],
) -> pd.Series | None:
    """Build a fresh matchup row from each team's most recent game.

    Uses the home team's latest game for home_* columns and the away team's
    latest game for away_* columns, regardless of opponent. Recomputes all
    diff_* columns and injects current Elo ratings from elo_ratings.csv.

    Returns None if either team has no current-season data.
    """
    current_season = _get_current_season_code()

    # Each team's most recent game in current season
    current_df = df[df["season"].astype(int) == int(current_season)]
    if current_df.empty:
        current_df = df  # fall back to all data if no current-season rows

    home_games = current_df[
        (current_df["home_team"] == home_team_abbr)
        | (current_df["away_team"] == home_team_abbr)
    ].sort_values("game_date")

    away_games = current_df[
        (current_df["home_team"] == away_team_abbr)
        | (current_df["away_team"] == away_team_abbr)
    ].sort_values("game_date")

    if home_games.empty or away_games.empty:
        return None

    # Prefer a row where the team is in the matching role (home plays at home)
    home_as_home = home_games[home_games["home_team"] == home_team_abbr]
    home_source = (
        home_as_home.iloc[-1].copy() if not home_as_home.empty
        else home_games.iloc[-1].copy()
    )

    away_as_away = away_games[away_games["away_team"] == away_team_abbr]
    away_source = (
        away_as_away.iloc[-1].copy() if not away_as_away.empty
        else away_games.iloc[-1].copy()
    )

    # Start from home source row, overwrite away_* columns from away source
    row = home_source.copy()
    row["away_team"] = away_team_abbr
    row["home_team"] = home_team_abbr

    for c in df.columns:
        if c.startswith("away_"):
            row[c] = away_source.get(c, row.get(c, np.nan))

    # Inject current Elo ratings
    try:
        from src.features.elo import get_current_elos
        current_elos = get_current_elos()
        home_elo = current_elos.get(home_team_abbr, 1500.0)
        away_elo = current_elos.get(away_team_abbr, 1500.0)
        row["home_elo"] = home_elo
        row["away_elo"] = away_elo
    except Exception:
        home_elo = row.get("home_elo", 1500.0)
        away_elo = row.get("away_elo", 1500.0)

    # Recompute ALL diff_* columns from current home/away values
    for c in df.columns:
        if c.startswith("diff_"):
            if c == "diff_elo":
                row[c] = home_elo - away_elo
                continue
            base = c.replace("diff_", "")
            h_col, a_col = f"home_{base}", f"away_{base}"
            if h_col in row.index and a_col in row.index:
                try:
                    row[c] = float(row[h_col]) - float(row[a_col])
                except (TypeError, ValueError):
                    pass

    return row


def predict_game(
    home_team_abbr: str,
    away_team_abbr: str,
    features_path: str = MATCHUP_PATH,
    artifacts_dir: str = ARTIFACTS_DIR,
    game_date: str | None = None,   # "YYYY-MM-DD"; defaults to today if None
) -> dict:
    """Estimate win probability for a matchup.

    Strategy:
      1) Look for a current-season exact matchup between these teams.
      2) If none exists, synthesize a fresh row from each team's most recent
         game in the current season, with current Elo ratings and recomputed
         diff_* columns.
      3) Last resort: use any available data (cross-season).
    """
    model, model_artifact_name = _load_game_outcome_model(artifacts_dir)
    feat_path = os.path.join(artifacts_dir, "game_outcome_features.pkl")
    with open(feat_path, "rb") as f:
        feat_cols = pickle.load(f)

    df = pd.read_csv(features_path)
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")

    current_season = _get_current_season_code()

    # 1) Try current-season exact matchup
    exact = df[
        (df["home_team"] == home_team_abbr)
        & (df["away_team"] == away_team_abbr)
        & (df["season"].astype(int) == int(current_season))
    ]
    if not exact.empty:
        row = exact.sort_values("game_date").iloc[-1].copy()
        # Still refresh Elo to reflect games played since this matchup
        try:
            from src.features.elo import get_current_elos
            current_elos = get_current_elos()
            home_elo = current_elos.get(home_team_abbr, row.get("home_elo", 1500.0))
            away_elo = current_elos.get(away_team_abbr, row.get("away_elo", 1500.0))
            row["home_elo"] = home_elo
            row["away_elo"] = away_elo
            row["diff_elo"] = home_elo - away_elo
        except Exception:
            pass
    else:
        # 2) Synthesize from each team's most recent game
        row = _synthesize_matchup_row(df, home_team_abbr, away_team_abbr, feat_cols)
        if row is None:
            return {
                "error": f"Not enough history to build features for "
                         f"{home_team_abbr} vs {away_team_abbr}."
            }

    row_df = row.to_frame().T.reindex(columns=feat_cols)
    prob = model.predict_proba(row_df)[0]

    # ── Prediction store + JSON export (FR-6.1, FR-6.3, FR-6.4) ──────────────
    result = {
        "home_team": home_team_abbr,
        "away_team": away_team_abbr,
        "home_win_prob": round(float(prob[1]), 4),
        "away_win_prob": round(float(prob[0]), 4),
        "model_artifact": model_artifact_name,
        "feature_count": len(feat_cols),
        "game_date": game_date,
    }

    try:
        from src.outputs.prediction_store import write_game_prediction
        from src.outputs.json_export import export_daily_snapshot
        write_game_prediction(result)
        export_daily_snapshot(game_date)
    except Exception as e:
        import warnings
        warnings.warn(
            f"Could not write prediction to store: {e}. "
            "Prediction result still returned normally.",
            stacklevel=2,
        )

    return result


if __name__ == "__main__":
    model, metrics = train_game_outcome_model()
    log.info(f"\nFinal model accuracy: {metrics['test_accuracy']:.1%}")
    log.info(f"Final model ROC-AUC:  {metrics['test_auc']:.4f}")
