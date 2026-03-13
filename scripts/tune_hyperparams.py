"""
Optuna Hyperparameter Optimization for the Game Outcome Model
=============================================================
Tunes LightGBM (and XGBoost if available) using the same expanding-window
cross-validation logic as game_outcome_model.py.

Usage:
    python scripts/tune_hyperparams.py
    python scripts/tune_hyperparams.py --trials 100
    python scripts/tune_hyperparams.py --trials 50 --no-retrain

Outputs:
    models/artifacts/best_hpo_params.json       -- best params + score
    models/artifacts/game_outcome_model_hpo.pkl -- retrained model (unless --no-retrain)
"""

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

# ── sys.path: PROJECT_ROOT must be importable so src.* modules resolve ─────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False

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

MATCHUP_PATH = str(PROJECT_ROOT / "data" / "features" / "game_matchup_features.csv")
ARTIFACTS_DIR = str(PROJECT_ROOT / "models" / "artifacts")
FEATURES_PKL = str(PROJECT_ROOT / "models" / "artifacts" / "game_outcome_features.pkl")

from src.config import get_current_season

TARGET = "home_win"
# Last 2 complete seasons are holdout test set
_cur = get_current_season()
TEST_SEASONS = [_cur - 202, _cur - 101]  # e.g. 202526 -> [202324, 202425]
MODERN_ERA_START = 201314
EXCLUDED_SEASONS = [201920, 202021]  # COVID bubble seasons (historical constant)
MIN_TRAIN_SEASONS_FOR_TUNING = 6

DEFAULT_N_TRIALS = 50

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(matchup_path: str = MATCHUP_PATH) -> pd.DataFrame:
    """Load, sort, and filter the matchup feature CSV.

    Applies modern-era filter and excludes anomalous seasons (bubble + COVID)
    to mirror the training configuration used in game_outcome_model.py.
    """
    df = pd.read_csv(matchup_path)
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")
    df = df.sort_values("game_date").reset_index(drop=True)

    n_before = len(df)
    df = df.dropna(subset=[TARGET])
    if len(df) < n_before:
        log.info("Dropped %d rows with missing target", n_before - len(df))

    df = df[df["season"].astype(int) >= MODERN_ERA_START].copy()
    df = df[~df["season"].astype(int).isin(EXCLUDED_SEASONS)].copy()
    log.info("Loaded %d games across %d seasons", len(df), df["season"].nunique())
    return df


def load_feature_cols(features_pkl: str = FEATURES_PKL) -> list | None:
    """Load the canonical feature list from the training artifact.

    The project stores model feature lists as pickle files -- this matches the
    project convention established in game_outcome_model.py and is trusted
    project-internal data, not external input.

    Returns None if the artifact does not yet exist (cold start).
    """
    if not os.path.exists(features_pkl):
        log.warning(
            "Feature pkl not found at '%s' -- deriving features from CSV. "
            "Run game_outcome_model.py first to generate the canonical list.",
            features_pkl,
        )
        return None

    import pickle  # noqa: PLC0415 -- deferred to keep import local to this path
    with open(features_pkl, "rb") as f:
        cols = pickle.load(f)  # project-internal artifact, not external input
    log.info("Loaded %d features from %s", len(cols), features_pkl)
    return cols


def derive_feature_cols(df: pd.DataFrame) -> list:
    """Derive feature columns from the DataFrame (fallback if pkl missing).

    Mirrors the logic in game_outcome_model.get_feature_cols().
    """
    exclude = {TARGET, "game_id", "season", "game_date", "home_team", "away_team"}
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
    context_cols = [c for c in numeric_cols if c in schedule_cols | injury_cols]
    if diff_cols:
        return sorted(set(diff_cols + context_cols))
    return numeric_cols


# ── Expanding-window CV ────────────────────────────────────────────────────────

def season_splits(train_df: pd.DataFrame) -> list:
    """Build expanding-window CV splits (identical to game_outcome_model.py).

    Each entry is (train_subset, val_subset, season_label).
    No future data ever enters the training window -- fully leakage-free.
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

    if not splits:
        cutoff = int(len(train_df) * 0.85)
        tr = train_df.iloc[:cutoff].copy()
        va = train_df.iloc[cutoff:].copy()
        if not tr.empty and not va.empty:
            splits.append((tr, va, "date_fallback"))

    return splits


def expanding_window_auc(pipeline, train_df: pd.DataFrame, feat_cols: list) -> float:
    """Evaluate a pipeline using expanding-window CV, returning mean ROC-AUC.

    Mirrors the split logic in game_outcome_model._season_splits() and the
    evaluation loop in train_game_outcome_model().  Each split sees only
    past seasons in training and one future season in validation.
    """
    splits = season_splits(train_df)
    if not splits:
        return 0.0

    aucs = []
    for tr, va, _ in splits:
        X_tr, y_tr = tr[feat_cols], tr[TARGET]
        X_va, y_va = va[feat_cols], va[TARGET]
        try:
            pipeline.fit(X_tr, y_tr)
            proba = pipeline.predict_proba(X_va)[:, 1]
            aucs.append(roc_auc_score(y_va, proba))
        except Exception:
            aucs.append(0.0)

    return float(np.mean(aucs))


# ── Optuna objectives ──────────────────────────────────────────────────────────

def build_lgbm_pipeline(trial) -> Pipeline:
    """Suggest LightGBM hyperparameters and return a fitted-ready pipeline."""
    params = {
        "n_estimators":      trial.suggest_int("lgbm_n_estimators", 100, 1000),
        "max_depth":         trial.suggest_int("lgbm_max_depth", 3, 12),
        "learning_rate":     trial.suggest_float("lgbm_learning_rate", 0.005, 0.3, log=True),
        "num_leaves":        trial.suggest_int("lgbm_num_leaves", 15, 127),
        "subsample":         trial.suggest_float("lgbm_subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("lgbm_colsample_bytree", 0.4, 1.0),
        "min_child_samples": trial.suggest_int("lgbm_min_child_samples", 5, 100),
        "reg_alpha":         trial.suggest_float("lgbm_reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("lgbm_reg_lambda", 1e-8, 10.0, log=True),
        "random_state": 42,
        "verbose": -1,
    }
    clf = LGBMClassifier(**params)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("clf", clf),
    ])


def build_xgb_pipeline(trial) -> Pipeline:
    """Suggest XGBoost hyperparameters and return a fitted-ready pipeline."""
    params = {
        "n_estimators":      trial.suggest_int("xgb_n_estimators", 100, 1000),
        "max_depth":         trial.suggest_int("xgb_max_depth", 3, 12),
        "learning_rate":     trial.suggest_float("xgb_learning_rate", 0.005, 0.3, log=True),
        "subsample":         trial.suggest_float("xgb_subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("xgb_colsample_bytree", 0.4, 1.0),
        "min_child_weight":  trial.suggest_int("xgb_min_child_weight", 1, 20),
        "reg_alpha":         trial.suggest_float("xgb_reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("xgb_reg_lambda", 1e-8, 10.0, log=True),
        "gamma":             trial.suggest_float("xgb_gamma", 0.0, 5.0),
        "random_state": 42,
        "eval_metric": "logloss",
        "verbosity": 0,
    }
    clf = XGBClassifier(**params)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("clf", clf),
    ])


def make_objective(train_df: pd.DataFrame, feat_cols: list, model_type: str):
    """Return an Optuna objective function that closes over training data.

    Args:
        train_df:   DataFrame containing only training seasons (test seasons
                    excluded before this function is called).
        feat_cols:  Canonical feature column list.
        model_type: "lgbm" or "xgb".

    Returns:
        Callable[trial] -> float (mean ROC-AUC across expanding CV splits).
    """
    if model_type == "lgbm" and not _LGBM_AVAILABLE:
        raise RuntimeError("LightGBM is not installed.")
    if model_type == "xgb" and not _XGB_AVAILABLE:
        raise RuntimeError("XGBoost is not installed.")

    def objective(trial):
        if model_type == "lgbm":
            pipeline = build_lgbm_pipeline(trial)
        else:
            pipeline = build_xgb_pipeline(trial)
        return expanding_window_auc(pipeline, train_df, feat_cols)

    return objective


# ── Study runner ───────────────────────────────────────────────────────────────

def run_study(
    train_df: pd.DataFrame,
    feat_cols: list,
    model_type: str,
    n_trials: int = DEFAULT_N_TRIALS,
) -> tuple:
    """Run an Optuna study for one model type.

    Returns (best_params_dict, best_auc_float).
    best_params_dict has a "model_type" key prepended for downstream use.
    """
    if not _OPTUNA_AVAILABLE:
        raise RuntimeError(
            "Optuna is not installed. Run: pip install optuna"
        )

    log.info(
        "Starting %s study with %d trials (expanding-window AUC objective)...",
        model_type.upper(), n_trials,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    objective = make_objective(train_df, feat_cols, model_type)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = {"model_type": model_type, **study.best_params}
    best_auc = study.best_value
    log.info(
        "%s best AUC: %.4f after %d trials",
        model_type.upper(), best_auc, n_trials,
    )
    return best_params, best_auc


# ── Final model retraining ─────────────────────────────────────────────────────

def retrain_with_best_params(
    train_df: pd.DataFrame,
    feat_cols: list,
    best_params: dict,
    artifacts_dir: str = ARTIFACTS_DIR,
) -> Pipeline:
    """Retrain on the full training set using the best HPO params.

    Strips the model-type prefix from param names before constructing the
    classifier (e.g. "lgbm_n_estimators" -> "n_estimators").

    Saves the fitted pipeline to models/artifacts/game_outcome_model_hpo.pkl
    using pickle -- project convention matches game_outcome_model.py.
    """
    import pickle  # noqa: PLC0415 -- local import to keep it near the write

    model_type = best_params.get("model_type", "lgbm")
    raw_params = {k: v for k, v in best_params.items() if k != "model_type"}

    # Strip model-type prefix (e.g. "lgbm_n_estimators" -> "n_estimators")
    prefix = f"{model_type}_"
    clean_params = {
        (k[len(prefix):] if k.startswith(prefix) else k): v
        for k, v in raw_params.items()
    }

    if model_type == "lgbm":
        clean_params.setdefault("random_state", 42)
        clean_params.setdefault("verbose", -1)
        clf = LGBMClassifier(**clean_params)
    elif model_type == "xgb":
        clean_params.setdefault("random_state", 42)
        clean_params.setdefault("eval_metric", "logloss")
        clean_params.setdefault("verbosity", 0)
        clf = XGBClassifier(**clean_params)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("clf", clf),
    ])

    X_train = train_df[feat_cols]
    y_train = train_df[TARGET]
    pipeline.fit(X_train, y_train)

    os.makedirs(artifacts_dir, exist_ok=True)
    out_path = os.path.join(artifacts_dir, "game_outcome_model_hpo.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(pipeline, f)
    log.info("HPO model saved -> %s", out_path)
    return pipeline


# ── CLI entry point ────────────────────────────────────────────────────────────

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Optuna HPO for NBA game outcome model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--trials", "-n",
        type=int,
        default=DEFAULT_N_TRIALS,
        dest="n_trials",
        help="Number of Optuna trials per model type",
    )
    parser.add_argument(
        "--no-retrain",
        action="store_true",
        default=False,
        help="Skip retraining the final model after HPO (only save best params JSON)",
    )
    parser.add_argument(
        "--matchup-path",
        default=MATCHUP_PATH,
        help="Path to game_matchup_features.csv",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=ARTIFACTS_DIR,
        help="Directory for saving artifacts",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if not _OPTUNA_AVAILABLE:
        log.error("Optuna not installed. Run: pip install optuna")
        sys.exit(1)

    # ── Load data ──────────────────────────────────────────────────────────────
    df = load_data(args.matchup_path)

    feat_cols_loaded = load_feature_cols(
        os.path.join(args.artifacts_dir, "game_outcome_features.pkl")
    )
    if feat_cols_loaded is None:
        feat_cols = derive_feature_cols(df)
        log.info("Derived %d feature columns from CSV", len(feat_cols))
    else:
        # Keep only columns that actually exist in the current CSV
        feat_cols = [c for c in feat_cols_loaded if c in df.columns]
        missing = [c for c in feat_cols_loaded if c not in df.columns]
        if missing:
            log.warning(
                "%d feature(s) in pkl missing from CSV -- skipped: %s",
                len(missing), missing[:5],
            )

    # Exclude holdout test seasons -- same as game_outcome_model.py
    train_df = df[~df["season"].astype(int).isin(TEST_SEASONS)].copy()
    log.info(
        "Train set: %d games across %d seasons (test seasons excluded: %s)",
        len(train_df), train_df["season"].nunique(), TEST_SEASONS,
    )

    # ── Run studies ────────────────────────────────────────────────────────────
    results = {}

    if _LGBM_AVAILABLE:
        lgbm_params, lgbm_auc = run_study(
            train_df, feat_cols, "lgbm", n_trials=args.n_trials
        )
        results["lgbm"] = {"params": lgbm_params, "auc": lgbm_auc}
    else:
        log.warning("LightGBM not available -- skipping LGBM study")

    if _XGB_AVAILABLE:
        xgb_params, xgb_auc = run_study(
            train_df, feat_cols, "xgb", n_trials=args.n_trials
        )
        results["xgb"] = {"params": xgb_params, "auc": xgb_auc}
    else:
        log.warning("XGBoost not available -- skipping XGB study")

    if not results:
        log.error("No model libraries available. Install lightgbm or xgboost.")
        sys.exit(1)

    # ── Pick overall winner ────────────────────────────────────────────────────
    best_model_type = max(results, key=lambda k: results[k]["auc"])
    best_params = results[best_model_type]["params"]
    best_auc = results[best_model_type]["auc"]

    print("\n" + "=" * 60)
    print("HPO RESULTS")
    print("=" * 60)
    for model_type, res in results.items():
        marker = " <-- BEST" if model_type == best_model_type else ""
        print(f"  {model_type.upper():6s}  AUC={res['auc']:.4f}{marker}")
    print(f"\nBest params ({best_model_type.upper()}):")
    for k, v in best_params.items():
        if k == "model_type":
            continue
        print(f"  {k}: {v}")

    # ── Save best params JSON ──────────────────────────────────────────────────
    os.makedirs(args.artifacts_dir, exist_ok=True)
    json_path = os.path.join(args.artifacts_dir, "best_hpo_params.json")
    payload = {
        "best_model_type": best_model_type,
        "best_auc": round(best_auc, 6),
        "best_params": best_params,
        "all_results": {
            k: {"auc": round(v["auc"], 6), "params": v["params"]}
            for k, v in results.items()
        },
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Best params saved -> %s", json_path)
    print(f"\nBest params saved -> {json_path}")

    # ── Optionally retrain final model ─────────────────────────────────────────
    if not args.no_retrain:
        log.info("Retraining final model with best params...")
        retrain_with_best_params(
            train_df, feat_cols, best_params, artifacts_dir=args.artifacts_dir
        )
        print(
            "HPO model saved -> "
            + os.path.join(args.artifacts_dir, "game_outcome_model_hpo.pkl")
        )
        print(
            "\nNOTE: Run src/models/calibration.py if you want a calibrated "
            "version of the HPO model for probability outputs."
        )
    else:
        log.info("--no-retrain set: skipping final model save")

    print("\nDone.")
    return payload


if __name__ == "__main__":
    main()
