"""
Player Performance Prediction Model
Train per-target regression models to forecast player next-game stats
(points, rebounds, assists) from pre-game features.

Enhancements:
  - Validation-season model selection per target.
  - Candidate ensemble regressors (Ridge, GBM, Random Forest, Extra Trees).
  - Better diagnostics and consistent artifact metadata.
Player Performance Prediction Model  (v2 — improved, numpy-only)
Trains regression models to forecast individual player stats for the next
game (points, rebounds, assists) using rolling pre-game features.

Key improvements over v1:
  • Collinearity removal  — removes one feature from every pair whose
    |Pearson r| > 0.92, keeping the one more correlated with the target.
  • Importance pruning    — two-pass training; prune near-zero-importance
    features before the final 500-tree pass.
  • Better hyperparameters — shallower trees (max_depth=3), more
    estimators, stronger leaf regularisation.
  • Pure numpy — no scikit-learn dependency.

Usage:
    python src/models/player_performance_model.py
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, os.path.dirname(__file__))

from numpy_gbm import (
    FastGBMRegressor,
    remove_collinear_features, prune_by_importance,
    mean_absolute_error, r2_score,
)

warnings.filterwarnings("ignore")


# ── Config ─────────────────────────────────────────────────────────────────────

FEATURES_PATH = "data/features/player_game_features.csv"
ARTIFACTS_DIR = "models/artifacts"
TARGETS = ["pts", "reb", "ast"]
TEST_SEASONS = ["202324", "202425"]
MIN_TRAIN_GAMES = 20
VALIDATION_SEASON = "202223"
TARGETS         = ["pts", "reb", "ast"]
TEST_SEASONS    = None   # auto-detect: last 2 seasons in the data
MIN_TRAIN_GAMES = 20

CORR_THRESHOLD      = 0.92
IMP_CUMULATIVE_FRAC = 0.999


# ── Feature utilities ──────────────────────────────────────────────────────────

_RAW_GAME_STATS = [
    "pts", "reb", "ast", "stl", "blk", "tov", "pf",
    "min", "fgm", "fga", "fg_pct",
    "fg3m", "fg3a", "fg3_pct",
    "ftm", "fta", "ft_pct",
    "oreb", "dreb",
    "plus_minus", "win", "wl",
]

def get_feature_cols(df: pd.DataFrame, target: str) -> list:
    """Return numeric predictors while excluding leakage columns."""
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
    cols = [
        c for c in df.columns
        if c not in exclude and df[c].dtype in [np.float64, np.int64, float, int]
_META_COLS = {
    "season", "player_id", "player_name", "team_id",
    "team_abbreviation", "game_id", "game_date", "matchup",
}


def get_feature_cols(df, target):
    """Return feature columns for predicting `target`."""
    exclude = set(_RAW_GAME_STATS) | _META_COLS | {target}
    return [
        c for c in df.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    # Prefer stable predictive groups first
    priority = [
        c for c in cols
        if any(k in c for k in [
            "roll", "std", "season_avg", "form_delta", "per_min",
            "opp_", "team_", "role_opportunity", "rest_advantage",
            "usg_pct", "ts_pct", "net_rating", "pie", "age",
            "scoring_", "clutch_",
        ])
    ]
    return sorted(set(priority if priority else cols))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_candidates() -> dict:
    return {
        "ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("model", GradientBoostingRegressor(
                n_estimators=350,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.9,
                random_state=42,
                loss="squared_error",
            )),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("model", RandomForestRegressor(
                n_estimators=700,
                max_depth=12,
                min_samples_leaf=6,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "extra_trees": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("model", ExtraTreesRegressor(
                n_estimators=700,
                max_depth=14,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    }


def _split_train_validation(train_df: pd.DataFrame) -> tuple:
    subtrain = train_df[train_df["season"].astype(str) < VALIDATION_SEASON].copy()
    valid = train_df[train_df["season"].astype(str) == VALIDATION_SEASON].copy()
    if valid.empty or subtrain.empty:
        cutoff = int(len(train_df) * 0.85)
        subtrain = train_df.iloc[:cutoff].copy()
        valid = train_df.iloc[cutoff:].copy()
    return subtrain, valid


def _extract_importance(model_pipe: Pipeline, feat_cols: list) -> pd.Series:
    model = model_pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
    if hasattr(model, "coef_"):
        return pd.Series(np.abs(model.coef_), index=feat_cols).sort_values(ascending=False)
    return pd.Series(np.zeros(len(feat_cols)), index=feat_cols)
# ── Ridge baseline (pure numpy) ────────────────────────────────────────────────

class _RidgeBaseline:
    """Thin Ridge regression using numpy normal equations."""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.w_ = None
        self.b_ = 0.0

    def _prep(self, X, fit=False):
        Xc = np.where(np.isnan(X), 0.0, X)
        if fit:
            self._mn  = Xc.mean(axis=0)
            self._std = Xc.std(axis=0)
            self._std[self._std == 0] = 1.0
        return (Xc - self._mn) / self._std

    def fit(self, X, y):
        Xs = self._prep(X, fit=True)
        n, p = Xs.shape
        A = Xs.T @ Xs + self.alpha * np.eye(p)
        b = Xs.T @ y
        self.w_ = np.linalg.solve(A, b)
        self.b_ = float(y.mean() - Xs.mean(axis=0) @ self.w_)
        return self

    def predict(self, X):
        return self._prep(X) @ self.w_ + self.b_

def train_player_models(
    features_path: str = FEATURES_PATH,
    artifacts_dir: str = ARTIFACTS_DIR,
    test_seasons: list = TEST_SEASONS,
    targets: list = TARGETS,
) -> tuple:
    """Train one selected regression pipeline per stat target."""

def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


# ── Train one stat model ───────────────────────────────────────────────────────

def _train_one(train_df, test_df, all_feat_cols, target):
    """
    Full training pipeline for a single stat target with feature selection.

    Returns:
        (final_model, final_feat_cols, metrics_dict)
    """
    t_train = train_df.dropna(subset=[target])
    t_test  = test_df.dropna(subset=[target])

    y_train = t_train[target].values.astype(float)
    y_test  = t_test[target].values.astype(float)

    X_train_full = t_train[all_feat_cols].values.astype(float)
    X_test_full  = t_test[all_feat_cols].values.astype(float)

    # ── Baseline: Ridge regression ─────────────────────────────────────────
    ridge = _RidgeBaseline(alpha=1.0)
    ridge.fit(X_train_full, y_train)
    ridge_pred = ridge.predict(X_test_full)
    ridge_mae  = mean_absolute_error(y_test, ridge_pred)
    ridge_rmse = _rmse(y_test, ridge_pred)
    print(f"  Baseline Ridge → MAE: {ridge_mae:.3f} | RMSE: {ridge_rmse:.3f}")

    # ── Step 1: Collinearity removal ───────────────────────────────────────
    # Use a sample for speed (correlation doesn't need all rows)
    sample_n   = min(len(t_train), 30_000)
    samp_idx   = np.random.RandomState(42).choice(len(t_train), sample_n, replace=False)
    kept_corr  = remove_collinear_features(
        t_train[all_feat_cols].iloc[samp_idx],
        y_train[samp_idx],
        threshold=CORR_THRESHOLD,
    )
    print(f"  After collinearity removal: {len(kept_corr)} features "
          f"(dropped {len(all_feat_cols) - len(kept_corr)})")

    corr_idx       = [all_feat_cols.index(c) for c in kept_corr]
    X_train_corr   = X_train_full[:, corr_idx]
    X_test_corr    = X_test_full[:,  corr_idx]

    # ── Step 2: Pass 1 — initial fit for importances ───────────────────────
    # Use a subsample for pass-1 to speed up importance estimation
    n_imp_sample = min(len(X_train_corr), 40_000)
    imp_idx = np.random.RandomState(42).choice(len(X_train_corr), n_imp_sample, replace=False)
    gb_pass1 = FastGBMRegressor(
        n_estimators=100, learning_rate=0.05, max_depth=3,
        min_samples_leaf=15, subsample=0.8, max_features=0.4,
        random_state=42,
    )
    gb_pass1.fit(X_train_corr[imp_idx], y_train[imp_idx])
    importances_pass1 = gb_pass1.feature_importances_

    # ── Step 3: Importance pruning ─────────────────────────────────────────
    kept_final_names = prune_by_importance(
        importances_pass1, kept_corr,
        cumulative_frac=IMP_CUMULATIVE_FRAC, min_features=10,
    )
    print(f"  After importance pruning:   {len(kept_final_names)} features "
          f"(dropped {len(kept_corr) - len(kept_final_names)})")

    final_idx    = [kept_corr.index(c) for c in kept_final_names]
    X_train_fin  = X_train_corr[:, final_idx]
    X_test_fin   = X_test_corr[:,  final_idx]

    # ── Step 4: Final GBM on pruned features (full training set, ~50K rows) ─
    # Cap training rows at 50K to keep runtime reasonable while retaining accuracy
    n_train_cap = min(len(X_train_fin), 50_000)
    cap_idx = np.random.RandomState(0).choice(len(X_train_fin), n_train_cap, replace=False)
    print(f"  Training on {n_train_cap:,} rows (capped from {len(X_train_fin):,})")
    gb_final = FastGBMRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=3,
        min_samples_leaf=15, subsample=0.8, max_features=0.4,
        random_state=42,
    )
    gb_final.fit(X_train_fin[cap_idx], y_train[cap_idx])

    gb_pred  = gb_final.predict(X_test_fin)
    gb_mae   = mean_absolute_error(y_test, gb_pred)
    gb_rmse  = _rmse(y_test, gb_pred)
    gb_r2    = r2_score(y_test, gb_pred)
    gb_bias  = float(np.mean(gb_pred - y_test))

    print(f"  Final GBM  → MAE: {gb_mae:.3f} | RMSE: {gb_rmse:.3f} | "
          f"R²: {gb_r2:.3f} | Bias: {gb_bias:+.3f}")

    # ── High-usage breakdown ───────────────────────────────────────────────
    q75 = float(np.percentile(y_test, 75))
    high_mask = y_test >= q75
    if high_mask.sum() > 10:
        hi_mae = mean_absolute_error(y_test[high_mask], gb_pred[high_mask])
        lo_mae = mean_absolute_error(y_test[~high_mask], gb_pred[~high_mask])
        print(f"  High-value games (≥{q75:.0f}): MAE {hi_mae:.3f} | "
              f"Low-value games: MAE {lo_mae:.3f}")

    metrics = {
        "mae"       : gb_mae,
        "rmse"      : gb_rmse,
        "r2"        : gb_r2,
        "bias"      : gb_bias,
        "ridge_mae" : ridge_mae,
        "ridge_rmse": ridge_rmse,
        "n_test"    : len(y_test),
        "n_features": len(kept_final_names),
    }
    return gb_final, kept_final_names, metrics


# ── Main trainer ───────────────────────────────────────────────────────────────

def train_player_models(
    features_path=FEATURES_PATH,
    artifacts_dir=ARTIFACTS_DIR,
    test_seasons=TEST_SEASONS,
    targets=TARGETS,
):
    """Train one regression model per stat (pts, reb, ast)."""
    print("=" * 60)
    print("PLAYER PERFORMANCE PREDICTION MODELS  (v2 — numpy-only)")
    print("=" * 60)

    print("\nLoading player game features...")
    df = pd.read_csv(features_path, encoding="utf-8")
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)
    print(f"  Total rows: {len(df):,} | Players: {df.player_id.nunique():,} | "
          f"Seasons: {df.season.nunique()}")

    # Auto-detect test seasons: use last 2 seasons in the data
    if test_seasons is None:
        all_seasons = sorted(df["season"].astype(str).unique())
        test_seasons = all_seasons[-2:]
        print(f"  Auto-detected test seasons: {test_seasons}")

    train_df = df[~df["season"].astype(str).isin(test_seasons)].copy()
    test_df = df[df["season"].astype(str).isin(test_seasons)].copy()
    print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")

    # Filter out players with too few training appearances
    train_counts = train_df.groupby("player_id")["game_id"].transform("count")
    train_df = train_df[train_counts >= MIN_TRAIN_GAMES].copy()
    print(f"  Train after min-games filter ({MIN_TRAIN_GAMES}): {len(train_df):,}")

    models, metrics = {}, {}
    os.makedirs(artifacts_dir, exist_ok=True)

    for target in targets:
        print(f"\n{'─' * 46}")
    models  = {}
    metrics = {}
    all_importances = {}

    for target in targets:
        print(f"\n{'─'*50}")
        print(f"Target: {target.upper()}")
        print(f"{'─'*50}")

        feat_cols = get_feature_cols(df, target)

        t_train = train_df.dropna(subset=[target]).copy()
        t_test = test_df.dropna(subset=[target]).copy()
        subtrain, valid = _split_train_validation(t_train)

        X_sub, y_sub = subtrain[feat_cols], subtrain[target]
        X_val, y_val = valid[feat_cols], valid[target]
        X_train, y_train = t_train[feat_cols], t_train[target]
        X_test, y_test = t_test[feat_cols], t_test[target]

        # Candidate selection on validation split
        candidates = _build_candidates()
        selection_rows = []
        best_name = None
        best_val_mae = np.inf
        for name, pipe in candidates.items():
            pipe.fit(X_sub, y_sub)
            val_pred = pipe.predict(X_val)
            val_mae = mean_absolute_error(y_val, val_pred)
            val_rmse = root_mean_squared_error(y_val, val_pred)
            selection_rows.append((name, val_mae, val_rmse))
            print(f"  {name:>16} | val_MAE={val_mae:.3f} | val_RMSE={val_rmse:.3f}")
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_name = name

        best_model = candidates[best_name]
        best_model.fit(X_train, y_train)
        test_pred = best_model.predict(X_test)
        test_mae = mean_absolute_error(y_test, test_pred)
        test_rmse = root_mean_squared_error(y_test, test_pred)

        print(f"  Selected      | {best_name}")
        print(f"  Test          | MAE={test_mae:.3f} | RMSE={test_rmse:.3f}")

        models[target] = best_model
        metrics[target] = {
            "selected_model": best_name,
            "validation_mae": float(best_val_mae),
            "mae": float(test_mae),
            "rmse": float(test_rmse),
            "n_test": int(len(y_test)),
            "n_features": int(len(feat_cols)),
        }

        imp = _extract_importance(best_model, feat_cols)
        print(f"\n  Top 10 features for {target}:")
        print(imp.head(10).to_string())

        all_feat_cols = get_feature_cols(df, target)
        print(f"  Initial features: {len(all_feat_cols)}")

        model, kept_cols, m = _train_one(train_df, test_df, all_feat_cols, target)
        models[target]  = model
        metrics[target] = m

        # Feature importances
        imp = pd.Series(
            model.feature_importances_,
            index=kept_cols,
        ).sort_values(ascending=False)
        all_importances[target] = imp
        print(f"\n  Top 10 features for {target.upper()}:")
        print(imp.head(10).to_string())

        # Save artifacts
        os.makedirs(artifacts_dir, exist_ok=True)
        with open(os.path.join(artifacts_dir, f"player_{target}_model.pkl"), "wb") as f:
            pickle.dump(best_model, f)
        with open(os.path.join(artifacts_dir, f"player_{target}_features.pkl"), "wb") as f:
            pickle.dump(feat_cols, f)
        imp.reset_index().rename(columns={"index": "feature", 0: "importance"}).to_csv(
            os.path.join(artifacts_dir, f"player_{target}_importances.csv"), index=False
        )

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    for target, m in metrics.items():
        print(
            f"  {target.upper():>4}: model={m['selected_model']:<16} "
            f"MAE={m['mae']:.3f} | RMSE={m['rmse']:.3f} | val_MAE={m['validation_mae']:.3f}"
        )
            pickle.dump(kept_cols, f)
        imp.reset_index().rename(
            columns={"index": "feature", 0: "importance"}
        ).to_csv(
            os.path.join(artifacts_dir, f"player_{target}_importances.csv"), index=False
        )

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for t, m in metrics.items():
        print(f"  {t.upper():>4}:  MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  "
              f"R²={m['r2']:.3f}  Bias={m['bias']:+.3f}  "
              f"(baseline MAE={m['ridge_mae']:.3f}, features={m['n_features']})")

    return models, metrics


def predict_player_next_game(
    player_name: str,
    features_path: str = FEATURES_PATH,
    artifacts_dir: str = ARTIFACTS_DIR,
    targets: list = TARGETS,
) -> dict:
    """Predict next-game PTS/REB/AST for a player from latest feature row."""
    df = pd.read_csv(features_path)
# ── Predict single player ──────────────────────────────────────────────────────

def predict_player_next_game(
    player_name,
    features_path=FEATURES_PATH,
    artifacts_dir=ARTIFACTS_DIR,
    targets=TARGETS,
):
    """Predict a player's next game stats using their most recent rolling features."""
    df = pd.read_csv(features_path, encoding="utf-8")
    df["game_date"] = pd.to_datetime(df["game_date"])

    player_df = df[df["player_name"] == player_name].sort_values("game_date")
    if player_df.empty:
        return {"error": f"No data found for player: {player_name}"}

    latest  = player_df.iloc[-1]
    results = {
        "player"    : player_name,
        "last_game" : str(latest["game_date"].date()),
    }

    for target in targets:
        feat_path = os.path.join(artifacts_dir, f"player_{target}_features.pkl")
        model_path = os.path.join(artifacts_dir, f"player_{target}_model.pkl")

        with open(feat_path, "rb") as f:
            feat_cols = pickle.load(f)
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        X = latest[feat_cols].to_frame().T.fillna(0)
        results[f"pred_{target}"] = round(float(model.predict(X)[0]), 1)
        with open(feat_path,  "rb") as f: feat_cols = pickle.load(f)
        with open(model_path, "rb") as f: model     = pickle.load(f)

        X    = np.array([[latest.get(c, 0.0) for c in feat_cols]], dtype=float)
        pred = model.predict(X)[0]
        results[f"pred_{target}"] = round(float(pred), 1)

    return results


if __name__ == "__main__":
    train_player_models()
