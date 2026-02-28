"""
Player Performance Prediction Model
=====================================
Train per-target regression models to forecast player next-game stats
(points, rebounds, assists) from pre-game features.

Enhancements:
  - Validation-season model selection per target.
  - Candidate ensemble regressors (Ridge, GBM, Random Forest, Extra Trees).
  - Better diagnostics and consistent artifact metadata.
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

warnings.filterwarnings("ignore")


# ── Config ─────────────────────────────────────────────────────────────────────

FEATURES_PATH = "data/features/player_game_features.csv"
ARTIFACTS_DIR = "models/artifacts"
TARGETS = ["pts", "reb", "ast"]
TEST_SEASONS = ["202324", "202425"]
MIN_TRAIN_GAMES = 20
VALIDATION_SEASON = "202223"


# ── Feature selection ──────────────────────────────────────────────────────────

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


# ── Main trainer ───────────────────────────────────────────────────────────────

def train_player_models(
    features_path: str = FEATURES_PATH,
    artifacts_dir: str = ARTIFACTS_DIR,
    test_seasons: list = TEST_SEASONS,
    targets: list = TARGETS,
) -> tuple:
    """Train one selected regression pipeline per stat target."""
    print("=" * 60)
    print("PLAYER PERFORMANCE PREDICTION MODELS")
    print("=" * 60)

    print("\nLoading player game features...")
    df = pd.read_csv(features_path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)
    print(f"  Total rows: {len(df):,} | Players: {df.player_id.nunique():,} | Seasons: {df.season.nunique()}")

    train_df = df[~df["season"].astype(str).isin(test_seasons)].copy()
    test_df = df[df["season"].astype(str).isin(test_seasons)].copy()
    print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")

    train_counts = train_df.groupby("player_id")["game_id"].transform("count")
    train_df = train_df[train_counts >= MIN_TRAIN_GAMES]
    print(f"  Train after min-games filter ({MIN_TRAIN_GAMES}): {len(train_df):,}")

    models, metrics = {}, {}
    os.makedirs(artifacts_dir, exist_ok=True)

    for target in targets:
        print(f"\n{'─' * 46}")
        print(f"Target: {target.upper()}")

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

    return models, metrics


def predict_player_next_game(
    player_name: str,
    features_path: str = FEATURES_PATH,
    artifacts_dir: str = ARTIFACTS_DIR,
    targets: list = TARGETS,
) -> dict:
    """Predict next-game PTS/REB/AST for a player from latest feature row."""
    df = pd.read_csv(features_path)
    df["game_date"] = pd.to_datetime(df["game_date"])

    player_df = df[df["player_name"] == player_name].sort_values("game_date")
    if player_df.empty:
        return {"error": f"No data found for player: {player_name}"}

    latest = player_df.iloc[-1]
    results = {"player": player_name, "last_game": str(latest["game_date"].date())}

    for target in targets:
        feat_path = os.path.join(artifacts_dir, f"player_{target}_features.pkl")
        model_path = os.path.join(artifacts_dir, f"player_{target}_model.pkl")

        with open(feat_path, "rb") as f:
            feat_cols = pickle.load(f)
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        X = latest[feat_cols].to_frame().T.fillna(0)
        results[f"pred_{target}"] = round(float(model.predict(X)[0]), 1)

    return results


if __name__ == "__main__":
    train_player_models()
