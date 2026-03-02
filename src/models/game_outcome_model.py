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

warnings.filterwarnings("ignore")


# ── Config ─────────────────────────────────────────────────────────────────────

MATCHUP_PATH = "data/features/game_matchup_features.csv"
ARTIFACTS_DIR = "models/artifacts"
TARGET = "home_win"
TEST_SEASONS = ["202324", "202425"]
TRAIN_START_SEASON = "200001"
MIN_TRAIN_SEASONS_FOR_TUNING = 6

# When True, restrict training data to the modern 3-Point Revolution era
# (2013-14 onward). Cuts historical noise at the cost of less training data.
# See docs/model_advisor_notes.md Proposal 5 for rationale.
# CHANGED: default to modern era (was False)
MODERN_ERA_ONLY = True
# CHANGED: include 2013-14 per SC1 (was "201415")
MODERN_ERA_START = "201314"
# NEW: exclude anomalous seasons (bubble + shortened COVID)
EXCLUDED_SEASONS = ["201920", "202021"]


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
        print("  [null audit] Columns with partial nulls (will be imputed):")
        for col, rate in partial.items():
            print(f"    {col}: {rate:.1%}")


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

    # Rest and schedule context
    schedule_cols = {
        "home_days_rest", "away_days_rest",
        "home_is_back_to_back", "away_is_back_to_back",
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
            return pickle.load(f), "game_outcome_model_calibrated.pkl"

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

    start_season = MODERN_ERA_START if modern_era_only else TRAIN_START_SEASON
    df = df[df["season"].astype(str) >= start_season].copy()
    label = f"modern era only ({MODERN_ERA_START}+)" if modern_era_only else f"from {TRAIN_START_SEASON}"
    print(f"  Training data {label}: {len(df):,} games")

    if modern_era_only and excluded_seasons:
        before = len(df)
        df = df[~df["season"].astype(str).isin(excluded_seasons)].copy()
        n_excluded = before - len(df)
        print(f"  Excluded {n_excluded:,} games from anomalous seasons: {excluded_seasons}")

    train = df[~df["season"].astype(str).isin(test_seasons)].copy()
    test = df[df["season"].astype(str).isin(test_seasons)].copy()
    print(f"  Train: {len(train):,} games | Test: {len(test):,} games")
    print(f"  Test seasons: {test_seasons}")

    feat_cols = get_feature_cols(df)

    print("\nValidating feature null rates...")
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
    print(f"  Metadata saved → {meta_path}")

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
    game_date: str | None = None,   # "YYYY-MM-DD"; defaults to today if None
) -> dict:
    """Estimate win probability for a matchup.

    Strategy:
      1) Use most recent exact historical pairing if available.
      2) Fallback: synthesize a matchup row using each team's most recent
         home/away context and differential columns.
    """
    model, model_artifact_name = _load_game_outcome_model(artifacts_dir)
    feat_path = os.path.join(artifacts_dir, "game_outcome_features.pkl")
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
    print(f"\nFinal model accuracy: {metrics['test_accuracy']:.1%}")
    print(f"Final model ROC-AUC:  {metrics['test_auc']:.4f}")
