import json, os, pickle, sys, warnings
import numpy as np, pandas as pd
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.elo import get_current_elos

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MATCHUP_PATH = "data/features/game_matchup_features.csv"
GAME_LOGS_PATH = "data/processed/team_game_logs.csv"
ARTIFACTS_DIR = "models/artifacts"
OUTCOME_FEATURES_PKL = "models/artifacts/game_outcome_features.pkl"

TARGET = "point_diff"
TEST_SEASONS = [202324, 202425]
MODERN_ERA_START = 201314
MIN_TRAIN_SEASONS_FOR_CV = 6
EXCLUDED_SEASONS = [201920, 202021]


def _validate_null_rates(df, feat_cols, threshold=0.95):
    """Raise ValueError if any feature column has null rate >= threshold."""
    null_rates = df[feat_cols].isnull().mean()
    bad = null_rates[null_rates >= threshold]
    if not bad.empty:
        lines = [f"  {col}: {rate:.1%}" for col, rate in bad.items()]
        raise ValueError(
            f"Feature columns exceed {threshold:.0%} null threshold "
            f"(broken upstream join):\n" + "\n".join(lines)
        )
    partial = null_rates[(null_rates > 0) & (null_rates < threshold)]
    if not partial.empty:
        print("  [null audit] Columns with partial nulls (will be imputed):")
        for col, rate in partial.items():
            print(f"    {col}: {rate:.1%}")


def _get_feature_cols(df, outcome_features=None):
    """Return the feature column list for the margin model.

    Priority:
      1. If game_outcome_features pkl is available, use the same feature set.
      2. Otherwise fall back to deriving diff_ + context cols from DataFrame.
    """
    if outcome_features is not None:
        available = set(df.columns)
        return [c for c in outcome_features if c in available]

    exclude = {
        TARGET, "home_win", "game_id", "season", "game_date",
        "home_team", "away_team",
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
        "home_missing_minutes", "away_missing_minutes",
        "home_missing_usg_pct", "away_missing_usg_pct",
        "home_rotation_availability", "away_rotation_availability",
        "home_star_player_out", "away_star_player_out",
    }
    context_cols = [c for c in numeric_cols if c in schedule_cols | injury_cols]
    if diff_cols:
        return sorted(set(diff_cols + context_cols))
    return numeric_cols


def _derive_point_diff(matchup, game_logs_path=GAME_LOGS_PATH):
    """Attach point_diff (home_pts - away_pts) to the matchup DataFrame.

    Strategy: join team_game_logs home-team rows on game_id.
    The plus_minus column for home teams equals home_pts - away_pts.
    Rows without a matching log entry are dropped (cannot label them).
    """
    logs = pd.read_csv(
        game_logs_path, usecols=["game_id", "matchup", "plus_minus"]
    )
    logs["is_home"] = (
        logs["matchup"].str.contains(r"vs\.", regex=True).astype(int)
    )
    home_logs = (
        logs[logs["is_home"] == 1][["game_id", "plus_minus"]]
        .rename(columns={"plus_minus": TARGET})
        .drop_duplicates(subset=["game_id"])
    )
    n_before = len(matchup)
    merged = matchup.merge(home_logs, on="game_id", how="inner")
    n_dropped = n_before - len(merged)
    if n_dropped:
        print(f"  Target derivation: dropped {n_dropped:,} rows with no log entry")
    print(
        f"  point_diff range: {merged[TARGET].min():.0f}"
        f" to {merged[TARGET].max():.0f}"
        f" (mean={merged[TARGET].mean():.2f})"
    )
    return merged


def _season_splits(train_df):
    """Build expanding-window season splits for CV.

    Each split: train on seasons 0..i-1, validate on season i.
    Falls back to an 85/15 index split when there are too few seasons.

    Returns:
        List of (train_subset, valid_subset, valid_season_label) tuples.
    """
    seasons = sorted(train_df["season"].astype(int).unique())
    splits = []
    for i in range(max(1, MIN_TRAIN_SEASONS_FOR_CV - 1), len(seasons)):
        train_seasons = seasons[:i]
        valid_season = seasons[i]
        tr = train_df[train_df["season"].astype(int).isin(train_seasons)].copy()
        va = train_df[train_df["season"].astype(int) == valid_season].copy()
        if not tr.empty and not va.empty:
            splits.append((tr, va, valid_season))
    if not splits:
        cutoff = int(len(train_df) * 0.85)
        tr = train_df.iloc[:cutoff].copy()
        va = train_df.iloc[cutoff:].copy()
        if not tr.empty and not va.empty:
            splits.append((tr, va, "date_fallback"))
    return splits


def train_margin_model(
    matchup_path=MATCHUP_PATH,
    game_logs_path=GAME_LOGS_PATH,
    artifacts_dir=ARTIFACTS_DIR,
    test_seasons=TEST_SEASONS,
):
    """Train a point-differential regression model and return (pipeline, metrics).

    Steps:
      1. Load matchup features and derive point_diff target.
      2. Filter to modern era (2013-14+), exclude anomalous seasons.
      3. Time-based train/test split by season.
      4. Expanding-window CV; select model by lowest MAE.
      5. Retrain winner on full train set; evaluate on holdout test seasons.
      6. Save model + feature list to models/artifacts/.

    Returns:
        (best_pipeline, metrics_dict)
    """
    print("=" * 60)
    print("MARGIN REGRESSION MODEL (point differential)")
    print("=" * 60)

    print("\nLoading matchup features...")
    matchup = pd.read_csv(matchup_path)
    matchup["game_date"] = pd.to_datetime(matchup["game_date"], format="mixed")
    matchup = matchup.sort_values("game_date").reset_index(drop=True)
    print(
        f"  Total matchup rows: {len(matchup):,}"
        f" | Seasons: {matchup.season.nunique()}"
    )

    print("\nDeriving point_diff target from team_game_logs...")
    df = _derive_point_diff(matchup, game_logs_path)

    df = df[df["season"].astype(int) >= int(MODERN_ERA_START)].copy()
    df = df[~df["season"].astype(int).isin(EXCLUDED_SEASONS)].copy()
    print(f"  After era/exclusion filter: {len(df):,} games")

    train = df[~df["season"].astype(int).isin(test_seasons)].copy()
    test = df[df["season"].astype(int).isin(test_seasons)].copy()
    print(f"  Train: {len(train):,} | Test: {len(test):,}")
    print(f"  Test seasons: {test_seasons}")

    outcome_features = None
    if os.path.exists(OUTCOME_FEATURES_PKL):
        with open(OUTCOME_FEATURES_PKL, "rb") as fh:
            outcome_features = pickle.load(fh)
        print(f"  Using game_outcome_features.pkl ({len(outcome_features)} cols)")
    else:
        print(
            "  game_outcome_features.pkl not found;"
            " deriving features from DataFrame"
        )

    feat_cols = _get_feature_cols(df, outcome_features)
    print(f"  Feature columns: {len(feat_cols)}")

    print("\nValidating feature null rates...")
    _validate_null_rates(df, feat_cols)

    X_train = train[feat_cols]
    y_train = train[TARGET]
    X_test = test[feat_cols]
    y_test = test[TARGET]

    candidates = {
        "ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=10.0)),
        ]),
        "lasso": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("reg", Lasso(alpha=0.5, max_iter=5000)),
        ]),
        "gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("reg", GradientBoostingRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )),
        ]),
        "huber_gbm": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("reg", GradientBoostingRegressor(
                loss="huber",
                alpha=0.9,
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
    }

    splits = _season_splits(train)
    print(f"\n--- Model selection across {len(splits)} validation split(s) ---")
    model_scores = {name: [] for name in candidates}

    for name, pipe in candidates.items():
        for tr, va, split_label in splits:
            X_sub, y_sub = tr[feat_cols], tr[TARGET]
            X_val, y_val = va[feat_cols], va[TARGET]
            pipe.fit(X_sub, y_sub)
            val_pred = pipe.predict(X_val)
            val_mae = mean_absolute_error(y_val, val_pred)
            model_scores[name].append(val_mae)
            print(
                f"  {name:>20} | split={split_label} | val_MAE={val_mae:.3f}"
            )

    mean_maes = {n: float(np.mean(s)) for n, s in model_scores.items()}
    best_name = min(mean_maes, key=mean_maes.__getitem__)
    cv_mae = mean_maes[best_name]
    print(f"\nSelected model: {best_name} (mean CV MAE = {cv_mae:.3f})")
    for name, mae in sorted(mean_maes.items(), key=lambda x: x[1]):
        marker = " <-- selected" if name == best_name else ""
        print(f"  {name:>20}: {mae:.3f}{marker}")

    best_pipe = candidates[best_name]
    best_pipe.fit(X_train, y_train)

    if len(X_test) == 0:
        print("\nNo test-set rows (test_seasons not present in data); skipping holdout eval.")
        test_mae = float("nan")
        test_rmse = float("nan")
    else:
        test_pred = best_pipe.predict(X_test)
        test_mae = mean_absolute_error(y_test, test_pred)
        test_rmse = float(np.sqrt(np.mean((y_test.values - test_pred) ** 2)))
        print(f"\nTest MAE  : {test_mae:.3f}")
        print(f"Test RMSE : {test_rmse:.3f}")

    reg_step = best_pipe.named_steps.get("reg")
    if reg_step is not None and hasattr(reg_step, "feature_importances_"):
        importances = pd.Series(reg_step.feature_importances_, index=feat_cols)
        print("\nTop 15 Most Important Features:")
        print(importances.sort_values(ascending=False).head(15).to_string())
    elif reg_step is not None and hasattr(reg_step, "coef_"):
        importances = pd.Series(np.abs(reg_step.coef_), index=feat_cols)
        print("\nTop 15 Features by |coefficient|:")
        print(importances.sort_values(ascending=False).head(15).to_string())

    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(artifacts_dir, "margin_model.pkl")
    feat_path = os.path.join(artifacts_dir, "margin_model_features.pkl")
    with open(model_path, "wb") as fh:
        pickle.dump(best_pipe, fh)
    with open(feat_path, "wb") as fh:
        pickle.dump(feat_cols, fh)
    print(f"\nModel saved -> {model_path}")
    print(f"Features saved -> {feat_path}")

    # Compute and save residual_std for BettingRouter spread probability
    y_train_pred = best_pipe.predict(X_train)
    residual_std = float(np.std(y_train.values - y_train_pred))
    residual_std_path = os.path.join(artifacts_dir, "margin_residual_std.json")
    with open(residual_std_path, "w") as f:
        json.dump({"residual_std": residual_std}, f)
    print(f"  Residual std: {residual_std:.2f} (saved to {residual_std_path})")

    # Segmented MAE by predicted margin bucket
    if len(X_test) > 0:
        y_test_arr = y_test.values if hasattr(y_test, "values") else np.array(y_test)
        abs_pred = np.abs(test_pred)
        residuals = np.abs(y_test_arr - test_pred)

        tight = abs_pred <= 3
        medium = (abs_pred > 3) & (abs_pred <= 7)
        wide = abs_pred > 7

        for label, mask in [
            ("Tight (0-3)", tight),
            ("Medium (3-7)", medium),
            ("Wide (7+)", wide),
        ]:
            if np.sum(mask) > 0:
                seg_mae = float(np.mean(residuals[mask]))
                print(f"  MAE [{label}]: {seg_mae:.2f} ({np.sum(mask)} games)")

    metrics = {
        "selected_model": best_name,
        "cv_mean_mae": cv_mae,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "n_train": len(train),
        "n_test": len(test),
        "test_seasons": test_seasons,
        "n_features": len(feat_cols),
        "trained_at": datetime.now().isoformat(),
    }
    return best_pipe, metrics


def predict_margin(
    home_team,
    away_team,
    game_date=None,
    matchup_path=MATCHUP_PATH,
    artifacts_dir=ARTIFACTS_DIR,
):
    """Return predicted point differential (home - away) for a matchup.

    Lookup strategy:
      1. Most recent exact historical pairing (home_team vs away_team).
      2. Fallback: latest home-team row + latest away-team row, spliced.

    Args:
        home_team: Home team abbreviation (e.g., "LAL").
        away_team: Away team abbreviation (e.g., "GSW").
        game_date: Optional date string (informational only).
        matchup_path: Path to game_matchup_features.csv.
        artifacts_dir: Directory with margin_model.pkl and features pkl.

    Returns:
        Predicted point differential as float (positive = home favored).

    Raises:
        FileNotFoundError: If model artifacts have not been trained yet.
        ValueError: If no feature rows can be found for either team.
    """
    model_path = os.path.join(artifacts_dir, "margin_model.pkl")
    feat_path = os.path.join(artifacts_dir, "margin_model_features.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Margin model artifact not found at {model_path!r}. "
            "Run: python src/models/margin_model.py"
        )
    if not os.path.exists(feat_path):
        raise FileNotFoundError(
            f"Margin model features not found at {feat_path!r}. "
            "Run: python src/models/margin_model.py"
        )
    with open(model_path, "rb") as fh:
        model = pickle.load(fh)
    with open(feat_path, "rb") as fh:
        feat_cols = pickle.load(fh)

    df = pd.read_csv(matchup_path)
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")

    # FIX: Only consider rows from the most recent season to avoid stale
    # cross-season features that can catastrophically invert predictions.
    # Always fall back to the synthesized-row path if no current-season exact
    # matchup exists — this uses each team's latest stats rather than a frozen
    # historical snapshot.
    latest_season = int(df["season"].astype(int).max())
    current_df = df[df["season"].astype(int) == latest_season]

    exact = current_df[
        (current_df["home_team"] == home_team) & (current_df["away_team"] == away_team)
    ]
    if not exact.empty:
        row = exact.sort_values("game_date").iloc[-1].copy()
    else:
        # Synthesize from each team's most recent current-season row
        home_rows = current_df[current_df["home_team"] == home_team].sort_values("game_date")
        away_rows = current_df[current_df["away_team"] == away_team].sort_values("game_date")
        if home_rows.empty or away_rows.empty:
            raise ValueError(
                f"Not enough current-season history to build features"
                f" for {home_team} vs {away_team}."
            )
        row = home_rows.iloc[-1].copy()
        away_source = away_rows.iloc[-1]
        for col in current_df.columns:
            if col.startswith("away_"):
                row[col] = away_source.get(col, row.get(col, np.nan))
        for col in feat_cols:
            if col.startswith("diff_"):
                base = col.replace("diff_", "", 1)
                h_col, a_col = f"home_{base}", f"away_{base}"
                if h_col in row.index and a_col in row.index:
                    row[col] = row[h_col] - row[a_col]

    # Refresh Elo with current ratings (not stale CSV values)
    current_elos = get_current_elos()
    home_elo = current_elos.get(home_team, 1500.0)
    away_elo = current_elos.get(away_team, 1500.0)
    if "diff_elo" in row.index:
        row["diff_elo"] = home_elo - away_elo
    if "home_elo_pre" in row.index:
        row["home_elo_pre"] = home_elo
    if "away_elo_pre" in row.index:
        row["away_elo_pre"] = away_elo
    # Note: diff_elo_fast (K=40) and diff_elo_momentum are separate signals.
    # get_current_elos() only returns standard Elo (K=20). Leave fast/momentum
    # from the CSV row (still fresher than cross-season data). Only override
    # diff_elo which is 37.3% feature importance.
    # TODO (Phase 3): extend get_current_elos() to return fast + momentum Elo.

    row_df = row.to_frame().T.reindex(columns=feat_cols)
    prediction = float(model.predict(row_df)[0])
    return round(prediction, 2)


if __name__ == "__main__":
    _pipeline, _metrics = train_margin_model()
    print(f"\nSelected model : {_metrics['selected_model']}")
    print(f"CV MAE         : {_metrics['cv_mean_mae']:.3f}")
    print(f"Test MAE       : {_metrics['test_mae']:.3f}")
    print(f"Test RMSE      : {_metrics['test_rmse']:.3f}")
