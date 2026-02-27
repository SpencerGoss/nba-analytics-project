"""
Walk-Forward Backtesting Framework
=====================================
Evaluates prediction models by rolling forward one season at a time —
training on all history up to season N, testing on season N+1.

This is the gold standard for time-series model evaluation because it
simulates exactly how the model would have performed in real deployment:
you never peek at future data, and you retrain after each season just as
you would in production.

Why this matters over a single train/test split:
  - A single split only gives you accuracy on 2 seasons.
  - Walk-forward gives you per-season accuracy curves, letting you spot
    if performance is degrading over time, improving, or unstable.
  - It surfaces era effects — e.g., did accuracy drop when the 3-point
    revolution changed team styles?

Usage:
    python src/models/backtesting.py

    Or import specific runners:
        from src.models.backtesting import (
            run_game_outcome_backtest,
            run_player_model_backtest,
        )

Output:
    reports/backtest_game_outcome.csv   — per-season metrics
    reports/backtest_player_pts.csv     — per-season MAE for pts
    reports/backtest_player_reb.csv     — per-season MAE for reb
    reports/backtest_player_ast.csv     — per-season MAE for ast
    reports/backtest_summary.txt        — human-readable summary
"""

import os
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    mean_absolute_error, root_mean_squared_error,
    brier_score_loss,
)

# ── Config ─────────────────────────────────────────────────────────────────────

MATCHUP_PATH     = "data/features/game_matchup_features.csv"
PLAYER_PATH      = "data/features/player_game_features.csv"
REPORTS_DIR      = "reports"

# Minimum seasons in the training window before we start testing.
# Fewer than this and the model is seeing only a handful of seasons.
MIN_TRAIN_SEASONS = 5

# For the player model, we only backtest the modern era (data is richer).
# Full historical player game logs only exist from ~2000 onward.
PLAYER_BACKTEST_START = "200001"

# Restrict game outcome backtest to modern era — pre-2000 seasons have different
# statistical distributions that add noise for predicting modern games.
GAME_BACKTEST_START = "200001"

# Minimum games played in a training set for a player to be included.
MIN_PLAYER_GAMES = 20

TARGET_CLASS  = "home_win"
PLAYER_TARGETS = ["pts", "reb", "ast"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sorted_seasons(df: pd.DataFrame, col: str = "season") -> list:
    return sorted(df[col].astype(str).unique())


def _get_feature_cols_game(df: pd.DataFrame) -> list:
    exclude = {TARGET_CLASS, "game_id", "season", "game_date", "home_team", "away_team"}
    return [
        c for c in df.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def _get_feature_cols_player(df: pd.DataFrame, target: str) -> list:
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
    return [
        c for c in df.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def _build_gb_classifier() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("clf",     GradientBoostingClassifier(
            n_estimators=100,        # faster than 200 for backtesting
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )),
    ])


def _build_gb_regressor() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("model",   GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            loss="squared_error",
        )),
    ])


# ── Game outcome walk-forward ──────────────────────────────────────────────────

def run_game_outcome_backtest(
    matchup_path: str = MATCHUP_PATH,
    reports_dir: str  = REPORTS_DIR,
    min_train:    int = MIN_TRAIN_SEASONS,
) -> pd.DataFrame:
    """
    Walk-forward backtest for the game outcome classifier.

    For each season after the minimum training window:
      - Train on all prior seasons
      - Test on this season
      - Record accuracy, ROC-AUC, Brier score, and n_games

    Returns a DataFrame indexed by test season.
    """
    print("=" * 65)
    print("WALK-FORWARD BACKTEST — Game Outcome Model")
    print("=" * 65)

    df = pd.read_csv(matchup_path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["season"]    = df["season"].astype(str)
    df = df.sort_values("game_date").reset_index(drop=True)
    df = df.dropna(subset=[TARGET_CLASS])           # drop unplayed / future games
    df = df[df["season"] >= GAME_BACKTEST_START]    # modern era only

    seasons     = _sorted_seasons(df)
    feat_cols   = _get_feature_cols_game(df)
    results     = []

    print(f"\nTotal seasons available : {len(seasons)}  ({seasons[0]} → {seasons[-1]})")
    print(f"Minimum training window : {min_train} seasons")
    print(f"Test seasons            : {len(seasons) - min_train}")
    print(f"Features                : {len(feat_cols)}")
    print()
    print(f"{'Season':<10} {'Train Games':>12} {'Test Games':>11} "
          f"{'Accuracy':>10} {'ROC-AUC':>9} {'Brier':>8}")
    print("─" * 65)

    for i in range(min_train, len(seasons)):
        train_seasons = seasons[:i]
        test_season   = seasons[i]

        train = df[df["season"].isin(train_seasons)]
        test  = df[df["season"] == test_season]

        if len(test) < 50:          # skip partial / lockout seasons
            continue
        if TARGET_CLASS not in test.columns:
            continue

        X_train = train[feat_cols]
        y_train = train[TARGET_CLASS]
        X_test  = test[feat_cols]
        y_test  = test[TARGET_CLASS]

        model = _build_gb_classifier()
        model.fit(X_train, y_train)

        pred  = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        acc    = accuracy_score(y_test, pred)
        auc    = roc_auc_score(y_test, proba)
        brier  = brier_score_loss(y_test, proba)

        results.append({
            "test_season":  test_season,
            "n_train":      len(train),
            "n_test":       len(test),
            "accuracy":     round(acc, 4),
            "roc_auc":      round(auc, 4),
            "brier_score":  round(brier, 4),
        })

        print(f"{test_season:<10} {len(train):>12,} {len(test):>11,} "
              f"{acc:>10.3%} {auc:>9.4f} {brier:>8.4f}")

    results_df = pd.DataFrame(results)

    # ── Summary stats ──────────────────────────────────────────────────────────
    print("\n── Summary ─────────────────────────────────────────────────────")
    print(f"  Mean accuracy  : {results_df['accuracy'].mean():.3%}  "
          f"(σ = {results_df['accuracy'].std():.4f})")
    print(f"  Mean ROC-AUC   : {results_df['roc_auc'].mean():.4f}  "
          f"(σ = {results_df['roc_auc'].std():.4f})")
    print(f"  Mean Brier     : {results_df['brier_score'].mean():.4f}  "
          f"(σ = {results_df['brier_score'].std():.4f})")
    print(f"  Best season    : {results_df.loc[results_df['accuracy'].idxmax(), 'test_season']}  "
          f"({results_df['accuracy'].max():.3%})")
    print(f"  Worst season   : {results_df.loc[results_df['accuracy'].idxmin(), 'test_season']}  "
          f"({results_df['accuracy'].min():.3%})")

    # ── Era-level breakdown ────────────────────────────────────────────────────
    try:
        from src.features.era_labels import label_eras
        results_df["era_num"] = None
        results_df["era_name"] = None
        for idx, row in results_df.iterrows():
            from src.features.era_labels import get_era
            era = get_era(row["test_season"])
            results_df.at[idx, "era_num"]  = era["era_num"]
            results_df.at[idx, "era_name"] = era["era_name"]

        print("\n── Accuracy by Era ─────────────────────────────────────────────")
        era_summary = (
            results_df.groupby(["era_num", "era_name"])["accuracy"]
            .agg(["mean", "std", "count"])
            .rename(columns={"mean": "avg_acc", "std": "std_acc", "count": "n_seasons"})
        )
        for (era_num, era_name), row in era_summary.iterrows():
            print(f"  Era {era_num} — {era_name:<25}: "
                  f"{row['avg_acc']:.3%}  (σ={row['std_acc']:.4f}, n={int(row['n_seasons'])})")
    except Exception:
        pass

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(reports_dir, exist_ok=True)
    out_path = os.path.join(reports_dir, "backtest_game_outcome.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved → {out_path}")

    return results_df


# ── Player model walk-forward ──────────────────────────────────────────────────

def run_player_model_backtest(
    player_path:  str  = PLAYER_PATH,
    reports_dir:  str  = REPORTS_DIR,
    targets:      list = PLAYER_TARGETS,
    start_season: str  = PLAYER_BACKTEST_START,
) -> dict:
    """
    Walk-forward backtest for the player performance regression models.

    For each target stat (pts, reb, ast):
      - Walk forward season by season from start_season
      - Train on all prior seasons (within the available data)
      - Test on the current season
      - Record MAE, RMSE, and baseline (mean predictor) MAE

    Returns:
        dict of {target: results_DataFrame}
    """
    print("\n" + "=" * 65)
    print("WALK-FORWARD BACKTEST — Player Performance Models")
    print("=" * 65)

    df = pd.read_csv(player_path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["season"]    = df["season"].astype(str)
    df = df.sort_values("game_date").reset_index(drop=True)

    all_seasons = _sorted_seasons(df)
    # Only backtest from start_season forward
    backtest_seasons = [s for s in all_seasons if s >= start_season]

    print(f"\nBacktest seasons: {backtest_seasons[0]} → {backtest_seasons[-1]}  "
          f"({len(backtest_seasons)} seasons)")

    all_results = {}

    for target in targets:
        print(f"\n{'─'*65}")
        print(f"Target: {target.upper()}")
        print(f"{'Season':<10} {'Train Rows':>12} {'Test Rows':>10} "
              f"{'MAE':>8} {'RMSE':>8} {'Baseline MAE':>14}")
        print("─" * 65)

        feat_cols = _get_feature_cols_player(df, target)
        results   = []

        for i, test_season in enumerate(backtest_seasons):
            train_seasons = [s for s in all_seasons if s < test_season]
            if len(train_seasons) < MIN_TRAIN_SEASONS:
                continue

            train = df[df["season"].isin(train_seasons)].copy()
            test  = df[df["season"] == test_season].copy()

            # Filter to players with enough training history
            train_counts = train.groupby("player_id")["game_id"].transform("count")
            train = train[train_counts >= MIN_PLAYER_GAMES]

            # Drop rows where target is NaN
            train = train.dropna(subset=[target])
            test  = test.dropna(subset=[target])

            if len(test) < 100:
                continue

            X_train = train[feat_cols]
            y_train = train[target]
            X_test  = test[feat_cols]
            y_test  = test[target]

            model = _build_gb_regressor()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            mae      = mean_absolute_error(y_test, preds)
            rmse     = root_mean_squared_error(y_test, preds)
            baseline = mean_absolute_error(y_test, np.full(len(y_test), y_train.mean()))

            results.append({
                "test_season":   test_season,
                "n_train":       len(train),
                "n_test":        len(test),
                "mae":           round(mae, 4),
                "rmse":          round(rmse, 4),
                "baseline_mae":  round(baseline, 4),
                "mae_vs_baseline": round(baseline - mae, 4),
            })

            print(f"{test_season:<10} {len(train):>12,} {len(test):>10,} "
                  f"{mae:>8.3f} {rmse:>8.3f} {baseline:>14.3f}")

        results_df = pd.DataFrame(results)
        all_results[target] = results_df

        if len(results_df) > 0:
            print(f"\n  Mean MAE       : {results_df['mae'].mean():.3f}  "
                  f"(σ = {results_df['mae'].std():.4f})")
            print(f"  Mean RMSE      : {results_df['rmse'].mean():.3f}")
            print(f"  Avg vs baseline: +{results_df['mae_vs_baseline'].mean():.3f} MAE improvement")

        # ── Save ──────────────────────────────────────────────────────────────
        os.makedirs(reports_dir, exist_ok=True)
        out_path = os.path.join(reports_dir, f"backtest_player_{target}.csv")
        results_df.to_csv(out_path, index=False)
        print(f"  Saved → {out_path}")

    return all_results


# ── Summary report ─────────────────────────────────────────────────────────────

def write_summary_report(
    game_results:   pd.DataFrame,
    player_results: dict,
    reports_dir:    str = REPORTS_DIR,
) -> None:
    """Write a human-readable summary of all backtest results."""
    os.makedirs(reports_dir, exist_ok=True)
    path = os.path.join(reports_dir, "backtest_summary.txt")

    lines = [
        "NBA ANALYTICS PROJECT — BACKTESTING SUMMARY",
        "=" * 60,
        "",
        "GAME OUTCOME MODEL (Walk-Forward by Season)",
        "─" * 60,
        f"  Seasons evaluated   : {len(game_results)}",
        f"  Mean accuracy       : {game_results['accuracy'].mean():.3%}",
        f"  Mean ROC-AUC        : {game_results['roc_auc'].mean():.4f}",
        f"  Mean Brier score    : {game_results['brier_score'].mean():.4f}",
        f"  Accuracy range      : {game_results['accuracy'].min():.3%} → "
                                  f"{game_results['accuracy'].max():.3%}",
        "",
    ]

    for target, df in player_results.items():
        if len(df) == 0:
            continue
        lines += [
            f"PLAYER MODEL — {target.upper()}",
            "─" * 60,
            f"  Seasons evaluated   : {len(df)}",
            f"  Mean MAE            : {df['mae'].mean():.3f}",
            f"  Mean RMSE           : {df['rmse'].mean():.3f}",
            f"  Avg improvement vs baseline MAE : {df['mae_vs_baseline'].mean():.3f}",
            "",
        ]

    lines += [
        "NOTE: Walk-forward validation trains on all history up to",
        "season N and tests on season N alone — simulating real-world",
        "deployment. Results are generally more conservative than a",
        "single train/test split.",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nSummary report saved → {path}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    game_results   = run_game_outcome_backtest()
    player_results = run_player_model_backtest()
    write_summary_report(game_results, player_results)
