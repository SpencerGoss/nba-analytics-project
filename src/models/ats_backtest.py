"""
ATS Backtest Harness
====================
Evaluates the trained ATS model against historical game data, reporting:
  - ROI (flat betting at standard -110 vig)
  - CLV (closing line value -- opening spread as proxy; see note below)
  - Hit rate (% of spread predictions correct)

Two backtest modes:
  1. Baseline (all games): Evaluates every non-push game the model predicts.
  2. Value-bet filtered: Evaluates only games flagged as value bets by the
     win-probability disagreement detector (edge > threshold).

Hard minimum: 500 games required for the baseline backtest. Raises ValueError
if the total non-push game count is below this threshold.

CLV Note
--------
The Kaggle dataset (game_ats_features.csv) stores a single 'spread' column
that represents the opening line. There is no separate closing line column.
As a result, CLV is computed as 0.0 for all games in this dataset (opening
vs opening is always zero movement). This is documented as a data limitation --
a closing-line dataset would be needed to measure true CLV.

Usage:
    python src/models/ats_backtest.py

    Or import:
        from src.models.ats_backtest import run_ats_backtest, write_backtest_reports
        results = run_ats_backtest()
        write_backtest_reports(results)

Output:
    reports/ats_backtest.csv         -- per-season breakdown
    reports/ats_backtest_summary.txt -- human-readable summary
"""

import json
import os
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# -- Config -------------------------------------------------------------------

MIN_BACKTEST_GAMES = 500
ATS_FEATURES_PATH = "data/features/game_ats_features.csv"
ARTIFACTS_DIR = "models/artifacts"
REPORTS_DIR = "reports"
TARGET = "covers_spread"
VALUE_BET_THRESHOLD = 0.05


# -- ROI / CLV helpers --------------------------------------------------------

def compute_roi_flat_110(covers: pd.Series) -> dict:
    """Compute ROI for flat $110 bets at standard -110 vig.

    Each winning bet returns +$100 net (risked $110 to win $100).
    Each losing bet costs $110.

    Args:
        covers: Boolean/int Series where 1=bet was correct (prediction matched
                actual outcome), 0=bet was wrong.
                NaN values are treated as pushes and excluded.

    Returns:
        dict with:
            n_bets:     int -- total bets placed (excluding pushes)
            wins:       int -- number of winning bets
            losses:     int -- number of losing bets
            hit_rate:   float -- wins / n_bets
            net_units:  float -- net profit in units of $110 stake
            roi:        float -- net_units / n_bets (e.g., 0.05 = 5% ROI)

    Raises:
        ValueError: If fewer than MIN_BACKTEST_GAMES valid bets remain.
    """
    valid = covers.dropna()
    n = len(valid)

    if n < MIN_BACKTEST_GAMES:
        raise ValueError(
            f"Backtest requires >= {MIN_BACKTEST_GAMES} games, got {n}. "
            "Ensure the dataset covers a sufficient historical range."
        )

    wins = int((valid == 1).sum())
    losses = int((valid == 0).sum())
    hit_rate = float(wins / n)

    # -110 vig arithmetic: win $100 per $110 staked
    # net_pnl per bet: +100/110 = +0.909 units on win, -1.0 unit on loss
    net_units = float(wins * (100.0 / 110.0) - losses * 1.0)
    roi = float(net_units / n)

    return {
        "n_bets": int(n),
        "wins": wins,
        "losses": losses,
        "hit_rate": round(hit_rate, 6),
        "net_units": round(net_units, 4),
        "roi": round(roi, 6),
    }


def compute_clv_spread(bet_spread, closing_spread, bet_side) -> float:
    """Compute closing line value (CLV) for a spread bet.

    CLV measures whether we got a better number than the closing line.
    Positive CLV means we bet on a more favorable spread than what closed.

    For home bets: CLV = bet_spread - closing_spread
        (positive if opening was larger/more points than close, i.e., we
        got more points when taking the home side)
    For away bets: CLV = closing_spread - bet_spread
        (positive if opening was smaller/tighter than close, i.e., we got
        a better spread as underdogs)

    NOTE: This dataset only has an opening spread, not a closing spread.
    When closing_spread equals opening_spread (no movement data), CLV = 0.0
    for every game. This is a data limitation -- a source with opening AND
    closing lines would enable meaningful CLV measurement.

    Args:
        bet_spread:     Opening spread at bet time (home perspective, e.g. -5.5)
        closing_spread: Closing spread (use opening spread as proxy if unavailable)
        bet_side:       "home" or "away"

    Returns:
        float: CLV in spread points. Positive = got better number than close.
    """
    try:
        bet_spread = float(bet_spread)
        closing_spread = float(closing_spread)
        if pd.isna(bet_spread) or pd.isna(closing_spread):
            return 0.0
        if bet_side == "home":
            return float(bet_spread - closing_spread)
        else:
            return float(closing_spread - bet_spread)
    except (TypeError, ValueError):
        return 0.0


# -- Model loading ------------------------------------------------------------

def _load_ats_model(artifacts_dir: str = ARTIFACTS_DIR):
    """Load trained ATS model pipeline, feature list, and metadata.

    Returns:
        (model, feat_cols, metadata): tuple
    """
    model_path = os.path.join(artifacts_dir, "ats_model.pkl")
    feat_path = os.path.join(artifacts_dir, "ats_model_features.pkl")
    meta_path = os.path.join(artifacts_dir, "ats_model_metadata.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"ATS model artifact not found at '{model_path}'. "
            "Run: python src/models/ats_model.py"
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(feat_path, "rb") as f:
        feat_cols = pickle.load(f)

    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)

    return model, feat_cols, metadata


# -- Per-game edge computation ------------------------------------------------

def _compute_edge(row) -> float:
    """Compute edge between ATS model probability and market implied probability.

    Edge = covers_spread_prob - home_implied_prob
    Positive edge means model thinks home team covers more than market implies.

    Returns NaN if implied probability is missing.
    """
    model_prob = row.get("covers_spread_prob", float("nan"))
    market_prob = row.get("home_implied_prob", float("nan"))
    if pd.isna(model_prob) or pd.isna(market_prob):
        return float("nan")
    return float(model_prob) - float(market_prob)


# -- Per-season breakdown -----------------------------------------------------

def _compute_season_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-season metrics for the given prediction/result DataFrame.

    Args:
        df: DataFrame with covers_spread (actual), covers_spread_pred (predicted),
            covers_spread_prob (probability), spread, season, and avg_clv.

    Returns:
        DataFrame with columns: season, n_games, wins, losses, hit_rate,
        net_units, roi, avg_clv, avg_edge
    """
    rows = []
    for season, grp in df.groupby("season"):
        # Count correct predictions
        correct = (grp["covers_spread"] == grp["covers_spread_pred"]).sum()
        n = len(grp)
        wins = int(correct)
        losses = int(n - correct)
        hit_rate = float(wins / n) if n > 0 else float("nan")

        # ROI (flat -110 betting) -- skip guard for per-season (may be < 500)
        valid = grp["covers_spread"].dropna()
        n_valid = len(valid)
        covers_pred_matched = grp[grp["covers_spread"].notna()]["covers_spread"] == \
                               grp[grp["covers_spread"].notna()]["covers_spread_pred"]
        wins_roi = int(covers_pred_matched.sum())
        losses_roi = int(n_valid - wins_roi)
        net_units = float(wins_roi * (100.0 / 110.0) - losses_roi * 1.0)
        roi = float(net_units / n_valid) if n_valid > 0 else float("nan")

        avg_clv = float(grp["clv"].mean()) if "clv" in grp.columns else 0.0
        avg_edge = float(grp["edge"].mean()) if "edge" in grp.columns else float("nan")

        rows.append({
            "season": str(season),
            "n_games": int(n),
            "wins": wins_roi,
            "losses": losses_roi,
            "hit_rate": round(hit_rate, 6),
            "net_units": round(net_units, 4),
            "roi": round(roi, 6),
            "avg_clv": round(avg_clv, 6),
            "avg_edge": round(avg_edge, 6) if not pd.isna(avg_edge) else float("nan"),
        })

    return pd.DataFrame(rows).sort_values("season").reset_index(drop=True)


# -- Main backtest function ---------------------------------------------------

def run_ats_backtest(
    ats_features_path: str = ATS_FEATURES_PATH,
    artifacts_dir: str = ARTIFACTS_DIR,
    value_bet_threshold: float = VALUE_BET_THRESHOLD,
) -> dict:
    """Run the full ATS backtest against historical game data.

    Evaluates the trained ATS model on ALL non-push historical games (baseline)
    and on the subset flagged as value bets by edge detection (value-bet mode).

    The backtest uses ALL available games including training seasons -- this is
    intentional for the purposes of this historical performance report. The model
    was selected using expanding-window cross-validation so in-sample results
    are expected to be optimistic. The test holdout seasons (202324, 202425) are
    explicitly called out in the summary for honest out-of-sample comparison.

    Args:
        ats_features_path: Path to game_ats_features.csv.
        artifacts_dir: Directory containing ats_model.pkl and related artifacts.
        value_bet_threshold: Minimum |edge| to flag as a value bet. Default 0.05.

    Returns:
        dict with:
            baseline: dict -- overall metrics for all games
            baseline_by_season: DataFrame -- per-season breakdown (all games)
            value_bet: dict or None -- overall metrics for value-bet subset
            value_bet_by_season: DataFrame or None -- per-season for value bets
            metadata: dict -- model metadata
            n_total_games: int -- total games before push exclusion
            n_push_games: int -- number of push/unknown games excluded
            n_backtest_games: int -- games used in baseline backtest
            n_value_bet_games: int -- games in value-bet filtered backtest
            test_seasons: list -- holdout seasons (not used in model training)
            clv_note: str -- explanation of CLV data limitation
    """
    print("=" * 65)
    print("ATS BACKTEST HARNESS")
    print("=" * 65)

    # -- Load data ----------------------------------------------------------
    print(f"\nLoading ATS features from {ats_features_path}...")
    df = pd.read_csv(ats_features_path, low_memory=False)
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")
    df = df.sort_values("game_date").reset_index(drop=True)
    n_total = len(df)
    print(f"  Total games loaded: {n_total:,}")

    # -- Drop push rows (NaN covers_spread) ---------------------------------
    n_before_push = len(df)
    df = df.dropna(subset=[TARGET])
    n_push = n_before_push - len(df)
    print(f"  Push/unknown games excluded: {n_push:,}")
    print(f"  Games for backtest: {len(df):,}")

    # -- Load ATS model -----------------------------------------------------
    print("\nLoading ATS model...")
    model, feat_cols, metadata = _load_ats_model(artifacts_dir)
    threshold = metadata.get("threshold", 0.50)
    model_type = metadata.get("model_type", "unknown")
    n_features = metadata.get("n_features", len(feat_cols))
    training_date = metadata.get("training_date", "unknown")
    test_seasons = metadata.get("test_seasons", [])
    print(f"  Model type: {model_type}")
    print(f"  Features: {n_features}")
    print(f"  Decision threshold: {threshold}")
    print(f"  Training date: {training_date}")
    print(f"  Holdout test seasons: {test_seasons}")

    # -- Generate predictions for all games ----------------------------------
    print("\nGenerating ATS predictions...")
    X = df.reindex(columns=feat_cols)
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    df = df.copy()
    df["covers_spread_prob"] = proba
    df["covers_spread_pred"] = pred

    # -- Compute edge (model prob vs market implied prob) -------------------
    df["edge"] = df.apply(_compute_edge, axis=1)

    # -- Compute CLV (opening spread as proxy for closing) ------------------
    # CLV = 0.0 for all games since no separate closing line exists
    # bet_side: if model predicts home covers (pred=1), bet home; else bet away
    df["bet_side"] = df["covers_spread_pred"].apply(
        lambda p: "home" if p == 1 else "away"
    )
    # closing_spread = opening spread (no movement data available)
    df["clv"] = df.apply(
        lambda row: compute_clv_spread(
            bet_spread=row.get("spread", float("nan")),
            closing_spread=row.get("spread", float("nan")),  # same = no closing data
            bet_side=row["bet_side"],
        ),
        axis=1,
    )

    print(f"  Predictions generated for {len(df):,} games")
    # bet_correct: 1 if prediction matched actual outcome, 0 otherwise
    # This is what we pass to compute_roi_flat_110 -- we bet on every game
    # and "win" the bet when our prediction is correct.
    df["bet_correct"] = (df["covers_spread"] == df["covers_spread_pred"]).astype(int)
    correct = int(df["bet_correct"].sum())
    overall_hit_rate = float(correct / len(df))
    print(f"  Overall hit rate (all games): {overall_hit_rate:.4f} ({overall_hit_rate:.1%})")

    # -- Baseline backtest (all games) ----------------------------------------
    print("\n--- Baseline Backtest (all non-push games) ---")
    n_backtest = len(df)
    baseline_roi = compute_roi_flat_110(df["bet_correct"])  # raises if < 500
    baseline_by_season = _compute_season_breakdown(df)
    avg_clv_baseline = float(df["clv"].mean())
    avg_edge_baseline = float(df["edge"].dropna().mean())

    print(f"  Games: {baseline_roi['n_bets']:,}")
    print(f"  Hit rate: {baseline_roi['hit_rate']:.4f} ({baseline_roi['hit_rate']:.1%})")
    print(f"  Net units: {baseline_roi['net_units']:.2f}")
    print(f"  ROI: {baseline_roi['roi']:.4f} ({baseline_roi['roi']:.1%})")
    print(f"  Avg CLV: {avg_clv_baseline:.4f} (NOTE: 0.0 -- no closing line data)")
    print(f"  Avg edge: {avg_edge_baseline:.4f}")

    baseline_result = {
        **baseline_roi,
        "avg_clv": round(avg_clv_baseline, 6),
        "avg_edge": round(avg_edge_baseline, 6),
    }

    # -- Value-bet filtered backtest -----------------------------------------
    print(f"\n--- Value-Bet Filtered Backtest (edge > {value_bet_threshold:.2%}) ---")
    value_df = df[df["edge"].abs() > value_bet_threshold].copy()
    n_value_bets = len(value_df)
    print(f"  Value-bet games: {n_value_bets:,} of {n_backtest:,} "
          f"({n_value_bets / n_backtest:.1%})")

    value_bet_result = None
    value_bet_by_season = None

    if n_value_bets >= MIN_BACKTEST_GAMES:
        vb_roi = compute_roi_flat_110(value_df["bet_correct"])
        avg_clv_vb = float(value_df["clv"].mean())
        avg_edge_vb = float(value_df["edge"].dropna().mean())
        value_bet_by_season = _compute_season_breakdown(value_df)

        print(f"  Hit rate: {vb_roi['hit_rate']:.4f} ({vb_roi['hit_rate']:.1%})")
        print(f"  Net units: {vb_roi['net_units']:.2f}")
        print(f"  ROI: {vb_roi['roi']:.4f} ({vb_roi['roi']:.1%})")
        print(f"  Avg CLV: {avg_clv_vb:.4f}")
        print(f"  Avg edge: {avg_edge_vb:.4f}")

        value_bet_result = {
            **vb_roi,
            "avg_clv": round(avg_clv_vb, 6),
            "avg_edge": round(avg_edge_vb, 6),
        }
    else:
        print(f"  NOTE: Only {n_value_bets} value-bet games -- below {MIN_BACKTEST_GAMES} "
              f"game minimum. Value-bet backtest skipped for primary report.")
        print(f"  Reporting limited metrics (no ROI guard applied to subset).")
        if n_value_bets > 0:
            vb_correct = int(value_df["bet_correct"].sum())
            vb_hit_rate = float(vb_correct / n_value_bets)
            avg_edge_vb = float(value_df["edge"].dropna().mean())
            print(f"  Hit rate (unreported, small sample): {vb_hit_rate:.4f}")
            value_bet_result = {
                "n_bets": int(n_value_bets),
                "wins": int(vb_correct),
                "losses": int(n_value_bets - vb_correct),
                "hit_rate": round(vb_hit_rate, 6),
                "net_units": float("nan"),
                "roi": float("nan"),
                "avg_clv": 0.0,
                "avg_edge": round(avg_edge_vb, 6),
                "note": f"Only {n_value_bets} games (below {MIN_BACKTEST_GAMES} minimum). ROI not reported.",
            }

    # -- Holdout-only metrics (honest out-of-sample) -------------------------
    print(f"\n--- Holdout Seasons Only ({test_seasons}) ---")
    holdout_df = df[df["season"].astype(int).isin(test_seasons)].copy()
    if len(holdout_df) > 0:
        h_correct = int(holdout_df["bet_correct"].sum())
        h_hit_rate = float(h_correct / len(holdout_df))
        print(f"  Holdout games: {len(holdout_df):,}")
        print(f"  Holdout hit rate: {h_hit_rate:.4f} ({h_hit_rate:.1%})")
        if len(holdout_df) >= MIN_BACKTEST_GAMES:
            holdout_roi = compute_roi_flat_110(holdout_df["bet_correct"])
            print(f"  Holdout ROI: {holdout_roi['roi']:.4f} ({holdout_roi['roi']:.1%})")
        else:
            holdout_roi = {"roi": float("nan"), "hit_rate": h_hit_rate, "n_bets": len(holdout_df)}
            print(f"  NOTE: {len(holdout_df)} holdout games -- ROI guard requires {MIN_BACKTEST_GAMES}.")
    else:
        holdout_roi = {}
        h_hit_rate = float("nan")
        print("  No holdout games found.")

    clv_note = (
        "CLV is 0.0 for all games. The game_ats_features.csv dataset stores only an "
        "opening spread column ('spread'). No separate closing line is available, "
        "so opening-vs-opening CLV is always zero. A dataset with opening AND closing "
        "lines would enable true CLV measurement."
    )

    return {
        "baseline": baseline_result,
        "baseline_by_season": baseline_by_season,
        "value_bet": value_bet_result,
        "value_bet_by_season": value_bet_by_season,
        "holdout_roi": holdout_roi,
        "holdout_hit_rate": h_hit_rate,
        "metadata": metadata,
        "n_total_games": int(n_total),
        "n_push_games": int(n_push),
        "n_backtest_games": int(n_backtest),
        "n_value_bet_games": int(n_value_bets),
        "value_bet_threshold": float(value_bet_threshold),
        "test_seasons": test_seasons,
        "clv_note": clv_note,
        "vig_breakeven": 0.5238,
    }


# -- Report generation --------------------------------------------------------

def write_backtest_reports(results: dict, reports_dir: str = REPORTS_DIR) -> None:
    """Write ATS backtest results to reports directory.

    Creates two files:
      reports/ats_backtest.csv         -- per-season breakdown (baseline)
      reports/ats_backtest_summary.txt -- human-readable summary

    Args:
        results: dict returned by run_ats_backtest()
        reports_dir: output directory (created if it doesn't exist)
    """
    os.makedirs(reports_dir, exist_ok=True)

    # -- Per-season CSV -------------------------------------------------------
    csv_path = os.path.join(reports_dir, "ats_backtest.csv")
    baseline_by_season = results["baseline_by_season"]
    baseline_by_season.to_csv(csv_path, index=False)
    print(f"\nPer-season CSV saved -> {csv_path}")

    # -- Human-readable summary -----------------------------------------------
    txt_path = os.path.join(reports_dir, "ats_backtest_summary.txt")
    baseline = results["baseline"]
    value_bet = results.get("value_bet") or {}
    metadata = results.get("metadata", {})
    test_seasons = results.get("test_seasons", [])
    vig_breakeven = results.get("vig_breakeven", 0.5238)

    lines = [
        "ATS BACKTEST REPORT",
        "=" * 65,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "SUMMARY",
        "-" * 65,
        f"  Total games loaded:          {results['n_total_games']:,}",
        f"  Push/unknown games excluded: {results['n_push_games']:,}",
        f"  Games in baseline backtest:  {results['n_backtest_games']:,}",
        f"  Value-bet games:             {results['n_value_bet_games']:,} "
        f"(edge > {results['value_bet_threshold']:.2%})",
        f"  Holdout seasons (OOS):       {', '.join(str(s) for s in test_seasons)}",
        "",
        "BASELINE BACKTEST (All non-push games)",
        "-" * 65,
        f"  Total bets:     {baseline['n_bets']:,}",
        f"  Wins:           {baseline['wins']:,}",
        f"  Losses:         {baseline['losses']:,}",
        f"  Hit rate:       {baseline['hit_rate']:.4f} ({baseline['hit_rate']:.2%})",
        f"  Vig breakeven:  {vig_breakeven:.4f} ({vig_breakeven:.2%})",
        f"  Net units:      {baseline['net_units']:.2f}",
        f"  ROI:            {baseline['roi']:.4f} ({baseline['roi']:.2%})",
        f"  Avg CLV:        {baseline['avg_clv']:.4f}",
        f"  Avg edge:       {baseline['avg_edge']:.4f}",
        "",
    ]

    # Holdout section
    holdout_roi = results.get("holdout_roi", {})
    holdout_hit_rate = results.get("holdout_hit_rate", float("nan"))
    if holdout_roi:
        lines += [
            "HOLDOUT BACKTEST (Out-of-sample test seasons only)",
            "-" * 65,
            f"  Test seasons:   {', '.join(str(s) for s in test_seasons)}",
            f"  Games:          {holdout_roi.get('n_bets', '?'):,}" if isinstance(holdout_roi.get('n_bets'), int) else f"  Games:          {holdout_roi.get('n_bets', '?')}",
        ]
        if not pd.isna(holdout_hit_rate):
            lines.append(f"  Hit rate:       {holdout_hit_rate:.4f} ({holdout_hit_rate:.2%})")
        if not pd.isna(holdout_roi.get("roi", float("nan"))):
            lines.append(f"  ROI:            {holdout_roi['roi']:.4f} ({holdout_roi['roi']:.2%})")
        else:
            lines.append(f"  ROI:            N/A (sample too small for guard)")
        lines.append("")

    # Value-bet section
    if value_bet:
        lines += [
            "VALUE-BET FILTERED BACKTEST",
            "-" * 65,
            f"  Threshold:      edge > {results['value_bet_threshold']:.2%}",
            f"  Total bets:     {value_bet['n_bets']:,}",
        ]
        if "note" in value_bet:
            lines.append(f"  NOTE:           {value_bet['note']}")
        else:
            lines += [
                f"  Wins:           {value_bet['wins']:,}",
                f"  Losses:         {value_bet['losses']:,}",
                f"  Hit rate:       {value_bet['hit_rate']:.4f} ({value_bet['hit_rate']:.2%})",
                f"  Net units:      {value_bet['net_units']:.2f}",
                f"  ROI:            {value_bet['roi']:.4f} ({value_bet['roi']:.2%})",
                f"  Avg CLV:        {value_bet['avg_clv']:.4f}",
                f"  Avg edge:       {value_bet['avg_edge']:.4f}",
            ]
        lines.append("")

    # Per-season table
    lines += [
        "PER-SEASON BREAKDOWN (Baseline)",
        "-" * 65,
        f"{'Season':<10} {'Games':>6} {'Wins':>5} {'Losses':>7} "
        f"{'Hit Rate':>10} {'ROI':>8} {'Avg Edge':>10}",
        "-" * 65,
    ]
    for _, row in baseline_by_season.iterrows():
        season_tag = str(row["season"])
        oos_flag = " *" if season_tag in [str(s) for s in test_seasons] else "  "
        edge_str = f"{row['avg_edge']:.4f}" if not pd.isna(row.get("avg_edge", float("nan"))) else "   N/A"
        lines.append(
            f"{season_tag:<10} {row['n_games']:>6,} {row['wins']:>5,} {row['losses']:>7,} "
            f"{row['hit_rate']:>10.2%} {row['roi']:>8.4f} {edge_str:>10}{oos_flag}"
        )

    lines += [
        "",
        "  * = Out-of-sample holdout season (not used in model training)",
        "",
        "CLV NOTE",
        "-" * 65,
        results.get("clv_note", ""),
        "",
        "MODEL METADATA",
        "-" * 65,
        f"  Model type:         {metadata.get('model_type', 'unknown')}",
        f"  n_features:         {metadata.get('n_features', '?')}",
        f"  Training date:      {metadata.get('training_date', 'unknown')}",
        f"  Train rows:         {metadata.get('n_train_rows', '?'):,}" if isinstance(metadata.get('n_train_rows'), int) else f"  Train rows:         {metadata.get('n_train_rows', '?')}",
        f"  Test rows:          {metadata.get('n_test_rows', '?'):,}" if isinstance(metadata.get('n_test_rows'), int) else f"  Test rows:          {metadata.get('n_test_rows', '?')}",
        f"  Test accuracy:      {metadata.get('test_accuracy', 'unknown'):.4f}" if isinstance(metadata.get('test_accuracy'), float) else f"  Test accuracy:      {metadata.get('test_accuracy', 'unknown')}",
        f"  Test ROC-AUC:       {metadata.get('test_auc', 'unknown'):.4f}" if isinstance(metadata.get('test_auc'), float) else f"  Test ROC-AUC:       {metadata.get('test_auc', 'unknown')}",
        f"  Decision threshold: {metadata.get('threshold', 0.50):.2f}",
        f"  Val mean accuracy:  {metadata.get('validation_mean_accuracy', 'unknown'):.4f}" if isinstance(metadata.get('validation_mean_accuracy'), float) else f"  Val mean accuracy:  {metadata.get('validation_mean_accuracy', 'unknown')}",
        f"  Holdout seasons:    {', '.join(str(s) for s in test_seasons)}",
        "",
        "INTERPRETATION",
        "-" * 65,
        f"  ATS vig breakeven is ~52.38% at -110 standard lines.",
        f"  Hit rates below 52.38% lose money at standard vig.",
        f"  This model's overall hit rate of {baseline['hit_rate']:.2%} "
        f"{'is above' if baseline['hit_rate'] > 0.5238 else 'is below'} breakeven.",
        f"  The 51.2% test accuracy reflects the known difficulty of ATS prediction",
        f"  (Vegas lines are set to be near-50/50 for bettors).",
    ]

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Summary report saved  -> {txt_path}")


# -- Entry point --------------------------------------------------------------

if __name__ == "__main__":
    results = run_ats_backtest()
    write_backtest_reports(results)

    # Print key metrics to console
    baseline = results["baseline"]
    print("\n" + "=" * 65)
    print("KEY METRICS")
    print("=" * 65)
    print(f"  Total games backtested:  {results['n_backtest_games']:,}")
    print(f"  Overall hit rate:        {baseline['hit_rate']:.4f} ({baseline['hit_rate']:.2%})")
    print(f"  Overall ROI (-110 flat): {baseline['roi']:.4f} ({baseline['roi']:.2%})")
    print(f"  Value-bet games flagged: {results['n_value_bet_games']:,}")
    vb = results.get("value_bet") or {}
    if vb and "roi" in vb and not pd.isna(vb.get("roi", float("nan"))):
        print(f"  Value-bet hit rate:      {vb['hit_rate']:.4f} ({vb['hit_rate']:.2%})")
        print(f"  Value-bet ROI:           {vb['roi']:.4f} ({vb['roi']:.2%})")
    elif vb and "hit_rate" in vb:
        print(f"  Value-bet hit rate:      {vb['hit_rate']:.4f} (small sample -- ROI not reported)")
    holdout_hr = results.get("holdout_hit_rate", float("nan"))
    if not pd.isna(holdout_hr):
        print(f"  Holdout hit rate (OOS):  {holdout_hr:.4f} ({holdout_hr:.2%})")
