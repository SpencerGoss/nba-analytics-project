"""
build_accuracy_history.py -- generate dashboard/data/accuracy_history.json

Combines:
  1. Season-level backtest data from reports/backtest_game_outcome.csv
     (expanding-window CV results, one row per test season)
  2. Live game-by-game accuracy from predictions_history.db game_predictions
     (actual_home_win filled by backfill_outcomes.py)

The accuracy_history.json format expected by build_performance.py:
[
  {
    "date": "2026-01-03",
    "daily_accuracy": 0.625,
    "rolling_accuracy": 0.671,
    "games": 8,
    "correct": 5,
    "cumulative_games": 8
  },
  ...
]

Run: python scripts/build_accuracy_history.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BACKTEST_CSV = PROJECT_ROOT / "reports" / "backtest_game_outcome.csv"
DB_PATH = PROJECT_ROOT / "database" / "predictions_history.db"
OUT_JSON = PROJECT_ROOT / "dashboard" / "data" / "accuracy_history.json"

ROLLING_WINDOW = 14  # games for rolling accuracy


def load_live_predictions() -> pd.DataFrame:
    """Load resolved predictions from predictions_history.db."""
    if not DB_PATH.exists():
        return pd.DataFrame()
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """SELECT game_date, home_win_prob, actual_home_win
           FROM game_predictions
           WHERE actual_home_win IS NOT NULL
           ORDER BY game_date""",
        con,
    )
    con.close()
    return df


def load_backtest_seasons() -> pd.DataFrame:
    """Load season-level backtest accuracy from reports/."""
    if not BACKTEST_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(BACKTEST_CSV)
    return df


def build_history() -> list[dict]:
    """
    Build game-by-game accuracy history.

    If live predictions exist, use them for recent dates.
    For historical context, synthesise daily rows from the most recent
    backtest season result (repeating the season accuracy as a stable baseline).
    """
    live = load_live_predictions()
    backtest = load_backtest_seasons()

    results: list[dict] = []
    cumulative_games = 0
    cumulative_correct = 0

    # --- Use live predictions (game-by-game) ---
    if not live.empty:
        live["game_date"] = pd.to_datetime(live["game_date"], format="mixed")
        live["correct"] = (
            ((live["home_win_prob"] >= 0.5) & (live["actual_home_win"] == 1)) |
            ((live["home_win_prob"] < 0.5) & (live["actual_home_win"] == 0))
        ).astype(int)
        live = live.sort_values("game_date")

        for date_val, day_df in live.groupby(live["game_date"].dt.date):
            games = len(day_df)
            correct = int(day_df["correct"].sum())
            daily_acc = correct / games if games > 0 else 0.0
            cumulative_games += games
            cumulative_correct += correct
            rolling_acc = cumulative_correct / cumulative_games

            results.append({
                "date": str(date_val),
                "daily_accuracy": round(daily_acc, 4),
                "rolling_accuracy": round(rolling_acc, 4),
                "games": games,
                "correct": correct,
                "cumulative_games": cumulative_games,
            })

    # --- Supplement with backtest context if few live games ---
    if len(results) < 5 and not backtest.empty:
        # Use the last 3 backtest seasons as synthetic "weekly" accuracy points
        # to give the performance chart meaningful historical data
        recent = backtest.sort_values("test_season").tail(5)
        import datetime as dt
        # Spread them across the current season timeline (Oct 2025 - Mar 2026)
        start = dt.date(2025, 10, 22)
        total_weeks = 20
        week_delta = dt.timedelta(days=7)
        games_per_week = 1230 // 30  # approx games per week across season

        synthetic: list[dict] = []
        week_date = start
        for i, (_, row) in enumerate(recent.iterrows()):
            acc = float(row["accuracy"])
            games_week = games_per_week
            correct_week = round(acc * games_week)
            cumulative_games_s = (i + 1) * games_week
            cumulative_correct_s = round(acc * cumulative_games_s)
            synthetic.append({
                "date": str(week_date),
                "daily_accuracy": round(acc, 4),
                "rolling_accuracy": round(acc, 4),
                "games": games_week,
                "correct": correct_week,
                "cumulative_games": cumulative_games_s,
                "backtest": True,
            })
            week_date += week_delta * 4  # roughly monthly

        # Only add synthetic rows before the earliest live prediction
        if results:
            first_live_date = results[0]["date"]
            synthetic = [r for r in synthetic if r["date"] < first_live_date]

        results = synthetic + results

    return results


def main() -> None:
    print("Building accuracy history...")
    history = build_history()
    print(f"  {len(history)} entries ({sum(1 for r in history if r.get('backtest'))} synthetic backtest, "
          f"{sum(1 for r in history if not r.get('backtest'))} live)")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2, ensure_ascii=False)
    print(f"  Written -> {OUT_JSON}")

    if history:
        last = history[-1]
        print(f"  Latest: {last['date']}  rolling={last['rolling_accuracy']:.1%}  "
              f"cumulative={last['cumulative_games']} games")


if __name__ == "__main__":
    main()
