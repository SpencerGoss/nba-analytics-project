"""
Build dashboard/data/performance.json from predictions_history.db and accuracy_history.json.

Computes:
- Season accuracy and rolling accuracy (7-day window)
- ROI by market (ML, ATS, totals) from game_predictions table
- CLV summary from clv_tracking table (if it exists)
- Calibration buckets: 5% confidence intervals -> actual accuracy
- Win/loss streaks (current and best)

Run: python scripts/build_performance.py
"""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "database" / "predictions_history.db"
ACCURACY_HISTORY_PATH = PROJECT_ROOT / "dashboard" / "data" / "accuracy_history.json"
OUT_PATH = PROJECT_ROOT / "dashboard" / "data" / "performance.json"

# Confidence bucket boundaries for calibration (lower inclusive, upper exclusive)
CALIBRATION_BUCKETS = [
    (0.50, 0.55, "50-55%"),
    (0.55, 0.60, "55-60%"),
    (0.60, 0.65, "60-65%"),
    (0.65, 0.70, "65-70%"),
    (0.70, 0.75, "70-75%"),
    (0.75, 0.80, "75-80%"),
    (0.80, 1.01, "80%+"),
]


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    return cur.fetchone() is not None


# ---------------------------------------------------------------------------
# Accuracy history (from JSON file)
# ---------------------------------------------------------------------------

def load_accuracy_history(path: Path) -> list[dict]:
    """Load accuracy_history.json; return empty list if file missing."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compute_season_accuracy(history: list[dict]) -> tuple[float | None, int]:
    """Return (season_accuracy, total_games) from accuracy_history entries."""
    if not history:
        return None, 0
    total_games = sum(row.get("games", 0) for row in history)
    total_correct = sum(row.get("correct", 0) for row in history)
    if total_games == 0:
        return None, 0
    return round(total_correct / total_games, 4), total_games


def compute_rolling_accuracy_7d(history: list[dict]) -> float | None:
    """Compute accuracy over the last 7 calendar-date entries in history."""
    if not history:
        return None
    last7 = history[-7:]
    games = sum(row.get("games", 0) for row in last7)
    correct = sum(row.get("correct", 0) for row in last7)
    if games == 0:
        return None
    return round(correct / games, 4)


# ---------------------------------------------------------------------------
# Streak computation
# ---------------------------------------------------------------------------

def _daily_results(history: list[dict]) -> list[bool]:
    """
    Expand accuracy_history into a flat list of True/False per game,
    in chronological order, treating each day's games as a block.
    Used only for streak detection — we approximate by treating each
    day as W if daily_accuracy >= 0.5, L otherwise.
    """
    results: list[bool] = []
    for row in history:
        win = (row.get("daily_accuracy", 0.0) or 0.0) >= 0.5
        results.append(win)
    return results


def compute_streaks(history: list[dict]) -> tuple[dict, dict]:
    """
    Return (current_streak, best_streak).
    Each is {"type": "W"|"L", "length": int}.
    Uses per-day win/loss (daily_accuracy >= 0.5).
    """
    results = _daily_results(history)
    if not results:
        return {"type": "W", "length": 0}, {"type": "W", "length": 0}

    # Current streak (from end)
    current_type = "W" if results[-1] else "L"
    current_len = 0
    for r in reversed(results):
        if (r and current_type == "W") or (not r and current_type == "L"):
            current_len += 1
        else:
            break

    # Best streak (scan entire history)
    best_type = "W"
    best_len = 0
    streak_type = "W" if results[0] else "L"
    streak_len = 1
    for r in results[1:]:
        this_type = "W" if r else "L"
        if this_type == streak_type:
            streak_len += 1
        else:
            if streak_len > best_len:
                best_len = streak_len
                best_type = streak_type
            streak_type = this_type
            streak_len = 1
    if streak_len > best_len:
        best_len = streak_len
        best_type = streak_type

    return (
        {"type": current_type, "length": current_len},
        {"type": best_type, "length": best_len},
    )


# ---------------------------------------------------------------------------
# ROI by market (from game_predictions table)
# ---------------------------------------------------------------------------

def compute_roi_by_market(conn: sqlite3.Connection) -> dict:
    """
    Derive ROI metrics from game_predictions.
    - ML: rows where actual_home_win IS NOT NULL (outcome resolved)
    - ATS: not tracked in DB yet -> fill with null
    - totals: not tracked -> fill with null
    """
    empty_market = {"bets": 0, "wins": 0, "win_pct": None, "roi_pct": None}

    if not _table_exists(conn, "game_predictions"):
        return {"ML": empty_market, "ATS": empty_market, "totals": empty_market}

    cur = conn.execute(
        """
        SELECT
            home_win_prob,
            away_win_prob,
            actual_home_win
        FROM game_predictions
        WHERE actual_home_win IS NOT NULL
        """,
    )
    rows = cur.fetchall()

    ml_bets = 0
    ml_wins = 0
    for row in rows:
        home_prob = row["home_win_prob"] or 0.0
        away_prob = row["away_win_prob"] or 0.0
        actual = row["actual_home_win"]  # 1 = home won, 0 = away won

        # Predicted winner is whoever has higher prob
        predicted_home_wins = home_prob >= away_prob
        actual_home_wins = bool(actual)

        ml_bets += 1
        if predicted_home_wins == actual_home_wins:
            ml_wins += 1

    if ml_bets > 0:
        win_pct = round(ml_wins / ml_bets, 4)
        # Simplified flat-bet ROI at -110 (vig = 4.55% per loss)
        wins = ml_wins
        losses = ml_bets - ml_wins
        roi_pct = round((wins * (100 / 110) - losses) / ml_bets * 100, 2)
        ml_market: dict = {
            "bets": ml_bets,
            "wins": ml_wins,
            "win_pct": win_pct,
            "roi_pct": roi_pct,
        }
    else:
        ml_market = empty_market

    return {
        "ML": ml_market,
        "ATS": empty_market,
        "totals": empty_market,
    }


# ---------------------------------------------------------------------------
# CLV summary (from clv_tracking table if present)
# ---------------------------------------------------------------------------

def compute_clv_summary(conn: sqlite3.Connection) -> dict:
    """Read clv_tracking table if it exists; otherwise return null summary."""
    null_summary = {
        "mean_clv": None,
        "positive_clv_rate": None,
        "has_edge": False,
        "sample_size": 0,
    }

    if not _table_exists(conn, "clv_tracking"):
        return null_summary

    cur = conn.execute("SELECT clv FROM clv_tracking WHERE clv IS NOT NULL")
    clv_values = [row[0] for row in cur.fetchall()]

    if not clv_values:
        return null_summary

    mean_clv = round(sum(clv_values) / len(clv_values), 4)
    positive_count = sum(1 for v in clv_values if v > 0)
    positive_rate = round(positive_count / len(clv_values), 4)

    return {
        "mean_clv": mean_clv,
        "positive_clv_rate": positive_rate,
        "has_edge": mean_clv > 0,
        "sample_size": len(clv_values),
    }


# ---------------------------------------------------------------------------
# Calibration buckets (from game_predictions)
# ---------------------------------------------------------------------------

def compute_calibration(conn: sqlite3.Connection) -> list[dict]:
    """
    Bucket resolved predictions by model confidence (max of home/away prob),
    then compute actual accuracy per bucket.
    Returns list of dicts with keys: confidence_bucket, predicted_pct, actual_pct, n.
    """
    if not _table_exists(conn, "game_predictions"):
        return []

    cur = conn.execute(
        """
        SELECT
            home_win_prob,
            away_win_prob,
            actual_home_win
        FROM game_predictions
        WHERE actual_home_win IS NOT NULL
          AND home_win_prob IS NOT NULL
          AND away_win_prob IS NOT NULL
        """,
    )
    rows = cur.fetchall()

    if not rows:
        return []

    # Bucket rows
    buckets: dict[str, dict] = {
        label: {"predicted_sum": 0.0, "correct": 0, "n": 0}
        for (_, _, label) in CALIBRATION_BUCKETS
    }

    for row in rows:
        home_prob = row["home_win_prob"]
        away_prob = row["away_win_prob"]
        actual = bool(row["actual_home_win"])

        confidence = max(home_prob, away_prob)
        predicted_correct_prob = confidence  # prob of predicted outcome
        predicted_home_wins = home_prob >= away_prob
        is_correct = predicted_home_wins == actual

        for lo, hi, label in CALIBRATION_BUCKETS:
            if lo <= confidence < hi:
                buckets[label]["predicted_sum"] += predicted_correct_prob
                buckets[label]["correct"] += int(is_correct)
                buckets[label]["n"] += 1
                break

    result = []
    for lo, hi, label in CALIBRATION_BUCKETS:
        bucket = buckets[label]
        n = bucket["n"]
        if n == 0:
            continue
        result.append({
            "confidence_bucket": label,
            "predicted_pct": round(bucket["predicted_sum"] / n, 4),
            "actual_pct": round(bucket["correct"] / n, 4),
            "n": n,
        })

    return result


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_performance(
    db_path: Path = DB_PATH,
    accuracy_history_path: Path = ACCURACY_HISTORY_PATH,
    out_path: Path = OUT_PATH,
) -> dict:
    """Compute all performance metrics and write performance.json."""
    history = load_accuracy_history(accuracy_history_path)
    season_accuracy, total_games = compute_season_accuracy(history)
    rolling_accuracy_7d = compute_rolling_accuracy_7d(history)
    current_streak, best_streak = compute_streaks(history)

    if db_path.exists():
        conn = _open_db(db_path)
        try:
            roi_by_market = compute_roi_by_market(conn)
            clv_summary = compute_clv_summary(conn)
            calibration = compute_calibration(conn)
        finally:
            conn.close()
    else:
        print(f"WARNING: DB not found at {db_path} -- using null values for DB-derived fields")
        roi_by_market = {
            "ML": {"bets": 0, "wins": 0, "win_pct": None, "roi_pct": None},
            "ATS": {"bets": 0, "wins": 0, "win_pct": None, "roi_pct": None},
            "totals": {"bets": 0, "wins": 0, "win_pct": None, "roi_pct": None},
        }
        clv_summary = {"mean_clv": None, "positive_clv_rate": None, "has_edge": False, "sample_size": 0}
        calibration = []

    payload = {
        "season_accuracy": season_accuracy,
        "total_games": total_games,
        "rolling_accuracy_7d": rolling_accuracy_7d,
        "accuracy_history": history,
        "roi_by_market": roi_by_market,
        "clv_summary": clv_summary,
        "calibration": calibration,
        "current_streak": current_streak,
        "best_streak": best_streak,
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))

    print(f"performance.json written to {out_path}")
    print(f"  season_accuracy={season_accuracy}  total_games={total_games}")
    print(f"  rolling_accuracy_7d={rolling_accuracy_7d}")
    print(f"  current_streak={current_streak}  best_streak={best_streak}")
    print(f"  calibration buckets={len(calibration)}")

    return payload


if __name__ == "__main__":
    build_performance()
