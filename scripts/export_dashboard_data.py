"""Export prediction data from predictions_history.db to dashboard JSON files.

Reads from database/predictions_history.db and writes:
  - dashboard/data/todays_picks.json
  - dashboard/data/accuracy_history.json
  - dashboard/data/value_bets.json
  - dashboard/data/player_predictions.json

Handles missing/empty data gracefully (writes empty arrays, never crashes).

Usage:
    python scripts/export_dashboard_data.py
"""

import json
import os
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "database" / "predictions_history.db"
DASHBOARD_DATA = PROJECT_ROOT / "dashboard" / "data"


def _connect():
    """Return a sqlite3 connection or None if DB is missing."""
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _write_json(filename: str, data) -> None:
    """Write data to a JSON file in the dashboard data directory."""
    DASHBOARD_DATA.mkdir(parents=True, exist_ok=True)
    path = DASHBOARD_DATA / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Wrote {path} ({len(data) if isinstance(data, list) else 'object'})")


def export_todays_picks(conn) -> None:
    """Export today's game predictions with win prob, ATS pick, value bet flag."""
    today = date.today().isoformat()
    if conn is None:
        _write_json("todays_picks.json", [])
        return

    rows = conn.execute(
        """SELECT game_date, home_team, away_team, home_win_prob, away_win_prob,
                  model_name, notes, created_at
           FROM game_predictions
           WHERE game_date = ?
           ORDER BY created_at DESC""",
        (today,),
    ).fetchall()

    picks = []
    for row in rows:
        notes = {}
        if row["notes"]:
            try:
                notes = json.loads(row["notes"])
            except (json.JSONDecodeError, TypeError):
                notes = {"raw": row["notes"]}

        picks.append({
            "game_date": row["game_date"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_win_prob": round(row["home_win_prob"], 4),
            "away_win_prob": round(row["away_win_prob"], 4),
            "predicted_winner": (
                row["home_team"]
                if row["home_win_prob"] >= 0.5
                else row["away_team"]
            ),
            "ats_pick": notes.get("ats_pick", None),
            "spread": notes.get("spread", None),
            "value_bet": notes.get("value_bet", False),
            "edge_pct": notes.get("edge_pct", None),
            "model_name": row["model_name"],
            "created_at": row["created_at"],
        })

    _write_json("todays_picks.json", picks)


def export_accuracy_history(conn) -> None:
    """Export daily and rolling accuracy for games that have actual results."""
    if conn is None:
        _write_json("accuracy_history.json", [])
        return

    rows = conn.execute(
        """SELECT game_date,
                  COUNT(*) AS total,
                  SUM(CASE
                      WHEN (home_win_prob >= 0.5 AND actual_home_win = 1)
                        OR (home_win_prob < 0.5 AND actual_home_win = 0)
                      THEN 1 ELSE 0 END) AS correct
           FROM game_predictions
           WHERE actual_home_win IS NOT NULL
             AND game_date IS NOT NULL
           GROUP BY game_date
           ORDER BY game_date"""
    ).fetchall()

    history = []
    cumulative_correct = 0
    cumulative_total = 0
    for row in rows:
        cumulative_correct += row["correct"]
        cumulative_total += row["total"]
        daily_acc = row["correct"] / row["total"] if row["total"] > 0 else 0.0
        running_acc = (
            cumulative_correct / cumulative_total if cumulative_total > 0 else 0.0
        )
        history.append({
            "date": row["game_date"],
            "daily_accuracy": round(daily_acc, 4),
            "rolling_accuracy": round(running_acc, 4),
            "games": row["total"],
            "correct": row["correct"],
            "cumulative_games": cumulative_total,
        })

    _write_json("accuracy_history.json", history)


def export_value_bets(conn) -> None:
    """Export recent value bet alerts with edge size."""
    if conn is None:
        _write_json("value_bets.json", [])
        return

    rows = conn.execute(
        """SELECT game_date, home_team, away_team, home_win_prob, away_win_prob,
                  notes, created_at
           FROM game_predictions
           WHERE notes LIKE '%value_bet%'
           ORDER BY created_at DESC
           LIMIT 50"""
    ).fetchall()

    bets = []
    for row in rows:
        notes = {}
        if row["notes"]:
            try:
                notes = json.loads(row["notes"])
            except (json.JSONDecodeError, TypeError):
                continue

        if not notes.get("value_bet", False):
            continue

        bets.append({
            "game_date": row["game_date"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "model_prob": round(row["home_win_prob"], 4),
            "market_prob": notes.get("market_prob", None),
            "edge_pct": notes.get("edge_pct", None),
            "recommended_side": notes.get("recommended_side", None),
            "created_at": row["created_at"],
        })

    _write_json("value_bets.json", bets)


def export_player_predictions(conn) -> None:
    """Export latest player prediction data from notes JSON."""
    if conn is None:
        _write_json("player_predictions.json", [])
        return

    rows = conn.execute(
        """SELECT notes, created_at
           FROM game_predictions
           WHERE notes LIKE '%player%'
           ORDER BY created_at DESC
           LIMIT 200"""
    ).fetchall()

    players = []
    for row in rows:
        if not row["notes"]:
            continue
        try:
            notes = json.loads(row["notes"])
        except (json.JSONDecodeError, TypeError):
            continue

        if "player_predictions" in notes:
            for pp in notes["player_predictions"]:
                players.append({
                    "player_name": pp.get("player_name", "Unknown"),
                    "pts": pp.get("pts", None),
                    "reb": pp.get("reb", None),
                    "ast": pp.get("ast", None),
                    "team": pp.get("team", None),
                    "opponent": pp.get("opponent", None),
                    "created_at": row["created_at"],
                })

    _write_json("player_predictions.json", players)


def main():
    print("Exporting dashboard data...")
    print(f"  DB: {DB_PATH} (exists={DB_PATH.exists()})")
    conn = _connect()
    try:
        export_todays_picks(conn)
        export_accuracy_history(conn)
        export_value_bets(conn)
        export_player_predictions(conn)
    finally:
        if conn:
            conn.close()

    # Write metadata
    meta = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "db_exists": DB_PATH.exists(),
    }
    _write_json("meta.json", meta)
    print("Done.")


if __name__ == "__main__":
    main()
