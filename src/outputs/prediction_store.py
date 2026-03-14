"""Append-only SQLite prediction store.

Usage:
    from src.outputs.prediction_store import write_game_prediction
    write_game_prediction({"home_team": "BOS", "away_team": "LAL",
                           "home_win_prob": 0.62, "away_win_prob": 0.38})

The store is initialized on first write. WAL mode is always set on open
to allow concurrent reads from a web process during update.py writes.
FR-6.1, FR-6.2, FR-6.4.
"""
import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path

STORE_PATH = "database/predictions_history.db"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS game_predictions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT NOT NULL,
    game_date           TEXT,
    home_team           TEXT NOT NULL,
    away_team           TEXT NOT NULL,
    home_win_prob       REAL NOT NULL,
    away_win_prob       REAL NOT NULL,
    model_name          TEXT,
    model_artifact      TEXT,
    decision_threshold  REAL,
    feature_count       INTEGER,
    actual_home_win     INTEGER,
    notes               TEXT
);
"""
_INDEX_GAME_DATE = "CREATE INDEX IF NOT EXISTS idx_game_date ON game_predictions(game_date);"
_INDEX_CREATED   = "CREATE INDEX IF NOT EXISTS idx_created_at ON game_predictions(created_at);"
_INDEX_TEAMS     = "CREATE INDEX IF NOT EXISTS idx_teams ON game_predictions(home_team, away_team);"


def _get_connection(store_path: str) -> sqlite3.Connection:
    """Open connection with WAL mode enabled (persists after close)."""
    Path(store_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(store_path)
    con.execute("PRAGMA journal_mode=WAL;")
    return con


def init_store(store_path: str = STORE_PATH) -> None:
    """Create the database, enable WAL mode, and create tables/indexes if absent."""
    con = _get_connection(store_path)
    con.execute(_CREATE_TABLE_SQL)
    con.execute(_INDEX_GAME_DATE)
    con.execute(_INDEX_CREATED)
    con.execute(_INDEX_TEAMS)
    con.commit()
    con.close()


def write_game_prediction(
    prediction: dict,
    store_path: str = STORE_PATH,
) -> int:
    """Append one game prediction record to the store.

    prediction must include: home_team, away_team, home_win_prob, away_win_prob.
    Optional keys: game_date (YYYY-MM-DD), model_name, model_artifact,
                    decision_threshold, feature_count, notes (str or dict).
    Returns the rowid of the inserted record.
    """
    init_store(store_path)
    game_date = prediction.get("game_date")
    home = prediction["home_team"]
    away = prediction["away_team"]

    # Skip if prediction already exists for this game on this date
    con = _get_connection(store_path)
    existing = con.execute(
        "SELECT id FROM game_predictions WHERE game_date = ? AND home_team = ? AND away_team = ?",
        (game_date, home, away),
    ).fetchone()
    if existing:
        con.close()
        return existing[0]

    now = datetime.now(timezone.utc).isoformat()
    notes = prediction.get("notes")
    if notes is not None and not isinstance(notes, str):
        notes = json.dumps(notes)
    cur = con.execute(
        """INSERT INTO game_predictions
           (created_at, game_date, home_team, away_team,
            home_win_prob, away_win_prob, model_name, model_artifact,
            decision_threshold, feature_count, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            now,
            game_date,
            home,
            away,
            float(prediction["home_win_prob"]),
            float(prediction["away_win_prob"]),
            prediction.get("model_name"),
            prediction.get("model_artifact"),
            prediction.get("decision_threshold"),
            prediction.get("feature_count"),
            notes,
        ),
    )
    rowid = cur.lastrowid
    con.commit()
    con.close()
    return rowid
