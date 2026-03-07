"""
build_picks.py -- regenerate dashboard/data/todays_picks.json

Reads predictions from database/predictions_history.db.
Finds predictions for today's date, or falls back to the most recent date
with predictions. Joins with odds from data/processed/game_lines.csv if
it exists.

If the DB has no predictions at all, reads the existing todays_picks.json
and returns it unchanged (pass-through).

Run: python scripts/build_picks.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "database" / "predictions_history.db"
GAME_LINES_CSV = PROJECT_ROOT / "data" / "processed" / "game_lines.csv"
OUT_JSON = PROJECT_ROOT / "dashboard" / "data" / "todays_picks.json"

TEAMS_CSV = PROJECT_ROOT / "data" / "processed" / "teams.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_team_names() -> dict[str, str]:
    """Return {abbreviation: full_name} from teams.csv."""
    if not TEAMS_CSV.exists():
        return {}
    df = pd.read_csv(TEAMS_CSV)
    return dict(zip(df["abbreviation"], df["full_name"]))


def _load_predictions(conn: sqlite3.Connection, target_date: str) -> list[dict]:
    """
    Fetch all predictions for target_date from game_predictions table.
    Returns list of row dicts.
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM game_predictions WHERE game_date = ? ORDER BY id",
        (target_date,),
    )
    col_names = [desc[0] for desc in cur.description]
    return [dict(zip(col_names, row)) for row in cur.fetchall()]


def _latest_prediction_date(conn: sqlite3.Connection) -> str | None:
    """Return the most recent game_date that has predictions, or None."""
    cur = conn.cursor()
    cur.execute("SELECT MAX(game_date) FROM game_predictions")
    row = cur.fetchone()
    if row and row[0]:
        return row[0]
    return None


def _load_game_lines(target_date: str) -> dict[tuple[str, str], dict]:
    """
    Load game_lines.csv (if present) and return a dict keyed by
    (home_team, away_team) -> {spread, home_market_prob, ...}.
    Filters to target_date if a date column is present.
    """
    if not GAME_LINES_CSV.exists():
        return {}
    try:
        df = pd.read_csv(GAME_LINES_CSV)
    except Exception as exc:
        print(f"  WARN: could not read game_lines.csv: {exc}")
        return {}

    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], format="mixed").dt.date.astype(str)
        df = df[df["game_date"] == target_date]

    result: dict[tuple[str, str], dict] = {}
    for _, row in df.iterrows():
        home = str(row.get("home_team", ""))
        away = str(row.get("away_team", ""))
        if home and away:
            result[(home, away)] = row.to_dict()
    return result


def _build_pick_row(
    pred: dict,
    team_names: dict[str, str],
    lines: dict[tuple[str, str], dict],
) -> dict:
    """Convert a DB prediction row into the todays_picks.json schema."""
    home = pred["home_team"]
    away = pred["away_team"]
    home_prob = float(pred.get("home_win_prob") or 0.0)
    away_prob = float(pred.get("away_win_prob") or 0.0)

    # Determine predicted winner and ATS pick
    predicted_winner = home if home_prob >= away_prob else away

    # Odds enrichment
    line_data = lines.get((home, away), {})
    spread = line_data.get("spread") if line_data else None
    market_prob = line_data.get("home_market_prob") if line_data else None

    # ATS pick: if spread exists, pick the side with positive expected value
    ats_pick: str | None = None
    edge_pct: float | None = None
    value_bet = False

    if spread is not None and market_prob is not None:
        market_prob_f = float(market_prob)
        edge = home_prob - market_prob_f
        edge_pct = round(abs(edge), 3)
        if abs(edge) >= 0.05:
            value_bet = True
        ats_pick = home if edge >= 0 else away
    else:
        ats_pick = predicted_winner

    return {
        "game_date": pred.get("game_date", ""),
        "home_team": home,
        "away_team": away,
        "home_team_name": team_names.get(home, home),
        "away_team_name": team_names.get(away, away),
        "home_win_prob": round(home_prob, 4),
        "away_win_prob": round(away_prob, 4),
        "predicted_winner": predicted_winner,
        "ats_pick": ats_pick,
        "spread": spread,
        "value_bet": value_bet,
        "edge_pct": edge_pct,
        "model_name": pred.get("model_name") or pred.get("model_artifact", "unknown"),
        "created_at": pred.get("created_at", datetime.now(timezone.utc).isoformat()),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_picks(
    db_path: Path = DB_PATH,
    out_path: Path = OUT_JSON,
    target_date: str | None = None,
) -> list[dict]:
    """
    Main entry point.
    target_date: ISO date string (YYYY-MM-DD). Defaults to today.
    Returns the picks list (also writes JSON).
    """
    if target_date is None:
        target_date = date.today().isoformat()

    if not db_path.exists():
        print(f"  WARN: predictions DB not found at {db_path}")
        return _passthrough(out_path)

    try:
        conn = sqlite3.connect(str(db_path))
    except Exception as exc:
        print(f"  ERROR: could not connect to DB: {exc}")
        return _passthrough(out_path)

    try:
        preds = _load_predictions(conn, target_date)

        if not preds:
            # Fall back to the most recent date with predictions
            latest_date = _latest_prediction_date(conn)
            if latest_date:
                print(f"  No predictions for {target_date}, falling back to {latest_date}")
                preds = _load_predictions(conn, latest_date)
                target_date = latest_date
            else:
                print("  No predictions in DB at all -- passing through existing JSON")
                conn.close()
                return _passthrough(out_path)
    finally:
        conn.close()

    if not preds:
        print("  Predictions query returned empty -- passing through existing JSON")
        return _passthrough(out_path)

    print(f"  Found {len(preds)} predictions for {target_date}")

    team_names = _load_team_names()
    lines = _load_game_lines(target_date)
    if lines:
        print(f"  Loaded {len(lines)} lines from game_lines.csv for {target_date}")
    else:
        print("  No game_lines.csv data -- odds fields will be null")

    picks = [_build_pick_row(p, team_names, lines) for p in preds]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(picks, fh, indent=2, default=str)

    print(f"Written -> {out_path}  ({len(picks)} picks)")
    return picks


def _passthrough(out_path: Path) -> list[dict]:
    """Return existing JSON unchanged (or empty list if file missing)."""
    if out_path.exists():
        try:
            with open(out_path, encoding="utf-8") as fh:
                data = json.load(fh)
            print(f"  Pass-through: returning existing {out_path.name} unchanged")
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  WARN: could not read existing JSON: {exc}")
    return []


if __name__ == "__main__":
    build_picks()
