"""
build_value_bets.py -- regenerate dashboard/data/value_bets.json

Reads predictions from database/predictions_history.db and game odds from
data/odds/game_lines.csv.

A game is a value bet when: model_prob - market_prob > 0.05 (5% edge).

If no odds data is available, the existing value_bets.json is returned
unchanged (pass-through).

Run: python scripts/build_value_bets.py
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

DB_PATH = PROJECT_ROOT / "database" / "predictions_history.db"
GAME_LINES_CSV = PROJECT_ROOT / "data" / "odds" / "game_lines.csv"
OUT_JSON = PROJECT_ROOT / "dashboard" / "data" / "value_bets.json"
TEAMS_CSV = PROJECT_ROOT / "data" / "processed" / "teams.csv"

EDGE_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_team_names() -> dict[str, str]:
    """Return {abbreviation: full_name} from teams.csv."""
    if not TEAMS_CSV.exists():
        return {}
    df = pd.read_csv(TEAMS_CSV)
    return dict(zip(df["abbreviation"], df["full_name"]))


def _load_all_predictions(conn: sqlite3.Connection) -> list[dict]:
    """Load all predictions from DB, most recent first."""
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM game_predictions ORDER BY game_date DESC, id DESC"
    )
    col_names = [desc[0] for desc in cur.description]
    return [dict(zip(col_names, row)) for row in cur.fetchall()]


def _american_to_prob(ml) -> float | None:
    """Convert American moneyline to implied probability (vig-inclusive)."""
    try:
        ml = float(ml)
    except (TypeError, ValueError):
        return None
    if ml > 0:
        return round(100 / (ml + 100), 4)
    else:
        return round(abs(ml) / (abs(ml) + 100), 4)


def _load_game_lines_all() -> pd.DataFrame:
    """
    Load all game_lines.csv rows. Returns empty DataFrame if file missing.

    fetch_odds.py writes: date, home_team, away_team, home_moneyline,
    away_moneyline, spread.  We normalise to the columns expected downstream:
    game_date, home_team, away_team, home_market_prob, spread.
    """
    if not GAME_LINES_CSV.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(GAME_LINES_CSV)

        # Rename 'date' -> 'game_date' if needed (fetch_odds.py writes 'date')
        if "date" in df.columns and "game_date" not in df.columns:
            df = df.rename(columns={"date": "game_date"})

        if "game_date" in df.columns:
            df["game_date"] = pd.to_datetime(df["game_date"], format="mixed").dt.date.astype(str)

        # Compute home_market_prob from home_moneyline if not already present
        if "home_market_prob" not in df.columns and "home_moneyline" in df.columns:
            df = df.copy()
            df["home_market_prob"] = df["home_moneyline"].apply(_american_to_prob)

        return df
    except Exception as exc:
        print(f"  WARN: could not read game_lines.csv: {exc}")
        return pd.DataFrame()


def _build_lines_index(lines_df: pd.DataFrame) -> dict[tuple[str, str, str], dict]:
    """Build (game_date, home_team, away_team) -> row dict index."""
    index: dict[tuple[str, str, str], dict] = {}
    if lines_df.empty:
        return index
    for _, row in lines_df.iterrows():
        key = (
            str(row.get("game_date", "")),
            str(row.get("home_team", "")),
            str(row.get("away_team", "")),
        )
        index[key] = row.to_dict()
    return index


def _compute_value_bets(
    predictions: list[dict],
    lines_index: dict[tuple[str, str, str], dict],
    team_names: dict[str, str],
    edge_threshold: float = EDGE_THRESHOLD,
) -> list[dict]:
    """
    For each prediction, compute edge vs market. Return rows where edge >= threshold.
    Sorted by edge_pct descending.
    """
    value_bets: list[dict] = []
    now_str = datetime.now(timezone.utc).isoformat()

    for pred in predictions:
        game_date = str(pred.get("game_date", ""))
        home = str(pred.get("home_team", ""))
        away = str(pred.get("away_team", ""))
        home_prob = float(pred.get("home_win_prob") or 0.0)

        key = (game_date, home, away)
        line_row = lines_index.get(key)
        if line_row is None:
            continue

        market_prob_raw = line_row.get("home_market_prob")
        if market_prob_raw is None:
            continue

        try:
            market_prob = float(market_prob_raw)
        except (TypeError, ValueError):
            continue

        edge = home_prob - market_prob

        if edge >= edge_threshold:
            # Model favours home side
            recommended_side = home
            model_prob = home_prob
        elif -edge >= edge_threshold:
            # Model favours away side
            recommended_side = away
            model_prob = 1.0 - home_prob
            market_prob = 1.0 - market_prob
            edge = -edge
        else:
            continue

        # Half-Kelly criterion: f = 0.5 * (p*b - (1-p)) / b
        b = market_prob / (1.0 - market_prob) if market_prob < 1.0 else 0.0
        kelly_raw = (model_prob * b - (1.0 - model_prob)) / b if b > 0 else 0.0
        kelly_fraction = round(max(0.0, 0.5 * kelly_raw), 4)

        value_bets.append({
            "game_date": game_date,
            "home_team": home,
            "away_team": away,
            "home_team_name": team_names.get(home, home),
            "away_team_name": team_names.get(away, away),
            "model_prob": round(model_prob, 4),
            "market_prob": round(market_prob, 4),
            "edge_pct": round(edge, 4),
            "kelly_fraction": kelly_fraction,
            "recommended_side": recommended_side,
            "created_at": pred.get("created_at", now_str),
        })

    value_bets.sort(key=lambda r: r["edge_pct"], reverse=True)
    return value_bets


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_value_bets(
    db_path: Path = DB_PATH,
    out_path: Path = OUT_JSON,
    edge_threshold: float = EDGE_THRESHOLD,
) -> list[dict]:
    """
    Main entry point.
    Returns the value bets list (also writes JSON).
    """
    if not db_path.exists():
        print(f"  WARN: predictions DB not found at {db_path}")
        return _passthrough(out_path)

    lines_df = _load_game_lines_all()
    if lines_df.empty:
        print("  No game_lines.csv data -- passing through existing value_bets.json")
        return _passthrough(out_path)

    print(f"  Loaded {len(lines_df)} lines rows from game_lines.csv")
    lines_index = _build_lines_index(lines_df)

    try:
        conn = sqlite3.connect(str(db_path))
        predictions = _load_all_predictions(conn)
        conn.close()
    except Exception as exc:
        print(f"  ERROR: could not read predictions DB: {exc}")
        return _passthrough(out_path)

    if not predictions:
        print("  No predictions in DB -- passing through existing value_bets.json")
        return _passthrough(out_path)

    print(f"  {len(predictions)} predictions loaded from DB")
    team_names = _load_team_names()

    value_bets = _compute_value_bets(predictions, lines_index, team_names, edge_threshold)
    print(f"  Found {len(value_bets)} value bets (edge >= {edge_threshold:.0%})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(value_bets, fh, indent=2, default=str)

    print(f"Written -> {out_path}")
    return value_bets


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
    build_value_bets()
