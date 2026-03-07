"""
build_line_movement.py  --  produce dashboard/data/line_movement.json

Reads opening vs closing spreads from predictions_history.db (clv_tracking
table if present, else game_predictions) and optionally from
data/processed/game_lines.csv if it exists.

Movement classification:
  |movement| >= 1.5  -> "sharp_action"
  |movement| >= 0.5  -> "moderate_move"
  otherwise          -> "stable"

Direction convention:
  movement = closing_spread - opening_spread
  movement < 0  (spread grew toward home) -> "toward_home"
  movement > 0  (spread shrank)           -> "toward_away"
  movement == 0                           -> "no_movement"
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "database" / "predictions_history.db"
GAME_LINES_CSV = PROJECT_ROOT / "data" / "processed" / "game_lines.csv"
OUT_JSON = PROJECT_ROOT / "dashboard" / "data" / "line_movement.json"

SHARP_THRESHOLD = 1.5
MODERATE_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify(movement: float) -> str:
    abs_move = abs(movement)
    if abs_move >= SHARP_THRESHOLD:
        return "sharp_action"
    if abs_move >= MODERATE_THRESHOLD:
        return "moderate_move"
    return "stable"


def _direction(movement: float) -> str:
    if movement < -0.05:
        return "toward_home"
    if movement > 0.05:
        return "toward_away"
    return "no_movement"


def _interpretation(home: str, away: str, movement: float, opening: float, closing: float) -> str:
    cls = _classify(movement)
    direction = _direction(movement)
    abs_move = abs(movement)

    if cls == "stable":
        return f"Line stable at {closing:+.1f} -- no significant money movement"

    side = home if direction == "toward_home" else away
    qualifier = "Sharp" if cls == "sharp_action" else "Moderate"
    return (
        f"Line moved {abs_move:.1f} pt(s) toward {side} "
        f"({opening:+.1f} -> {closing:+.1f}) -- {qualifier} money on {side}"
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _db_tables(con: sqlite3.Connection) -> list[str]:
    rows = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return [r[0] for r in rows]


def load_from_db() -> pd.DataFrame:
    """
    Attempt to load line movement data from predictions_history.db.

    Tries clv_tracking table first; falls back to game_predictions
    (which has home_win_prob but no explicit spread -- in that case we
    derive an implied spread proxy and surface it as opening == closing
    with zero movement, so downstream consumers still get game metadata).
    """
    if not DB_PATH.exists():
        return pd.DataFrame()

    with sqlite3.connect(DB_PATH) as con:
        tables = _db_tables(con)

        if "clv_tracking" in tables:
            df = pd.read_sql_query(
                "SELECT * FROM clv_tracking ORDER BY game_date DESC",
                con,
            )
            required = {"home_team", "away_team", "game_date", "opening_spread", "closing_spread"}
            if required.issubset(set(df.columns)):
                return df

        if "game_predictions" in tables:
            df = pd.read_sql_query(
                """
                SELECT game_date,
                       home_team,
                       away_team,
                       home_win_prob,
                       notes
                FROM game_predictions
                ORDER BY game_date DESC
                """,
                con,
            )
            return df

    return pd.DataFrame()


def load_from_csv() -> pd.DataFrame:
    """Load game_lines.csv if it exists."""
    if not GAME_LINES_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(GAME_LINES_CSV)
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")
    return df


def _prob_to_spread(prob: float) -> float:
    """
    Very rough conversion: win probability -> implied point spread.
    Based on a linear approximation: 50% ~ 0, each ~3% ~ 1 point.
    Used only as a fallback when no explicit spread is available.
    """
    return round((0.5 - prob) * 33.3, 1)


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_line_movement(
    db_df: pd.DataFrame | None = None,
    csv_df: pd.DataFrame | None = None,
) -> list[dict]:
    if db_df is None:
        db_df = load_from_db()
    if csv_df is None:
        csv_df = load_from_csv()

    results: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    # --- Prefer CSV if it has the required spread columns ---
    if not csv_df.empty:
        required = {"home_team", "away_team", "game_date", "opening_spread", "closing_spread"}
        if required.issubset(set(csv_df.columns)):
            csv_df = csv_df.copy()
            csv_df["game_date"] = pd.to_datetime(csv_df["game_date"], format="mixed").dt.strftime("%Y-%m-%d")
            for _, row in csv_df.iterrows():
                home = str(row["home_team"])
                away = str(row["away_team"])
                date = str(row["game_date"])
                key = (date, home, away)
                if key in seen:
                    continue
                seen.add(key)

                opening = float(row["opening_spread"])
                closing = float(row["closing_spread"])
                movement = round(closing - opening, 2)

                results.append({
                    "home_team": home,
                    "away_team": away,
                    "game_date": date,
                    "opening_spread": opening,
                    "current_spread": closing,
                    "movement": movement,
                    "direction": _direction(movement),
                    "classification": _classify(movement),
                    "interpretation": _interpretation(home, away, movement, opening, closing),
                })

    # --- Use DB data ---
    if not db_df.empty:
        db_df = db_df.copy()

        # CLV table path (has explicit spreads)
        if "opening_spread" in db_df.columns and "closing_spread" in db_df.columns:
            db_df["game_date"] = pd.to_datetime(db_df["game_date"], format="mixed").dt.strftime("%Y-%m-%d")
            for _, row in db_df.iterrows():
                home = str(row["home_team"])
                away = str(row["away_team"])
                date = str(row["game_date"])
                key = (date, home, away)
                if key in seen:
                    continue
                seen.add(key)

                opening = float(row["opening_spread"])
                closing = float(row["closing_spread"])
                movement = round(closing - opening, 2)

                results.append({
                    "home_team": home,
                    "away_team": away,
                    "game_date": date,
                    "opening_spread": opening,
                    "current_spread": closing,
                    "movement": movement,
                    "direction": _direction(movement),
                    "classification": _classify(movement),
                    "interpretation": _interpretation(home, away, movement, opening, closing),
                })

        # game_predictions fallback (no explicit spreads; derive from prob)
        elif "home_win_prob" in db_df.columns:
            db_df["game_date"] = pd.to_datetime(db_df["game_date"], format="mixed").dt.strftime("%Y-%m-%d")
            for _, row in db_df.iterrows():
                home = str(row["home_team"])
                away = str(row["away_team"])
                date = str(row["game_date"])
                key = (date, home, away)
                if key in seen:
                    continue
                seen.add(key)

                implied_spread = _prob_to_spread(float(row["home_win_prob"]))
                # With only one prob, opening == closing -> movement = 0
                results.append({
                    "home_team": home,
                    "away_team": away,
                    "game_date": date,
                    "opening_spread": implied_spread,
                    "current_spread": implied_spread,
                    "movement": 0.0,
                    "direction": "no_movement",
                    "classification": "stable",
                    "interpretation": (
                        f"No historical line data -- implied spread {implied_spread:+.1f} "
                        f"derived from model probability ({float(row['home_win_prob']):.1%})"
                    ),
                })

    # Sort by date descending
    results.sort(key=lambda x: x["game_date"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading DB data...")
    db_df = load_from_db()
    print(f"  {len(db_df)} rows from predictions_history.db")

    csv_df = load_from_csv()
    if not csv_df.empty:
        print(f"  {len(csv_df)} rows from game_lines.csv")
    else:
        print("  game_lines.csv not found -- using DB only")

    results = build_line_movement(db_df, csv_df)
    print(f"  Built {len(results)} line movement entries")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    print(f"  Written -> {OUT_JSON}")
    for entry in results[:5]:
        print(f"  {entry['away_team']} @ {entry['home_team']}  "
              f"{entry['game_date']}  "
              f"open={entry['opening_spread']:+.1f}  "
              f"close={entry['current_spread']:+.1f}  "
              f"move={entry['movement']:+.2f}  "
              f"cls={entry['classification']}")


if __name__ == "__main__":
    main()
