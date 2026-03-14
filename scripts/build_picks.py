"""
build_picks.py -- regenerate dashboard/data/todays_picks.json

Reads predictions from database/predictions_history.db.
Finds predictions for today's date, or falls back to the most recent date
with predictions. Joins with odds from data/odds/game_lines.csv if
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
GAME_LINES_CSV = PROJECT_ROOT / "data" / "odds" / "game_lines.csv"
OUT_JSON = PROJECT_ROOT / "dashboard" / "data" / "todays_picks.json"

TEAMS_CSV = PROJECT_ROOT / "data" / "processed" / "teams.csv"
MATCHUP_CSV = PROJECT_ROOT / "data" / "features" / "game_matchup_features.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_team_names() -> dict[str, str]:
    from scripts.builder_helpers import load_team_names
    return load_team_names(TEAMS_CSV)


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

    # fetch_odds.py writes the column as "date"; normalize to "game_date"
    if "date" in df.columns and "game_date" not in df.columns:
        df = df.rename(columns={"date": "game_date"})
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


def _build_margin_lookup(
    preds: list[dict],
) -> dict[tuple[str, str, str], float | None]:
    """
    Attempt to produce a projected_margin for each prediction by running the
    NBAEnsemble margin model on the most-recent matchup features.

    Returns a dict keyed by (home_team, away_team, game_date) -> float | None.
    Returns an all-None dict (gracefully) if the margin model or feature CSV is
    unavailable -- callers treat None as "not available".
    """
    result: dict[tuple[str, str, str], float | None] = {
        (p["home_team"], p["away_team"], p.get("game_date", "")): None
        for p in preds
    }

    if not MATCHUP_CSV.exists():
        print("  WARN: matchup features CSV not found -- projected_margin will be null")
        return result

    margin_pkl = ARTIFACTS_DIR / "margin_model.pkl"
    if not margin_pkl.exists():
        print("  INFO: margin_model.pkl not found -- projected_margin will be null")
        return result

    try:
        import warnings

        # Load ensemble through src module (handles its own artifact loading)
        from src.models.ensemble import NBAEnsemble

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ensemble = NBAEnsemble.load(ARTIFACTS_DIR)

        if ensemble.margin_model is None:
            print("  INFO: ensemble margin model not available -- projected_margin will be null")
            return result

        matchup_df = pd.read_csv(MATCHUP_CSV, low_memory=False)
        matchup_df["game_date"] = pd.to_datetime(
            matchup_df["game_date"], format="mixed"
        )

        from src.models.game_outcome_model import (
            _get_current_season_code,
            _synthesize_matchup_row,
        )

        current_season = _get_current_season_code()

        # Build most-recent-row index per (home, away) — current season only
        current_df = matchup_df[
            matchup_df["season"].astype(int) == int(current_season)
        ]
        latest = (
            current_df.sort_values("game_date")
            .groupby(["home_team", "away_team"], sort=False)
            .last()
            .reset_index()
        )
        feat_index: dict[tuple[str, str], pd.Series] = {}
        for _, row in latest.iterrows():
            feat_index[(str(row["home_team"]), str(row["away_team"]))] = row

        for p in preds:
            home = p["home_team"]
            away = p["away_team"]
            game_date_str = p.get("game_date", "")
            key = (home, away, game_date_str)

            feat_row = feat_index.get((home, away))
            if feat_row is None:
                # Synthesize from each team's most recent game
                feat_row = _synthesize_matchup_row(
                    matchup_df, home, away,
                    ensemble.margin_feats or [],
                )
                if feat_row is None:
                    continue

            margin_feats = ensemble.margin_feats
            if margin_feats is not None:
                X = pd.DataFrame([feat_row]).reindex(columns=margin_feats)
            else:
                X = pd.DataFrame([feat_row])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                margin_val = float(ensemble.margin_model.predict(X)[0])

            result[key] = round(margin_val, 2)

        n_filled = sum(1 for v in result.values() if v is not None)
        print(f"  Projected margins computed for {n_filled}/{len(preds)} games")

    except Exception as exc:
        print(f"  WARN: margin model inference failed ({exc}) -- projected_margin will be null")

    return result


def _compute_kelly_fraction(
    edge_pct: float | None,
    market_prob: float | None,
    ats_pick: str,
    home: str,
    home_prob: float,
) -> float | None:
    """
    Half-Kelly sizing: kelly = (p * b - (1 - p)) / b * 0.5
    where p = model win prob for the bet side,
          b = (1 - market_prob) / market_prob (decimal odds - 1).
    Returns None if inputs are missing or market_prob is degenerate.
    Returns 0.0 if Kelly is negative (no edge).
    """
    if edge_pct is None or market_prob is None:
        return None

    market_prob_f = float(market_prob)
    if market_prob_f <= 0.0 or market_prob_f >= 1.0:
        return None

    # Determine model probability for the bet side
    if ats_pick == home:
        p = home_prob
        q = market_prob_f
    else:
        p = 1.0 - home_prob
        q = 1.0 - market_prob_f

    if q <= 0.0 or q >= 1.0:
        return None

    b = (1.0 - q) / q
    kelly = (p * b - (1.0 - p)) / b
    return round(max(0.0, 0.5 * kelly), 4)


def _confidence_tier(
    edge: float | None,
    home_prob: float,
    projected_margin: float | None,
) -> str:
    """Assign confidence tier using BettingRouter's edge-based system.

    Tiers: Best Bet (>=8% edge + agree), Solid Pick (>=4%), Lean (>=2%), Skip.
    Falls back to edge-only when projected_margin is unavailable.
    """
    from src.models.betting_router import confidence_tier, model_agreement

    agree = True
    if projected_margin is not None:
        agree = model_agreement(home_prob, projected_margin)

    if edge is not None and edge > 0:
        return confidence_tier(edge, agree)

    # No odds available — use model probability but cap at Solid Pick
    # (Best Bet requires market confirmation via edge > 0)
    max_prob = max(home_prob, 1.0 - home_prob)
    if max_prob >= 0.72:
        return "Solid Pick" if agree else "Lean"
    if max_prob >= 0.60:
        return "Lean"
    return "Skip"


def _build_pick_row(
    pred: dict,
    team_names: dict[str, str],
    lines: dict[tuple[str, str], dict],
    margin_lookup: dict[tuple[str, str, str], float | None] | None = None,
) -> dict:
    """Convert a DB prediction row into the todays_picks.json schema."""
    home = pred["home_team"]
    away = pred["away_team"]
    game_date = pred.get("game_date", "")
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

    # Projected margin: pull from margin_lookup keyed by (home, away, game_date)
    projected_margin: float | None = (margin_lookup or {}).get((home, away, game_date))

    # Derived enrichment fields
    kelly = _compute_kelly_fraction(edge_pct, market_prob, ats_pick, home, home_prob)
    tier = _confidence_tier(edge_pct, home_prob, projected_margin)
    model_confidence = round(
        home_prob * 100 if predicted_winner == home else away_prob * 100
    )

    return {
        "game_date": game_date,
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
        "kelly_fraction": kelly,
        "projected_margin": projected_margin,
        "confidence_tier": tier,
        "model_confidence": model_confidence,
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

    margin_lookup = _build_margin_lookup(preds)

    picks = [_build_pick_row(p, team_names, lines, margin_lookup) for p in preds]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(picks, fh, separators=(",", ":"), default=str)

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
