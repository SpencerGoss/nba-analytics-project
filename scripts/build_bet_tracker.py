"""
build_bet_tracker.py  --  produce dashboard/data/bet_tracker.json

Reads game_predictions and clv_tracking from predictions_history.db,
joins them to produce a bet tracker export for the dashboard.

Output per record:
  game_date, home_team, away_team, predicted_winner,
  home_win_prob, away_win_prob, ats_pick, spread, edge_pct,
  value_bet, actual_winner, result, created_at

Sorted by game_date descending, limited to last 60 days.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import logging

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "database" / "predictions_history.db"
VALUE_BETS_JSON = PROJECT_ROOT / "dashboard" / "data" / "value_bets.json"
OUT_JSON = PROJECT_ROOT / "dashboard" / "data" / "bet_tracker.json"

LOOKBACK_DAYS = 60


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result_label(actual_home_win, predicted_winner: str, home_team: str) -> str | None:
    """Return 'WIN', 'LOSS', or None (not yet played)."""
    if actual_home_win is None or pd.isna(actual_home_win):
        return None
    home_won = bool(int(actual_home_win))
    predicted_home = predicted_winner == home_team
    correct = (predicted_home and home_won) or (not predicted_home and not home_won)
    return "WIN" if correct else "LOSS"


def _actual_winner(actual_home_win, home_team: str, away_team: str) -> str | None:
    if actual_home_win is None or pd.isna(actual_home_win):
        return None
    return home_team if bool(int(actual_home_win)) else away_team


# ---------------------------------------------------------------------------
# Value bet lookup
# ---------------------------------------------------------------------------

def _load_value_bets() -> dict[tuple[str, str, str], dict]:
    """Return a dict keyed by (game_date, home_team, away_team) -> vbet record."""
    if not VALUE_BETS_JSON.exists():
        return {}
    try:
        with VALUE_BETS_JSON.open(encoding="utf-8") as fh:
            bets = json.load(fh)
        return {
            (b.get("game_date", ""), b.get("home_team", ""), b.get("away_team", "")): b
            for b in bets
            if isinstance(b, dict)
        }
    except (json.JSONDecodeError, OSError):
        return {}


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_bet_tracker() -> list[dict]:
    if not DB_PATH.exists():
        log.warning("  predictions_history.db not found -- outputting []")
        return []

    cutoff = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")

    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql_query(
            """
            SELECT id, created_at, game_date, home_team, away_team,
                   home_win_prob, away_win_prob, actual_home_win, notes
            FROM game_predictions
            WHERE game_date >= ?
            ORDER BY game_date DESC, id DESC
            """,
            con,
            params=(cutoff,),
        )

        # Deduplicate: keep the latest prediction per (game_date, home, away)
        df = df.drop_duplicates(subset=["game_date", "home_team", "away_team"], keep="first")

        # Load CLV tracking for spread info
        tables = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {r[0] for r in tables}

        clv_df = pd.DataFrame()
        if "clv_tracking" in table_names:
            clv_df = pd.read_sql_query(
                "SELECT game_date, home_team, away_team, opening_spread FROM clv_tracking",
                con,
            )

    value_bets = _load_value_bets()

    # Build CLV lookup: (game_date, home, away) -> opening_spread
    clv_lookup: dict[tuple[str, str, str], float | None] = {}
    if not clv_df.empty:
        for _, row in clv_df.iterrows():
            key = (str(row["game_date"]), str(row["home_team"]), str(row["away_team"]))
            spread = None if pd.isna(row["opening_spread"]) else float(row["opening_spread"])
            clv_lookup[key] = spread

    results: list[dict] = []
    for _, row in df.iterrows():
        game_date = str(row["game_date"])
        home = str(row["home_team"])
        away = str(row["away_team"])
        home_prob = float(row["home_win_prob"])
        away_prob = float(row["away_win_prob"])

        predicted_winner = home if home_prob >= away_prob else away

        clv_key = (game_date, home, away)
        spread = clv_lookup.get(clv_key)
        spread_str = f"{spread:+.1f}" if spread is not None else None

        ats_pick = f"{predicted_winner} {spread_str}" if spread_str else predicted_winner

        vbet = value_bets.get(clv_key)
        is_value = vbet is not None
        edge_pct = float(vbet["edge_pct"]) if vbet and vbet.get("edge_pct") is not None else None

        actual_win = row["actual_home_win"]
        actual_winner_val = _actual_winner(actual_win, home, away)
        result_label = _result_label(actual_win, predicted_winner, home)

        results.append({
            "game_date": game_date,
            "home_team": home,
            "away_team": away,
            "predicted_winner": predicted_winner,
            "home_win_prob": round(home_prob, 4),
            "away_win_prob": round(away_prob, 4),
            "ats_pick": ats_pick,
            "spread": spread_str,
            "edge_pct": round(edge_pct * 100, 2) if edge_pct is not None else None,
            "value_bet": is_value,
            "actual_winner": actual_winner_val,
            "result": result_label,
            "created_at": str(row["created_at"]),
        })

    results.sort(key=lambda x: x["game_date"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("Building bet tracker...")
    records = build_bet_tracker()
    log.info(f"  {len(records)} records (last {LOOKBACK_DAYS} days)")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)

    log.info(f"  Written -> {OUT_JSON}")
    for rec in records[:5]:
        result_str = rec["result"] or "pending"
        vb_str = "VALUE" if rec["value_bet"] else ""
        log.info(f"  {rec['away_team']} @ {rec['home_team']}  {rec['game_date']}  "
            f"pred={rec['predicted_winner']}  {result_str}  {vb_str}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
