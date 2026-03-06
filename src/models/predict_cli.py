"""Prediction CLI for game, player, ATS, and value-bet predictions.

Examples:
  python src/models/predict_cli.py game --home BOS --away LAL
  python src/models/predict_cli.py player --name "LeBron James"
  python src/models/predict_cli.py ats --home BOS --away LAL --spread -3.5
  python src/models/predict_cli.py value-bet --threshold 0.07
"""

import argparse
import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd

from src.models.ats_model import predict_ats
from src.models.game_outcome_model import predict_game
from src.models.player_performance_model import predict_player_next_game
from src.models.value_bet_detector import run_value_bet_scan, no_vig_prob


def parse_args():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="command", required=True)

    g = sp.add_parser("game", help="Predict game outcome probability")
    g.add_argument("--home", required=True, help="Home team abbreviation (e.g., BOS)")
    g.add_argument("--away", required=True, help="Away team abbreviation (e.g., LAL)")
    g.add_argument(
        "--date",
        default=None,
        help="Game date as YYYY-MM-DD (default: today). Stored in prediction history.",
    )

    pl = sp.add_parser("player", help="Predict player PTS/REB/AST")
    pl.add_argument("--name", required=True, help="Exact player name")

    ats = sp.add_parser("ats", help="Predict ATS (against the spread) outcome")
    ats.add_argument("--home", required=True, help="Home team abbreviation (e.g., BOS)")
    ats.add_argument("--away", required=True, help="Away team abbreviation (e.g., LAL)")
    ats.add_argument("--spread", required=True, type=float, help="Point spread (home perspective, e.g., -3.5)")
    ats.add_argument("--home-ml", type=float, default=None, help="Home moneyline (e.g., -150)")
    ats.add_argument("--away-ml", type=float, default=None, help="Away moneyline (e.g., +130)")

    vb = sp.add_parser("value-bet", help="Scan for value bets (model vs market disagreement)")
    vb.add_argument("--live", action="store_true", help="Use live odds from Pinnacle (free, no API key required)")
    vb.add_argument("--threshold", type=float, default=0.05, help="Edge threshold for flagging value bets (default: 0.05)")

    return p.parse_args()


def _handle_ats(args) -> dict:
    """Build a feature row from latest matchup data + CLI spread/moneylines, then predict ATS."""
    matchup_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "features", "game_matchup_features.csv",
    )
    if not os.path.exists(matchup_path):
        return {"error": f"Matchup features not found at {matchup_path}. Run feature build first."}

    df = pd.read_csv(matchup_path)
    df["game_date"] = pd.to_datetime(df["game_date"])

    exact = df[(df["home_team"] == args.home) & (df["away_team"] == args.away)]
    if not exact.empty:
        row = exact.sort_values("game_date").iloc[-1].copy()
    else:
        home_rows = df[df["home_team"] == args.home].sort_values("game_date")
        away_rows = df[df["away_team"] == args.away].sort_values("game_date")
        if home_rows.empty or away_rows.empty:
            return {"error": f"Not enough history for {args.home} vs {args.away}."}
        row = home_rows.iloc[-1].copy()
        away_source = away_rows.iloc[-1]
        for c in df.columns:
            if c.startswith("away_"):
                row[c] = away_source.get(c, row.get(c, np.nan))
        for c in df.columns:
            if c.startswith("diff_"):
                base = c.replace("diff_", "")
                h_col, a_col = f"home_{base}", f"away_{base}"
                if h_col in row.index and a_col in row.index:
                    row[c] = row[h_col] - row[a_col]

    row["spread"] = args.spread
    if args.home_ml is not None and args.away_ml is not None:
        home_nv, away_nv = no_vig_prob(args.home_ml, args.away_ml)
        row["home_implied_prob"] = home_nv
        row["away_implied_prob"] = away_nv

    row_df = row.to_frame().T
    result = predict_ats(row_df)

    return {
        "home_team": args.home,
        "away_team": args.away,
        "spread": args.spread,
        "covers_spread_prob": round(float(result["covers_spread_prob"].iloc[0]), 4),
        "covers_spread_pred": int(result["covers_spread_pred"].iloc[0]),
    }


def _handle_value_bet(args) -> list:
    """Run value-bet scan and return results."""
    return run_value_bet_scan(
        use_live_odds=args.live,
        threshold=args.threshold,
    )


def main() -> None:
    args = parse_args()
    if args.command == "game":
        out = predict_game(args.home, args.away, game_date=args.date)
    elif args.command == "player":
        out = predict_player_next_game(args.name)
    elif args.command == "ats":
        out = _handle_ats(args)
    else:
        out = _handle_value_bet(args)

    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
