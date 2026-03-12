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
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")

    from src.models.game_outcome_model import _get_current_season_code, _synthesize_matchup_row
    current_season = _get_current_season_code()

    # Try current-season exact matchup first; fall back to synthesis
    exact = df[
        (df["home_team"] == args.home)
        & (df["away_team"] == args.away)
        & (df["season"].astype(str) == current_season)
    ]
    if not exact.empty:
        row = exact.sort_values("game_date").iloc[-1].copy()
        # Refresh Elo ratings
        try:
            from src.features.elo import get_current_elos
            current_elos = get_current_elos()
            home_elo = current_elos.get(args.home, row.get("home_elo", 1500.0))
            away_elo = current_elos.get(args.away, row.get("away_elo", 1500.0))
            row["home_elo"] = home_elo
            row["away_elo"] = away_elo
            row["diff_elo"] = home_elo - away_elo
        except Exception:
            pass
    else:
        feat_cols = [c for c in df.columns if c.startswith("diff_")]
        row = _synthesize_matchup_row(df, args.home, args.away, feat_cols)
        if row is None:
            return {"error": f"Not enough history for {args.home} vs {args.away}."}

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
