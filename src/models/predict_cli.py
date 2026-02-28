"""Prediction CLI for game and player props.

Examples:
  python src/models/predict_cli.py game --home BOS --away LAL
  python src/models/predict_cli.py player --name "LeBron James"
"""

import argparse
import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.game_outcome_model import predict_game
from src.models.player_performance_model import predict_player_next_game


def parse_args():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="command", required=True)

    g = sp.add_parser("game", help="Predict game outcome probability")
    g.add_argument("--home", required=True, help="Home team abbreviation (e.g., BOS)")
    g.add_argument("--away", required=True, help="Away team abbreviation (e.g., LAL)")

    pl = sp.add_parser("player", help="Predict player PTS/REB/AST")
    pl.add_argument("--name", required=True, help="Exact player name")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "game":
        out = predict_game(args.home, args.away)
    else:
        out = predict_player_next_game(args.name)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
