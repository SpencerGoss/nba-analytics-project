"""Train one model per core task.

Consolidated entrypoint that can also rebuild feature tables before training.

Usage:
    python src/models/train_all_models.py
    python src/models/train_all_models.py --rebuild-features
"""

import argparse
import os
import sys
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.game_outcome_model import train_game_outcome_model
from src.models.player_performance_model import train_player_models
from src.models.playoff_odds_model import simulate_playoff_odds


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rebuild-features", action="store_true",
                   help="Recompute team + player feature tables before training models")
    return p.parse_args()


def maybe_rebuild_features(enabled: bool) -> None:
    if not enabled:
        return

    print("\n" + "-" * 72)
    print("FEATURE ENGINEERING")
    print("-" * 72)
    from src.features.team_game_features import build_team_game_features, build_matchup_dataset
    from src.features.player_features import build_player_game_features

    build_team_game_features()
    build_matchup_dataset()
    build_player_game_features()


def main() -> None:
    args = parse_args()
    start = datetime.now()
    print("=" * 72)
    print("NBA ANALYTICS — CONSOLIDATED TASK TRAINER")
    print("=" * 72)
    print("This run trains one workflow per core task:")
    print("  1) Game outcomes")
    print("  2) Player performance")
    print("  3) Playoff odds")

    maybe_rebuild_features(args.rebuild_features)

    print("\n" + "-" * 72)
    print("TASK 1/3 — GAME OUTCOME")
    print("-" * 72)
    _, game_metrics = train_game_outcome_model()

    print("\n" + "-" * 72)
    print("TASK 2/3 — PLAYER PERFORMANCE")
    print("-" * 72)
    _, player_metrics = train_player_models()

    print("\n" + "-" * 72)
    print("TASK 3/3 — PLAYOFF ODDS")
    print("-" * 72)
    playoff_df = simulate_playoff_odds()

    elapsed = datetime.now() - start
    print("\n" + "=" * 72)
    print("RUN SUMMARY")
    print("=" * 72)
    print(
        f"Game outcome → model={game_metrics.get('selected_model', 'n/a')} | "
        f"test_acc={game_metrics.get('test_accuracy', game_metrics.get('gb_accuracy', 0)):.4f}"
    )
    print(
        "Player performance → "
        + ", ".join([f"{k.upper()} ({v.get('selected_model','n/a')}): MAE={v.get('mae', 0):.3f}"
                       for k, v in player_metrics.items()])
    )
    print(f"Playoff odds → teams simulated={len(playoff_df):,}")
    print(f"Elapsed: {elapsed.seconds // 60}m {elapsed.seconds % 60}s")
    print("=" * 72)


if __name__ == "__main__":
    main()
