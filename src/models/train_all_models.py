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
from src.models.calibration import run_calibration_analysis
from src.models.ats_model import train_ats_model
import logging

log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rebuild-features", action="store_true",
                   help="Recompute team + player feature tables before training models")
    return p.parse_args()


def maybe_rebuild_features(enabled: bool) -> None:
    if not enabled:
        return

    log.info("\n" + "-" * 72)
    log.info("FEATURE ENGINEERING")
    print("-" * 72)
    from src.features.team_game_features import build_team_game_features, build_matchup_dataset
    from src.features.player_features import build_player_game_features

    build_team_game_features()
    build_matchup_dataset()
    build_player_game_features()


def main() -> None:
    args = parse_args()
    start = datetime.now()
    log.info("=" * 72)
    log.info("NBA ANALYTICS — CONSOLIDATED TASK TRAINER")
    log.info("=" * 72)
    log.info("This run trains one workflow per core task:")
    log.info("  1) Game outcomes")
    log.info("  2) Player performance")
    log.info("  3) Playoff odds")
    log.info("  4) Calibration analysis")
    log.info("  5) ATS spread model")

    maybe_rebuild_features(args.rebuild_features)

    log.info("\n" + "-" * 72)
    log.info("TASK 1/5 — GAME OUTCOME")
    print("-" * 72)
    _, game_metrics = train_game_outcome_model()

    log.info("\n" + "-" * 72)
    log.info("TASK 2/5 — PLAYER PERFORMANCE")
    print("-" * 72)
    _, player_metrics = train_player_models()

    log.info("\n" + "-" * 72)
    log.info("TASK 3/5 — PLAYOFF ODDS")
    print("-" * 72)
    playoff_df = simulate_playoff_odds()

    log.info("\n" + "-" * 72)
    log.info("TASK 4/5 -- CALIBRATION")
    print("-" * 72)
    cal_metrics = run_calibration_analysis()

    log.info("\n" + "-" * 72)
    log.info("TASK 5/5 -- ATS SPREAD MODEL")
    print("-" * 72)
    _, ats_metrics = train_ats_model()

    elapsed = datetime.now() - start
    log.info("\n" + "=" * 72)
    log.info("RUN SUMMARY")
    log.info("=" * 72)
    log.info(f"Game outcome -> model={game_metrics.get('selected_model', 'n/a')} | "
        f"test_acc={game_metrics.get('test_accuracy', game_metrics.get('gb_accuracy', 0)):.4f}")
    log.info("Player performance -> "
        + ", ".join([f"{k.upper()} ({v.get('selected_model','n/a')}): MAE={v.get('mae', 0):.3f}"
                       for k, v in player_metrics.items()]))
    log.info(f"Playoff odds -> teams simulated={len(playoff_df):,}")
    log.info(f"Calibration -> Brier={cal_metrics.get('brier_calibrated', 0):.5f} | "
        f"ECE={cal_metrics.get('ece_calibrated', 0):.5f}")
    log.info(f"ATS model -> model={ats_metrics.get('model_type', 'n/a')} | "
        f"test_acc={ats_metrics.get('test_accuracy', 0):.4f}")
    log.info(f"Elapsed: {elapsed.seconds // 60}m {elapsed.seconds % 60}s")
    log.info("=" * 72)


if __name__ == "__main__":
    main()
