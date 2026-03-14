"""
Model Evaluation Runner
=========================
Runs all three evaluation tools in sequence:
  1. Walk-forward backtesting  (backtesting.py)
  2. Calibration analysis      (calibration.py)
  3. Explainability / SHAP     (model_explainability.py)

All outputs are written to the reports/ folder.

Usage:
    python src/models/run_evaluation.py

    Optional flags:
        --skip-backtest     skip the walk-forward backtest (slowest step)
        --skip-calibration  skip calibration analysis
        --skip-explain      skip SHAP/permutation explainability

Notes:
  - The models must be trained first (run the individual model scripts).
  - Walk-forward backtesting retrains the model for each season and can
    take 5-15 minutes depending on hardware.
  - SHAP requires the shap package (pip install shap).
    If not installed, permutation importance is used as a fallback.
"""

import argparse
import sys
import os
from datetime import datetime
import logging

log = logging.getLogger(__name__)

# Ensure the project root is on sys.path so that `src.*` imports resolve
# regardless of which directory the script is launched from.
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

start = datetime.now()
log.info(f"[{start.strftime('%Y-%m-%d %H:%M:%S')}] Starting model evaluation...")
print()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-backtest",    action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--skip-explain",     action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    # -- 1. Walk-forward backtesting -------------------------------------------
    if not args.skip_backtest:
        print("-" * 60)
        log.info("STEP 1 OF 3 -- Walk-Forward Backtesting")
        print("-" * 60)
        try:
            from src.models.backtesting import (
                run_game_outcome_backtest,
                run_player_model_backtest,
                write_summary_report,
            )
            game_results   = run_game_outcome_backtest()
            player_results = run_player_model_backtest()
            write_summary_report(game_results, player_results)
            log.info("\n[OK] Backtesting complete")
        except Exception as e:
            log.error(f"\n[FAIL] Backtesting failed: {e}")
            import traceback; traceback.print_exc()
    else:
        log.warning("Step 1: Backtesting skipped (--skip-backtest)")

    print()

    # -- 2. Calibration analysis -----------------------------------------------
    if not args.skip_calibration:
        print("-" * 60)
        log.info("STEP 2 OF 3 -- Calibration Analysis")
        print("-" * 60)
        try:
            from src.models.calibration import run_calibration_analysis
            metrics = run_calibration_analysis()
            log.info("\n[OK] Calibration analysis complete")
        except FileNotFoundError as e:
            log.error(f"\n[FAIL] Calibration failed: {e}")
        except Exception as e:
            log.error(f"\n[FAIL] Calibration failed: {e}")
            import traceback; traceback.print_exc()
    else:
        log.warning("Step 2: Calibration analysis skipped (--skip-calibration)")

    print()

    # -- 3. Explainability -----------------------------------------------------
    if not args.skip_explain:
        print("-" * 60)
        log.info("STEP 3 OF 3 -- Explainability")
        print("-" * 60)
        try:
            from src.models.model_explainability import (
                explain_game_outcome_model,
                explain_player_model,
            )
            explain_game_outcome_model()
            explain_player_model()
            log.info("\n[OK] Explainability analysis complete")
        except FileNotFoundError as e:
            log.error(f"\n[FAIL] Explainability failed: {e}")
        except Exception as e:
            log.error(f"\n[FAIL] Explainability failed: {e}")
            import traceback; traceback.print_exc()
    else:
        log.warning("Step 3: Explainability skipped (--skip-explain)")

    # -- Done ------------------------------------------------------------------
    elapsed = datetime.now() - start
    print()
    print("=" * 60)
    log.info(f"Evaluation complete -- {elapsed.seconds // 60}m {elapsed.seconds % 60}s")
    print()
    log.info("Reports written to:")
    log.info("  reports/backtest_game_outcome.csv")
    log.info("  reports/backtest_player_*.csv")
    log.info("  reports/backtest_summary.txt")
    log.info("  reports/calibration/calibration_curve.png")
    log.info("  reports/calibration/brier_score_by_season.png")
    log.info("  reports/calibration/calibration_metrics.csv")
    log.info("  reports/explainability/  (charts + CSVs)")
    print("=" * 60)


if __name__ == "__main__":
    main()
