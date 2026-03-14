"""Weekly model retraining — runs Saturday 3:00 AM.

Retrains the game outcome model, ATS model, and prop models,
then runs calibration. Ensures models stay fresh even if the
daily pipeline's weekly retrain logic gets skipped due to errors.

Usage:
    python scripts/weekly_retrain.py
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Subprocesses need PYTHONPATH to import from src.*
_ENV = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}


def run_step(name, cmd):
    """Run a pipeline step, return success bool."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {name}...")
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        env=_ENV,
    )
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[:500]}")
        return False
    print(f"  OK")
    return True


def main():
    python = sys.executable
    print(f"=== Weekly Model Retrain — {datetime.now().strftime('%Y-%m-%d %H:%M')} ===\n")

    steps = [
        ("Step 1: Rebuild Elo ratings",
         [python, "-c",
          "from src.features.elo import build_elo_ratings; "
          "build_elo_ratings()"]),

        ("Step 2: Rebuild team game features",
         [python, "-c",
          "from src.features.team_game_features import build_team_game_features; "
          "build_team_game_features()"]),

        ("Step 3: Rebuild matchup dataset",
         [python, "-c",
          "from src.features.team_game_features import build_matchup_dataset; "
          "build_matchup_dataset()"]),

        ("Step 4: Retrain game outcome model",
         [python, "-c",
          "from src.models.game_outcome_model import train_game_outcome_model; "
          "train_game_outcome_model()"]),

        ("Step 5: Run calibration",
         [python, "-c",
          "from src.models.calibration import run_calibration_analysis; "
          "run_calibration_analysis()"]),

        ("Step 6: Retrain ATS model",
         [python, "-c",
          "from src.models.ats_model import train_ats_model; "
          "train_ats_model()"]),

        ("Step 7: Retrain margin model",
         [python, "-c",
          "from src.models.margin_model import train_margin_model; "
          "train_margin_model()"]),
    ]

    results = {}
    for name, cmd in steps:
        results[name] = run_step(name, cmd)

    # Summary
    print(f"\n=== Results ===")
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")

    failed = sum(1 for ok in results.values() if not ok)
    if failed:
        print(f"\n{failed} step(s) failed. Check logs above.")
        sys.exit(1)
    else:
        print(f"\nAll steps passed. Models retrained successfully.")


if __name__ == "__main__":
    main()
