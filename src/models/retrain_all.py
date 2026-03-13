"""
Retrain all production models after feature pipeline updates.

Run this script after expanding lineup data or changing features:

    python src/models/retrain_all.py

Steps:
  1. Rebuild lineup features (from data/raw/lineups/)
  2. Rebuild game matchup features (includes lineup net rating diff columns)
  3. Retrain game outcome model (saves to models/artifacts/game_outcome_model.pkl)
  4. Re-run calibration (saves game_outcome_model_calibrated.pkl)
  5. Retrain ATS model (saves to models/artifacts/ats_model.pkl)

After retraining, verify with:
    python -m pytest tests/ -q
"""

import subprocess
import sys
import os

# Ensure we run from project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)


def run_step(description: str, cmd: list) -> bool:
    """Run a subprocess step and return True on success."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    # Ensure project root is on PYTHONPATH so 'from src...' imports work
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(cmd, capture_output=False, text=True, env=env)
    if result.returncode != 0:
        print(f"ERROR: Step failed with return code {result.returncode}")
        return False
    return True


def main():
    print("NBA Analytics — Full Model Retraining Pipeline")
    print("=" * 60)

    # Step 1: Rebuild lineup features
    ok = run_step(
        "Rebuild lineup features from raw CSV files",
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, '.'); "
         "from src.features.lineup_features import build_lineup_features; "
         "df = build_lineup_features(); "
         "print(f'Lineup features: {df.shape}'); "
         "print(f'Seasons covered: {sorted(df[\"season\"].unique())}')"],
    )
    if not ok:
        print("Lineup feature rebuild failed. Check data/raw/lineups/ directory.")
        sys.exit(1)

    # Step 2: Rebuild game matchup features
    ok = run_step(
        "Rebuild game matchup features (includes lineup diff columns)",
        [sys.executable, "src/features/team_game_features.py"],
    )
    if not ok:
        print("Matchup feature rebuild failed.")
        sys.exit(1)

    # Step 3: Retrain game outcome model
    ok = run_step(
        "Retrain game outcome model",
        [sys.executable, "src/models/game_outcome_model.py"],
    )
    if not ok:
        print("Game outcome model training failed.")
        sys.exit(1)

    # Step 4: Re-calibrate
    ok = run_step(
        "Re-run probability calibration",
        [sys.executable, "src/models/calibration.py"],
    )
    if not ok:
        print("Calibration failed.")
        sys.exit(1)

    # Step 5: Retrain ATS model
    ok = run_step(
        "Retrain ATS model",
        [sys.executable, "src/models/ats_model.py"],
    )
    if not ok:
        print("ATS model training failed.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("ALL STEPS COMPLETE")
    print("Run tests: python -m pytest tests/ -q")
    print("=" * 60)


if __name__ == "__main__":
    main()
