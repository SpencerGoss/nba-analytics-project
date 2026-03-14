"""Capture closing-line odds snapshot.

Run at ~9 PM after games tip off. Captures current lines as near-closing
lines for CLV tracking. Uses the existing fetch_odds infrastructure.

Usage:
    python scripts/odds_snapshot.py
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    python = sys.executable
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"[{timestamp}] Starting closing-line odds snapshot...")

    # Run fetch_odds to get latest lines
    result = subprocess.run(
        [python, str(PROJECT_ROOT / "scripts" / "fetch_odds.py")],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"  fetch_odds failed: {result.stderr[:500]}")
        sys.exit(1)

    print(f"  Odds snapshot captured successfully")

    # Also run build_line_movement to track movement
    result2 = subprocess.run(
        [python, str(PROJECT_ROOT / "scripts" / "pipeline_runner.py"),
         "--builder", "build_line_movement"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )

    if result2.returncode != 0:
        print(f"  build_line_movement warning: {result2.stderr[:300]}")
    else:
        print(f"  Line movement updated")

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Odds snapshot complete")


if __name__ == "__main__":
    main()
