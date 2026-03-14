"""Wrapper for Task Scheduler: sets working directory then runs update.py.

Task Scheduler's schtasks.exe doesn't support setting a working directory
without elevated privileges. This wrapper ensures update.py runs from
the project root regardless of how Task Scheduler invokes it.

Usage (Task Scheduler):
    .venv/Scripts/python.exe scripts/run_daily_full.py
"""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

sys.exit(
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "update.py")],
        cwd=str(PROJECT_ROOT),
    ).returncode
)
