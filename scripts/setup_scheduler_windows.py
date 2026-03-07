#!/usr/bin/env python3
"""
Create a Windows Task Scheduler entry to run scheduler.py every hour.

Run once (as Administrator if Task Scheduler requires elevated access):
  python scripts/setup_scheduler_windows.py

Options:
  --task-name NAME    Override the default task name
  --interval-hours N  Override the repeat interval (default: 1)
  --remove            Delete the scheduled task instead of creating it
"""
import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON_EXE = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
SCHEDULER_SCRIPT = PROJECT_ROOT / "scripts" / "scheduler.py"

DEFAULT_TASK_NAME = "NBAAnalyticsHourlyRefresh"


def _schtasks(*args: str) -> subprocess.CompletedProcess:
    """Run schtasks.exe with the given arguments; print output."""
    cmd = ["schtasks"] + list(args)
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip(), file=sys.stderr)
    return result


def create_task(task_name: str, interval_hours: int) -> int:
    """Create (or replace) a scheduled task."""
    if not PYTHON_EXE.exists():
        print(f"ERROR: Python executable not found: {PYTHON_EXE}")
        print("Activate the venv and retry, or create the venv first.")
        return 1

    if not SCHEDULER_SCRIPT.exists():
        print(f"ERROR: Scheduler script not found: {SCHEDULER_SCRIPT}")
        return 1

    # Build the task action: "<venv_python>" "<scheduler.py>"
    run_cmd = f'"{PYTHON_EXE}" "{SCHEDULER_SCRIPT}"'

    # /SC HOURLY /MO <interval> repeats every N hours
    result = _schtasks(
        "/Create",
        "/F",                       # overwrite if exists
        "/TN", task_name,
        "/TR", run_cmd,
        "/SC", "HOURLY",
        "/MO", str(interval_hours),
        "/RL", "HIGHEST",           # run with highest available privileges
        "/IT",                      # run only when user is logged on (interactive)
    )

    if result.returncode == 0:
        print(f"\nTask '{task_name}' created successfully.")
        print(f"  Runs every {interval_hours} hour(s).")
        print(f"  Command: {run_cmd}")
        print("\nVerify with:  schtasks /Query /TN", task_name, "/FO LIST")
        return 0

    print(f"\nERROR: schtasks exited with code {result.returncode}")
    return result.returncode


def remove_task(task_name: str) -> int:
    """Delete an existing scheduled task by name."""
    result = _schtasks("/Delete", "/TN", task_name, "/F")
    if result.returncode == 0:
        print(f"Task '{task_name}' removed.")
        return 0
    print(f"ERROR: Could not remove task '{task_name}' (exit {result.returncode})")
    return result.returncode


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register the NBA analytics hourly scheduler in Windows Task Scheduler.",
    )
    parser.add_argument(
        "--task-name",
        default=DEFAULT_TASK_NAME,
        help=f"Task name (default: {DEFAULT_TASK_NAME})",
    )
    parser.add_argument(
        "--interval-hours",
        type=int,
        default=1,
        metavar="N",
        help="Repeat interval in hours (default: 1)",
    )
    parser.add_argument(
        "--remove",
        action="store_true",
        help="Remove the scheduled task instead of creating it.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.remove:
        return remove_task(args.task_name)

    return create_task(args.task_name, args.interval_hours)


if __name__ == "__main__":
    sys.exit(main())
