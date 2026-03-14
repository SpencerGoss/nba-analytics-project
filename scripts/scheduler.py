#!/usr/bin/env python3
"""
Hourly dashboard refresh scheduler.

Run via Windows Task Scheduler or cron.

Usage:
  python scripts/scheduler.py                    # run all builders
  python scripts/scheduler.py --dry-run          # list what would run
  python scripts/scheduler.py --builder build_performance.py
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import logging

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
META_PATH = PROJECT_ROOT / "dashboard" / "data" / "meta.json"
PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

# ---------------------------------------------------------------------------
# Builder registry — ordered by dependency
# ---------------------------------------------------------------------------
# Stage 1: live data (no dependencies)
# Stage 2: game context (needs game logs)
# Stage 3: picks / value bets (needs context)
# Stage 4: explainers (needs picks)
# Stage 5: independent builders (parallel-safe)
BUILDERS: list[list[str]] = [
    # Stage 1
    ["scripts/fetch_live_scores.py"],
    # Stage 2
    ["scripts/build_game_context.py"],
    # Stage 3
    ["scripts/build_picks.py"],
    ["scripts/build_value_bets.py"],
    # Stage 4
    ["scripts/build_explainers.py"],
    # Stage 5
    ["scripts/build_trends.py"],
    ["scripts/build_h2h.py"],
    ["scripts/build_matchup_analysis.py"],
    ["scripts/build_props.py"],
    ["scripts/build_totals.py"],
    ["scripts/build_power_rankings.py"],
    ["scripts/build_injuries.py"],
    ["scripts/build_line_movement.py"],
    ["scripts/build_performance.py"],
    ["scripts/build_standings.py"],
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_builder_path(entry: list[str]) -> Path:
    """Return absolute Path for a builder entry (relative to project root)."""
    return PROJECT_ROOT / entry[0]


def _builder_name(entry: list[str]) -> str:
    return entry[0]


def _python_exe() -> str:
    """Return the venv python path, falling back to sys.executable."""
    if PYTHON.exists():
        return str(PYTHON)
    return sys.executable


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(msg: str) -> None:
    log.info(f"[{_timestamp()}] {msg}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_builder(entry: list[str]) -> tuple[bool, str]:
    """
    Run a single builder script.

    Returns (success, message).
    Skips gracefully if the script file does not exist.
    """
    script_path = _resolve_builder_path(entry)
    name = _builder_name(entry)

    if not script_path.exists():
        msg = f"SKIP  {name} (file not found)"
        _log(msg)
        return True, msg  # skipped counts as non-failure

    cmd = [_python_exe(), str(script_path)]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5-minute per-builder timeout
        )
        if result.returncode == 0:
            msg = f"OK    {name}"
            _log(msg)
            return True, msg
        else:
            stderr_snippet = (result.stderr or "").strip()[-500:]
            msg = f"FAIL  {name} (exit {result.returncode}): {stderr_snippet}"
            _log(msg)
            return False, msg
    except subprocess.TimeoutExpired:
        msg = f"FAIL  {name} (timed out after 300s)"
        _log(msg)
        return False, msg
    except Exception as exc:
        msg = f"FAIL  {name} (exception: {exc})"
        _log(msg)
        return False, msg


def run_all(builders: list[list[str]]) -> tuple[list[str], list[str]]:
    """Run builders in order; return (run_names, failed_names)."""
    run_names: list[str] = []
    failed_names: list[str] = []

    for entry in builders:
        success, _ = run_builder(entry)
        name = _builder_name(entry)
        script_path = _resolve_builder_path(entry)
        if script_path.exists():
            run_names.append(name)
        if not success:
            failed_names.append(name)

    return run_names, failed_names


def update_meta(run_names: list[str], failed_names: list[str]) -> None:
    """Merge scheduler results into dashboard/data/meta.json."""
    existing: dict = {}
    if META_PATH.exists():
        try:
            with open(META_PATH, encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    existing.update(
        {
            "last_updated": _timestamp(),
            "builders_run": run_names,
            "failed_builders": failed_names,
        }
    )

    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)

    _log(f"meta.json updated  ({len(run_names)} run, {len(failed_names)} failed)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NBA analytics hourly dashboard refresh scheduler.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List builders that would run without executing them.",
    )
    parser.add_argument(
        "--builder",
        metavar="NAME",
        help=(
            "Run a single builder by script name "
            "(e.g. build_performance.py). "
            "Skips meta.json update."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.dry_run:
        _log("DRY-RUN -- builders that would execute:")
        for entry in BUILDERS:
            path = _resolve_builder_path(entry)
            status = "EXISTS" if path.exists() else "MISSING"
            log.info(f"  [{status}] {_builder_name(entry)}")
        return 0

    if args.builder:
        # Find the matching entry
        target = args.builder.strip()
        matched = [e for e in BUILDERS if _builder_name(e) == target or Path(_builder_name(e)).name == target]
        if not matched:
            _log(f"ERROR: builder '{target}' not found in registry")
            return 1
        success, _ = run_builder(matched[0])
        return 0 if success else 1

    # Full run
    _log(f"Starting scheduler run ({len(BUILDERS)} builders registered)")
    run_names, failed_names = run_all(BUILDERS)
    update_meta(run_names, failed_names)

    if failed_names:
        _log(f"Completed with failures: {failed_names}")
        return 1

    _log("All builders completed successfully.")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
