"""Unified NBA pipeline runner with health reporting, modes, and dry-run.

Replaces the fragmented update.py Step 7 approach with a structured,
mode-aware runner that tracks success/failure per builder and writes
a health report to dashboard/data/pipeline_report.json.

Modes:
    full          -- Post-game (4 AM): fetch all data, rebuild features,
                     retrain (weekly), generate predictions, build all
                     dashboard JSONs, deploy
    injuries_odds -- Midday refresh: injuries + odds only, rebuild
                     affected JSONs
    pretip        -- Pre-tip (6 PM): final odds refresh, regenerate
                     predictions + picks

Usage:
    python scripts/pipeline_runner.py --mode full
    python scripts/pipeline_runner.py --mode full --dry-run
    python scripts/pipeline_runner.py --builder build_standings
    python scripts/pipeline_runner.py --mode pretip --resume
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PYTHON = sys.executable
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
LOGS_DIR = PROJECT_ROOT / "logs"
STATE_FILE = LOGS_DIR / "pipeline_state.json"


# ---------------------------------------------------------------------------
# Builder Registry
# ---------------------------------------------------------------------------
# Phase number determines execution order. Builders within a phase run
# sequentially (to avoid API rate limits and honour data dependencies).
#
# Script names match actual files in scripts/.
# update.py Steps 1-6 remain in update.py; the runner handles Step 7+.
# ---------------------------------------------------------------------------

BUILDERS = {
    # Phase 1 -- Resolve yesterday's prediction outcomes (must be first)
    "backfill_outcomes": {
        "phase": 1,
        "script": "backfill_outcomes.py",
        "label": "Backfill prediction outcomes",
        "modes": ["full"],
    },
    # Phase 2 -- Fresh odds (needed by picks, value bets, props, line movement)
    "fetch_odds": {
        "phase": 2,
        "script": "fetch_odds.py",
        "label": "Fetch sportsbook odds and props",
        "modes": ["full", "injuries_odds", "pretip"],
    },
    # Phase 3 -- Picks & value bets (depend on fresh odds)
    "build_picks": {
        "phase": 3,
        "script": "build_picks.py",
        "label": "Today's picks",
        "modes": ["full", "pretip"],
    },
    "build_value_bets": {
        "phase": 3,
        "script": "build_value_bets.py",
        "label": "Value bets",
        "modes": ["full", "pretip"],
    },
    # Phase 4 -- Static / team / standings builders (no inter-dependency)
    "build_standings": {
        "phase": 4,
        "script": "build_standings.py",
        "label": "Standings",
        "modes": ["full"],
    },
    "build_injuries": {
        "phase": 4,
        "script": "build_injuries.py",
        "label": "Injury report",
        "modes": ["full", "injuries_odds"],
    },
    "build_power_rankings": {
        "phase": 4,
        "script": "build_power_rankings.py",
        "label": "Power rankings",
        "modes": ["full"],
    },
    "build_h2h": {
        "phase": 4,
        "script": "build_h2h.py",
        "label": "Head-to-head",
        "modes": ["full"],
    },
    "build_streaks": {
        "phase": 4,
        "script": "build_streaks.py",
        "label": "Hot/cold streaks",
        "modes": ["full"],
    },
    "build_advanced_stats": {
        "phase": 4,
        "script": "build_advanced_stats.py",
        "label": "Advanced stats",
        "modes": ["full"],
    },
    "build_live_scores": {
        "phase": 4,
        "script": "build_live_scores.py",
        "label": "Live scores",
        "modes": ["full", "pretip"],
    },
    "build_playoff_odds": {
        "phase": 4,
        "script": "build_playoff_odds.py",
        "label": "Playoff odds",
        "modes": ["full"],
    },
    "build_trends": {
        "phase": 4,
        "script": "build_trends.py",
        "label": "Trends",
        "modes": ["full"],
    },
    "build_totals": {
        "phase": 4,
        "script": "build_totals.py",
        "label": "Totals",
        "modes": ["full"],
    },
    "build_game_context": {
        "phase": 4,
        "script": "build_game_context.py",
        "label": "Game context",
        "modes": ["full", "pretip"],
    },
    "build_explainers": {
        "phase": 4,
        "script": "build_explainers.py",
        "label": "Explainers",
        "modes": ["full"],
    },
    "build_matchup_analysis": {
        "phase": 4,
        "script": "build_matchup_analysis.py",
        "label": "Matchup analysis",
        "modes": ["full"],
    },
    "build_game_detail": {
        "phase": 4,
        "script": "build_game_detail.py",
        "label": "Game detail",
        "modes": ["full", "pretip"],
    },
    "build_elo_timeline": {
        "phase": 4,
        "script": "build_elo_timeline.py",
        "label": "Elo timeline",
        "modes": ["full"],
    },
    # Phase 5 -- Performance & betting builders (depend on backfill_outcomes)
    "build_performance": {
        "phase": 5,
        "script": "build_performance.py",
        "label": "Performance history",
        "modes": ["full"],
    },
    "build_accuracy_history": {
        "phase": 5,
        "script": "build_accuracy_history.py",
        "label": "Accuracy history",
        "modes": ["full"],
    },
    "build_line_movement": {
        "phase": 5,
        "script": "build_line_movement.py",
        "label": "Line movement",
        "modes": ["full", "pretip"],
    },
    "build_sharp_money": {
        "phase": 5,
        "script": "build_sharp_money.py",
        "label": "Sharp money signals",
        "modes": ["full", "pretip"],
    },
    "build_bet_tracker": {
        "phase": 5,
        "script": "build_bet_tracker.py",
        "label": "Bet tracker export",
        "modes": ["full"],
    },
    "build_props": {
        "phase": 5,
        "script": "build_props.py",
        "label": "Player props",
        "modes": ["full", "pretip"],
    },
    "build_clv": {
        "phase": 5,
        "script": "build_clv.py",
        "label": "CLV summary",
        "modes": ["full"],
    },
    # Phase 6 -- Heavy / player builders (run later to avoid slowing picks)
    "build_player_comparison": {
        "phase": 6,
        "script": "build_player_comparison.py",
        "label": "Player comparison",
        "modes": ["full"],
    },
    "build_player_detail": {
        "phase": 6,
        "script": "build_player_detail.py",
        "label": "Player detail stats",
        "modes": ["full"],
    },
    "build_season_history": {
        "phase": 6,
        "script": "build_season_history.py",
        "label": "Season history",
        "modes": ["full"],
    },
    # Phase 7 -- Meta (always last -- timestamps the completed run)
    "build_meta": {
        "phase": 7,
        "script": "build_meta.py",
        "label": "Dashboard metadata",
        "modes": ["full", "injuries_odds", "pretip"],
    },
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    """Configure structured logging to file and console."""
    LOGS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"pipeline_{timestamp}.log"

    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers on repeated calls
    if not logger.handlers:
        # File handler
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
        )
        logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# State persistence (for --resume)
# ---------------------------------------------------------------------------

def load_state() -> dict:
    """Load last pipeline state from disk (or return empty dict)."""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_state(state: dict) -> None:
    """Persist pipeline state to disk."""
    LOGS_DIR.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Builder execution
# ---------------------------------------------------------------------------

def run_builder(
    name: str,
    info: dict,
    dry_run: bool,
    logger: logging.Logger,
) -> dict:
    """Run a single builder script. Returns a result dict."""
    script = info["script"]
    script_path = SCRIPTS_DIR / script

    if not script_path.exists():
        logger.warning("  MISSING %s: %s", name, script_path)
        return {"name": name, "status": "missing", "duration": 0}

    if dry_run:
        logger.info("  [DRY RUN] %s -> %s", name, script)
        return {"name": name, "status": "dry_run", "duration": 0}

    logger.info("  Running %s ...", name)
    start = time.time()

    try:
        result = subprocess.run(
            [PYTHON, str(script_path)],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(PROJECT_ROOT),
        )
        duration = round(time.time() - start, 1)

        if result.returncode == 0:
            logger.info("  OK  %s (%ss)", name, duration)
            return {"name": name, "status": "success", "duration": duration}

        stderr_snippet = (result.stderr or "")[:500]
        logger.error("  FAIL %s (%ss): %s", name, duration, stderr_snippet[:200])
        return {
            "name": name,
            "status": "failed",
            "duration": duration,
            "error": stderr_snippet,
        }

    except subprocess.TimeoutExpired:
        duration = round(time.time() - start, 1)
        logger.error("  TIMEOUT %s (%ss)", name, duration)
        return {"name": name, "status": "timeout", "duration": duration}

    except Exception as exc:
        duration = round(time.time() - start, 1)
        logger.error("  ERROR %s: %s", name, exc)
        return {
            "name": name,
            "status": "error",
            "duration": duration,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Main pipeline orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(
    mode: str,
    dry_run: bool = False,
    single_builder: str | None = None,
    resume: bool = False,
) -> dict:
    """Run the pipeline in the specified mode.

    Returns a health-report dict with per-builder results.
    """
    logger = setup_logging()
    logger.info(
        "Pipeline starting: mode=%s, dry_run=%s, resume=%s",
        mode, dry_run, resume,
    )
    start_time = time.time()

    # Determine which builders to run ----------------------------------
    if single_builder:
        if single_builder not in BUILDERS:
            logger.error("Unknown builder: %s", single_builder)
            logger.info("Available: %s", ", ".join(sorted(BUILDERS)))
            return {"error": f"Unknown builder: {single_builder}"}
        builders = {single_builder: BUILDERS[single_builder]}
    else:
        builders = {
            name: info
            for name, info in BUILDERS.items()
            if mode in info["modes"]
        }

    # Resume support: skip already-succeeded builders from last run ----
    completed_names: set[str] = set()
    if resume:
        prev_state = load_state()
        if prev_state.get("mode") == mode:
            completed_names = {
                r["name"]
                for r in prev_state.get("results", [])
                if r["status"] == "success"
            }
            logger.info(
                "Resuming: skipping %d already-succeeded builder(s)",
                len(completed_names),
            )
        else:
            logger.info("Resume requested but no matching prior run found; starting fresh")

    # Group by phase ---------------------------------------------------
    phases: dict[int, list[tuple[str, dict]]] = {}
    for name, info in builders.items():
        phases.setdefault(info["phase"], []).append((name, info))

    results: list[dict] = []

    for phase_num in sorted(phases):
        phase_builders = phases[phase_num]
        logger.info(
            "\n--- Phase %d (%d builder%s) ---",
            phase_num,
            len(phase_builders),
            "s" if len(phase_builders) != 1 else "",
        )

        for name, info in phase_builders:
            if name in completed_names:
                logger.info("  SKIP %s (already succeeded)", name)
                results.append({"name": name, "status": "resumed_skip", "duration": 0})
                continue

            result = run_builder(name, info, dry_run, logger)
            results.append(result)

    # Summary ----------------------------------------------------------
    total_duration = round(time.time() - start_time, 1)
    succeeded = sum(1 for r in results if r["status"] in ("success", "resumed_skip"))
    failed = sum(1 for r in results if r["status"] == "failed")
    timed_out = sum(1 for r in results if r["status"] == "timeout")
    missing = sum(1 for r in results if r["status"] == "missing")
    skipped = sum(1 for r in results if r["status"] in ("dry_run", "resumed_skip"))

    logger.info("\n%s", "=" * 55)
    logger.info(
        "Pipeline complete: %d OK, %d FAILED, %d timeout, %d missing, %d skipped (%ss)",
        succeeded, failed, timed_out, missing, skipped, total_duration,
    )

    if failed or timed_out:
        logger.info("\nProblematic builders:")
        for r in results:
            if r["status"] in ("failed", "timeout"):
                logger.info(
                    "  - %s [%s]: %s",
                    r["name"],
                    r["status"],
                    r.get("error", "")[:100],
                )

    # Health report ----------------------------------------------------
    report = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "dry_run": dry_run,
        "duration_seconds": total_duration,
        "summary": {
            "succeeded": succeeded,
            "failed": failed,
            "timed_out": timed_out,
            "missing": missing,
            "skipped": skipped,
            "total": len(results),
        },
        "results": results,
    }

    # Save for --resume
    save_state(report)

    # Save dashboard-visible health report
    report_path = PROJECT_ROOT / "dashboard" / "data" / "pipeline_report.json"
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Health report -> %s", report_path)
    except Exception as exc:
        logger.warning("Could not save health report: %s", exc)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NBA Pipeline Runner -- unified orchestrator for dashboard builders",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "injuries_odds", "pretip"],
        default="full",
        help="Pipeline mode (default: full)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the execution plan without running any scripts",
    )
    parser.add_argument(
        "--builder",
        type=str,
        help="Run a single builder by name (ignores --mode filtering)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip builders that succeeded in the previous run of the same mode",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_builders",
        help="List all registered builders and exit",
    )
    args = parser.parse_args()

    if args.list_builders:
        log.info(f"{'Builder':<30} {'Phase':>5}  {'Modes'}")
        print("-" * 60)
        for name, info in sorted(BUILDERS.items(), key=lambda x: (x[1]["phase"], x[0])):
            modes_str = ", ".join(info["modes"])
            log.info(f"{name:<30} {info['phase']:>5}  {modes_str}")
        return

    run_pipeline(
        mode=args.mode,
        dry_run=args.dry_run,
        single_builder=args.builder,
        resume=args.resume,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
