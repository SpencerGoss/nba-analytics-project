"""
NBA Analytics Historical Backfill Script

Fetches historical NBA data ranges that are not covered by the daily updater,
then rebuilds processed CSVs.

Run once (or rarely):
    python backfill.py
"""

from datetime import datetime
from pathlib import Path
from typing import Callable

from src.data.get_game_log import get_team_game_logs
from src.data.get_player_game_logs import get_player_game_logs
from src.data.get_player_game_logs_playoffs import get_player_game_logs_playoffs
from src.data.get_player_stats import get_player_stats_all_seasons
from src.data.get_player_stats_advanced import get_player_stats_advanced
from src.data.get_player_stats_playoffs import get_player_stats_playoffs
from src.data.get_standings import get_standings
from src.data.get_team_game_logs_playoffs import get_team_game_logs_playoffs
from src.data.get_team_stats import get_team_stats_all_seasons
from src.data.get_team_stats_advanced import get_team_stats_advanced
from src.data.get_team_stats_playoffs import get_team_stats_playoffs
from src.processing.preprocessing import run_preprocessing
from src.features.team_game_features import build_team_game_features


ERROR_LOG_PATH = Path("logs/pipeline_errors.log")


def _log_pipeline_error(script_name: str, error: Exception) -> None:
    """Append pipeline errors to logs/pipeline_errors.log."""
    ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with ERROR_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(f"[{timestamp}] {script_name} failed: {error}\n")


def run_tasks(
    step_title: str,
    note: str,
    tasks: list[tuple[str, Callable[..., None], int, int]],
) -> None:
    print(f"\n=== {step_title} ===")
    if note:
        print(note)
    for label, func, start_year, end_year in tasks:
        print(f"--- {label} ({start_year}-{end_year}) ---")
        func(start_year=start_year, end_year=end_year)


def main() -> None:
    try:
        _run_backfill()
    except Exception as error:
        _log_pipeline_error("backfill.py", error)
        print(f"Backfill failed: {error}. See logs/pipeline_errors.log for details.")
        raise SystemExit(1) from error


def _run_backfill() -> None:
    start_time = datetime.now()
    print(f"[{start_time.strftime('%Y-%m-%d %H:%M:%S')}] Starting NBA historical backfill...")

    leaguedash_tasks = [
        ("Player stats (basic)", get_player_stats_all_seasons, 1996, 1999),
        ("Player stats (advanced)", get_player_stats_advanced, 1996, 1999),
        ("Player stats playoffs (basic + advanced)", get_player_stats_playoffs, 1996, 1999),
        ("Team stats (basic)", get_team_stats_all_seasons, 1996, 1999),
        ("Team stats (advanced)", get_team_stats_advanced, 1996, 1999),
        ("Team stats playoffs (basic + advanced)", get_team_stats_playoffs, 1996, 1999),
    ]
    run_tasks(
        "Step 1: LeagueDash stats",
        "Earliest reliable range for these endpoints is 1996-97 onward.",
        leaguedash_tasks,
    )

    gamelog_tasks = [
        ("Team game logs (regular season)", get_team_game_logs, 1946, 1999),
        ("Team game logs (playoffs)", get_team_game_logs_playoffs, 1946, 1999),
        ("Player game logs (regular season)", get_player_game_logs, 1946, 1999),
        ("Player game logs (playoffs)", get_player_game_logs_playoffs, 1946, 1999),
    ]
    run_tasks(
        "Step 2: Game logs",
        "These endpoints can go back to 1946-47 (with sparse early coverage).",
        gamelog_tasks,
    )

    standings_tasks = [("Standings", get_standings, 1979, 1999)]
    run_tasks(
        "Step 3: Standings",
        "LeagueStandingsV3 is most reliable from 1979-80 onward.",
        standings_tasks,
    )

    print("\n=== Step 4: Rebuilding processed CSVs (full rebuild) ===")
    run_preprocessing(full_rebuild=True)

    print("\n=== Step 5: Rebuilding derived feature CSVs ===")
    try:
        build_team_game_features()
        try:
            from src.features.player_features import build_player_game_features
            build_player_game_features()
        except Exception as player_feat_err:
            print(f"Player feature rebuild skipped (non-fatal): {player_feat_err}")
    except Exception as feat_err:
        _log_pipeline_error("backfill.py:feature_rebuild", feat_err)
        print(f"Feature rebuild failed (non-fatal): {feat_err}")

    elapsed = datetime.now() - start_time
    total_seconds = int(elapsed.total_seconds())
    print(
        f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        f"Backfill complete in {total_seconds // 60}m {total_seconds % 60}s"
    )


if __name__ == "__main__":
    main()
