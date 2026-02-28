"""
NBA Analytics Update Script

Refreshes the current NBA season from NBA API endpoints, rebuilds processed CSVs,
and reloads the SQLite database.

Run daily (or on demand):
    python update.py
"""

from datetime import datetime
from typing import Callable

from src.data.get_game_log import get_team_game_logs
from src.data.get_player_game_logs import get_player_game_logs
from src.data.get_player_master import get_player_master
from src.data.get_player_stats import get_player_stats_all_seasons
from src.data.get_player_stats_advanced import get_player_stats_advanced
from src.data.get_player_stats_clutch import get_player_stats_clutch
from src.data.get_player_stats_scoring import get_player_stats_scoring
from src.data.get_standings import get_standings
from src.data.get_team_stats import get_team_stats_all_seasons
from src.data.get_team_stats_advanced import get_team_stats_advanced
from src.data.get_teams import get_teams
from src.processing.load_to_sql import load_all_tables
from src.processing.preprocessing import run_preprocessing


def get_current_season_year(reference: datetime | None = None) -> int:
    """Return NBA season start year (season flips in October)."""
    now = reference or datetime.now()
    return now.year if now.month >= 10 else now.year - 1


def run_fetchers(
    title: str,
    fetchers: list[tuple[str, Callable[..., None]]],
    start_year: int,
    end_year: int,
) -> None:
    print(f"\n=== {title} ===")
    for label, fetcher in fetchers:
        print(f"--- {label} ---")
        fetcher(start_year=start_year, end_year=end_year)


def fetch_current_season_data(current_year: int, now: datetime) -> None:
    regular_fetchers = [
        ("Player stats (basic)", get_player_stats_all_seasons),
        ("Player stats (advanced)", get_player_stats_advanced),
        ("Player stats (scoring)", get_player_stats_scoring),
        ("Player stats (clutch)", get_player_stats_clutch),
        ("Player game logs", get_player_game_logs),
        ("Team stats (basic)", get_team_stats_all_seasons),
        ("Team stats (advanced)", get_team_stats_advanced),
        ("Team game logs", get_team_game_logs),
        ("Standings", get_standings),
    ]
    run_fetchers("Step 1A: Current-season regular data", regular_fetchers, current_year, current_year)

    if current_year >= 2015:
        from src.data.get_hustle_stats import get_hustle_stats

        run_fetchers(
            "Step 1B: Hustle stats",
            [("Hustle stats", get_hustle_stats)],
            current_year,
            current_year,
        )
    else:
        print("\n=== Step 1B: Hustle stats ===")
        print("Skipping hustle stats (NBA tracking starts in 2015-16).")

    if now.month >= 4:
        from src.data.get_player_game_logs_playoffs import get_player_game_logs_playoffs
        from src.data.get_player_stats_playoffs import get_player_stats_playoffs
        from src.data.get_team_game_logs_playoffs import get_team_game_logs_playoffs
        from src.data.get_team_stats_playoffs import get_team_stats_playoffs

        playoff_fetchers = [
            ("Player playoffs stats", get_player_stats_playoffs),
            ("Team playoffs stats", get_team_stats_playoffs),
            ("Player playoffs game logs", get_player_game_logs_playoffs),
            ("Team playoffs game logs", get_team_game_logs_playoffs),
        ]
        run_fetchers("Step 1C: Playoff data", playoff_fetchers, current_year, current_year)
    else:
        print("\n=== Step 1C: Playoff data ===")
        print(f"Skipping playoff pulls (current month={now.month}; playoffs start in April).")

    print("\n=== Step 1D: Reference tables ===")
    print("--- Player master ---")
    get_player_master()
    print("--- Teams ---")
    get_teams()


def main() -> None:
    start_time = datetime.now()
    now = datetime.now()
    current_year = get_current_season_year(now)
    current_season = f"{current_year}-{str(current_year + 1)[-2:]}"

    print(f"[{start_time.strftime('%Y-%m-%d %H:%M:%S')}] Starting NBA data update...")
    print(f"Current NBA season: {current_season}")

    fetch_current_season_data(current_year, now)

    print("\n=== Step 2: Rebuilding processed CSVs ===")
    run_preprocessing()

    print("\n=== Step 3: Reloading database ===")
    load_all_tables()

    elapsed = datetime.now() - start_time
    total_seconds = int(elapsed.total_seconds())
    print(
        f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        f"Update complete in {total_seconds // 60}m {total_seconds % 60}s"
    )


if __name__ == "__main__":
    main()
