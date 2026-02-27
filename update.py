"""
NBA Analytics Update Script
============================
Fetches the current NBA season's data from the NBA API and refreshes
the processed CSVs and SQLite database.

Run this daily (or manually) to keep the database up to date.

Usage:
    python update.py

Automated setup (Windows Task Scheduler):
    See scripts/run_update.bat for step-by-step Task Scheduler instructions.

What this does:
    1. Determines the current NBA season year automatically
    2. Re-fetches all data types for the current season only (~10-15 API calls)
    3. Rebuilds all processed CSVs from raw data (fast — no API calls)
    4. Reloads the SQLite database from the processed CSVs

Playoff data is only fetched from April onward (when playoffs begin).
Hustle stats are only fetched from 2015-16 onward (when the NBA started tracking them).
"""

import sys
import os
from datetime import datetime

start_time = datetime.now()
print(f"[{start_time.strftime('%Y-%m-%d %H:%M:%S')}] Starting NBA data update...")


# ── Determine current NBA season ───────────────────────────────────────────────
# The NBA season starts in October. Before October = current season started last year.
# Example: February 2026 → season started Oct 2025 → CURRENT_YEAR = 2025 → "2025-26"
def get_current_season_year():
    now = datetime.now()
    return now.year if now.month >= 10 else now.year - 1


CURRENT_YEAR = get_current_season_year()
CURRENT_SEASON = f"{CURRENT_YEAR}-{str(CURRENT_YEAR + 1)[-2:]}"
print(f"Current NBA season: {CURRENT_SEASON}\n")


# ── Step 1: Fetch current season from NBA API ──────────────────────────────────
print(f"=== Step 1: Fetching {CURRENT_SEASON} data from NBA API ===")

from src.data.get_player_stats import get_player_stats_all_seasons
from src.data.get_player_stats_advanced import get_player_stats_advanced
from src.data.get_player_stats_scoring import get_player_stats_scoring
from src.data.get_player_stats_clutch import get_player_stats_clutch
from src.data.get_player_game_logs import get_player_game_logs
from src.data.get_team_stats import get_team_stats_all_seasons
from src.data.get_team_stats_advanced import get_team_stats_advanced
from src.data.get_game_log import get_team_game_logs
from src.data.get_standings import get_standings
from src.data.get_player_master import get_player_master
from src.data.get_teams import get_teams

# Regular season data
get_player_stats_all_seasons(start_year=CURRENT_YEAR, end_year=CURRENT_YEAR)
get_player_stats_advanced(start_year=CURRENT_YEAR, end_year=CURRENT_YEAR)
get_player_stats_scoring(start_year=CURRENT_YEAR, end_year=CURRENT_YEAR)
get_player_stats_clutch(start_year=CURRENT_YEAR, end_year=CURRENT_YEAR)
get_player_game_logs(start_year=CURRENT_YEAR, end_year=CURRENT_YEAR)
get_team_stats_all_seasons(start_year=CURRENT_YEAR, end_year=CURRENT_YEAR)
get_team_stats_advanced(start_year=CURRENT_YEAR, end_year=CURRENT_YEAR)
get_team_game_logs(start_year=CURRENT_YEAR, end_year=CURRENT_YEAR)
get_standings(start_year=CURRENT_YEAR, end_year=CURRENT_YEAR)

# Hustle stats: available from 2015-16 onward
if CURRENT_YEAR >= 2015:
    from src.data.get_hustle_stats import get_hustle_stats
    get_hustle_stats(start_year=CURRENT_YEAR, end_year=CURRENT_YEAR)

# Playoff data: only fetch during/after the playoff window (April–September)
# Avoids empty-data errors when the playoffs haven't started yet
now = datetime.now()
if now.month >= 4:
    from src.data.get_player_stats_playoffs import get_player_stats_playoffs
    from src.data.get_team_stats_playoffs import get_team_stats_playoffs
    from src.data.get_player_game_logs_playoffs import get_player_game_logs_playoffs
    from src.data.get_team_game_logs_playoffs import get_team_game_logs_playoffs
    get_player_stats_playoffs(start_year=CURRENT_YEAR, end_year=CURRENT_YEAR)
    get_team_stats_playoffs(start_year=CURRENT_YEAR, end_year=CURRENT_YEAR)
    get_player_game_logs_playoffs(start_year=CURRENT_YEAR, end_year=CURRENT_YEAR)
    get_team_game_logs_playoffs(start_year=CURRENT_YEAR, end_year=CURRENT_YEAR)
else:
    print(f"Skipping playoff data (month={now.month}, playoffs start in April)")

# Always refresh player master and teams (fast: 1 API call each)
get_player_master()
get_teams()


# ── Step 2: Rebuild all processed CSVs from raw data ──────────────────────────
print("\n=== Step 2: Rebuilding processed CSVs ===")
from src.processing.preprocessing import run_preprocessing
run_preprocessing()


# ── Step 3: Reload the SQLite database ────────────────────────────────────────
print("\n=== Step 3: Reloading database ===")
from src.processing.load_to_sql import load_all_tables
load_all_tables()


# ── Done ───────────────────────────────────────────────────────────────────────
elapsed = datetime.now() - start_time
print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Update complete in {elapsed.seconds // 60}m {elapsed.seconds % 60}s")
