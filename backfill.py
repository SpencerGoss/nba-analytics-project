"""
NBA Analytics Historical Backfill Script
=========================================
Fetches all available historical data not yet in the database (pre-2000 seasons)
and extends certain data types back to the earliest year each endpoint supports.

Run this ONCE. After it completes, use update.py for daily updates.

Historical availability by endpoint:
  LeagueDashPlayerStats / LeagueDashTeamStats   ->  1996-97  (first reliable season)
  LeagueGameLog (player & team, reg + playoffs) ->  1946-47  (first NBA/BAA season)
  LeagueStandingsV3                             ->  1979-80  (3-point era / reliable floor)
  Scoring / Clutch stats (tracking data)        ->  2004-05  (no pre-2004 data exists)
  Hustle stats (tracking data)                  ->  2015-16  (already in DB, skip)

What this script fetches (existing 2000-2024 raw files are not touched):
  player_stats                1996-1999
  player_stats_advanced       1996-1999
  player_stats_playoffs       1996-1999  (base + advanced)
  team_stats                  1996-1999
  team_stats_advanced         1996-1999
  team_stats_playoffs         1996-1999  (base + advanced)
  player_game_logs            1946-1999
  player_game_logs_playoffs   1946-1999
  team_game_logs              1946-1999
  team_game_logs_playoffs     1946-1999
  standings                   1979-1999

After fetching, rebuilds all processed CSVs and reloads the SQLite database.

Estimated run time: 30-60 minutes (rate-limited to ~1 req/sec).
"""

import sys
import os
from datetime import datetime

start_time = datetime.now()
print(f"[{start_time.strftime('%Y-%m-%d %H:%M:%S')}] Starting NBA historical backfill...")
print()


# ── Imports ────────────────────────────────────────────────────────────────────
from src.data.get_player_stats import get_player_stats_all_seasons
from src.data.get_player_stats_advanced import get_player_stats_advanced
from src.data.get_player_stats_playoffs import get_player_stats_playoffs
from src.data.get_player_game_logs import get_player_game_logs
from src.data.get_player_game_logs_playoffs import get_player_game_logs_playoffs
from src.data.get_team_stats import get_team_stats_all_seasons
from src.data.get_team_stats_advanced import get_team_stats_advanced
from src.data.get_team_stats_playoffs import get_team_stats_playoffs
from src.data.get_game_log import get_team_game_logs
from src.data.get_team_game_logs_playoffs import get_team_game_logs_playoffs
from src.data.get_standings import get_standings


# ── Step 1: LeagueDash endpoints (1996-97 is the earliest reliable season) ────
print("=== Step 1: LeagueDash stats (1996–1999) ===")
print("Note: If seasons before 1996-97 return errors they will be skipped automatically.\n")

print("--- Player stats (basic) ---")
get_player_stats_all_seasons(start_year=1996, end_year=1999)

print("\n--- Player stats (advanced) ---")
get_player_stats_advanced(start_year=1996, end_year=1999)

print("\n--- Player stats playoffs (basic + advanced) ---")
get_player_stats_playoffs(start_year=1996, end_year=1999)

print("\n--- Team stats (basic) ---")
get_team_stats_all_seasons(start_year=1996, end_year=1999)

print("\n--- Team stats (advanced) ---")
get_team_stats_advanced(start_year=1996, end_year=1999)

print("\n--- Team stats playoffs (basic + advanced) ---")
get_team_stats_playoffs(start_year=1996, end_year=1999)


# ── Step 2: LeagueGameLog endpoints (goes all the way back to 1946-47) ────────
print("\n=== Step 2: Game logs (1946–1999) ===")
print("Note: Very old seasons (pre-1960s) may be sparse. Gaps are skipped automatically.\n")

print("--- Team game logs (regular season) ---")
get_team_game_logs(start_year=1946, end_year=1999)

print("\n--- Team game logs (playoffs) ---")
get_team_game_logs_playoffs(start_year=1946, end_year=1999)

print("\n--- Player game logs (regular season) ---")
get_player_game_logs(start_year=1946, end_year=1999)

print("\n--- Player game logs (playoffs) ---")
get_player_game_logs_playoffs(start_year=1946, end_year=1999)


# ── Step 3: Standings (reliable from 1979-80 / 3-point era) ───────────────────
print("\n=== Step 3: Standings (1979–1999) ===")
print("Note: LeagueStandingsV3 becomes unreliable before 1979-80.\n")

get_standings(start_year=1979, end_year=1999)


# ── Step 4: Rebuild processed CSVs ────────────────────────────────────────────
print("\n=== Step 4: Rebuilding all processed CSVs ===")
from src.processing.preprocessing import run_preprocessing
run_preprocessing()


# ── Step 5: Reload the SQLite database ────────────────────────────────────────
print("\n=== Step 5: Reloading SQLite database ===")
from src.processing.load_to_sql import load_all_tables
load_all_tables()


# ── Done ───────────────────────────────────────────────────────────────────────
elapsed = datetime.now() - start_time
minutes = elapsed.seconds // 60
seconds = elapsed.seconds % 60
print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Backfill complete in {minutes}m {seconds}s")
print()
print("Database now covers:")
print("  Game logs:    1946-47 onward")
print("  Standings:    1979-80 onward")
print("  Player/team stats (LeagueDash): 1996-97 onward")
print("  Scoring/Clutch stats:           2000-01 onward  (tracking data: actual data starts 2004-05)")
print("  Hustle stats:                   2015-16 onward")
