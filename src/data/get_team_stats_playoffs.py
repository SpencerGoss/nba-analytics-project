# Import required libraries
from nba_api.stats.endpoints import leaguedashteamstats
import pandas as pd
import time
import os


HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nba.com/",
    "Connection": "keep-alive",
    "Origin": "https://www.nba.com"
}

MAX_RETRIES = 3
RETRY_DELAY = 10


def fetch_with_retry(fetch_fn, season):
    for attempt in range(MAX_RETRIES):
        try:
            return fetch_fn()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  Attempt {attempt + 1} failed for {season}: {e}. Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"  All {MAX_RETRIES} retries failed for {season}. Skipping.")
                return None


def get_team_stats_playoffs(start_year=2000, end_year=2024):
    os.makedirs("data/raw/team_stats_playoffs", exist_ok=True)
    os.makedirs("data/raw/team_stats_advanced_playoffs", exist_ok=True)

    for year in range(start_year, end_year + 1):
        season = f"{year}-{str(year+1)[-2:]}"
        print(f"Fetching {season}...")
        time.sleep(1)

        # Base stats
        base = fetch_with_retry(
            lambda s=season: leaguedashteamstats.LeagueDashTeamStats(
                season=s,
                season_type_all_star="Playoffs",
                headers=HEADERS,
                timeout=60
            ).get_data_frames()[0],
            season
        )

        if base is not None and len(base) > 0:
            base.to_csv(f"data/raw/team_stats_playoffs/team_stats_playoffs_{season.replace('-', '')}.csv", index=False)
            print(f"  Saved base {season} ({len(base)} rows)")
        else:
            print(f"  No base playoff data for {season}. Skipping.")

        time.sleep(1)

        # Advanced stats
        advanced = fetch_with_retry(
            lambda s=season: leaguedashteamstats.LeagueDashTeamStats(
                season=s,
                season_type_all_star="Playoffs",
                measure_type_detailed_defense="Advanced",
                headers=HEADERS,
                timeout=60
            ).get_data_frames()[0],
            season
        )

        if advanced is not None and len(advanced) > 0:
            advanced.to_csv(f"data/raw/team_stats_advanced_playoffs/team_stats_advanced_playoffs_{season.replace('-', '')}.csv", index=False)
            print(f"  Saved advanced {season} ({len(advanced)} rows)")
        else:
            print(f"  No advanced playoff data for {season}. Skipping.")


if __name__ == "__main__":
    get_team_stats_playoffs()
