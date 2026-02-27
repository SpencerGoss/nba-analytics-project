# Import required libraries
from nba_api.stats.endpoints import leaguehustlestatsplayer, leaguehustlestatsteam
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


# Download hustle stats for every season
# Available from 2015-16 onward
# Includes: charges drawn, deflections, screen assists, contested shots, loose balls recovered
def get_hustle_stats(start_year=2015, end_year=2024):
    os.makedirs("data/raw/player_hustle_stats", exist_ok=True)
    os.makedirs("data/raw/team_hustle_stats", exist_ok=True)

    for year in range(start_year, end_year + 1):
        season = f"{year}-{str(year+1)[-2:]}"
        print(f"Fetching {season}...")
        time.sleep(1)

        # Player hustle stats
        player_data = fetch_with_retry(
            lambda s=season: leaguehustlestatsplayer.LeagueHustleStatsPlayer(
                season=s,
                season_type_all_star="Regular Season",
                headers=HEADERS,
                timeout=60
            ).get_data_frames()[0],
            season
        )

        if player_data is not None and len(player_data) > 0:
            player_data.to_csv(f"data/raw/player_hustle_stats/player_hustle_stats_{season.replace('-', '')}.csv", index=False)
            print(f"  Saved player hustle {season} ({len(player_data)} rows)")
        else:
            print(f"  No player hustle data for {season}. Skipping.")

        time.sleep(1)

        # Team hustle stats
        team_data = fetch_with_retry(
            lambda s=season: leaguehustlestatsteam.LeagueHustleStatsTeam(
                season=s,
                season_type_all_star="Regular Season",
                headers=HEADERS,
                timeout=60
            ).get_data_frames()[0],
            season
        )

        if team_data is not None and len(team_data) > 0:
            team_data.to_csv(f"data/raw/team_hustle_stats/team_hustle_stats_{season.replace('-', '')}.csv", index=False)
            print(f"  Saved team hustle {season} ({len(team_data)} rows)")
        else:
            print(f"  No team hustle data for {season}. Skipping.")


if __name__ == "__main__":
    get_hustle_stats()
