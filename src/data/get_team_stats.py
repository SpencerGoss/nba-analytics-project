# Import required libraries
from nba_api.stats.endpoints import leaguedashteamstats
import pandas as pd
import time
import os

from src.data.api_client import fetch_with_retry, HEADERS


# Download team stats for all seasons
def get_team_stats_all_seasons(start_year=2000, end_year=2024):
    os.makedirs("data/raw/team_stats", exist_ok=True)

    for year in range(start_year, end_year + 1):
        season = f"{year}-{str(year+1)[-2:]}"
        print(f"Fetching {season}...")
        time.sleep(1)

        result = fetch_with_retry(
            lambda s=season: leaguedashteamstats.LeagueDashTeamStats(
                season=s,
                headers=HEADERS,
                timeout=60
            ).get_data_frames()[0],
            season
        )

        if not result["success"]:
            continue

        data = result["data"]
        output_path = f"data/raw/team_stats/team_stats_{season.replace('-', '')}.csv"
        data.to_csv(output_path, index=False)
        print(f"  Saved {season} ({len(data)} rows)")


# Run script
if __name__ == "__main__":
    get_team_stats_all_seasons()
