# Import required libraries
from nba_api.stats.endpoints import leaguedashplayerbiostats
import pandas as pd
import time
import os

from src.data.api_client import fetch_with_retry, HEADERS


# Download player biographical stats for every season in a range.
#
# Provides height, weight, position, draft year, draft round, and draft pick
# for every player who appeared in each season. One row per player per season.
#
# This data is missing from the base player_stats and players tables, making
# it useful for filtering by position or comparing draft class performance.
def get_player_bio_stats(start_year=2000, end_year=2024):
    os.makedirs("data/raw/player_bio_stats", exist_ok=True)

    for year in range(start_year, end_year + 1):
        season = f"{year}-{str(year+1)[-2:]}"
        print(f"Fetching {season}...")
        time.sleep(1)

        result = fetch_with_retry(
            lambda s=season: leaguedashplayerbiostats.LeagueDashPlayerBioStats(
                season=s,
                headers=HEADERS,
                timeout=60
            ).get_data_frames()[0],
            season
        )

        if not result["success"]:
            continue

        data = result["data"]
        output_path = f"data/raw/player_bio_stats/player_bio_stats_{season.replace('-', '')}.csv"
        data.to_csv(output_path, index=False)
        print(f"  Saved {season} ({len(data)} rows)")


if __name__ == "__main__":
    get_player_bio_stats()
