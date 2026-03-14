# Import required libraries
from nba_api.stats.endpoints import leaguedashplayerclutch
import pandas as pd
import time
import os

from src.data.api_client import fetch_with_retry, HEADERS
import logging

log = logging.getLogger(__name__)


# Download clutch player stats for every season
# Clutch = last 5 minutes of games decided by 5 points or fewer
def get_player_stats_clutch(start_year=2000, end_year=2024):
    os.makedirs("data/raw/player_stats_clutch", exist_ok=True)

    for year in range(start_year, end_year + 1):
        season = f"{year}-{str(year+1)[-2:]}"
        log.info(f"Fetching {season}...")
        time.sleep(1)

        result = fetch_with_retry(
            lambda s=season: leaguedashplayerclutch.LeagueDashPlayerClutch(
                season=s,
                clutch_time="Last 5 Minutes",
                ahead_behind="Ahead or Behind",
                point_diff=5,
                headers=HEADERS,
                timeout=60
            ).get_data_frames()[0],
            season
        )

        if not result["success"]:
            continue

        data = result["data"]
        output_path = f"data/raw/player_stats_clutch/player_stats_clutch_{season.replace('-', '')}.csv"
        data.to_csv(output_path, index=False)
        log.info(f"  Saved {season} ({len(data)} rows)")


if __name__ == "__main__":
    get_player_stats_clutch()
