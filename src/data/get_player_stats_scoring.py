# Import required libraries
from nba_api.stats.endpoints import leaguedashplayerstats
import pandas as pd
import time
import os

from src.data.api_client import fetch_with_retry, HEADERS
import logging

log = logging.getLogger(__name__)


# Download scoring breakdown stats for every season
# Breaks down how players score: pull-up jumpers, catch-and-shoot, driving layups, etc.
def get_player_stats_scoring(start_year=2000, end_year=2024):
    os.makedirs("data/raw/player_stats_scoring", exist_ok=True)

    for year in range(start_year, end_year + 1):
        season = f"{year}-{str(year+1)[-2:]}"
        log.info(f"Fetching {season}...")
        time.sleep(1)

        result = fetch_with_retry(
            lambda s=season: leaguedashplayerstats.LeagueDashPlayerStats(
                season=s,
                measure_type_detailed_defense="Scoring",
                headers=HEADERS,
                timeout=60
            ).get_data_frames()[0],
            season
        )

        if not result["success"]:
            continue

        data = result["data"]
        output_path = f"data/raw/player_stats_scoring/player_stats_scoring_{season.replace('-', '')}.csv"
        data.to_csv(output_path, index=False)
        log.info(f"  Saved {season} ({len(data)} rows)")


if __name__ == "__main__":
    get_player_stats_scoring()
