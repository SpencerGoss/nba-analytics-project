# Import required libraries
from nba_api.stats.endpoints import leaguedashplayerstats
import pandas as pd
import time
import os

from src.data.api_client import fetch_with_retry, HEADERS


def get_player_stats_playoffs(start_year=2000, end_year=2024):
    os.makedirs("data/raw/player_stats_playoffs", exist_ok=True)
    os.makedirs("data/raw/player_stats_advanced_playoffs", exist_ok=True)

    for year in range(start_year, end_year + 1):
        season = f"{year}-{str(year+1)[-2:]}"
        print(f"Fetching {season}...")
        time.sleep(1)

        # Base stats
        base_result = fetch_with_retry(
            lambda s=season: leaguedashplayerstats.LeagueDashPlayerStats(
                season=s,
                season_type_all_star="Playoffs",
                headers=HEADERS,
                timeout=60
            ).get_data_frames()[0],
            season
        )

        if base_result["success"] and len(base_result["data"]) > 0:
            base_result["data"].to_csv(f"data/raw/player_stats_playoffs/player_stats_playoffs_{season.replace('-', '')}.csv", index=False)
            print(f"  Saved base {season} ({len(base_result['data'])} rows)")
        else:
            print(f"  No base playoff data for {season}. Skipping.")

        time.sleep(1)

        # Advanced stats
        adv_result = fetch_with_retry(
            lambda s=season: leaguedashplayerstats.LeagueDashPlayerStats(
                season=s,
                season_type_all_star="Playoffs",
                measure_type_detailed_defense="Advanced",
                headers=HEADERS,
                timeout=60
            ).get_data_frames()[0],
            season
        )

        if adv_result["success"] and len(adv_result["data"]) > 0:
            adv_result["data"].to_csv(f"data/raw/player_stats_advanced_playoffs/player_stats_advanced_playoffs_{season.replace('-', '')}.csv", index=False)
            print(f"  Saved advanced {season} ({len(adv_result['data'])} rows)")
        else:
            print(f"  No advanced playoff data for {season}. Skipping.")


if __name__ == "__main__":
    get_player_stats_playoffs()
