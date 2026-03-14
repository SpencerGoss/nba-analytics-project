# Import required libraries
from nba_api.stats.endpoints import leaguegamelog
import pandas as pd
import time
import os

from src.data.api_client import fetch_with_retry, HEADERS
import logging

log = logging.getLogger(__name__)


def get_player_game_logs_playoffs(start_year=2000, end_year=2024):
    os.makedirs("data/raw/player_game_logs_playoffs", exist_ok=True)

    for year in range(start_year, end_year + 1):
        season = f"{year}-{str(year+1)[-2:]}"
        log.info(f"Fetching {season}...")
        time.sleep(1)

        result = fetch_with_retry(
            lambda s=season: leaguegamelog.LeagueGameLog(
                season=s,
                season_type_all_star="Playoffs",
                player_or_team_abbreviation="P",
                headers=HEADERS,
                timeout=60
            ).get_data_frames()[0],
            season
        )

        if not result["success"] or len(result["data"]) == 0:
            log.warning(f"  No playoff data for {season}. Skipping.")
            continue

        data = result["data"]
        output_path = f"data/raw/player_game_logs_playoffs/player_game_logs_playoffs_{season.replace('-', '')}.csv"
        data.to_csv(output_path, index=False)
        log.info(f"  Saved {season} ({len(data)} rows)")


if __name__ == "__main__":
    get_player_game_logs_playoffs()
