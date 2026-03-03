# Import required libraries
from nba_api.stats.endpoints import leaguehustlestatsplayer, leaguehustlestatsteam
import pandas as pd
import time
import os

from src.data.api_client import fetch_with_retry, HEADERS


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
        player_result = fetch_with_retry(
            lambda s=season: leaguehustlestatsplayer.LeagueHustleStatsPlayer(
                season=s,
                season_type_all_star="Regular Season",
                headers=HEADERS,
                timeout=60
            ).get_data_frames()[0],
            season
        )

        if player_result["success"] and len(player_result["data"]) > 0:
            player_result["data"].to_csv(f"data/raw/player_hustle_stats/player_hustle_stats_{season.replace('-', '')}.csv", index=False)
            print(f"  Saved player hustle {season} ({len(player_result['data'])} rows)")
        else:
            print(f"  No player hustle data for {season}. Skipping.")

        time.sleep(1)

        # Team hustle stats
        team_result = fetch_with_retry(
            lambda s=season: leaguehustlestatsteam.LeagueHustleStatsTeam(
                season=s,
                season_type_all_star="Regular Season",
                headers=HEADERS,
                timeout=60
            ).get_data_frames()[0],
            season
        )

        if team_result["success"] and len(team_result["data"]) > 0:
            team_result["data"].to_csv(f"data/raw/team_hustle_stats/team_hustle_stats_{season.replace('-', '')}.csv", index=False)
            print(f"  Saved team hustle {season} ({len(team_result['data'])} rows)")
        else:
            print(f"  No team hustle data for {season}. Skipping.")


if __name__ == "__main__":
    get_hustle_stats()
