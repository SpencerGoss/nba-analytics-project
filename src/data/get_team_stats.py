# Import required libraries
from nba_api.stats.endpoints import leaguedashteamstats
import pandas as pd
import time
import os


# Headers used in nba_api GitHub examples

HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nba.com/"
}

# Download team stats for all seasons
def get_team_stats_all_seasons(start_year=2000, end_year=2024):
    os.makedirs("data/raw/team_stats", exist_ok=True)

    for year in range(start_year, end_year + 1):
        season = f"{year}-{str(year+1)[-2:]}"
        time.sleep(1)

        data = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            headers=HEADERS,
            timeout=60
        ).get_data_frames()[0]

        output_path = f"data/raw/team_stats/team_stats_{season.replace('-', '')}.csv"
        data.to_csv(output_path, index=False)

# Run script
if __name__ == "__main__":
    get_team_stats_all_seasons()
