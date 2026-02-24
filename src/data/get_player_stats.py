# Import required libraries
from nba_api.stats.endpoints import leaguedashplayerstats
import pandas as pd
import time
import os


# Headers used in nba_api gethub. These headers help the NBA API accept the request
HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nba.com/",
    "Connection": "keep-alive",
    "Origin": "https://www.nba.com"
}


# Function to pull player stats for every season in a range
def get_player_stats_all_seasons(start_year=2000, end_year=2024):
    os.makedirs("data/raw/player_stats", exist_ok=True)

    for year in range(start_year, end_year + 1):
        season = f"{year}-{str(year+1)[-2:]}"
        time.sleep(1)

        data = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            headers=HEADERS,
            timeout=60
        ).get_data_frames()[0]

        output_path = f"data/raw/player_stats/player_stats_{season.replace('-', '')}.csv"
        data.to_csv(output_path, index=False)


# Run the script
if __name__ == "__main__":
    get_player_stats_all_seasons()



