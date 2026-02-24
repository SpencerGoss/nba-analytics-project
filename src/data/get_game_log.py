# Import required libraries
from nba_api.stats.endpoints import leaguegamelog
import pandas as pd
import time
import os

# Headers used in nba_api GitHub examples
HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nba.com/",
    "Connection": "keep-alive",
    "Origin": "https://www.nba.com"
}

# Download team game logs for each season
def get_team_game_logs(start_year=2000, end_year=2024):
    os.makedirs("data/raw/team_game_logs", exist_ok=True)

    for year in range(start_year, end_year + 1):
        season = f"{year}-{str(year+1)[-2:]}"
        time.sleep(1)

        data = leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star="Regular Season",
            headers=HEADERS,
            timeout=60
        ).get_data_frames()[0]

        output_path = f"data/raw/team_game_logs/team_game_logs_{season.replace('-', '')}.csv"
        data.to_csv(output_path, index=False)

# Run script
if __name__ == "__main__":
    get_team_game_logs()



