# Import required libraries
from nba_api.stats.endpoints import commonallplayers
import pandas as pd
import os

# Headers used in nba_api GitHub examples
HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nba.com/"
}


# Download player master table
def get_player_master():
    os.makedirs("data/raw/player_master", exist_ok=True)

    data = commonallplayers.CommonAllPlayers(
        headers=HEADERS
    ).get_data_frames()[0]

    data.to_csv("data/raw/player_master/player_master.csv", index=False)


# Run script
if __name__ == "__main__":
    get_player_master()
