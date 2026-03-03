# Import required libraries
from nba_api.stats.endpoints import commonallplayers
import pandas as pd
import time
import os

from src.data.api_client import fetch_with_retry, HEADERS


# Download player master table
def get_player_master():
    os.makedirs("data/raw/players", exist_ok=True)
    print("Fetching player master...")

    result = fetch_with_retry(
        lambda: commonallplayers.CommonAllPlayers(
            headers=HEADERS
        ).get_data_frames()[0],
        "player_master"
    )

    if not result["success"]:
        return

    data = result["data"]
    data.to_csv("data/raw/players/player_master.csv", index=False)
    print(f"  Saved player_master ({len(data)} rows)")


# Run script
if __name__ == "__main__":
    get_player_master()
