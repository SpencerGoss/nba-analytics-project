# Import required libraries
from nba_api.stats.endpoints import commonallplayers
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

MAX_RETRIES = 3
RETRY_DELAY = 10


def fetch_with_retry(fetch_fn, label):
    for attempt in range(MAX_RETRIES):
        try:
            return fetch_fn()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  Attempt {attempt + 1} failed for {label}: {e}. Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"  All {MAX_RETRIES} retries failed for {label}. Skipping.")
                return None


# Download player master table
def get_player_master():
    os.makedirs("data/raw/players", exist_ok=True)
    print("Fetching player master...")

    data = fetch_with_retry(
        lambda: commonallplayers.CommonAllPlayers(
            headers=HEADERS
        ).get_data_frames()[0],
        "player_master"
    )

    if data is None:
        return

    data.to_csv("data/raw/players/player_master.csv", index=False)
    print(f"  Saved player_master ({len(data)} rows)")


# Run script
if __name__ == "__main__":
    get_player_master()
