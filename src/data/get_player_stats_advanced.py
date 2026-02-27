# Import required libraries
from nba_api.stats.endpoints import leaguedashplayerstats
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

MAX_RETRIES = 3
RETRY_DELAY = 10


def fetch_with_retry(fetch_fn, season):
    for attempt in range(MAX_RETRIES):
        try:
            return fetch_fn()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  Attempt {attempt + 1} failed for {season}: {e}. Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"  All {MAX_RETRIES} retries failed for {season}. Skipping.")
                return None


# Download advanced player stats for every season in a range
# Uses the same LeagueDashPlayerStats endpoint as base stats but with
# measure_type_detailed_defense="Advanced" to get metrics like TS%, USG%, ORTG, DRTG, etc.
def get_player_stats_advanced(start_year=2000, end_year=2024):
    os.makedirs("data/raw/player_stats_advanced", exist_ok=True)

    for year in range(start_year, end_year + 1):
        season = f"{year}-{str(year+1)[-2:]}"
        print(f"Fetching {season}...")
        time.sleep(1)

        data = fetch_with_retry(
            lambda s=season: leaguedashplayerstats.LeagueDashPlayerStats(
                season=s,
                measure_type_detailed_defense="Advanced",
                headers=HEADERS,
                timeout=60
            ).get_data_frames()[0],
            season
        )

        if data is None:
            continue

        output_path = f"data/raw/player_stats_advanced/player_stats_advanced_{season.replace('-', '')}.csv"
        data.to_csv(output_path, index=False)
        print(f"  Saved {season} ({len(data)} rows)")


# Run script
if __name__ == "__main__":
    get_player_stats_advanced()
