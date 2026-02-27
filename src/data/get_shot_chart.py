# Import required libraries
from nba_api.stats.endpoints import shotchartdetail
import pandas as pd
import glob
import time
import os


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
RETRY_DELAY = 15


def fetch_with_retry(fetch_fn, label):
    for attempt in range(MAX_RETRIES):
        try:
            return fetch_fn()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"    Attempt {attempt + 1} failed for {label}: {e}. Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"    All {MAX_RETRIES} retries failed for {label}. Skipping.")
                return None


# Download shot chart data for every player in every season
#
# WARNING: This script makes one API call per player per season.
# With ~450 players across 25 seasons that is ~11,000+ API calls.
# Estimated run time: 3-4 hours. Run overnight or as a background task.
#
# Shot data includes court coordinates (LOC_X, LOC_Y) which are ideal
# for heat maps and shot chart visualizations on a website.
def get_shot_chart(start_year=2000, end_year=2024):
    os.makedirs("data/raw/shot_chart", exist_ok=True)

    for year in range(start_year, end_year + 1):
        season = f"{year}-{str(year+1)[-2:]}"
        season_code = f"{year}{str(year+1)[-2:]}"

        # Get player IDs for this season from the raw player_stats file
        raw_path = f"data/raw/player_stats/player_stats_{season_code}.csv"
        if not os.path.exists(raw_path):
            print(f"No player stats file for {season}. Run get_player_stats.py first. Skipping.")
            continue

        players_df = pd.read_csv(raw_path)
        player_ids = players_df["PLAYER_ID"].unique()

        print(f"Fetching shot charts for {season} ({len(player_ids)} players)...")
        season_shots = []

        for i, player_id in enumerate(player_ids):
            time.sleep(0.8)

            data = fetch_with_retry(
                lambda p=player_id, s=season: shotchartdetail.ShotChartDetail(
                    player_id=p,
                    team_id=0,
                    season_nullable=s,
                    season_type_all_star="Regular Season",
                    context_measure_simple="FGA",
                    headers=HEADERS,
                    timeout=60
                ).get_data_frames()[0],
                f"{season} player {player_id}"
            )

            if data is not None and len(data) > 0:
                season_shots.append(data)

            # Print progress every 50 players
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(player_ids)} players processed...")

        if season_shots:
            combined = pd.concat(season_shots, ignore_index=True)
            output_path = f"data/raw/shot_chart/shot_chart_{season_code}.csv"
            combined.to_csv(output_path, index=False)
            print(f"  Saved {season} ({len(combined)} shots)")
        else:
            print(f"  No shots found for {season}.")


if __name__ == "__main__":
    get_shot_chart()
