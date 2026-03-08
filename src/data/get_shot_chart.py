# Import required libraries
from nba_api.stats.endpoints import shotchartdetail
import pandas as pd
import glob
import time
import os

from src.data.api_client import fetch_with_retry, HEADERS


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

        output_path = f"data/raw/shot_chart/shot_chart_{season_code}.csv"
        if os.path.exists(output_path):
            print(f"  SKIP: {season} already fetched -> {output_path}")
            continue

        players_df = pd.read_csv(raw_path)
        player_ids = players_df["PLAYER_ID"].unique()

        print(f"Fetching shot charts for {season} ({len(player_ids)} players)...")
        season_shots = []

        for i, player_id in enumerate(player_ids):
            time.sleep(1.0)

            result = fetch_with_retry(
                lambda p=player_id, s=season: shotchartdetail.ShotChartDetail(
                    player_id=p,
                    team_id=0,
                    season_nullable=s,
                    season_type_all_star="Regular Season",
                    context_measure_simple="FGA",
                    headers=HEADERS,
                    timeout=60
                ).get_data_frames()[0],
                f"{season} player {player_id}",
                retry_delay=15
            )

            if result["success"] and len(result["data"]) > 0:
                season_shots.append(result["data"])

            # Print progress every 50 players
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(player_ids)} players processed...")

        if season_shots:
            combined = pd.concat(season_shots, ignore_index=True)
            combined.to_csv(output_path, index=False)
            print(f"  Saved {season} ({len(combined)} shots)")
        else:
            print(f"  No shots found for {season}.")


if __name__ == "__main__":
    get_shot_chart()
