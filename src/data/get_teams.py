# Import required libraries
from nba_api.stats.static import teams
import pandas as pd
import os


# Fetch the static teams lookup table.
#
# This uses nba_api's local static data — no HTTP call is made, so it
# runs instantly and never hits a rate limit.
#
# Provides: team_id, full_name, abbreviation, nickname, city, state, year_founded
def get_teams():
    os.makedirs("data/raw/teams", exist_ok=True)

    print("Fetching teams static lookup...")
    all_teams = teams.get_teams()

    df = pd.DataFrame(all_teams)

    df.to_csv("data/raw/teams/teams.csv", index=False)
    print(f"  Saved teams ({len(df)} rows)")


if __name__ == "__main__":
    get_teams()
