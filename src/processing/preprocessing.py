"""
Preprocessing module for NBA Analytics Project.

Converts all raw seasonal CSV files from data/raw/ into clean, consolidated
processed CSVs in data/processed/.

Mirrors the logic from preprocessing_data.ipynb but runs as a plain Python
script, which enables automated scheduling via update.py.

Usage:
    python src/processing/preprocessing.py
"""

import pandas as pd
import glob
import os


# ── Shared helpers ─────────────────────────────────────────────────────────────

def clean_columns(df):
    """Standardize all column names to lowercase_underscore format."""
    df.columns = (
        df.columns
            .str.lower()
            .str.strip()
            .str.replace(" ", "_", regex=False)
            .str.replace("-", "_", regex=False)
            .str.replace("/", "_", regex=False)
    )
    return df


def load_season_folder(path, prefix):
    """
    Load all seasonal CSV files from a raw data folder.

    Extracts the season code from each filename (e.g., 'player_stats_202425.csv'
    → season='202425') and adds it as a column so rows can be identified by season
    after all files are concatenated.
    """
    files = sorted(glob.glob(path))
    if not files:
        raise FileNotFoundError(
            f"No files found at {path}. Run the corresponding ingestion script first."
        )
    rows = []
    for file in files:
        season = os.path.basename(file).replace(prefix, "").replace(".csv", "")
        temp_df = pd.read_csv(file)
        temp_df["season"] = season
        rows.append(temp_df)
    return pd.concat(rows, ignore_index=True)


# ── Main preprocessing function ────────────────────────────────────────────────

def run_preprocessing():
    """
    Rebuild all processed CSVs from raw data files.

    Reads all raw seasonal files, applies consistent cleaning, and saves
    consolidated CSVs to data/processed/. Safe to re-run at any time.
    """
    os.makedirs("data/processed", exist_ok=True)

    # ── players ──────────────────────────────────────────────────────────────
    df = pd.read_csv("data/raw/players/player_master.csv")
    df = clean_columns(df)
    df = df.rename(columns={
        "person_id":                  "player_id",
        "display_first_last":         "player_name",
        "display_last_comma_first":   "player_name_last_first",
        "from_year":                  "from_season",
        "to_year":                    "to_season",
    })
    df["player_id"]   = df["player_id"].astype(int)
    df["from_season"] = df["from_season"].astype(int)
    df["to_season"]   = df["to_season"].astype(int)
    df = df.drop_duplicates(subset=["player_id"])
    df.to_csv("data/processed/players.csv", index=False)
    print(f"players: {len(df):,} rows")

    # ── player_stats ─────────────────────────────────────────────────────────
    df = load_season_folder("data/raw/player_stats/*.csv", "player_stats_")
    df = clean_columns(df)
    df["player_id"] = df["player_id"].astype(int)
    df["team_id"]   = df["team_id"].astype(int)
    df["age"]       = df["age"].astype(float)
    df["gp"]        = df["gp"].astype(int)
    df["w"]         = df["w"].astype(int)
    df["l"]         = df["l"].astype(int)
    df = df.drop_duplicates()
    df.to_csv("data/processed/player_stats.csv", index=False)
    print(f"player_stats: {len(df):,} rows")

    # ── player_stats_advanced ─────────────────────────────────────────────────
    df = load_season_folder("data/raw/player_stats_advanced/*.csv", "player_stats_advanced_")
    df = clean_columns(df)
    df["player_id"] = df["player_id"].astype(int)
    df["team_id"]   = df["team_id"].astype(int)
    df["age"]       = df["age"].astype(float)
    df["gp"]        = df["gp"].astype(int)
    df = df.drop_duplicates()
    df.to_csv("data/processed/player_stats_advanced.csv", index=False)
    print(f"player_stats_advanced: {len(df):,} rows")

    # ── player_stats_clutch ───────────────────────────────────────────────────
    df = load_season_folder("data/raw/player_stats_clutch/*.csv", "player_stats_clutch_")
    df = clean_columns(df)
    df["player_id"] = df["player_id"].astype(int)
    df["team_id"]   = df["team_id"].astype(int)
    df["age"]       = df["age"].astype(float)
    df = df.drop_duplicates()
    df.to_csv("data/processed/player_stats_clutch.csv", index=False)
    print(f"player_stats_clutch: {len(df):,} rows")

    # ── player_stats_scoring ──────────────────────────────────────────────────
    df = load_season_folder("data/raw/player_stats_scoring/*.csv", "player_stats_scoring_")
    df = clean_columns(df)
    df["player_id"] = df["player_id"].astype(int)
    df["team_id"]   = df["team_id"].astype(int)
    df["age"]       = df["age"].astype(float)
    df = df.drop_duplicates()
    df.to_csv("data/processed/player_stats_scoring.csv", index=False)
    print(f"player_stats_scoring: {len(df):,} rows")

    # ── player_game_logs ──────────────────────────────────────────────────────
    df = load_season_folder("data/raw/player_game_logs/*.csv", "player_game_logs_")
    df = clean_columns(df)
    df["player_id"] = df["player_id"].astype(int)
    df["team_id"]   = df["team_id"].astype(int)
    df["game_id"]   = df["game_id"].astype(int)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.drop_duplicates()
    df.to_csv("data/processed/player_game_logs.csv", index=False)
    print(f"player_game_logs: {len(df):,} rows")

    # ── team_stats ────────────────────────────────────────────────────────────
    df = load_season_folder("data/raw/team_stats/*.csv", "team_stats_")
    df = clean_columns(df)
    df["team_id"] = df["team_id"].astype(int)
    df["gp"]      = df["gp"].astype(int)
    df["w"]       = df["w"].astype(int)
    df["l"]       = df["l"].astype(int)
    df = df.drop_duplicates()
    df.to_csv("data/processed/team_stats.csv", index=False)
    print(f"team_stats: {len(df):,} rows")

    # ── team_stats_advanced ───────────────────────────────────────────────────
    df = load_season_folder("data/raw/team_stats_advanced/*.csv", "team_stats_advanced_")
    df = clean_columns(df)
    df["team_id"] = df["team_id"].astype(int)
    df["gp"]      = df["gp"].astype(int)
    df["w"]       = df["w"].astype(int)
    df["l"]       = df["l"].astype(int)
    df = df.drop_duplicates()
    df.to_csv("data/processed/team_stats_advanced.csv", index=False)
    print(f"team_stats_advanced: {len(df):,} rows")

    # ── team_game_logs ────────────────────────────────────────────────────────
    df = load_season_folder("data/raw/team_game_logs/*.csv", "team_game_logs_")
    df = clean_columns(df)
    df["team_id"]   = df["team_id"].astype(int)
    df["game_id"]   = df["game_id"].astype(int)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.drop_duplicates()
    df.to_csv("data/processed/team_game_logs.csv", index=False)
    print(f"team_game_logs: {len(df):,} rows")

    # ── standings ─────────────────────────────────────────────────────────────
    df = load_season_folder("data/raw/standings/*.csv", "standings_")
    df = clean_columns(df)
    df = df.rename(columns={
        "teamid":   "team_id",
        "teamcity": "team_city",
        "teamname": "team_name",
        "teamslug": "team_slug",
        "wins":     "w",
        "losses":   "l",
        "winpct":   "w_pct",
    })
    df["team_id"] = df["team_id"].astype(int)
    df["w"]       = df["w"].astype(int)
    df["l"]       = df["l"].astype(int)
    df = df.drop_duplicates()
    df.to_csv("data/processed/standings.csv", index=False)
    print(f"standings: {len(df):,} rows")

    # ── player_game_logs_playoffs ─────────────────────────────────────────────
    df = load_season_folder(
        "data/raw/player_game_logs_playoffs/*.csv", "player_game_logs_playoffs_"
    )
    df = clean_columns(df)
    df["player_id"] = df["player_id"].astype(int)
    df["team_id"]   = df["team_id"].astype(int)
    df["game_id"]   = df["game_id"].astype(int)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.drop_duplicates()
    df.to_csv("data/processed/player_game_logs_playoffs.csv", index=False)
    print(f"player_game_logs_playoffs: {len(df):,} rows")

    # ── team_game_logs_playoffs ───────────────────────────────────────────────
    df = load_season_folder(
        "data/raw/team_game_logs_playoffs/*.csv", "team_game_logs_playoffs_"
    )
    df = clean_columns(df)
    df["team_id"]   = df["team_id"].astype(int)
    df["game_id"]   = df["game_id"].astype(int)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.drop_duplicates()
    df.to_csv("data/processed/team_game_logs_playoffs.csv", index=False)
    print(f"team_game_logs_playoffs: {len(df):,} rows")

    # ── player_stats_playoffs ─────────────────────────────────────────────────
    df = load_season_folder(
        "data/raw/player_stats_playoffs/*.csv", "player_stats_playoffs_"
    )
    df = clean_columns(df)
    df["player_id"] = df["player_id"].astype(int)
    df["team_id"]   = df["team_id"].astype(int)
    df["age"]       = df["age"].astype(float)
    df = df.drop_duplicates()
    df.to_csv("data/processed/player_stats_playoffs.csv", index=False)
    print(f"player_stats_playoffs: {len(df):,} rows")

    # ── player_stats_advanced_playoffs ────────────────────────────────────────
    df = load_season_folder(
        "data/raw/player_stats_advanced_playoffs/*.csv",
        "player_stats_advanced_playoffs_",
    )
    df = clean_columns(df)
    df["player_id"] = df["player_id"].astype(int)
    df["team_id"]   = df["team_id"].astype(int)
    df["age"]       = df["age"].astype(float)
    df = df.drop_duplicates()
    df.to_csv("data/processed/player_stats_advanced_playoffs.csv", index=False)
    print(f"player_stats_advanced_playoffs: {len(df):,} rows")

    # ── team_stats_playoffs ───────────────────────────────────────────────────
    df = load_season_folder(
        "data/raw/team_stats_playoffs/*.csv", "team_stats_playoffs_"
    )
    df = clean_columns(df)
    df["team_id"] = df["team_id"].astype(int)
    df["gp"]      = df["gp"].astype(int)
    df["w"]       = df["w"].astype(int)
    df["l"]       = df["l"].astype(int)
    df = df.drop_duplicates()
    df.to_csv("data/processed/team_stats_playoffs.csv", index=False)
    print(f"team_stats_playoffs: {len(df):,} rows")

    # ── team_stats_advanced_playoffs ──────────────────────────────────────────
    df = load_season_folder(
        "data/raw/team_stats_advanced_playoffs/*.csv",
        "team_stats_advanced_playoffs_",
    )
    df = clean_columns(df)
    df["team_id"] = df["team_id"].astype(int)
    df["gp"]      = df["gp"].astype(int)
    df["w"]       = df["w"].astype(int)
    df["l"]       = df["l"].astype(int)
    df = df.drop_duplicates()
    df.to_csv("data/processed/team_stats_advanced_playoffs.csv", index=False)
    print(f"team_stats_advanced_playoffs: {len(df):,} rows")

    # ── player_hustle_stats (2015-16 onward) ──────────────────────────────────
    df = load_season_folder(
        "data/raw/player_hustle_stats/*.csv", "player_hustle_stats_"
    )
    df = clean_columns(df)
    df["player_id"] = df["player_id"].astype(int)
    df["team_id"]   = df["team_id"].astype(int)
    df = df.drop_duplicates()
    df.to_csv("data/processed/player_hustle_stats.csv", index=False)
    print(f"player_hustle_stats: {len(df):,} rows")

    # ── team_hustle_stats (2015-16 onward) ────────────────────────────────────
    df = load_season_folder(
        "data/raw/team_hustle_stats/*.csv", "team_hustle_stats_"
    )
    df = clean_columns(df)
    df["team_id"] = df["team_id"].astype(int)
    df = df.drop_duplicates()
    df.to_csv("data/processed/team_hustle_stats.csv", index=False)
    print(f"team_hustle_stats: {len(df):,} rows")

    # ── player_bio_stats (optional — skip if not yet downloaded) ──────────────
    bio_files = glob.glob("data/raw/player_bio_stats/*.csv")
    if bio_files:
        df = load_season_folder(
            "data/raw/player_bio_stats/*.csv", "player_bio_stats_"
        )
        df = clean_columns(df)
        df["player_id"] = df["player_id"].astype(int)
        df["team_id"]   = df["team_id"].astype(int)
        df = df.drop_duplicates()
        df.to_csv("data/processed/player_bio_stats.csv", index=False)
        print(f"player_bio_stats: {len(df):,} rows")
    else:
        print("player_bio_stats: skipped (run src/data/get_player_bio_stats.py to download)")

    # ── teams (optional — skip if not yet downloaded) ─────────────────────────
    teams_path = "data/raw/teams/teams.csv"
    if os.path.exists(teams_path):
        df = pd.read_csv(teams_path)
        df = clean_columns(df)
        df = df.rename(columns={"id": "team_id"})
        df["team_id"] = df["team_id"].astype(int)
        df = df.drop_duplicates(subset=["team_id"])
        df.to_csv("data/processed/teams.csv", index=False)
        print(f"teams: {len(df):,} rows")
    else:
        print("teams: skipped (run src/data/get_teams.py to download)")

    # ── shot_chart (optional — skip if not yet downloaded) ────────────────────
    shot_files = glob.glob("data/raw/shot_chart/*.csv")
    if shot_files:
        df = load_season_folder("data/raw/shot_chart/*.csv", "shot_chart_")
        df = clean_columns(df)
        df["player_id"] = df["player_id"].astype(int)
        df["team_id"]   = df["team_id"].astype(int)
        df["game_id"]   = df["game_id"].astype(str)
        df["game_date"] = pd.to_datetime(df["game_date"])
        df = df.drop_duplicates()
        df.to_csv("data/processed/shot_chart.csv", index=False)
        print(f"shot_chart: {len(df):,} rows")
    else:
        print("shot_chart: skipped (run src/data/get_shot_chart.py to download — takes 3-4 hours)")

    print("\nAll processed CSVs rebuilt successfully.")


if __name__ == "__main__":
    run_preprocessing()
