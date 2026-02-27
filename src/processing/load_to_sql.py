import sqlite3
import pandas as pd
import os


def load_all_tables():
    """Load all processed CSVs into the SQLite database."""
    conn = sqlite3.connect("database/nba.db")

    # --- Core tables ---
    pd.read_csv("data/processed/players.csv").to_sql("players", conn, if_exists="replace", index=False)
    pd.read_csv("data/processed/player_stats.csv").to_sql("player_stats", conn, if_exists="replace", index=False)
    pd.read_csv("data/processed/player_stats_advanced.csv").to_sql("player_stats_advanced", conn, if_exists="replace", index=False)
    pd.read_csv("data/processed/player_game_logs.csv").to_sql("player_game_logs", conn, if_exists="replace", index=False)
    pd.read_csv("data/processed/team_stats.csv").to_sql("team_stats", conn, if_exists="replace", index=False)
    pd.read_csv("data/processed/team_stats_advanced.csv").to_sql("team_stats_advanced", conn, if_exists="replace", index=False)
    pd.read_csv("data/processed/team_game_logs.csv").to_sql("team_game_logs", conn, if_exists="replace", index=False)
    pd.read_csv("data/processed/standings.csv").to_sql("standings", conn, if_exists="replace", index=False)

    # --- Playoff tables ---
    pd.read_csv("data/processed/player_game_logs_playoffs.csv").to_sql("player_game_logs_playoffs", conn, if_exists="replace", index=False)
    pd.read_csv("data/processed/team_game_logs_playoffs.csv").to_sql("team_game_logs_playoffs", conn, if_exists="replace", index=False)
    pd.read_csv("data/processed/player_stats_playoffs.csv").to_sql("player_stats_playoffs", conn, if_exists="replace", index=False)
    pd.read_csv("data/processed/player_stats_advanced_playoffs.csv").to_sql("player_stats_advanced_playoffs", conn, if_exists="replace", index=False)
    pd.read_csv("data/processed/team_stats_playoffs.csv").to_sql("team_stats_playoffs", conn, if_exists="replace", index=False)
    pd.read_csv("data/processed/team_stats_advanced_playoffs.csv").to_sql("team_stats_advanced_playoffs", conn, if_exists="replace", index=False)

    # --- Clutch and scoring tables ---
    pd.read_csv("data/processed/player_stats_clutch.csv").to_sql("player_stats_clutch", conn, if_exists="replace", index=False)
    pd.read_csv("data/processed/player_stats_scoring.csv").to_sql("player_stats_scoring", conn, if_exists="replace", index=False)

    # --- Hustle tables ---
    pd.read_csv("data/processed/player_hustle_stats.csv").to_sql("player_hustle_stats", conn, if_exists="replace", index=False)
    pd.read_csv("data/processed/team_hustle_stats.csv").to_sql("team_hustle_stats", conn, if_exists="replace", index=False)

    # --- Teams dimension (load if available) ---
    if os.path.exists("data/processed/teams.csv"):
        pd.read_csv("data/processed/teams.csv").to_sql("teams", conn, if_exists="replace", index=False)
        print("teams loaded.")

    # --- Player bio stats (load if available) ---
    if os.path.exists("data/processed/player_bio_stats.csv"):
        pd.read_csv("data/processed/player_bio_stats.csv").to_sql("player_bio_stats", conn, if_exists="replace", index=False)
        print("player_bio_stats loaded.")

    # --- Shot chart (load if available) ---
    if os.path.exists("data/processed/shot_chart.csv"):
        pd.read_csv("data/processed/shot_chart.csv").to_sql("shot_chart", conn, if_exists="replace", index=False)
        print("shot_chart loaded.")

    conn.close()
    print("All tables loaded into SQLite successfully.")


if __name__ == "__main__":
    load_all_tables()
