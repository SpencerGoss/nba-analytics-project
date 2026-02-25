import sqlite3
import pandas as pd

# Load cleaned data from processed folder
df_players = pd.read_csv("data/processed/players.csv")
df_player_stats = pd.read_csv("data/processed/player_stats.csv")
df_team_stats = pd.read_csv("data/processed/team_stats.csv")
df_team_game_logs = pd.read_csv("data/processed/team_game_logs.csv")

# Connect to SQLite database
conn = sqlite3.connect("database/nba.db")

# Load each table into SQL
df_players.to_sql("players", conn, if_exists="replace", index=False)
df_player_stats.to_sql("player_stats", conn, if_exists="replace", index=False)
df_team_stats.to_sql("team_stats", conn, if_exists="replace", index=False)
df_team_game_logs.to_sql("team_game_logs", conn, if_exists="replace", index=False)

conn.close()

print("All tables loaded into SQLite successfully.")

