"""
Shared pytest fixtures for NBA analytics tests.

All fixtures produce small synthetic DataFrames (5-10 rows) that mirror
the structure of real data without depending on actual data files.
"""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def team_game_logs_df():
    """
    Synthetic team game log with 10 rows for a single team (LAL, team_id=1610612747).

    Covers 10 consecutive games in the 202425 season with realistic matchup strings,
    dates, and box-score stats. Two games are back-to-back (days_rest=1).
    """
    dates = pd.date_range("2024-10-22", periods=10, freq="2D")
    # Make games 4 and 5 back-to-back (1 day apart)
    dates = dates.to_list()
    dates[4] = dates[3] + pd.Timedelta(days=1)

    opponents = ["GSW", "PHX", "DEN", "SAC", "LAC", "BOS", "MIA", "CHI", "NYK", "BKN"]
    home_flags = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    matchups = []
    for i, (opp, home) in enumerate(zip(opponents, home_flags)):
        if home:
            matchups.append(f"LAL vs. {opp}")
        else:
            matchups.append(f"LAL @ {opp}")

    wl = ["W", "L", "W", "W", "L", "W", "W", "L", "W", "W"]

    return pd.DataFrame({
        "season_id": [22024] * 10,
        "season": [202425] * 10,
        "team_id": [1610612747] * 10,
        "team_abbreviation": ["LAL"] * 10,
        "team_name": ["Los Angeles Lakers"] * 10,
        "game_id": [f"002240{i:04d}" for i in range(1, 11)],
        "game_date": dates,
        "matchup": matchups,
        "wl": wl,
        "pts": [110, 98, 115, 105, 95, 112, 108, 100, 120, 118],
        "fgm": [42, 36, 44, 40, 35, 43, 41, 38, 46, 45],
        "fga": [88, 85, 90, 86, 82, 89, 87, 84, 92, 91],
        "fg_pct": [0.477, 0.424, 0.489, 0.465, 0.427, 0.483, 0.471, 0.452, 0.500, 0.495],
        "fg3m": [12, 8, 14, 10, 9, 13, 11, 7, 15, 14],
        "fg3a": [32, 28, 35, 30, 26, 34, 31, 25, 36, 33],
        "fg3_pct": [0.375, 0.286, 0.400, 0.333, 0.346, 0.382, 0.355, 0.280, 0.417, 0.424],
        "ftm": [14, 18, 13, 15, 16, 13, 15, 17, 13, 14],
        "fta": [18, 22, 16, 20, 21, 17, 19, 23, 16, 18],
        "ft_pct": [0.778, 0.818, 0.813, 0.750, 0.762, 0.765, 0.789, 0.739, 0.813, 0.778],
        "oreb": [10, 8, 12, 9, 7, 11, 10, 8, 13, 11],
        "dreb": [32, 30, 35, 31, 28, 33, 32, 29, 36, 34],
        "reb": [42, 38, 47, 40, 35, 44, 42, 37, 49, 45],
        "ast": [25, 20, 28, 23, 18, 26, 24, 21, 29, 27],
        "stl": [8, 5, 9, 7, 4, 8, 7, 6, 10, 9],
        "blk": [5, 3, 6, 4, 2, 5, 4, 3, 7, 6],
        "tov": [12, 15, 10, 13, 16, 11, 12, 14, 9, 10],
        "pf": [20, 22, 18, 21, 24, 19, 20, 23, 17, 18],
        "plus_minus": [8, -10, 12, 5, -7, 6, 10, -8, 15, 12],
    })


@pytest.fixture
def two_team_game_logs_df(team_game_logs_df):
    """
    Game logs for two teams (LAL and GSW) so we can test opponent joins.

    The first game (game_id 0022400001) is LAL vs GSW, so both teams
    share that game_id, enabling opponent-stat self-join tests.
    """
    lal = team_game_logs_df.copy()
    gsw = lal.copy()
    gsw["team_id"] = 1610612744
    gsw["team_abbreviation"] = "GSW"
    gsw["team_name"] = "Golden State Warriors"
    # Flip matchups and results for GSW perspective
    gsw["matchup"] = gsw["matchup"].str.replace("LAL", "___").str.replace("GSW", "LAL").str.replace("___", "GSW")
    gsw["matchup"] = gsw["matchup"].apply(
        lambda m: m.replace(" vs. ", " TEMP ").replace(" @ ", " vs. ").replace(" TEMP ", " @ ")
    )
    gsw["wl"] = gsw["wl"].map({"W": "L", "L": "W"})
    gsw["plus_minus"] = -gsw["plus_minus"]
    gsw["pts"] = lal["pts"] - lal["plus_minus"]  # opp_pts from LAL perspective

    return pd.concat([lal, gsw], ignore_index=True)


@pytest.fixture
def player_game_logs_df():
    """
    Synthetic player game log for injury proxy tests.

    Player A (id=101) plays games 1-5, misses 6-8, plays 9-10.
    Player B (id=102) plays all 10 games with low minutes (bench player).
    Player C (id=103) plays all 10 games as a star (high usage, high minutes).

    All on team_id=1610612747 (LAL), season=202425.
    """
    dates = pd.date_range("2024-10-22", periods=10, freq="3D")
    game_ids = [f"002240{i:04d}" for i in range(1, 11)]

    rows = []
    for i in range(10):
        # Player A: plays games 0-4 and 8-9 (misses 5, 6, 7)
        if i not in [5, 6, 7]:
            rows.append({
                "season": 202425,
                "player_id": 101,
                "player_name": "Player A",
                "team_id": 1610612747,
                "team_abbreviation": "LAL",
                "game_id": game_ids[i],
                "game_date": dates[i],
                "min": 30,
            })

        # Player B: plays every game, low minutes
        rows.append({
            "season": 202425,
            "player_id": 102,
            "player_name": "Player B",
            "team_id": 1610612747,
            "team_abbreviation": "LAL",
            "game_id": game_ids[i],
            "game_date": dates[i],
            "min": 10,
        })

        # Player C: plays every game, star player
        rows.append({
            "season": 202425,
            "player_id": 103,
            "player_name": "Player C",
            "team_id": 1610612747,
            "team_abbreviation": "LAL",
            "game_id": game_ids[i],
            "game_date": dates[i],
            "min": 35,
        })

    return pd.DataFrame(rows)


@pytest.fixture
def adv_stats_df():
    """
    Synthetic advanced stats with usage rates for injury proxy tests.

    Player A: usg_pct = 0.22 (rotation player, not a star)
    Player B: usg_pct = 0.10 (bench player, low usage)
    Player C: usg_pct = 0.28 (star player, above STAR_USG_THRESHOLD=0.25)
    """
    return pd.DataFrame({
        "player_id": [101, 102, 103],
        "season": [202425, 202425, 202425],
        "usg_pct": [0.22, 0.10, 0.28],
    })
