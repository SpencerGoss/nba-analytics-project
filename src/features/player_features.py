"""
Player Game Feature Engineering
=================================
Builds rolling, lag, and contextual features from player_game_logs for use in
player performance prediction models.

Each output row is one player-game enriched with pre-game rolling stats
(shift-1 to prevent data leakage).

Memory-efficient implementation: processes season-by-season and streams
output to CSV to avoid holding the full dataset in memory.

New in v2:
  - Opponent defensive context: opp_pts_allowed_roll20, opp_net_rating_roll20
    Tells the model how strong the defense is that this player is facing.
  - Player age and height/weight from player_bio_stats
    Age captures career trajectory; size matters for rebound/block prediction.

Usage:
    from src.features.player_features import build_player_game_features
    df = build_player_game_features()

    python src/features/player_features.py
"""

import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.features.era_labels import label_eras


# ── Config ─────────────────────────────────────────────────────────────────────

GAME_LOG_PATH      = "data/processed/player_game_logs.csv"
ADV_STATS_PATH     = "data/processed/player_stats_advanced.csv"
BIO_STATS_PATH     = "data/processed/player_bio_stats.csv"
TEAM_FEATURES_PATH = "data/features/team_game_features.csv"
OUTPUT_PATH        = "data/features/player_game_features.csv"

ROLL_WINDOWS = [5, 10, 20]

ROLL_STATS = [
    "pts", "reb", "ast", "stl", "blk", "tov",
    "min", "fg_pct", "fg3_pct", "ft_pct",
    "plus_minus",
]

VOL_STATS = ["pts", "min", "ast", "reb"]

ADV_CONTEXT_COLS = [
    "player_id", "season",
    "usg_pct", "ts_pct", "net_rating", "pie",
    "ast_pct", "oreb_pct", "dreb_pct",
]

BIO_COLS = [
    "player_id", "season",
    "age", "player_height_inches", "player_weight",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_home_away(matchup: pd.Series) -> pd.Series:
    return matchup.str.contains(" vs\\.", regex=True).astype(int)


def _extract_opponent_abbr(matchup: pd.Series) -> pd.Series:
    """
    Extract opponent abbreviation from player game log matchup string.
      'LBJ vs. BOS'  →  'BOS'
      'LBJ @ BOS'    →  'BOS'
    """
    return matchup.str.split(r" vs\. | @ ", regex=True).str[-1].str.strip()


def _compute_player_rolling(group: pd.DataFrame, windows: list) -> pd.DataFrame:
    """
    Given a DataFrame for ONE player (sorted by game_date), compute all
    rolling features using shift(1) to avoid leakage.
    """
    g = group.copy()

    for col in ROLL_STATS:
        if col not in g.columns:
            continue
        shifted = g[col].shift(1)
        for w in windows:
            g[f"{col}_roll{w}"] = shifted.rolling(window=w, min_periods=1).mean()

    for col in VOL_STATS:
        if col not in g.columns:
            continue
        shifted = g[col].shift(1)
        for w in windows:
            g[f"{col}_std{w}"] = shifted.rolling(window=w, min_periods=2).std()

    for col in ["pts", "min", "ast", "reb"]:
        if col in g.columns:
            g[f"{col}_season_avg"] = g[col].shift(1).expanding().mean()

    return g


def _load_opponent_defense(team_features_path: str = TEAM_FEATURES_PATH) -> pd.DataFrame:
    """
    Build an opponent defensive context lookup table from team_game_features.

    For each (team_abbreviation, game_id), extract the team's rolling
    defensive stats (points allowed and net rating) so we can join these
    to player game rows by matching the player's opponent to a team.

    Returns a DataFrame with columns:
        opponent_abbr, game_id,
        opp_pts_allowed_roll5/10/20,
        opp_net_rating_roll5/10/20
    """
    if not os.path.exists(team_features_path):
        print(f"  Warning: {team_features_path} not found. "
              "Run team_game_features.py first to get opponent defensive context.")
        return None

    print("  Loading opponent defensive context from team_game_features...")
    tdf = pd.read_csv(team_features_path, usecols=lambda c: (
        c in ["team_abbreviation", "game_id"]
        or "opp_pts_roll" in c
        or "plus_minus_roll" in c
    ))

    # Rename so they clearly indicate opponent defense when joined
    rename = {}
    for c in tdf.columns:
        if "opp_pts_roll" in c:
            rename[c] = c.replace("opp_pts_roll", "opp_pts_allowed_roll")
        elif "plus_minus_roll" in c:
            rename[c] = c.replace("plus_minus_roll", "opp_net_rating_roll")
    tdf = tdf.rename(columns=rename)
    tdf = tdf.rename(columns={"team_abbreviation": "opponent_abbr"})

    return tdf


def _load_player_bio(bio_path: str = BIO_STATS_PATH) -> pd.DataFrame:
    """
    Load player bio stats (age, height, weight) for joining to player features.
    Only available for recent seasons (2020-21+), will be NaN for earlier rows.
    """
    if not os.path.exists(bio_path):
        print(f"  Warning: {bio_path} not found. Skipping bio features.")
        return None

    print("  Loading player bio stats...")
    bio = pd.read_csv(bio_path, usecols=BIO_COLS)
    bio = bio.drop_duplicates(subset=["player_id", "season"])
    return bio


# ── Main builder ───────────────────────────────────────────────────────────────

def build_player_game_features(
    game_log_path: str  = GAME_LOG_PATH,
    adv_stats_path: str = ADV_STATS_PATH,
    bio_stats_path: str = BIO_STATS_PATH,
    team_features_path: str = TEAM_FEATURES_PATH,
    output_path: str    = OUTPUT_PATH,
    roll_windows: list  = ROLL_WINDOWS,
    min_games: int      = 5,
    start_season: str   = "199697",   # modern era default
) -> pd.DataFrame:
    """
    Build pre-game rolling features for every player-game.

    Processes players one at a time to stay memory-efficient.
    Streams results to CSV by season batch.

    Args:
        start_season: Only include seasons >= this value (default: 1996-97)
    """
    print("Loading player_game_logs...")
    df = pd.read_csv(
        game_log_path,
        usecols=["season", "player_id", "player_name", "team_id", "team_abbreviation",
                 "game_id", "game_date", "matchup", "wl"] + ROLL_STATS + ["stl", "blk", "tov", "pf"],
    )
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Limit to modern era
    df = df[df["season"].astype(str) >= start_season].copy()
    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    print(f"  Filtered to {start_season}+: {len(df):,} rows | "
          f"{df.player_id.nunique():,} players | {df.season.nunique()} seasons")

    # ── Context features ──────────────────────────────────────────────────────
    df["is_home"]       = _parse_home_away(df["matchup"])
    df["opponent_abbr"] = _extract_opponent_abbr(df["matchup"])
    df["win"]           = (df["wl"] == "W").astype(int)

    df["days_rest"] = (
        df.groupby("player_id")["game_date"]
        .diff()
        .dt.days
        .fillna(7)
        .clip(upper=14)
    )
    df["is_back_to_back"] = (df["days_rest"] <= 1).astype(int)
    df["player_season_game_num"] = (
        df.groupby(["player_id", "season"]).cumcount() + 1
    )

    # ── Load supplementary tables ─────────────────────────────────────────────
    print("Loading advanced stats...")
    adv = pd.read_csv(adv_stats_path)
    adv = adv[ADV_CONTEXT_COLS].drop_duplicates(subset=["player_id", "season"])

    opp_defense = _load_opponent_defense(team_features_path)
    bio         = _load_player_bio(bio_stats_path)

    # ── Process player-by-player ──────────────────────────────────────────────
    print("Computing rolling features per player...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    player_ids = df["player_id"].unique()
    total      = len(player_ids)
    results    = []
    batch_size = 500

    for i, pid in enumerate(player_ids):
        pdata = df[df["player_id"] == pid].copy()
        if len(pdata) < min_games:
            continue
        pdata = _compute_player_rolling(pdata, roll_windows)
        results.append(pdata)

        # Flush to disk every batch_size players to control memory
        if (i + 1) % batch_size == 0 or (i + 1) == total:
            chunk = pd.concat(results, ignore_index=True)

            # Join season-level advanced stats
            chunk = chunk.merge(adv, on=["player_id", "season"], how="left")

            # Join opponent defensive context (game-level)
            # Match player's opponent_abbr + game_id to the opposing team's
            # rolling defensive stats from team_game_features
            if opp_defense is not None:
                chunk = chunk.merge(
                    opp_defense,
                    on=["opponent_abbr", "game_id"],
                    how="left",
                )

            # Join player bio stats (age, height, weight)
            # Available from 2020-21+; will be NaN for earlier seasons
            if bio is not None:
                chunk = chunk.merge(bio, on=["player_id", "season"], how="left")

            # Add era labels
            chunk = label_eras(chunk, season_col="season")

            # Write header only on first batch
            write_header = (i + 1) <= batch_size
            chunk.to_csv(output_path, mode="w" if write_header else "a",
                         header=write_header, index=False)
            print(f"  Processed {i+1:,}/{total:,} players, "
                  f"flushed {len(chunk):,} rows to disk")
            results = []
            del chunk

    # ── Verify output ─────────────────────────────────────────────────────────
    final = pd.read_csv(output_path, nrows=5)
    total_rows = sum(1 for _ in open(output_path)) - 1
    print(f"\nSaved {total_rows:,} rows × {len(final.columns)} cols → {output_path}")
    return final


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    build_player_game_features()
    print("\nPlayer feature engineering complete.")
