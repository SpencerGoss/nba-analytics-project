"""
Team Game Feature Engineering
==============================
Builds rolling, lag, and contextual features from team_game_logs for use in
game outcome prediction and standings/playoff models.

Each row in the output represents one team's perspective for one game, enriched
with pre-game rolling stats (computed ONLY from prior games to prevent leakage).

New in v2:
  - opp_pts (points allowed) rolling means — key defensive signal
  - Strength of schedule (rolling average of opponent win% at game time)
  - Explicit matchup differential features in build_matchup_dataset()

Usage:
    from src.features.team_game_features import build_team_game_features
    df = build_team_game_features()

    # Or run directly:
    python src/features/team_game_features.py
"""

import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.features.era_labels import label_eras


# ── Config ─────────────────────────────────────────────────────────────────────

DATA_PATH     = "data/processed/team_game_logs.csv"
OUTPUT_PATH   = "data/features/team_game_features.csv"
ROLL_WINDOWS  = [5, 10, 20]   # rolling window sizes (games)

ROLL_STATS = [
    "pts", "fg_pct", "fg3_pct", "ft_pct",
    "reb", "oreb", "dreb",
    "ast", "stl", "blk", "tov", "pf",
    "plus_minus",
    "opp_pts",        # points allowed — key defensive signal
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_home_away(matchup: pd.Series) -> pd.Series:
    """
    Extract home/away flag from matchup string.
      'LAL vs. GSW'  →  home (1)
      'LAL @ GSW'    →  away (0)
    """
    return matchup.str.contains(" vs\\.", regex=True).astype(int)


def _extract_opponent_abbr(matchup: pd.Series) -> pd.Series:
    """
    Extract the opponent team abbreviation from the matchup string.
      'LAL vs. GSW'  →  'GSW'
      'LAL @ GSW'    →  'GSW'
    """
    return matchup.str.split(r" vs\. | @ ", regex=True).str[-1].str.strip()


def _rolling_mean_shift(group: pd.DataFrame, col: str, window: int) -> pd.Series:
    """
    Compute a shifted rolling mean to avoid data leakage.
    Shift(1) ensures we only use games BEFORE the current one.
    min_periods=1 so early games still get a value (smaller sample).
    """
    return (
        group[col]
        .shift(1)
        .rolling(window=window, min_periods=1)
        .mean()
    )


def _rolling_win_pct(group: pd.DataFrame, window: int) -> pd.Series:
    """Rolling win percentage over the last N games (shift to avoid leakage)."""
    win_flag = (group["wl"] == "W").astype(int)
    return (
        win_flag
        .shift(1)
        .rolling(window=window, min_periods=1)
        .mean()
    )


# ── Main builder ───────────────────────────────────────────────────────────────

def build_team_game_features(
    data_path: str = DATA_PATH,
    output_path: str = OUTPUT_PATH,
    roll_windows: list = ROLL_WINDOWS,
) -> pd.DataFrame:
    """
    Build pre-game rolling features for every team-game.

    Steps:
      1. Load and prep team game logs
      2. Compute opp_pts (points allowed) from pts and plus_minus
      3. Sort by team + game_date (chronological)
      4. Compute rolling stats (shift-1 to prevent leakage)
      5. Add context features: home/away, days_rest, back_to_back, win_flag
      6. Add strength of schedule (rolling opponent win% at game time)
      7. Save to data/features/team_game_features.csv

    Returns:
        pd.DataFrame with one row per team-game, featuring:
          - Identifiers: team_id, game_id, game_date, season, matchup
          - Context: is_home, opponent_abbr, days_rest, is_back_to_back
          - Target: win (1=W, 0=L)
          - Rolling stats: e.g. pts_roll5, fg_pct_roll10, win_pct_roll5, …
          - New: opp_pts_roll20 (defensive), sos_roll10/20 (schedule strength)
    """
    print("Loading team_game_logs...")
    df = pd.read_csv(data_path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["team_id", "game_date"]).reset_index(drop=True)

    # ── Basic context features ────────────────────────────────────────────────
    df["is_home"]       = _parse_home_away(df["matchup"])
    df["opponent_abbr"] = _extract_opponent_abbr(df["matchup"])
    df["win"]           = (df["wl"] == "W").astype(int)

    # ── Opponent points (points allowed) ─────────────────────────────────────
    # plus_minus = team_pts - opp_pts  →  opp_pts = pts - plus_minus
    # This is the team's defensive output: how many points did they allow?
    df["opp_pts"] = df["pts"] - df["plus_minus"]

    # ── Days rest (days since last game for this team) ────────────────────────
    df["days_rest"] = (
        df.groupby("team_id")["game_date"]
        .diff()
        .dt.days
        .fillna(7)    # first game of season gets a neutral 7-day rest
        .clip(upper=14)
    )
    df["is_back_to_back"] = (df["days_rest"] <= 1).astype(int)

    # ── Cumulative wins and games played (season context) ─────────────────────
    df["season_game_num"] = df.groupby(["team_id", "season"]).cumcount() + 1

    # ── Rolling stats per team ────────────────────────────────────────────────
    print("Computing rolling features...")

    for window in roll_windows:
        group = df.groupby("team_id", group_keys=False)

        # Rolling mean for each stat
        for stat in ROLL_STATS:
            if stat in df.columns:
                col_name = f"{stat}_roll{window}"
                df[col_name] = group.apply(
                    lambda g: _rolling_mean_shift(g, stat, window),
                    include_groups=False,
                ).values

        # Rolling win percentage
        df[f"win_pct_roll{window}"] = group.apply(
            lambda g: _rolling_win_pct(g, window),
            include_groups=False,
        ).values

    # ── Season-level cumulative win% (prior to this game) ────────────────────
    df["cum_wins"] = (
        df.groupby(["team_id", "season"])["win"]
        .transform(lambda x: x.shift(1).cumsum())
        .fillna(0)
    )
    df["cum_losses"] = (
        df.groupby(["team_id", "season"])["win"]
        .transform(lambda x: (1 - x).shift(1).cumsum())
        .fillna(0)
    )
    df["cum_win_pct"] = df["cum_wins"] / (df["cum_wins"] + df["cum_losses"]).replace(0, np.nan)
    df["cum_win_pct"] = df["cum_win_pct"].fillna(0.5)   # neutral at game 1

    # ── Strength of schedule ──────────────────────────────────────────────────
    # For each game, look up the opponent's cumulative win% entering that game.
    # Then roll those values to get a team's average strength of schedule.
    # The opponent's cum_win_pct is already pre-game (no leakage risk).
    print("Computing strength of schedule...")
    opp_strength = df[["team_abbreviation", "game_id", "cum_win_pct"]].copy()
    opp_strength.columns = ["opponent_abbr", "game_id", "opp_pre_game_win_pct"]

    df = df.merge(opp_strength, on=["opponent_abbr", "game_id"], how="left")

    # Rolling SOS (shift-1 so current game's opponent is not included)
    sos_group = df.groupby("team_id", group_keys=False)
    for window in [10, 20]:
        df[f"sos_roll{window}"] = sos_group.apply(
            lambda g: g["opp_pre_game_win_pct"].shift(1).rolling(window, min_periods=1).mean(),
            include_groups=False,
        ).values

    # ── Drop raw game stats (they'd leak the result) ──────────────────────────
    id_cols = [
        "season_id", "season", "team_id", "team_abbreviation", "team_name",
        "game_id", "game_date", "matchup", "wl",
    ]
    context_cols = [
        "is_home", "opponent_abbr", "days_rest", "is_back_to_back",
        "season_game_num", "cum_wins", "cum_losses", "cum_win_pct", "win",
        "sos_roll10", "sos_roll20",
    ]
    roll_cols = [c for c in df.columns if "_roll" in c and "sos_roll" not in c]

    output = df[id_cols + context_cols + roll_cols].copy()

    # ── Add era labels ────────────────────────────────────────────────────────
    print("Labeling eras...")
    output = label_eras(output, season_col="season")

    # ── Join injury proxy features ────────────────────────────────────────────
    # Build (or load) lineup availability features and join on (team_id, game_id).
    # These columns tell the model how healthy each team's rotation is entering
    # this game: missing_minutes, missing_usg_pct, rotation_availability,
    # star_player_out.
    injury_path = "data/features/injury_proxy_features.csv"
    try:
        from src.features.injury_proxy import build_injury_proxy_features
        print("Building injury proxy features...")
        injury_df = build_injury_proxy_features()
        output = output.merge(injury_df, on=["team_id", "game_id"], how="left")
        # Teams with no rotation data (very early seasons) get neutral defaults
        output["missing_minutes"]       = output["missing_minutes"].fillna(0)
        output["missing_usg_pct"]       = output["missing_usg_pct"].fillna(0)
        output["rotation_availability"] = output["rotation_availability"].fillna(1.0)
        output["star_player_out"]       = output["star_player_out"].fillna(0)
        output["n_missing_rotation"]    = output["n_missing_rotation"].fillna(0)
        print(f"  Injury features joined: {injury_df.columns.tolist()}")
    except Exception as e:
        print(f"  Warning: could not build injury proxy features ({e}). Skipping.")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Saved {len(output):,} rows × {len(output.columns)} cols → {output_path}")
    return output


def build_matchup_dataset(
    features_path: str = OUTPUT_PATH,
    output_path: str = "data/features/game_matchup_features.csv",
) -> pd.DataFrame:
    """
    Merge team-level features into a single game-level row for modeling.

    For each game, we get:
      - home_team features (prefixed 'home_')
      - away_team features (prefixed 'away_')
      - target: home_win (1 if home team won)
      - NEW: diff_ columns (home minus away for key stats)

    The differential features give the model the direct gap between teams
    rather than requiring it to infer the gap from two separate columns.
    This is the game outcome model's primary input.
    """
    print("Loading team game features...")
    df = pd.read_csv(features_path)
    df["game_date"] = pd.to_datetime(df["game_date"])

    home = df[df["is_home"] == 1].copy()
    away = df[df["is_home"] == 0].copy()

    roll_cols    = [c for c in df.columns if "_roll" in c]
    injury_cols  = [c for c in ["missing_minutes", "missing_usg_pct",
                                "rotation_availability", "star_player_out",
                                "n_missing_rotation"] if c in df.columns]
    context_cols = ["days_rest", "is_back_to_back", "season_game_num",
                    "cum_wins", "cum_losses", "cum_win_pct",
                    "sos_roll10", "sos_roll20"] + injury_cols
    feature_cols = context_cols + roll_cols

    home_feat = home[["game_id"] + feature_cols + ["win"]].copy()
    home_feat.columns = (
        ["game_id"]
        + [f"home_{c}" for c in feature_cols]
        + ["home_win"]
    )

    away_feat = away[["game_id"] + feature_cols].copy()
    away_feat.columns = (
        ["game_id"]
        + [f"away_{c}" for c in feature_cols]
    )

    # Meta from home side
    meta = home[["game_id", "season", "game_date", "team_abbreviation", "opponent_abbr"]].copy()
    meta.columns = ["game_id", "season", "game_date", "home_team", "away_team"]

    matchup = (
        meta
        .merge(home_feat, on="game_id", how="inner")
        .merge(away_feat, on="game_id", how="inner")
    )

    # ── Differential features ─────────────────────────────────────────────────
    # Explicit home-minus-away gaps for the most predictive stats.
    # These give the model the head-to-head edge directly.
    print("Computing matchup differential features...")

    diff_stats = [
        "pts_roll5", "pts_roll10", "pts_roll20",
        "opp_pts_roll5", "opp_pts_roll10", "opp_pts_roll20",  # defensive diff
        "plus_minus_roll5", "plus_minus_roll10", "plus_minus_roll20",
        "win_pct_roll5", "win_pct_roll10", "win_pct_roll20",
        "fg_pct_roll20", "fg3_pct_roll20",
        "ast_roll20", "tov_roll20",
        "cum_win_pct",
        "sos_roll10", "sos_roll20",
        # Injury differentials — positive means home team has MORE missing minutes
        # (i.e. is MORE depleted). A large negative value = home team is healthy,
        # away team is banged up — a significant edge.
        "missing_minutes", "missing_usg_pct", "rotation_availability",
    ]

    for stat in diff_stats:
        h_col = f"home_{stat}"
        a_col = f"away_{stat}"
        if h_col in matchup.columns and a_col in matchup.columns:
            matchup[f"diff_{stat}"] = matchup[h_col] - matchup[a_col]

    matchup = matchup.dropna(subset=["home_win"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    matchup.to_csv(output_path, index=False)
    print(f"Saved {len(matchup):,} matchup rows × {len(matchup.columns)} cols → {output_path}")
    return matchup


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    build_team_game_features()
    build_matchup_dataset()
    print("\nTeam game feature engineering complete.")
