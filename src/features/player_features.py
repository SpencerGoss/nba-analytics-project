"""
Player Game Feature Engineering
=================================
Builds rolling, lag, and contextual features from player_game_logs for use in
player performance prediction models (props, fantasy, player props model).

Each output row is one player-game enriched with pre-game rolling stats
(shift-1 to prevent data leakage).

Memory-efficient: processes season-by-season and streams output to CSV.

New in v2:
  - Opponent defensive context: opp_pts_allowed_roll20, opp_net_rating_roll20
  - Player age and height/weight from player_bio_stats

New in v3 (comprehensive props-model features):
  MINUTES & ROLE:
    minutes_last_5_avg, minutes_last_10_avg, minutes_trend_5,
    games_started_last_5, starter_flag

  USAGE & OPPORTUNITY:
    usage_rate_roll10 (box score proxy), fga_roll10, touches_roll10,
    assist_opportunities_roll10, top2_usage_share

  MATCHUP VS POSITION:
    opp_pts/ast/reb/fg_pct_allowed_to_position_roll10
    (position inferred from player height; G <78", F 78-82", C >=82")

  GAME ENVIRONMENT:
    team_pace_roll10, opp_pace_roll10, game_pace_estimate,
    implied_team_points, implied_game_total

  INJURY CONTEXT:
    missing_teammate_minutes, missing_teammate_usage_pct, role_expansion_flag

  REST & LOCATION:
    rest_days, is_back_to_back, home_away_flag,
    pts/ast/reb_home_avg, pts/ast/reb_away_avg

  CONSISTENCY / DISTRIBUTION:
    pts/ast/reb_std10, over_rate_pts/ast/reb_10, pts/ast/reb_trend_5

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

GAME_LOG_PATH        = "data/processed/player_game_logs.csv"
ADV_STATS_PATH       = "data/processed/player_stats_advanced.csv"
BIO_STATS_PATH       = "data/processed/player_bio_stats.csv"
POSITIONS_PATH       = "data/processed/player_positions.csv"   # from get_player_positions.py
TEAM_FEATURES_PATH   = "data/features/team_game_features.csv"
INJURY_PROXY_PATH    = "data/features/injury_proxy_features.csv"
OUTPUT_PATH          = "data/features/player_game_features.csv"

ROLL_WINDOWS = [5, 10, 20]

ROLL_STATS = [
    "pts", "reb", "ast", "stl", "blk", "tov",
    "min", "fga", "fta",               # fga/fta needed for usage proxy
    "fg_pct", "fg3_pct", "ft_pct",
    "plus_minus",
]

VOL_STATS = ["pts", "min", "ast", "reb"]   # compute std for these

# Stats to compute home/away splits and over-rate for
SPLIT_STATS = ["pts", "ast", "reb"]

ADV_CONTEXT_COLS = [
    "player_id", "season",
    "usg_pct", "ts_pct", "net_rating", "pie",
    "ast_pct", "oreb_pct", "dreb_pct",
]

BIO_COLS = [
    "player_id", "season",
    "age", "player_height_inches", "player_weight",
]

# Position bucket height thresholds (inches) — proxy since position not in dataset
POS_G_MAX_HEIGHT = 78    # < 78"  = Guard (up to 6'6")
POS_F_MAX_HEIGHT = 82    # < 82"  = Forward (up to 6'10")
                          # >= 82" = Center


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_home_away(matchup: pd.Series) -> pd.Series:
    return matchup.str.contains(" vs\\.", regex=True).astype(int)


def _extract_opponent_abbr(matchup: pd.Series) -> pd.Series:
    """
    Extract opponent abbreviation from player game log matchup string.
      'LAL vs. BOS'  →  'BOS'
      'LAL @ BOS'    →  'BOS'
    """
    return matchup.str.split(r" vs\. | @ ", regex=True).str[-1].str.strip()


def _rolling_slope(values: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Compute rolling linear regression slope using shift-1 semantics.
    At position i, fits a line through values[max(0, i-window) : i].
    Returns 0.0 for positions with fewer than 2 prior values.
    """
    n = len(values)
    slopes = np.full(n, 0.0)
    for i in range(1, n):
        start = max(0, i - window)
        y = values[start:i]
        if len(y) < 2:
            slopes[i] = 0.0
            continue
        x = np.arange(len(y), dtype=float)
        slopes[i] = np.polyfit(x, y, 1)[0]
    return slopes


# ── Pre-computation helpers (cross-player lookups) ─────────────────────────────

def _build_starter_flag_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Infer starter status from minutes played within each (team_id, game_id).
    Top-5 players by minutes on a team in a game are flagged as starters.

    This is a proxy since nba_api player game logs don't include an explicit
    'started' column. Works well for ~95%+ of cases.

    Returns: DataFrame[player_id, game_id, starter_flag]
    """
    work = df[["player_id", "team_id", "game_id", "min"]].dropna(subset=["min"])
    work = work.copy()
    work["rank_in_game"] = (
        work.groupby(["team_id", "game_id"])["min"]
        .rank(method="first", ascending=False)
    )
    work["starter_flag"] = (work["rank_in_game"] <= 5).astype(int)
    return work[["player_id", "game_id", "starter_flag"]]


def _build_team_usage_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (team_id, game_id), compute top2_usage_share: the combined
    usage proxy of the top 2 usage players divided by total team usage.

    Usage proxy = FGA + 0.44*FTA + TOV  (standard formula component).

    Then roll each team's top2_usage_share over the prior 10 games so each
    player row can see how concentrated their team's usage has been.

    Returns: DataFrame[team_id, game_id, top2_usage_share]
    """
    work = df[["player_id", "team_id", "game_id", "game_date",
               "fga", "fta", "tov"]].copy()
    work["fga"]  = work["fga"].fillna(0)
    work["fta"]  = work["fta"].fillna(0)
    work["tov"]  = work["tov"].fillna(0)
    work["usage_proxy"] = work["fga"] + 0.44 * work["fta"] + work["tov"]

    def _top2_share(grp):
        total = grp["usage_proxy"].sum()
        if total <= 0:
            return 0.5
        return grp["usage_proxy"].nlargest(2).sum() / total

    tg = (
        work.groupby(["team_id", "game_id"])
        .apply(_top2_share, include_groups=False)
        .reset_index(name="top2_usage_raw")
    )

    # Add game_date for sorting
    date_map = work[["team_id", "game_id", "game_date"]].drop_duplicates()
    tg = tg.merge(date_map, on=["team_id", "game_id"], how="left")
    tg = tg.sort_values(["team_id", "game_date"])

    # Roll per team with shift-1 to prevent leakage
    tg["top2_usage_share"] = (
        tg.groupby("team_id")["top2_usage_raw"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    )
    return tg[["team_id", "game_id", "top2_usage_share"]]


def _build_position_map(
    positions_path: str = POSITIONS_PATH,
    bio_path:       str = BIO_STATS_PATH,
) -> pd.Series:
    """
    Build a player_id → position_bucket (G / F / C) mapping.

    Priority order:
      1. data/processed/player_positions.csv  — official NBA position from
         the nba_api PlayerIndex endpoint (run src/data/get_player_positions.py
         to generate this file; it only needs to be run once).

      2. Height-based inference from player_bio_stats.csv (fallback when the
         positions file hasn't been fetched yet).

         Height thresholds:
           < 78"  → Guard   (up to 6'6")
           78-82" → Forward (6'6"–6'10")
           >= 82" → Center  (taller than 6'10")

    Returns a Series mapping player_id → "G" | "F" | "C".
    """
    # ── Source 1: official NBA positions ──────────────────────────────────────
    if os.path.exists(positions_path):
        print(f"  Loading player positions from {positions_path}...")
        pos_df = pd.read_csv(
            positions_path,
            usecols=["player_id", "position_bucket"],
        )
        pos_df = pos_df.dropna(subset=["position_bucket"])
        pos_df = pos_df.drop_duplicates("player_id", keep="first")
        pos_map = pos_df.set_index("player_id")["position_bucket"]

        fill_rate = (pos_map != "").mean() * 100 if len(pos_map) else 0
        print(f"    {len(pos_map):,} players loaded "
              f"| distribution: {pos_map.value_counts().to_dict()}")

        # If we got meaningful data, return it (may still have gaps for very old
        # players — the fallback below will fill those in the caller via .fillna)
        if len(pos_map) >= 100:
            return pos_map

        print("  Warning: player_positions.csv has fewer than 100 entries — "
              "falling through to height-based fallback.")

    else:
        print(f"  Note: {positions_path} not found.")
        print("  Run  python src/data/get_player_positions.py  to fetch official positions.")
        print("  Falling back to height-based position inference...")

    # ── Source 2: height-based fallback ───────────────────────────────────────
    if not os.path.exists(bio_path):
        print(f"  Warning: {bio_path} also not found. Position bucket will default to 'F'.")
        return pd.Series(dtype=str)

    bio = pd.read_csv(bio_path, usecols=["player_id", "player_height_inches"])
    bio = bio.dropna(subset=["player_height_inches"])
    bio = bio.drop_duplicates("player_id", keep="last")

    conditions = [
        bio["player_height_inches"] < POS_G_MAX_HEIGHT,
        (bio["player_height_inches"] >= POS_G_MAX_HEIGHT)
        & (bio["player_height_inches"] < POS_F_MAX_HEIGHT),
        bio["player_height_inches"] >= POS_F_MAX_HEIGHT,
    ]
    bio["position_bucket"] = np.select(conditions, ["G", "F", "C"], default="F")
    print(f"  Height-inferred positions for {len(bio):,} players "
          f"| distribution: {bio['position_bucket'].value_counts().to_dict()}")
    return bio.set_index("player_id")["position_bucket"]


def _build_opp_positional_defense(
    df: pd.DataFrame,
    pos_map: pd.Series,
) -> pd.DataFrame:
    """
    For each (opponent_abbr, game_id, position_bucket), compute the rolling
    10-game average of pts / ast / reb / fg_pct allowed to players of that
    position group.

    Logic: When player P (a Guard) plays against team T, we look at team T's
    last 10 games and ask: "how many points per game did Guards average against T?"

    The 'opponent_abbr' of the offensive player's team is the defensive team.
    Shift-1 on the rolling ensures no leakage from the current game.

    Returns: DataFrame[opponent_abbr, game_id, position_bucket,
                        opp_pts/ast/reb/fg_pct_allowed_to_position_roll10]
    """
    work = df[["team_abbreviation", "opponent_abbr", "game_id", "game_date",
               "player_id", "pts", "ast", "reb", "fg_pct"]].copy()
    work["position_bucket"] = work["player_id"].map(pos_map).fillna("F")

    # Average stats per (defensive team, game, position)
    pos_game = (
        work.groupby(["opponent_abbr", "game_id", "game_date", "position_bucket"])
        [["pts", "ast", "reb", "fg_pct"]]
        .mean()
        .reset_index()
    )
    pos_game = pos_game.sort_values(["opponent_abbr", "position_bucket", "game_date"])

    # Rolling 10-game avg with shift-1 per (defensive team, position)
    for stat in ["pts", "ast", "reb", "fg_pct"]:
        pos_game[f"opp_{stat}_allowed_to_position_roll10"] = (
            pos_game.groupby(["opponent_abbr", "position_bucket"])[stat]
            .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        )

    out_cols = [
        "opponent_abbr", "game_id", "position_bucket",
        "opp_pts_allowed_to_position_roll10",
        "opp_ast_allowed_to_position_roll10",
        "opp_reb_allowed_to_position_roll10",
        "opp_fg_pct_allowed_to_position_roll10",
    ]
    return pos_game[out_cols]


def _build_team_pace_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate per-game team pace from player game logs aggregated to team level.

    Pace proxy: 2 × (FGA - OREB + TOV + 0.44×FTA)
    This is proportional to possessions and is commonly used in basketball
    analytics when full play-by-play data isn't available.

    Computes rolling 10-game average per team with shift-1 for leakage prevention.

    Returns: DataFrame[team_id, team_abbreviation, game_id, team_pace_roll10]
    """
    agg = (
        df.groupby(["team_id", "team_abbreviation", "game_id", "game_date"])
        [["fga", "oreb", "tov", "fta"]]
        .sum()
        .reset_index()
    )
    agg["pace_raw"] = 2 * (
        agg["fga"] - agg["oreb"] + agg["tov"] + 0.44 * agg["fta"]
    )
    agg = agg.sort_values(["team_id", "game_date"])
    agg["team_pace_roll10"] = (
        agg.groupby("team_id")["pace_raw"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    )
    return agg[["team_id", "team_abbreviation", "game_id", "team_pace_roll10"]]


def _load_opponent_defense(team_features_path: str = TEAM_FEATURES_PATH) -> pd.DataFrame:
    """
    Build an opponent defensive context lookup from team_game_features.

    For each (team_abbreviation, game_id), extract the team's rolling
    defensive stats (points allowed and net rating) for joining to player rows.

    Returns: DataFrame[opponent_abbr, game_id,
                        opp_pts_allowed_roll5/10/20,
                        opp_net_rating_roll5/10/20]
    """
    if not os.path.exists(team_features_path):
        print(f"  Warning: {team_features_path} not found. "
              "Run team_game_features.py first.")
        return None

    print("  Loading opponent defensive context from team_game_features...")
    tdf = pd.read_csv(team_features_path, usecols=lambda c: (
        c in ["team_abbreviation", "game_id"]
        or "opp_pts_roll" in c
        or "plus_minus_roll" in c
        or c in ["pts_roll10", "opp_pts_roll10"]   # for implied points
    ))

    rename = {}
    for c in tdf.columns:
        if "opp_pts_roll" in c:
            rename[c] = c.replace("opp_pts_roll", "opp_pts_allowed_roll")
        elif "plus_minus_roll" in c:
            rename[c] = c.replace("plus_minus_roll", "opp_net_rating_roll")
    tdf = tdf.rename(columns=rename)
    tdf = tdf.rename(columns={"team_abbreviation": "opponent_abbr"})
    return tdf


def _load_team_context(team_features_path: str = TEAM_FEATURES_PATH) -> pd.DataFrame:
    """
    Load team-game level context for joining to player rows:
      - pts_roll10 (team's expected scoring = implied_team_points proxy)
      - opp_pts_roll10 (opponent expected scoring)
      - days_rest, is_back_to_back, games_last_5_days, games_last_7_days
        (team-level rest context that may differ slightly from player-level)

    Returns: DataFrame[team_abbreviation, game_id, ...]
    """
    if not os.path.exists(team_features_path):
        return None

    cols_to_load = [
        "team_abbreviation", "team_id", "game_id",
        "pts_roll10", "opp_pts_roll10",
        "games_last_5_days", "games_last_7_days",
    ]
    tdf = pd.read_csv(
        team_features_path,
        usecols=lambda c: c in cols_to_load,
    )
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


# ── Per-player rolling features ────────────────────────────────────────────────

def _compute_player_rolling(group: pd.DataFrame, windows: list) -> pd.DataFrame:
    """
    Given a DataFrame for ONE player (sorted by game_date), compute all
    rolling features using shift(1) to avoid leakage.

    Includes: rolling means, stds, slopes, home/away splits, over-rates,
    and all role/usage features.
    """
    g = group.copy().sort_values("game_date")

    # ── Standard rolling means ──────────────────────────────────────────────
    for col in ROLL_STATS:
        if col not in g.columns:
            continue
        shifted = g[col].shift(1)
        for w in windows:
            g[f"{col}_roll{w}"] = shifted.rolling(window=w, min_periods=1).mean()

    # ── Rolling standard deviations ─────────────────────────────────────────
    for col in VOL_STATS:
        if col not in g.columns:
            continue
        shifted = g[col].shift(1)
        for w in windows:
            g[f"{col}_std{w}"] = shifted.rolling(window=w, min_periods=2).std().fillna(0)

    # ── Season expanding averages ────────────────────────────────────────────
    for col in ["pts", "min", "ast", "reb"]:
        if col in g.columns:
            g[f"{col}_season_avg"] = g[col].shift(1).expanding().mean()

    # ── MINUTES & ROLE ───────────────────────────────────────────────────────
    # minutes_last_5_avg / minutes_last_10_avg are captured in min_roll5/10 above.
    # Add explicit aliases for clarity:
    if "min_roll5" in g.columns:
        g["minutes_last_5_avg"]  = g["min_roll5"]
    if "min_roll10" in g.columns:
        g["minutes_last_10_avg"] = g["min_roll10"]

    # minutes_trend_5: slope of minutes over last 5 games
    if "min" in g.columns:
        g["minutes_trend_5"] = _rolling_slope(g["min"].values, window=5)

    # games_started_last_5: rolling sum of starter_flag over prior 5 games
    if "starter_flag" in g.columns:
        g["games_started_last_5"] = (
            g["starter_flag"].shift(1).rolling(5, min_periods=1).sum()
        )

    # ── USAGE & OPPORTUNITY ──────────────────────────────────────────────────
    # usage_rate_roll10: box-score-based usage proxy (no team denominator available)
    # usage_proxy = FGA + 0.44*FTA + TOV (possessions used, unnormalized)
    if all(c in g.columns for c in ["fga", "fta", "tov"]):
        g["usage_proxy_game"] = (
            g["fga"].fillna(0)
            + 0.44 * g["fta"].fillna(0)
            + g["tov"].fillna(0)
        )
        g["usage_rate_roll10"] = (
            g["usage_proxy_game"].shift(1).rolling(10, min_periods=1).mean()
        )

        # touches_roll10: proxy for ball-handling opportunities
        # = FGA + AST + TOV (creation/possession events)
        if "ast" in g.columns:
            g["touches_proxy_game"] = (
                g["fga"].fillna(0)
                + g["ast"].fillna(0)
                + g["tov"].fillna(0)
            )
            g["touches_roll10"] = (
                g["touches_proxy_game"].shift(1).rolling(10, min_periods=1).mean()
            )

    # fga_roll10: already computed above as part of ROLL_STATS
    # (alias for clarity)
    if "fga_roll10" in g.columns:
        g["fga_roll10"] = g["fga_roll10"]    # no-op, already present

    # assist_opportunities_roll10: proxy using ast (potential assists tracked
    # in tracking data but not standard box score; ast is the closest proxy)
    if "ast_roll10" in g.columns:
        g["assist_opportunities_roll10"] = g["ast_roll10"]

    # ── TREND & SLOPE FEATURES ───────────────────────────────────────────────
    for stat in SPLIT_STATS:     # pts, ast, reb
        if stat in g.columns:
            g[f"{stat}_trend_5"] = _rolling_slope(g[stat].values, window=5)

    # ── HOME / AWAY STAT SPLITS ──────────────────────────────────────────────
    if "is_home" in g.columns:
        for stat in SPLIT_STATS:
            if stat not in g.columns:
                continue
            home_result = pd.Series(np.nan, index=g.index)
            away_result = pd.Series(np.nan, index=g.index)

            for season, sg in g.groupby("season"):
                sg = sg.sort_values("game_date")
                home_vals = sg[stat].where(sg["is_home"] == 1)
                away_vals = sg[stat].where(sg["is_home"] == 0)
                home_avg  = home_vals.expanding().mean().shift(1).ffill().fillna(0)
                away_avg  = away_vals.expanding().mean().shift(1).ffill().fillna(0)
                home_result.loc[sg.index] = home_avg.values
                away_result.loc[sg.index] = away_avg.values

            g[f"{stat}_home_avg"] = home_result
            g[f"{stat}_away_avg"] = away_result

        # home_away_flag alias for is_home
        g["home_away_flag"] = g["is_home"]

    # ── OVER-RATE: fraction of last 10 games above rolling 20-game avg ───────
    for stat in SPLIT_STATS:
        roll20_col = f"{stat}_roll20"
        if stat not in g.columns or roll20_col not in g.columns:
            continue
        # above_mean: 1 if this game's result exceeded the pre-game rolling avg
        above = (g[stat] > g[roll20_col]).astype(int)
        # over_rate: fraction of the PRIOR 10 games where the player went over
        g[f"over_rate_{stat}_10"] = (
            above.shift(1).rolling(10, min_periods=1).mean()
        )

    return g


# ── Main builder ───────────────────────────────────────────────────────────────

def build_player_game_features(
    game_log_path:       str  = GAME_LOG_PATH,
    adv_stats_path:      str  = ADV_STATS_PATH,
    bio_stats_path:      str  = BIO_STATS_PATH,
    positions_path:      str  = POSITIONS_PATH,
    team_features_path:  str  = TEAM_FEATURES_PATH,
    injury_proxy_path:   str  = INJURY_PROXY_PATH,
    output_path:         str  = OUTPUT_PATH,
    roll_windows:        list = ROLL_WINDOWS,
    min_games:           int  = 5,
    start_season:        str  = "199697",
) -> pd.DataFrame:
    """
    Build pre-game rolling features for every player-game.

    Steps:
      1. Load player_game_logs
      2. Pre-compute cross-player lookup tables (starter_flag, team usage,
         positional defense, team pace)
      3. Process player-by-player: rolling features + per-player joins
      4. Flush results to CSV in batches to control memory

    Args:
        start_season: Only include seasons >= this value (default: 1996-97)
    """
    print("Loading player_game_logs...")
    load_cols = (
        ["season", "player_id", "player_name", "team_id", "team_abbreviation",
         "game_id", "game_date", "matchup", "wl"]
        + ROLL_STATS
        + [c for c in ["stl", "blk", "tov", "pf", "oreb", "dreb"]
           if c not in ROLL_STATS]
    )
    df = pd.read_csv(game_log_path, usecols=lambda c: c in load_cols or c in [
        "fga", "fta", "oreb",   # ensure these are loaded for usage/pace
    ])
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Limit to modern era
    df = df[df["season"].astype(str) >= start_season].copy()
    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    print(f"  Filtered to {start_season}+: {len(df):,} rows | "
          f"{df.player_id.nunique():,} players | {df.season.nunique()} seasons")

    # ── Basic context features ─────────────────────────────────────────────
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

    # ── Pre-compute cross-player lookup tables ─────────────────────────────
    print("Pre-computing cross-player lookup tables...")

    # 1. Starter flag (top-5 minutes per team per game)
    print("  Building starter flag table...")
    starter_table = _build_starter_flag_table(df)
    df = df.merge(starter_table, on=["player_id", "game_id"], how="left")
    df["starter_flag"] = df["starter_flag"].fillna(0).astype(int)

    # 2. Team usage concentration (top-2 usage share per team, rolling 10)
    print("  Building team usage table...")
    usage_table = _build_team_usage_table(df)
    df = df.merge(usage_table, on=["team_id", "game_id"], how="left")
    df["top2_usage_share"] = df["top2_usage_share"].fillna(0.5)

    # 3. Position map (player → G/F/C)
    # Uses official NBA positions from player_positions.csv if available;
    # falls back to height-based inference from player_bio_stats.csv.
    print("  Building position map...")
    pos_map = _build_position_map(
        positions_path=positions_path,
        bio_path=bio_stats_path,
    )
    df["position_bucket"] = df["player_id"].map(pos_map).fillna("F")

    # 4. Opponent positional defense (rolling 10-game avg pts/ast/reb/fg% by position)
    print("  Building opponent positional defense table...")
    opp_pos_def = _build_opp_positional_defense(df, pos_map)
    # Join using (opponent_abbr, game_id, position_bucket)
    df = df.merge(
        opp_pos_def,
        on=["opponent_abbr", "game_id", "position_bucket"],
        how="left",
    )
    pos_def_cols = [
        "opp_pts_allowed_to_position_roll10",
        "opp_ast_allowed_to_position_roll10",
        "opp_reb_allowed_to_position_roll10",
        "opp_fg_pct_allowed_to_position_roll10",
    ]
    for c in pos_def_cols:
        df[c] = df[c].fillna(df[c].median())

    # 5. Team pace (rolling 10-game estimate from player box scores)
    print("  Building team pace table...")
    pace_table = _build_team_pace_table(df)
    # Merge only the columns we need — drop team_abbreviation from the join
    # to prevent pandas from creating team_abbreviation_x / team_abbreviation_y
    # (df already has team_abbreviation; pace_table also contains it)
    df = df.merge(
        pace_table[["team_id", "game_id", "team_pace_roll10"]],
        on=["team_id", "game_id"],
        how="left",
    )

    # Opponent pace: self-join on (team_abbreviation == opponent_abbr) + game_id
    opp_pace = pace_table[["team_abbreviation", "game_id", "team_pace_roll10"]].copy()
    opp_pace.columns = ["opponent_abbr", "game_id", "opp_pace_roll10"]
    df = df.merge(opp_pace, on=["opponent_abbr", "game_id"], how="left")
    df["game_pace_estimate"] = (
        (df["team_pace_roll10"].fillna(0) + df["opp_pace_roll10"].fillna(0)) / 2
    )

    # 6. Implied team/game points from team_game_features
    print("  Loading team-level implied points context...")
    team_ctx = _load_team_context(team_features_path)
    if team_ctx is not None:
        # Drop team_id from team_ctx before merging — df already has team_id and
        # merging on (team_abbreviation, game_id) is sufficient to identify the row.
        # Keeping team_id in team_ctx would cause pandas to emit team_id_x / team_id_y
        # columns requiring fragile post-merge cleanup.
        team_ctx_clean = team_ctx.drop(columns=["team_id"], errors="ignore")
        df = df.merge(
            team_ctx_clean.rename(columns={"team_abbreviation": "team_abbr_for_join"}),
            left_on=["team_abbreviation", "game_id"],
            right_on=["team_abbr_for_join", "game_id"],
            how="left",
        )
        # Clean up the temporary join key
        if "team_abbr_for_join" in df.columns:
            df = df.drop(columns=["team_abbr_for_join"])

        df["implied_team_points"] = df["pts_roll10"].fillna(df["pts_roll10"].median())
        df["implied_game_total"]  = (
            df["pts_roll10"].fillna(0) + df["opp_pts_roll10"].fillna(0)
        )
    else:
        df["implied_team_points"] = np.nan
        df["implied_game_total"]  = np.nan

    # 7. Injury context from injury proxy features
    print("  Loading injury proxy features...")
    if os.path.exists(injury_proxy_path):
        inj = pd.read_csv(
            injury_proxy_path,
            usecols=lambda c: c in [
                "team_id", "game_id",
                "missing_minutes", "missing_usg_pct",
                "star_player_out", "rotation_availability",
            ],
        )
        inj = inj.rename(columns={
            "missing_minutes":     "missing_teammate_minutes",
            "missing_usg_pct":     "missing_teammate_usage_pct",
        })
        df = df.merge(inj, on=["team_id", "game_id"], how="left")
        for c in ["missing_teammate_minutes", "missing_teammate_usage_pct",
                  "star_player_out", "rotation_availability"]:
            if c in df.columns:
                df[c] = df[c].fillna(0 if c != "rotation_availability" else 1.0)

        # role_expansion_flag: 1 if a star is out OR >10% usage is missing
        df["role_expansion_flag"] = (
            (df.get("star_player_out", 0) == 1)
            | (df.get("missing_teammate_usage_pct", 0) >= 0.10)
        ).astype(int)
    else:
        print(f"  Warning: {injury_proxy_path} not found. Injury features will be NaN.")
        for c in ["missing_teammate_minutes", "missing_teammate_usage_pct",
                  "star_player_out", "rotation_availability", "role_expansion_flag"]:
            df[c] = np.nan

    # ── Load supplementary season-level tables ─────────────────────────────
    print("Loading advanced stats...")
    adv = pd.read_csv(adv_stats_path)
    adv = adv[ADV_CONTEXT_COLS].drop_duplicates(subset=["player_id", "season"])

    opp_defense = _load_opponent_defense(team_features_path)
    bio         = _load_player_bio(bio_stats_path)

    # ── Process player-by-player ───────────────────────────────────────────
    print("Computing rolling features per player...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Defragment after many joins
    df = df.copy()

    player_ids = df["player_id"].unique()
    total      = len(player_ids)
    results    = []
    batch_size = 500

    for i, pid in enumerate(player_ids):
        pdata = df[df["player_id"] == pid].copy()
        if len(pdata) < min_games:
            continue

        # Compute per-player rolling features (including trends, home/away, over-rates)
        pdata = _compute_player_rolling(pdata, roll_windows)
        results.append(pdata)

        # Flush to disk every batch_size players to control memory
        if (i + 1) % batch_size == 0 or (i + 1) == total:
            chunk = pd.concat(results, ignore_index=True)

            # Join season-level advanced stats
            chunk = chunk.merge(adv, on=["player_id", "season"], how="left")

            # Join opponent defensive context (team-game level)
            if opp_defense is not None:
                chunk = chunk.merge(
                    opp_defense,
                    on=["opponent_abbr", "game_id"],
                    how="left",
                )

            # Join player bio stats (age, height, weight)
            if bio is not None:
                chunk = chunk.merge(bio, on=["player_id", "season"], how="left")

            # Rename for reporting clarity (player's position proxy)
            if "player_height_inches_x" in chunk.columns:
                chunk = chunk.rename(columns={
                    "player_height_inches_x": "player_height_inches",
                    "age_x": "age",
                })
                for dup in ["player_height_inches_y", "age_y", "player_weight_y"]:
                    if dup in chunk.columns:
                        chunk = chunk.drop(columns=[dup])

            # Add era labels
            chunk = label_eras(chunk, season_col="season")

            # Write header only on first batch
            write_header = (i + 1) <= batch_size
            chunk.to_csv(
                output_path,
                mode="w" if write_header else "a",
                header=write_header,
                index=False,
            )
            print(f"  Processed {i+1:,}/{total:,} players, "
                  f"flushed {len(chunk):,} rows")
            results = []
            del chunk

    # ── Verify output ──────────────────────────────────────────────────────
    final      = pd.read_csv(output_path, nrows=5)
    total_rows = sum(1 for _ in open(output_path, encoding="utf-8")) - 1
    print(f"\nSaved {total_rows:,} rows × {len(final.columns)} cols → {output_path}")

    # Print feature coverage summary
    new_features = [
        "minutes_last_5_avg", "minutes_last_10_avg", "minutes_trend_5",
        "games_started_last_5", "starter_flag",
        "usage_rate_roll10", "fga_roll10", "touches_roll10",
        "assist_opportunities_roll10", "top2_usage_share",
        "opp_pts_allowed_to_position_roll10", "opp_ast_allowed_to_position_roll10",
        "opp_reb_allowed_to_position_roll10", "opp_fg_pct_allowed_to_position_roll10",
        "team_pace_roll10", "opp_pace_roll10", "game_pace_estimate",
        "implied_team_points", "implied_game_total",
        "missing_teammate_minutes", "missing_teammate_usage_pct", "role_expansion_flag",
        "days_rest", "is_back_to_back", "home_away_flag",
        "pts_home_avg", "pts_away_avg", "ast_home_avg", "ast_away_avg",
        "reb_home_avg", "reb_away_avg",
        "pts_std10", "ast_std10", "reb_std10",
        "over_rate_pts_10", "over_rate_ast_10", "over_rate_reb_10",
        "pts_trend_5", "ast_trend_5", "reb_trend_5",
    ]
    all_cols = set(pd.read_csv(output_path, nrows=0).columns)
    present  = [f for f in new_features if f in all_cols]
    missing  = [f for f in new_features if f not in all_cols]
    print(f"\nNew feature coverage: {len(present)}/{len(new_features)} present")
    if missing:
        print(f"  Missing from output: {missing}")

    return final


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    build_player_game_features()
    print("\nPlayer feature engineering complete.")
