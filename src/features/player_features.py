"""
Player Game Feature Engineering
=================================
Builds rolling, lag, and contextual features from player_game_logs for use in
player performance prediction models.

Each output row is one player-game enriched with pre-game rolling stats
(shift-1 to prevent data leakage).

Enhancements:
  - Opponent + own-team context from team_game_features (defense, pace proxy,
    injury availability).
  - Optional season priors from player scoring + clutch tables when available.
  - Derived features capturing form, per-minute production, and opportunity.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.features.era_labels import label_eras


# ── Config ─────────────────────────────────────────────────────────────────────

GAME_LOG_PATH = "data/processed/player_game_logs.csv"
ADV_STATS_PATH = "data/processed/player_stats_advanced.csv"
BIO_STATS_PATH = "data/processed/player_bio_stats.csv"
SCORING_STATS_PATH = "data/processed/player_stats_scoring.csv"
CLUTCH_STATS_PATH = "data/processed/player_stats_clutch.csv"
TEAM_FEATURES_PATH = "data/features/team_game_features.csv"
OUTPUT_PATH = "data/features/player_game_features.csv"

ROLL_WINDOWS = [5, 10, 20]

ROLL_STATS = [
    "pts", "reb", "ast", "stl", "blk", "tov",
    "min", "fg_pct", "fg3_pct", "ft_pct", "plus_minus",
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
    return matchup.str.split(r" vs\. | @ ", regex=True).str[-1].str.strip()


def _compute_player_rolling(group: pd.DataFrame, windows: list) -> pd.DataFrame:
    """Compute shifted rolling features for one player's game history."""
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
        g[f"{col}_season_avg"] = g[col].shift(1).expanding().mean()

    # Recent-vs-long trend features
    g["pts_form_delta"] = g["pts_roll5"] - g["pts_roll20"]
    g["ast_form_delta"] = g["ast_roll5"] - g["ast_roll20"]
    g["reb_form_delta"] = g["reb_roll5"] - g["reb_roll20"]
    g["min_form_delta"] = g["min_roll5"] - g["min_roll20"]

    # Per-minute production trends
    g["pts_per_min_roll10"] = g["pts_roll10"] / g["min_roll10"].replace(0, np.nan)
    g["reb_per_min_roll10"] = g["reb_roll10"] / g["min_roll10"].replace(0, np.nan)
    g["ast_per_min_roll10"] = g["ast_roll10"] / g["min_roll10"].replace(0, np.nan)

    return g


def _load_advanced(adv_stats_path: str) -> pd.DataFrame:
    print("Loading advanced stats...")
    adv = pd.read_csv(adv_stats_path)
    return adv[ADV_CONTEXT_COLS].drop_duplicates(subset=["player_id", "season"])


def _load_player_bio(bio_path: str = BIO_STATS_PATH) -> pd.DataFrame:
    if not os.path.exists(bio_path):
        print(f"  Warning: {bio_path} not found. Skipping bio features.")
        return None
    print("  Loading player bio stats...")
    bio = pd.read_csv(bio_path, usecols=BIO_COLS)
    return bio.drop_duplicates(subset=["player_id", "season"])


def _load_optional_priors(path: str, value_cols: list, prefix: str) -> pd.DataFrame:
    """Load optional season-level prior stats when available."""
    if not os.path.exists(path):
        print(f"  Optional table missing: {path} (skipping {prefix} priors)")
        return None

    raw = pd.read_csv(path)
    keep = [c for c in ["player_id", "season"] + value_cols if c in raw.columns]
    if len(keep) <= 2:
        return None

    out = raw[keep].copy().drop_duplicates(subset=["player_id", "season"])
    rename = {c: f"{prefix}_{c}" for c in keep if c not in ["player_id", "season"]}
    return out.rename(columns=rename)


def _load_team_context(team_features_path: str = TEAM_FEATURES_PATH) -> tuple:
    """
    Load team-game context for both opponent and player's own team.

    Returns:
      (opp_context_df, own_context_df)
    """
    if not os.path.exists(team_features_path):
        print("  Warning: team_game_features.csv missing; skipping team context joins.")
        return None, None

    print("  Loading opponent and team context from team_game_features...")
    tdf = pd.read_csv(team_features_path)

    base_cols = [
        "team_abbreviation", "game_id",
        "opp_pts_roll10", "opp_pts_roll20",
        "plus_minus_roll10", "plus_minus_roll20",
        "sos_roll10", "sos_roll20",
        "is_back_to_back", "days_rest",
        "missing_minutes", "missing_usg_pct", "rotation_availability", "star_player_out",
    ]
    keep_cols = [c for c in base_cols if c in tdf.columns]
    slim = tdf[keep_cols].copy()

    opp_context = slim.rename(columns={
        "team_abbreviation": "opponent_abbr",
        "opp_pts_roll10": "opp_pts_allowed_roll10",
        "opp_pts_roll20": "opp_pts_allowed_roll20",
        "plus_minus_roll10": "opp_net_rating_roll10",
        "plus_minus_roll20": "opp_net_rating_roll20",
        "sos_roll10": "opp_sos_roll10",
        "sos_roll20": "opp_sos_roll20",
        "is_back_to_back": "opp_is_back_to_back",
        "days_rest": "opp_days_rest",
        "missing_minutes": "opp_missing_minutes",
        "missing_usg_pct": "opp_missing_usg_pct",
        "rotation_availability": "opp_rotation_availability",
        "star_player_out": "opp_star_player_out",
    })

    own_context = slim.rename(columns={
        "team_abbreviation": "team_abbreviation",
        "opp_pts_roll10": "team_opp_pts_allowed_roll10",
        "opp_pts_roll20": "team_opp_pts_allowed_roll20",
        "plus_minus_roll10": "team_net_rating_roll10",
        "plus_minus_roll20": "team_net_rating_roll20",
        "sos_roll10": "team_sos_roll10",
        "sos_roll20": "team_sos_roll20",
        "is_back_to_back": "team_is_back_to_back",
        "days_rest": "team_days_rest",
        "missing_minutes": "team_missing_minutes",
        "missing_usg_pct": "team_missing_usg_pct",
        "rotation_availability": "team_rotation_availability",
        "star_player_out": "team_star_player_out",
    })

    return opp_context, own_context


def _add_derived_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create cross-table features from player + team context."""
    out = df.copy()

    if {"team_missing_minutes", "team_missing_usg_pct"}.issubset(out.columns):
        out["role_opportunity_index"] = (
            out["team_missing_minutes"].fillna(0) + 0.8 * out["team_missing_usg_pct"].fillna(0)
        )

    if {"opp_pts_allowed_roll20", "pts_roll10"}.issubset(out.columns):
        out["scoring_matchup_edge"] = out["pts_roll10"] - out["opp_pts_allowed_roll20"] * 0.1

    if {"opp_net_rating_roll20", "plus_minus_roll10"}.issubset(out.columns):
        out["impact_matchup_edge"] = out["plus_minus_roll10"] - out["opp_net_rating_roll20"]

    if {"team_days_rest", "opp_days_rest"}.issubset(out.columns):
        out["rest_advantage"] = out["team_days_rest"] - out["opp_days_rest"]

    return out


# ── Main builder ───────────────────────────────────────────────────────────────

def build_player_game_features(
    game_log_path: str = GAME_LOG_PATH,
    adv_stats_path: str = ADV_STATS_PATH,
    bio_stats_path: str = BIO_STATS_PATH,
    team_features_path: str = TEAM_FEATURES_PATH,
    output_path: str = OUTPUT_PATH,
    roll_windows: list = ROLL_WINDOWS,
    min_games: int = 5,
    start_season: str = "199697",
) -> pd.DataFrame:
    """Build enriched pre-game features for every player-game."""
    print("Loading player_game_logs...")
    df = pd.read_csv(
        game_log_path,
        usecols=[
            "season", "player_id", "player_name", "team_id", "team_abbreviation",
            "game_id", "game_date", "matchup", "wl",
        ] + ROLL_STATS + ["stl", "blk", "tov", "pf"],
    )
    df["game_date"] = pd.to_datetime(df["game_date"])

    df = df[df["season"].astype(str) >= start_season].copy()
    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    print(f"  Filtered to {start_season}+: {len(df):,} rows | "
          f"{df.player_id.nunique():,} players | {df.season.nunique()} seasons")

    df["is_home"] = _parse_home_away(df["matchup"])
    df["opponent_abbr"] = _extract_opponent_abbr(df["matchup"])
    df["win"] = (df["wl"] == "W").astype(int)

    df["days_rest"] = (
        df.groupby("player_id")["game_date"].diff().dt.days.fillna(7).clip(upper=14)
    )
    df["is_back_to_back"] = (df["days_rest"] <= 1).astype(int)
    df["player_season_game_num"] = df.groupby(["player_id", "season"]).cumcount() + 1

    adv = _load_advanced(adv_stats_path)
    bio = _load_player_bio(bio_stats_path)
    scoring_priors = _load_optional_priors(
        SCORING_STATS_PATH,
        value_cols=["pct_fga_3pt", "pct_pts_2pt_mr", "pct_pts_fast_break", "pct_pts_off_to", "points_in_paint_per_game"],
        prefix="scoring",
    )
    clutch_priors = _load_optional_priors(
        CLUTCH_STATS_PATH,
        value_cols=["clutch_pts", "clutch_ast", "clutch_fga", "clutch_fg_pct"],
        prefix="clutch",
    )

    opp_context, own_context = _load_team_context(team_features_path)

    print("Computing rolling features per player...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    player_ids = df["player_id"].unique()
    total = len(player_ids)
    results = []
    batch_size = 500

    for i, pid in enumerate(player_ids):
        pdata = df[df["player_id"] == pid].copy()
        if len(pdata) < min_games:
            continue
        pdata = _compute_player_rolling(pdata, roll_windows)
        results.append(pdata)

        if (i + 1) % batch_size == 0 or (i + 1) == total:
            chunk = pd.concat(results, ignore_index=True)

            chunk = chunk.merge(adv, on=["player_id", "season"], how="left")
            if bio is not None:
                chunk = chunk.merge(bio, on=["player_id", "season"], how="left")
            if scoring_priors is not None:
                chunk = chunk.merge(scoring_priors, on=["player_id", "season"], how="left")
            if clutch_priors is not None:
                chunk = chunk.merge(clutch_priors, on=["player_id", "season"], how="left")

            if opp_context is not None:
                chunk = chunk.merge(opp_context, on=["opponent_abbr", "game_id"], how="left")
            if own_context is not None:
                chunk = chunk.merge(own_context, on=["team_abbreviation", "game_id"], how="left")

            chunk = _add_derived_context_features(chunk)
            chunk = label_eras(chunk, season_col="season")

            write_header = (i + 1) <= batch_size
            chunk.to_csv(
                output_path,
                mode="w" if write_header else "a",
                header=write_header,
                index=False,
            )
            print(f"  Processed {i + 1:,}/{total:,} players, flushed {len(chunk):,} rows")
            results = []
            del chunk

    final = pd.read_csv(output_path, nrows=5)
    total_rows = sum(1 for _ in open(output_path, "r", encoding="utf-8")) - 1
    print(f"\nSaved {total_rows:,} rows × {len(final.columns)} cols → {output_path}")
    return final


if __name__ == "__main__":
    build_player_game_features()
    print("\nPlayer feature engineering complete.")
