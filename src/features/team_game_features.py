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

New in v3:
  - REST & FATIGUE: games_last_5_days, games_last_7_days
  - TREND & VOLATILITY: net_rating_trend_5, off/def_rating_trend_5, net_rating_std_10
  - HOME COURT STRENGTH: home_net_rating_season, away_net_rating_season
  - STYLE MISMATCH: three_rate, opponent_three_allowed_rate, three_style_mismatch,
                    rebounding_edge (oreb vs opp dreb)
  - MOTIVATION / PSYCHOLOGICAL: revenge_game, blowout_loss_last_game,
                                  close_playoff_race
  - Matchup-level diff_ columns for all new features above

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

DATA_PATH      = "data/processed/team_game_logs.csv"
STANDINGS_PATH = "data/processed/standings.csv"
OUTPUT_PATH    = "data/features/team_game_features.csv"
ROLL_WINDOWS   = [5, 10, 20]   # rolling window sizes (games)

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


def _games_in_window(group: pd.DataFrame, window_days: int) -> pd.Series:
    """
    Count games played in the N calendar days BEFORE each game (exclusive).

    For game on date D, counts games in (D - window_days, D).
    Uses numpy searchsorted for O(n log n) efficiency on sorted date arrays.

    The group must be sorted by game_date ascending.
    """
    dates = group["game_date"].values   # numpy datetime64, sorted ascending
    result = np.zeros(len(dates), dtype=int)
    for i in range(1, len(dates)):
        cutoff = dates[i] - np.timedelta64(window_days, "D")
        # searchsorted on dates[:i] (all prior games) to find the left boundary
        lo = np.searchsorted(dates[:i], cutoff, side="right")
        result[i] = i - lo   # games between cutoff and current (exclusive)
    return pd.Series(result, index=group.index)


def _rolling_slope(values: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Compute rolling linear regression slope using shift-1 semantics.

    At position i, fits a line through values[max(0, i-window) : i]
    (i.e., only prior games — no leakage).
    Returns 0.0 for positions with fewer than 2 prior values,
    and NaN at position 0 (no prior games at all).
    """
    n = len(values)
    slopes = np.full(n, np.nan)
    for i in range(1, n):
        start = max(0, i - window)
        y = values[start:i]
        if len(y) < 2:
            slopes[i] = 0.0
            continue
        x = np.arange(len(y), dtype=float)
        slopes[i] = np.polyfit(x, y, 1)[0]
    slopes[0] = 0.0   # first game: no prior games, neutral slope
    return slopes


def _compute_home_away_net_rating(group: pd.DataFrame) -> tuple:
    """
    Season-to-date average net rating in home-only / away-only games.

    For each game G entering the season, computes the cumulative mean of
    plus_minus across all PRIOR home (or away) games in the same season.
    ffill() propagates the last known value on the opposite game type.
    fillna(0) for the neutral start-of-season case.

    Returns (home_series, away_series) with the same index as `group`.
    group must be a single team's rows, sorted by game_date.
    """
    home_result = pd.Series(np.nan, index=group.index)
    away_result = pd.Series(np.nan, index=group.index)

    for season, sg in group.groupby("season"):
        sg = sg.sort_values("game_date")

        # Mask plus_minus by game type
        home_pm = sg["plus_minus"].where(sg["is_home"] == 1)
        away_pm = sg["plus_minus"].where(sg["is_home"] == 0)

        # expanding().mean() skips NaN → cumulative mean over home/away games only
        # shift(1) → only prior games (no leakage)
        # ffill() → carries the value forward through the opposite game type
        # fillna(0) → neutral at start of season before any home/away game
        home_cum = home_pm.expanding().mean().shift(1).ffill().fillna(0)
        away_cum = away_pm.expanding().mean().shift(1).ffill().fillna(0)

        home_result.loc[sg.index] = home_cum.values
        away_result.loc[sg.index] = away_cum.values

    return home_result, away_result


def _compute_close_playoff_race(df: pd.DataFrame, conf_map: pd.Series) -> pd.Series:
    """
    Flag (1/0) whether a team is within 2 seeds of the 6th or 10th place
    in their conference standings entering each game.

    Uses cum_wins/cum_losses (already pre-game via shift-cumsum) and
    builds rolling conference standings date by date per (season, conference).

    The 6th seed boundary is the traditional playoff cutoff; the 10th seed
    boundary covers the modern play-in tournament bubble.
    """
    work = df[["team_id", "season", "game_date", "cum_wins", "cum_losses"]].copy()
    work["conference"] = work["team_id"].map(conf_map)
    work["gp"]         = work["cum_wins"] + work["cum_losses"]
    work["win_pct"]    = (work["cum_wins"] / work["gp"].replace(0, np.nan)).fillna(0.5)
    work = work.sort_values(["season", "conference", "game_date"])

    result = pd.Series(0, index=df.index, dtype=int)

    for (season, conf), conf_grp in work.groupby(["season", "conference"]):
        if pd.isna(conf) or conf == "":
            continue

        conf_grp_sorted = conf_grp.sort_values("game_date")

        # Build per-team history as numpy arrays for fast date lookups
        team_histories = {}
        for tid, tgrp in conf_grp_sorted.groupby("team_id"):
            team_histories[tid] = tgrp[["game_date", "win_pct"]].to_numpy()

        all_tids = list(team_histories.keys())

        for date in conf_grp_sorted["game_date"].unique():
            # Get most recent record for each team on or before this date
            latest = {}
            for tid in all_tids:
                hist = team_histories[tid]
                mask = hist[:, 0] <= date
                if mask.any():
                    latest[tid] = float(hist[mask, 1][-1])

            if len(latest) < 6:
                continue

            n_teams     = len(latest)
            sorted_tids = sorted(latest, key=lambda t: latest[t], reverse=True)
            ranks       = {tid: i + 1 for i, tid in enumerate(sorted_tids)}

            # Flag teams playing today that are close to 6th or 10th seed
            playing_today = conf_grp_sorted[conf_grp_sorted["game_date"] == date]
            playing_ranks = playing_today["team_id"].map(ranks)

            close_to_6th  = (playing_ranks - 6).abs() <= 2
            close_to_10th = (n_teams >= 10) & ((playing_ranks - 10).abs() <= 2)
            close_mask    = close_to_6th | close_to_10th

            result.loc[playing_today.index[close_mask.values]] = 1

    return result


# ── Main builder ───────────────────────────────────────────────────────────────

def build_team_game_features(
    data_path:      str  = DATA_PATH,
    standings_path: str  = STANDINGS_PATH,
    output_path:    str  = OUTPUT_PATH,
    roll_windows:   list = ROLL_WINDOWS,
) -> pd.DataFrame:
    """
    Build pre-game rolling features for every team-game.

    Steps:
      1. Load and prep team game logs
      2. Compute opp_pts and three_rate_raw
      3. Sort by team + game_date (chronological)
      4. Compute rest/fatigue features (days_rest, games_last_N_days)
      5. Compute rolling stats (shift-1 to prevent leakage)
      6. Compute cumulative stats and SOS
      7. Compute style mismatch features (self-join for opponent stats)
      8. Compute trend and volatility features
      9. Compute home/away split net rating
      10. Compute motivation/psychological features
      11. Attach era labels and injury proxy
      12. Save to data/features/team_game_features.csv

    Returns:
        pd.DataFrame with one row per team-game.
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
    df["opp_pts"] = df["pts"] - df["plus_minus"]

    # ── Three-point attempt rate (raw, per game) ──────────────────────────────
    if "fg3a" in df.columns and "fga" in df.columns:
        df["three_rate_raw"] = (
            df["fg3a"] / df["fga"].replace(0, np.nan)
        ).fillna(0)
    else:
        print("  Warning: fg3a or fga not in data — three_rate features will be NaN")
        df["three_rate_raw"] = np.nan

    # ── Days rest (days since last game for this team) ────────────────────────
    df["days_rest"] = (
        df.groupby("team_id")["game_date"]
        .diff()
        .dt.days
        .fillna(7)    # first game of season gets a neutral 7-day rest
        .clip(upper=14)
    )
    df["rest_days"]       = df["days_rest"]           # alias requested by user
    df["is_back_to_back"] = (df["days_rest"] <= 1).astype(int)

    # ── Games in last 5 / 7 calendar days (fatigue load) ─────────────────────
    print("Computing fatigue load features...")
    for window_days in [5, 7]:
        col = f"games_last_{window_days}_days"
        df[col] = (
            df.groupby("team_id", group_keys=False)
            .apply(lambda g: _games_in_window(g, window_days), include_groups=False)
            .values
        )

    # ── Cumulative wins/games in season (context) ─────────────────────────────
    df["season_game_num"] = df.groupby(["team_id", "season"]).cumcount() + 1

    # ── Rolling stats per team ────────────────────────────────────────────────
    print("Computing rolling features...")

    for window in roll_windows:
        group = df.groupby("team_id", group_keys=False)

        for stat in ROLL_STATS:
            if stat in df.columns:
                col_name = f"{stat}_roll{window}"
                df[col_name] = group.apply(
                    lambda g: _rolling_mean_shift(g, stat, window),
                    include_groups=False,
                ).values

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
    print("Computing strength of schedule...")
    opp_strength = df[["team_abbreviation", "game_id", "cum_win_pct"]].copy()
    opp_strength.columns = ["opponent_abbr", "game_id", "opp_pre_game_win_pct"]

    df = df.merge(opp_strength, on=["opponent_abbr", "game_id"], how="left")

    sos_group = df.groupby("team_id", group_keys=False)
    for window in [10, 20]:
        df[f"sos_roll{window}"] = sos_group.apply(
            lambda g: g["opp_pre_game_win_pct"].shift(1).rolling(window, min_periods=1).mean(),
            include_groups=False,
        ).values

    # ── Style mismatch: join opponent's three_rate + dreb ────────────────────
    # Use self-join keyed on (opponent_abbr, game_id) to get the opposing
    # team's fg3a rate and dreb in the same game (raw, pre-rolling).
    print("Computing style mismatch features...")
    opp_style = df[["team_abbreviation", "game_id", "three_rate_raw", "dreb"]].copy()
    opp_style.columns = [
        "opponent_abbr", "game_id", "opp_three_rate_game", "opp_dreb_game"
    ]
    df = df.merge(opp_style, on=["opponent_abbr", "game_id"], how="left")

    style_group = df.groupby("team_id", group_keys=False)

    # Rolling own three_rate (fraction of FGA from 3PT range, prior 20 games)
    df["three_rate"] = style_group.apply(
        lambda g: g["three_rate_raw"].shift(1).rolling(20, min_periods=1).mean(),
        include_groups=False,
    ).values

    # Rolling opponent three_rate allowed (how much 3PT shooting team's defense allows)
    df["opponent_three_allowed_rate"] = style_group.apply(
        lambda g: g["opp_three_rate_game"].shift(1).rolling(20, min_periods=1).mean(),
        include_groups=False,
    ).values

    # Rolling opponent defensive rebound rate (opponent's dreb per game, last 20)
    df["opp_dreb_roll20"] = style_group.apply(
        lambda g: g["opp_dreb_game"].shift(1).rolling(20, min_periods=1).mean(),
        include_groups=False,
    ).values

    # Composite style features
    df["three_style_mismatch"] = df["three_rate"] - df["opponent_three_allowed_rate"]
    df["rebounding_edge"]      = df["oreb_roll20"] - df["opp_dreb_roll20"]

    # ── Trend features (rolling linear regression slope) ──────────────────────
    print("Computing trend & volatility features...")

    for stat_col, feat_name in [
        ("plus_minus", "net_rating_trend_5"),
        ("pts",        "off_rating_trend_5"),
        ("opp_pts",    "def_rating_trend_5"),
    ]:
        if stat_col not in df.columns:
            continue
        df[feat_name] = (
            df.groupby("team_id", group_keys=False)
            .apply(
                lambda g, sc=stat_col: pd.Series(
                    _rolling_slope(g[sc].values, window=5),
                    index=g.index,
                ),
                include_groups=False,
            )
            .values
        )

    # Rolling standard deviation of net rating (consistency measure)
    df["net_rating_std_10"] = (
        df.groupby("team_id", group_keys=False)
        .apply(
            lambda g: g["plus_minus"].shift(1).rolling(10, min_periods=2).std(),
            include_groups=False,
        )
        .values
    )
    df["net_rating_std_10"] = df["net_rating_std_10"].fillna(0)

    # ── Home/away split net rating (season-to-date) ───────────────────────────
    print("Computing home/away net rating splits...")
    home_ratings = []
    away_ratings = []

    for tid, tgrp in df.groupby("team_id"):
        hr, ar = _compute_home_away_net_rating(tgrp)
        home_ratings.append(hr)
        away_ratings.append(ar)

    df["home_net_rating_season"] = pd.concat(home_ratings).reindex(df.index)
    df["away_net_rating_season"] = pd.concat(away_ratings).reindex(df.index)

    # Defragment the DataFrame before the motivation section (avoids PerformanceWarning
    # from many prior column insertions) and resets the memory layout for faster access.
    df = df.copy()

    # ── Motivation / Psychological features ───────────────────────────────────
    print("Computing motivation & psychological features...")

    # Revenge game: 1 if team lost their most recent prior matchup vs this opponent
    # Sort by (team_id, opponent_abbr, game_date) then shift within matchup pairs
    df_matchup = df[["team_id", "opponent_abbr", "game_date", "win"]].copy()
    df_matchup = df_matchup.sort_values(
        ["team_id", "opponent_abbr", "game_date"]
    ).reset_index()   # preserves original index in 'index' column
    prev_win = df_matchup.groupby(["team_id", "opponent_abbr"])["win"].shift(1)
    df_matchup["revenge_game"] = ((prev_win == 0).astype(int)).fillna(0)
    df["revenge_game"] = (
        df_matchup.set_index("index")["revenge_game"]
        .reindex(df.index)
        .fillna(0)
        .astype(int)
    )

    # Blowout loss last game: 1 if previous game was a loss by 20+ points
    df["blowout_loss_last_game"] = (
        df.groupby("team_id")["plus_minus"]
        .shift(1)
        .le(-20)
        .astype(int)
        .fillna(0)
        .values
    )

    # Close playoff race: 1 if within 2 seeds of 6th or 10th in conference
    if os.path.exists(standings_path):
        print("Computing close playoff race flags...")
        try:
            conf_df  = pd.read_csv(standings_path, usecols=["team_id", "conference"])
            conf_map = (
                conf_df
                .drop_duplicates("team_id", keep="last")
                .set_index("team_id")["conference"]
            )
            df["close_playoff_race"] = _compute_close_playoff_race(df, conf_map)
        except Exception as e:
            print(f"  Warning: could not compute close_playoff_race ({e}). Defaulting to 0.")
            df["close_playoff_race"] = 0
    else:
        print(f"  Warning: {standings_path} not found — close_playoff_race defaulted to 0")
        df["close_playoff_race"] = 0

    # ── Drop raw game stats (they'd leak the result) ──────────────────────────
    id_cols = [
        "season_id", "season", "team_id", "team_abbreviation", "team_name",
        "game_id", "game_date", "matchup", "wl",
    ]
    context_cols = [
        "is_home", "opponent_abbr",
        # Rest & fatigue
        "days_rest", "rest_days", "is_back_to_back",
        "games_last_5_days", "games_last_7_days",
        # Season context
        "season_game_num", "cum_wins", "cum_losses", "cum_win_pct", "win",
        # SOS
        "sos_roll10", "sos_roll20",
        # Trend & volatility
        "net_rating_trend_5", "off_rating_trend_5", "def_rating_trend_5",
        "net_rating_std_10",
        # Home/away splits
        "home_net_rating_season", "away_net_rating_season",
        # Style mismatch
        "three_rate", "opponent_three_allowed_rate",
        "three_style_mismatch", "rebounding_edge",
        # Motivation
        "revenge_game", "blowout_loss_last_game", "close_playoff_race",
    ]
    # Filter to only include columns that exist in df
    context_cols = [c for c in context_cols if c in df.columns]

    roll_cols = [c for c in df.columns if "_roll" in c and "sos_roll" not in c]

    output = df[id_cols + context_cols + roll_cols].copy()

    # ── Add era labels ────────────────────────────────────────────────────────
    print("Labeling eras...")
    output = label_eras(output, season_col="season")

    # ── Join injury proxy features ────────────────────────────────────────────
    try:
        from src.features.injury_proxy import build_injury_proxy_features
        print("Building injury proxy features...")
        injury_df = build_injury_proxy_features()
        output = output.merge(injury_df, on=["team_id", "game_id"], how="left")
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
    output_path:   str = "data/features/game_matchup_features.csv",
) -> pd.DataFrame:
    """
    Merge team-level features into a single game-level row for modeling.

    For each game, we get:
      - home_team features (prefixed 'home_')
      - away_team features (prefixed 'away_')
      - target: home_win (1 if home team won)
      - diff_ columns (home minus away) for key stats

    The differential features give the model the direct gap between teams
    rather than requiring it to infer the gap from two separate columns.
    """
    print("Loading team game features...")
    df = pd.read_csv(features_path)
    df["game_date"] = pd.to_datetime(df["game_date"])

    home = df[df["is_home"] == 1].copy()
    away = df[df["is_home"] == 0].copy()

    # Exclude sos_roll* here because they are already included in context_cols below.
    # Including them in both lists would create duplicate columns → ValueError on subtraction.
    roll_cols    = [c for c in df.columns if "_roll" in c and "sos_roll" not in c]
    injury_cols  = [c for c in [
        "missing_minutes", "missing_usg_pct",
        "rotation_availability", "star_player_out",
        "n_missing_rotation",
    ] if c in df.columns]

    context_cols = [
        "days_rest", "rest_days", "is_back_to_back",
        "games_last_5_days", "games_last_7_days",
        "season_game_num", "cum_wins", "cum_losses", "cum_win_pct",
        "sos_roll10", "sos_roll20",
        # Trend & volatility
        "net_rating_trend_5", "off_rating_trend_5", "def_rating_trend_5",
        "net_rating_std_10",
        # Home/away splits
        "home_net_rating_season", "away_net_rating_season",
        # Style mismatch
        "three_rate", "opponent_three_allowed_rate",
        "three_style_mismatch", "rebounding_edge",
        # Motivation
        "revenge_game", "blowout_loss_last_game", "close_playoff_race",
    ] + injury_cols

    # Keep only context_cols that actually exist in df
    context_cols = [c for c in context_cols if c in df.columns]
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
    print("Computing matchup differential features...")

    diff_stats = [
        # Core rolling performance
        "pts_roll5", "pts_roll10", "pts_roll20",
        "opp_pts_roll5", "opp_pts_roll10", "opp_pts_roll20",   # defensive diff
        "plus_minus_roll5", "plus_minus_roll10", "plus_minus_roll20",
        "win_pct_roll5", "win_pct_roll10", "win_pct_roll20",
        "fg_pct_roll20", "fg3_pct_roll20",
        "ast_roll20", "tov_roll20",
        # Season context
        "cum_win_pct", "sos_roll10", "sos_roll20",
        # Rest & fatigue
        "days_rest", "games_last_5_days", "games_last_7_days",
        # Trend & volatility
        "net_rating_trend_5", "net_rating_std_10",
        # Home/away splits
        "home_net_rating_season", "away_net_rating_season",
        # Style mismatch
        "three_style_mismatch", "rebounding_edge",
        # Motivation / psychological
        "revenge_game", "blowout_loss_last_game", "close_playoff_race",
        # Injury differentials — positive = home team more depleted
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
