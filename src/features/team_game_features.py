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
from src.features.lineup_features import build_lineup_features


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

ADV_ROLL_STATS = [
    "off_rtg_game", "def_rtg_game", "net_rtg_game", "pace_game",
    "efg_game", "ts_game", "tov_poss_game", "tov_pct_game",
    "oreb_pct_game", "ft_rate_game",
]

# ── Arena coordinates (lat, lon) for travel distance computation ──────────────
# Covers all 30 modern NBA teams. LAC and LAL share Crypto.com Arena.
ARENA_COORDS = {
    'ATL': (33.7573, -84.3963),
    'BKN': (40.6826, -73.9754),
    'BOS': (42.3662, -71.0621),
    'CHA': (35.2251, -80.8392),
    'CHI': (41.8807, -87.6742),
    'CLE': (41.4965, -81.6882),
    'DAL': (32.7905, -96.8103),
    'DEN': (39.7487, -105.0077),
    'DET': (42.3410, -83.0553),
    'GSW': (37.7680, -122.3877),
    'HOU': (29.7508, -95.3621),
    'IND': (39.7638, -86.1555),
    'LAC': (34.0430, -118.2673),
    'LAL': (34.0430, -118.2673),
    'MEM': (35.1382, -90.0505),
    'MIA': (25.7814, -80.1870),
    'MIL': (43.0451, -87.9170),
    'MIN': (44.9795, -93.2760),
    'NOP': (29.9490, -90.0812),
    'NYK': (40.7505, -73.9934),
    'OKC': (35.4634, -97.5151),
    'ORL': (28.5392, -81.3836),
    'PHI': (39.9012, -75.1720),
    'PHX': (33.4457, -112.0712),
    'POR': (45.5316, -122.6668),
    'SAC': (38.5802, -121.4996),
    'SAS': (29.4270, -98.4375),
    'TOR': (43.6435, -79.3791),
    'UTA': (40.7683, -111.9011),
    'WAS': (38.8981, -77.0209),
}

# ── Arena timezone zones (rough geographic grouping for cross-country flag) ────
# PHX is Mountain year-round (no DST), but "Mountain" is correct for the
# geographic cross-country indicator purpose.
ARENA_TIMEZONE = {
    'ATL': 'Eastern', 'BKN': 'Eastern', 'BOS': 'Eastern', 'CHA': 'Eastern',
    'CHI': 'Central', 'CLE': 'Eastern', 'DAL': 'Central', 'DEN': 'Mountain',
    'DET': 'Eastern', 'GSW': 'Pacific', 'HOU': 'Central', 'IND': 'Eastern',
    'LAC': 'Pacific', 'LAL': 'Pacific', 'MEM': 'Central', 'MIA': 'Eastern',
    'MIL': 'Central', 'MIN': 'Central', 'NOP': 'Central', 'NYK': 'Eastern',
    'OKC': 'Central', 'ORL': 'Eastern', 'PHI': 'Eastern', 'PHX': 'Mountain',
    'POR': 'Pacific', 'SAC': 'Pacific', 'SAS': 'Central', 'TOR': 'Eastern',
    'UTA': 'Mountain', 'WAS': 'Eastern',
}


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


def _haversine_miles(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in miles. Accepts numpy arrays.

    Accuracy within 0.2% of geodesic -- negligible for features ranging 0-2700 mi.
    Used instead of geopy.distance.geodesic for 1000x performance on batch data.
    """
    R = 3958.8  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


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
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")
    df = df.sort_values(["team_id", "game_date"]).reset_index(drop=True)

    # ── Basic context features ────────────────────────────────────────────────
    df["is_home"]       = _parse_home_away(df["matchup"])
    df["opponent_abbr"] = _extract_opponent_abbr(df["matchup"])

    # -- Travel distance and timezone-change features (Phase 4, FR-3.2/FR-3.3) --
    print("Computing travel distance features...")
    _arena_lat = {k: v[0] for k, v in ARENA_COORDS.items()}
    _arena_lon = {k: v[1] for k, v in ARENA_COORDS.items()}

    # Current game arena: team's own arena if home, opponent's arena if away
    curr_arena_abbr = np.where(
        df["is_home"] == 1, df["team_abbreviation"], df["opponent_abbr"]
    )
    curr_series = pd.Series(curr_arena_abbr, index=df.index)

    df["_curr_lat"] = curr_series.map(_arena_lat)
    df["_curr_lon"] = curr_series.map(_arena_lon)
    df["_curr_tz"]  = curr_series.map(ARENA_TIMEZONE)

    # Previous game arena (shift-1 within team -- no leakage, NFR-1)
    df["_prev_lat"] = df.groupby("team_id")["_curr_lat"].shift(1)
    df["_prev_lon"] = df.groupby("team_id")["_curr_lon"].shift(1)
    df["_prev_tz"]  = df.groupby("team_id")["_curr_tz"].shift(1)

    # travel_miles: haversine distance from prev game arena to current game arena
    has_both = df["_prev_lat"].notna() & df["_curr_lat"].notna()
    df["travel_miles"] = np.nan
    df.loc[has_both, "travel_miles"] = _haversine_miles(
        df.loc[has_both, "_prev_lat"].values,
        df.loc[has_both, "_prev_lon"].values,
        df.loc[has_both, "_curr_lat"].values,
        df.loc[has_both, "_curr_lon"].values,
    )
    df["travel_miles"] = df["travel_miles"].fillna(0)  # first game of season = no travel burden

    # cross_country_travel: 1 if timezone changed from prev game to this game
    df["cross_country_travel"] = (
        df["_prev_tz"].notna()
        & df["_curr_tz"].notna()
        & (df["_prev_tz"] != df["_curr_tz"])
    ).astype(int)

    # Drop internal working columns
    df = df.drop(columns=["_curr_lat", "_curr_lon", "_curr_tz",
                           "_prev_lat", "_prev_lon", "_prev_tz"])

    n_cross = df["cross_country_travel"].sum()
    print(f"  travel_miles: min={df['travel_miles'].min():.0f}mi max={df['travel_miles'].max():.0f}mi")
    print(f"  cross_country_travel: {n_cross:,} games flagged ({n_cross/len(df):.1%})")

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
        print("  Warning: fg3a or fga not in data - three_rate features will be NaN")
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
    # 0-indexed: value at row N = games played BEFORE this game (no current-game leakage)
    df["season_game_num"] = df.groupby(["team_id", "season"]).cumcount()

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

    _n_before_sos = len(df)
    df = df.merge(opp_strength, on=["opponent_abbr", "game_id"], how="left")
    _n_after_sos = len(df)
    print(f"  SOS join: {_n_before_sos:,} rows -> {_n_after_sos:,} rows (dropped {_n_before_sos - _n_after_sos:,})")

    sos_group = df.groupby("team_id", group_keys=False)
    for window in [10, 20]:
        df[f"sos_roll{window}"] = sos_group.apply(
            lambda g: g["opp_pre_game_win_pct"].shift(1).rolling(window, min_periods=1).mean(),
            include_groups=False,
        ).values

    # ── Style mismatch + advanced metrics: join opponent's box score ─────────
    # Use a single expanded self-join keyed on (opponent_abbr, game_id) to get
    # the opposing team's full box score columns for both style mismatch features
    # and pace-normalized advanced metric computation.
    print("Computing style mismatch features...")
    opp_box = df[[
        "team_abbreviation", "game_id",
        "pts", "fga", "fg3m", "fgm", "oreb", "tov", "fta", "dreb",
        "three_rate_raw",
    ]].copy()
    opp_box.columns = [
        "opponent_abbr", "game_id",
        "opp_pts_raw", "opp_fga", "opp_fg3m", "opp_fgm",
        "opp_oreb", "opp_tov", "opp_fta", "opp_dreb",
        "opp_three_rate_game",
    ]
    _n_before_opp = len(df)
    df = df.merge(opp_box, on=["opponent_abbr", "game_id"], how="left")
    _n_after_opp = len(df)
    print(f"  Opponent box join: {_n_before_opp:,} rows -> {_n_after_opp:,} rows (dropped {_n_before_opp - _n_after_opp:,})")

    if df["opp_fga"].notna().sum() == 0:
        raise ValueError(
            "Opponent box score join matched zero rows -- check opponent_abbr/game_id alignment"
        )

    # ── Possession estimates (Oliver formula) ────────────────────────────────
    df["poss_est"]     = df["fga"] - df["oreb"] + df["tov"] + 0.44 * df["fta"]
    df["opp_poss_est"] = df["opp_fga"] - df["opp_oreb"] + df["opp_tov"] + 0.44 * df["opp_fta"]
    df["avg_poss"]     = (df["poss_est"] + df["opp_poss_est"]) / 2

    # ── Per-game advanced metrics (raw, not yet rolled) ───────────────────────
    # IMPORTANT: These are computed here after the opponent join; do NOT add to
    # ROLL_STATS (which runs before the join). They are rolled separately via
    # ADV_ROLL_STATS loop further below.
    df["off_rtg_game"]  = df["pts"] / df["avg_poss"].replace(0, np.nan) * 100
    df["def_rtg_game"]  = df["opp_pts_raw"] / df["avg_poss"].replace(0, np.nan) * 100
    df["net_rtg_game"]  = df["off_rtg_game"] - df["def_rtg_game"]
    df["pace_game"]     = df["avg_poss"]
    df["efg_game"]      = (df["fgm"] + 0.5 * df["fg3m"]) / df["fga"].replace(0, np.nan)
    df["ts_game"]       = df["pts"] / (2 * (df["fga"] + 0.44 * df["fta"])).replace(0, np.nan)
    df["tov_poss_game"] = df["tov"] / df["poss_est"].replace(0, np.nan)
    df["tov_pct_game"]  = df["tov"] / (df["fga"] + 0.44 * df["fta"] + df["tov"]).replace(0, np.nan)
    df["oreb_pct_game"] = df["oreb"] / (df["oreb"] + df["opp_dreb"]).replace(0, np.nan)
    df["ft_rate_game"]  = df["fta"] / df["fga"].replace(0, np.nan)

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
        lambda g: g["opp_dreb"].shift(1).rolling(20, min_periods=1).mean(),
        include_groups=False,
    ).values

    # Composite style features
    df["three_style_mismatch"] = df["three_rate"] - df["opponent_three_allowed_rate"]
    df["rebounding_edge"]      = df["oreb_roll20"] - df["opp_dreb_roll20"]

    # ── Advanced metric rolling features ──────────────────────────────────────
    print("Computing advanced metric rolling features...")
    adv_group = df.groupby("team_id", group_keys=False)
    for window in roll_windows:
        for stat in ADV_ROLL_STATS:
            col_name = f"{stat}_roll{window}"
            df[col_name] = adv_group.apply(
                lambda g, s=stat: _rolling_mean_shift(g, s, window),
                include_groups=False,
            ).values

    # Sanity check: print null rates for new rolling features
    for stat in ADV_ROLL_STATS:
        col = f"{stat}_roll20"
        nn = df[col].notna().mean()
        print(f"  {col}: {nn:.1%} non-null")

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
        print(f"  Warning: {standings_path} not found - close_playoff_race defaulted to 0")
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
        "travel_miles", "cross_country_travel",    # Phase 4: travel burden
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
        # Referee features (NaN when no scrape data -- do NOT fillna(0))
        "ref_crew_fta_rate_roll10", "ref_crew_fta_rate_roll20",
        "ref_crew_pace_impact_roll10",
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

        # Normalize join keys on BOTH sides before merging to prevent silent
        # type-mismatch failures (e.g. int vs str game_id across CSV sources).
        output["game_id"]    = output["game_id"].astype(str).str.strip()
        output["team_id"]    = output["team_id"].astype(int)
        injury_df["game_id"] = injury_df["game_id"].astype(str).str.strip()
        injury_df["team_id"] = injury_df["team_id"].astype(int)

        output = output.merge(injury_df, on=["team_id", "game_id"], how="left")

        # Assert the join actually matched rows — zero matches means a key format
        # mismatch slipped through normalization (NFR-1 guard).
        n_matched = output["missing_minutes"].notna().sum()
        print(f"  Injury join: {n_matched:,} rows matched out of {len(output):,}")
        if not injury_df.empty:
            assert n_matched > 0, (
                "Injury proxy join matched zero rows — game_id or team_id type mismatch. "
                "Check that player_game_logs and team_game_logs game_id formats are compatible."
            )

        output["missing_minutes"]       = output["missing_minutes"].fillna(0)
        output["missing_usg_pct"]       = output["missing_usg_pct"].fillna(0)
        output["rotation_availability"] = output["rotation_availability"].fillna(1.0)
        output["star_player_out"]       = output["star_player_out"].fillna(0)
        output["n_missing_rotation"]    = output["n_missing_rotation"].fillna(0)
        print(f"  Injury features joined: {injury_df.columns.tolist()}")
    except Exception as e:
        print(f"  Warning: could not build injury proxy features ({e}). Skipping.")

    # ── Join referee features ─────────────────────────────────────────────────
    # Referee features are per-game (same crew officiates both home and away).
    # ref_df has one row per game keyed on (game_date, home_team abbreviation).
    # We join to both home and away team rows by matching on game_date + home team.
    # Referee features stay NaN when no scrape data exists (do NOT fillna(0) --
    # zero would inject false signal for pre-scrape seasons; Pitfall 6, RESEARCH.md).
    try:
        from src.features.referee_features import build_referee_features
        print("Building referee features...")
        ref_df = build_referee_features()

        if not ref_df.empty:
            # Normalize game_date to string for join
            output["game_date_str"] = pd.to_datetime(output["game_date"]).dt.strftime("%Y-%m-%d")
            ref_df["game_date"] = ref_df["game_date"].astype(str)

            # ref_df has one row per game with home_team as the join key.
            # team_game_features has two rows per game (home + away).
            # We extract the home team from each game in output to join on.
            # is_home == 1 -> team_abbreviation == home_team in ref_df
            # is_home == 0 -> opponent_abbr == home_team in ref_df
            # Merge home team rows: join on (game_date, team_abbreviation == home_team)
            ref_cols = [
                "game_date", "home_team",
                "ref_crew_fta_rate_roll10", "ref_crew_fta_rate_roll20",
                "ref_crew_pace_impact_roll10",
            ]
            ref_home_merge = ref_df[ref_cols].rename(columns={"home_team": "join_team"})

            # For each team-game row we need to identify which team is home.
            # Instead of two separate merges, we create a "home_abbr" column on output
            # that gives the home team abbreviation for both home and away rows:
            #   is_home==1: team's own abbreviation
            #   is_home==0: the opponent's abbreviation
            output["home_abbr_for_join"] = np.where(
                output["is_home"] == 1,
                output["team_abbreviation"],
                output["opponent_abbr"],
            )

            output = output.merge(
                ref_home_merge.rename(columns={"join_team": "home_abbr_for_join"}),
                on=["game_date_str", "home_abbr_for_join"],
                how="left",
                suffixes=("", "_ref"),
            )
            # Drop the temporary join key columns
            output = output.drop(columns=["home_abbr_for_join", "game_date_str"])

            n_matched = output["ref_crew_fta_rate_roll10"].notna().sum()
            print(f"  Referee feature join: {n_matched:,} rows matched out of {len(output):,}")
        else:
            print("  Referee features: no scrape data yet (empty DataFrame). Columns will be absent.")
    except Exception as e:
        print(f"  Warning: could not build referee features ({e}). Skipping.")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Saved {len(output):,} rows x {len(output.columns)} cols -> {output_path}")
    return output


# ── Four Factors composite (Dean Oliver, "Basketball on Paper" 2004) ──────────

FOUR_FACTORS_WEIGHTS = {
    "efg_game_roll20":      +0.40,
    "tov_pct_game_roll20":  -0.25,   # negative: more turnovers = disadvantage
    "oreb_pct_game_roll20": +0.20,
    "ft_rate_game_roll20":  +0.15,
}


def _four_factors_composite(matchup: pd.DataFrame, weights: dict) -> pd.Series:
    """Compute weighted Four Factors differential (home advantage - away advantage).

    Uses Dean Oliver's canonical weights. The result is a single composite
    that captures the home team's efficiency advantage across all four dimensions.
    Positive = home team more efficient overall.
    """
    composite = pd.Series(0.0, index=matchup.index)
    for feat, weight in weights.items():
        h_col = f"home_{feat}"
        a_col = f"away_{feat}"
        if h_col in matchup.columns and a_col in matchup.columns:
            diff = matchup[h_col] - matchup[a_col]
            composite += weight * diff.fillna(0)
    return composite


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
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")

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
        "travel_miles", "cross_country_travel",    # Phase 4: travel burden
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
        # Referee features (NaN when no scrape data -- do NOT fillna(0))
        "ref_crew_fta_rate_roll10", "ref_crew_fta_rate_roll20",
        "ref_crew_pace_impact_roll10",
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

    _n_meta = len(meta)
    matchup = (
        meta
        .merge(home_feat, on="game_id", how="inner")
        .merge(away_feat, on="game_id", how="inner")
    )
    print(f"  Home/away matchup join: {_n_meta:,} games -> {len(matchup):,} rows (dropped {_n_meta - len(matchup):,})")

    # -- Season-segment context (Phase 4, FR-3.4) --
    matchup["season_month"] = pd.to_datetime(matchup["game_date"]).dt.month
    print(f"  season_month: {matchup['season_month'].nunique()} unique months, "
          f"range {int(matchup['season_month'].min())}-{int(matchup['season_month'].max())}")

    # ── Lineup efficiency features (Phase 8, FEAT-02) ─────────────────────────
    # Lineup data exists for 2023-24 and 2024-25 only. Left join so earlier
    # seasons simply receive 0.0 (neutral / unknown) values.
    print("Joining lineup efficiency features...")
    lineup_cols = [
        "top1_lineup_net_rtg", "top3_lineup_net_rtg", "avg_lineup_net_rtg",
        "lineup_net_rtg_std", "best_off_rating", "best_def_rating", "n_lineups",
    ]
    try:
        lineup_feat = build_lineup_features()

        if not lineup_feat.empty:
            # Normalize key types for safe join
            lineup_feat["season"]  = lineup_feat["season"].astype(int)
            lineup_feat["team_id"] = lineup_feat["team_id"].astype(int)
            matchup["season"]      = matchup["season"].astype(int)

            # Extract home team_id from meta for join
            home_ids = home[["game_id", "team_id"]].copy()
            home_ids.columns = ["game_id", "home_team_id_for_join"]
            away_ids = away[["game_id", "team_id"]].copy()
            away_ids.columns = ["game_id", "away_team_id_for_join"]
            matchup = matchup.merge(home_ids, on="game_id", how="left")
            matchup = matchup.merge(away_ids, on="game_id", how="left")

            # Join home lineup features
            home_lineup = lineup_feat[["season", "team_id"] + lineup_cols].copy()
            home_lineup.columns = (
                ["season", "home_team_id_for_join"]
                + [f"home_{c}" for c in lineup_cols]
            )
            matchup = matchup.merge(
                home_lineup, on=["season", "home_team_id_for_join"], how="left"
            )

            # Join away lineup features
            away_lineup = lineup_feat[["season", "team_id"] + lineup_cols].copy()
            away_lineup.columns = (
                ["season", "away_team_id_for_join"]
                + [f"away_{c}" for c in lineup_cols]
            )
            matchup = matchup.merge(
                away_lineup, on=["season", "away_team_id_for_join"], how="left"
            )

            # Drop the temporary join key columns
            matchup = matchup.drop(
                columns=["home_team_id_for_join", "away_team_id_for_join"]
            )

            # Fill NaN lineup columns with 0.0 for seasons without lineup data
            for side in ["home", "away"]:
                for col in lineup_cols:
                    full = f"{side}_{col}"
                    if full in matchup.columns:
                        matchup[full] = matchup[full].fillna(0.0)

            # Non-null rate for lineup features (diagnostic)
            recent_mask = matchup["season"].isin(
                lineup_feat["season"].unique()
            )
            n_recent = recent_mask.sum()
            if n_recent > 0:
                nn_rate = matchup.loc[recent_mask, "home_avg_lineup_net_rtg"].notna().mean()
                print(
                    f"  Lineup features: {len(lineup_cols)*2} new columns added. "
                    f"Non-null rate for recent seasons: {nn_rate:.1%} "
                    f"({n_recent:,} rows)"
                )
        else:
            print("  Lineup features: empty DataFrame — skipping join.")
            for side in ["home", "away"]:
                for col in lineup_cols:
                    matchup[f"{side}_{col}"] = 0.0

    except Exception as e:
        print(f"  Warning: could not build lineup features ({e}). Defaulting to 0.0.")
        for side in ["home", "away"]:
            for col in lineup_cols:
                matchup[f"{side}_{col}"] = 0.0

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
        "is_back_to_back",                         # Phase 4: fatigue asymmetry signal
        "travel_miles",                            # Phase 4: travel burden
        "cross_country_travel",                    # Phase 4: timezone disruption
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
        # Advanced efficiency differentials (Phase 2)
        "off_rtg_game_roll5", "off_rtg_game_roll10", "off_rtg_game_roll20",
        "def_rtg_game_roll5", "def_rtg_game_roll10", "def_rtg_game_roll20",
        "net_rtg_game_roll5", "net_rtg_game_roll10", "net_rtg_game_roll20",
        "pace_game_roll20",
        "efg_game_roll20", "ts_game_roll20",
        "tov_poss_game_roll20",
        # Referee foul rate (same crew for both teams, but diff captures any asymmetry
        # from imperfect joins; home/away raw values are more meaningful signal)
        "ref_crew_fta_rate_roll10", "ref_crew_fta_rate_roll20",
        # Lineup efficiency differentials (Phase 8, FEAT-02)
        "top1_lineup_net_rtg", "top3_lineup_net_rtg", "avg_lineup_net_rtg",
        "best_off_rating", "best_def_rating",
    ]

    for stat in diff_stats:
        h_col = f"home_{stat}"
        a_col = f"away_{stat}"
        if h_col in matchup.columns and a_col in matchup.columns:
            matchup[f"diff_{stat}"] = matchup[h_col] - matchup[a_col]

    # Four Factors differential composite (FR-2.4)
    matchup["diff_four_factors_composite"] = _four_factors_composite(
        matchup, FOUR_FACTORS_WEIGHTS
    )
    print(f"  Four Factors composite: {matchup['diff_four_factors_composite'].notna().sum():,} non-null values")

    matchup = matchup.dropna(subset=["home_win"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    matchup.to_csv(output_path, index=False)
    print(f"Saved {len(matchup):,} matchup rows x {len(matchup.columns)} cols -> {output_path}")
    return matchup


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    build_team_game_features()
    build_matchup_dataset()
    print("\nTeam game feature engineering complete.")
