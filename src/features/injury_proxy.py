"""
Injury Proxy Feature Engineering
===================================
Builds pre-game lineup availability features from player game logs.

Since the NBA API does not expose a historical injury timeline, we infer
player availability using a proxy approach: a rotation player is anyone
who has averaged 15+ minutes over their last 5 games. If such a player
has no entry in the game log for a given game, they almost certainly
did not play -- whether due to injury, rest, or coach's decision.

This gives the model four meaningful signals for each team entering a game:

  missing_minutes        -- Total expected minutes from absent rotation players.
                           A value of 35 means ~35 min of expected production
                           is unavailable, roughly equivalent to losing a starter.

  missing_usg_pct        -- Combined usage rate of absent players. Usage rate
                           measures what % of team possessions a player uses;
                           a missing player at 0.28 usg_pct is a much bigger
                           deal than one at 0.12.

  rotation_availability  -- Expected minutes available / total expected minutes.
                           1.0 = full strength. 0.7 = team is missing 30% of
                           their normal rotation's playing time.

  star_player_out        -- Binary flag. 1 if any absent player has a season
                           usage rate >= 25% (the threshold commonly used to
                           define a team's primary ball-handler / star).

These features are computed per (team_id, game_id) and merged into
team_game_features, then flow through to the matchup table as
home_/away_ prefixed columns and a diff_missing_minutes differential.

Primary data source:
  data/processed/player_absences.csv (built by get_historical_absences.py).
  When this file exists, the expensive merge_asof derivation is skipped and
  the pre-computed absence records are used directly.

Fallback:
  If player_absences.csv is not found, absence data is re-derived from
  player_game_logs.csv using the original merge_asof logic. A warning is
  printed. Run src/data/get_historical_absences.py to rebuild the CSV.

Leakage note:
  All rolling stats use shift(1) so only prior games inform the
  "expected minutes" baseline. The fact that a player is absent in the
  current game's log is pre-game-observable information -- NBA teams
  publish official injury reports ~1 hour before tip-off. For real-time
  predictions, pair this module with get_todays_injury_report() below.

Code path boundary (FR-4.4):
  This module is the TRAINING PATH for injury features. It builds
  historical injury proxy from game logs. It must NEVER import or call
  src.data.external.injury_report (the INFERENCE PATH).

  The inference path lives in src/data/external/injury_report.py and
  fetches live NBA pre-game reports. These two modules must never share
  inputs or call each other. Violations break the training/inference
  separation that prevents data leakage.

Usage:
    from src.features.injury_proxy import build_injury_proxy_features
    df = build_injury_proxy_features()

    # For live game predictions use the canonical INFERENCE PATH module:
    #   src.data.external.injury_report.get_todays_nba_injury_report()

    python src/features/injury_proxy.py
"""

import os
import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# -- Code path boundary (FR-4.4) -----------------------------------------------
# This module is TRAINING PATH ONLY.
# NEVER import src.data.external.injury_report here.
# NEVER call get_todays_nba_injury_report() from build_injury_proxy_features().
# The inference path is src/data/external/injury_report.py.
_CODE_PATH = "TRAINING"

# ── Config ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GAME_LOG_PATH     = "data/processed/player_game_logs.csv"
ADV_STATS_PATH    = "data/processed/player_stats_advanced.csv"
OUTPUT_PATH       = "data/features/injury_proxy_features.csv"
ABSENCES_CSV_PATH = "data/processed/player_absences.csv"

# pathlib constant — used by load_real_absences() and build_injury_proxy_features()
ABSENCES_PATH = PROJECT_ROOT / "data" / "processed" / "player_absences.csv"

# A player is considered "in the rotation" if they've averaged at least
# this many minutes over the last 5 games they played.
MIN_ROTATION_MINUTES = 15.0

# Minimum games in the rolling window before we consider a player part
# of the rotation — avoids flagging a player as "expected" based on 1 game.
MIN_ROTATION_GAMES = 2

# Usage rate threshold defining a "star" player.
# 0.25 = player uses 25% of team possessions while on the floor.
STAR_USG_THRESHOLD = 0.25

# Rolling window for establishing rotation baseline
ROLL_WINDOW = 5


# ── Load real absence data ─────────────────────────────────────────────────────

def load_real_absences(season: int | None = None) -> pd.DataFrame | None:
    """
    Load pre-computed player absence records from ``player_absences.csv``.

    The CSV was built by Phase 10 (get_historical_absences.py) and contains
    one row per (player_id, game_date) covering all seasons in the pipeline.
    Each row records whether the player was absent for that specific game
    (``was_absent=1``) or present (``was_absent=0``).

    Leakage safety: the ``was_absent`` value for game N is pre-game-observable
    information (teams publish official injury reports ~1 hour before tip-off),
    so joining on ``player_id + game_date`` does NOT introduce data leakage.

    Args:
        season: Optional integer season code (e.g. ``202425``).  When provided,
                only rows matching that season are returned.  Pass ``None`` to
                load all seasons.

    Returns:
        DataFrame with at least: ``player_id``, ``game_date``, ``season``,
        ``team_id``, ``min_roll5``, ``usg_pct``, ``was_absent``.
        Returns ``None`` when the CSV does not exist, which triggers the
        rolling-proxy fallback in ``build_injury_proxy_features()``.
    """
    if not ABSENCES_PATH.exists():
        return None

    required_cols = ["player_id", "game_date", "season", "team_id",
                     "min_roll5", "usg_pct", "was_absent"]
    try:
        df = pd.read_csv(ABSENCES_PATH, usecols=required_cols)
    except ValueError as exc:
        # Column subset missing — file schema mismatch; treat as unavailable.
        print(
            f"  WARNING: player_absences.csv column mismatch ({exc}). "
            "Falling back to rolling proxy."
        )
        return None

    if season is not None:
        df = df[df["season"] == season].copy()

    df["game_date"] = pd.to_datetime(df["game_date"])
    df["player_id"] = df["player_id"].astype(int)
    df["team_id"]   = df["team_id"].astype(int)
    df["was_absent"] = df["was_absent"].astype(int)
    df["min_roll5"]  = pd.to_numeric(df["min_roll5"],  errors="coerce").fillna(0.0)
    df["usg_pct"]    = pd.to_numeric(df["usg_pct"],    errors="coerce").fillna(0.18)

    return df


# ── Helper: aggregate pre-built absences DataFrame ────────────────────────────

def _aggregate_from_absences_df(absences: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate a player_absences DataFrame (schema from get_historical_absences.py)
    into per-(team_id, game_id) injury proxy aggregate rows.

    Expected input columns:
        player_id, team_id, game_id, min_roll5, usg_pct, was_absent

    The min_roll5 values in the input were already computed with shift(1) before
    rolling by get_historical_absences.py, so no additional shift is needed here.

    Returns a DataFrame with columns:
        game_id, team_id,
        missing_minutes, missing_usg_pct, n_missing_rotation,
        star_player_out, total_expected_minutes
    """
    absences = absences.copy()
    absences["game_id"] = absences["game_id"].astype(str).str.strip()
    absences["team_id"] = absences["team_id"].astype(int)

    # All rotation player-games (absent + present) contribute to total expected minutes.
    # This mirrors the original logic: expected_df (absent) + played_rotation combined.
    total_agg = (
        absences
        .groupby(["game_id", "team_id"])["min_roll5"]
        .sum()
        .reset_index()
        .rename(columns={"min_roll5": "total_expected_minutes"})
    )

    absent = absences[absences["was_absent"] == 1].copy()

    if absent.empty:
        absent_agg = pd.DataFrame(
            columns=["game_id", "team_id", "missing_minutes",
                     "missing_usg_pct", "n_missing_rotation", "star_player_out"]
        )
    else:
        absent_agg = (
            absent
            .groupby(["game_id", "team_id"])
            .agg(
                missing_minutes    =("min_roll5",  "sum"),
                missing_usg_pct    =("usg_pct",    "sum"),
                n_missing_rotation =("player_id",  "count"),
            )
            .reset_index()
        )

        # Star player out: any absent player with usg >= threshold
        star_out = (
            absent[absent["usg_pct"] >= STAR_USG_THRESHOLD]
            .groupby(["game_id", "team_id"])
            .size()
            .reset_index(name="star_player_out")
        )
        star_out["star_player_out"] = 1

        absent_agg = absent_agg.merge(star_out, on=["game_id", "team_id"], how="left")
        absent_agg["star_player_out"] = absent_agg["star_player_out"].fillna(0).astype(int)

    result = total_agg.merge(absent_agg, on=["game_id", "team_id"], how="left")
    return result


# ── Core builder ───────────────────────────────────────────────────────────────

def build_injury_proxy_features(
    game_log_path:  str = GAME_LOG_PATH,
    adv_stats_path: str = ADV_STATS_PATH,
    output_path:    str = OUTPUT_PATH,
    absences_path:  str = ABSENCES_CSV_PATH,
) -> pd.DataFrame:
    """
    Build per-(team_id, game_id) lineup availability features.

    Primary source: absences_path (player_absences.csv). When present, the
    pre-built per-player absence records are aggregated directly, bypassing
    the expensive merge_asof derivation.

    Fallback: if absences_path is None or does not exist, absence data is
    derived from game_log_path + adv_stats_path using the original logic.
    A warning is logged. This path is also used by unit tests that supply
    synthetic game logs directly.

    Returns a DataFrame with columns:
        team_id, game_id,
        missing_minutes,        — total expected min from absent rotation players
        missing_usg_pct,        — combined usage rate of absent rotation players
        rotation_availability,  — fraction of rotation minutes that are available
        star_player_out,        — 1 if any absent player has usg_pct >= 0.25
        n_missing_rotation      — headcount of absent rotation players
    """
    print("=" * 60)
    print("INJURY PROXY FEATURE ENGINEERING")
    print("=" * 60)

    # ── Primary path: load player_absences.csv via load_real_absences() ─────────
    # load_real_absences() returns None when the file is missing, which triggers
    # the fallback.  We only use this path when the caller has not overridden the
    # game log (i.e. it's a standard pipeline run, not a unit test with synthetic
    # data).
    using_default_game_log = (game_log_path == GAME_LOG_PATH)
    real_absences = load_real_absences() if using_default_game_log else None

    # Legacy CSV path guard: the old ABSENCES_CSV_PATH constant (expects game_id
    # column) is kept for backward compatibility.  If load_real_absences() already
    # returned data we skip the legacy check entirely.
    if real_absences is None and using_default_game_log:
        # Check legacy path that has game_id column (pre-Phase-10 format)
        legacy_path = absences_path or ABSENCES_CSV_PATH
        if legacy_path and os.path.exists(legacy_path):
            try:
                absences_legacy = pd.read_csv(
                    legacy_path,
                    usecols=["player_id", "team_id", "game_id",
                             "min_roll5", "usg_pct", "was_absent"],
                )
                print(f"\nPrimary source (legacy game_id format): {legacy_path}")
                print(f"  Loaded {len(absences_legacy):,} rotation player-game rows")
                print(
                    f"  Absent rows (was_absent=1): "
                    f"{(absences_legacy['was_absent'] == 1).sum():,}"
                )
                result_intermediate = _aggregate_from_absences_df(absences_legacy)
                # Jump to final output section
                _use_intermediate = True
            except (ValueError, KeyError) as exc:
                print(
                    f"\n  WARNING: {legacy_path} missing expected columns ({exc}). "
                    "Falling back to rolling proxy."
                )
                _use_intermediate = False
        else:
            _use_intermediate = False
    else:
        _use_intermediate = False

    if real_absences is not None:
        # ── New primary path: game_date-based player_absences.csv ─────────────
        # The CSV schema from Phase 10 uses game_date (not game_id).  We join
        # with the game log's (team_id, game_date) → game_id mapping so that
        # _aggregate_from_absences_df() (which groups by game_id) can be reused.
        print(f"\nPrimary source: {ABSENCES_PATH}")
        print(f"  Loaded {len(real_absences):,} rotation player-game rows")
        print(
            f"  Absent rows (was_absent=1): "
            f"{(real_absences['was_absent'] == 1).sum():,}"
        )

        # Build game_date → game_id mapping from the game log
        print("  Resolving game_id from player_game_logs...")
        log_map_cols = ["team_id", "game_id", "game_date"]
        log_map = pd.read_csv(game_log_path, usecols=log_map_cols)
        log_map["game_date"] = pd.to_datetime(log_map["game_date"])
        log_map["team_id"]   = log_map["team_id"].astype(int)
        log_map["game_id"]   = log_map["game_id"].astype(str).str.strip()
        log_map = log_map.drop_duplicates(subset=["team_id", "game_date"])

        absences_with_id = real_absences.merge(
            log_map[["team_id", "game_date", "game_id"]],
            on=["team_id", "game_date"],
            how="left",
        )

        missing_game_id = absences_with_id["game_id"].isna().sum()
        if missing_game_id > 0:
            print(
                f"  WARNING: {missing_game_id:,} absence rows could not be matched "
                "to a game_id and will be dropped."
            )
        absences_with_id = absences_with_id.dropna(subset=["game_id"])

        result_intermediate = _aggregate_from_absences_df(absences_with_id)
        _use_intermediate = True

    if not _use_intermediate:
        # ── Fallback path: derive from raw game logs ──────────────────────────
        if absences_path:
            print(
                f"\n  WARNING: {absences_path} not found — "
                "falling back to deriving absence data from game logs. "
                "Run src/data/get_historical_absences.py to build the primary data source."
            )

        print("\nLoading player_game_logs...")
        cols = ["season", "player_id", "player_name", "team_id",
                "team_abbreviation", "game_id", "game_date", "min"]
        df = pd.read_csv(game_log_path, usecols=cols)
        df["game_date"] = pd.to_datetime(df["game_date"])

        # Ensure minutes is numeric — handle "MM:SS" strings from older seasons
        df["min"] = pd.to_numeric(df["min"], errors="coerce").fillna(0)

        # Sort chronologically within each player
        df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)
        print(f"  Loaded {len(df):,} player-game rows")

        # ── Load season-level usage rates ──────────────────────────────────────
        print("Loading advanced stats for usage rates...")
        adv = pd.read_csv(adv_stats_path, usecols=["player_id", "season", "usg_pct"])
        adv = adv.drop_duplicates(subset=["player_id", "season"])
        # Fill missing usage with a neutral league-average estimate
        adv["usg_pct"] = adv["usg_pct"].fillna(0.18)

        df = df.merge(adv, on=["player_id", "season"], how="left")
        df["usg_pct"] = df["usg_pct"].fillna(0.18)

        # ── Compute rolling baseline minutes per player ────────────────────────
        # CRITICAL: shift(1) BEFORE rolling so row N only uses rows 0..N-1.
        print("Computing rolling baseline minutes per player...")

        df["min_roll5"] = (
            df.groupby("player_id")["min"]
            .transform(lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=MIN_ROTATION_GAMES).mean())
        )
        df["games_in_roll5"] = (
            df.groupby("player_id")["min"]
            .transform(lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=1).count())
        )

        # A player is "in the rotation" for a given game if:
        #   - They averaged MIN_ROTATION_MINUTES+ over their last 5 games (shift-1)
        #   - They appeared in at least MIN_ROTATION_GAMES of those 5 games
        # NOTE: in_rotation is based on prior-games rolling stats (shift-1), so it is
        # the pre-game expectation — it does NOT use the current game's minutes.
        df["in_rotation"] = (
            (df["min_roll5"] >= MIN_ROTATION_MINUTES) &
            (df["games_in_roll5"] >= MIN_ROTATION_GAMES)
        )

        # ── Identify absent rotation players (vectorized) ──────────────────────
        # Design: for each player-team, use pd.merge_asof to project their last
        # known rotation status onto ALL team-game dates (sorted merge on game_date).
        # A player was "expected" to play game G if their most recent appearance
        # before G shows in_rotation=True and was within MAX_STALE_DAYS days.
        # Then anti-join against actual appearances to find who was absent.
        print("Identifying absent rotation players per team-game...")

        MAX_STALE_DAYS = ROLL_WINDOW * 5   # ~25 days; beyond this, don't flag player

        # All (team_id, game_id, game_date) pairs that actually occurred.
        team_games = (
            df[["team_id", "game_id", "game_date"]]
            .drop_duplicates()
            .copy()
        )

        # Player status timeline per player per team, sorted for merge_asof
        player_status = (
            df[["player_id", "team_id", "game_date", "in_rotation", "min_roll5", "usg_pct"]]
            .copy()
            .sort_values(["player_id", "team_id", "game_date"])
        )

        # Actual played: all (game_id, team_id, player_id) entries in the logs
        # (game logs only include rows where the player actually played)
        actual_played = (
            df[["game_id", "team_id", "player_id"]]
            .drop_duplicates()
            .assign(played=True)
        )

        # Vectorized expected-rotation builder using merge_asof per (team, player)
        expected_parts = []
        for team_id_val, tg_grp in team_games.groupby("team_id"):
            ps_team = player_status[player_status["team_id"] == team_id_val]
            if ps_team.empty:
                continue
            tg_sorted = tg_grp.sort_values("game_date")[["team_id", "game_id", "game_date"]]

            for player_id_val, player_grp in ps_team.groupby("player_id"):
                ps_sorted = (
                    player_grp
                    .sort_values("game_date")
                    .rename(columns={"game_date": "last_date"})
                    [["last_date", "in_rotation", "min_roll5", "usg_pct"]]
                )
                # merge_asof: for each team-game date, find the player's last prior appearance
                merged = pd.merge_asof(
                    tg_sorted,
                    ps_sorted,
                    left_on="game_date",
                    right_on="last_date",
                    direction="backward",
                )
                # Keep rows where the player had a prior appearance showing in_rotation=True
                # and that appearance was strictly before the game (days_since > 0)
                # and not stale (days_since <= MAX_STALE_DAYS)
                merged = merged[merged["last_date"].notna()].copy()
                merged["days_since"] = (merged["game_date"] - merged["last_date"]).dt.days
                merged = merged[
                    (merged["in_rotation"])
                    & (merged["days_since"] > 0)
                    & (merged["days_since"] <= MAX_STALE_DAYS)
                ]
                if merged.empty:
                    continue
                merged["player_id"] = player_id_val
                expected_parts.append(
                    merged[["game_id", "team_id", "player_id", "min_roll5", "usg_pct"]]
                )

        expected_df = (
            pd.concat(expected_parts, ignore_index=True)
            if expected_parts
            else pd.DataFrame(columns=["game_id", "team_id", "player_id", "min_roll5", "usg_pct"])
        )
        print(f"  Expected rotation appearances: {len(expected_df):,}")

        # Anti-join: expected players who did NOT appear in the actual game log
        merged_check = expected_df.merge(
            actual_played,
            on=["game_id", "team_id", "player_id"],
            how="left",
        )
        # played=NaN → player was expected but did not appear in the game log
        absent = merged_check[merged_check["played"].isna()].copy()

        print(f"  Absent rotation instances: {len(absent):,}")
        if len(absent) > 0:
            print(f"  Games with at least one absent rotation player: "
                  f"{absent['game_id'].nunique():,}")

        # ── Aggregate absent players per (team_id, game_id) ──────────────────
        absent_agg = (
            absent
            .groupby(["game_id", "team_id"])
            .agg(
                missing_minutes    =("min_roll5",  "sum"),
                missing_usg_pct    =("usg_pct",    "sum"),
                n_missing_rotation =("player_id",  "count"),
            )
            .reset_index()
        )

        # Star player out: any absent player with usg >= threshold
        star_out = (
            absent[absent["usg_pct"] >= STAR_USG_THRESHOLD]
            .groupby(["game_id", "team_id"])
            .size()
            .reset_index(name="star_player_out")
        )
        star_out["star_player_out"] = 1

        absent_agg = absent_agg.merge(star_out, on=["game_id", "team_id"], how="left")
        absent_agg["star_player_out"] = absent_agg["star_player_out"].fillna(0).astype(int)

        # ── Compute total expected minutes per team-game ──────────────────────
        # total_expected_minutes = absent rotation minutes + played rotation minutes.
        # expected_df only contains absent players; we also need rotation players
        # who DID play (their min_roll5 from the game log, in_rotation=True).
        played_rotation = df[df["in_rotation"]][
            ["game_id", "team_id", "min_roll5"]
        ].copy()

        combined_for_total = pd.concat([
            expected_df[["game_id", "team_id", "min_roll5"]],
            played_rotation[["game_id", "team_id", "min_roll5"]],
        ], ignore_index=True)

        expected_agg = (
            combined_for_total
            .groupby(["game_id", "team_id"])["min_roll5"]
            .sum()
            .reset_index()
            .rename(columns={"min_roll5": "total_expected_minutes"})
        )

        result_intermediate = expected_agg.merge(absent_agg, on=["game_id", "team_id"], how="left")

    # ── Build final output (shared by both paths) ──────────────────────────────
    result = result_intermediate.copy()

    result["missing_minutes"]    = result["missing_minutes"].fillna(0)
    result["missing_usg_pct"]    = result["missing_usg_pct"].fillna(0)
    result["n_missing_rotation"] = result["n_missing_rotation"].fillna(0).astype(int)
    result["star_player_out"]    = result["star_player_out"].fillna(0).astype(int)

    result["rotation_availability"] = (
        1.0 - result["missing_minutes"] / result["total_expected_minutes"].replace(0, np.nan)
    ).fillna(1.0).clip(0.0, 1.0)

    # Round floats for cleaner storage
    result["missing_minutes"]       = result["missing_minutes"].round(2)
    result["missing_usg_pct"]       = result["missing_usg_pct"].round(4)
    result["rotation_availability"] = result["rotation_availability"].round(4)

    # Drop the intermediate column
    result = result.drop(columns=["total_expected_minutes"])

    # ── Normalize join keys before returning ──────────────────────────────────
    # Cast game_id to str.strip() and team_id to int so that downstream merges
    # in team_game_features.py use consistent types regardless of how the source
    # CSVs were written. This prevents silent left-join key mismatches.
    result["game_id"] = result["game_id"].astype(str).str.strip()
    result["team_id"] = result["team_id"].astype(int)

    # ── Summary stats ─────────────────────────────────────────────────────────
    print(f"\n-- Summary -------------------------------------------------")
    print(f"  Total team-game rows    : {len(result):,}")
    print(f"  Games with missing mins : "
          f"{(result['missing_minutes'] > 0).sum():,}  "
          f"({(result['missing_minutes'] > 0).mean():.1%})")
    print(f"  Games with star out     : "
          f"{result['star_player_out'].sum():,}  "
          f"({result['star_player_out'].mean():.1%})")
    print(f"  Avg missing minutes     : {result['missing_minutes'].mean():.1f}")
    print(f"  Avg rotation availability: {result['rotation_availability'].mean():.3f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"\nSaved {len(result):,} rows -> {output_path}")

    return result


# ── Real-time injury report helper ────────────────────────────────────────────

def get_todays_injury_report() -> pd.DataFrame:
    """
    Fetch today's official NBA injury report and return a DataFrame of
    players who are OUT or QUESTIONABLE for tonight's games.

    NOTE: Prefer src.data.external.injury_report.get_todays_nba_injury_report()
    which includes PDF fallback and snapshot saving. This function is retained
    for backward compatibility only.

    This is the real-time complement to the historical proxy above.
    For live predictions, use this to identify which rotation players
    will be missing and apply the same missing_minutes logic.

    Returns:
        DataFrame with columns: player_name, team, status, reason, game_date
        Returns empty DataFrame if the report cannot be fetched.

    Note:
        The NBA publishes the official injury report at:
        https://www.nba.com/players/injury-report
        The nba_api package does not expose a clean historical endpoint,
        but the current report is accessible via the CommonPlayerInfo
        and LeagueInjuryReport endpoints.
    """
    try:
        from nba_api.stats.endpoints import leagueinjuryreport
        import time

        time.sleep(1)   # be polite to the API
        report = leagueinjuryreport.LeagueInjuryReport(
            league_id="00",
            season_type="Regular Season",
        ).get_data_frames()[0]

        if report.empty:
            print("  NBA injury report returned empty - may be off-season or no games today.")
            return pd.DataFrame()

        # Normalize column names
        report.columns = [c.lower() for c in report.columns]

        # Filter to OUT and QUESTIONABLE players — PROBABLE players usually play
        report = report[report["player_status"].isin(["Out", "Questionable"])].copy()
        print(f"  Fetched {len(report)} OUT/QUESTIONABLE entries from NBA injury report")
        return report

    except Exception as e:
        print(f"  Could not fetch live injury report: {e}")
        return pd.DataFrame()


def apply_live_injuries(
    matchup_row: pd.Series,
    scratched_player_ids: list,
    player_features_path: str = "data/features/player_game_features.csv",
    artifacts_dir: str = "models/artifacts",
) -> pd.Series:
    """
    Adjust a matchup feature row to reflect known pre-game injuries.

    Given a set of player IDs that won't play tonight, recalculates
    missing_minutes and rotation_availability for the affected team
    and patches the matchup row before feeding it to the model.

    Code path: INFERENCE. This function adjusts features for live prediction.
    It is NOT called during training -- training uses build_injury_proxy_features()
    which derives availability from historical game logs only.

    Args:
        matchup_row:         A single row from game_matchup_features
        scratched_player_ids: List of player_ids confirmed out tonight
        player_features_path: Path to player_game_features.csv

    Returns:
        Updated matchup_row with adjusted injury proxy columns.
    """
    if not scratched_player_ids:
        return matchup_row

    pf = pd.read_csv(player_features_path)
    pf["game_date"] = pd.to_datetime(pf["game_date"])

    row = matchup_row.copy()

    for side in ["home", "away"]:
        team_col = f"{side}_team"
        if team_col not in row.index:
            continue
        team_abbr = row[team_col]

        # Get the latest rolling stats for each scratched player on this team
        team_players = pf[
            (pf["team_abbreviation"] == team_abbr) &
            (pf["player_id"].isin(scratched_player_ids))
        ]
        if team_players.empty:
            continue

        latest = (
            team_players
            .sort_values("game_date")
            .groupby("player_id")
            .last()
            .reset_index()
        )

        added_missing_min = latest["min_roll5"].fillna(0).sum() if "min_roll5" in latest else 0
        added_missing_usg = latest["usg_pct"].fillna(0).sum()   if "usg_pct"  in latest else 0

        miss_col  = f"{side}_missing_minutes"
        usg_col   = f"{side}_missing_usg_pct"
        avail_col = f"{side}_rotation_availability"
        star_col  = f"{side}_star_player_out"

        if miss_col in row.index:
            row[miss_col] = row[miss_col] + added_missing_min
        if usg_col in row.index:
            row[usg_col] = row[usg_col] + added_missing_usg
        if avail_col in row.index:
            row[avail_col] = max(0.0, row[avail_col] - (added_missing_min / 240))
        if star_col in row.index and any(latest.get("usg_pct", pd.Series()).fillna(0) >= STAR_USG_THRESHOLD):
            row[star_col] = 1

    return row


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = build_injury_proxy_features()
    print("\nSample output:")
    print(result[result["missing_minutes"] > 0].head(10).to_string(index=False))
