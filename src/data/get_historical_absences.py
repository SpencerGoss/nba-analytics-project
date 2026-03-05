"""
Historical Absence Dataset Builder
====================================
Scans existing player game logs to produce a structured per-player absence
record file at data/processed/player_absences.csv.

Purpose:
  This script backfills historical absence data from game logs using the same
  rotation-detection proxy logic as injury_proxy.py — but outputs per-player
  rows rather than team-level aggregates.

  Downstream consumers (injury_proxy.py feature builder, future real-absence
  replacement) can use this file to look up who was absent for a given game.

Algorithm:
  1. Load data/processed/player_game_logs.csv (same source as injury_proxy.py).
  2. Compute per-player rolling-average minutes using shift(1) before rolling(5)
     so that row N only sees data from rows 0..N-1 (no leakage).
  3. For each team-game, determine which players were in the rotation before
     that game (min_roll5 >= 15.0 over at least 2 of the prior 5 games).
  4. Use merge_asof (direction="backward") to project each player's last known
     rotation status onto all team-game dates.
  5. Anti-join against actual appearances: rotation players with no game-log
     entry for a given game_id are marked was_absent=1.
  6. Append rows for rotation players who DID appear (was_absent=0).
  7. Save to data/processed/player_absences.csv and return the DataFrame.

Output schema:
  player_id   (int)    — NBA player ID
  player_name (str)    — player display name
  team_id     (int)    — NBA team ID
  game_id     (str)    — game identifier, normalized to string
  game_date   (str)    — YYYY-MM-DD
  season      (int)    — integer season code, e.g. 202425
  min_roll5   (float)  — rolling avg minutes from prior 5 games (shift-1)
  usg_pct     (float)  — season usage rate, filled 0.18 if missing
  was_absent  (int)    — 1 if rotation player absent, 0 if played

Leakage note:
  shift(1) is applied BEFORE .rolling() so the current game's minutes are
  never included in min_roll5. This is the same pattern as injury_proxy.py.

Code path boundary:
  TRAINING PATH only. Never import from src.data.external.injury_report here.

Usage:
  python src/data/get_historical_absences.py
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# ── Config ────────────────────────────────────────────────────────────────────

GAME_LOG_PATH = "data/processed/player_game_logs.csv"
ADV_STATS_PATH = "data/processed/player_stats_advanced.csv"
OUTPUT_PATH = "data/processed/player_absences.csv"

# Rotation threshold: player must average >= this many minutes over last 5 games
MIN_ROTATION_MINUTES = 15.0

# Minimum games in the rolling window before flagging a player as rotation
MIN_ROTATION_GAMES = 2

# Rolling window size (games)
ROLL_WINDOW = 5

# A player's last known rotation status is considered stale after this many days.
# Beyond this cutoff, we don't project their status onto new games.
MAX_STALE_DAYS = ROLL_WINDOW * 5   # 25 days


# ── Core builder ──────────────────────────────────────────────────────────────

def build_player_absences(
    game_log_path: str = GAME_LOG_PATH,
    adv_stats_path: str = ADV_STATS_PATH,
    output_path: str = OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Derive per-player absence records from existing game log CSVs.

    Returns a DataFrame with columns:
        player_id, player_name, team_id, game_id, game_date, season,
        min_roll5, usg_pct, was_absent
    """
    print("=" * 60)
    print("HISTORICAL ABSENCE DATASET BUILDER")
    print("=" * 60)

    # ── Load player game logs ─────────────────────────────────────────────────
    print("\nLoading player_game_logs...")
    cols = [
        "season", "player_id", "player_name", "team_id",
        "team_abbreviation", "game_id", "game_date", "min",
    ]
    df = pd.read_csv(game_log_path, usecols=cols)
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Ensure minutes is numeric — handle "MM:SS" strings from older seasons
    df["min"] = pd.to_numeric(df["min"], errors="coerce").fillna(0)

    # Normalize join keys: game_id as str, team_id as int, season as int
    df["game_id"] = df["game_id"].astype(str).str.strip()
    df["team_id"] = df["team_id"].astype(int)
    df["season"] = df["season"].astype(int)

    # Sort chronologically within each player for correct rolling computation
    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    print(f"  Loaded {len(df):,} player-game rows")

    # ── Load season-level usage rates ─────────────────────────────────────────
    print("Loading advanced stats for usage rates...")
    try:
        adv = pd.read_csv(adv_stats_path, usecols=["player_id", "season", "usg_pct"])
        adv = adv.drop_duplicates(subset=["player_id", "season"])
        adv["usg_pct"] = adv["usg_pct"].fillna(0.18)
        adv["season"] = adv["season"].astype(int)
        df = df.merge(adv, on=["player_id", "season"], how="left")
    except Exception as e:
        print(f"  Warning: could not load advanced stats ({e}). Using default usg_pct=0.18")
        df["usg_pct"] = 0.18

    df["usg_pct"] = df["usg_pct"].fillna(0.18)

    # ── Compute rolling baseline minutes per player ───────────────────────────
    # CRITICAL: shift(1) BEFORE rolling so row N only uses rows 0..N-1.
    # Do NOT use .rolling().shift() — the shift must come first.
    print("Computing rolling baseline minutes per player (shift-1 before rolling)...")

    df["min_roll5"] = (
        df.groupby("player_id")["min"]
        .transform(
            lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=MIN_ROTATION_GAMES).mean()
        )
    )
    df["games_in_roll5"] = (
        df.groupby("player_id")["min"]
        .transform(
            lambda x: x.shift(1).rolling(ROLL_WINDOW, min_periods=1).count()
        )
    )

    # in_rotation: based entirely on prior-game rolling stats (shift-1).
    # This is the pre-game expectation — does NOT use the current game's minutes.
    df["in_rotation"] = (
        (df["min_roll5"] >= MIN_ROTATION_MINUTES)
        & (df["games_in_roll5"] >= MIN_ROTATION_GAMES)
    )

    print(f"  Rotation player-games identified: {df['in_rotation'].sum():,}")

    # ── Build all (team_id, game_id, game_date) pairs ─────────────────────────
    team_games = (
        df[["team_id", "game_id", "game_date"]]
        .drop_duplicates()
        .copy()
    )

    # Player status timeline per player per team, sorted for merge_asof
    player_status = (
        df[["player_id", "player_name", "team_id", "game_date",
            "in_rotation", "min_roll5", "usg_pct"]]
        .copy()
        .sort_values(["player_id", "team_id", "game_date"])
    )

    # Actual appearances: all (game_id, team_id, player_id) in the log
    actual_played = (
        df[["game_id", "team_id", "player_id"]]
        .drop_duplicates()
        .assign(played=True)
    )

    # ── Vectorized expected-rotation builder using merge_asof ─────────────────
    # For each (team, player), project last known rotation status onto all
    # team-game dates. This is the same approach as injury_proxy.py.
    print("Projecting rotation status onto all team-game dates...")
    expected_parts = []

    for team_id_val, tg_grp in team_games.groupby("team_id"):
        ps_team = player_status[player_status["team_id"] == team_id_val]
        if ps_team.empty:
            continue

        tg_sorted = (
            tg_grp.sort_values("game_date")[["team_id", "game_id", "game_date"]]
        )

        for player_id_val, player_grp in ps_team.groupby("player_id"):
            # Get player name from the group
            player_name_val = player_grp["player_name"].iloc[0]

            ps_sorted = (
                player_grp
                .sort_values("game_date")
                .rename(columns={"game_date": "last_date"})
                [["last_date", "in_rotation", "min_roll5", "usg_pct"]]
            )

            # merge_asof: for each team-game date, find the player's last
            # prior appearance (direction="backward")
            merged = pd.merge_asof(
                tg_sorted,
                ps_sorted,
                left_on="game_date",
                right_on="last_date",
                direction="backward",
            )

            # Only keep rows where player had a prior appearance as a rotation player
            # AND that appearance was strictly before the game (days_since > 0)
            # AND not stale (days_since <= MAX_STALE_DAYS)
            merged = merged[merged["last_date"].notna()].copy()
            if merged.empty:
                continue

            merged["days_since"] = (merged["game_date"] - merged["last_date"]).dt.days
            merged = merged[
                (merged["in_rotation"])
                & (merged["days_since"] > 0)
                & (merged["days_since"] <= MAX_STALE_DAYS)
            ]
            if merged.empty:
                continue

            merged["player_id"] = player_id_val
            merged["player_name"] = player_name_val
            expected_parts.append(
                merged[["game_id", "team_id", "game_date", "player_id",
                         "player_name", "min_roll5", "usg_pct"]]
            )

    # ── Build played-rotation rows (was_absent=0) — always computed ───────────
    # These are rotation players who appeared in the game log. We include them
    # so the output covers all rotation player-games (absent AND present), enabling
    # downstream consumers to look up any player's rolling stats for a given game.
    team_game_season = (
        df[["game_id", "team_id", "season"]]
        .drop_duplicates(subset=["game_id", "team_id"])
    )

    played_rotation = df[df["in_rotation"]].copy()
    played_rotation["was_absent"] = 0
    played_rotation["game_date"] = pd.to_datetime(
        played_rotation["game_date"]
    ).dt.strftime("%Y-%m-%d")

    _PLAYED_COLS = [
        "player_id", "player_name", "team_id", "game_id",
        "game_date", "season", "min_roll5", "usg_pct", "was_absent",
    ]
    played_rotation_slim = played_rotation[_PLAYED_COLS].copy()

    _OUTPUT_COLS = _PLAYED_COLS  # same schema

    if not expected_parts:
        print("  No absent rotation appearances found — only played rows in output.")
        result = played_rotation_slim.copy()
        result["player_id"] = result["player_id"].astype(int)
        result["team_id"] = result["team_id"].astype(int)
        result["game_id"] = result["game_id"].astype(str).str.strip()
        result["season"] = result["season"].astype(int)
        result["min_roll5"] = result["min_roll5"].round(2)
        result["usg_pct"] = result["usg_pct"].round(4)
        result["was_absent"] = result["was_absent"].astype(int)
        result = result.dropna(subset=["season"])
        result["season"] = result["season"].astype(int)
        result = result.sort_values(
            ["season", "team_id", "game_date", "player_id"]
        ).reset_index(drop=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result.to_csv(output_path, index=False)
        print(f"\nSaved {len(result):,} rows -> {output_path}")
        return result

    expected_df = pd.concat(expected_parts, ignore_index=True)
    print(f"  Expected rotation appearances: {len(expected_df):,}")

    # ── Anti-join: expected players who did NOT appear in the actual log ───────
    merged_check = expected_df.merge(
        actual_played,
        on=["game_id", "team_id", "player_id"],
        how="left",
    )

    # was_absent=1 when played is NaN (player expected but not in game log)
    merged_check["was_absent"] = merged_check["played"].isna().astype(int)
    merged_check = merged_check.drop(columns=["played"])

    print(
        f"  Absent rotation instances (was_absent=1): "
        f"{(merged_check['was_absent'] == 1).sum():,}"
    )

    # ── Attach season and format game_date on absent rows ─────────────────────
    merged_check = merged_check.merge(
        team_game_season,
        on=["game_id", "team_id"],
        how="left",
    )
    merged_check["game_date"] = pd.to_datetime(
        merged_check["game_date"]
    ).dt.strftime("%Y-%m-%d")

    # ── Build final output ────────────────────────────────────────────────────
    _OUTPUT_COLS = [
        "player_id", "player_name", "team_id", "game_id",
        "game_date", "season", "min_roll5", "usg_pct", "was_absent",
    ]

    # Combine absent rows (was_absent=1) and played-rotation rows (was_absent=0)
    # Prefer absent rows if there's a conflict (shouldn't happen, but guard it)
    combined = pd.concat(
        [merged_check[_OUTPUT_COLS], played_rotation_slim[_OUTPUT_COLS]],
        ignore_index=True,
    )
    # If a player-game appears in both (edge case), keep was_absent=1 version
    combined = combined.sort_values("was_absent", ascending=False)
    combined = combined.drop_duplicates(
        subset=["player_id", "game_id", "team_id"], keep="first"
    )

    result = combined.copy()

    # Normalize types
    result["player_id"] = result["player_id"].astype(int)
    result["team_id"] = result["team_id"].astype(int)
    result["game_id"] = result["game_id"].astype(str).str.strip()
    result["game_date"] = pd.to_datetime(result["game_date"]).dt.strftime("%Y-%m-%d")
    result["season"] = result["season"].astype(int)
    result["min_roll5"] = result["min_roll5"].round(2)
    result["usg_pct"] = result["usg_pct"].round(4)
    result["was_absent"] = result["was_absent"].astype(int)

    # Drop rows with null season (shouldn't happen but guard against it)
    result = result.dropna(subset=["season"])
    result["season"] = result["season"].astype(int)

    # Sort for reproducibility
    result = result.sort_values(
        ["season", "team_id", "game_date", "player_id"]
    ).reset_index(drop=True)

    # ── Summary stats ─────────────────────────────────────────────────────────
    absent_rate = (result["was_absent"] == 1).mean()
    print(f"\n-- Summary -------------------------------------------------")
    print(f"  Total rows            : {len(result):,}")
    print(f"  Unique players        : {result['player_id'].nunique():,}")
    print(f"  Unique games          : {result['game_id'].nunique():,}")
    print(f"  was_absent=1 (absent) : {(result['was_absent'] == 1).sum():,} ({absent_rate:.1%})")
    print(f"  was_absent=0 (played) : {(result['was_absent'] == 0).sum():,}")
    print(f"  Seasons covered       : {sorted(result['season'].unique().tolist())}")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"\nSaved {len(result):,} rows -> {output_path}")

    return result


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = build_player_absences()
    print("\nSample absent rows:")
    absent_sample = result[result["was_absent"] == 1].head(5)
    print(absent_sample.to_string(index=False))
