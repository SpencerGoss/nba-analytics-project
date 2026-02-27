"""
Injury Proxy Feature Engineering
===================================
Builds pre-game lineup availability features from player game logs.

Since the NBA API does not expose a historical injury timeline, we infer
player availability using a proxy approach: a rotation player is anyone
who has averaged 15+ minutes over their last 5 games. If such a player
has no entry in the game log for a given game, they almost certainly
did not play — whether due to injury, rest, or coach's decision.

This gives the model four meaningful signals for each team entering a game:

  missing_minutes        — Total expected minutes from absent rotation players.
                           A value of 35 means ~35 min of expected production
                           is unavailable, roughly equivalent to losing a starter.

  missing_usg_pct        — Combined usage rate of absent players. Usage rate
                           measures what % of team possessions a player uses;
                           a missing player at 0.28 usg_pct is a much bigger
                           deal than one at 0.12.

  rotation_availability  — Expected minutes available / total expected minutes.
                           1.0 = full strength. 0.7 = team is missing 30% of
                           their normal rotation's playing time.

  star_player_out        — Binary flag. 1 if any absent player has a season
                           usage rate >= 25% (the threshold commonly used to
                           define a team's primary ball-handler / star).

These features are computed per (team_id, game_id) and merged into
team_game_features, then flow through to the matchup table as
home_/away_ prefixed columns and a diff_missing_minutes differential.

Leakage note:
  All rolling stats use shift(1) so only prior games inform the
  "expected minutes" baseline. The fact that a player is absent in the
  current game's log is pre-game-observable information — NBA teams
  publish official injury reports ~1 hour before tip-off. For real-time
  predictions, pair this module with get_todays_injury_report() below.

Usage:
    from src.features.injury_proxy import build_injury_proxy_features
    df = build_injury_proxy_features()

    # Or for a live game prediction:
    from src.features.injury_proxy import get_todays_injury_report
    scratched = get_todays_injury_report()

    python src/features/injury_proxy.py
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ── Config ─────────────────────────────────────────────────────────────────────

GAME_LOG_PATH  = "data/processed/player_game_logs.csv"
ADV_STATS_PATH = "data/processed/player_stats_advanced.csv"
OUTPUT_PATH    = "data/features/injury_proxy_features.csv"

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


# ── Core builder ───────────────────────────────────────────────────────────────

def build_injury_proxy_features(
    game_log_path:  str = GAME_LOG_PATH,
    adv_stats_path: str = ADV_STATS_PATH,
    output_path:    str = OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Build per-(team_id, game_id) lineup availability features.

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

    # ── Load player game logs ─────────────────────────────────────────────────
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

    # ── Load season-level usage rates ─────────────────────────────────────────
    print("Loading advanced stats for usage rates...")
    adv = pd.read_csv(adv_stats_path, usecols=["player_id", "season", "usg_pct"])
    adv = adv.drop_duplicates(subset=["player_id", "season"])
    # Fill missing usage with a neutral league-average estimate
    adv["usg_pct"] = adv["usg_pct"].fillna(0.18)

    df = df.merge(adv, on=["player_id", "season"], how="left")
    df["usg_pct"] = df["usg_pct"].fillna(0.18)

    # ── Compute rolling baseline minutes per player ───────────────────────────
    # shift(1) means we only know what they AVERAGED in the 5 games before this one.
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
    #   - They averaged MIN_ROTATION_MINUTES+ over their last 5 games
    #   - They appeared in at least MIN_ROTATION_GAMES of those 5 games
    df["in_rotation"] = (
        (df["min_roll5"] >= MIN_ROTATION_MINUTES) &
        (df["games_in_roll5"] >= MIN_ROTATION_GAMES)
    )

    rotation = df[df["in_rotation"]].copy()
    print(f"  Rotation player-game entries: {len(rotation):,}")

    # ── Build actual-played lookup ────────────────────────────────────────────
    # This is who DID play — our reference set for "absent = not in this table"
    actual_played = (
        df[["game_id", "team_id", "player_id"]]
        .assign(played=True)
    )

    # ── Anti-join: rotation members who did NOT play ──────────────────────────
    print("Identifying absent rotation players per team-game...")

    merged = rotation[["game_id", "team_id", "player_id", "min_roll5", "usg_pct"]].merge(
        actual_played,
        on=["game_id", "team_id", "player_id"],
        how="left",
    )
    merged["played"] = merged["played"].fillna(False)
    absent = merged[~merged["played"]].copy()

    print(f"  Absent rotation instances: {len(absent):,}")
    if len(absent) > 0:
        print(f"  Games with at least one absent rotation player: "
              f"{absent['game_id'].nunique():,}")

    # ── Aggregate absent players per (team_id, game_id) ──────────────────────
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

    # ── Compute total expected minutes per team-game ──────────────────────────
    expected_agg = (
        rotation
        .groupby(["game_id", "team_id"])["min_roll5"]
        .sum()
        .reset_index()
        .rename(columns={"min_roll5": "total_expected_minutes"})
    )

    # ── Build final output ────────────────────────────────────────────────────
    # Start from expected_agg (all team-games with a rotation)
    result = expected_agg.merge(absent_agg, on=["game_id", "team_id"], how="left")

    result["missing_minutes"]     = result["missing_minutes"].fillna(0)
    result["missing_usg_pct"]     = result["missing_usg_pct"].fillna(0)
    result["n_missing_rotation"]  = result["n_missing_rotation"].fillna(0).astype(int)
    result["star_player_out"]     = result["star_player_out"].fillna(0).astype(int)

    result["rotation_availability"] = (
        1.0 - result["missing_minutes"] / result["total_expected_minutes"].replace(0, np.nan)
    ).fillna(1.0).clip(0.0, 1.0)

    # Round floats for cleaner storage
    result["missing_minutes"]    = result["missing_minutes"].round(2)
    result["missing_usg_pct"]    = result["missing_usg_pct"].round(4)
    result["rotation_availability"] = result["rotation_availability"].round(4)

    # Drop the intermediate column
    result = result.drop(columns=["total_expected_minutes"])

    # ── Summary stats ─────────────────────────────────────────────────────────
    print(f"\n── Summary ──────────────────────────────────────────────")
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
    print(f"\nSaved {len(result):,} rows → {output_path}")

    return result


# ── Real-time injury report helper ────────────────────────────────────────────

def get_todays_injury_report() -> pd.DataFrame:
    """
    Fetch today's official NBA injury report and return a DataFrame of
    players who are OUT or QUESTIONABLE for tonight's games.

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
            print("  NBA injury report returned empty — may be off-season or no games today.")
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
