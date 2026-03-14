"""
Referee Foul-Rate Rolling Feature Builder
==========================================
Computes per-crew rolling foul-rate features from scraped Basketball Reference
referee assignment data.

Features produced (per game, crew-level):
    ref_crew_fta_rate_roll10  -- rolling 10-game mean FTA/game for this crew
    ref_crew_fta_rate_roll20  -- rolling 20-game mean FTA/game for this crew
    ref_crew_pace_impact_roll10 -- crew's rolling avg possessions vs league avg

Data dependency:
    data/raw/external/referee_crew/referee_crew_*.csv   (from bref_scraper.py)
    data/processed/team_game_logs.csv                   (FTA and pace per game)

Pitfall 6 compliance (RESEARCH.md):
    All features default to NaN (NOT zero) for pre-scrape games.
    Do NOT fillna(0) on referee features -- zero would inject false signal
    into training data for seasons without referee coverage.
    The model's SimpleImputer handles NaN via mean imputation.

NFR-1 (no lookahead) compliance:
    All rolling windows use shift(1) BEFORE rolling() so the current game's
    referee assignments are never used to compute their own rolling stat.

Usage:
    from src.features.referee_features import build_referee_features
    ref_df = build_referee_features()

    # Returns DataFrame with one row per game that has referee data.
    # Games without referee data are not included; the join in
    # team_game_features.py uses a left join so those games get NaN.
"""

import os
import glob
import warnings

import numpy as np
import pandas as pd
import logging

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

REFEREE_DATA_DIR = "data/raw/external/referee_crew/"
OUTPUT_PATH = "data/features/referee_features.csv"
TEAM_GAME_LOG_PATH = "data/processed/team_game_logs.csv"

ROLL_WINDOWS = [10, 20]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_referee_assignments(referee_data_dir: str) -> pd.DataFrame:
    """Load and concatenate all referee crew CSV files from the data directory.

    Args:
        referee_data_dir: Path to directory containing referee_crew_*.csv files.

    Returns:
        DataFrame with columns: game_date, game_id_bref, home_team, away_team,
        referee_1, referee_2, referee_3.
        Returns empty DataFrame if no files found.
    """
    pattern = os.path.join(referee_data_dir, "referee_crew_*.csv")
    files = glob.glob(pattern)

    if not files:
        return pd.DataFrame(columns=[
            "game_date", "game_id_bref", "home_team", "away_team",
            "referee_1", "referee_2", "referee_3",
        ])

    dfs = []
    for f in sorted(files):
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            warnings.warn(f"Could not read {f}: {e}")

    if not dfs:
        return pd.DataFrame(columns=[
            "game_date", "game_id_bref", "home_team", "away_team",
            "referee_1", "referee_2", "referee_3",
        ])

    combined = pd.concat(dfs, ignore_index=True)

    # Normalize game_date to datetime then back to string YYYY-MM-DD
    combined["game_date"] = pd.to_datetime(combined["game_date"], format="mixed").dt.strftime("%Y-%m-%d")

    # Drop duplicate games (same game_id_bref may appear in overlapping date ranges)
    combined = combined.drop_duplicates(subset=["game_id_bref"])

    return combined


def _melt_to_long_format(crew_df: pd.DataFrame) -> pd.DataFrame:
    """Expand referee crew assignments to one row per (game, referee).

    Input: one row per game with referee_1/2/3 columns.
    Output: one row per (game, referee) with a single 'referee' column.

    Filters out rows where referee is NaN (game missing official data).

    Args:
        crew_df: DataFrame with game_date, game_id_bref, home_team, away_team,
                 referee_1, referee_2, referee_3.

    Returns:
        DataFrame with columns: game_date, game_id_bref, home_team, away_team,
        referee (name string).
    """
    ref_cols = [c for c in ["referee_1", "referee_2", "referee_3"] if c in crew_df.columns]

    long = crew_df.melt(
        id_vars=["game_date", "game_id_bref", "home_team", "away_team"],
        value_vars=ref_cols,
        var_name="ref_slot",
        value_name="referee",
    )
    # Drop games/slots with no referee data
    long = long.dropna(subset=["referee"])
    long = long[long["referee"].str.strip() != ""]
    long = long.drop(columns=["ref_slot"]).reset_index(drop=True)

    return long


def _join_game_fta(long_df: pd.DataFrame, game_logs: pd.DataFrame) -> pd.DataFrame:
    """Join FTA per game onto the referee-per-game long format.

    Strategy:
    - game_logs has one row per team per game (home AND away)
    - For each game, average the home and away FTA to get one FTA number
    - Join this averaged FTA to referee assignments on game_date + team_abbr

    The bref_scraper stores home_team in Basketball Reference 3-char format
    (e.g. "LAL"), which matches team_abbreviation in NBA API game logs.

    Args:
        long_df: One row per (game, referee) with home_team/away_team columns.
        game_logs: team_game_logs with team_abbreviation, game_date, fta.

    Returns:
        long_df enriched with: fta_per_game (avg of home + away FTA), avg_poss.
    """
    # Compute average FTA and avg_poss for each game (average home + away values)
    game_logs_clean = game_logs.copy()
    game_logs_clean["game_date"] = pd.to_datetime(game_logs_clean["game_date"], format="mixed").dt.strftime("%Y-%m-%d")

    # Possession estimate (Oliver formula): FGA - OREB + TOV + 0.44 * FTA
    has_poss_cols = all(c in game_logs_clean.columns for c in ["fga", "oreb", "tov", "fta"])
    if has_poss_cols:
        game_logs_clean["poss_est"] = (
            game_logs_clean["fga"]
            - game_logs_clean["oreb"].fillna(0)
            + game_logs_clean["tov"].fillna(0)
            + 0.44 * game_logs_clean["fta"].fillna(0)
        )
    else:
        game_logs_clean["poss_est"] = np.nan

    # Per-game aggregates: average across home and away team
    # We join referee to home_team abbreviation; FTA averaged across both teams
    per_game_agg = (
        game_logs_clean
        .groupby(["game_date", "game_id"])[["fta", "poss_est"]]
        .mean()
        .reset_index()
        .rename(columns={
            "fta": "fta_per_game",
            "poss_est": "avg_poss",
            "game_id": "nba_game_id",
        })
    )

    # Join using game_date + home_team: match bref home_team abbr to NBA API
    # The home team row in game_logs has matchup "TEAM vs. OPP" (contains "vs.")
    home_rows = game_logs_clean[
        game_logs_clean["matchup"].str.contains(" vs\\.", regex=True, na=False)
    ][["game_date", "game_id", "team_abbreviation"]].copy()
    home_rows = home_rows.rename(columns={"game_id": "nba_game_id"})

    # Merge home team abbr back to per-game agg so we can join on home_team name
    per_game_with_home = per_game_agg.merge(
        home_rows,
        on=["game_date", "nba_game_id"],
        how="left",
    )
    # team_abbreviation is now the home team's abbreviation
    per_game_with_home = per_game_with_home.rename(
        columns={"team_abbreviation": "home_team_abbr"}
    )

    # Join to referee long_df on (game_date, home_team)
    # NOTE: bref home_team uses 3-char abbreviations from BR, which match NBA API
    # abbreviations for modern teams (minor historical exceptions exist post-relocation)
    result = long_df.merge(
        per_game_with_home[["game_date", "home_team_abbr", "fta_per_game", "avg_poss", "nba_game_id"]],
        left_on=["game_date", "home_team"],
        right_on=["game_date", "home_team_abbr"],
        how="left",
    ).drop(columns=["home_team_abbr"])

    return result


def _compute_referee_rolling_stats(long_df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling FTA/game and pace stats per referee.

    For each referee, sorts their game history by date and computes:
      - ref_fta_rate_roll10: rolling 10-game mean of fta_per_game (shift-1)
      - ref_fta_rate_roll20: rolling 20-game mean of fta_per_game (shift-1)
      - ref_poss_roll10: rolling 10-game mean of avg_poss (shift-1)

    shift(1) before rolling() ensures we only use games BEFORE the current one
    (NFR-1 no-lookahead compliance).

    Args:
        long_df: One row per (game, referee) with fta_per_game and avg_poss.

    Returns:
        long_df with additional columns: ref_fta_rate_roll10, ref_fta_rate_roll20,
        ref_poss_roll10.
    """
    long_df = long_df.sort_values(["referee", "game_date"]).reset_index(drop=True)

    # Use transform so the result always aligns back to the original index
    # regardless of the number of unique referees in the input.
    # Rolling FTA/game per referee (shift-1 to prevent leakage)
    long_df["ref_fta_rate_roll10"] = (
        long_df.groupby("referee")["fta_per_game"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    )

    long_df["ref_fta_rate_roll20"] = (
        long_df.groupby("referee")["fta_per_game"]
        .transform(lambda x: x.shift(1).rolling(20, min_periods=1).mean())
    )

    # Rolling pace (avg possessions) per referee
    long_df["ref_poss_roll10"] = (
        long_df.groupby("referee")["avg_poss"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    )

    return long_df


def _aggregate_crew_stats(long_df: pd.DataFrame, game_logs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate individual referee rolling stats to crew-level per game.

    For each game (identified by game_id_bref), average the rolling stats
    across all referees in the crew to get crew-level features.

    Also computes ref_crew_pace_impact_roll10:
        crew's rolling avg possessions minus league-wide rolling avg possessions.
    Positive = this crew's games tend to have more possessions than average.

    Args:
        long_df: One row per (game, referee) with per-referee rolling stats.
        game_logs: team_game_logs for computing league avg possessions.

    Returns:
        DataFrame with one row per game: game_date, game_id_bref, home_team,
        away_team, ref_crew_fta_rate_roll10, ref_crew_fta_rate_roll20,
        ref_crew_pace_impact_roll10.
    """
    # Average rolling stats across the crew for each game
    crew_stats = (
        long_df
        .groupby(["game_date", "game_id_bref", "home_team", "away_team"])[
            ["ref_fta_rate_roll10", "ref_fta_rate_roll20", "ref_poss_roll10"]
        ]
        .mean()
        .reset_index()
        .rename(columns={
            "ref_fta_rate_roll10": "ref_crew_fta_rate_roll10",
            "ref_fta_rate_roll20": "ref_crew_fta_rate_roll20",
        })
    )

    # Compute league-wide rolling average possessions to use as baseline
    # for pace impact: use a 10-game rolling mean of all games' avg_poss
    # sorted by date
    game_logs_clean = game_logs.copy()
    game_logs_clean["game_date"] = pd.to_datetime(
        game_logs_clean["game_date"]
    ).dt.strftime("%Y-%m-%d")

    has_poss_cols = all(c in game_logs_clean.columns for c in ["fga", "oreb", "tov", "fta"])
    if has_poss_cols:
        game_logs_clean["poss_est"] = (
            game_logs_clean["fga"]
            - game_logs_clean["oreb"].fillna(0)
            + game_logs_clean["tov"].fillna(0)
            + 0.44 * game_logs_clean["fta"].fillna(0)
        )
        # League avg pace per game date: average across all teams playing that day
        league_pace = (
            game_logs_clean
            .groupby("game_date")["poss_est"]
            .mean()
            .reset_index()
            .rename(columns={"poss_est": "league_avg_poss"})
            .sort_values("game_date")
        )
        # Rolling 10-game league average (shift-1 for no leakage)
        league_pace["league_poss_roll10"] = (
            league_pace["league_avg_poss"]
            .shift(1)
            .rolling(10, min_periods=1)
            .mean()
        )

        crew_stats = crew_stats.merge(league_pace[["game_date", "league_poss_roll10"]], on="game_date", how="left")
        crew_stats["ref_crew_pace_impact_roll10"] = (
            crew_stats["ref_poss_roll10"] - crew_stats["league_poss_roll10"]
        )
        crew_stats = crew_stats.drop(columns=["ref_poss_roll10", "league_poss_roll10"])
    else:
        crew_stats["ref_crew_pace_impact_roll10"] = np.nan
        crew_stats = crew_stats.drop(columns=["ref_poss_roll10"], errors="ignore")

    return crew_stats


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def build_referee_features(
    referee_data_dir: str = REFEREE_DATA_DIR,
    team_game_log_path: str = TEAM_GAME_LOG_PATH,
    output_path: str = OUTPUT_PATH,
) -> pd.DataFrame:
    """Build per-game referee foul-rate features from scraped crew assignments.

    For each game with referee data:
    1. Load all scraped referee CSV files from referee_data_dir
    2. Melt to long format (one row per game x referee)
    3. Join FTA and pace from team_game_logs
    4. Compute rolling 10 and 20-game FTA averages per referee (shift-1, no leakage)
    5. Average across the 3-ref crew to get crew-level stats
    6. Compute pace impact relative to league average

    Returns DataFrame with columns:
        game_date (str: YYYY-MM-DD),
        game_id_bref (str: Basketball Reference game ID),
        home_team (str: 3-char abbreviation),
        away_team (str: 3-char abbreviation or None),
        ref_crew_fta_rate_roll10 (float or NaN),
        ref_crew_fta_rate_roll20 (float or NaN),
        ref_crew_pace_impact_roll10 (float or NaN)

    TRAINING PATH ONLY for historical games.
    NaN for games without referee data (SimpleImputer handles it).
    Do NOT fillna(0) on referee features -- zero injects false signal.

    Args:
        referee_data_dir: Directory containing referee_crew_*.csv files.
        team_game_log_path: Path to team_game_logs.csv for FTA/pace data.
        output_path: Path to save the output CSV (directory created if needed).

    Returns:
        DataFrame with referee crew features. Empty DataFrame if no referee
        data exists yet (returns correct empty schema for graceful degradation).
    """
    log.info(f"Loading referee assignments from {referee_data_dir}...")
    crew_df = _load_referee_assignments(referee_data_dir)

    if crew_df.empty:
        log.warning("  No referee CSV files found (no scrape data yet). Returning empty DataFrame.")
        return pd.DataFrame(columns=[
            "game_date", "game_id_bref", "home_team", "away_team",
            "ref_crew_fta_rate_roll10", "ref_crew_fta_rate_roll20",
            "ref_crew_pace_impact_roll10",
        ])

    n_games = len(crew_df)
    log.info(f"  Loaded {n_games:,} game assignments from referee CSV files.")

    # Load team game logs (needed for FTA and pace per game)
    if not os.path.exists(team_game_log_path):
        warnings.warn(
            f"team_game_logs not found at {team_game_log_path}. "
            "Returning referee assignments without FTA features."
        )
        # Return crew assignments with NaN for all feature columns
        result = crew_df[["game_date", "game_id_bref", "home_team", "away_team"]].copy()
        result["ref_crew_fta_rate_roll10"] = np.nan
        result["ref_crew_fta_rate_roll20"] = np.nan
        result["ref_crew_pace_impact_roll10"] = np.nan
        return result

    log.info(f"Loading team game logs from {team_game_log_path}...")
    game_logs = pd.read_csv(team_game_log_path)
    game_logs["game_date"] = pd.to_datetime(game_logs["game_date"], format="mixed").dt.strftime("%Y-%m-%d")

    # Step 1: Melt referee assignments to long format (one row per game x referee)
    log.info("Building long-format referee assignments...")
    long_df = _melt_to_long_format(crew_df)

    if long_df.empty:
        log.warning("  No valid referee names found in CSV files. Returning empty DataFrame.")
        return pd.DataFrame(columns=[
            "game_date", "game_id_bref", "home_team", "away_team",
            "ref_crew_fta_rate_roll10", "ref_crew_fta_rate_roll20",
            "ref_crew_pace_impact_roll10",
        ])

    log.info(f"  Long format: {len(long_df):,} rows ({len(long_df['referee'].unique()):,} unique referees)")

    # Step 2: Join FTA and pace from team game logs
    log.info("Joining game FTA and pace from team_game_logs...")
    long_df = _join_game_fta(long_df, game_logs)
    n_fta_matched = long_df["fta_per_game"].notna().sum()
    log.info(f"  FTA join: {n_fta_matched:,} / {len(long_df):,} referee-game rows matched")

    # Step 3: Compute per-referee rolling stats (shift-1 for no lookahead)
    log.info("Computing per-referee rolling FTA stats...")
    long_df = _compute_referee_rolling_stats(long_df)

    # Step 4: Aggregate to crew level (average across 3 referees per game)
    log.info("Aggregating to crew-level features...")
    result = _aggregate_crew_stats(long_df, game_logs)

    n_with_features = result["ref_crew_fta_rate_roll10"].notna().sum()
    log.info(f"  Crew features: {n_with_features:,} / {len(result):,} games have non-NaN ref_crew_fta_rate_roll10")

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_csv(output_path, index=False)
    log.info(f"Saved {len(result):,} rows -> {output_path}")

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = build_referee_features()
    if df.empty:
        log.info("No referee data available yet. Run bref_scraper.py to scrape historical data.")
    else:
        log.debug(df.head(10).to_string(index=False))
        log.info(f"\nShape: {df.shape}")
        log.info(f"NaN rates:\n{df.isna().mean()}")
