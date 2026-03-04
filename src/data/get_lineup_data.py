"""
Fetches 5-man lineup efficiency data from nba_api for all teams.
Saves per-season CSV to data/raw/lineups/lineup_data_{season}.csv

Uses nba_api.stats.endpoints.TeamDashLineups (Advanced, Per100Possessions)
to collect on-court lineup combinations per team per season, then merges
them into a single league-wide file per season.

Output columns:
    season          -- int e.g. 202425
    team_id         -- NBA team ID
    team_abbreviation
    group_name      -- "Player1 - Player2 - Player3 - Player4 - Player5"
    gp              -- games played
    min             -- minutes played
    net_rating      -- offensive rating minus defensive rating
    off_rating      -- offensive rating (pts per 100 possessions)
    def_rating      -- defensive rating (pts allowed per 100 possessions)
    ts_pct          -- true shooting %
    ast_ratio       -- assist ratio
    oreb_pct        -- offensive rebound %
    dreb_pct        -- defensive rebound %
    reb_pct         -- total rebound %

Usage:
    python src/data/get_lineup_data.py
    from src.data.get_lineup_data import get_lineup_data
"""

import os
import sys
import time
import logging

import pandas as pd

# Allow running as a script from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from nba_api.stats.endpoints import teamdashlineups
from nba_api.stats.static import teams as nba_teams_static

from src.data.api_client import fetch_with_retry, HEADERS, configure_logging

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

RAW_LINEUPS_DIR = "data/raw/lineups"

# Minimum games played filter — removes rarely-used lineups (noise)
MIN_GAMES_PLAYED = 5

# All 30 current NBA teams
def _get_all_team_ids() -> list[dict]:
    """Return list of {id, abbreviation} for all active NBA teams."""
    all_teams = nba_teams_static.get_teams()
    return [{"id": t["id"], "abbreviation": t["abbreviation"]} for t in all_teams]


# ── Core fetcher ─────────────────────────────────────────────────────────────

def _fetch_team_lineups(team_id: int, team_abbr: str, season: str) -> pd.DataFrame:
    """
    Fetch 5-man lineup data for one team and season.

    Returns a DataFrame with columns matching the output schema,
    or an empty DataFrame on failure.
    """
    label = f"{team_abbr} {season}"

    result = fetch_with_retry(
        lambda: teamdashlineups.TeamDashLineups(
            team_id=team_id,
            group_quantity=5,
            season=season,
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="Per100Possessions",
            headers=HEADERS,
            timeout=60,
        ).get_data_frames()[1],  # frame[1] = lineup-level data
        label,
    )

    if not result["success"]:
        logger.warning("Skipping %s: %s", label, result["error"])
        return pd.DataFrame()

    df = result["data"]

    if df.empty:
        logger.info("%s: no lineup data returned", label)
        return pd.DataFrame()

    # Filter to lineups with enough games to be meaningful
    if "GP" in df.columns:
        df = df[df["GP"] >= MIN_GAMES_PLAYED].copy()

    if df.empty:
        logger.info("%s: no lineups with GP >= %d", label, MIN_GAMES_PLAYED)
        return pd.DataFrame()

    # Select and rename output columns
    column_map = {
        "GROUP_NAME":    "group_name",
        "GP":            "gp",
        "MIN":           "min",
        "NET_RATING":    "net_rating",
        "OFF_RATING":    "off_rating",
        "DEF_RATING":    "def_rating",
        "TS_PCT":        "ts_pct",
        "AST_RATIO":     "ast_ratio",
        "OREB_PCT":      "oreb_pct",
        "DREB_PCT":      "dreb_pct",
        "REB_PCT":       "reb_pct",
    }

    # Keep only columns that exist in the response
    existing_cols = {k: v for k, v in column_map.items() if k in df.columns}
    output = df[list(existing_cols.keys())].rename(columns=existing_cols)

    # Add team identifiers
    output = output.copy()
    output.insert(0, "team_id", team_id)
    output.insert(1, "team_abbreviation", team_abbr)

    return output


# ── Public interface ──────────────────────────────────────────────────────────

def get_lineup_data(start_year: int = 2015, end_year: int = 2024) -> None:
    """
    Fetch 5-man lineup efficiency data for all teams across a range of seasons.

    Iterates from start_year to end_year (inclusive), fetching per-team lineup
    data from nba_api TeamDashLineups (Advanced, Per100Possessions). Saves one
    CSV file per season to data/raw/lineups/lineup_data_{season}.csv.

    Note: Lineup data is only meaningful from ~2014-15 onward (sufficient sample
    sizes and consistent endpoint availability).

    Parameters
    ----------
    start_year : int
        First season start year (e.g. 2015 for 2015-16 season).
    end_year : int
        Last season start year (e.g. 2024 for 2024-25 season).
    """
    os.makedirs(RAW_LINEUPS_DIR, exist_ok=True)

    all_teams = _get_all_team_ids()
    logger.info(
        "Fetching lineup data for %d teams, seasons %d-%d to %d-%d",
        len(all_teams),
        start_year, (start_year + 1) % 100,
        end_year, (end_year + 1) % 100,
    )

    for year in range(start_year, end_year + 1):
        season_str = f"{year}-{str(year + 1)[-2:]}"
        season_code = f"{year}{str(year + 1)[-2:]}"  # e.g. "202425"
        output_path = os.path.join(RAW_LINEUPS_DIR, f"lineup_data_{season_code}.csv")

        logger.info("=== Season %s ===", season_str)

        season_frames = []

        for team in all_teams:
            team_id = team["id"]
            team_abbr = team["abbreviation"]

            try:
                time.sleep(1)  # rate limit: 1 second between API calls
                df = _fetch_team_lineups(team_id, team_abbr, season_str)

                if not df.empty:
                    season_frames.append(df)
                    logger.info(
                        "  %s: %d lineups (GP >= %d)",
                        team_abbr, len(df), MIN_GAMES_PLAYED,
                    )

            except Exception as exc:
                logger.warning(
                    "  Unexpected error for %s %s: %s — skipping",
                    team_abbr, season_str, exc,
                )
                continue

        if not season_frames:
            logger.warning("No lineup data collected for %s — skipping save", season_str)
            continue

        season_df = pd.concat(season_frames, ignore_index=True)

        # Add season column
        season_df.insert(0, "season", int(season_code))

        season_df.to_csv(output_path, index=False)
        logger.info(
            "Saved %s: %d lineups across %d teams -> %s",
            season_str, len(season_df), len(season_frames), output_path,
        )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    configure_logging()
    # Default: fetch last 2 seasons as a quick test
    get_lineup_data(start_year=2023, end_year=2024)
