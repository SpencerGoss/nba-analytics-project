"""
Backfill historical player career stats using PlayerCareerStats endpoint.

The LeagueDashPlayerStats endpoint only goes back to 1996-97, but
PlayerCareerStats returns season-by-season stats for ANY player going
all the way back to the NBA's founding in 1946.

This script:
1. Loads all player IDs from nba_api.stats.static.players
2. Finds players not yet in historical_player_seasons.csv
3. Fetches their career stats via PlayerCareerStats (PerGame)
4. Appends to historical_player_seasons.csv in the same format

Run: python scripts/backfill_career_stats.py
     python scripts/backfill_career_stats.py --limit 100  (test with 100 players)
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.static import players
import logging

log = logging.getLogger(__name__)

PLAYERS_CSV = PROJECT_ROOT / "data" / "raw" / "historical_player_seasons.csv"
THROTTLE_SEC = 0.65
RETRY_SLEEP_SEC = 3.0
FLUSH_EVERY = 50

# Column mapping from PlayerCareerStats -> our CSV format
COL_MAP = {
    "PLAYER_ID": "player_id",
    "SEASON_ID": "season_str",
    "TEAM_ID": "team_id",
    "TEAM_ABBREVIATION": "team_abbreviation",
    "PLAYER_AGE": "age",
    "GP": "gp",
    "GS": "gs",
    "MIN": "min",
    "FGM": "fgm",
    "FGA": "fga",
    "FG_PCT": "fg_pct",
    "FG3M": "fg3m",
    "FG3A": "fg3a",
    "FG3_PCT": "fg3_pct",
    "FTM": "ftm",
    "FTA": "fta",
    "FT_PCT": "ft_pct",
    "OREB": "oreb",
    "DREB": "dreb",
    "REB": "reb",
    "AST": "ast",
    "STL": "stl",
    "BLK": "blk",
    "TOV": "tov",
    "PF": "pf",
    "PTS": "pts",
}


def _season_str_to_int(s: str) -> int:
    """'1969-70' -> 196970"""
    parts = s.split("-")
    return int(parts[0]) * 100 + int(parts[1])


def _load_existing_player_ids() -> set[int]:
    """Return set of player_ids already in the CSV."""
    if not PLAYERS_CSV.exists():
        return set()
    df = pd.read_csv(PLAYERS_CSV, usecols=["player_id"], low_memory=False)
    return set(df["player_id"].unique())


def _fetch_career(player_id: int) -> pd.DataFrame | None:
    """Fetch career stats for one player. Returns None on failure."""
    try:
        r = playercareerstats.PlayerCareerStats(
            player_id=player_id, per_mode36="PerGame"
        )
        df = r.get_data_frames()[0]
        time.sleep(THROTTLE_SEC)
        # Filter to regular season NBA only (league_id 00)
        if "LEAGUE_ID" in df.columns:
            df = df[df["LEAGUE_ID"] == "00"]
        return df if len(df) > 0 else None
    except Exception:
        time.sleep(RETRY_SLEEP_SEC)
        try:
            r = playercareerstats.PlayerCareerStats(
                player_id=player_id, per_mode36="PerGame"
            )
            df = r.get_data_frames()[0]
            time.sleep(THROTTLE_SEC)
            if "LEAGUE_ID" in df.columns:
                df = df[df["LEAGUE_ID"] == "00"]
            return df if len(df) > 0 else None
        except Exception:
            return None


def _get_csv_columns() -> list[str]:
    """Read the header row from the existing CSV to get exact column order."""
    if PLAYERS_CSV.exists():
        with open(PLAYERS_CSV) as f:
            header = f.readline().strip()
        return header.split(",")
    # Fallback: known column order from historical_player_seasons.csv
    return [
        "player_id", "player_name", "nickname", "team_id", "team_abbreviation",
        "age", "gp", "w", "l", "w_pct", "min", "fgm", "fga", "fg_pct",
        "fg3m", "fg3a", "fg3_pct", "ftm", "fta", "ft_pct", "oreb", "dreb",
        "reb", "ast", "tov", "stl", "blk", "blka", "pf", "pfd", "pts",
        "plus_minus", "nba_fantasy_pts", "dd2", "td3", "wnba_fantasy_pts",
        "gp_rank", "w_rank", "l_rank", "w_pct_rank", "min_rank", "fgm_rank",
        "fga_rank", "fg_pct_rank", "fg3m_rank", "fg3a_rank", "fg3_pct_rank",
        "ftm_rank", "fta_rank", "ft_pct_rank", "oreb_rank", "dreb_rank",
        "reb_rank", "ast_rank", "tov_rank", "stl_rank", "blk_rank",
        "blka_rank", "pf_rank", "pfd_rank", "pts_rank", "plus_minus_rank",
        "nba_fantasy_pts_rank", "dd2_rank", "td3_rank", "wnba_fantasy_pts_rank",
        "team_count", "season_str", "season",
    ]


def _normalize(df: pd.DataFrame, player_name: str) -> pd.DataFrame:
    """Rename columns to match historical_player_seasons.csv format exactly."""
    # Only keep columns we have mappings for
    available = {k: v for k, v in COL_MAP.items() if k in df.columns}
    out = df[list(available.keys())].rename(columns=available).copy()

    # Add player_name
    out["player_name"] = player_name

    # Add season integer column
    out["season"] = out["season_str"].apply(_season_str_to_int)

    # Reindex to match the exact CSV column order, filling missing cols with NaN
    csv_cols = _get_csv_columns()
    for col in csv_cols:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[csv_cols]
    return out


def _append_to_csv(frames: list[pd.DataFrame]) -> None:
    """Append frames to the CSV file."""
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    header = not PLAYERS_CSV.exists()
    combined.to_csv(PLAYERS_CSV, mode="a", header=header, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill career stats for players missing from historical CSV."
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max players to fetch (0 = all missing)"
    )
    args = parser.parse_args()

    log.info("Loading existing player IDs...")
    existing = _load_existing_player_ids()
    log.info(f"  {len(existing)} players already in CSV")

    all_players = players.get_players()
    missing = [p for p in all_players if p["id"] not in existing]
    log.warning(f"  {len(missing)} players to backfill")

    if args.limit > 0:
        missing = missing[:args.limit]
        log.info(f"  Limited to {args.limit} players")

    if not missing:
        log.info("Nothing to do.")
        return

    frames: list[pd.DataFrame] = []
    fetched = 0
    skipped = 0
    total_seasons = 0

    for i, p in enumerate(missing, 1):
        pid = p["id"]
        name = p["full_name"]

        if i % 100 == 0 or i == 1:
            log.warning(f"  [{i}/{len(missing)}] Fetching {name}...")

        df = _fetch_career(pid)
        if df is None or df.empty:
            skipped += 1
            continue

        normalized = _normalize(df, name)
        frames.append(normalized)
        fetched += 1
        total_seasons += len(normalized)

        # Flush periodically
        if len(frames) >= FLUSH_EVERY:
            _append_to_csv(frames)
            frames = []
            log.warning(f"    Flushed at {i}/{len(missing)} "
                  f"({fetched} fetched, {skipped} skipped, "
                  f"{total_seasons} total seasons)")

    # Final flush
    _append_to_csv(frames)

    log.warning(f"\nDone. Fetched {fetched} players ({total_seasons} seasons), "
          f"skipped {skipped}.")

    # Verify totals
    df_final = pd.read_csv(PLAYERS_CSV, usecols=["player_id"], low_memory=False)
    log.info(f"CSV now has {df_final['player_id'].nunique()} unique players, "
          f"{len(df_final)} total rows.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
