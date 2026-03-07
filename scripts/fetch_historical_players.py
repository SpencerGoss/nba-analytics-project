"""
Fetch historical NBA player and team season stats from 1946-47 to 2024-25.

Uses LeagueDashPlayerStats / LeagueDashTeamStats — one API call per season.
80 seasons x 1 call = ~80 calls total (much better than per-player endpoints).

Output:
  data/raw/historical_player_seasons.csv
  data/raw/historical_team_seasons.csv

Run: python scripts/fetch_historical_players.py [--start 1996-97] [--end 2024-25]
     python scripts/fetch_historical_players.py --players-only
     python scripts/fetch_historical_players.py --teams-only
"""

import argparse
import time
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

PLAYERS_CSV = RAW_DIR / "historical_player_seasons.csv"
TEAMS_CSV = RAW_DIR / "historical_team_seasons.csv"

# NBA's first season. Three-point line started 1979-80 — fg3 cols will be 0 before that.
FIRST_SEASON_YEAR = 1946
LAST_SEASON_YEAR = 2024  # "2024-25"

THROTTLE_SEC = 0.7  # slightly above the 0.6 floor for safety
RETRY_SLEEP_SEC = 2.0


# ---------------------------------------------------------------------------
# Season string helpers
# ---------------------------------------------------------------------------

def season_int_to_str(season_int: int) -> str:
    """194647 -> '1946-47'"""
    year = season_int // 100
    end_yy = season_int % 100
    return f"{year}-{end_yy:02d}"


def season_str_to_int(season_str: str) -> int:
    """'1946-47' -> 194647"""
    start, end_yy = season_str.split("-")
    return int(start) * 100 + int(end_yy)


def generate_season_strings(first_year: int = FIRST_SEASON_YEAR,
                             last_year: int = LAST_SEASON_YEAR) -> list[str]:
    """Generate NBA season strings, e.g. ['1946-47', '1947-48', ...]."""
    seasons = []
    for year in range(first_year, last_year + 1):
        end_yy = (year + 1) % 100
        seasons.append(f"{year}-{end_yy:02d}")
    return seasons


def parse_season_range(start_arg: str | None, end_arg: str | None) -> list[str]:
    """Parse --start / --end CLI args into a list of season strings."""
    first = int(start_arg.split("-")[0]) if start_arg else FIRST_SEASON_YEAR
    last = int(end_arg.split("-")[0]) if end_arg else LAST_SEASON_YEAR
    return generate_season_strings(first, last)


# ---------------------------------------------------------------------------
# Existing data helpers
# ---------------------------------------------------------------------------

def load_existing_seasons(csv_path: Path) -> set[str]:
    """Return set of season_str values already in csv_path."""
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=["season_str"])
        return set(df["season_str"].unique())
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# API fetch helpers
# ---------------------------------------------------------------------------

def _fetch_player_season(season_str: str) -> pd.DataFrame | None:
    """Fetch all player PerGame stats for one season. Returns None on failure."""
    from nba_api.stats.endpoints import LeagueDashPlayerStats  # local import to allow --help without nba_api

    try:
        df = LeagueDashPlayerStats(
            season=season_str,
            season_type_all_star="Regular Season",
            per_mode_detailed="PerGame",
        ).get_data_frames()[0]
        time.sleep(THROTTLE_SEC)
        return df
    except Exception as exc:
        print(f"  WARNING: first attempt failed for {season_str}: {exc}")
        time.sleep(RETRY_SLEEP_SEC)
        try:
            df = LeagueDashPlayerStats(
                season=season_str,
                season_type_all_star="Regular Season",
                per_mode_detailed="PerGame",
            ).get_data_frames()[0]
            time.sleep(THROTTLE_SEC)
            return df
        except Exception as exc2:
            print(f"  ERROR: skipping {season_str} after retry: {exc2}")
            return None


def _fetch_team_season(season_str: str) -> pd.DataFrame | None:
    """Fetch all team PerGame stats for one season. Returns None on failure."""
    from nba_api.stats.endpoints import LeagueDashTeamStats

    try:
        df = LeagueDashTeamStats(
            season=season_str,
            season_type_all_star="Regular Season",
            per_mode_detailed="PerGame",
        ).get_data_frames()[0]
        time.sleep(THROTTLE_SEC)
        return df
    except Exception as exc:
        print(f"  WARNING: first attempt failed for {season_str}: {exc}")
        time.sleep(RETRY_SLEEP_SEC)
        try:
            df = LeagueDashTeamStats(
                season=season_str,
                season_type_all_star="Regular Season",
                per_mode_detailed="PerGame",
            ).get_data_frames()[0]
            time.sleep(THROTTLE_SEC)
            return df
        except Exception as exc2:
            print(f"  ERROR: skipping {season_str} after retry: {exc2}")
            return None


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase all column names."""
    return df.rename(columns=str.lower)


def _add_season_cols(df: pd.DataFrame, season_str: str) -> pd.DataFrame:
    """Attach season metadata columns."""
    df = df.copy()
    df["season_str"] = season_str
    df["season"] = season_str_to_int(season_str)
    return df


# ---------------------------------------------------------------------------
# Fetch loops
# ---------------------------------------------------------------------------

def fetch_players(seasons: list[str]) -> None:
    """Fetch player stats for all seasons, appending to PLAYERS_CSV."""
    existing = load_existing_seasons(PLAYERS_CSV)
    to_fetch = [s for s in seasons if s not in existing]

    if not to_fetch:
        print("fetch_historical_players: player data already up to date, nothing to fetch.")
        return

    print(f"fetch_historical_players: fetching {len(to_fetch)} player seasons "
          f"({to_fetch[0]} -> {to_fetch[-1]})")

    first_write = not PLAYERS_CSV.exists()
    frames: list[pd.DataFrame] = []

    for i, season_str in enumerate(to_fetch, 1):
        print(f"  [{i}/{len(to_fetch)}] Fetching players {season_str}...", end=" ", flush=True)
        df = _fetch_player_season(season_str)
        if df is None or df.empty:
            print("skipped (no data)")
            continue
        df = _normalize_columns(df)
        df = _add_season_cols(df, season_str)
        frames.append(df)
        print(f"done ({len(df)} players)")

        # Flush every 10 seasons to avoid losing progress on network error
        if len(frames) >= 10:
            _append_frames(frames, PLAYERS_CSV, first_write and i <= len(frames))
            first_write = False
            frames = []

    if frames:
        _append_frames(frames, PLAYERS_CSV, first_write)

    print(f"fetch_historical_players: player data saved -> {PLAYERS_CSV}")


def fetch_teams(seasons: list[str]) -> None:
    """Fetch team stats for all seasons, appending to TEAMS_CSV."""
    existing = load_existing_seasons(TEAMS_CSV)
    to_fetch = [s for s in seasons if s not in existing]

    if not to_fetch:
        print("fetch_historical_players: team data already up to date, nothing to fetch.")
        return

    print(f"fetch_historical_players: fetching {len(to_fetch)} team seasons "
          f"({to_fetch[0]} -> {to_fetch[-1]})")

    first_write = not TEAMS_CSV.exists()
    frames: list[pd.DataFrame] = []

    for i, season_str in enumerate(to_fetch, 1):
        print(f"  [{i}/{len(to_fetch)}] Fetching teams  {season_str}...", end=" ", flush=True)
        df = _fetch_team_season(season_str)
        if df is None or df.empty:
            print("skipped (no data)")
            continue
        df = _normalize_columns(df)
        df = _add_season_cols(df, season_str)
        frames.append(df)
        print(f"done ({len(df)} teams)")

        if len(frames) >= 10:
            _append_frames(frames, TEAMS_CSV, first_write and i <= len(frames))
            first_write = False
            frames = []

    if frames:
        _append_frames(frames, TEAMS_CSV, first_write)

    print(f"fetch_historical_players: team data saved -> {TEAMS_CSV}")


def _append_frames(frames: list[pd.DataFrame], csv_path: Path, write_header: bool) -> None:
    """Concatenate frames and append (or write) to csv_path."""
    combined = pd.concat(frames, ignore_index=True)
    mode = "w" if write_header else "a"
    combined.to_csv(csv_path, mode=mode, header=write_header, index=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fetch historical NBA player/team season stats (1946-47 to 2024-25)."
    )
    p.add_argument("--start", default=None, metavar="YYYY-YY",
                   help="First season to fetch, e.g. 1996-97 (default: 1946-47)")
    p.add_argument("--end", default=None, metavar="YYYY-YY",
                   help="Last season to fetch, e.g. 2024-25 (default: 2024-25)")
    p.add_argument("--players-only", action="store_true",
                   help="Only fetch player stats, skip team stats")
    p.add_argument("--teams-only", action="store_true",
                   help="Only fetch team stats, skip player stats")
    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    seasons = parse_season_range(args.start, args.end)

    if not args.teams_only:
        fetch_players(seasons)

    if not args.players_only:
        fetch_teams(seasons)

    print("fetch_historical_players: done.")


if __name__ == "__main__":
    main()
