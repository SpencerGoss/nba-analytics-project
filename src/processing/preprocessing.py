"""
Preprocessing module for NBA Analytics Project.

Converts all raw seasonal CSV files from data/raw/ into clean, consolidated
processed CSVs in data/processed/.

Supports incremental mode (default): only re-processes tables whose raw source
files have changed since the last processed output was written.  Use
``--full-rebuild`` to force a complete rebuild of every table.

Usage:
    python src/processing/preprocessing.py              # incremental
    python src/processing/preprocessing.py --full-rebuild  # full rebuild
"""

import argparse
import glob
import logging
import os
from pathlib import Path
from typing import Callable

import pandas as pd


# ── Logging setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ── Shared helpers ────────────────────────────────────────────────────────────


def _coerce_int_col(df: pd.DataFrame, col: str) -> None:
    """Coerce column to int, logging any rows that become NaN and dropping them."""
    before = len(df)
    df[col] = pd.to_numeric(df[col], errors="coerce")
    lost = df[col].isna().sum()
    if lost:
        print(f"  WARN: {lost}/{before} rows have non-numeric {col}")
    df.dropna(subset=[col], inplace=True)
    df[col] = df[col].astype(int)


def _needs_rebuild(raw_path: Path, out_path: Path) -> bool:
    """Return True if out_path is missing or older than raw_path."""
    if not out_path.exists():
        return True
    return raw_path.stat().st_mtime > out_path.stat().st_mtime


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize all column names to lowercase_underscore format."""
    df = df.copy()
    df.columns = (
        df.columns
            .str.lower()
            .str.strip()
            .str.replace(" ", "_", regex=False)
            .str.replace("-", "_", regex=False)
            .str.replace("/", "_", regex=False)
    )
    return df


def get_stale_seasons(raw_dir: str, processed_path: str) -> list[str]:
    """
    Return raw CSV file paths whose mtime is newer than the processed output.

    If the processed file does not exist (first run), returns ALL raw files
    so a full build occurs for that table.

    Parameters
    ----------
    raw_dir : str
        Glob pattern for the raw season files, e.g. "data/raw/player_stats/*.csv".
    processed_path : str
        Path to the single processed output file, e.g. "data/processed/player_stats.csv".

    Returns
    -------
    list[str]
        Sorted list of raw file paths that are newer than the processed output.
    """
    raw_files = sorted(glob.glob(raw_dir))
    if not raw_files:
        return []

    if not os.path.exists(processed_path):
        return raw_files

    processed_mtime = os.path.getmtime(processed_path)
    return sorted(f for f in raw_files if os.path.getmtime(f) > processed_mtime)


def load_season_files(files: list[str], prefix: str) -> pd.DataFrame:
    """
    Load a specific list of seasonal CSV files.

    Extracts the season code from each filename (e.g., 'player_stats_202425.csv'
    -> season='202425') and adds it as a column.
    """
    if not files:
        raise FileNotFoundError("No files provided to load_season_files.")
    rows = []
    for file in files:
        season = os.path.basename(file).replace(prefix, "").replace(".csv", "")
        temp_df = pd.read_csv(file)
        temp_df["season"] = season
        rows.append(temp_df)
    return pd.concat(rows, ignore_index=True)


def load_season_folder(path: str, prefix: str) -> pd.DataFrame:
    """
    Load ALL seasonal CSV files from a raw data folder.

    Kept for backward compatibility with any external callers.
    """
    files = sorted(glob.glob(path))
    if not files:
        raise FileNotFoundError(
            f"No files found at {path}. Run the corresponding ingestion script first."
        )
    return load_season_files(files, prefix)


def _season_label(file_path: str, prefix: str) -> str:
    """Extract a human-readable season label like '2024-25' from a raw filename."""
    code = os.path.basename(file_path).replace(prefix, "").replace(".csv", "")
    if len(code) == 6:
        return f"{code[:4]}-{code[4:]}"
    return code


def merge_incremental(
    new_df: pd.DataFrame,
    processed_path: str,
    stale_files: list[str],
    prefix: str,
    dedup_subset: list[str] | None = None,
) -> pd.DataFrame:
    """
    Merge newly processed rows into the existing processed CSV.

    Removes old rows for the updated seasons, appends the new rows,
    and deduplicates.

    Parameters
    ----------
    new_df : pd.DataFrame
        Freshly processed rows for the stale seasons.
    processed_path : str
        Path to the existing processed CSV on disk.
    stale_files : list[str]
        The raw files that were re-processed (used to determine which seasons
        to replace in the existing data).
    prefix : str
        Filename prefix used to extract season codes.
    dedup_subset : list[str] | None
        Columns to use for drop_duplicates. If None, deduplicates on all columns.

    Returns
    -------
    pd.DataFrame
        The merged dataframe ready to be saved.
    """
    stale_seasons = {
        os.path.basename(f).replace(prefix, "").replace(".csv", "")
        for f in stale_files
    }

    if os.path.exists(processed_path):
        existing_df = pd.read_csv(processed_path)
        # Keep only rows from seasons that were NOT rebuilt
        if "season" in existing_df.columns:
            existing_df["season"] = existing_df["season"].astype(str)
            kept_df = existing_df[~existing_df["season"].isin(stale_seasons)]
        else:
            # No season column — full replace
            kept_df = pd.DataFrame(columns=existing_df.columns)
        merged = pd.concat([kept_df, new_df], ignore_index=True)
    else:
        merged = new_df

    if dedup_subset:
        merged = merged.drop_duplicates(subset=dedup_subset)
    else:
        merged = merged.drop_duplicates()
    return merged


# ── Table processing functions ────────────────────────────────────────────────

def _process_seasonal_table(
    table_name: str,
    raw_glob: str,
    prefix: str,
    processed_path: str,
    transform_fn: Callable[[pd.DataFrame], pd.DataFrame],
    full_rebuild: bool,
    dedup_subset: list[str] | None = None,
    optional: bool = False,
) -> None:
    """
    Process a single seasonal table with incremental or full rebuild support.

    Parameters
    ----------
    table_name : str
        Human-readable name for logging.
    raw_glob : str
        Glob pattern for raw season files.
    prefix : str
        Filename prefix to strip for season extraction.
    processed_path : str
        Output path for the processed CSV.
    transform_fn : callable
        Function that takes a raw DataFrame and returns the cleaned DataFrame.
    full_rebuild : bool
        If True, always rebuild from all raw files.
    dedup_subset : list[str] | None
        Columns for deduplication. None means all columns.
    optional : bool
        If True, skip silently when no raw files exist.
    """
    all_raw_files = sorted(glob.glob(raw_glob))
    if not all_raw_files:
        if optional:
            logger.info(
                "%s: skipped (no raw files found)", table_name
            )
            return
        raise FileNotFoundError(
            f"No files found at {raw_glob}. "
            f"Run the corresponding ingestion script first."
        )

    if full_rebuild:
        files_to_process = all_raw_files
    else:
        files_to_process = get_stale_seasons(raw_glob, processed_path)

    if not files_to_process:
        logger.info("Skipping %s (no changes)", table_name)
        return

    season_labels = [_season_label(f, prefix) for f in files_to_process]
    is_incremental = len(files_to_process) < len(all_raw_files)
    mode = "incremental" if is_incremental else "full"
    logger.info(
        "Rebuilding %s (%s): %s (%d season%s)",
        table_name,
        mode,
        ", ".join(season_labels),
        len(files_to_process),
        "s" if len(files_to_process) != 1 else "",
    )

    df = load_season_files(files_to_process, prefix)
    df = transform_fn(df)

    if is_incremental:
        df = merge_incremental(df, processed_path, files_to_process, prefix, dedup_subset)

    df.to_csv(processed_path, index=False)
    logger.info("%s: %s rows", table_name, f"{len(df):,}")


# ── Transform functions (one per table type) ──────────────────────────────────

def _transform_player_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_columns(df)
    _coerce_int_col(df, "player_id")
    _coerce_int_col(df, "team_id")
    df["age"] = df["age"].astype(float)
    df["gp"]  = df["gp"].astype(int)
    df["w"]   = df["w"].astype(int)
    df["l"]   = df["l"].astype(int)
    return df


def _transform_player_stats_advanced(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_columns(df)
    _coerce_int_col(df, "player_id")
    _coerce_int_col(df, "team_id")
    df["age"] = df["age"].astype(float)
    df["gp"]  = df["gp"].astype(int)
    return df


def _transform_player_stats_age(df: pd.DataFrame) -> pd.DataFrame:
    """For clutch, scoring, playoffs with player_id+team_id+age."""
    df = clean_columns(df)
    _coerce_int_col(df, "player_id")
    _coerce_int_col(df, "team_id")
    df["age"] = df["age"].astype(float)
    return df


def _transform_player_game_logs(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_columns(df)
    _coerce_int_col(df, "player_id")
    _coerce_int_col(df, "team_id")
    _coerce_int_col(df, "game_id")
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def _transform_team_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_columns(df)
    _coerce_int_col(df, "team_id")
    df["gp"] = df["gp"].astype(int)
    df["w"]  = df["w"].astype(int)
    df["l"]  = df["l"].astype(int)
    return df


def _transform_team_game_logs(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_columns(df)
    _coerce_int_col(df, "team_id")
    _coerce_int_col(df, "game_id")
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def _transform_standings(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_columns(df)
    df = df.rename(columns={
        "teamid":   "team_id",
        "teamcity": "team_city",
        "teamname": "team_name",
        "teamslug": "team_slug",
        "wins":     "w",
        "losses":   "l",
        "winpct":   "w_pct",
    })
    _coerce_int_col(df, "team_id")
    df["w"] = df["w"].astype(int)
    df["l"] = df["l"].astype(int)
    return df


def _transform_player_hustle(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_columns(df)
    _coerce_int_col(df, "player_id")
    _coerce_int_col(df, "team_id")
    return df


def _transform_team_hustle(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_columns(df)
    _coerce_int_col(df, "team_id")
    return df


def _transform_player_bio(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_columns(df)
    _coerce_int_col(df, "player_id")
    _coerce_int_col(df, "team_id")
    return df


def _transform_shot_chart(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_columns(df)
    df["player_id"] = df["player_id"].astype(int)
    df["team_id"]   = df["team_id"].astype(int)
    df["game_id"]   = df["game_id"].astype(str)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def _transform_lineup_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize lineup efficiency data.

    Lineup CSVs are already written with lowercase column names by get_lineup_data.py,
    so clean_columns() is a no-op but is applied for consistency.
    """
    df = clean_columns(df)
    if "team_id" in df.columns:
        _coerce_int_col(df, "team_id")
    if "gp" in df.columns:
        df["gp"] = df["gp"].astype(int)
    if "season" in df.columns:
        _coerce_int_col(df, "season")
    return df


# ── Table registry ────────────────────────────────────────────────────────────

# Each entry: (table_name, raw_glob, prefix, output_file, transform_fn, dedup_subset, optional)
SEASONAL_TABLES: list[tuple[str, str, str, str, Callable, list[str] | None, bool]] = [
    ("player_stats", "data/raw/player_stats/*.csv", "player_stats_",
     "data/processed/player_stats.csv", _transform_player_stats, None, False),
    ("player_stats_advanced", "data/raw/player_stats_advanced/*.csv", "player_stats_advanced_",
     "data/processed/player_stats_advanced.csv", _transform_player_stats_advanced, None, False),
    ("player_stats_clutch", "data/raw/player_stats_clutch/*.csv", "player_stats_clutch_",
     "data/processed/player_stats_clutch.csv", _transform_player_stats_age, None, False),
    ("player_stats_scoring", "data/raw/player_stats_scoring/*.csv", "player_stats_scoring_",
     "data/processed/player_stats_scoring.csv", _transform_player_stats_age, None, False),
    ("player_game_logs", "data/raw/player_game_logs/*.csv", "player_game_logs_",
     "data/processed/player_game_logs.csv", _transform_player_game_logs, None, False),
    ("team_stats", "data/raw/team_stats/*.csv", "team_stats_",
     "data/processed/team_stats.csv", _transform_team_stats, None, False),
    ("team_stats_advanced", "data/raw/team_stats_advanced/*.csv", "team_stats_advanced_",
     "data/processed/team_stats_advanced.csv", _transform_team_stats, None, False),
    ("team_game_logs", "data/raw/team_game_logs/*.csv", "team_game_logs_",
     "data/processed/team_game_logs.csv", _transform_team_game_logs, None, False),
    ("standings", "data/raw/standings/*.csv", "standings_",
     "data/processed/standings.csv", _transform_standings, None, False),
    ("player_game_logs_playoffs", "data/raw/player_game_logs_playoffs/*.csv", "player_game_logs_playoffs_",
     "data/processed/player_game_logs_playoffs.csv", _transform_player_game_logs, None, False),
    ("team_game_logs_playoffs", "data/raw/team_game_logs_playoffs/*.csv", "team_game_logs_playoffs_",
     "data/processed/team_game_logs_playoffs.csv", _transform_team_game_logs, None, False),
    ("player_stats_playoffs", "data/raw/player_stats_playoffs/*.csv", "player_stats_playoffs_",
     "data/processed/player_stats_playoffs.csv", _transform_player_stats_age, None, False),
    ("player_stats_advanced_playoffs", "data/raw/player_stats_advanced_playoffs/*.csv",
     "player_stats_advanced_playoffs_",
     "data/processed/player_stats_advanced_playoffs.csv", _transform_player_stats_age, None, False),
    ("team_stats_playoffs", "data/raw/team_stats_playoffs/*.csv", "team_stats_playoffs_",
     "data/processed/team_stats_playoffs.csv", _transform_team_stats, None, False),
    ("team_stats_advanced_playoffs", "data/raw/team_stats_advanced_playoffs/*.csv",
     "team_stats_advanced_playoffs_",
     "data/processed/team_stats_advanced_playoffs.csv", _transform_team_stats, None, False),
    ("player_hustle_stats", "data/raw/player_hustle_stats/*.csv", "player_hustle_stats_",
     "data/processed/player_hustle_stats.csv", _transform_player_hustle, None, False),
    ("team_hustle_stats", "data/raw/team_hustle_stats/*.csv", "team_hustle_stats_",
     "data/processed/team_hustle_stats.csv", _transform_team_hustle, None, False),
    ("player_bio_stats", "data/raw/player_bio_stats/*.csv", "player_bio_stats_",
     "data/processed/player_bio_stats.csv", _transform_player_bio, None, True),
    ("shot_chart", "data/raw/shot_chart/*.csv", "shot_chart_",
     "data/processed/shot_chart.csv", _transform_shot_chart, None, True),
    ("lineup_data", "data/raw/lineups/lineup_data_*.csv", "lineup_data_",
     "data/processed/lineup_data.csv", _transform_lineup_data, None, True),
]


# ── Single-file tables (no seasonal split) ────────────────────────────────────

def _process_players(full_rebuild: bool) -> None:
    """Process the player master table (single file, not seasonal)."""
    raw_path = "data/raw/players/player_master.csv"
    processed_path = "data/processed/players.csv"

    if not os.path.exists(raw_path):
        logger.info("players: skipped (no raw file)")
        return

    if not full_rebuild and os.path.exists(processed_path):
        if os.path.getmtime(raw_path) <= os.path.getmtime(processed_path):
            logger.info("Skipping players (no changes)")
            return

    logger.info("Rebuilding players (single-file table)")
    df = pd.read_csv(raw_path)
    df = clean_columns(df)
    df = df.rename(columns={
        "person_id":                  "player_id",
        "display_first_last":         "player_name",
        "display_last_comma_first":   "player_name_last_first",
        "from_year":                  "from_season",
        "to_year":                    "to_season",
    })
    _coerce_int_col(df, "player_id")
    _coerce_int_col(df, "from_season")
    _coerce_int_col(df, "to_season")
    df = df.drop_duplicates(subset=["player_id"])
    df.to_csv(processed_path, index=False)
    logger.info("players: %s rows", f"{len(df):,}")


def _process_teams(full_rebuild: bool) -> None:
    """Process the teams table (single file, not seasonal)."""
    raw_path = "data/raw/teams/teams.csv"
    processed_path = "data/processed/teams.csv"

    if not os.path.exists(raw_path):
        logger.info("teams: skipped (run src/data/get_teams.py to download)")
        return

    if not full_rebuild and os.path.exists(processed_path):
        if os.path.getmtime(raw_path) <= os.path.getmtime(processed_path):
            logger.info("Skipping teams (no changes)")
            return

    logger.info("Rebuilding teams (single-file table)")
    df = pd.read_csv(raw_path)
    df = clean_columns(df)
    df = df.rename(columns={"id": "team_id"})
    _coerce_int_col(df, "team_id")
    df = df.drop_duplicates(subset=["team_id"])
    df.to_csv(processed_path, index=False)
    logger.info("teams: %s rows", f"{len(df):,}")


# ── Main preprocessing function ──────────────────────────────────────────────

def run_preprocessing(full_rebuild: bool = False) -> None:
    """
    Process raw CSVs into clean, consolidated processed CSVs.

    Parameters
    ----------
    full_rebuild : bool
        If True, rebuild every table from scratch regardless of file timestamps.
        If False (default), only re-process tables whose raw source files are
        newer than the processed output (incremental mode).
    """
    os.makedirs("data/processed", exist_ok=True)

    mode_label = "full rebuild" if full_rebuild else "incremental"
    logger.info("Starting preprocessing (%s mode)", mode_label)

    # Single-file tables
    _process_players(full_rebuild)
    _process_teams(full_rebuild)

    # Seasonal tables
    for table_name, raw_glob, prefix, output_path, transform_fn, dedup_subset, optional in SEASONAL_TABLES:
        _process_seasonal_table(
            table_name=table_name,
            raw_glob=raw_glob,
            prefix=prefix,
            processed_path=output_path,
            transform_fn=transform_fn,
            full_rebuild=full_rebuild,
            dedup_subset=dedup_subset,
            optional=optional,
        )

    logger.info("Preprocessing complete.")


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess raw NBA data into consolidated CSVs."
    )
    parser.add_argument(
        "--full-rebuild",
        action="store_true",
        default=False,
        help="Force a complete rebuild of all tables, ignoring file timestamps.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_preprocessing(full_rebuild=args.full_rebuild)
