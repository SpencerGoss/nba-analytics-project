"""
Player Positions Ingestion
===========================
Fetches the official NBA position for every player using the nba_api
PlayerIndex endpoint — one call per season, then deduplicated.

This matches the season-loop pattern used by all other get_*.py scripts in
this project (get_player_bio_stats.py, get_player_stats.py, etc.) and avoids
relying on version-specific flags like 'historical' that vary across nba_api
releases.

Saves to:
  data/processed/player_positions.csv

Output columns:
  player_id, player_name, position, position_bucket

Position bucket mapping (used by player_features.py):
  PG / SG / G / G-F → "G"  (Guard)
  SF / PF / F / F-C → "F"  (Forward)
  C  / C-F           → "C"  (Center)

For dual-position players (e.g. "G-F") the PRIMARY position (first listed)
determines the bucket so every player gets exactly one bucket.

Usage:
    python src/data/get_player_positions.py

Run once; re-run to pick up new rookies at the start of each season.
"""

from nba_api.stats.endpoints import playerindex
import pandas as pd
import time
import os

from src.data.api_client import fetch_with_retry, HEADERS
import logging

log = logging.getLogger(__name__)

REQUEST_DELAY = 1          # seconds between season calls (be polite to the API)
OUTPUT_PATH   = "data/processed/player_positions.csv"


# ── Position bucket mapping ────────────────────────────────────────────────────

_POSITION_MAP = {
    # Pure positions
    "PG": "G", "SG": "G", "G": "G",
    "SF": "F", "PF": "F", "F": "F",
    "C":  "C",
    # Dual positions — bucket = primary (first listed)
    "G-F": "G", "G-C": "G",
    "F-G": "F", "F-C": "F",
    "C-F": "C", "C-G": "C",
    # Alternate / historic spellings
    "GUARD":          "G", "FORWARD":        "F", "CENTER":          "C",
    "GUARD-FORWARD":  "G", "FORWARD-GUARD":  "F",
    "FORWARD-CENTER": "F", "CENTER-FORWARD": "C",
}


def _to_bucket(pos_str: str) -> str:
    """
    Map a raw NBA position string to G / F / C.

      1. Exact lookup in _POSITION_MAP
      2. First token before any hyphen
      3. First character (G/F/C)
      4. Default 'F' (most common NBA position)
    """
    if pd.isna(pos_str) or str(pos_str).strip() == "":
        return "F"
    cleaned = str(pos_str).strip().upper()
    if cleaned in _POSITION_MAP:
        return _POSITION_MAP[cleaned]
    primary = cleaned.split("-")[0].strip()
    if primary in _POSITION_MAP:
        return _POSITION_MAP[primary]
    return {"G": "G", "F": "F", "C": "C"}.get(primary[:1], "F")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_response(raw: pd.DataFrame) -> pd.DataFrame | None:
    """
    Normalise a single PlayerIndex response DataFrame into a consistent
    (player_id, player_name, position) table regardless of column-name
    variations across nba_api versions.
    """
    # Standardise column names
    raw = raw.copy()
    raw.columns = (
        raw.columns
        .str.lower()
        .str.strip()
        .str.replace(" ", "_", regex=False)
    )

    # Locate player ID column
    id_col = next(
        (c for c in ["person_id", "player_id", "personid"] if c in raw.columns),
        None,
    )
    if id_col is None:
        log.error(f"    Could not find player ID column. Available: {list(raw.columns)}")
        return None
    raw = raw.rename(columns={id_col: "player_id"})

    # Build player_name
    if "player_name" not in raw.columns:
        if "player_first_name" in raw.columns and "player_last_name" in raw.columns:
            raw["player_name"] = (
                raw["player_first_name"].fillna("").str.strip()
                + " "
                + raw["player_last_name"].fillna("").str.strip()
            ).str.strip()
        elif "display_first_last" in raw.columns:
            raw["player_name"] = raw["display_first_last"]
        else:
            raw["player_name"] = raw["player_id"].astype(str)

    # Locate position column
    pos_col = next(
        (c for c in ["position", "pos"] if c in raw.columns),
        None,
    )
    if pos_col is None:
        # Position column absent — keep rows but flag them empty
        raw["position"] = ""
    else:
        raw = raw.rename(columns={pos_col: "position"})

    out = raw[["player_id", "player_name", "position"]].copy()
    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce")
    out = out.dropna(subset=["player_id"])
    out["player_id"] = out["player_id"].astype(int)
    out["position"]  = out["position"].fillna("").str.strip()
    return out


# ── Main fetcher ───────────────────────────────────────────────────────────────

def get_player_positions(
    start_year: int = 1996,
    end_year:   int = 2024,
) -> pd.DataFrame:
    """
    Fetch position data for every NBA player by looping through seasons.

    For each season in [start_year, end_year], calls PlayerIndex and collects
    player + position rows.  After all seasons are processed, deduplicates so
    each player appears once — using the MOST RECENT season's position
    (handles rare cases where a player switched from G to G-F mid-career).

    Args:
        start_year: First season start year (default 1996 → 1996-97).
        end_year:   Last season start year  (default 2024 → 2024-25).

    Returns:
        DataFrame saved to data/processed/player_positions.csv.
    """
    os.makedirs("data/processed", exist_ok=True)
    all_frames = []

    log.info(f"Fetching player positions via PlayerIndex "
          f"({start_year}-{str(start_year+1)[-2:]} -> "
          f"{end_year}-{str(end_year+1)[-2:]})...")
    log.info(f"  {end_year - start_year + 1} seasons to fetch.\n")

    for year in range(start_year, end_year + 1):
        season = f"{year}-{str(year + 1)[-2:]}"
        time.sleep(REQUEST_DELAY)

        result = fetch_with_retry(
            lambda s=season: playerindex.PlayerIndex(
                league_id="00",
                season=s,
                headers=HEADERS,
                timeout=60,
            ).get_data_frames()[0],
            season,
        )

        if not result["success"]:
            log.error(f"  Skipping {season} (all retries failed).")
            continue

        parsed = _parse_response(result["data"])
        if parsed is None or len(parsed) == 0:
            log.warning(f"  {season}: no usable rows, skipping.")
            continue

        parsed["season_year"] = year   # used for deduplication ordering
        all_frames.append(parsed)

        filled = (parsed["position"] != "").sum()
        log.info(f"  {season}: {len(parsed):,} players | "
              f"{filled:,} with position ({filled/len(parsed)*100:.0f}%)")

    if not all_frames:
        log.warning("\nNo data collected. Check your network connection and nba_api version.")
        return None

    # ── Deduplicate ────────────────────────────────────────────────────────────
    combined = pd.concat(all_frames, ignore_index=True)

    # Prefer rows that have a non-empty position
    combined["has_position"] = (combined["position"] != "").astype(int)
    combined = combined.sort_values(
        ["player_id", "has_position", "season_year"],
        ascending=[True, False, False],   # keep highest has_position, then newest season
    )
    final = combined.drop_duplicates(subset=["player_id"], keep="first")
    final = final.drop(columns=["has_position", "season_year"])

    # Add position bucket
    final["position_bucket"] = final["position"].apply(_to_bucket)
    final = final.sort_values("player_id").reset_index(drop=True)

    # ── Summary ────────────────────────────────────────────────────────────────
    pct_filled    = (final["position"] != "").mean() * 100
    bucket_counts = final["position_bucket"].value_counts()
    top_positions = final["position"].value_counts().head(10)

    log.info(f"\n{'='*50}")
    log.info(f"  Total unique players : {len(final):,}")
    log.info(f"  Position fill rate   : {pct_filled:.1f}%")
    log.info(f"  Bucket distribution  :")
    for bucket, count in bucket_counts.items():
        log.info(f"    {bucket} : {count:,}")
    log.info(f"  Top 10 raw positions :")
    for pos, count in top_positions.items():
        label = pos if pos else "(empty)"
        log.info(f"    {label:<15} {count:,}")
    log.info(f"{'='*50}")

    final.to_csv(OUTPUT_PATH, index=False)
    log.info(f"\nSaved {len(final):,} players -> {OUTPUT_PATH}")

    return final


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = get_player_positions()
    if df is not None:
        log.info(f"\nSample output:")
        log.debug(df[["player_id", "player_name", "position", "position_bucket"]].head(10).to_string(index=False))
