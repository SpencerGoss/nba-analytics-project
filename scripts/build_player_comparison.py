"""
Build player_comparison.json and player_index.json from historical_player_seasons.csv.

Does NOT call the NBA API — reads from data/raw/ CSVs only.

Era normalization approach:
  - Compute league average PPG, RPG, APG, SPG, BPG per season from team totals
    divided by team count (approximating per-player league average).
  - historical_avg_* = mean of those league averages across all seasons.
  - normalized_stat = player_stat / league_avg_stat * historical_avg_stat
  - This lets you compare Wilt Chamberlain's 50.4 PPG (1961-62, high-scoring era)
    to modern players fairly.
  - pace_factor = season_avg_pace / historical_avg_pace  (team data; NaN for
    pre-pace-era seasons, stored as null in JSON).
  - per36 stats computed from raw totals / minutes * 36.

Output files:
  dashboard/data/player_comparison.json
  dashboard/data/player_index.json

Run: python scripts/build_player_comparison.py
     python scripts/build_player_comparison.py --min-seasons 3 --min-games 200
"""

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DASHBOARD_DATA = PROJECT_ROOT / "dashboard" / "data"
DASHBOARD_DATA.mkdir(parents=True, exist_ok=True)

PLAYERS_CSV = RAW_DIR / "historical_player_seasons.csv"
TEAMS_CSV = RAW_DIR / "historical_team_seasons.csv"
POS_CSV = PROJECT_ROOT / "data" / "processed" / "player_positions.csv"
GAME_LOGS_DIR = RAW_DIR / "player_game_logs"

OUT_COMPARISON = DASHBOARD_DATA / "player_comparison.json"
OUT_INDEX = DASHBOARD_DATA / "player_index.json"

# Eligibility filter defaults (override via CLI)
DEFAULT_MIN_SEASONS = 5
DEFAULT_MIN_CAREER_GAMES = 500

# Target historical averages (all-time mean of per-season league averages)
# These are hard-coded anchors so normalization is stable even when new
# seasons are added. Recomputed at runtime from the actual data but capped
# against these floor values so the scale stays interpretable.
HIST_AVG_PTS = 14.5
HIST_AVG_REB = 7.0
HIST_AVG_AST = 3.5
HIST_AVG_STL = 0.9
HIST_AVG_BLK = 0.4
HIST_AVG_PACE = 95.0   # possessions per 48 min — pre-2014 approximated


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Season string helpers (duplicated here so this module has no dependency on
# fetch_historical_players.py and can be tested / run independently)
# ---------------------------------------------------------------------------

def season_str_to_int(season_str: str) -> int:
    """'1946-47' -> 194647"""
    start, end_yy = season_str.split("-")
    return int(start) * 100 + int(end_yy)


def season_int_to_str(season_int: int) -> str:
    """194647 -> '1946-47'"""
    year = season_int // 100
    end_yy = season_int % 100
    return f"{year}-{end_yy:02d}"


def generate_season_strings(first_year: int = 1946, last_year: int = 2024) -> list[str]:
    """Generate NBA season strings, e.g. ['1946-47', '1947-48', ...]."""
    seasons = []
    for year in range(first_year, last_year + 1):
        end_yy = (year + 1) % 100
        seasons.append(f"{year}-{end_yy:02d}")
    return seasons


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
# Load helpers
# ---------------------------------------------------------------------------

def _load_players(csv_path: Path) -> pd.DataFrame:
    """Load historical_player_seasons.csv with column validation."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Player CSV not found: {csv_path}\n"
            "Run scripts/fetch_historical_players.py first."
        )
    df = pd.read_csv(csv_path, low_memory=False)

    # Normalise column names — raw file may have mixed case depending on nba_api version
    df.columns = [c.lower() for c in df.columns]

    required = {"player_id", "player_name", "season_str", "season", "gp", "min", "pts"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Player CSV missing required columns: {missing}")

    return df


def _load_teams(csv_path: Path) -> pd.DataFrame | None:
    """Load historical_team_seasons.csv. Returns None if file absent."""
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# League averages
# ---------------------------------------------------------------------------

def compute_league_averages(players: pd.DataFrame,
                             teams: pd.DataFrame | None) -> pd.DataFrame:
    """
    Compute per-season league averages for key stats.

    Strategy: average across all player rows for that season (only players
    with min >= 5 to exclude garbage-time outliers).  Team-level pace comes
    from the teams dataframe when available.
    """
    # Filter to players with meaningful minutes
    active = players[players["min"].fillna(0) >= 5].copy()

    agg: dict[str, str] = {}
    for col in ("pts", "reb", "ast", "stl", "blk", "min"):
        if col in active.columns:
            agg[col] = "mean"

    league = (
        active.groupby("season_str")
        .agg(agg)
        .reset_index()
        .rename(columns={
            "pts": "avg_pts",
            "reb": "avg_reb",
            "ast": "avg_ast",
            "stl": "avg_stl",
            "blk": "avg_blk",
            "min": "avg_min",
        })
    )

    # Attach season_int
    league["season_int"] = league["season_str"].apply(
        lambda s: int(s.split("-")[0]) * 100 + int(s.split("-")[1])
    )

    # Pace from team data (optional)
    league["avg_pace"] = float("nan")
    if teams is not None and "pace" in teams.columns:
        pace_by_season = (
            teams.groupby("season_str")["pace"].mean().reset_index()
            .rename(columns={"pace": "avg_pace"})
        )
        league = league.merge(pace_by_season, on="season_str", how="left", suffixes=("", "_team"))
        if "avg_pace_team" in league.columns:
            league["avg_pace"] = league["avg_pace_team"].combine_first(league["avg_pace"])
            league = league.drop(columns=["avg_pace_team"])

    league = league.sort_values("season_int").reset_index(drop=True)
    return league


def _global_avg(series: pd.Series, fallback: float) -> float:
    """Mean of a series, with NaN-safe fallback."""
    val = series.dropna().mean()
    return float(val) if not math.isnan(val) else fallback


# ---------------------------------------------------------------------------
# Player stat enrichment
# ---------------------------------------------------------------------------

def _safe_div(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """Division that returns fallback when denominator is 0 or NaN."""
    if denominator and not math.isnan(denominator) and denominator != 0:
        return numerator / denominator
    return fallback


def enrich_player_seasons(players: pd.DataFrame,
                           league: pd.DataFrame,
                           hist_avgs: dict[str, float]) -> pd.DataFrame:
    """
    Merge league averages into player rows and compute derived stats.

    Adds columns:
      pts_normalized, reb_normalized, ast_normalized, stl_normalized, blk_normalized
      per36_pts, per36_reb, per36_ast
      pace_factor
      ts_pct  (true shooting %)
    """
    # Only select league columns that actually exist (stl/blk may be absent for
    # very old seasons if the raw data had no such column)
    _want_league_cols = ["season_str", "avg_pts", "avg_reb", "avg_ast",
                         "avg_stl", "avg_blk", "avg_pace"]
    _available_league_cols = [c for c in _want_league_cols if c in league.columns]
    df = players.merge(
        league[_available_league_cols],
        on="season_str",
        how="left",
    )

    # Era-normalized stats
    for stat, hist_key in [
        ("pts", "pts"), ("reb", "reb"), ("ast", "ast"),
        ("stl", "stl"), ("blk", "blk"),
    ]:
        avg_col = f"avg_{stat}"
        hist_val = hist_avgs[hist_key]
        if stat in df.columns and avg_col in df.columns:
            df[f"{stat}_normalized"] = df.apply(
                lambda row, s=stat, a=avg_col, h=hist_val: (
                    _safe_div(row[s], row[a], fallback=row[s]) * h
                    if pd.notna(row.get(a)) and row.get(a, 0) != 0
                    else row[s]
                ),
                axis=1,
            )
        else:
            df[f"{stat}_normalized"] = df.get(stat, 0)

    # Per-36 stats (requires total minutes — nba_api PerGame gives avg min/gp)
    # Total minutes = min (per-game avg) * gp; then per36 = total_stat/total_min*36
    if "min" in df.columns and "gp" in df.columns:
        total_min = df["min"] * df["gp"]
        for stat in ("pts", "reb", "ast"):
            if stat in df.columns:
                total_stat = df[stat] * df["gp"]
                df[f"per36_{stat}"] = (total_stat / total_min * 36).where(total_min > 0)
            else:
                df[f"per36_{stat}"] = float("nan")

    # Pace factor
    hist_pace = hist_avgs["pace"]
    df["pace_factor"] = df["avg_pace"].apply(
        lambda p: round(p / hist_pace, 4) if pd.notna(p) and hist_pace > 0 else None
    )

    # True shooting % — requires pts, fta, fga
    has_ts_cols = all(c in df.columns for c in ("pts", "fta", "fga", "gp"))
    if has_ts_cols:
        gp = df["gp"]
        total_pts = df["pts"] * gp
        total_fta = df["fta"] * gp
        total_fga = df["fga"] * gp
        ts_denom = 2 * (total_fga + 0.44 * total_fta)
        df["ts_pct"] = (total_pts / ts_denom).where(ts_denom > 0).round(4)
    else:
        df["ts_pct"] = float("nan")

    return df


# ---------------------------------------------------------------------------
# Build output structures
# ---------------------------------------------------------------------------

def _round_or_none(val, decimals: int = 1):
    """Round float; return None for NaN/None."""
    try:
        if val is None or math.isnan(float(val)):
            return None
        return round(float(val), decimals)
    except (TypeError, ValueError):
        return None


def _season_row(row: pd.Series) -> dict:
    """Convert one enriched player-season row to the JSON dict."""
    def g(col, decimals=1):
        return _round_or_none(row.get(col), decimals)

    return {
        "season": row.get("season_str", ""),
        "season_int": int(row.get("season", 0)),
        "team": str(row.get("team_abbreviation", "")).strip(),
        "age": _round_or_none(row.get("age"), 0),
        "gp": _round_or_none(row.get("gp"), 0),
        "min": g("min"),
        "pts": g("pts"),
        "reb": g("reb"),
        "ast": g("ast"),
        "stl": g("stl"),
        "blk": g("blk"),
        "pts_normalized": g("pts_normalized"),
        "reb_normalized": g("reb_normalized"),
        "ast_normalized": g("ast_normalized"),
        "stl_normalized": g("stl_normalized"),
        "blk_normalized": g("blk_normalized"),
        "per36_pts": g("per36_pts"),
        "per36_reb": g("per36_reb"),
        "per36_ast": g("per36_ast"),
        "pace_factor": _round_or_none(row.get("pace_factor"), 4),
        "ts_pct": _round_or_none(row.get("ts_pct"), 4),
        "fg_pct": _round_or_none(row.get("fg_pct"), 4),
        "fg3_pct": _round_or_none(row.get("fg3_pct"), 4),
        "ft_pct": _round_or_none(row.get("ft_pct"), 4),
    }


def _load_position_lookup() -> dict[int, dict]:
    """Load player positions from player_positions.csv if available."""
    if not POS_CSV.exists():
        return {}
    try:
        df = pd.read_csv(POS_CSV)
        result = {}
        for _, row in df.iterrows():
            pid = int(row["player_id"])
            result[pid] = {
                "position": str(row.get("position") or ""),
                "positions": str(row.get("positions") or ""),
                "position_primary": str(row.get("position_primary") or ""),
                "jersey_number": str(int(row["jersey_number"])) if pd.notna(row.get("jersey_number")) else "",
            }
        return result
    except Exception:
        return {}


def _heuristic_positions(career_avgs: dict) -> str:
    """Guess single granular position from career averages (historical players)."""
    pts = career_avgs.get("pts") or 0
    reb = career_avgs.get("reb") or 0
    ast = career_avgs.get("ast") or 0
    blk = career_avgs.get("blk") or 0
    if reb >= 9 and blk >= 1.5:
        return "C"
    if reb >= 9:
        return "PF"
    if ast >= 7:
        return "PG"
    if reb >= 6:
        return "SF"
    if ast >= 4:
        return "SG"
    if pts >= 15 and reb >= 5:
        return "SF"
    return "SG"


def build_player_records(enriched: pd.DataFrame,
                          min_seasons: int,
                          min_career_games: int,
                          pos_lookup: dict[int, dict] | None = None) -> list[dict]:
    """
    Group by player, build career summary, filter to eligibility threshold.
    Returns list of player dicts sorted by career pts_normalized desc.
    """
    pos_lookup = pos_lookup or {}
    records = []

    for (player_id, player_name), grp in enriched.groupby(
        ["player_id", "player_name"], sort=False
    ):
        grp = grp.sort_values("season")
        total_gp = grp["gp"].fillna(0).sum()
        n_seasons = len(grp)

        if n_seasons < min_seasons and total_gp < min_career_games:
            continue

        season_rows = [_season_row(row) for _, row in grp.iterrows()]

        # Career averages — weighted by games played
        gp_arr = grp["gp"].fillna(0)
        total_gp_safe = gp_arr.sum() if gp_arr.sum() > 0 else 1

        def wavg(col):
            if col not in grp.columns:
                return None
            vals = pd.to_numeric(grp[col], errors="coerce")
            return _round_or_none((vals * gp_arr).sum() / total_gp_safe)

        career_avgs = {
            "pts": wavg("pts"),
            "reb": wavg("reb"),
            "ast": wavg("ast"),
            "stl": wavg("stl"),
            "blk": wavg("blk"),
            "pts_normalized": wavg("pts_normalized"),
        }

        # Best season by normalized PPG
        norm_pts = pd.to_numeric(grp.get("pts_normalized", grp.get("pts", pd.Series())),
                                  errors="coerce")
        best_idx = norm_pts.idxmax() if not norm_pts.isna().all() else grp.index[0]
        best_season_str = grp.loc[best_idx, "season_str"] if best_idx in grp.index else ""

        # Season span string for index
        min_year = int(str(grp["season"].min())[:4])
        max_year = int(str(grp["season"].max())[:4]) + 1
        seasons_span = f"{min_year}-{max_year}"

        # Position: prefer roster data, fall back to heuristic
        pos_info = pos_lookup.get(int(player_id), {})
        position = pos_info.get("position", "")
        positions = pos_info.get("positions", "")
        position_primary = pos_info.get("position_primary", "")
        jersey_number = pos_info.get("jersey_number", "")
        if not positions:
            positions = _heuristic_positions(career_avgs)
            position_primary = positions

        records.append({
            "player_id": int(player_id),
            "player_name": str(player_name),
            "position": position,
            "positions": positions,
            "position_primary": position_primary,
            "jersey_number": jersey_number,
            "seasons": season_rows,
            "career_avgs": career_avgs,
            "best_season": str(best_season_str),
            "career_gp": int(total_gp),
            "seasons_span": seasons_span,
        })

    # Sort by career normalized PPG descending (nulls last)
    records.sort(
        key=lambda r: r["career_avgs"].get("pts_normalized") or 0,
        reverse=True,
    )
    return records


def build_league_by_season(league: pd.DataFrame) -> list[dict]:
    """Convert league averages dataframe to JSON-ready list."""
    rows = []
    for _, row in league.iterrows():
        rows.append({
            "season": str(row.get("season_str", "")),
            "season_int": int(row.get("season_int", 0)),
            "avg_pts": _round_or_none(row.get("avg_pts"), 2),
            "avg_reb": _round_or_none(row.get("avg_reb"), 2),
            "avg_ast": _round_or_none(row.get("avg_ast"), 2),
            "avg_stl": _round_or_none(row.get("avg_stl"), 2),
            "avg_blk": _round_or_none(row.get("avg_blk"), 2),
            "avg_pace": _round_or_none(row.get("avg_pace"), 2),
        })
    return rows


def build_player_index(player_records: list[dict]) -> list[dict]:
    """Lightweight index for autocomplete search."""
    return [
        {
            "id": r["player_id"],
            "name": r["player_name"],
            "seasons": r["seasons_span"],
        }
        for r in player_records
    ]


# ---------------------------------------------------------------------------
# Legend overrides — pre-1996 stars not fully covered by NBA API
# Stats: Basketball Reference career regular-season averages (per game).
# These records REPLACE any partial API-sourced record for the same player.
# ---------------------------------------------------------------------------

_LEGENDS: list[dict] = [
    {
        "player_id": -1, "player_name": "Michael Jordan",
        "positions": "SG", "position_primary": "SG", "jersey_number": "23",
        "seasons_span": "1984-2003", "career_gp": 1072,
        "career_avgs": {"pts": 30.1, "reb": 6.2, "ast": 5.3, "stl": 2.35, "blk": 0.83,
                        "pts_normalized": None},
        "best_season": "1986-87",
        "ts_pct": 0.570, "fg_pct": 0.497, "fg3_pct": 0.327, "ft_pct": 0.835,
        "seasons": [
            {"season_str": "1984-85", "pts": 28.2, "reb": 6.5, "ast": 5.9},
            {"season_str": "1986-87", "pts": 37.1, "reb": 5.2, "ast": 4.6},
            {"season_str": "1987-88", "pts": 35.0, "reb": 5.5, "ast": 5.9},
            {"season_str": "1990-91", "pts": 31.5, "reb": 6.0, "ast": 5.5},
            {"season_str": "1995-96", "pts": 30.4, "reb": 6.6, "ast": 4.3},
            {"season_str": "1996-97", "pts": 29.6, "reb": 5.9, "ast": 4.3},
            {"season_str": "1997-98", "pts": 28.7, "reb": 5.8, "ast": 3.5},
            {"season_str": "2001-02", "pts": 22.9, "reb": 5.7, "ast": 5.2},
            {"season_str": "2002-03", "pts": 20.0, "reb": 6.1, "ast": 3.8},
        ],
        "_legend": True, "_note": "Career stats from Basketball Reference. NBA API only covers 1996-2003.",
    },
    {
        "player_id": -2, "player_name": "Larry Bird",
        "positions": "SF, PF", "position_primary": "SF", "jersey_number": "33",
        "seasons_span": "1979-1992", "career_gp": 897,
        "career_avgs": {"pts": 24.3, "reb": 10.0, "ast": 6.3, "stl": 1.74, "blk": 0.84,
                        "pts_normalized": None},
        "best_season": "1987-88",
        "ts_pct": 0.584, "fg_pct": 0.496, "fg3_pct": 0.376, "ft_pct": 0.886,
        "seasons": [],
        "_legend": True, "_note": "Career stats from Basketball Reference.",
    },
    {
        "player_id": -3, "player_name": "Magic Johnson",
        "positions": "PG", "position_primary": "PG", "jersey_number": "32",
        "seasons_span": "1979-1996", "career_gp": 906,
        "career_avgs": {"pts": 19.5, "reb": 7.2, "ast": 11.2, "stl": 1.90, "blk": 0.37,
                        "pts_normalized": None},
        "best_season": "1988-89",
        "ts_pct": 0.584, "fg_pct": 0.520, "fg3_pct": 0.303, "ft_pct": 0.848,
        "seasons": [],
        "_legend": True, "_note": "Career stats from Basketball Reference.",
    },
    {
        "player_id": -4, "player_name": "Kareem Abdul-Jabbar",
        "positions": "C", "position_primary": "C", "jersey_number": "33",
        "seasons_span": "1969-1989", "career_gp": 1560,
        "career_avgs": {"pts": 24.6, "reb": 11.2, "ast": 3.6, "stl": 0.94, "blk": 2.60,
                        "pts_normalized": None},
        "best_season": "1971-72",
        "ts_pct": 0.579, "fg_pct": 0.559, "fg3_pct": 0.056, "ft_pct": 0.721,
        "seasons": [],
        "_legend": True, "_note": "Career stats from Basketball Reference.",
    },
    {
        "player_id": -5, "player_name": "Kobe Bryant",
        "positions": "SG", "position_primary": "SG", "jersey_number": "24",
        "seasons_span": "1996-2016", "career_gp": 1346,
        "career_avgs": {"pts": 25.0, "reb": 5.2, "ast": 4.7, "stl": 1.38, "blk": 0.52,
                        "pts_normalized": None},
        "best_season": "2005-06",
        "ts_pct": 0.551, "fg_pct": 0.447, "fg3_pct": 0.329, "ft_pct": 0.837,
        "seasons": [],
        "_legend": True, "_note": "Career stats from Basketball Reference.",
    },
    {
        "player_id": -6, "player_name": "Wilt Chamberlain",
        "positions": "C", "position_primary": "C", "jersey_number": "13",
        "seasons_span": "1959-1973", "career_gp": 1045,
        "career_avgs": {"pts": 30.1, "reb": 22.9, "ast": 4.4, "stl": None, "blk": None,
                        "pts_normalized": None},
        "best_season": "1961-62",
        "ts_pct": 0.541, "fg_pct": 0.540, "fg3_pct": None, "ft_pct": 0.511,
        "seasons": [],
        "_legend": True, "_note": "Career stats from Basketball Reference.",
    },
]


def _inject_legends(player_records: list[dict]) -> list[dict]:
    """Replace partial API records with curated legend stats; append if not present."""
    existing_names = {r["player_name"] for r in player_records}
    result = []
    replaced = set()
    for rec in player_records:
        # Drop partial API record if a legend override exists
        leg = next((l for l in _LEGENDS if l["player_name"] == rec["player_name"]), None)
        if leg:
            replaced.add(leg["player_name"])
        else:
            result.append(rec)
    # Add all legends (replaced or new)
    for leg in _LEGENDS:
        result.append(dict(leg))
    return result


# ---------------------------------------------------------------------------
# Single-game records (computed from game logs + pre-1996 historical floor)
# ---------------------------------------------------------------------------

# Pre-1996 records that can't be derived from game logs (NBA API starts 1996-97)
_PRE_1996_RECORDS: dict[str, list[dict]] = {
    "pts": [
        {"name": "Wilt Chamberlain", "team": "PHI", "val": 100, "detail": "vs NYK, Mar 2 1962"},
        {"name": "Wilt Chamberlain", "team": "PHI", "val": 78, "detail": "vs LAL, Dec 8 1961"},
        {"name": "David Thompson", "team": "DEN", "val": 73, "detail": "vs DET, Apr 9 1978"},
        {"name": "Wilt Chamberlain", "team": "SFW", "val": 73, "detail": "vs NYK, Nov 16 1962"},
        {"name": "Elgin Baylor", "team": "LAL", "val": 71, "detail": "vs NYK, Nov 15 1960"},
        {"name": "Wilt Chamberlain", "team": "PHI", "val": 68, "detail": "vs CHI, Dec 16 1967"},
        {"name": "Pete Maravich", "team": "NOJ", "val": 68, "detail": "vs NYK, Feb 25 1977"},
    ],
    "reb": [
        {"name": "Wilt Chamberlain", "team": "PHI", "val": 55, "detail": "vs BOS, Nov 24 1960"},
        {"name": "Bill Russell", "team": "BOS", "val": 51, "detail": "vs SYR, Feb 5 1960"},
        {"name": "Bill Russell", "team": "BOS", "val": 49, "detail": "vs PHI, Nov 16 1957"},
        {"name": "Bill Russell", "team": "BOS", "val": 49, "detail": "vs DET, Mar 11 1965"},
        {"name": "Wilt Chamberlain", "team": "PHI", "val": 45, "detail": "vs SYR, Feb 6 1960"},
        {"name": "Wilt Chamberlain", "team": "PHI", "val": 45, "detail": "vs LAL, Jan 21 1961"},
        {"name": "Wilt Chamberlain", "team": "SFW", "val": 43, "detail": "vs BOS, Jan 15 1963"},
        {"name": "Bill Russell", "team": "BOS", "val": 43, "detail": "vs LAL, Jan 20 1963"},
        {"name": "Wilt Chamberlain", "team": "PHI", "val": 42, "detail": "vs BOS, Mar 6 1965"},
        {"name": "Nate Thurmond", "team": "SFW", "val": 42, "detail": "vs DET, Nov 9 1965"},
    ],
    "ast": [
        {"name": "Bob Cousy", "team": "BOS", "val": 28, "detail": "vs MIN, Feb 27 1959"},
        {"name": "Guy Rodgers", "team": "SFW", "val": 28, "detail": "vs STL, Mar 14 1963"},
        {"name": "Nate McMillan", "team": "SEA", "val": 25, "detail": "vs LAC, Feb 23 1987"},
    ],
}

# Map NBA API team abbreviations to standard 3-letter codes
_TEAM_ABB_MAP = {
    "PHX": "PHX", "PHO": "PHX",
    "BKN": "BKN", "NJN": "NJN",
    "CHA": "CHA", "CHH": "CHA",
    "NOH": "NOP", "NOK": "NOP", "NOP": "NOP",
}


def _build_single_game_records(top_n: int = 15) -> dict[str, list[dict]]:
    """Compute top single-game performances from player game logs.

    Merges with pre-1996 historical records and returns top_n per stat.
    """
    stat_cols = {"pts": "PTS", "reb": "REB", "ast": "AST"}
    records: dict[str, list[dict]] = {s: list(recs) for s, recs in _PRE_1996_RECORDS.items()}

    if not GAME_LOGS_DIR.exists():
        print("  WARN: game logs directory not found, using pre-1996 records only")
        for stat in records:
            records[stat].sort(key=lambda r: r["val"], reverse=True)
            records[stat] = records[stat][:top_n]
        return records

    # Load all game log CSVs
    csv_files = sorted(GAME_LOGS_DIR.glob("*.csv"))
    if not csv_files:
        print("  WARN: no game log CSVs found")
        for stat in records:
            records[stat].sort(key=lambda r: r["val"], reverse=True)
            records[stat] = records[stat][:top_n]
        return records

    for stat_key, col_name in stat_cols.items():
        best_from_logs: list[dict] = []
        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path, usecols=["PLAYER_NAME", "MATCHUP", "GAME_DATE", col_name],
                                 low_memory=False)
            except (ValueError, KeyError):
                continue
            df = df.dropna(subset=[col_name])
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
            df = df.dropna(subset=[col_name])
            # Keep top performances from this file
            top = df.nlargest(top_n, col_name)
            for _, row in top.iterrows():
                matchup = str(row.get("MATCHUP", ""))
                game_date = str(row.get("GAME_DATE", ""))
                # Parse matchup for team abbreviation
                team = matchup.split(" ")[0] if matchup else ""
                # Format detail string
                opponent = ""
                if " vs. " in matchup:
                    opponent = "vs " + matchup.split(" vs. ")[-1]
                elif " @ " in matchup:
                    opponent = "@ " + matchup.split(" @ ")[-1]
                # Format date
                try:
                    dt = pd.to_datetime(game_date, format="mixed")
                    detail = f"{opponent}, {dt.strftime('%b %-d %Y')}" if opponent else game_date
                except Exception:
                    detail = f"{opponent}, {game_date}" if opponent else game_date
                # Windows strftime doesn't support %-d, use %#d instead
                try:
                    dt = pd.to_datetime(game_date, format="mixed")
                    detail = f"{opponent}, {dt.strftime('%b %d %Y').replace(' 0', ' ')}" if opponent else game_date
                except Exception:
                    pass

                best_from_logs.append({
                    "name": str(row["PLAYER_NAME"]),
                    "team": _TEAM_ABB_MAP.get(team, team),
                    "val": int(row[col_name]),
                    "detail": detail,
                })

        # Merge with pre-1996 records, sort, dedupe, take top_n
        combined = records.get(stat_key, []) + best_from_logs
        combined.sort(key=lambda r: r["val"], reverse=True)
        # Deduplicate by (name, val, detail prefix) — keep first occurrence
        seen = set()
        deduped = []
        for r in combined:
            key = (r["name"], r["val"])
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        records[stat_key] = deduped[:top_n]

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(min_seasons: int = DEFAULT_MIN_SEASONS,
        min_career_games: int = DEFAULT_MIN_CAREER_GAMES) -> dict:
    """Execute the full build. Returns summary dict for logging."""
    print("build_player_comparison: loading player data...")
    try:
        players = _load_players(PLAYERS_CSV)
    except FileNotFoundError as exc:
        print(f"WARN: {exc} -- skipping player_comparison.json write")
        return {"players": 0, "seasons": 0}
    print(f"  {len(players):,} player-season rows across "
          f"{players['season_str'].nunique()} seasons")

    print("build_player_comparison: loading team data (for pace)...")
    teams = _load_teams(TEAMS_CSV)

    print("build_player_comparison: computing league averages...")
    league = compute_league_averages(players, teams)

    # Historical averages across all seasons
    hist_avgs = {
        "pts": _global_avg(league.get("avg_pts", pd.Series()), HIST_AVG_PTS),
        "reb": _global_avg(league.get("avg_reb", pd.Series()), HIST_AVG_REB),
        "ast": _global_avg(league.get("avg_ast", pd.Series()), HIST_AVG_AST),
        "stl": _global_avg(league.get("avg_stl", pd.Series()), HIST_AVG_STL),
        "blk": _global_avg(league.get("avg_blk", pd.Series()), HIST_AVG_BLK),
        "pace": _global_avg(league.get("avg_pace", pd.Series()), HIST_AVG_PACE),
    }
    print(f"  Historical averages: pts={hist_avgs['pts']:.2f} reb={hist_avgs['reb']:.2f} "
          f"ast={hist_avgs['ast']:.2f} pace={hist_avgs['pace']:.1f}")

    print("build_player_comparison: enriching player seasons...")
    enriched = enrich_player_seasons(players, league, hist_avgs)

    print("build_player_comparison: loading position data...")
    pos_lookup = _load_position_lookup()
    print(f"  {len(pos_lookup)} players with roster positions")

    print(f"build_player_comparison: building player records "
          f"(min_seasons={min_seasons}, min_career_games={min_career_games})...")
    player_records = build_player_records(enriched, min_seasons, min_career_games, pos_lookup)
    league_rows = build_league_by_season(league)

    print(f"  {len(player_records):,} eligible players")

    # Inject curated legend overrides for pre-1996 players missing from NBA API.
    # Stats sourced from Basketball Reference career regular-season averages.
    # Must happen BEFORE building the player index so legends appear in search.
    player_records = _inject_legends(player_records)
    print(f"  {len(player_records):,} players after legend injection")

    player_index = build_player_index(player_records)

    # Build single-game records from game logs + historical floor
    print("build_player_comparison: computing single-game records...")
    single_game_records = _build_single_game_records(top_n=15)
    for stat_key, recs in single_game_records.items():
        print(f"  {stat_key}: top={recs[0]['name']} ({recs[0]['val']})" if recs else f"  {stat_key}: empty")

    # Write comparison JSON
    comparison = {
        "players": player_records,
        "league_by_season": league_rows,
        "single_game_records": single_game_records,
        "meta": {
            "seasons_covered": league["season_str"].tolist(),
            "total_seasons": len(league),
            "total_players": len(player_records),
            "historical_avgs": hist_avgs,
            "min_seasons_filter": min_seasons,
            "min_career_games_filter": min_career_games,
        },
    }

    with open(OUT_COMPARISON, "w", encoding="utf-8") as f:
        json.dump(comparison, f, separators=(",", ":"))

    print(f"build_player_comparison: written -> {OUT_COMPARISON} "
          f"({OUT_COMPARISON.stat().st_size / 1024:.0f} KB)")

    # Write player index
    with open(OUT_INDEX, "w", encoding="utf-8") as f:
        json.dump(player_index, f, separators=(",", ":"))

    print(f"build_player_comparison: written -> {OUT_INDEX} "
          f"({OUT_INDEX.stat().st_size / 1024:.0f} KB)")

    return {
        "players": len(player_records),
        "seasons": len(league),
        "league_rows": len(league_rows),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Build player_comparison.json from historical_player_seasons.csv."
    )
    p.add_argument("--min-seasons", type=int, default=DEFAULT_MIN_SEASONS,
                   help=f"Minimum seasons to include player (default: {DEFAULT_MIN_SEASONS})")
    p.add_argument("--min-games", type=int, default=DEFAULT_MIN_CAREER_GAMES,
                   help=f"Minimum career games to include player (default: {DEFAULT_MIN_CAREER_GAMES})")
    args = p.parse_args(argv)
    summary = run(min_seasons=args.min_seasons, min_career_games=args.min_games)
    print(f"build_player_comparison: done. "
          f"{summary['players']} players, {summary['seasons']} seasons.")


if __name__ == "__main__":
    main()
