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
import logging

log = logging.getLogger(__name__)

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


TEAM_CSV_COLUMNS = [
    "team_id", "team_name", "gp", "w", "l", "w_pct", "min", "fgm", "fga",
    "fg_pct", "fg3m", "fg3a", "fg3_pct", "ftm", "fta", "ft_pct", "oreb",
    "dreb", "reb", "ast", "tov", "stl", "blk", "blka", "pf", "pfd", "pts",
    "plus_minus",
    # Rank columns (28..53)
    "gp_rank", "w_rank", "l_rank", "w_pct_rank", "min_rank", "fgm_rank",
    "fga_rank", "fg_pct_rank", "fg3m_rank", "fg3a_rank", "fg3_pct_rank",
    "ftm_rank", "fta_rank", "ft_pct_rank", "oreb_rank", "dreb_rank",
    "reb_rank", "ast_rank", "tov_rank", "stl_rank", "blk_rank", "blka_rank",
    "pf_rank", "pfd_rank", "pts_rank", "plus_minus_rank",
    # Trailing identifiers
    "season_str", "season",
]


def _load_teams(csv_path: Path) -> pd.DataFrame | None:
    """Load historical_team_seasons.csv (no header row). Returns None if file absent."""
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, header=None, low_memory=False)
    # Assign column names — file has no header row
    if len(df.columns) == len(TEAM_CSV_COLUMNS):
        df.columns = TEAM_CSV_COLUMNS
    else:
        # Fallback: assign as many names as possible
        df.columns = [
            TEAM_CSV_COLUMNS[i] if i < len(TEAM_CSV_COLUMNS) else f"col_{i}"
            for i in range(len(df.columns))
        ]
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


def _build_team_season_lookup(teams: pd.DataFrame | None) -> dict[tuple[str, str], dict]:
    """Build a lookup of (team_abbreviation, season_str) -> team per-game stats.

    Returns empty dict if teams is None.  All values in the dict are per-game
    averages (as stored in the CSV).
    """
    if teams is None or teams.empty:
        return {}
    lookup: dict[tuple[str, str], dict] = {}
    numeric_cols = ["gp", "min", "fgm", "fga", "fg3m",
                    "ftm", "fta", "oreb", "dreb", "reb", "ast", "tov", "stl", "blk", "pts"]
    available = [c for c in numeric_cols if c in teams.columns]
    for _, row in teams.iterrows():
        team_name = str(row.get("team_name", ""))
        season_str = str(row.get("season_str", ""))
        key = (team_name, season_str)
        lookup[key] = {c: float(row[c]) if pd.notna(row.get(c)) else 0.0 for c in available}
    return lookup


def _league_avg_reb_by_season(teams: pd.DataFrame | None) -> dict[str, dict]:
    """Compute league-average per-game OREB and DREB per season from team data.

    Returns {season_str: {"oreb": float, "dreb": float}}.
    """
    if teams is None or teams.empty:
        return {}
    result: dict[str, dict] = {}
    for season_str, grp in teams.groupby("season_str"):
        result[str(season_str)] = {
            "oreb": float(grp["oreb"].mean()) if "oreb" in grp.columns else 0.0,
            "dreb": float(grp["dreb"].mean()) if "dreb" in grp.columns else 0.0,
        }
    return result


# Maps team_abbreviation -> team_name for matching player rows to team rows.
# Built lazily at runtime from team data.
_TEAM_NAME_BY_ABBR: dict[tuple[str, str], str] = {}


def _resolve_team_name(team_abbr: str, season_str: str,
                       team_lookup: dict[tuple[str, str], dict]) -> str | None:
    """Find the team_name that matches a player's team_abbreviation + season."""
    # Try cache first
    cache_key = (team_abbr, season_str)
    if cache_key in _TEAM_NAME_BY_ABBR:
        return _TEAM_NAME_BY_ABBR[cache_key]
    # Search through lookup keys
    for (tname, sstr), _ in team_lookup.items():
        if sstr == season_str:
            # Match by common NBA abbreviation patterns
            # The team CSV has full names; we need to map abbreviation -> name
            # Store all matches for this season
            pass
    return None


def _build_abbr_to_name_map(teams: pd.DataFrame | None) -> dict[tuple[str, str], str]:
    """Build (abbreviation, season_str) -> team_name from team data.

    Uses a known mapping of NBA team abbreviations to full names.
    """
    if teams is None or teams.empty:
        return {}

    # Build map from team_name -> abbreviation by matching known patterns
    _ABBR_MAP = {
        "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
        "Charlotte Hornets": "CHA", "Charlotte Bobcats": "CHA",
        "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
        "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
        "Detroit Pistons": "DET", "Golden State Warriors": "GSW",
        "Houston Rockets": "HOU", "Indiana Pacers": "IND",
        "LA Clippers": "LAC", "Los Angeles Clippers": "LAC",
        "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
        "Vancouver Grizzlies": "VAN", "Miami Heat": "MIA",
        "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
        "New Jersey Nets": "NJN", "New Orleans Pelicans": "NOP",
        "New Orleans Hornets": "NOH", "New Orleans/Oklahoma City Hornets": "NOK",
        "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
        "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI",
        "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR",
        "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
        "Seattle SuperSonics": "SEA", "Toronto Raptors": "TOR",
        "Utah Jazz": "UTA", "Washington Wizards": "WAS",
        "Washington Bullets": "WAS",
    }

    result: dict[tuple[str, str], str] = {}
    for _, row in teams.iterrows():
        tname = str(row.get("team_name", ""))
        sstr = str(row.get("season_str", ""))
        abbr = _ABBR_MAP.get(tname, "")
        if abbr:
            result[(abbr, sstr)] = tname
    return result


def compute_advanced_stats(players: pd.DataFrame,
                           teams: pd.DataFrame | None) -> pd.DataFrame:
    """Compute USG%, AST%, TOV%, eFG%, OREB%, DREB%, STL% for each player-season.

    All input stats (player and team) are per-game averages.  Formulas use
    per-game values directly — the GP factors cancel out in most ratios.

    Returns the players DataFrame with new columns appended.
    """
    if teams is None or teams.empty:
        for col in ("usg_pct", "ast_pct", "tov_pct", "efg_pct",
                     "oreb_pct", "dreb_pct", "stl_pct"):
            players[col] = float("nan")
        return players

    team_lookup = _build_team_season_lookup(teams)
    abbr_to_name = _build_abbr_to_name_map(teams)
    lg_reb = _league_avg_reb_by_season(teams)

    # Pre-build (team_name, season_str) -> team stats for fast lookup
    # Also build (abbr, season_str) -> team stats
    abbr_lookup: dict[tuple[str, str], dict] = {}
    for (abbr, sstr), tname in abbr_to_name.items():
        key = (tname, sstr)
        if key in team_lookup:
            abbr_lookup[(abbr, sstr)] = team_lookup[key]

    usg_vals = []
    ast_vals = []
    tov_vals = []
    efg_vals = []
    oreb_vals = []
    dreb_vals = []
    stl_vals = []

    for _, row in players.iterrows():
        abbr = str(row.get("team_abbreviation", "")).strip()
        sstr = str(row.get("season_str", ""))

        tm = abbr_lookup.get((abbr, sstr))
        lg = lg_reb.get(sstr, {})

        # Player per-game stats
        p_min = float(row.get("min") or 0)
        p_fga = float(row.get("fga") or 0)
        p_fta = float(row.get("fta") or 0)
        p_tov = float(row.get("tov") or 0)
        p_fgm = float(row.get("fgm") or 0)
        p_fg3m = float(row.get("fg3m") or 0)
        p_ast = float(row.get("ast") or 0)
        p_oreb = float(row.get("oreb") or 0)
        p_dreb = float(row.get("dreb") or 0)
        p_stl = float(row.get("stl") or 0)

        if tm is None or p_min <= 0:
            usg_vals.append(float("nan"))
            ast_vals.append(float("nan"))
            tov_vals.append(float("nan"))
            efg_vals.append(float("nan") if p_fga <= 0 else
                            round((p_fgm + 0.5 * p_fg3m) / p_fga, 4))
            oreb_vals.append(float("nan"))
            dreb_vals.append(float("nan"))
            stl_vals.append(float("nan"))
            continue

        # Team per-game stats
        tm_min = float(tm.get("min") or 0)
        tm_fga = float(tm.get("fga") or 0)
        tm_fta = float(tm.get("fta") or 0)
        tm_tov = float(tm.get("tov") or 0)
        tm_fgm = float(tm.get("fgm") or 0)
        tm_oreb = float(tm.get("oreb") or 0)
        tm_dreb = float(tm.get("dreb") or 0)
        tm_stl = float(tm.get("stl") or 0)
        tm_gp = float(tm.get("gp") or 0)

        # Tm_MP = total team player-minutes per game = game_length * 5
        # (the CSV `min` column is game length ~48, not total player-minutes)
        # Tm_MP/5 = game_length = tm_min
        tm_min_per5 = tm_min if tm_min > 0 else 0

        # --- USG% ---
        # 100 * ((FGA + 0.44*FTA + TOV) * (Tm_MP/5)) / (MP * (Tm_FGA + 0.44*Tm_FTA + Tm_TOV))
        usg_num = (p_fga + 0.44 * p_fta + p_tov) * tm_min_per5
        usg_den = p_min * (tm_fga + 0.44 * tm_fta + tm_tov)
        if usg_den > 0:
            usg_vals.append(round(100 * usg_num / usg_den, 1))
        else:
            usg_vals.append(float("nan"))

        # --- AST% ---
        # 100 * AST / (((MP / (Tm_MP/5)) * Tm_FGM) - FGM)
        if tm_min_per5 > 0:
            teammate_fgm = (p_min / tm_min_per5) * tm_fgm - p_fgm
            if teammate_fgm > 0:
                ast_vals.append(round(100 * p_ast / teammate_fgm, 1))
            else:
                ast_vals.append(float("nan"))
        else:
            ast_vals.append(float("nan"))

        # --- TOV% ---
        # 100 * TOV / (FGA + 0.44*FTA + TOV)
        tov_den = p_fga + 0.44 * p_fta + p_tov
        if tov_den > 0:
            tov_vals.append(round(100 * p_tov / tov_den, 1))
        else:
            tov_vals.append(float("nan"))

        # --- eFG% ---
        # (FGM + 0.5*FG3M) / FGA
        if p_fga > 0:
            efg_vals.append(round((p_fgm + 0.5 * p_fg3m) / p_fga, 4))
        else:
            efg_vals.append(float("nan"))

        # --- OREB% ---
        # 100 * (OREB * (Tm_MP/5)) / (MP * (Tm_OREB + Opp_DRB))
        # Use league avg DRB as proxy for opponent DRB
        lg_dreb = lg.get("dreb", 0)
        oreb_den = p_min * (tm_oreb + lg_dreb)
        if oreb_den > 0:
            oreb_vals.append(round(100 * (p_oreb * tm_min_per5) / oreb_den, 1))
        else:
            oreb_vals.append(float("nan"))

        # --- DREB% ---
        # 100 * (DREB * (Tm_MP/5)) / (MP * (Tm_DREB + Opp_ORB))
        lg_oreb = lg.get("oreb", 0)
        dreb_den = p_min * (tm_dreb + lg_oreb)
        if dreb_den > 0:
            dreb_vals.append(round(100 * (p_dreb * tm_min_per5) / dreb_den, 1))
        else:
            dreb_vals.append(float("nan"))

        # --- STL% ---
        # 100 * (STL * (Tm_MP/5)) / (MP * Tm_Poss)
        # Tm_Poss per game ~ (Tm_PTS / (2 * eFG%)) approximately, or use pace proxy
        # Better: Poss ~ FGA - OREB + TOV + 0.44*FTA (team per-game)
        tm_poss = tm_fga - tm_oreb + tm_tov + 0.44 * tm_fta
        stl_den = p_min * tm_poss
        if stl_den > 0:
            stl_vals.append(round(100 * (p_stl * tm_min_per5) / stl_den, 1))
        else:
            stl_vals.append(float("nan"))

    players = players.copy()
    players["usg_pct"] = usg_vals
    players["ast_pct"] = ast_vals
    players["tov_pct"] = tov_vals
    players["efg_pct"] = efg_vals
    players["oreb_pct"] = oreb_vals
    players["dreb_pct"] = dreb_vals
    players["stl_pct"] = stl_vals

    return players


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
        "usg_pct": g("usg_pct"),
        "ast_pct": g("ast_pct"),
        "tov_pct": g("tov_pct"),
        "efg_pct": _round_or_none(row.get("efg_pct"), 4),
        "oreb_pct": g("oreb_pct"),
        "dreb_pct": g("dreb_pct"),
        "stl_pct": g("stl_pct"),
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


# Position centroids: median stats for each position from 534 known-position
# players (player_positions.csv, 2025-26 rosters).  Used by _infer_position()
# to classify historical players via weighted Euclidean distance.
_POS_CENTROIDS: dict[str, dict[str, float]] = {
    "C":  {"ast_pct": 8.8,  "oreb_pct": 11.1, "dreb_pct": 23.2, "blk": 1.0, "ast": 1.4, "reb": 7.9},
    "PF": {"ast_pct": 8.8,  "oreb_pct": 8.3,  "dreb_pct": 20.4, "blk": 0.7, "ast": 1.5, "reb": 6.4},
    "SF": {"ast_pct": 9.6,  "oreb_pct": 4.5,  "dreb_pct": 15.2, "blk": 0.5, "ast": 1.6, "reb": 4.4},
    "SG": {"ast_pct": 13.2, "oreb_pct": 2.8,  "dreb_pct": 11.9, "blk": 0.3, "ast": 2.3, "reb": 3.4},
    "PG": {"ast_pct": 21.8, "oreb_pct": 2.4,  "dreb_pct": 10.6, "blk": 0.3, "ast": 3.8, "reb": 3.3},
}
_POS_WEIGHTS: dict[str, float] = {
    "ast_pct": 3.0, "oreb_pct": 2.5, "dreb_pct": 2.5,
    "blk": 2.0, "ast": 1.5, "reb": 1.5,
}
_POS_RANGES: dict[str, float] = {}
for _feat in list(_POS_CENTROIDS["C"].keys()):
    _vals = [_POS_CENTROIDS[_p][_feat] for _p in _POS_CENTROIDS]
    _POS_RANGES[_feat] = max(_vals) - min(_vals) or 1.0

# Generic position (G/F/C/G-F etc.) -> allowed specific positions
_GENERIC_POS_MAP: dict[str, set[str]] = {
    "G": {"PG", "SG"}, "F": {"SF", "PF"}, "C": {"C"},
    "G-F": {"SG", "SF"}, "F-G": {"SF", "SG"},
    "F-C": {"PF", "C"}, "C-F": {"C", "PF"},
}


def _centroid_distances(player_feats: dict[str, float]) -> list[tuple[float, str]]:
    """Return (distance, position) pairs sorted by nearest centroid."""
    scored = []
    for pos, centroid in _POS_CENTROIDS.items():
        dist = sum(
            _POS_WEIGHTS[f] * ((player_feats[f] - centroid[f]) / _POS_RANGES[f]) ** 2
            for f in _POS_WEIGHTS
        )
        scored.append((dist, pos))
    scored.sort()
    return scored


def _infer_position(career_avgs: dict, generic_pos: str = "") -> str:
    """Infer position from career averages using centroid-distance scoring.

    When advanced stats (AST%, OREB%, DREB%) are available, uses weighted
    Euclidean distance to position centroids derived from 534 known-position
    players.  Falls back to basic counting stats for pre-tracking-era players.

    *generic_pos* (e.g. "G", "F", "F-C") constrains the result to matching
    specific positions when available.

    Accuracy on 250 known-position test set: ~71% (vs. 14% old heuristic
    which defaulted 73.5% of players to SG).
    """
    ast_pct = career_avgs.get("ast_pct") or 0
    oreb_pct = career_avgs.get("oreb_pct") or 0
    dreb_pct = career_avgs.get("dreb_pct") or 0
    blk = career_avgs.get("blk") or 0
    ast = career_avgs.get("ast") or 0
    reb = career_avgs.get("reb") or 0

    allowed = _GENERIC_POS_MAP.get(generic_pos)
    ast_reb_ratio = ast / max(reb, 0.1)
    has_advanced = ast_pct > 0 and oreb_pct > 0

    if has_advanced:
        feats = {
            "ast_pct": ast_pct, "oreb_pct": oreb_pct, "dreb_pct": dreb_pct,
            "blk": blk, "ast": ast, "reb": reb,
        }
        scored = _centroid_distances(feats)

        if allowed:
            for _, pos in scored:
                if pos in allowed:
                    return pos

        # PG/SG tiebreaker: ast/reb ratio (PG median 1.21, SG median 0.65)
        top, runner = scored[0][1], scored[1][1]
        if {top, runner} == {"PG", "SG"}:
            return "PG" if ast_reb_ratio >= 0.95 else "SG"
        return top

    # Fallback: basic counting stats for pre-tracking-era players
    if reb >= 8 and blk >= 1.0:
        best = "C"
    elif reb >= 7.5:
        best = "PF"
    elif ast >= 6:
        best = "PG"
    elif reb >= 5 and blk >= 0.5:
        best = "SF"
    elif reb >= 4.5:
        best = "SF"
    elif ast_reb_ratio >= 1.0:
        best = "PG"
    elif ast >= 2.5 and reb < 3.5:
        best = "SG"
    elif reb >= 3.5:
        best = "SF"
    else:
        best = "SG"

    if allowed and best not in allowed:
        if "PG" in allowed and ast_reb_ratio > 0.8:
            return "PG"
        if "SG" in allowed:
            return "SG"
        return min(allowed)
    return best


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
            "usg_pct": wavg("usg_pct"),
            "ast_pct": wavg("ast_pct"),
            "tov_pct": wavg("tov_pct"),
            "efg_pct": wavg("efg_pct"),
            "oreb_pct": wavg("oreb_pct"),
            "dreb_pct": wavg("dreb_pct"),
            "stl_pct": wavg("stl_pct"),
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

        # Position: prefer roster data, fall back to centroid-based inference
        pos_info = pos_lookup.get(int(player_id), {})
        position = pos_info.get("position", "")
        positions = pos_info.get("positions", "")
        position_primary = pos_info.get("position_primary", "")
        jersey_number = pos_info.get("jersey_number", "")
        if not positions:
            positions = _infer_position(career_avgs, position)
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
        "player_id": -1, "nba_id": 893, "player_name": "Michael Jordan",
        "positions": "SG", "position_primary": "SG", "jersey_number": "23",
        "seasons_span": "1984-2003", "career_gp": 1072,
        "career_avgs": {"pts": 30.1, "reb": 6.2, "ast": 5.3, "stl": 2.35, "blk": 0.83,
                        "pts_normalized": None,
                        "usg_pct": 33.3, "ast_pct": 24.9, "tov_pct": 9.3,
                        "efg_pct": 0.509, "oreb_pct": 4.7, "dreb_pct": 14.1, "stl_pct": 3.1},
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
        "player_id": -2, "nba_id": 1449, "player_name": "Larry Bird",
        "positions": "SF, PF", "position_primary": "SF", "jersey_number": "33",
        "seasons_span": "1979-1992", "career_gp": 897,
        "career_avgs": {"pts": 24.3, "reb": 10.0, "ast": 6.3, "stl": 1.74, "blk": 0.84,
                        "pts_normalized": None,
                        "usg_pct": 26.5, "ast_pct": 24.7, "tov_pct": 12.7,
                        "efg_pct": 0.514, "oreb_pct": 5.9, "dreb_pct": 22.4, "stl_pct": 2.2},
        "best_season": "1987-88",
        "ts_pct": 0.584, "fg_pct": 0.496, "fg3_pct": 0.376, "ft_pct": 0.886,
        "seasons": [],
        "_legend": True, "_note": "Career stats from Basketball Reference.",
    },
    {
        "player_id": -3, "nba_id": 77142, "player_name": "Magic Johnson",
        "positions": "PG", "position_primary": "PG", "jersey_number": "32",
        "seasons_span": "1979-1996", "career_gp": 906,
        "career_avgs": {"pts": 19.5, "reb": 7.2, "ast": 11.2, "stl": 1.90, "blk": 0.37,
                        "pts_normalized": None,
                        "usg_pct": 22.3, "ast_pct": 40.9, "tov_pct": 19.4,
                        "efg_pct": 0.533, "oreb_pct": 5.7, "dreb_pct": 15.8, "stl_pct": 2.5},
        "best_season": "1988-89",
        "ts_pct": 0.584, "fg_pct": 0.520, "fg3_pct": 0.303, "ft_pct": 0.848,
        "seasons": [],
        "_legend": True, "_note": "Career stats from Basketball Reference.",
    },
    {
        "player_id": -4, "nba_id": 76003, "player_name": "Kareem Abdul-Jabbar",
        "positions": "C", "position_primary": "C", "jersey_number": "33",
        "seasons_span": "1969-1989", "career_gp": 1560,
        "career_avgs": {"pts": 24.6, "reb": 11.2, "ast": 3.6, "stl": 0.94, "blk": 2.60,
                        "pts_normalized": None,
                        "usg_pct": 24.3, "ast_pct": 14.6, "tov_pct": 13.4,
                        "efg_pct": 0.559, "oreb_pct": 7.7, "dreb_pct": 21.7, "stl_pct": 1.2},
        "best_season": "1971-72",
        "ts_pct": 0.579, "fg_pct": 0.559, "fg3_pct": 0.056, "ft_pct": 0.721,
        "seasons": [],
        "_legend": True, "_note": "Career stats from Basketball Reference.",
    },
    {
        "player_id": -5, "nba_id": 977, "player_name": "Kobe Bryant",
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
        "player_id": -6, "nba_id": 76375, "player_name": "Wilt Chamberlain",
        "positions": "C", "position_primary": "C", "jersey_number": "13",
        "seasons_span": "1959-1973", "career_gp": 1045,
        "career_avgs": {"pts": 30.1, "reb": 22.9, "ast": 4.4, "stl": None, "blk": None,
                        "pts_normalized": None,
                        "usg_pct": None, "ast_pct": 15.8, "tov_pct": None,
                        "efg_pct": 0.540, "oreb_pct": None, "dreb_pct": None, "stl_pct": None},
        "best_season": "1961-62",
        "ts_pct": 0.541, "fg_pct": 0.540, "fg3_pct": None, "ft_pct": 0.511,
        "seasons": [],
        "_legend": True, "_note": "Career stats from Basketball Reference.",
    },
]


def _inject_legends(player_records: list[dict],
                    enriched: pd.DataFrame | None = None) -> list[dict]:
    """Replace partial API records with curated legend stats; append if not present.

    If *enriched* is provided and a legend has an empty ``seasons`` list, populate
    it from the enriched DataFrame (which already has normalized/per36/pace stats)
    so the output matches the format produced by ``build_player_records()``.
    """
    # Build a lookup: legend player_name -> enriched rows from CSV
    _legend_seasons_from_csv: dict[str, list[dict]] = {}
    if enriched is not None and not enriched.empty:
        for leg in _LEGENDS:
            if leg["seasons"]:
                # Legend already has hardcoded season data — skip CSV lookup
                continue
            name = leg["player_name"]
            mask = enriched["player_name"] == name
            grp = enriched.loc[mask].sort_values("season")
            if grp.empty:
                continue
            _legend_seasons_from_csv[name] = [
                _season_row(row) for _, row in grp.iterrows()
            ]

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
        entry = dict(leg)
        entry["career_avgs"] = dict(entry.get("career_avgs") or {})
        # Fill empty seasons from CSV-enriched data when available
        if not entry["seasons"] and entry["player_name"] in _legend_seasons_from_csv:
            entry["seasons"] = _legend_seasons_from_csv[entry["player_name"]]
        # Compute GP-weighted advanced stat averages from filled seasons
        adv_keys = ("usg_pct", "ast_pct", "tov_pct", "efg_pct",
                    "oreb_pct", "dreb_pct", "stl_pct")
        for key in adv_keys:
            if entry["career_avgs"].get(key) is not None:
                continue  # already set
            total_gp = 0
            weighted_sum = 0.0
            for s in entry["seasons"]:
                gp = s.get("gp") or 0
                val = s.get(key)
                if gp > 0 and val is not None:
                    weighted_sum += val * gp
                    total_gp += gp
            if total_gp > 0:
                entry["career_avgs"][key] = round(weighted_sum / total_gp, 1)
        result.append(entry)
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
        log.warning("  WARN: game logs directory not found, using pre-1996 records only")
        for stat in records:
            records[stat].sort(key=lambda r: r["val"], reverse=True)
            records[stat] = records[stat][:top_n]
        return records

    # Load all game log CSVs
    csv_files = sorted(GAME_LOGS_DIR.glob("*.csv"))
    if not csv_files:
        log.warning("  WARN: no game log CSVs found")
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
    log.info("build_player_comparison: loading player data...")
    try:
        players = _load_players(PLAYERS_CSV)
    except FileNotFoundError as exc:
        log.warning(f"WARN: {exc} -- skipping player_comparison.json write")
        return {"players": 0, "seasons": 0}
    log.info(f"  {len(players):,} player-season rows across "
          f"{players['season_str'].nunique()} seasons")

    log.info("build_player_comparison: loading team data (for pace)...")
    teams = _load_teams(TEAMS_CSV)

    log.info("build_player_comparison: computing league averages...")
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
    log.info(f"  Historical averages: pts={hist_avgs['pts']:.2f} reb={hist_avgs['reb']:.2f} "
          f"ast={hist_avgs['ast']:.2f} pace={hist_avgs['pace']:.1f}")

    log.info("build_player_comparison: enriching player seasons...")
    enriched = enrich_player_seasons(players, league, hist_avgs)

    log.info("build_player_comparison: computing advanced stats (USG%, AST%, etc.)...")
    enriched = compute_advanced_stats(enriched, teams)
    adv_count = enriched["usg_pct"].notna().sum()
    log.info(f"  {adv_count:,} player-seasons with advanced stats")

    log.info("build_player_comparison: loading position data...")
    pos_lookup = _load_position_lookup()
    log.info(f"  {len(pos_lookup)} players with roster positions")

    log.info(f"build_player_comparison: building player records "
          f"(min_seasons={min_seasons}, min_career_games={min_career_games})...")
    player_records = build_player_records(enriched, min_seasons, min_career_games, pos_lookup)
    league_rows = build_league_by_season(league)

    log.info(f"  {len(player_records):,} eligible players")

    # Inject curated legend overrides for pre-1996 players missing from NBA API.
    # Stats sourced from Basketball Reference career regular-season averages.
    # Must happen BEFORE building the player index so legends appear in search.
    player_records = _inject_legends(player_records, enriched=enriched)
    log.info(f"  {len(player_records):,} players after legend injection")

    player_index = build_player_index(player_records)

    # Build single-game records from game logs + historical floor
    log.info("build_player_comparison: computing single-game records...")
    single_game_records = _build_single_game_records(top_n=15)
    for stat_key, recs in single_game_records.items():
        log.warning(f"  {stat_key}: top={recs[0]['name']} ({recs[0]['val']})" if recs else f"  {stat_key}: empty")

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

    log.info(f"build_player_comparison: written -> {OUT_COMPARISON} "
          f"({OUT_COMPARISON.stat().st_size / 1024:.0f} KB)")

    # Write player index
    with open(OUT_INDEX, "w", encoding="utf-8") as f:
        json.dump(player_index, f, separators=(",", ":"))

    log.info(f"build_player_comparison: written -> {OUT_INDEX} "
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
    log.info(f"build_player_comparison: done. "
          f"{summary['players']} players, {summary['seasons']} seasons.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
