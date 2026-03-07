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


def build_player_records(enriched: pd.DataFrame,
                          min_seasons: int,
                          min_career_games: int) -> list[dict]:
    """
    Group by player, build career summary, filter to eligibility threshold.
    Returns list of player dicts sorted by career pts_normalized desc.
    """
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

        records.append({
            "player_id": int(player_id),
            "player_name": str(player_name),
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
# Main
# ---------------------------------------------------------------------------

def run(min_seasons: int = DEFAULT_MIN_SEASONS,
        min_career_games: int = DEFAULT_MIN_CAREER_GAMES) -> dict:
    """Execute the full build. Returns summary dict for logging."""
    print("build_player_comparison: loading player data...")
    players = _load_players(PLAYERS_CSV)
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

    print(f"build_player_comparison: building player records "
          f"(min_seasons={min_seasons}, min_career_games={min_career_games})...")
    player_records = build_player_records(enriched, min_seasons, min_career_games)
    league_rows = build_league_by_season(league)
    player_index = build_player_index(player_records)

    print(f"  {len(player_records):,} eligible players")

    # Write comparison JSON
    comparison = {
        "players": player_records,
        "league_by_season": league_rows,
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
