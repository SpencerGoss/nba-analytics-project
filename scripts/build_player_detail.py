"""
build_player_detail.py -- produce dashboard/data/player_detail.json

Aggregates rich per-player data from multiple CSVs into a single JSON keyed
by player name, for use in the player detail modal in the dashboard.

Data sources (all from data/processed/):
  - player_stats.csv          -- season totals (divide by gp for per-game)
  - player_stats_advanced.csv -- TS%, USG%, PIE, off/def/net ratings
  - player_stats_scoring.csv  -- shot-zone distribution percentages
  - player_stats_clutch.csv   -- last-5-min/margin<=5 stats
  - player_hustle_stats.csv   -- screen assists, deflections, box-outs
  - player_game_logs.csv      -- per-game results for last-5 form

JSON format (keyed by player_name):
{
  "Shai Gilgeous-Alexander": {
    "player_id": 1628983,
    "name": "Shai Gilgeous-Alexander",
    "team": "OKC",
    "season_stats": {
      "ppg": 32.1, "rpg": 5.5, "apg": 6.4, "spg": 2.0, "bpg": 1.1,
      "topg": 2.8, "mpg": 34.7,
      "fg_pct": 0.535, "fg3_pct": 0.342, "ft_pct": 0.874,
      "ts_pct": 0.638, "usg_pct": 0.322, "pie": 0.175,
      "efg_pct": 0.562, "off_rating": 125.3, "def_rating": 107.2,
      "net_rating": 18.1, "gp": 55
    },
    "shooting_splits": {
      "pct_fga_2pt": 0.62, "pct_fga_3pt": 0.38,
      "pct_pts_paint": 0.28, "pct_pts_3pt": 0.24,
      "pct_pts_ft": 0.28, "pct_pts_2pt_mr": 0.12,
      "pct_ast_fgm": 0.42, "pct_uast_fgm": 0.58
    },
    "clutch_stats": {
      "ppg": 6.2, "rpg": 1.1, "apg": 1.8, "fg_pct": 0.512, "gp": 32
    },
    "hustle_stats": {
      "contested_shots_pg": 4.2, "deflections_pg": 1.8,
      "screen_assists_pg": 0.3, "box_outs_pg": 1.6,
      "charges_drawn_pg": 0.2
    },
    "last_5_games": [
      {"date": "2026-03-06", "opp": "GSW", "pts": 38, "reb": 6, "ast": 7,
       "stl": 2, "blk": 1, "fg_pct": 0.571, "result": "W"}
    ],
    "prop_trends": {
      "pts_last5_avg": 31.4, "pts_season_avg": 32.1, "pts_trend": "flat",
      "reb_last5_avg": 5.2, "reb_season_avg": 5.5, "reb_trend": "flat",
      "ast_last5_avg": 7.0, "ast_season_avg": 6.4, "ast_trend": "up"
    }
  }
}

Run: python scripts/build_player_detail.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# --- Input paths ---
STATS_CSV     = PROJECT_ROOT / "data" / "processed" / "player_stats.csv"
ADV_CSV       = PROJECT_ROOT / "data" / "processed" / "player_stats_advanced.csv"
SCORING_CSV   = PROJECT_ROOT / "data" / "processed" / "player_stats_scoring.csv"
CLUTCH_CSV    = PROJECT_ROOT / "data" / "processed" / "player_stats_clutch.csv"
HUSTLE_CSV    = PROJECT_ROOT / "data" / "processed" / "player_hustle_stats.csv"
LOGS_CSV      = PROJECT_ROOT / "data" / "processed" / "player_game_logs.csv"

# --- Output path ---
OUT_JSON = PROJECT_ROOT / "dashboard" / "data" / "player_detail.json"

CURRENT_SEASON = 202526
MIN_GP = 5           # minimum games played to include
LAST_N_GAMES = 5     # number of recent games for form section
TREND_THRESHOLD = 0.10  # 10% above/below season avg -> up/down


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(val: Any, decimals: int = 3) -> float | None:
    """Return rounded float or None if missing/NaN."""
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if pd.isna(f):
        return None
    return round(f, decimals)


def _safe_int(val: Any) -> int | None:
    """Return int or None if missing/NaN."""
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if pd.isna(f):
        return None
    return int(f)


def _per_game(total: Any, gp: int) -> float | None:
    """Divide total by gp; return None if either is missing."""
    t = _safe_float(total)
    if t is None or not gp:
        return None
    return round(t / gp, 1)


def _trend(last5_avg: float | None, season_avg: float | None) -> str:
    """Classify stat trend relative to season average."""
    if last5_avg is None or season_avg is None or season_avg == 0:
        return "flat"
    ratio = (last5_avg - season_avg) / season_avg
    if ratio > TREND_THRESHOLD:
        return "up"
    if ratio < -TREND_THRESHOLD:
        return "down"
    return "flat"


# ---------------------------------------------------------------------------
# Per-source loaders  (return dicts keyed by player_id)
# ---------------------------------------------------------------------------

def _load_season_stats(df_stats: pd.DataFrame) -> dict[int, dict]:
    """Load season totals -> per-game averages from player_stats.csv."""
    cur = df_stats[df_stats["season"] == CURRENT_SEASON].copy()
    # Keep highest-gp row when a player traded mid-season (team_count > 1 rows)
    cur = cur.sort_values("gp", ascending=False).drop_duplicates(
        subset=["player_id"], keep="first"
    )

    result: dict[int, dict] = {}
    for _, row in cur.iterrows():
        pid = _safe_int(row["player_id"])
        gp = _safe_int(row["gp"]) or 0
        if pid is None or gp < MIN_GP:
            continue

        result[pid] = {
            "ppg":   _per_game(row.get("pts"), gp),
            "rpg":   _per_game(row.get("reb"), gp),
            "apg":   _per_game(row.get("ast"), gp),
            "spg":   _per_game(row.get("stl"), gp),
            "bpg":   _per_game(row.get("blk"), gp),
            "topg":  _per_game(row.get("tov"), gp),
            "mpg":   _per_game(row.get("min"), gp),
            "fg_pct":  _safe_float(row.get("fg_pct")),
            "fg3_pct": _safe_float(row.get("fg3_pct")),
            "ft_pct":  _safe_float(row.get("ft_pct")),
            "gp": gp,
            "team": str(row.get("team_abbreviation") or ""),
            "name": str(row.get("player_name") or ""),
        }
    return result


def _load_advanced(df_adv: pd.DataFrame) -> dict[int, dict]:
    """Load advanced metrics from player_stats_advanced.csv."""
    cur = df_adv[df_adv["season"] == CURRENT_SEASON].copy()
    cur = cur.sort_values("gp", ascending=False).drop_duplicates(
        subset=["player_id"], keep="first"
    )

    result: dict[int, dict] = {}
    for _, row in cur.iterrows():
        pid = _safe_int(row["player_id"])
        if pid is None:
            continue
        result[pid] = {
            "ts_pct":     _safe_float(row.get("ts_pct")),
            "usg_pct":    _safe_float(row.get("usg_pct")),
            "pie":        _safe_float(row.get("pie")),
            "efg_pct":    _safe_float(row.get("efg_pct")),
            "off_rating": _safe_float(row.get("off_rating"), decimals=1),
            "def_rating": _safe_float(row.get("def_rating"), decimals=1),
            "net_rating": _safe_float(row.get("net_rating"), decimals=1),
            "ast_pct":    _safe_float(row.get("ast_pct")),
            "oreb_pct":   _safe_float(row.get("oreb_pct")),
            "dreb_pct":   _safe_float(row.get("dreb_pct")),
        }
    return result


def _load_scoring(df_scoring: pd.DataFrame) -> dict[int, dict]:
    """Load shot-distribution splits from player_stats_scoring.csv."""
    cur = df_scoring[df_scoring["season"] == CURRENT_SEASON].copy()
    cur = cur.sort_values("gp", ascending=False).drop_duplicates(
        subset=["player_id"], keep="first"
    )

    result: dict[int, dict] = {}
    for _, row in cur.iterrows():
        pid = _safe_int(row["player_id"])
        if pid is None:
            continue
        result[pid] = {
            "pct_fga_2pt":    _safe_float(row.get("pct_fga_2pt")),
            "pct_fga_3pt":    _safe_float(row.get("pct_fga_3pt")),
            "pct_pts_2pt":    _safe_float(row.get("pct_pts_2pt")),
            "pct_pts_2pt_mr": _safe_float(row.get("pct_pts_2pt_mr")),
            "pct_pts_3pt":    _safe_float(row.get("pct_pts_3pt")),
            "pct_pts_paint":  _safe_float(row.get("pct_pts_paint")),
            "pct_pts_ft":     _safe_float(row.get("pct_pts_ft")),
            "pct_pts_fb":     _safe_float(row.get("pct_pts_fb")),
            "pct_ast_fgm":    _safe_float(row.get("pct_ast_fgm")),
            "pct_uast_fgm":   _safe_float(row.get("pct_uast_fgm")),
        }
    return result


def _load_clutch(df_clutch: pd.DataFrame) -> dict[int, dict]:
    """Load last-5-min clutch stats from player_stats_clutch.csv."""
    cur = df_clutch[df_clutch["season"] == CURRENT_SEASON].copy()
    cur = cur.sort_values("gp", ascending=False).drop_duplicates(
        subset=["player_id"], keep="first"
    )

    result: dict[int, dict] = {}
    for _, row in cur.iterrows():
        pid = _safe_int(row["player_id"])
        gp = _safe_int(row.get("gp")) or 0
        if pid is None or gp == 0:
            continue
        result[pid] = {
            "ppg":    _per_game(row.get("pts"), gp),
            "rpg":    _per_game(row.get("reb"), gp),
            "apg":    _per_game(row.get("ast"), gp),
            "fg_pct": _safe_float(row.get("fg_pct")),
            "ft_pct": _safe_float(row.get("ft_pct")),
            "plus_minus_pg": _per_game(row.get("plus_minus"), gp),
            "gp": gp,
        }
    return result


def _load_hustle(df_hustle: pd.DataFrame) -> dict[int, dict]:
    """Load hustle stats per-game from player_hustle_stats.csv."""
    cur = df_hustle[df_hustle["season"] == CURRENT_SEASON].copy()
    cur = cur.sort_values("g", ascending=False).drop_duplicates(
        subset=["player_id"], keep="first"
    )

    result: dict[int, dict] = {}
    for _, row in cur.iterrows():
        pid = _safe_int(row["player_id"])
        gp = _safe_int(row.get("g")) or 0
        if pid is None or gp == 0:
            continue
        result[pid] = {
            "contested_shots_pg": _per_game(row.get("contested_shots"), gp),
            "deflections_pg":     _per_game(row.get("deflections"), gp),
            "screen_assists_pg":  _per_game(row.get("screen_assists"), gp),
            "box_outs_pg":        _per_game(row.get("box_outs"), gp),
            "charges_drawn_pg":   _per_game(row.get("charges_drawn"), gp),
            "loose_balls_pg":     _per_game(row.get("loose_balls_recovered"), gp),
        }
    return result


def _load_game_logs(df_logs: pd.DataFrame) -> dict[int, list[dict]]:
    """Load last-N games and compute prop-trend data from player_game_logs.csv."""
    cur = df_logs[df_logs["season"] == CURRENT_SEASON].copy()
    cur["game_date"] = pd.to_datetime(cur["game_date"], format="mixed")
    cur = cur.sort_values("game_date", ascending=False)

    # Index by player_id, keep most recent LAST_N_GAMES
    result: dict[int, list[dict]] = {}
    for pid, grp in cur.groupby("player_id"):
        pid_int = _safe_int(pid)
        if pid_int is None:
            continue
        recent = grp.head(LAST_N_GAMES)
        games = []
        for _, row in recent.iterrows():
            # Extract opponent from matchup string (e.g. "OKC vs. HOU" or "OKC @ HOU")
            matchup = str(row.get("matchup") or "")
            team_abbr = str(row.get("team_abbreviation") or "")
            opp = _parse_opponent(matchup, team_abbr)

            games.append({
                "date":   row["game_date"].strftime("%Y-%m-%d"),
                "opp":    opp,
                "pts":    _safe_int(row.get("pts")),
                "reb":    _safe_int(row.get("reb")),
                "ast":    _safe_int(row.get("ast")),
                "stl":    _safe_int(row.get("stl")),
                "blk":    _safe_int(row.get("blk")),
                "fg_pct": _safe_float(row.get("fg_pct")),
                "min":    _safe_int(row.get("min")),
                "result": str(row.get("wl") or ""),
            })
        result[pid_int] = games
    return result


def _compute_prop_trends(
    last5: list[dict],
    season_stats: dict,
) -> dict:
    """Compute last-5 averages and trend direction vs season averages."""
    if not last5:
        return {}

    def avg(key: str) -> float | None:
        vals = [g[key] for g in last5 if g.get(key) is not None]
        if not vals:
            return None
        return round(sum(vals) / len(vals), 1)

    pts_l5  = avg("pts")
    reb_l5  = avg("reb")
    ast_l5  = avg("ast")
    pts_sa  = season_stats.get("ppg")
    reb_sa  = season_stats.get("rpg")
    ast_sa  = season_stats.get("apg")

    return {
        "pts_last5_avg":  pts_l5,
        "pts_season_avg": pts_sa,
        "pts_trend":      _trend(pts_l5, pts_sa),
        "reb_last5_avg":  reb_l5,
        "reb_season_avg": reb_sa,
        "reb_trend":      _trend(reb_l5, reb_sa),
        "ast_last5_avg":  ast_l5,
        "ast_season_avg": ast_sa,
        "ast_trend":      _trend(ast_l5, ast_sa),
    }


def _parse_opponent(matchup: str, team: str) -> str:
    """Extract the opposing team abbreviation from a matchup string."""
    # Format: "OKC vs. HOU"  or "OKC @ HOU"
    if " vs. " in matchup:
        parts = matchup.split(" vs. ")
    elif " @ " in matchup:
        parts = matchup.split(" @ ")
    else:
        return ""
    if len(parts) != 2:
        return ""
    home, away = parts[0].strip(), parts[1].strip()
    if home == team:
        return away
    return home


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build_player_detail() -> dict:
    """Build and return the full player detail dict keyed by player name."""

    # --- Load all CSVs (guard each with exists check) ---
    df_stats   = pd.read_csv(STATS_CSV)   if STATS_CSV.exists()   else pd.DataFrame()
    df_adv     = pd.read_csv(ADV_CSV)     if ADV_CSV.exists()     else pd.DataFrame()
    df_scoring = pd.read_csv(SCORING_CSV) if SCORING_CSV.exists() else pd.DataFrame()
    df_clutch  = pd.read_csv(CLUTCH_CSV)  if CLUTCH_CSV.exists()  else pd.DataFrame()
    df_hustle  = pd.read_csv(HUSTLE_CSV)  if HUSTLE_CSV.exists()  else pd.DataFrame()
    df_logs    = pd.read_csv(LOGS_CSV)    if LOGS_CSV.exists()    else pd.DataFrame()

    print(f"  Loaded stats:{len(df_stats)} adv:{len(df_adv)} "
          f"scoring:{len(df_scoring)} clutch:{len(df_clutch)} "
          f"hustle:{len(df_hustle)} logs:{len(df_logs)}")

    # --- Build per-source lookup dicts ---
    season_by_pid  = _load_season_stats(df_stats)  if not df_stats.empty   else {}
    adv_by_pid     = _load_advanced(df_adv)         if not df_adv.empty     else {}
    scoring_by_pid = _load_scoring(df_scoring)      if not df_scoring.empty else {}
    clutch_by_pid  = _load_clutch(df_clutch)        if not df_clutch.empty  else {}
    hustle_by_pid  = _load_hustle(df_hustle)        if not df_hustle.empty  else {}
    logs_by_pid    = _load_game_logs(df_logs)       if not df_logs.empty    else {}

    print(f"  Current-season players: stats={len(season_by_pid)} "
          f"adv={len(adv_by_pid)} scoring={len(scoring_by_pid)} "
          f"clutch={len(clutch_by_pid)} hustle={len(hustle_by_pid)}")

    # --- Merge into output records keyed by player name ---
    result: dict[str, dict] = {}

    for pid, base in season_by_pid.items():
        name = base.get("name") or ""
        if not name:
            continue

        # Merge advanced metrics into season_stats
        adv = adv_by_pid.get(pid, {})
        season_stats = {
            **base,
            "ts_pct":     adv.get("ts_pct"),
            "usg_pct":    adv.get("usg_pct"),
            "pie":        adv.get("pie"),
            "efg_pct":    adv.get("efg_pct"),
            "off_rating": adv.get("off_rating"),
            "def_rating": adv.get("def_rating"),
            "net_rating": adv.get("net_rating"),
            "ast_pct":    adv.get("ast_pct"),
            "oreb_pct":   adv.get("oreb_pct"),
            "dreb_pct":   adv.get("dreb_pct"),
        }
        # Remove internal-only keys from the output
        season_stats.pop("name", None)
        season_stats.pop("team", None)

        last5 = logs_by_pid.get(pid, [])
        prop_trends = _compute_prop_trends(last5, season_stats)

        result[name] = {
            "player_id":      pid,
            "name":           name,
            "team":           base.get("team") or "",
            "season_stats":   season_stats,
            "shooting_splits": scoring_by_pid.get(pid),
            "clutch_stats":   clutch_by_pid.get(pid),
            "hustle_stats":   hustle_by_pid.get(pid),
            "last_5_games":   last5,
            "prop_trends":    prop_trends if prop_trends else None,
        }

    return result


def main() -> None:
    print("Building player detail JSON...")
    data = build_player_detail()
    print(f"  Built {len(data)} player records")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "players": data,
                "season": CURRENT_SEASON,
                "exported_at": datetime.now(timezone.utc).isoformat(),
            },
            fh,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    size_kb = OUT_JSON.stat().st_size // 1024
    print(f"  Written -> {OUT_JSON} ({size_kb} KB)")

    # Spot-check a few notable players
    for name in [
        "Shai Gilgeous-Alexander", "Nikola Jokic",
        "LeBron James", "Stephen Curry", "Jayson Tatum",
    ]:
        rec = data.get(name)
        if rec:
            ss = rec["season_stats"]
            cl = rec.get("clutch_stats") or {}
            trend = rec.get("prop_trends") or {}
            print(
                f"  {name}: {ss.get('ppg')} ppg/{ss.get('rpg')} rpg/"
                f"{ss.get('apg')} apg | clutch_gp={cl.get('gp')} | "
                f"pts_trend={trend.get('pts_trend')}"
            )
        else:
            print(f"  {name}: not in {CURRENT_SEASON}")


if __name__ == "__main__":
    main()
