"""
build_advanced_stats.py -- produce dashboard/data/advanced_stats.json

Reads data/processed/player_stats_advanced.csv (current season) and
outputs real TS%, USG%, off/def/net ratings for all players.

Used by dashboard to replace the hardcoded ADV constant with real data.

JSON format:
{
  "PlayerName": {
    "ts": 66.6,      -- True Shooting % (0-100)
    "usg": 32.4,     -- Usage Rate % (0-100)
    "off_rtg": 120.7,
    "def_rtg": 105.2,
    "net_rtg": 15.4,
    "efg": 58.2,     -- Effective FG % (0-100)
    "pie": 21.0,     -- Player Impact Estimate % (0-100, nba.com metric)
    "gp": 52
  },
  ...
}

Run: python scripts/build_advanced_stats.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ADV_CSV = PROJECT_ROOT / "data" / "processed" / "player_stats_advanced.csv"
OUT_JSON = PROJECT_ROOT / "dashboard" / "data" / "advanced_stats.json"

from src.config import get_current_season

CURRENT_SEASON = get_current_season()
MIN_GP = 5  # minimum games played to include


def build_advanced_stats() -> dict:
    if not ADV_CSV.exists():
        print(f"  advanced stats CSV not found: {ADV_CSV}")
        return {}

    df = pd.read_csv(ADV_CSV)

    # Filter to current season
    cur = df[df["season"] == CURRENT_SEASON].copy()
    print(f"  {len(cur)} player-rows in season {CURRENT_SEASON}")

    # When a player appears on multiple teams, keep the row with most games
    cur = cur.sort_values("gp", ascending=False).drop_duplicates(
        subset=["player_name"], keep="first"
    )
    print(f"  {len(cur)} unique players after dedup")

    # Filter out low-sample players
    cur = cur[cur["gp"] >= MIN_GP]
    print(f"  {len(cur)} players with >= {MIN_GP} GP")

    result: dict[str, dict] = {}
    for _, row in cur.iterrows():
        name = str(row["player_name"])

        def _pct(val: float | None, multiply: bool = True) -> float | None:
            """Convert 0-1 fraction to 0-100 pct, or None if missing."""
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return None
            return round(float(val) * 100 if multiply else float(val), 1)

        def _flt(val: float | None) -> float | None:
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return None
            return round(float(val), 1)

        result[name] = {
            "ts":      _pct(row.get("ts_pct")),
            "usg":     _pct(row.get("usg_pct")),
            "off_rtg": _flt(row.get("off_rating")),
            "def_rtg": _flt(row.get("def_rating")),
            "net_rtg": _flt(row.get("net_rating")),
            "efg":     _pct(row.get("efg_pct")),
            "pie":     _pct(row.get("pie")),
            "gp":      int(row["gp"]) if pd.notna(row["gp"]) else None,
        }

    return result


def main() -> None:
    print("Building advanced stats...")
    data = build_advanced_stats()
    print(f"  Built {len(data)} player entries")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(
            {"players": data, "season": CURRENT_SEASON,
             "exported_at": datetime.now(timezone.utc).isoformat()},
            fh, separators=(",", ":"), ensure_ascii=False
        )
    print(f"  Written -> {OUT_JSON}")

    # Preview a few players
    for name in ["Shai Gilgeous-Alexander", "Nikola Jokic", "Giannis Antetokounmpo",
                 "LeBron James", "Stephen Curry"]:
        if name in data:
            d = data[name]
            print(f"  {name}: ts={d['ts']}% usg={d['usg']}% net={d['net_rtg']}")
        else:
            print(f"  {name}: not found in {CURRENT_SEASON}")


if __name__ == "__main__":
    main()
