"""
Fetch player positions from NBA team rosters for all 30 teams.

Saves to data/processed/player_positions.csv with columns:
  player_id, player_name, position, height, apg, positions, position_primary, team

Positions from the roster endpoint use NBA format:
  G, G-F, F, F-C, C, F-G, C-F, etc.

We map these to granular positions (PG/SG/SF/PF/C) using NBA label + height +
per-game assists from player_stats.csv.

Run: python scripts/fetch_player_positions.py
"""
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nba_api.stats.endpoints import commonteamroster
from nba_api.stats.static import teams

OUT_CSV = PROJECT_ROOT / "data" / "processed" / "player_positions.csv"
STATS_CSV = PROJECT_ROOT / "data" / "processed" / "player_stats.csv"
from src.config import get_current_season
import logging

log = logging.getLogger(__name__)

CURRENT_SEASON_INT = get_current_season()
# NBA API expects "2025-26" format string
_s = str(CURRENT_SEASON_INT)
SEASON = f"{_s[:4]}-{_s[4:]}"


def _parse_height_inches(height_str: str) -> int | None:
    """Convert NBA height string like '6-6' to total inches (78)."""
    if not height_str or height_str.strip() == "":
        return None
    parts = height_str.strip().split("-")
    if len(parts) != 2:
        return None
    try:
        feet, inches = int(parts[0]), int(parts[1])
        return feet * 12 + inches
    except (ValueError, TypeError):
        return None


def _load_apg_lookup() -> dict[int, float]:
    """Load per-game assists from player_stats.csv for current season."""
    if not STATS_CSV.exists():
        return {}
    df = pd.read_csv(STATS_CSV)
    cur = df[df["season"] == CURRENT_SEASON_INT].copy()
    # Keep highest-gp row for traded players
    cur = cur.sort_values("gp", ascending=False).drop_duplicates(
        subset=["player_id"], keep="first"
    )
    result: dict[int, float] = {}
    for _, row in cur.iterrows():
        pid = int(row["player_id"])
        gp = int(row.get("gp") or 0)
        ast = float(row.get("ast") or 0)
        if gp > 0:
            result[pid] = round(ast / gp, 1)
    return result


def _map_positions(nba_label: str, height_in: int | None, apg: float) -> str:
    """Map NBA roster position + height + APG to granular positions string.

    Returns comma-separated positions like 'PG, SG' or 'SF, PF'.
    """
    label = (nba_label or "").strip().upper()
    if not label:
        return ""

    h = height_in or 78  # default ~6'6" if unknown

    match label:
        case "G":
            if h <= 76:
                return "PG"
            if 77 <= h <= 78:
                return "PG, SG" if apg >= 5 else "SG"
            return "SG"  # 79+
        case "G-F":
            return "SG, SF"
        case "F-G":
            return "PG, SG" if apg >= 5 else "SG, SF"
        case "F":
            if h <= 79:
                return "SF"
            if 80 <= h <= 81:
                return "SF, PF"
            return "PF"  # 82+
        case "F-C":
            return "PF, C"
        case "C-F":
            return "C, PF"
        case "C":
            return "C"
        case _:
            # Unexpected label — best-effort from first character
            first = label.split("-")[0]
            if first == "G":
                return "SG"
            if first == "F":
                return "SF"
            if first == "C":
                return "C"
            return ""


def fetch_all_positions() -> pd.DataFrame:
    """Fetch roster positions for all 30 NBA teams."""
    all_teams = teams.get_teams()
    apg_lookup = _load_apg_lookup()
    rows = []

    for i, team in enumerate(all_teams):
        team_id = team["id"]
        abbr = team["abbreviation"]
        log.info(f"  [{i+1}/30] Fetching {abbr} roster...")

        try:
            roster = commonteamroster.CommonTeamRoster(
                team_id=team_id, season=SEASON
            )
            df = roster.get_data_frames()[0]

            for _, row in df.iterrows():
                pid = int(row["PLAYER_ID"])
                nba_pos = str(row.get("POSITION") or "")
                height_str = str(row.get("HEIGHT") or "")
                height_in = _parse_height_inches(height_str)
                apg = apg_lookup.get(pid, 0.0)
                positions = _map_positions(nba_pos, height_in, apg)
                primary = positions.split(",")[0].strip() if positions else ""
                jersey = str(row.get("NUM") or "")

                rows.append({
                    "player_id": pid,
                    "player_name": str(row["PLAYER"]),
                    "position": nba_pos,
                    "height": height_str,
                    "apg": apg,
                    "positions": positions,
                    "position_primary": primary,
                    "jersey_number": jersey,
                    "team": abbr,
                })
        except Exception as e:
            log.error(f"    WARN: Failed to fetch {abbr}: {e}")

        time.sleep(0.6)

    result = pd.DataFrame(rows)
    # Deduplicate (player traded mid-season appears on multiple rosters)
    # Keep the most recent team entry (last in list = latest roster update)
    result = result.drop_duplicates(subset=["player_id"], keep="last")
    return result


def main() -> None:
    log.info("Fetching player positions from NBA rosters...")
    df = fetch_all_positions()
    log.info(f"  {len(df)} players with positions")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    log.info(f"  Written -> {OUT_CSV}")

    # Show position distribution
    log.info("\nPosition distribution (primary):")
    log.debug(df["position_primary"].value_counts().to_string())

    # Spot-check notable players
    log.info("\nSpot checks:")
    checks = [
        "Luka Doncic", "Jayson Tatum", "Shai Gilgeous-Alexander",
        "Stephen Curry", "LeBron James", "Anthony Davis",
        "Nikola Jokic", "Jaylen Brown", "Austin Reaves",
    ]
    for name in checks:
        match = df[df["player_name"] == name]
        if not match.empty:
            r = match.iloc[0]
            log.info(f"  {name}: {r['position']} {r['height']} "
                  f"APG={r['apg']} -> {r['positions']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
