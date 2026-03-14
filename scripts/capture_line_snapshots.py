"""capture_line_snapshots.py -- Log Pinnacle spread snapshots for line movement analysis.

Captures current NBA game spreads at regular intervals throughout the day.
Designed to be run via Windows Task Scheduler multiple times on game days:
  - 10:00 AM ET (opening / early)
  - 1:00 PM ET
  - 4:00 PM ET
  - 6:00 PM ET (pre-tip)

Each run appends rows to data/odds/line_snapshots.csv. Downstream analysis
compares opening vs closing to detect:
  - Reverse line movement (RLM) = strongest sharp-money signal
  - Steam moves (sudden large shifts)
  - Opening-to-closing drift direction

Usage:
    python scripts/capture_line_snapshots.py

No API key required -- uses Pinnacle guest API.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SNAPSHOTS_CSV = PROJECT_ROOT / "data" / "odds" / "line_snapshots.csv"

SNAPSHOT_COLUMNS = [
    "snapshot_ts",     # UTC timestamp of this capture
    "game_date",       # date of the game (YYYY-MM-DD)
    "home_team",       # 3-letter abbreviation
    "away_team",       # 3-letter abbreviation
    "spread",          # home spread (negative = home favored)
    "home_moneyline",  # American odds
    "away_moneyline",  # American odds
    "total",           # over/under line
]


def capture_snapshot() -> pd.DataFrame:
    """Fetch current Pinnacle lines and return as a DataFrame with snapshot timestamp."""
    from scripts.fetch_odds import fetch_game_lines

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines_df = fetch_game_lines()
    if lines_df.empty:
        log.warning("No game lines returned -- nothing to snapshot.")
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)

    snapshot = pd.DataFrame({
        "snapshot_ts": now_utc,
        "game_date": lines_df["date"],
        "home_team": lines_df["home_team"],
        "away_team": lines_df["away_team"],
        "spread": lines_df.get("spread"),
        "home_moneyline": lines_df.get("home_moneyline"),
        "away_moneyline": lines_df.get("away_moneyline"),
        "total": lines_df.get("total"),
    })

    return snapshot


def append_snapshot(snapshot: pd.DataFrame) -> int:
    """Append snapshot rows to the CSV file. Returns number of rows written."""
    if snapshot.empty:
        return 0

    write_header = not SNAPSHOTS_CSV.exists()
    snapshot.to_csv(SNAPSHOTS_CSV, mode="a", index=False, header=write_header)
    return len(snapshot)


def compute_line_movement(game_date: str | None = None) -> pd.DataFrame:
    """Compute opening-to-latest spread movement for each game.

    Returns DataFrame with columns:
        game_date, home_team, away_team, opening_spread, latest_spread,
        spread_move, n_snapshots, is_reverse_line_movement

    Reverse line movement (RLM): spread moves AWAY from the side receiving
    majority of public bets. Approximated here as: spread moved toward the
    team that opened as underdog (sharps often back underdogs).
    """
    if not SNAPSHOTS_CSV.exists():
        log.warning("No snapshots file found -- run capture_snapshot first.")
        return pd.DataFrame()

    df = pd.read_csv(SNAPSHOTS_CSV, parse_dates=["snapshot_ts"])

    if game_date:
        df = df[df["game_date"] == game_date]

    if df.empty:
        return pd.DataFrame()

    # Sort by time to get opening and closing
    df = df.sort_values("snapshot_ts")

    results = []
    for (gd, home, away), group in df.groupby(["game_date", "home_team", "away_team"]):
        valid_spreads = group.dropna(subset=["spread"])
        if valid_spreads.empty:
            continue

        opening = valid_spreads.iloc[0]["spread"]
        latest = valid_spreads.iloc[-1]["spread"]
        move = latest - opening

        # RLM heuristic: if home opened as underdog (spread > 0) but spread
        # moved more negative (toward home), that's reverse movement.
        # Similarly, if home opened favored (spread < 0) and spread moved
        # more positive (toward away), that's also RLM.
        is_rlm = False
        if abs(move) >= 0.5:  # minimum half-point move to flag
            if opening > 0 and move < 0:
                is_rlm = True  # home was dog, line moved toward home
            elif opening < 0 and move > 0:
                is_rlm = True  # home was fav, line moved toward away

        results.append({
            "game_date": gd,
            "home_team": home,
            "away_team": away,
            "opening_spread": opening,
            "latest_spread": latest,
            "spread_move": round(move, 1),
            "n_snapshots": len(valid_spreads),
            "is_reverse_line_movement": is_rlm,
        })

    return pd.DataFrame(results)


def main():
    log.info("Capturing Pinnacle line snapshot...")
    snapshot = capture_snapshot()
    n_rows = append_snapshot(snapshot)
    log.info(f"Wrote {n_rows} rows to {SNAPSHOTS_CSV}")

    if n_rows > 0:
        movement = compute_line_movement()
        rlm_games = movement[movement["is_reverse_line_movement"]]
        if not rlm_games.empty:
            log.info(f"Reverse line movement detected in {len(rlm_games)} game(s):")
            for _, row in rlm_games.iterrows():
                log.info(
                    f"  {row['away_team']} @ {row['home_team']}: "
                    f"{row['opening_spread']} -> {row['latest_spread']} "
                    f"(move: {row['spread_move']:+.1f})"
                )


if __name__ == "__main__":
    main()
