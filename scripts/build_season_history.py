"""
Build dashboard/data/season_history.json

Reads data/processed/team_game_logs.csv and produces per-season standings
and game results for the last 5 seasons (202021 through 202425).

Output shape:
{
  "seasons": ["2024-25", ...],
  "data": {
    "2024-25": {
      "standings": [{"abbr": "BOS", "name": "...", "w": 61, "l": 21, "pct": 0.744}, ...],
      "games": [{"date": "2024-10-22", "home": "BOS", "away": "NYK",
                 "home_pts": 108, "away_pts": 99, "winner": "BOS"}, ...]
    }
  }
}

Games are deduplicated to one row per game (home team row, identified by "vs." in matchup).
Standings are sorted by win% descending. Games are sorted by date descending.
"""

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "team_game_logs.csv"
OUTPUT_PATH = PROJECT_ROOT / "dashboard" / "data" / "season_history.json"

SEASON_CODES = [202021, 202122, 202223, 202324, 202425]
SEASON_LABELS = {
    202021: "2020-21",
    202122: "2021-22",
    202223: "2022-23",
    202324: "2023-24",
    202425: "2024-25",
}

TEAM_NAMES = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
    # Historical teams
    "NOH": "New Orleans Hornets",
    "NOK": "New Orleans/Oklahoma City Hornets",
    "NJN": "New Jersey Nets",
    "SEA": "Seattle SuperSonics",
    "VAN": "Vancouver Grizzlies",
    "WSB": "Washington Bullets",
}


def season_label(code: int) -> str:
    return SEASON_LABELS.get(code, str(code))


def load_logs() -> pd.DataFrame:
    if not DATA_PATH.exists():
        print(f"ERROR: Data file not found: {DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH, dtype={"season": int, "season_id": str})
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")
    return df


def filter_seasons(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["season"].isin(SEASON_CODES)].copy()


def build_standings(season_df: pd.DataFrame) -> list[dict]:
    """Calculate W/L record per team for one season."""
    grp = (
        season_df.groupby(["team_abbreviation", "team_name"])["wl"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )
    # Ensure both W and L columns exist even if a team went undefeated or winless
    for col in ("W", "L"):
        if col not in grp.columns:
            grp[col] = 0

    grp = grp.rename(columns={"team_abbreviation": "abbr", "team_name": "name"})
    grp["w"] = grp["W"].astype(int)
    grp["l"] = grp["L"].astype(int)
    grp["games"] = grp["w"] + grp["l"]
    grp["pct"] = grp.apply(
        lambda r: round(r["w"] / r["games"], 3) if r["games"] > 0 else 0.0, axis=1
    )

    standings = (
        grp[["abbr", "name", "w", "l", "pct"]]
        .sort_values("pct", ascending=False)
        .reset_index(drop=True)
    )
    return standings.to_dict(orient="records")


def build_games(season_df: pd.DataFrame) -> list[dict]:
    """Return one game record per game_id using home-team rows (matchup contains 'vs.')."""
    home_rows = season_df[season_df["matchup"].str.contains(r"\bvs\.", na=False)].copy()

    records = []
    for _, row in home_rows.iterrows():
        # matchup format: "BOS vs. NYK"
        parts = row["matchup"].split(" vs. ")
        if len(parts) != 2:
            continue
        home_abbr = parts[0].strip()
        away_abbr = parts[1].strip()
        home_name = TEAM_NAMES.get(home_abbr, home_abbr)
        away_name = TEAM_NAMES.get(away_abbr, away_abbr)

        records.append(
            {
                "date": row["game_date"].strftime("%Y-%m-%d"),
                "home": home_name,
                "away": away_name,
                "home_abbr": home_abbr,
                "away_abbr": away_abbr,
                "home_pts": int(row["pts"]) if pd.notna(row["pts"]) else None,
                "away_pts": None,  # filled below from away row if available
                "winner": home_name if row["wl"] == "W" else away_name,
                "_game_id": row["game_id"],
                "_home_pts": int(row["pts"]) if pd.notna(row["pts"]) else None,
                "_home_wl": row["wl"],
            }
        )

    # Build away pts lookup from away rows (matchup contains "@")
    away_rows = season_df[season_df["matchup"].str.contains(r"\s@\s", na=False)].copy()
    away_pts_map: dict = {}
    for _, row in away_rows.iterrows():
        away_pts_map[row["game_id"]] = int(row["pts"]) if pd.notna(row["pts"]) else None

    games = []
    for rec in records:
        gid = rec.pop("_game_id")
        rec.pop("_home_pts")
        rec.pop("_home_wl")
        rec["away_pts"] = away_pts_map.get(gid)
        games.append(rec)

    # Sort by date descending
    games.sort(key=lambda g: g["date"], reverse=True)
    return games


def build_output(df: pd.DataFrame) -> dict:
    seasons_ordered = [season_label(c) for c in SEASON_CODES]  # oldest to newest
    seasons_ordered.reverse()  # newest first

    data: dict = {}
    for code in SEASON_CODES:
        label = season_label(code)
        season_df = df[df["season"] == code]
        if season_df.empty:
            print(f"  WARNING: No rows found for season {code} ({label})")
            continue
        standings = build_standings(season_df)
        games = build_games(season_df)
        data[label] = {"standings": standings, "games": games}
        print(f"  {label}: {len(standings)} teams, {len(games)} games")

    return {"seasons": seasons_ordered, "data": data}


def main() -> None:
    print("Building season_history.json ...")
    df = load_logs()
    print(f"Loaded {len(df):,} rows from team_game_logs.csv")

    filtered = filter_seasons(df)
    print(f"Filtered to {len(filtered):,} rows across last 5 seasons")

    output = build_output(filtered)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, separators=(",", ":"))

    size_kb = OUTPUT_PATH.stat().st_size / 1024
    total_games = sum(len(v["games"]) for v in output["data"].values())
    print(f"Wrote {OUTPUT_PATH} ({size_kb:.1f} KB)")
    print(f"Seasons included: {len(output['data'])}")
    print(f"Total game records: {total_games:,}")


if __name__ == "__main__":
    main()
