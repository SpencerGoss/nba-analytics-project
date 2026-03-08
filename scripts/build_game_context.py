"""
Build dashboard/data/game_context.json from team_game_logs.csv.

For each game in todays_picks.json, computes:
- Back-to-back flag (home/away)
- Rest days since last game
- Current home/away win streak
- Last 10 record (SU)
- Season home/away record
- Situational flags

Run: python scripts/build_game_context.py
"""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEAM_LOGS = PROJECT_ROOT / "data" / "processed" / "team_game_logs.csv"
PICKS_JSON = PROJECT_ROOT / "dashboard" / "data" / "todays_picks.json"
INJURIES_JSON = PROJECT_ROOT / "dashboard" / "data" / "injuries.json"
OUT_JSON = PROJECT_ROOT / "dashboard" / "data" / "game_context.json"

# Thresholds
REST_ADV_DAYS_THRESHOLD = 2  # 2+ extra rest days = significant advantage
HOT_WINS_THRESHOLD = 5       # won 5+ of last 7 -> HOT
COLD_WINS_THRESHOLD = 2      # won 2 or fewer of last 7 -> COLD
L10_HOT_WINS = 7             # kept for backwards-compat situational_flags
L10_COLD_WINS = 3

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_game_logs() -> pd.DataFrame:
    """Load team_game_logs.csv and normalise game_date to date objects."""
    df = pd.read_csv(TEAM_LOGS)
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed").dt.date
    df["is_home"] = df["matchup"].str.contains(r"vs\.", regex=True)
    df["win"] = df["wl"].str.upper() == "W"
    return df


def current_season_logs(df: pd.DataFrame) -> pd.DataFrame:
    """Return only current (202526) season rows, sorted ascending."""
    curr = df[df["season"] == 202526].copy()
    curr = curr.sort_values("game_date", ascending=True)
    return curr


def last_game_date(team_logs: pd.DataFrame, team: str, before: date) -> date | None:
    """Return the most recent game_date for *team* strictly before *before*."""
    mask = (team_logs["team_abbreviation"] == team) & (team_logs["game_date"] < before)
    rows = team_logs[mask]
    if rows.empty:
        return None
    return rows["game_date"].max()


def rest_days(team_logs: pd.DataFrame, team: str, game_date: date) -> int | None:
    """Days since the team last played before game_date. None if no prior game."""
    prev = last_game_date(team_logs, team, game_date)
    if prev is None:
        return None
    return (game_date - prev).days


def is_b2b(team_logs: pd.DataFrame, team: str, game_date: date) -> bool:
    """True if team played yesterday."""
    prev = last_game_date(team_logs, team, game_date)
    if prev is None:
        return False
    return (game_date - prev).days == 1


def home_streak(season_logs: pd.DataFrame, team: str, before: date) -> int:
    """
    Current home win/loss streak going into *before*.
    Positive = consecutive home wins, negative = consecutive home losses.
    Streak resets when the sign flips.
    """
    mask = (
        (season_logs["team_abbreviation"] == team)
        & (season_logs["is_home"])
        & (season_logs["game_date"] < before)
    )
    rows = season_logs[mask].sort_values("game_date", ascending=False)
    if rows.empty:
        return 0
    return _compute_streak(rows["win"].tolist())


def away_streak(season_logs: pd.DataFrame, team: str, before: date) -> int:
    """
    Current road win/loss streak going into *before*.
    Positive = wins, negative = losses.
    """
    mask = (
        (season_logs["team_abbreviation"] == team)
        & (~season_logs["is_home"])
        & (season_logs["game_date"] < before)
    )
    rows = season_logs[mask].sort_values("game_date", ascending=False)
    if rows.empty:
        return 0
    return _compute_streak(rows["win"].tolist())


def _compute_streak(wins_desc: list[bool]) -> int:
    """
    Given a list of win booleans most-recent-first, return the current streak.
    Positive = win streak, negative = loss streak.
    """
    if not wins_desc:
        return 0
    first = wins_desc[0]
    count = 0
    for w in wins_desc:
        if w == first:
            count += 1
        else:
            break
    return count if first else -count


def last10_record(season_logs: pd.DataFrame, team: str, before: date) -> str:
    """Return 'W-L' string for last 10 games (all venues) before *before*."""
    mask = (
        (season_logs["team_abbreviation"] == team)
        & (season_logs["game_date"] < before)
    )
    rows = season_logs[mask].sort_values("game_date", ascending=False).head(10)
    if rows.empty:
        return "0-0"
    wins = int(rows["win"].sum())
    losses = len(rows) - wins
    return f"{wins}-{losses}"


def last7_record(season_logs: pd.DataFrame, team: str, before: date) -> str:
    """Return 'W-L' string for last 7 games (all venues) before *before*."""
    mask = (
        (season_logs["team_abbreviation"] == team)
        & (season_logs["game_date"] < before)
    )
    rows = season_logs[mask].sort_values("game_date", ascending=False).head(7)
    if rows.empty:
        return "0-0"
    wins = int(rows["win"].sum())
    losses = len(rows) - wins
    return f"{wins}-{losses}"


def season_home_record(season_logs: pd.DataFrame, team: str, before: date) -> str:
    """Return 'W-L' string for home games this season before *before*."""
    mask = (
        (season_logs["team_abbreviation"] == team)
        & (season_logs["is_home"])
        & (season_logs["game_date"] < before)
    )
    rows = season_logs[mask]
    if rows.empty:
        return "0-0"
    wins = int(rows["win"].sum())
    losses = len(rows) - wins
    return f"{wins}-{losses}"


def season_away_record(season_logs: pd.DataFrame, team: str, before: date) -> str:
    """Return 'W-L' string for road games this season before *before*."""
    mask = (
        (season_logs["team_abbreviation"] == team)
        & (~season_logs["is_home"])
        & (season_logs["game_date"] < before)
    )
    rows = season_logs[mask]
    if rows.empty:
        return "0-0"
    wins = int(rows["win"].sum())
    losses = len(rows) - wins
    return f"{wins}-{losses}"


def _parse_record(record: str) -> tuple[int, int]:
    """Parse 'W-L' string to (wins, losses) tuple."""
    parts = record.split("-")
    return int(parts[0]), int(parts[1])


def situational_flags(
    home_b2b: bool,
    away_b2b: bool,
    home_rest: int | None,
    away_rest: int | None,
    home_l10: str,
    away_l10: str,
    home_l7: str = "0-0",
    away_l7: str = "0-0",
    home_injuries: list[dict] | None = None,
    away_injuries: list[dict] | None = None,
) -> list[str]:
    """Derive situational flags from pre-computed context fields.

    Flags emitted:
      HOME_B2B / AWAY_B2B          -- played yesterday
      REST_ADV_HOME / REST_ADV_AWAY -- 2+ more rest days than opponent
      HOME_HOT / AWAY_HOT          -- won 5+ of last 7
      HOME_COLD / AWAY_COLD        -- won 2 or fewer of last 7
      INJ_IMPACT_HOME / INJ_IMPACT_AWAY -- key player(s) out (high-impact injury)
    """
    flags: list[str] = []

    if home_b2b:
        flags.append("HOME_B2B")
    if away_b2b:
        flags.append("AWAY_B2B")

    # REST_ADV: team has REST_ADV_DAYS_THRESHOLD+ more days rest than opponent
    if home_rest is not None and away_rest is not None:
        diff = home_rest - away_rest
        if diff >= REST_ADV_DAYS_THRESHOLD:
            flags.append("REST_ADV_HOME")
        elif diff <= -REST_ADV_DAYS_THRESHOLD:
            flags.append("REST_ADV_AWAY")
    elif home_rest is not None and home_rest >= REST_ADV_DAYS_THRESHOLD + 1:
        # Away has unknown rest (first game of season etc.) -- still flag
        flags.append("REST_ADV_HOME")
    elif away_rest is not None and away_rest >= REST_ADV_DAYS_THRESHOLD + 1:
        flags.append("REST_ADV_AWAY")

    # HOT/COLD based on last-7 record
    h7w, _ = _parse_record(home_l7)
    a7w, _ = _parse_record(away_l7)

    if h7w >= HOT_WINS_THRESHOLD:
        flags.append("HOME_HOT")
    elif h7w <= COLD_WINS_THRESHOLD:
        flags.append("HOME_COLD")

    if a7w >= HOT_WINS_THRESHOLD:
        flags.append("AWAY_HOT")
    elif a7w <= COLD_WINS_THRESHOLD:
        flags.append("AWAY_COLD")

    # Injury impact: any high-impact player out
    if home_injuries:
        has_high = any(
            str(inj.get("impact", "")).lower() == "high"
            for inj in home_injuries
        )
        if has_high:
            flags.append("INJ_IMPACT_HOME")

    if away_injuries:
        has_high = any(
            str(inj.get("impact", "")).lower() == "high"
            for inj in away_injuries
        )
        if has_high:
            flags.append("INJ_IMPACT_AWAY")

    return flags


def build_context_summary(
    home: str,
    away: str,
    home_name: str,
    away_name: str,
    hb2b: bool,
    ab2b: bool,
    h_rest: int | None,
    a_rest: int | None,
    h_l10: str,
    a_l10: str,
    flags: list[str],
) -> str:
    """
    Plain-English summary of the situational context for this game.
    Example: "BOS rested (2 days), MIA on back-to-back. BOS won 6 of last 8."
    """
    parts: list[str] = []

    # Rest / B2B situation
    home_display = home_name or home
    away_display = away_name or away

    if hb2b:
        parts.append(f"{home_display} on back-to-back")
    elif h_rest is not None:
        parts.append(f"{home_display} rested ({h_rest}d)")

    if ab2b:
        parts.append(f"{away_display} on back-to-back")
    elif a_rest is not None:
        parts.append(f"{away_display} rested ({a_rest}d)")

    # Form summary
    hw, hl = _parse_record(h_l10)
    aw, al = _parse_record(a_l10)
    n_h = hw + hl
    n_a = aw + al

    if n_h > 0:
        parts.append(f"{home_display} {hw}-{hl} in last {n_h}")
    if n_a > 0:
        parts.append(f"{away_display} {aw}-{al} in last {n_a}")

    # Injury callout
    if "INJ_IMPACT_HOME" in flags:
        parts.append(f"key {home_display} player(s) out")
    if "INJ_IMPACT_AWAY" in flags:
        parts.append(f"key {away_display} player(s) out")

    return ". ".join(parts) + "." if parts else "No notable situational factors."


def _load_injuries_index(path: Path) -> dict[tuple[str, str, str], dict]:
    """
    Load injuries.json and return index by (home_team, away_team, game_date).
    Returns empty dict if file is absent or malformed.
    """
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            return {}
        index: dict[tuple[str, str, str], dict] = {}
        for entry in data:
            key = (
                entry.get("home_team", ""),
                entry.get("away_team", ""),
                entry.get("game_date", ""),
            )
            index[key] = entry
        return index
    except (json.JSONDecodeError, OSError):
        return {}


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_context_for_game(
    pick: dict,
    all_logs: pd.DataFrame,
    season_logs: pd.DataFrame,
    injuries_index: dict[tuple[str, str, str], dict] | None = None,
) -> dict:
    """Compute all context fields for a single game from todays_picks.json."""
    game_date_str: str = pick["game_date"]
    game_date_parsed: date = pd.to_datetime(game_date_str, format="mixed").date()
    home: str = pick["home_team"]
    away: str = pick["away_team"]
    home_name: str = pick.get("home_team_name", home)
    away_name: str = pick.get("away_team_name", away)

    hb2b = is_b2b(all_logs, home, game_date_parsed)
    ab2b = is_b2b(all_logs, away, game_date_parsed)
    h_rest = rest_days(all_logs, home, game_date_parsed)
    a_rest = rest_days(all_logs, away, game_date_parsed)
    h_streak = home_streak(season_logs, home, game_date_parsed)
    a_streak = away_streak(season_logs, away, game_date_parsed)
    h_l10 = last10_record(season_logs, home, game_date_parsed)
    a_l10 = last10_record(season_logs, away, game_date_parsed)
    h_l7 = last7_record(season_logs, home, game_date_parsed)
    a_l7 = last7_record(season_logs, away, game_date_parsed)
    h_home_rec = season_home_record(season_logs, home, game_date_parsed)
    a_away_rec = season_away_record(season_logs, away, game_date_parsed)

    # Injury data for this game (from injuries.json if available)
    home_injuries: list[dict] | None = None
    away_injuries: list[dict] | None = None
    if injuries_index is not None:
        inj_entry = injuries_index.get((home, away, game_date_str))
        if inj_entry:
            home_injuries = inj_entry.get("home_injuries") or []
            away_injuries = inj_entry.get("away_injuries") or []

    flags = situational_flags(
        hb2b, ab2b, h_rest, a_rest, h_l10, a_l10,
        home_l7=h_l7, away_l7=a_l7,
        home_injuries=home_injuries,
        away_injuries=away_injuries,
    )

    summary = build_context_summary(
        home, away, home_name, away_name,
        hb2b, ab2b, h_rest, a_rest, h_l10, a_l10, flags,
    )

    return {
        "home_team": home,
        "away_team": away,
        "game_date": game_date_str,
        "home_b2b": hb2b,
        "away_b2b": ab2b,
        "home_rest_days": h_rest,
        "away_rest_days": a_rest,
        "home_last10": h_l10,
        "away_last10": a_l10,
        "home_last7": h_l7,
        "away_last7": a_l7,
        "home_streak": h_streak,
        "away_streak": a_streak,
        "home_season_home_record": h_home_rec,
        "away_season_away_record": a_away_rec,
        "situational_flags": flags,
        "context_summary": summary,
    }


def build_game_context(
    picks_path: Path = PICKS_JSON,
    logs_path: Path = TEAM_LOGS,
    out_path: Path = OUT_JSON,
) -> list[dict]:
    """Main entry point. Returns the list of context dicts (also writes JSON)."""
    if not picks_path.exists():
        print(f"WARN: picks file not found: {picks_path} -- skipping game context build")
        return []
    if not logs_path.exists():
        print(f"WARN: team_game_logs not found: {logs_path} -- skipping game context build")
        return []

    with open(picks_path, encoding="utf-8") as f:
        picks = json.load(f)

    if not picks:
        print("No picks found in todays_picks.json -- writing empty game_context.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("[]", encoding="utf-8")
        return []

    print(f"Loading team game logs from {logs_path} ...")
    all_logs = load_game_logs()
    season_logs = current_season_logs(all_logs)
    print(f"  All-time rows: {len(all_logs):,}  |  202526 rows: {len(season_logs):,}")

    injuries_index = _load_injuries_index(INJURIES_JSON)
    if injuries_index:
        print(f"  Loaded injury data for {len(injuries_index)} games")
    else:
        print("  No injuries.json data available -- INJ_IMPACT flags will be skipped")

    results = []
    for pick in picks:
        ctx = build_context_for_game(pick, all_logs, season_logs, injuries_index)
        results.append(ctx)
        print(
            f"  {ctx['away_team']} @ {ctx['home_team']} ({ctx['game_date']}) -> "
            f"flags={ctx['situational_flags']}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Wrote {len(results)} game contexts to {out_path}")
    return results


if __name__ == "__main__":
    build_game_context()
