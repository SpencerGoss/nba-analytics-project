"""
build_game_detail.py -- produce dashboard/data/game_detail.json

For each of today's games (from todays_picks.json), compiles:
  - Full team records and standings
  - H2H history (last 5 meetings from team_game_logs.csv)
  - Key matchup factors from game_context.json (rest, B2B, streaks, injury impact)
  - Prediction details (confidence, spread, projected margin)
  - Top prediction factors/reasons (why the model picks this team)

Output format:
{
  "games": [
    {
      "game_id": "HOME_AWAY_DATE",
      "home": {...},
      "away": {...},
      "prediction": {...},
      "context": {"factors": [...]},
      "h2h": {...},
      "injuries": {...}
    }
  ]
}

Run: python scripts/build_game_detail.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PICKS_JSON = PROJECT_ROOT / "dashboard" / "data" / "todays_picks.json"
CONTEXT_JSON = PROJECT_ROOT / "dashboard" / "data" / "game_context.json"
STANDINGS_JSON = PROJECT_ROOT / "dashboard" / "data" / "standings.json"
INJURIES_JSON = PROJECT_ROOT / "dashboard" / "data" / "injuries.json"
H2H_JSON = PROJECT_ROOT / "dashboard" / "data" / "head_to_head.json"
TEAM_LOGS_CSV = PROJECT_ROOT / "data" / "processed" / "team_game_logs.csv"
OUT_JSON = PROJECT_ROOT / "dashboard" / "data" / "game_detail.json"

LAST_N_H2H = 5


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> list | dict:
    """Load a JSON file, returning empty list/dict on failure."""
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return []


def _load_standings_index(standings: dict) -> dict[str, dict]:
    """Build {team_abbr: standings_dict} from standings.json."""
    index: dict[str, dict] = {}
    for conf in ("east", "west"):
        for entry in standings.get(conf, []):
            team = entry.get("team", "")
            if team:
                index[team] = entry
    return index


def _load_context_index(context: list) -> dict[tuple[str, str], dict]:
    """Build {(home, away): context_dict} from game_context.json."""
    index: dict[tuple[str, str], dict] = {}
    for entry in context:
        home = entry.get("home_team", "")
        away = entry.get("away_team", "")
        if home and away:
            index[(home, away)] = entry
    return index


def _load_injuries_index(injuries: list) -> dict[tuple[str, str], dict]:
    """Build {(home, away): injury_dict} from injuries.json."""
    index: dict[tuple[str, str], dict] = {}
    for entry in injuries:
        home = entry.get("home_team", "")
        away = entry.get("away_team", "")
        if home and away:
            index[(home, away)] = entry
    return index


def _load_h2h_index(h2h: list) -> dict[tuple[str, str], dict]:
    """Build {(home, away): h2h_dict} from head_to_head.json."""
    index: dict[tuple[str, str], dict] = {}
    for entry in h2h:
        home = entry.get("home_team", "")
        away = entry.get("away_team", "")
        if home and away:
            index[(home, away)] = entry
    return index


# ---------------------------------------------------------------------------
# H2H from CSV (fallback when h2h.json is missing)
# ---------------------------------------------------------------------------

def _build_h2h_from_logs(
    home: str, away: str, n: int = LAST_N_H2H,
) -> dict:
    """Build H2H data directly from team_game_logs.csv (last n meetings)."""
    if not TEAM_LOGS_CSV.exists():
        return {"series_record": "No data", "avg_total": None, "meetings": []}

    try:
        df = pd.read_csv(TEAM_LOGS_CSV)
        df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")
        df["is_home"] = df["matchup"].str.contains(" vs. ", regex=False)
    except Exception:
        return {"series_record": "No data", "avg_total": None, "meetings": []}

    # Build per-game view
    home_rows = df[df["is_home"]].copy()
    away_rows = df[~df["is_home"]].copy()

    home_rows = home_rows.rename(columns={
        "team_abbreviation": "h_team", "pts": "h_score", "wl": "h_wl",
    })[["game_id", "game_date", "h_team", "h_score", "h_wl"]]

    away_rows = away_rows.rename(columns={
        "team_abbreviation": "a_team", "pts": "a_score",
    })[["game_id", "a_team", "a_score"]]

    games = home_rows.merge(away_rows, on="game_id", how="inner")

    mask = (
        ((games["h_team"] == home) & (games["a_team"] == away))
        | ((games["h_team"] == away) & (games["a_team"] == home))
    )
    meetings = games[mask].sort_values("game_date", ascending=False).head(n)

    if meetings.empty:
        return {"series_record": "No historical meetings", "avg_total": None, "meetings": []}

    meeting_list = []
    home_wins = 0
    away_wins = 0
    for _, row in meetings.iterrows():
        winner = row["h_team"] if row["h_wl"] == "W" else row["a_team"]
        if winner == home:
            home_wins += 1
        else:
            away_wins += 1
        meeting_list.append({
            "date": row["game_date"].strftime("%Y-%m-%d"),
            "home_team": row["h_team"],
            "home_score": int(row["h_score"]),
            "away_score": int(row["a_score"]),
            "winner": winner,
            "margin": abs(int(row["h_score"]) - int(row["a_score"])),
        })

    total_games = len(meetings)
    if home_wins > away_wins:
        record = f"{home} leads {home_wins}-{away_wins} (last {total_games})"
    elif away_wins > home_wins:
        record = f"{away} leads {away_wins}-{home_wins} (last {total_games})"
    else:
        record = f"Tied {home_wins}-{away_wins} (last {total_games})"

    totals = (meetings["h_score"] + meetings["a_score"]).dropna()
    avg_total = round(float(totals.mean()), 1) if not totals.empty else None

    return {
        "series_record": record,
        "avg_total": avg_total,
        "meetings": meeting_list,
    }


# ---------------------------------------------------------------------------
# Factor generation
# ---------------------------------------------------------------------------

def _generate_factors(
    pick: dict,
    ctx: dict | None,
    home_standings: dict | None,
    away_standings: dict | None,
) -> list[str]:
    """Generate human-readable prediction factors from context and standings."""
    factors: list[str] = []
    home = pick.get("home_team", "")
    away = pick.get("away_team", "")
    winner = pick.get("predicted_winner", "")
    confidence = pick.get("model_confidence", 0)

    # Confidence level
    if confidence >= 80:
        factors.append(f"Strong model confidence ({confidence}%) favoring {winner}")
    elif confidence >= 65:
        factors.append(f"Moderate model confidence ({confidence}%) favoring {winner}")
    elif confidence > 50:
        factors.append(f"Slight edge ({confidence}%) favoring {winner}")

    # Situational flags from context
    if ctx:
        flags = ctx.get("situational_flags", [])
        if "HOME_B2B" in flags:
            factors.append(f"{home} on back-to-back (fatigue risk)")
        if "AWAY_B2B" in flags:
            factors.append(f"{away} on back-to-back (fatigue risk)")
        if "REST_ADV_HOME" in flags:
            factors.append(f"{home} has rest advantage")
        if "REST_ADV_AWAY" in flags:
            factors.append(f"{away} has rest advantage")
        if "HOME_HOT" in flags:
            factors.append(f"{home} is hot (5+ of last 7)")
        if "AWAY_HOT" in flags:
            factors.append(f"{away} is hot (5+ of last 7)")
        if "HOME_COLD" in flags:
            factors.append(f"{home} is cold (2 or fewer of last 7)")
        if "AWAY_COLD" in flags:
            factors.append(f"{away} is cold (2 or fewer of last 7)")
        if "INJ_IMPACT_HOME" in flags:
            factors.append(f"{home} missing key player(s)")
        if "INJ_IMPACT_AWAY" in flags:
            factors.append(f"{away} missing key player(s)")

        # Streak info
        h_streak = ctx.get("home_streak", 0)
        a_streak = ctx.get("away_streak", 0)
        if h_streak >= 4:
            factors.append(f"{home} on {h_streak}-game home win streak")
        elif h_streak <= -4:
            factors.append(f"{home} on {abs(h_streak)}-game home losing streak")
        if a_streak >= 4:
            factors.append(f"{away} on {a_streak}-game road win streak")
        elif a_streak <= -4:
            factors.append(f"{away} on {abs(a_streak)}-game road losing streak")

    # Standings-based factors
    if home_standings and away_standings:
        h_pct = home_standings.get("win_pct", 0)
        a_pct = away_standings.get("win_pct", 0)
        diff = abs(h_pct - a_pct)
        if diff >= 0.200:
            better = home if h_pct > a_pct else away
            worse = away if h_pct > a_pct else home
            factors.append(
                f"Significant record gap: {better} ({round(max(h_pct, a_pct) * 100)}%) "
                f"vs {worse} ({round(min(h_pct, a_pct) * 100)}%)"
            )

    # Projected margin
    margin = pick.get("projected_margin")
    if margin is not None:
        if abs(margin) >= 8:
            factors.append(f"Model projects decisive margin ({margin:+.1f} pts)")
        elif abs(margin) <= 3:
            factors.append(f"Model projects a tight game ({margin:+.1f} pts)")

    # Value bet flag
    if pick.get("value_bet"):
        edge = pick.get("edge_pct")
        if edge is not None:
            factors.append(f"Value bet detected (edge: {edge:.1%})")

    return factors


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_game_detail() -> dict:
    """Build game detail JSON for all of today's games."""
    picks = _load_json(PICKS_JSON)
    if not picks:
        print("No picks found in todays_picks.json -> writing empty game_detail.json")
        result = {"games": []}
        _write_output(result)
        return result

    # Load supporting data
    standings_raw = _load_json(STANDINGS_JSON)
    standings_idx = _load_standings_index(standings_raw) if isinstance(standings_raw, dict) else {}

    context_raw = _load_json(CONTEXT_JSON)
    context_idx = _load_context_index(context_raw) if isinstance(context_raw, list) else {}

    injuries_raw = _load_json(INJURIES_JSON)
    injuries_idx = _load_injuries_index(injuries_raw) if isinstance(injuries_raw, list) else {}

    h2h_raw = _load_json(H2H_JSON)
    h2h_idx = _load_h2h_index(h2h_raw) if isinstance(h2h_raw, list) else {}

    print(f"Loaded: {len(picks)} picks, {len(standings_idx)} standings, "
          f"{len(context_idx)} contexts, {len(injuries_idx)} injury reports, "
          f"{len(h2h_idx)} h2h records")

    games = []
    for pick in picks:
        home = pick.get("home_team", "")
        away = pick.get("away_team", "")
        game_date = pick.get("game_date", "")
        game_id = f"{home}_{away}_{game_date}"

        # Team info from standings
        home_standing = standings_idx.get(home, {})
        away_standing = standings_idx.get(away, {})

        home_info = {
            "team": home,
            "team_name": pick.get("home_team_name", home),
            "record": f"{home_standing.get('w', '?')}-{home_standing.get('l', '?')}",
            "win_pct": home_standing.get("win_pct"),
            "home_record": home_standing.get("home_record"),
            "conference": home_standing.get("conference"),
            "rank": home_standing.get("rank"),
        }
        away_info = {
            "team": away,
            "team_name": pick.get("away_team_name", away),
            "record": f"{away_standing.get('w', '?')}-{away_standing.get('l', '?')}",
            "win_pct": away_standing.get("win_pct"),
            "away_record": away_standing.get("away_record"),
            "conference": away_standing.get("conference"),
            "rank": away_standing.get("rank"),
        }

        # Prediction details
        prediction = {
            "predicted_winner": pick.get("predicted_winner"),
            "home_win_prob": pick.get("home_win_prob"),
            "away_win_prob": pick.get("away_win_prob"),
            "confidence_tier": pick.get("confidence_tier"),
            "model_confidence": pick.get("model_confidence"),
            "spread": pick.get("spread"),
            "projected_margin": pick.get("projected_margin"),
            "ats_pick": pick.get("ats_pick"),
            "value_bet": pick.get("value_bet", False),
            "edge_pct": pick.get("edge_pct"),
            "kelly_fraction": pick.get("kelly_fraction"),
        }

        # Context and factors
        ctx = context_idx.get((home, away))
        factors = _generate_factors(pick, ctx, home_standing, away_standing)

        context_block = {
            "factors": factors,
            "home_b2b": ctx.get("home_b2b") if ctx else None,
            "away_b2b": ctx.get("away_b2b") if ctx else None,
            "home_rest_days": ctx.get("home_rest_days") if ctx else None,
            "away_rest_days": ctx.get("away_rest_days") if ctx else None,
            "home_last10": ctx.get("home_last10") if ctx else None,
            "away_last10": ctx.get("away_last10") if ctx else None,
            "home_streak": ctx.get("home_streak") if ctx else None,
            "away_streak": ctx.get("away_streak") if ctx else None,
            "situational_flags": ctx.get("situational_flags", []) if ctx else [],
            "context_summary": ctx.get("context_summary", "") if ctx else "",
        }

        # H2H: prefer pre-built h2h.json, fall back to CSV
        h2h_entry = h2h_idx.get((home, away))
        if h2h_entry:
            # Trim to last 5 meetings
            meetings = h2h_entry.get("meetings", [])[:LAST_N_H2H]
            h2h_block = {
                "series_record": h2h_entry.get("series_record", ""),
                "avg_total": h2h_entry.get("avg_total"),
                "meetings": meetings,
            }
        else:
            h2h_block = _build_h2h_from_logs(home, away, LAST_N_H2H)

        # Injuries
        inj_entry = injuries_idx.get((home, away), {})
        injuries_block = {
            "home_injuries": inj_entry.get("home_injuries", []),
            "away_injuries": inj_entry.get("away_injuries", []),
        }

        games.append({
            "game_id": game_id,
            "game_date": game_date,
            "home": home_info,
            "away": away_info,
            "prediction": prediction,
            "context": context_block,
            "h2h": h2h_block,
            "injuries": injuries_block,
        })

        print(f"  {away} @ {home} ({game_date}) -> {len(factors)} factors")

    result = {"games": games}
    _write_output(result)
    print(f"Written -> {OUT_JSON} ({len(games)} games)")
    return result


def _write_output(data: dict) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(data, fh, separators=(",", ":"), default=str)


if __name__ == "__main__":
    build_game_detail()
