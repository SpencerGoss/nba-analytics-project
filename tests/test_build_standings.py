"""
Tests for scripts/build_standings.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_logs(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")
    df["is_home"] = df["matchup"].str.contains(r" vs\. ", regex=True)
    df["win"] = df["wl"].str.upper() == "W"
    return df.sort_values(["team_abbreviation", "game_date"]).reset_index(drop=True)


SAMPLE_LOGS = _make_logs([
    # BOS: 3W home, 1L away
    {"team_abbreviation": "BOS", "matchup": "BOS vs. MIA", "wl": "W", "game_date": "2025-11-01", "season": 202526},
    {"team_abbreviation": "BOS", "matchup": "BOS vs. NYK", "wl": "W", "game_date": "2025-11-03", "season": 202526},
    {"team_abbreviation": "BOS", "matchup": "BOS vs. PHI", "wl": "W", "game_date": "2025-11-05", "season": 202526},
    {"team_abbreviation": "BOS", "matchup": "BOS @ TOR", "wl": "L", "game_date": "2025-11-07", "season": 202526},
    # BKN: 1W, 2L
    {"team_abbreviation": "BKN", "matchup": "BKN vs. CLE", "wl": "W", "game_date": "2025-11-01", "season": 202526},
    {"team_abbreviation": "BKN", "matchup": "BKN @ BOS", "wl": "L", "game_date": "2025-11-03", "season": 202526},
    {"team_abbreviation": "BKN", "matchup": "BKN @ CHI", "wl": "L", "game_date": "2025-11-05", "season": 202526},
    # MIA: 2W, 1L (Southeast)
    {"team_abbreviation": "MIA", "matchup": "MIA @ BOS", "wl": "L", "game_date": "2025-11-01", "season": 202526},
    {"team_abbreviation": "MIA", "matchup": "MIA vs. ATL", "wl": "W", "game_date": "2025-11-03", "season": 202526},
    {"team_abbreviation": "MIA", "matchup": "MIA vs. ORL", "wl": "W", "game_date": "2025-11-05", "season": 202526},
])

TEAM_NAMES = {
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "MIA": "Miami Heat",
}


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

def test_import():
    from scripts.build_standings import (
        compute_team_record,
        build_conference_standings,
        _current_streak,
        _games_behind,
    )
    assert callable(compute_team_record)
    assert callable(build_conference_standings)


# ---------------------------------------------------------------------------
# _current_streak
# ---------------------------------------------------------------------------

def test_streak_win_run():
    from scripts.build_standings import _current_streak
    series = pd.Series(["L", "W", "W", "W"])
    assert _current_streak(series) == "+3"


def test_streak_loss_run():
    from scripts.build_standings import _current_streak
    series = pd.Series(["W", "W", "L", "L"])
    assert _current_streak(series) == "-2"


def test_streak_empty():
    from scripts.build_standings import _current_streak
    assert _current_streak(pd.Series([], dtype=str)) == "+0"


def test_streak_single_loss():
    from scripts.build_standings import _current_streak
    assert _current_streak(pd.Series(["L"])) == "-1"


# ---------------------------------------------------------------------------
# _games_behind
# ---------------------------------------------------------------------------

def test_games_behind_leader():
    from scripts.build_standings import _games_behind
    # Leader 10-2, team 8-4 -> GB = ((10-8)+(4-2))/2 = 2.0
    assert _games_behind(10, 2, 8, 4) == 2.0


def test_games_behind_same():
    from scripts.build_standings import _games_behind
    assert _games_behind(10, 5, 10, 5) == 0.0


def test_games_behind_half():
    from scripts.build_standings import _games_behind
    # 10-5 leader vs 10-6 team -> GB = 0.5
    assert _games_behind(10, 5, 10, 6) == 0.5


# ---------------------------------------------------------------------------
# compute_team_record
# ---------------------------------------------------------------------------

def test_compute_team_record_wins_losses():
    from scripts.build_standings import compute_team_record
    bos_logs = SAMPLE_LOGS[SAMPLE_LOGS["team_abbreviation"] == "BOS"].copy()
    rec = compute_team_record(bos_logs)
    assert rec["w"] == 3
    assert rec["l"] == 1
    assert rec["win_pct"] == 0.75


def test_compute_team_record_home_away():
    from scripts.build_standings import compute_team_record
    bos_logs = SAMPLE_LOGS[SAMPLE_LOGS["team_abbreviation"] == "BOS"].copy()
    rec = compute_team_record(bos_logs)
    assert rec["home_record"] == "3-0"
    assert rec["away_record"] == "0-1"


def test_compute_team_record_streak():
    from scripts.build_standings import compute_team_record
    bos_logs = SAMPLE_LOGS[SAMPLE_LOGS["team_abbreviation"] == "BOS"].copy()
    rec = compute_team_record(bos_logs)
    # Last game was L (Nov 7 away) -> streak -1
    assert rec["streak"] == "-1"


def test_compute_team_record_last10():
    from scripts.build_standings import compute_team_record
    bos_logs = SAMPLE_LOGS[SAMPLE_LOGS["team_abbreviation"] == "BOS"].copy()
    rec = compute_team_record(bos_logs)
    assert rec["last10"] == "3-1"


# ---------------------------------------------------------------------------
# build_conference_standings
# ---------------------------------------------------------------------------

def test_build_conference_standings_rank_order():
    """BOS (3-1) should rank above BKN (1-2) in East Atlantic."""
    from scripts.build_standings import build_conference_standings, EAST
    rows = build_conference_standings(SAMPLE_LOGS, TEAM_NAMES, "East", EAST)
    # Find BOS and BKN
    bos_row = next(r for r in rows if r["team"] == "BOS")
    bkn_row = next(r for r in rows if r["team"] == "BKN")
    assert bos_row["rank"] < bkn_row["rank"]


def test_build_conference_standings_leader_gb_zero():
    from scripts.build_standings import build_conference_standings, EAST
    rows = build_conference_standings(SAMPLE_LOGS, TEAM_NAMES, "East", EAST)
    leader = rows[0]
    assert leader["games_behind"] == 0.0


def test_build_conference_standings_all_30_teams():
    """All 15 East + 15 West teams should appear even with no game data."""
    from scripts.build_standings import build_conference_standings, EAST, WEST
    empty_logs = SAMPLE_LOGS.iloc[0:0].copy()  # empty DataFrame, same schema
    east_rows = build_conference_standings(empty_logs, {}, "East", EAST)
    west_rows = build_conference_standings(empty_logs, {}, "West", WEST)
    assert len(east_rows) == 15
    assert len(west_rows) == 15


def test_build_conference_standings_schema():
    """Each row must contain required keys."""
    from scripts.build_standings import build_conference_standings, EAST
    rows = build_conference_standings(SAMPLE_LOGS, TEAM_NAMES, "East", EAST)
    required_keys = {
        "rank", "team", "team_name", "conference", "division",
        "w", "l", "win_pct", "home_record", "away_record",
        "last10", "streak", "games_behind", "ats_record", "ou_record",
    }
    for row in rows:
        assert required_keys.issubset(row.keys()), f"Missing keys in {row['team']}: {required_keys - row.keys()}"


def test_build_conference_standings_ats_ou_null():
    from scripts.build_standings import build_conference_standings, EAST
    rows = build_conference_standings(SAMPLE_LOGS, TEAM_NAMES, "East", EAST)
    for row in rows:
        assert row["ats_record"] is None
        assert row["ou_record"] is None


def test_build_standings_writes_json(tmp_path):
    """build_standings() writes valid JSON with east/west/last_updated keys."""
    from scripts.build_standings import build_standings

    logs_csv = tmp_path / "team_game_logs.csv"
    teams_csv = tmp_path / "teams.csv"
    out_json = tmp_path / "standings.json"

    # Write minimal CSVs
    SAMPLE_LOGS.assign(
        game_date=SAMPLE_LOGS["game_date"].astype(str),
        season=202526,
    ).to_csv(logs_csv, index=False)

    pd.DataFrame([
        {"abbreviation": "BOS", "full_name": "Boston Celtics"},
        {"abbreviation": "BKN", "full_name": "Brooklyn Nets"},
        {"abbreviation": "MIA", "full_name": "Miami Heat"},
    ]).to_csv(teams_csv, index=False)

    result = build_standings(
        logs_path=logs_csv,
        teams_path=teams_csv,
        out_path=out_json,
        season=202526,
    )

    assert out_json.exists()
    with open(out_json) as f:
        data = json.load(f)
    assert "east" in data
    assert "west" in data
    assert "last_updated" in data
    assert isinstance(data["east"], list)
    assert isinstance(data["west"], list)
