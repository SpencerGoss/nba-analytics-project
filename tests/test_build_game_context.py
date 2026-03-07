"""
Tests for scripts/build_game_context.py

All tests use synthetic DataFrames -- no file I/O against real CSVs.
"""
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

# Make the scripts/ directory importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_game_context import (
    _compute_streak,
    away_streak,
    home_streak,
    is_b2b,
    last10_record,
    rest_days,
    season_away_record,
    season_home_record,
    situational_flags,
    build_context_for_game,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_logs(rows: list[dict]) -> pd.DataFrame:
    """
    Build a minimal team_game_logs DataFrame from a list of dicts.
    Required keys: team_abbreviation, game_date (date obj), matchup, wl, season
    Handles empty list by returning a properly-columned empty DataFrame.
    """
    _COLS = ["team_abbreviation", "game_date", "matchup", "wl", "season", "is_home", "win"]
    if not rows:
        return pd.DataFrame(columns=_COLS)
    df = pd.DataFrame(rows)
    df["is_home"] = df["matchup"].str.contains(r"vs\.", regex=True)
    df["win"] = df["wl"].str.upper() == "W"
    df = df.sort_values("game_date", ascending=True)
    return df


def _home_row(team: str, opp: str, game_date: date, wl: str, season: int = 202526) -> dict:
    return {
        "team_abbreviation": team,
        "game_date": game_date,
        "matchup": f"{team} vs. {opp}",
        "wl": wl,
        "season": season,
    }


def _away_row(team: str, opp: str, game_date: date, wl: str, season: int = 202526) -> dict:
    return {
        "team_abbreviation": team,
        "game_date": game_date,
        "matchup": f"{team} @ {opp}",
        "wl": wl,
        "season": season,
    }


# ---------------------------------------------------------------------------
# Test 1: is_b2b and rest_days
# ---------------------------------------------------------------------------

class TestRestAndB2B:
    def test_b2b_true_when_played_yesterday(self):
        logs = _make_logs([
            _away_row("OKC", "LAL", date(2026, 3, 4), "W"),
        ])
        assert is_b2b(logs, "OKC", date(2026, 3, 5)) is True

    def test_b2b_false_when_played_two_days_ago(self):
        logs = _make_logs([
            _away_row("OKC", "LAL", date(2026, 3, 3), "W"),
        ])
        assert is_b2b(logs, "OKC", date(2026, 3, 5)) is False

    def test_b2b_false_when_no_prior_game(self):
        logs = _make_logs([])
        assert is_b2b(logs, "OKC", date(2026, 3, 5)) is False

    def test_rest_days_exact(self):
        logs = _make_logs([
            _home_row("BOS", "MIA", date(2026, 3, 1), "W"),
        ])
        assert rest_days(logs, "BOS", date(2026, 3, 4)) == 3

    def test_rest_days_none_when_no_prior_game(self):
        logs = _make_logs([])
        assert rest_days(logs, "BOS", date(2026, 3, 4)) is None

    def test_rest_days_uses_most_recent_game(self):
        logs = _make_logs([
            _home_row("BOS", "MIA", date(2026, 2, 28), "W"),
            _away_row("BOS", "NYK", date(2026, 3, 2), "L"),
        ])
        # Most recent is Mar 2, so rest = 2 days before Mar 4
        assert rest_days(logs, "BOS", date(2026, 3, 4)) == 2


# ---------------------------------------------------------------------------
# Test 2: _compute_streak
# ---------------------------------------------------------------------------

class TestComputeStreak:
    def test_win_streak(self):
        assert _compute_streak([True, True, True, False]) == 3

    def test_loss_streak(self):
        assert _compute_streak([False, False, True]) == -2

    def test_empty_list(self):
        assert _compute_streak([]) == 0

    def test_single_win(self):
        assert _compute_streak([True]) == 1

    def test_single_loss(self):
        assert _compute_streak([False]) == -1

    def test_alternating_starts_win(self):
        # Most recent is win, then loss -- streak = 1
        assert _compute_streak([True, False, True]) == 1


# ---------------------------------------------------------------------------
# Test 3: home_streak and away_streak
# ---------------------------------------------------------------------------

class TestStreaks:
    def _logs(self):
        return _make_logs([
            # Home games for OKC -- W W W
            _home_row("OKC", "LAL", date(2026, 2, 20), "W"),
            _home_row("OKC", "DEN", date(2026, 2, 25), "W"),
            _home_row("OKC", "HOU", date(2026, 3, 1), "W"),
            # Away games for OKC -- L L
            _away_row("OKC", "BOS", date(2026, 2, 22), "L"),
            _away_row("OKC", "MIA", date(2026, 2, 27), "L"),
        ])

    def test_home_streak_three_wins(self):
        logs = self._logs()
        assert home_streak(logs, "OKC", date(2026, 3, 6)) == 3

    def test_away_streak_two_losses(self):
        logs = self._logs()
        assert away_streak(logs, "OKC", date(2026, 3, 6)) == -2

    def test_home_streak_excludes_game_on_same_date(self):
        logs = self._logs()
        # Asking for streak *before* Mar 1 -- only 2 home wins visible
        assert home_streak(logs, "OKC", date(2026, 3, 1)) == 2

    def test_home_streak_empty_returns_zero(self):
        logs = _make_logs([])
        assert home_streak(logs, "OKC", date(2026, 3, 6)) == 0

    def test_away_streak_empty_returns_zero(self):
        logs = _make_logs([])
        assert away_streak(logs, "OKC", date(2026, 3, 6)) == 0


# ---------------------------------------------------------------------------
# Test 4: last10_record and season records
# ---------------------------------------------------------------------------

class TestRecords:
    def _logs_7w_3l(self) -> pd.DataFrame:
        rows = []
        base = date(2026, 2, 1)
        wins = [True, True, True, True, True, True, True, False, False, False]
        for i, w in enumerate(wins):
            rows.append(_home_row("LAC", "OPP", date(2026, 2, i + 1), "W" if w else "L"))
        return _make_logs(rows)

    def test_last10_record_7w_3l(self):
        logs = self._logs_7w_3l()
        assert last10_record(logs, "LAC", date(2026, 3, 6)) == "7-3"

    def test_last10_record_only_uses_10_most_recent(self):
        # 15 home games: first 5 are losses, last 10 are all wins
        rows = []
        for i in range(5):
            rows.append(_home_row("LAC", "OPP", date(2026, 1, i + 1), "L"))
        for i in range(10):
            rows.append(_home_row("LAC", "OPP", date(2026, 2, i + 1), "W"))
        logs = _make_logs(rows)
        assert last10_record(logs, "LAC", date(2026, 3, 6)) == "10-0"

    def test_last10_record_empty(self):
        logs = _make_logs([])
        assert last10_record(logs, "LAC", date(2026, 3, 6)) == "0-0"

    def test_season_home_record(self):
        rows = [
            _home_row("DET", "OPP", date(2026, 2, 1), "W"),
            _home_row("DET", "OPP", date(2026, 2, 5), "W"),
            _home_row("DET", "OPP", date(2026, 2, 10), "L"),
            _away_row("DET", "OPP", date(2026, 2, 15), "W"),  # road -- not counted
        ]
        logs = _make_logs(rows)
        assert season_home_record(logs, "DET", date(2026, 3, 6)) == "2-1"

    def test_season_away_record(self):
        rows = [
            _away_row("MIA", "OPP", date(2026, 2, 1), "W"),
            _away_row("MIA", "OPP", date(2026, 2, 5), "L"),
            _away_row("MIA", "OPP", date(2026, 2, 10), "L"),
            _home_row("MIA", "OPP", date(2026, 2, 15), "W"),  # home -- not counted
        ]
        logs = _make_logs(rows)
        assert season_away_record(logs, "MIA", date(2026, 3, 6)) == "1-2"


# ---------------------------------------------------------------------------
# Test 5: situational_flags
# ---------------------------------------------------------------------------

class TestSituationalFlags:
    def test_away_b2b_flag(self):
        flags = situational_flags(
            home_b2b=False, away_b2b=True,
            home_rest=2, away_rest=1,
            home_l10="5-5", away_l10="5-5",
        )
        assert "AWAY_B2B" in flags
        assert "HOME_B2B" not in flags

    def test_home_b2b_flag(self):
        flags = situational_flags(
            home_b2b=True, away_b2b=False,
            home_rest=1, away_rest=3,
            home_l10="5-5", away_l10="5-5",
        )
        assert "HOME_B2B" in flags

    def test_home_hot_flag(self):
        flags = situational_flags(
            home_b2b=False, away_b2b=False,
            home_rest=2, away_rest=2,
            home_l10="7-3", away_l10="5-5",
        )
        assert "HOME_HOT" in flags
        assert "HOME_COLD" not in flags

    def test_away_cold_flag(self):
        flags = situational_flags(
            home_b2b=False, away_b2b=False,
            home_rest=2, away_rest=2,
            home_l10="5-5", away_l10="3-7",
        )
        assert "AWAY_COLD" in flags
        assert "AWAY_HOT" not in flags

    def test_home_rest_adv_requires_opponent_fewer_rest(self):
        # Home has 4 days rest, away has 2 -- home gets REST_ADV
        flags = situational_flags(
            home_b2b=False, away_b2b=False,
            home_rest=4, away_rest=2,
            home_l10="5-5", away_l10="5-5",
        )
        assert "HOME_REST_ADV" in flags
        assert "AWAY_REST_ADV" not in flags

    def test_away_rest_adv(self):
        flags = situational_flags(
            home_b2b=False, away_b2b=False,
            home_rest=2, away_rest=5,
            home_l10="5-5", away_l10="5-5",
        )
        assert "AWAY_REST_ADV" in flags
        assert "HOME_REST_ADV" not in flags

    def test_no_rest_adv_when_equal(self):
        flags = situational_flags(
            home_b2b=False, away_b2b=False,
            home_rest=3, away_rest=3,
            home_l10="5-5", away_l10="5-5",
        )
        # Neither gets REST_ADV when equal
        assert "HOME_REST_ADV" not in flags
        assert "AWAY_REST_ADV" not in flags

    def test_no_flags_neutral_game(self):
        flags = situational_flags(
            home_b2b=False, away_b2b=False,
            home_rest=2, away_rest=2,
            home_l10="5-5", away_l10="5-5",
        )
        assert flags == []

    def test_multiple_flags_combined(self):
        flags = situational_flags(
            home_b2b=True, away_b2b=True,
            home_rest=1, away_rest=1,
            home_l10="8-2", away_l10="2-8",
        )
        assert "HOME_B2B" in flags
        assert "AWAY_B2B" in flags
        assert "HOME_HOT" in flags
        assert "AWAY_COLD" in flags


# ---------------------------------------------------------------------------
# Test 6: build_context_for_game (integration-style with synthetic logs)
# ---------------------------------------------------------------------------

class TestBuildContextForGame:
    def _make_scenario(self) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
        """
        OKC (home) vs POR (away), game on 2026-03-06.
        OKC: played 2026-03-04 (2 days rest), 3 home wins in a row, 7-3 last 10
        POR: played 2026-03-05 (b2b), 2 road losses in a row, 3-7 last 10
        """
        game_date = date(2026, 3, 6)

        okc_rows = [
            # Away games (not counted for home streak)
            _away_row("OKC", "LAL", date(2026, 2, 10), "W"),
            _away_row("OKC", "DEN", date(2026, 2, 15), "L"),
            _away_row("OKC", "HOU", date(2026, 2, 20), "L"),
            # Home wins: 3 in a row
            _home_row("OKC", "MIA", date(2026, 2, 22), "W"),
            _home_row("OKC", "BOS", date(2026, 2, 26), "W"),
            _home_row("OKC", "CHI", date(2026, 3, 1), "W"),
            # Last game 2 days before -> 2 rest days
            _home_row("OKC", "ATL", date(2026, 3, 4), "W"),
        ]
        por_rows = [
            _away_row("POR", "OKC", date(2026, 2, 15), "W"),
            _away_row("POR", "NYK", date(2026, 2, 20), "W"),
            _away_row("POR", "MIL", date(2026, 2, 25), "W"),
            _away_row("POR", "PHX", date(2026, 2, 28), "L"),
            _away_row("POR", "LAL", date(2026, 3, 5), "L"),  # played yesterday -> b2b
        ]
        all_rows = okc_rows + por_rows
        logs = _make_logs(all_rows)
        season_logs = logs[logs["season"] == 202526].copy().sort_values("game_date")

        pick = {
            "game_date": "2026-03-06",
            "home_team": "OKC",
            "away_team": "POR",
        }
        return pick, logs, season_logs

    def test_away_b2b_detected(self):
        pick, logs, season_logs = self._make_scenario()
        ctx = build_context_for_game(pick, logs, season_logs)
        assert ctx["away_b2b"] is True
        assert ctx["home_b2b"] is False

    def test_home_rest_days(self):
        pick, logs, season_logs = self._make_scenario()
        ctx = build_context_for_game(pick, logs, season_logs)
        assert ctx["home_rest_days"] == 2

    def test_away_rest_days_b2b(self):
        pick, logs, season_logs = self._make_scenario()
        ctx = build_context_for_game(pick, logs, season_logs)
        assert ctx["away_rest_days"] == 1

    def test_home_streak_positive(self):
        pick, logs, season_logs = self._make_scenario()
        ctx = build_context_for_game(pick, logs, season_logs)
        # OKC has 4 consecutive home wins
        assert ctx["home_streak"] == 4

    def test_away_streak_negative(self):
        pick, logs, season_logs = self._make_scenario()
        ctx = build_context_for_game(pick, logs, season_logs)
        # POR last 2 road games = losses
        assert ctx["away_streak"] == -2

    def test_away_b2b_flag_in_situational_flags(self):
        pick, logs, season_logs = self._make_scenario()
        ctx = build_context_for_game(pick, logs, season_logs)
        assert "AWAY_B2B" in ctx["situational_flags"]

    def test_output_keys_present(self):
        pick, logs, season_logs = self._make_scenario()
        ctx = build_context_for_game(pick, logs, season_logs)
        required_keys = [
            "home_team", "away_team", "game_date",
            "home_b2b", "away_b2b",
            "home_rest_days", "away_rest_days",
            "home_last10", "away_last10",
            "home_streak", "away_streak",
            "home_season_home_record", "away_season_away_record",
            "situational_flags",
        ]
        for key in required_keys:
            assert key in ctx, f"Missing key: {key}"

    def test_last10_record_format(self):
        pick, logs, season_logs = self._make_scenario()
        ctx = build_context_for_game(pick, logs, season_logs)
        # Records must be in 'W-L' format
        for field in ("home_last10", "away_last10"):
            val = ctx[field]
            parts = val.split("-")
            assert len(parts) == 2, f"{field}={val!r} is not W-L format"
            assert parts[0].isdigit() and parts[1].isdigit()
