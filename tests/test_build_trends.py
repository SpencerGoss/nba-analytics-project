"""
Tests for scripts/build_trends.py
"""

from __future__ import annotations

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_logs(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")
    df["is_home"] = df["matchup"].str.contains(" vs. ", regex=False)
    # Build opp_pts via simple approach used in build_trends.load_logs
    pts_map = (
        df.groupby("game_id")[["team_abbreviation", "pts"]]
        .apply(lambda g: g.set_index("team_abbreviation")["pts"].to_dict())
        .to_dict()
    )

    def _opp(row):
        gmap = pts_map.get(row["game_id"], {})
        for t, p in gmap.items():
            if t != row["team_abbreviation"]:
                return float(p)
        return float("nan")

    df["opp_pts"] = df.apply(_opp, axis=1)
    return df.sort_values(["team_abbreviation", "game_date"]).reset_index(drop=True)


def _make_features(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")
    return df


SAMPLE_LOGS = _make_logs([
    # LAL home wins
    {"game_id": 1, "team_abbreviation": "LAL", "matchup": "LAL vs. BOS", "wl": "W", "pts": 115, "game_date": "2025-01-01", "season": 202526},
    {"game_id": 1, "team_abbreviation": "BOS", "matchup": "BOS @ LAL", "wl": "L", "pts": 100, "game_date": "2025-01-01", "season": 202526},
    # LAL away loss
    {"game_id": 2, "team_abbreviation": "LAL", "matchup": "LAL @ GSW", "wl": "L", "pts": 108, "game_date": "2025-01-03", "season": 202526},
    {"game_id": 2, "team_abbreviation": "GSW", "matchup": "GSW vs. LAL", "wl": "W", "pts": 120, "game_date": "2025-01-03", "season": 202526},
    # LAL home win
    {"game_id": 3, "team_abbreviation": "LAL", "matchup": "LAL vs. MIA", "wl": "W", "pts": 122, "game_date": "2025-01-05", "season": 202526},
    {"game_id": 3, "team_abbreviation": "MIA", "matchup": "MIA @ LAL", "wl": "L", "pts": 110, "game_date": "2025-01-05", "season": 202526},
])

SAMPLE_FEATURES = _make_features([
    {"team_abbreviation": "LAL", "game_date": "2025-01-05", "season": 202526,
     "off_rtg_game_roll10": 115.0, "def_rtg_game_roll10": 108.0,
     "pace_game_roll10": 98.5, "fg3_pct_roll10": 0.38,
     "off_rtg_game_roll20": 112.0, "def_rtg_game_roll20": 110.0},
])


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

def test_import():
    from scripts.build_trends import (
        _current_streak,
        _fmt_diff,
        _record_str,
        compute_team_trends,
    )
    assert callable(compute_team_trends)


# ---------------------------------------------------------------------------
# _record_str
# ---------------------------------------------------------------------------

def test_record_str_basic():
    from scripts.build_trends import _record_str
    assert _record_str(8, 2) == "8-2"
    assert _record_str(0, 10) == "0-10"


# ---------------------------------------------------------------------------
# _current_streak
# ---------------------------------------------------------------------------

def test_streak_win():
    from scripts.build_trends import _current_streak
    series = pd.Series(["L", "W", "W", "W"])
    assert _current_streak(series) == 3


def test_streak_loss():
    from scripts.build_trends import _current_streak
    series = pd.Series(["W", "W", "L", "L"])
    assert _current_streak(series) == -2


def test_streak_empty():
    from scripts.build_trends import _current_streak
    assert _current_streak(pd.Series([], dtype=str)) == 0


def test_streak_single_win():
    from scripts.build_trends import _current_streak
    assert _current_streak(pd.Series(["W"])) == 1


# ---------------------------------------------------------------------------
# _fmt_diff
# ---------------------------------------------------------------------------

def test_fmt_diff_positive():
    from scripts.build_trends import _fmt_diff
    assert _fmt_diff(4.2) == "+4.2"


def test_fmt_diff_negative():
    from scripts.build_trends import _fmt_diff
    assert _fmt_diff(-2.1) == "-2.1"


def test_fmt_diff_zero():
    from scripts.build_trends import _fmt_diff
    assert _fmt_diff(0.0) == "+0.0"


# ---------------------------------------------------------------------------
# compute_team_trends
# ---------------------------------------------------------------------------

def test_compute_team_trends_keys():
    from scripts.build_trends import compute_team_trends
    result = compute_team_trends(SAMPLE_LOGS, SAMPLE_FEATURES)
    assert "LAL" in result
    lal = result["LAL"]
    expected_keys = {
        "last10_su", "last10_avg_scored", "last10_avg_allowed",
        "home_record", "away_record", "streak",
        "trend_offense", "trend_defense", "last10_pace", "last10_fg3_rate",
    }
    assert expected_keys == set(lal.keys())


def test_compute_team_trends_home_record():
    from scripts.build_trends import compute_team_trends
    result = compute_team_trends(SAMPLE_LOGS, SAMPLE_FEATURES)
    lal = result["LAL"]
    # LAL home games: vs BOS (W), vs MIA (W) -> 2-0
    # LAL away game: @ GSW (L) -> 0-1
    assert lal["home_record"] == "2-0", f"Expected 2-0, got {lal['home_record']}"
    assert lal["away_record"] == "0-1", f"Expected 0-1, got {lal['away_record']}"


def test_compute_team_trends_last10_su():
    from scripts.build_trends import compute_team_trends
    result = compute_team_trends(SAMPLE_LOGS, SAMPLE_FEATURES)
    lal = result["LAL"]
    assert lal["last10_su"] == "2-1"


def test_compute_team_trends_avg_scored():
    from scripts.build_trends import compute_team_trends
    result = compute_team_trends(SAMPLE_LOGS, SAMPLE_FEATURES)
    lal = result["LAL"]
    expected_avg = round((115 + 108 + 122) / 3, 1)
    assert lal["last10_avg_scored"] == expected_avg


def test_compute_team_trends_streak():
    from scripts.build_trends import compute_team_trends
    result = compute_team_trends(SAMPLE_LOGS, SAMPLE_FEATURES)
    lal = result["LAL"]
    # Last game was W (vs MIA on Jan 5) -> streak +1
    assert lal["streak"] == 1


def test_compute_team_trends_trend_offense():
    from scripts.build_trends import compute_team_trends
    result = compute_team_trends(SAMPLE_LOGS, SAMPLE_FEATURES)
    lal = result["LAL"]
    # roll10=115 - roll20=112 = +3.0
    assert lal["trend_offense"] == "+3.0"


def test_compute_team_trends_trend_defense():
    from scripts.build_trends import compute_team_trends
    result = compute_team_trends(SAMPLE_LOGS, SAMPLE_FEATURES)
    lal = result["LAL"]
    # roll10=108 - roll20=110 = -2.0
    assert lal["trend_defense"] == "-2.0"


def test_compute_team_trends_no_features():
    """When features are absent for a team, rating fields should be None."""
    from scripts.build_trends import compute_team_trends
    empty_feat = pd.DataFrame(columns=SAMPLE_FEATURES.columns)
    result = compute_team_trends(SAMPLE_LOGS, empty_feat)
    lal = result["LAL"]
    assert lal["trend_offense"] is None
    assert lal["trend_defense"] is None
    assert lal["last10_pace"] is None


def test_compute_team_trends_all_teams_present():
    """All teams appearing in logs should appear in output."""
    from scripts.build_trends import compute_team_trends
    result = compute_team_trends(SAMPLE_LOGS, SAMPLE_FEATURES)
    for team in ["LAL", "BOS", "GSW", "MIA"]:
        assert team in result


# ---------------------------------------------------------------------------
# Additional _record_str edge cases
# ---------------------------------------------------------------------------

def test_record_str_all_wins():
    from scripts.build_trends import _record_str
    assert _record_str(10, 0) == "10-0"


def test_record_str_all_losses():
    from scripts.build_trends import _record_str
    assert _record_str(0, 10) == "0-10"


def test_record_str_returns_string():
    from scripts.build_trends import _record_str
    result = _record_str(5, 3)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Additional _fmt_diff edge cases
# ---------------------------------------------------------------------------

def test_fmt_diff_large_positive():
    from scripts.build_trends import _fmt_diff
    assert _fmt_diff(10.0) == "+10.0"


def test_fmt_diff_large_negative():
    from scripts.build_trends import _fmt_diff
    assert _fmt_diff(-10.0) == "-10.0"


# ---------------------------------------------------------------------------
# Additional compute_team_trends checks
# ---------------------------------------------------------------------------

def test_compute_team_trends_avg_allowed():
    from scripts.build_trends import compute_team_trends
    result = compute_team_trends(SAMPLE_LOGS, SAMPLE_FEATURES)
    lal = result["LAL"]
    # LAL opp_pts: BOS scored 100, GSW scored 120, MIA scored 110 -> avg = 110.0
    expected = round((100 + 120 + 110) / 3, 1)
    assert lal["last10_avg_allowed"] == expected


def test_compute_team_trends_last10_pace():
    from scripts.build_trends import compute_team_trends
    result = compute_team_trends(SAMPLE_LOGS, SAMPLE_FEATURES)
    lal = result["LAL"]
    # pace_game_roll10 = 98.5 from SAMPLE_FEATURES
    assert lal["last10_pace"] == 98.5
