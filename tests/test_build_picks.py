"""
Tests for scripts/build_picks.py
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_db(tmp_path: Path, rows: list[dict]) -> Path:
    """Create a predictions_history.db with given rows in game_predictions."""
    db_path = tmp_path / "predictions_history.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """CREATE TABLE game_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            game_date TEXT,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            home_win_prob REAL NOT NULL,
            away_win_prob REAL NOT NULL,
            model_name TEXT,
            model_artifact TEXT,
            decision_threshold REAL,
            feature_count INTEGER,
            actual_home_win INTEGER,
            notes TEXT
        )"""
    )
    for row in rows:
        conn.execute(
            """INSERT INTO game_predictions
               (created_at, game_date, home_team, away_team, home_win_prob, away_win_prob, model_name, model_artifact)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                row.get("created_at", "2026-03-06T00:00:00"),
                row.get("game_date", "2026-03-06"),
                row["home_team"],
                row["away_team"],
                row["home_win_prob"],
                row["away_win_prob"],
                row.get("model_name", "gradient_boosting_v2"),
                row.get("model_artifact", "game_outcome_model_calibrated.pkl"),
            ),
        )
    conn.commit()
    conn.close()
    return db_path


SAMPLE_ROWS = [
    {"game_date": "2026-03-06", "home_team": "BOS", "away_team": "NYK",
     "home_win_prob": 0.72, "away_win_prob": 0.28},
    {"game_date": "2026-03-06", "home_team": "LAL", "away_team": "GSW",
     "home_win_prob": 0.48, "away_win_prob": 0.52},
]


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

def test_import():
    from scripts.build_picks import build_picks, _build_pick_row, _passthrough
    assert callable(build_picks)


# ---------------------------------------------------------------------------
# _build_pick_row
# ---------------------------------------------------------------------------

def test_build_pick_row_basic():
    from scripts.build_picks import _build_pick_row
    pred = {
        "game_date": "2026-03-06",
        "home_team": "BOS",
        "away_team": "NYK",
        "home_win_prob": 0.72,
        "away_win_prob": 0.28,
        "model_name": "test_model",
        "created_at": "2026-03-06T00:00:00",
    }
    row = _build_pick_row(pred, {"BOS": "Boston Celtics", "NYK": "New York Knicks"}, {})
    assert row["home_team"] == "BOS"
    assert row["away_team"] == "NYK"
    assert row["predicted_winner"] == "BOS"
    assert row["home_win_prob"] == 0.72
    assert row["away_win_prob"] == 0.28


def test_build_pick_row_away_favourite():
    from scripts.build_picks import _build_pick_row
    pred = {
        "game_date": "2026-03-06",
        "home_team": "WAS",
        "away_team": "BOS",
        "home_win_prob": 0.3,
        "away_win_prob": 0.7,
        "model_name": "test_model",
        "created_at": "2026-03-06T00:00:00",
    }
    row = _build_pick_row(pred, {}, {})
    assert row["predicted_winner"] == "BOS"


def test_build_pick_row_value_bet_detected():
    from scripts.build_picks import _build_pick_row
    pred = {
        "game_date": "2026-03-06",
        "home_team": "BOS",
        "away_team": "NYK",
        "home_win_prob": 0.72,
        "away_win_prob": 0.28,
        "model_name": "test_model",
        "created_at": "2026-03-06T00:00:00",
    }
    # market_prob = 0.60, edge = 0.12 -> value_bet = True
    lines = {("BOS", "NYK"): {"spread": 5.0, "home_market_prob": 0.60}}
    row = _build_pick_row(pred, {}, lines)
    assert row["value_bet"] is True
    assert row["edge_pct"] is not None
    assert row["edge_pct"] > 0.05


def test_build_pick_row_no_value_bet_when_edge_small():
    from scripts.build_picks import _build_pick_row
    pred = {
        "game_date": "2026-03-06",
        "home_team": "BOS",
        "away_team": "NYK",
        "home_win_prob": 0.62,
        "away_win_prob": 0.38,
        "model_name": "test_model",
        "created_at": "2026-03-06T00:00:00",
    }
    # market_prob = 0.60, edge = 0.02 -> value_bet = False
    lines = {("BOS", "NYK"): {"spread": 3.0, "home_market_prob": 0.60}}
    row = _build_pick_row(pred, {}, lines)
    assert row["value_bet"] is False


def test_build_pick_row_schema_keys():
    from scripts.build_picks import _build_pick_row
    pred = {
        "game_date": "2026-03-06",
        "home_team": "BOS",
        "away_team": "NYK",
        "home_win_prob": 0.65,
        "away_win_prob": 0.35,
        "model_name": "test_model",
        "created_at": "2026-03-06T00:00:00",
    }
    row = _build_pick_row(pred, {}, {})
    required = {
        "game_date", "home_team", "away_team", "home_team_name", "away_team_name",
        "home_win_prob", "away_win_prob", "predicted_winner", "ats_pick",
        "spread", "value_bet", "edge_pct", "model_name", "created_at",
    }
    assert required.issubset(row.keys())


# ---------------------------------------------------------------------------
# build_picks
# ---------------------------------------------------------------------------

def test_build_picks_writes_json(tmp_path):
    db_path = _make_db(tmp_path, SAMPLE_ROWS)
    out_path = tmp_path / "todays_picks.json"
    from scripts.build_picks import build_picks
    result = build_picks(db_path=db_path, out_path=out_path, target_date="2026-03-06")
    assert out_path.exists()
    with open(out_path) as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) == 2


def test_build_picks_fallback_to_most_recent(tmp_path):
    """When target_date has no rows, fall back to most recent date."""
    db_path = _make_db(tmp_path, SAMPLE_ROWS)
    out_path = tmp_path / "todays_picks.json"
    from scripts.build_picks import build_picks
    # Request future date -- should fall back to 2026-03-06
    result = build_picks(db_path=db_path, out_path=out_path, target_date="2030-01-01")
    assert len(result) == 2
    assert result[0]["game_date"] == "2026-03-06"


def test_build_picks_passthrough_when_no_db(tmp_path):
    """Missing DB -> read existing JSON unchanged."""
    existing = [{"game_date": "2026-03-05", "home_team": "ORL", "away_team": "DAL"}]
    out_path = tmp_path / "todays_picks.json"
    with open(out_path, "w") as f:
        json.dump(existing, f)
    fake_db = tmp_path / "nonexistent.db"
    from scripts.build_picks import build_picks
    result = build_picks(db_path=fake_db, out_path=out_path, target_date="2026-03-06")
    assert result == existing


def test_build_picks_passthrough_empty_db(tmp_path):
    """Empty DB -> pass-through existing JSON."""
    db_path = _make_db(tmp_path, [])
    existing = [{"game_date": "2026-03-01"}]
    out_path = tmp_path / "todays_picks.json"
    with open(out_path, "w") as f:
        json.dump(existing, f)
    from scripts.build_picks import build_picks
    result = build_picks(db_path=db_path, out_path=out_path, target_date="2026-03-06")
    assert result == existing


def test_build_picks_team_names_populated(tmp_path):
    """Team names from teams.csv should appear in output."""
    import pandas as pd
    db_path = _make_db(tmp_path, SAMPLE_ROWS)
    teams_csv = tmp_path / "teams.csv"
    pd.DataFrame([
        {"abbreviation": "BOS", "full_name": "Boston Celtics"},
        {"abbreviation": "NYK", "full_name": "New York Knicks"},
        {"abbreviation": "LAL", "full_name": "Los Angeles Lakers"},
        {"abbreviation": "GSW", "full_name": "Golden State Warriors"},
    ]).to_csv(teams_csv, index=False)

    out_path = tmp_path / "todays_picks.json"

    # Patch TEAMS_CSV inside the module for this test
    import scripts.build_picks as bp
    original = bp.TEAMS_CSV
    bp.TEAMS_CSV = teams_csv
    try:
        result = bp.build_picks(db_path=db_path, out_path=out_path, target_date="2026-03-06")
    finally:
        bp.TEAMS_CSV = original

    bos_row = next(r for r in result if r["home_team"] == "BOS")
    assert bos_row["home_team_name"] == "Boston Celtics"


# ---------------------------------------------------------------------------
# _compute_kelly_fraction
# ---------------------------------------------------------------------------

def test_kelly_positive_edge():
    from scripts.build_picks import _compute_kelly_fraction
    # Model 70%, market 55% -> positive edge
    result = _compute_kelly_fraction(0.15, 0.55, "BOS", "BOS", 0.70)
    assert result is not None
    assert result > 0.0


def test_kelly_zero_when_negative_edge():
    from scripts.build_picks import _compute_kelly_fraction
    # Model 45%, market 55% -> negative edge -> 0.0
    result = _compute_kelly_fraction(-0.10, 0.55, "BOS", "BOS", 0.45)
    assert result == 0.0


def test_kelly_none_when_no_market_prob():
    from scripts.build_picks import _compute_kelly_fraction
    result = _compute_kelly_fraction(0.10, None, "BOS", "BOS", 0.65)
    assert result is None


def test_kelly_none_when_no_edge():
    from scripts.build_picks import _compute_kelly_fraction
    result = _compute_kelly_fraction(None, 0.55, "BOS", "BOS", 0.65)
    assert result is None


def test_kelly_none_when_market_degenerate():
    from scripts.build_picks import _compute_kelly_fraction
    # market_prob of 0 or 1 are degenerate
    assert _compute_kelly_fraction(0.10, 0.0, "BOS", "BOS", 0.65) is None
    assert _compute_kelly_fraction(0.10, 1.0, "BOS", "BOS", 0.65) is None


def test_kelly_away_side():
    from scripts.build_picks import _compute_kelly_fraction
    # Away team is ats_pick -- uses (1 - home_prob) as model prob
    result = _compute_kelly_fraction(0.15, 0.55, "NYK", "BOS", 0.40)
    assert result is not None
    assert result > 0.0


def test_kelly_half_kelly():
    from scripts.build_picks import _compute_kelly_fraction
    # Full Kelly = (p*b - (1-p)) / b; result should be half of that
    p = 0.65
    q = 0.50
    b = (1 - q) / q  # = 1.0
    full_kelly = (p * b - (1 - p)) / b  # = 0.30
    result = _compute_kelly_fraction(0.15, q, "BOS", "BOS", p)
    assert result == pytest.approx(full_kelly * 0.5, abs=0.001)


# ---------------------------------------------------------------------------
# _confidence_tier
# ---------------------------------------------------------------------------

def test_confidence_high():
    from scripts.build_picks import _confidence_tier
    assert _confidence_tier(0.75, "BOS", "BOS") == "HIGH"


def test_confidence_high_away():
    from scripts.build_picks import _confidence_tier
    # Away win prob = 1 - 0.25 = 0.75
    assert _confidence_tier(0.25, "NYK", "BOS") == "HIGH"


def test_confidence_medium():
    from scripts.build_picks import _confidence_tier
    assert _confidence_tier(0.62, "BOS", "BOS") == "MEDIUM"


def test_confidence_low():
    from scripts.build_picks import _confidence_tier
    assert _confidence_tier(0.55, "BOS", "BOS") == "LOW"


def test_confidence_boundary_70():
    from scripts.build_picks import _confidence_tier
    # Exactly 70% -> HIGH
    assert _confidence_tier(0.70, "BOS", "BOS") == "HIGH"


def test_confidence_boundary_60():
    from scripts.build_picks import _confidence_tier
    # Exactly 60% -> MEDIUM
    assert _confidence_tier(0.60, "BOS", "BOS") == "MEDIUM"


def test_confidence_returns_string():
    from scripts.build_picks import _confidence_tier
    result = _confidence_tier(0.65, "BOS", "BOS")
    assert isinstance(result, str)
    assert result in ("HIGH", "MEDIUM", "LOW")
