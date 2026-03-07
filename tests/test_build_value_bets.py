"""
Tests for scripts/build_value_bets.py
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_db(tmp_path: Path, rows: list[dict]) -> Path:
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
               (created_at, game_date, home_team, away_team, home_win_prob, away_win_prob)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                row.get("created_at", "2026-03-06T00:00:00"),
                row.get("game_date", "2026-03-06"),
                row["home_team"],
                row["away_team"],
                row["home_win_prob"],
                row["away_win_prob"],
            ),
        )
    conn.commit()
    conn.close()
    return db_path


def _make_lines_csv(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "game_lines.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

def test_import():
    from scripts.build_value_bets import (
        _compute_value_bets,
        _build_lines_index,
        build_value_bets,
    )
    assert callable(build_value_bets)


# ---------------------------------------------------------------------------
# _build_lines_index
# ---------------------------------------------------------------------------

def test_build_lines_index_keying():
    from scripts.build_value_bets import _build_lines_index
    df = pd.DataFrame([
        {"game_date": "2026-03-06", "home_team": "BOS", "away_team": "NYK", "home_market_prob": 0.6},
    ])
    idx = _build_lines_index(df)
    assert ("2026-03-06", "BOS", "NYK") in idx
    assert idx[("2026-03-06", "BOS", "NYK")]["home_market_prob"] == 0.6


def test_build_lines_index_empty_df():
    from scripts.build_value_bets import _build_lines_index
    idx = _build_lines_index(pd.DataFrame())
    assert idx == {}


# ---------------------------------------------------------------------------
# _compute_value_bets
# ---------------------------------------------------------------------------

def test_compute_value_bets_detects_edge():
    from scripts.build_value_bets import _compute_value_bets
    predictions = [
        {
            "game_date": "2026-03-06",
            "home_team": "BOS",
            "away_team": "NYK",
            "home_win_prob": 0.72,
            "created_at": "2026-03-06T00:00:00",
        }
    ]
    lines_index = {("2026-03-06", "BOS", "NYK"): {"home_market_prob": 0.60}}
    result = _compute_value_bets(predictions, lines_index, {}, edge_threshold=0.05)
    assert len(result) == 1
    assert result[0]["recommended_side"] == "BOS"
    assert result[0]["edge_pct"] == pytest.approx(0.12, abs=0.001)


def test_compute_value_bets_away_edge():
    """Edge on away side: model_prob for home is low, away side has the edge."""
    from scripts.build_value_bets import _compute_value_bets
    predictions = [
        {
            "game_date": "2026-03-06",
            "home_team": "WAS",
            "away_team": "BOS",
            "home_win_prob": 0.30,
            "created_at": "2026-03-06T00:00:00",
        }
    ]
    # market_prob for home (WAS) = 0.55, so market favours home
    # model says home 30%, away 70%. market says home 55%, away 45%.
    # edge on away = 0.70 - 0.45 = 0.25 -> value bet on BOS
    lines_index = {("2026-03-06", "WAS", "BOS"): {"home_market_prob": 0.55}}
    result = _compute_value_bets(predictions, lines_index, {}, edge_threshold=0.05)
    assert len(result) == 1
    assert result[0]["recommended_side"] == "BOS"


def test_compute_value_bets_below_threshold_filtered():
    from scripts.build_value_bets import _compute_value_bets
    predictions = [
        {
            "game_date": "2026-03-06",
            "home_team": "BOS",
            "away_team": "NYK",
            "home_win_prob": 0.62,
            "created_at": "2026-03-06T00:00:00",
        }
    ]
    # edge = 0.02 -> below 0.05 threshold
    lines_index = {("2026-03-06", "BOS", "NYK"): {"home_market_prob": 0.60}}
    result = _compute_value_bets(predictions, lines_index, {}, edge_threshold=0.05)
    assert len(result) == 0


def test_compute_value_bets_sorted_by_edge():
    from scripts.build_value_bets import _compute_value_bets
    predictions = [
        {"game_date": "2026-03-06", "home_team": "BOS", "away_team": "NYK",
         "home_win_prob": 0.80, "created_at": "2026-03-06T00:00:00"},
        {"game_date": "2026-03-06", "home_team": "LAL", "away_team": "GSW",
         "home_win_prob": 0.70, "created_at": "2026-03-06T00:00:00"},
    ]
    lines_index = {
        ("2026-03-06", "BOS", "NYK"): {"home_market_prob": 0.60},   # edge 0.20
        ("2026-03-06", "LAL", "GSW"): {"home_market_prob": 0.60},   # edge 0.10
    }
    result = _compute_value_bets(predictions, lines_index, {}, edge_threshold=0.05)
    assert len(result) == 2
    assert result[0]["edge_pct"] >= result[1]["edge_pct"]


def test_compute_value_bets_no_matching_lines():
    from scripts.build_value_bets import _compute_value_bets
    predictions = [
        {"game_date": "2026-03-06", "home_team": "BOS", "away_team": "NYK",
         "home_win_prob": 0.80, "created_at": "2026-03-06T00:00:00"},
    ]
    # Lines for a different game
    lines_index = {("2026-03-06", "OKC", "POR"): {"home_market_prob": 0.55}}
    result = _compute_value_bets(predictions, lines_index, {}, edge_threshold=0.05)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# build_value_bets (integration)
# ---------------------------------------------------------------------------

def test_build_value_bets_passthrough_no_lines(tmp_path):
    """No game_lines.csv -> pass through existing JSON."""
    existing = [{"game_date": "2026-03-05", "recommended_side": "BOS", "edge_pct": 0.10}]
    out_path = tmp_path / "value_bets.json"
    with open(out_path, "w") as f:
        json.dump(existing, f)

    db_path = _make_db(tmp_path, [
        {"game_date": "2026-03-06", "home_team": "BOS", "away_team": "NYK",
         "home_win_prob": 0.72, "away_win_prob": 0.28},
    ])

    import scripts.build_value_bets as bvb
    original = bvb.GAME_LINES_CSV
    bvb.GAME_LINES_CSV = tmp_path / "game_lines_nonexistent.csv"
    try:
        result = bvb.build_value_bets(db_path=db_path, out_path=out_path)
    finally:
        bvb.GAME_LINES_CSV = original

    assert result == existing


def test_build_value_bets_writes_json(tmp_path):
    """Full pipeline: DB + lines -> writes value_bets.json."""
    db_path = _make_db(tmp_path, [
        {"game_date": "2026-03-06", "home_team": "BOS", "away_team": "NYK",
         "home_win_prob": 0.72, "away_win_prob": 0.28},
        {"game_date": "2026-03-06", "home_team": "LAL", "away_team": "GSW",
         "home_win_prob": 0.48, "away_win_prob": 0.52},  # below threshold
    ])
    lines_path = _make_lines_csv(tmp_path, [
        {"game_date": "2026-03-06", "home_team": "BOS", "away_team": "NYK", "home_market_prob": 0.60},
        {"game_date": "2026-03-06", "home_team": "LAL", "away_team": "GSW", "home_market_prob": 0.47},
    ])
    out_path = tmp_path / "value_bets.json"

    import scripts.build_value_bets as bvb
    original = bvb.GAME_LINES_CSV
    bvb.GAME_LINES_CSV = lines_path
    try:
        result = bvb.build_value_bets(db_path=db_path, out_path=out_path)
    finally:
        bvb.GAME_LINES_CSV = original

    assert out_path.exists()
    with open(out_path) as f:
        data = json.load(f)
    assert isinstance(data, list)
    # BOS has 0.12 edge -> value bet; LAL has 0.01 edge -> not a value bet
    assert any(r["recommended_side"] == "BOS" for r in data)
    assert all(r["recommended_side"] != "LAL" for r in data)


def test_build_value_bets_schema(tmp_path):
    """Each row in output must have required schema keys."""
    db_path = _make_db(tmp_path, [
        {"game_date": "2026-03-06", "home_team": "OKC", "away_team": "POR",
         "home_win_prob": 0.75, "away_win_prob": 0.25},
    ])
    lines_path = _make_lines_csv(tmp_path, [
        {"game_date": "2026-03-06", "home_team": "OKC", "away_team": "POR", "home_market_prob": 0.55},
    ])
    out_path = tmp_path / "value_bets.json"

    import scripts.build_value_bets as bvb
    original = bvb.GAME_LINES_CSV
    bvb.GAME_LINES_CSV = lines_path
    try:
        result = bvb.build_value_bets(db_path=db_path, out_path=out_path)
    finally:
        bvb.GAME_LINES_CSV = original

    assert len(result) == 1
    required_keys = {
        "game_date", "home_team", "away_team", "home_team_name", "away_team_name",
        "model_prob", "market_prob", "edge_pct", "recommended_side", "created_at",
    }
    assert required_keys.issubset(result[0].keys())
