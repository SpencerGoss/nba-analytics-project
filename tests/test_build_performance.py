"""
Tests for scripts/build_performance.py.

Coverage targets:
- accuracy history loading and season accuracy computation
- rolling accuracy (7-day window)
- streak computation (current and best)
- ROI by market (from game_predictions)
- CLV summary (null path and data path)
- calibration buckets
- end-to-end build_performance() using temp files
"""
import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

# Make the scripts package importable
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_performance import (
    CALIBRATION_BUCKETS,
    build_performance,
    compute_calibration,
    compute_clv_summary,
    compute_roi_by_market,
    compute_rolling_accuracy_7d,
    compute_season_accuracy,
    compute_streaks,
    load_accuracy_history,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_HISTORY = [
    {"date": "2026-01-01", "daily_accuracy": 0.75, "rolling_accuracy": 0.75, "games": 8, "correct": 6, "cumulative_games": 8},
    {"date": "2026-01-02", "daily_accuracy": 0.50, "rolling_accuracy": 0.625, "games": 4, "correct": 2, "cumulative_games": 12},
    {"date": "2026-01-03", "daily_accuracy": 1.00, "rolling_accuracy": 0.727, "games": 3, "correct": 3, "cumulative_games": 15},
    {"date": "2026-01-04", "daily_accuracy": 0.60, "rolling_accuracy": 0.700, "games": 5, "correct": 3, "cumulative_games": 20},
    {"date": "2026-01-05", "daily_accuracy": 0.80, "rolling_accuracy": 0.714, "games": 10, "correct": 8, "cumulative_games": 30},
    {"date": "2026-01-06", "daily_accuracy": 0.33, "rolling_accuracy": 0.688, "games": 9, "correct": 3, "cumulative_games": 39},
    {"date": "2026-01-07", "daily_accuracy": 0.70, "rolling_accuracy": 0.692, "games": 10, "correct": 7, "cumulative_games": 49},
    {"date": "2026-01-08", "daily_accuracy": 0.75, "rolling_accuracy": 0.694, "games": 8, "correct": 6, "cumulative_games": 57},
]


def _make_db(path: Path, predictions: list[dict] | None = None, clv_rows: list[float] | None = None) -> None:
    """Create a minimal predictions_history.db for testing."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE game_predictions (
            id INTEGER PRIMARY KEY,
            created_at TEXT,
            game_date TEXT,
            home_team TEXT,
            away_team TEXT,
            home_win_prob REAL,
            away_win_prob REAL,
            model_name TEXT,
            model_artifact TEXT,
            decision_threshold REAL,
            feature_count INTEGER,
            actual_home_win INTEGER,
            notes TEXT
        )
        """
    )
    if predictions:
        conn.executemany(
            """
            INSERT INTO game_predictions
              (game_date, home_team, away_team, home_win_prob, away_win_prob, actual_home_win)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    p["game_date"],
                    p["home_team"],
                    p["away_team"],
                    p["home_win_prob"],
                    p["away_win_prob"],
                    p.get("actual_home_win"),
                )
                for p in predictions
            ],
        )
    if clv_rows is not None:
        conn.execute("CREATE TABLE clv_tracking (id INTEGER PRIMARY KEY, clv REAL)")
        conn.executemany("INSERT INTO clv_tracking (clv) VALUES (?)", [(v,) for v in clv_rows])
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# load_accuracy_history
# ---------------------------------------------------------------------------

def test_load_accuracy_history_missing_file(tmp_path):
    result = load_accuracy_history(tmp_path / "nonexistent.json")
    assert result == []


def test_load_accuracy_history_valid(tmp_path):
    p = tmp_path / "acc.json"
    p.write_text(json.dumps(SAMPLE_HISTORY), encoding="utf-8")
    result = load_accuracy_history(p)
    assert len(result) == len(SAMPLE_HISTORY)
    assert result[0]["date"] == "2026-01-01"


# ---------------------------------------------------------------------------
# compute_season_accuracy
# ---------------------------------------------------------------------------

def test_season_accuracy_empty():
    acc, total = compute_season_accuracy([])
    assert acc is None
    assert total == 0


def test_season_accuracy_correct():
    # 8+4+3+5+10+9+10+8 = 57 games, 6+2+3+3+8+3+7+6 = 38 correct
    acc, total = compute_season_accuracy(SAMPLE_HISTORY)
    assert total == 57
    assert abs(acc - round(38 / 57, 4)) < 1e-6


# ---------------------------------------------------------------------------
# compute_rolling_accuracy_7d
# ---------------------------------------------------------------------------

def test_rolling_accuracy_7d_empty():
    assert compute_rolling_accuracy_7d([]) is None


def test_rolling_accuracy_7d_uses_last_7():
    # last 7 entries: indices 1..7  (correct=2+3+3+8+3+7+6=32, games=4+3+5+10+9+10+8=49)
    result = compute_rolling_accuracy_7d(SAMPLE_HISTORY)
    expected = round(32 / 49, 4)
    assert abs(result - expected) < 1e-6


def test_rolling_accuracy_7d_fewer_than_7():
    short = SAMPLE_HISTORY[:3]
    # games=8+4+3=15, correct=6+2+3=11
    result = compute_rolling_accuracy_7d(short)
    expected = round(11 / 15, 4)
    assert abs(result - expected) < 1e-6


# ---------------------------------------------------------------------------
# compute_streaks
# ---------------------------------------------------------------------------

def test_streaks_empty():
    current, best = compute_streaks([])
    assert current["length"] == 0
    assert best["length"] == 0


def test_streaks_all_wins():
    history = [{"daily_accuracy": 0.8, "games": 5, "correct": 4}] * 6
    current, best = compute_streaks(history)
    assert current == {"type": "W", "length": 6}
    assert best == {"type": "W", "length": 6}


def test_streaks_all_losses():
    history = [{"daily_accuracy": 0.4, "games": 5, "correct": 2}] * 4
    current, best = compute_streaks(history)
    assert current == {"type": "L", "length": 4}
    assert best == {"type": "L", "length": 4}


def test_streaks_mixed():
    # W W L L L W W W  -> current W3, best L3 or W3
    results = [0.8, 0.7, 0.4, 0.3, 0.4, 0.9, 0.8, 0.7]
    history = [{"daily_accuracy": r, "games": 5, "correct": 3} for r in results]
    current, best = compute_streaks(history)
    assert current == {"type": "W", "length": 3}
    assert best["length"] == 3


def test_current_streak_ends_on_loss():
    results = [0.8, 0.8, 0.8, 0.4]
    history = [{"daily_accuracy": r, "games": 5, "correct": 3} for r in results]
    current, _ = compute_streaks(history)
    assert current == {"type": "L", "length": 1}


# ---------------------------------------------------------------------------
# compute_roi_by_market
# ---------------------------------------------------------------------------

def test_roi_no_resolved_predictions(tmp_path):
    db = tmp_path / "pred.db"
    # Predictions with actual_home_win = NULL
    _make_db(db, predictions=[
        {"game_date": "2026-01-01", "home_team": "ORL", "away_team": "DAL",
         "home_win_prob": 0.7, "away_win_prob": 0.3, "actual_home_win": None},
    ])
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    result = compute_roi_by_market(conn)
    conn.close()
    assert result["ML"]["bets"] == 0
    assert result["ML"]["win_pct"] is None


def test_roi_with_resolved_predictions(tmp_path):
    db = tmp_path / "pred.db"
    _make_db(db, predictions=[
        {"game_date": "2026-01-01", "home_team": "ORL", "away_team": "DAL",
         "home_win_prob": 0.7, "away_win_prob": 0.3, "actual_home_win": 1},   # correct
        {"game_date": "2026-01-02", "home_team": "BOS", "away_team": "NYK",
         "home_win_prob": 0.6, "away_win_prob": 0.4, "actual_home_win": 0},   # wrong
        {"game_date": "2026-01-03", "home_team": "MIA", "away_team": "BKN",
         "home_win_prob": 0.9, "away_win_prob": 0.1, "actual_home_win": 1},   # correct
    ])
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    result = compute_roi_by_market(conn)
    conn.close()
    assert result["ML"]["bets"] == 3
    assert result["ML"]["wins"] == 2
    assert abs(result["ML"]["win_pct"] - round(2 / 3, 4)) < 1e-6
    assert result["ATS"]["bets"] == 0


def test_roi_no_table(tmp_path):
    db = tmp_path / "empty.db"
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    result = compute_roi_by_market(conn)
    conn.close()
    assert result["ML"]["bets"] == 0


# ---------------------------------------------------------------------------
# compute_clv_summary
# ---------------------------------------------------------------------------

def test_clv_no_table(tmp_path):
    db = tmp_path / "pred.db"
    _make_db(db)
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    result = compute_clv_summary(conn)
    conn.close()
    assert result["sample_size"] == 0
    assert result["mean_clv"] is None
    assert result["has_edge"] is False


def test_clv_with_data(tmp_path):
    db = tmp_path / "pred.db"
    _make_db(db, clv_rows=[0.5, 1.0, -0.5, 2.0])
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    result = compute_clv_summary(conn)
    conn.close()
    assert result["sample_size"] == 4
    assert abs(result["mean_clv"] - round((0.5 + 1.0 - 0.5 + 2.0) / 4, 4)) < 1e-6
    assert result["has_edge"] is True
    assert abs(result["positive_clv_rate"] - 0.75) < 1e-6


def test_clv_negative_mean(tmp_path):
    db = tmp_path / "pred.db"
    _make_db(db, clv_rows=[-1.0, -2.0])
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    result = compute_clv_summary(conn)
    conn.close()
    assert result["has_edge"] is False
    assert result["mean_clv"] < 0


# ---------------------------------------------------------------------------
# compute_calibration
# ---------------------------------------------------------------------------

def test_calibration_no_resolved(tmp_path):
    db = tmp_path / "pred.db"
    _make_db(db, predictions=[
        {"game_date": "2026-01-01", "home_team": "ORL", "away_team": "DAL",
         "home_win_prob": 0.7, "away_win_prob": 0.3, "actual_home_win": None},
    ])
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    result = compute_calibration(conn)
    conn.close()
    assert result == []


def test_calibration_buckets_correct(tmp_path):
    db = tmp_path / "pred.db"
    # 3 predictions all in 70-75% bucket (home_win_prob 0.70..0.74), all correct
    predictions = [
        {"game_date": "2026-01-01", "home_team": "ORL", "away_team": "DAL",
         "home_win_prob": 0.72, "away_win_prob": 0.28, "actual_home_win": 1},
        {"game_date": "2026-01-02", "home_team": "BOS", "away_team": "NYK",
         "home_win_prob": 0.71, "away_win_prob": 0.29, "actual_home_win": 1},
        {"game_date": "2026-01-03", "home_team": "MIA", "away_team": "BKN",
         "home_win_prob": 0.73, "away_win_prob": 0.27, "actual_home_win": 1},
    ]
    _make_db(db, predictions=predictions)
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    result = compute_calibration(conn)
    conn.close()

    assert len(result) == 1
    bucket = result[0]
    assert bucket["confidence_bucket"] == "70-75%"
    assert bucket["n"] == 3
    assert bucket["actual_pct"] == 1.0


def test_calibration_multiple_buckets(tmp_path):
    db = tmp_path / "pred.db"
    predictions = [
        # 60-65% bucket, home wins (correct)
        {"game_date": "2026-01-01", "home_team": "A", "away_team": "B",
         "home_win_prob": 0.62, "away_win_prob": 0.38, "actual_home_win": 1},
        # 80%+ bucket, away has high prob (away_win_prob=0.85), home loses (actual_home_win=0) -> correct
        {"game_date": "2026-01-02", "home_team": "C", "away_team": "D",
         "home_win_prob": 0.15, "away_win_prob": 0.85, "actual_home_win": 0},
    ]
    _make_db(db, predictions=predictions)
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    result = compute_calibration(conn)
    conn.close()

    labels = [b["confidence_bucket"] for b in result]
    assert "60-65%" in labels
    assert "80%+" in labels
    for bucket in result:
        assert bucket["actual_pct"] == 1.0


# ---------------------------------------------------------------------------
# build_performance end-to-end
# ---------------------------------------------------------------------------

def test_build_performance_full(tmp_path):
    db = tmp_path / "pred.db"
    acc_path = tmp_path / "accuracy_history.json"
    out_path = tmp_path / "performance.json"

    acc_path.write_text(json.dumps(SAMPLE_HISTORY), encoding="utf-8")
    _make_db(db, predictions=[
        {"game_date": "2026-01-01", "home_team": "ORL", "away_team": "DAL",
         "home_win_prob": 0.7, "away_win_prob": 0.3, "actual_home_win": 1},
        {"game_date": "2026-01-02", "home_team": "BOS", "away_team": "NYK",
         "home_win_prob": 0.6, "away_win_prob": 0.4, "actual_home_win": 0},
    ])

    payload = build_performance(db_path=db, accuracy_history_path=acc_path, out_path=out_path)

    # File was written
    assert out_path.exists()
    with open(out_path, encoding="utf-8") as f:
        on_disk = json.load(f)

    # Top-level keys present
    for key in ("season_accuracy", "total_games", "rolling_accuracy_7d",
                 "accuracy_history", "roi_by_market", "clv_summary",
                 "calibration", "current_streak", "best_streak", "last_updated"):
        assert key in on_disk, f"Missing key: {key}"

    # Values are consistent with the fixture
    assert on_disk["total_games"] == 57
    assert on_disk["season_accuracy"] is not None
    assert on_disk["roi_by_market"]["ML"]["bets"] == 2  # both predictions have actual_home_win set
    assert on_disk["clv_summary"]["sample_size"] == 0


def test_build_performance_no_db(tmp_path):
    acc_path = tmp_path / "accuracy_history.json"
    out_path = tmp_path / "performance.json"
    acc_path.write_text(json.dumps(SAMPLE_HISTORY[:3]), encoding="utf-8")

    payload = build_performance(
        db_path=tmp_path / "nonexistent.db",
        accuracy_history_path=acc_path,
        out_path=out_path,
    )
    assert payload["roi_by_market"]["ML"]["bets"] == 0
    assert payload["clv_summary"]["sample_size"] == 0
    assert out_path.exists()


def test_build_performance_empty_history(tmp_path):
    db = tmp_path / "pred.db"
    acc_path = tmp_path / "accuracy_history.json"
    out_path = tmp_path / "performance.json"
    acc_path.write_text("[]", encoding="utf-8")
    _make_db(db)

    payload = build_performance(db_path=db, accuracy_history_path=acc_path, out_path=out_path)
    assert payload["season_accuracy"] is None
    assert payload["total_games"] == 0
    assert payload["rolling_accuracy_7d"] is None
