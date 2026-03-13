"""
Tests for src/models/clv_tracker.py

Covers:
  - CLVTracker.log_opening_line: inserts row, skips duplicate, returns True/False
  - CLVTracker.update_closing_line: computes CLV, handles missing opening, NULL opening
  - CLVTracker.get_clv_summary: empty DB, single game, has_edge threshold
  - CLV formula: opening - closing (positive = better than closing)
  - Code quality: no deprecated utcnow(), uses cursor.rowcount not conn.total_changes
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.clv_tracker import CLVTracker


# ---------------------------------------------------------------------------
# Code quality checks
# ---------------------------------------------------------------------------

def test_clv_tracker_no_utcnow():
    """Must not use deprecated datetime.utcnow()."""
    from src.models import clv_tracker
    source = inspect.getsource(clv_tracker)
    assert "utcnow()" not in source, "clv_tracker.py still uses deprecated utcnow()"


def test_clv_tracker_uses_cursor_rowcount():
    """Must use cursor.rowcount, not conn.total_changes."""
    from src.models import clv_tracker
    source = inspect.getsource(clv_tracker)
    assert "total_changes" not in source, "clv_tracker.py still uses unreliable conn.total_changes"
    assert "cursor.rowcount" in source


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tracker(tmp_path):
    """CLVTracker backed by an isolated tmp_path database."""
    db = tmp_path / "test_predictions.db"
    return CLVTracker(db_path=db)


# ---------------------------------------------------------------------------
# log_opening_line
# ---------------------------------------------------------------------------

class TestLogOpeningLine:
    def test_insert_returns_true(self, tracker):
        result = tracker.log_opening_line("2026-01-15", "LAL", "GSW", -3.5)
        assert result is True

    def test_duplicate_returns_false(self, tracker):
        tracker.log_opening_line("2026-01-15", "LAL", "GSW", -3.5)
        result = tracker.log_opening_line("2026-01-15", "LAL", "GSW", -3.5)
        assert result is False

    def test_different_games_both_insert(self, tracker):
        r1 = tracker.log_opening_line("2026-01-15", "LAL", "GSW", -3.5)
        r2 = tracker.log_opening_line("2026-01-16", "BOS", "NYK", -5.0)
        assert r1 is True
        assert r2 is True

    def test_moneylines_stored(self, tracker):
        tracker.log_opening_line("2026-01-15", "LAL", "GSW", -3.5, -165, 140)
        summary = tracker.get_clv_summary()
        # No CLV yet (no closing line), but game exists
        assert summary["n_games"] == 0  # no CLV computed yet

    def test_none_spread_accepted(self, tracker):
        result = tracker.log_opening_line("2026-01-15", "LAL", "GSW", None)
        assert result is True


# ---------------------------------------------------------------------------
# update_closing_line
# ---------------------------------------------------------------------------

class TestUpdateClosingLine:
    def test_positive_clv_when_opening_better_than_closing(self, tracker):
        """opening=-3.5, closing=-5.5 -> CLV=+2.0 (easier cover than market settled)."""
        tracker.log_opening_line("2026-01-15", "LAL", "GSW", -3.5)
        clv = tracker.update_closing_line("2026-01-15", "LAL", "GSW", -5.5)
        assert clv == pytest.approx(2.0)

    def test_negative_clv_when_opening_worse_than_closing(self, tracker):
        """opening=-5.5, closing=-3.5 -> CLV=-2.0 (harder cover than market)."""
        tracker.log_opening_line("2026-01-15", "LAL", "GSW", -5.5)
        clv = tracker.update_closing_line("2026-01-15", "LAL", "GSW", -3.5)
        assert clv == pytest.approx(-2.0)

    def test_zero_clv_when_opening_equals_closing(self, tracker):
        tracker.log_opening_line("2026-01-15", "LAL", "GSW", -3.5)
        clv = tracker.update_closing_line("2026-01-15", "LAL", "GSW", -3.5)
        assert clv == pytest.approx(0.0)

    def test_returns_none_when_no_opening_logged(self, tracker):
        """update_closing_line on a game with no opening logged returns None."""
        clv = tracker.update_closing_line("2026-01-15", "LAL", "GSW", -3.5)
        assert clv is None

    def test_returns_none_when_opening_spread_is_null(self, tracker):
        tracker.log_opening_line("2026-01-15", "LAL", "GSW", None)
        clv = tracker.update_closing_line("2026-01-15", "LAL", "GSW", -3.5)
        assert clv is None

    def test_clv_rounded_to_2_decimals(self, tracker):
        tracker.log_opening_line("2026-01-15", "LAL", "GSW", -3.5)
        clv = tracker.update_closing_line("2026-01-15", "LAL", "GSW", -5.75)
        assert clv == pytest.approx(2.25)


# ---------------------------------------------------------------------------
# get_clv_summary
# ---------------------------------------------------------------------------

class TestGetClvSummary:
    def test_empty_db_returns_zero_games(self, tracker):
        summary = tracker.get_clv_summary()
        assert summary["n_games"] == 0
        assert summary["has_edge"] is False
        assert summary["mean_clv"] is None

    def test_single_game_with_clv(self, tracker):
        tracker.log_opening_line("2026-01-15", "LAL", "GSW", -3.5)
        tracker.update_closing_line("2026-01-15", "LAL", "GSW", -5.5)
        summary = tracker.get_clv_summary()
        assert summary["n_games"] == 1
        assert summary["mean_clv"] == pytest.approx(2.0)

    def test_mean_clv_is_average(self, tracker):
        tracker.log_opening_line("2026-01-15", "LAL", "GSW", -3.5)
        tracker.update_closing_line("2026-01-15", "LAL", "GSW", -5.5)  # CLV=+2.0
        tracker.log_opening_line("2026-01-16", "BOS", "NYK", -4.0)
        tracker.update_closing_line("2026-01-16", "BOS", "NYK", -2.0)  # CLV=-2.0
        summary = tracker.get_clv_summary()
        assert summary["n_games"] == 2
        assert summary["mean_clv"] == pytest.approx(0.0)

    def test_positive_clv_rate(self, tracker):
        for i, (opening, closing) in enumerate([(-3.5, -5.5), (-4.0, -6.0), (-5.0, -3.0)]):
            tracker.log_opening_line(f"2026-01-{i+15:02d}", "LAL", "GSW", opening)
            tracker.update_closing_line(f"2026-01-{i+15:02d}", "LAL", "GSW", closing)
        summary = tracker.get_clv_summary()
        # 2 positive CLVs, 1 negative -> pos_rate = 2/3
        assert summary["positive_clv_rate"] == pytest.approx(2/3, abs=0.001)

    def test_has_edge_false_when_too_few_games(self, tracker):
        """has_edge requires >= 10 games by default."""
        for i in range(5):
            tracker.log_opening_line(f"2026-01-{i+15:02d}", "LAL", "GSW", -3.5)
            tracker.update_closing_line(f"2026-01-{i+15:02d}", "LAL", "GSW", -5.5)
        summary = tracker.get_clv_summary()
        # 5 games with positive CLV is still < 10 min_games
        assert summary["has_edge"] is False

    def test_has_edge_true_when_sufficient_games_and_positive_mean(self, tracker):
        """has_edge requires >= 10 games, mean_clv > 0, pos_rate > 0.5."""
        for i in range(10):
            tracker.log_opening_line(f"2026-01-{i+10:02d}", "LAL", "GSW", -3.5)
            tracker.update_closing_line(f"2026-01-{i+10:02d}", "LAL", "GSW", -5.5)
        summary = tracker.get_clv_summary()
        assert summary["has_edge"] is True

    def test_data_list_returned(self, tracker):
        tracker.log_opening_line("2026-01-15", "LAL", "GSW", -3.5)
        tracker.update_closing_line("2026-01-15", "LAL", "GSW", -5.5)
        summary = tracker.get_clv_summary()
        assert isinstance(summary["data"], list)
        assert len(summary["data"]) == 1
        assert "clv" in summary["data"][0]

    def test_game_without_closing_line_not_in_clv_data(self, tracker):
        """Games with only opening (no closing) should not appear in n_games count."""
        tracker.log_opening_line("2026-01-15", "LAL", "GSW", -3.5)  # no closing
        summary = tracker.get_clv_summary()
        assert summary["n_games"] == 0

    def test_summary_has_required_keys(self, tracker):
        """get_clv_summary must always return all documented keys."""
        summary = tracker.get_clv_summary()
        for key in ("n_games", "mean_clv", "has_edge", "positive_clv_rate", "data"):
            assert key in summary, f"Missing key: {key}"

    def test_data_row_has_required_fields(self, tracker):
        """Each entry in summary['data'] must have game_date, home_team, away_team, clv."""
        tracker.log_opening_line("2026-01-15", "LAL", "GSW", -3.5)
        tracker.update_closing_line("2026-01-15", "LAL", "GSW", -5.5)
        summary = tracker.get_clv_summary()
        row = summary["data"][0]
        for field in ("game_date", "home_team", "away_team", "clv"):
            assert field in row, f"Missing field in data row: {field}"

    def test_clv_value_is_float(self, tracker):
        """CLV returned by update_closing_line must be a float."""
        tracker.log_opening_line("2026-01-15", "LAL", "GSW", -3.5)
        clv = tracker.update_closing_line("2026-01-15", "LAL", "GSW", -5.5)
        assert isinstance(clv, float)


# ---------------------------------------------------------------------------
# backfill_closing_lines
# ---------------------------------------------------------------------------

from datetime import date

import pandas as pd

from src.models.clv_tracker import backfill_closing_lines


class TestBackfillClosingLines:
    """Tests for backfill_closing_lines() which reads game_lines.csv and
    updates closing spreads for completed games."""

    def _write_game_lines(self, path, rows):
        """Helper: write a game_lines.csv file."""
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)

    def test_backfills_past_game(self, tmp_path):
        """Past game with opening line but no closing -> gets backfilled."""
        db = tmp_path / "test.db"
        csv = tmp_path / "game_lines.csv"

        tracker = CLVTracker(db_path=db)
        tracker.log_opening_line("2026-01-15", "LAL", "GSW", -3.5)

        self._write_game_lines(csv, [{
            "date": "2026-01-15",
            "home_team": "LAL",
            "away_team": "GSW",
            "home_moneyline": -165,
            "away_moneyline": 140,
            "spread": -5.5,
            "total": 220.0,
        }])

        n = backfill_closing_lines(
            db_path=db,
            game_lines_path=csv,
            today=date(2026, 1, 16),
        )
        assert n == 1

        # Verify CLV was computed correctly
        summary = tracker.get_clv_summary()
        assert summary["n_games"] == 1
        assert summary["data"][0]["clv"] == pytest.approx(2.0)

    def test_skips_future_games(self, tmp_path):
        """Games on or after today should NOT be backfilled."""
        db = tmp_path / "test.db"
        csv = tmp_path / "game_lines.csv"

        tracker = CLVTracker(db_path=db)
        tracker.log_opening_line("2026-01-20", "LAL", "GSW", -3.5)

        self._write_game_lines(csv, [{
            "date": "2026-01-20",
            "home_team": "LAL",
            "away_team": "GSW",
            "home_moneyline": -165,
            "away_moneyline": 140,
            "spread": -5.5,
            "total": 220.0,
        }])

        n = backfill_closing_lines(
            db_path=db,
            game_lines_path=csv,
            today=date(2026, 1, 20),
        )
        assert n == 0

    def test_skips_already_closed_games(self, tmp_path):
        """Games that already have a closing line should not be updated again."""
        db = tmp_path / "test.db"
        csv = tmp_path / "game_lines.csv"

        tracker = CLVTracker(db_path=db)
        tracker.log_opening_line("2026-01-15", "LAL", "GSW", -3.5)
        tracker.update_closing_line("2026-01-15", "LAL", "GSW", -4.0)

        self._write_game_lines(csv, [{
            "date": "2026-01-15",
            "home_team": "LAL",
            "away_team": "GSW",
            "home_moneyline": -165,
            "away_moneyline": 140,
            "spread": -5.5,
            "total": 220.0,
        }])

        n = backfill_closing_lines(
            db_path=db,
            game_lines_path=csv,
            today=date(2026, 1, 16),
        )
        assert n == 0

        # Original closing line should be preserved
        summary = tracker.get_clv_summary()
        assert summary["data"][0]["closing_spread"] == -4.0

    def test_skips_games_without_opening_line(self, tmp_path):
        """If no opening line was logged, backfill should skip the game."""
        db = tmp_path / "test.db"
        csv = tmp_path / "game_lines.csv"

        # Create tracker but do NOT log an opening line
        CLVTracker(db_path=db)

        self._write_game_lines(csv, [{
            "date": "2026-01-15",
            "home_team": "LAL",
            "away_team": "GSW",
            "home_moneyline": -165,
            "away_moneyline": 140,
            "spread": -5.5,
            "total": 220.0,
        }])

        n = backfill_closing_lines(
            db_path=db,
            game_lines_path=csv,
            today=date(2026, 1, 16),
        )
        assert n == 0

    def test_skips_null_spread(self, tmp_path):
        """Rows with NaN/null spread should be skipped."""
        db = tmp_path / "test.db"
        csv = tmp_path / "game_lines.csv"

        tracker = CLVTracker(db_path=db)
        tracker.log_opening_line("2026-01-15", "LAL", "GSW", -3.5)

        self._write_game_lines(csv, [{
            "date": "2026-01-15",
            "home_team": "LAL",
            "away_team": "GSW",
            "home_moneyline": -165,
            "away_moneyline": 140,
            "spread": None,
            "total": 220.0,
        }])

        n = backfill_closing_lines(
            db_path=db,
            game_lines_path=csv,
            today=date(2026, 1, 16),
        )
        assert n == 0

    def test_returns_zero_when_csv_missing(self, tmp_path):
        """Missing game_lines.csv should return 0 without error."""
        db = tmp_path / "test.db"
        CLVTracker(db_path=db)
        n = backfill_closing_lines(
            db_path=db,
            game_lines_path=tmp_path / "nonexistent.csv",
            today=date(2026, 1, 16),
        )
        assert n == 0

    def test_returns_zero_when_csv_empty(self, tmp_path):
        """Empty game_lines.csv should return 0."""
        db = tmp_path / "test.db"
        csv = tmp_path / "game_lines.csv"
        CLVTracker(db_path=db)

        # Write header-only CSV
        pd.DataFrame(columns=[
            "date", "home_team", "away_team", "home_moneyline",
            "away_moneyline", "spread", "total"
        ]).to_csv(csv, index=False)

        n = backfill_closing_lines(
            db_path=db,
            game_lines_path=csv,
            today=date(2026, 1, 16),
        )
        assert n == 0

    def test_multiple_games_backfilled(self, tmp_path):
        """Multiple past games should all get closing lines."""
        db = tmp_path / "test.db"
        csv = tmp_path / "game_lines.csv"

        tracker = CLVTracker(db_path=db)
        tracker.log_opening_line("2026-01-15", "LAL", "GSW", -3.5)
        tracker.log_opening_line("2026-01-15", "BOS", "NYK", -5.0)

        self._write_game_lines(csv, [
            {
                "date": "2026-01-15", "home_team": "LAL", "away_team": "GSW",
                "home_moneyline": -165, "away_moneyline": 140,
                "spread": -5.5, "total": 220.0,
            },
            {
                "date": "2026-01-15", "home_team": "BOS", "away_team": "NYK",
                "home_moneyline": -200, "away_moneyline": 175,
                "spread": -6.0, "total": 215.0,
            },
        ])

        n = backfill_closing_lines(
            db_path=db,
            game_lines_path=csv,
            today=date(2026, 1, 16),
        )
        assert n == 2

        summary = tracker.get_clv_summary()
        assert summary["n_games"] == 2
