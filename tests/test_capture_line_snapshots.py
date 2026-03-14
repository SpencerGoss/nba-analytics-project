"""Tests for scripts/capture_line_snapshots.py."""
import pandas as pd
import pytest
from unittest.mock import patch
from pathlib import Path

from scripts.capture_line_snapshots import (
    append_snapshot,
    compute_line_movement,
    SNAPSHOT_COLUMNS,
)


@pytest.fixture
def sample_snapshot():
    """A minimal snapshot DataFrame."""
    return pd.DataFrame({
        "snapshot_ts": ["2026-03-13T14:00:00Z", "2026-03-13T14:00:00Z"],
        "game_date": ["2026-03-13", "2026-03-13"],
        "home_team": ["BOS", "LAL"],
        "away_team": ["NYK", "GSW"],
        "spread": [-5.5, 3.0],
        "home_moneyline": [-220, 130],
        "away_moneyline": [180, -155],
        "total": [218.5, 225.0],
    })


class TestAppendSnapshot:
    def test_creates_file_with_header(self, tmp_path, sample_snapshot):
        csv_path = tmp_path / "snapshots.csv"
        with patch("scripts.capture_line_snapshots.SNAPSHOTS_CSV", csv_path):
            n = append_snapshot(sample_snapshot)
        assert n == 2
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert list(df.columns) == SNAPSHOT_COLUMNS
        assert len(df) == 2

    def test_appends_without_duplicate_header(self, tmp_path, sample_snapshot):
        csv_path = tmp_path / "snapshots.csv"
        with patch("scripts.capture_line_snapshots.SNAPSHOTS_CSV", csv_path):
            append_snapshot(sample_snapshot)
            append_snapshot(sample_snapshot)
        df = pd.read_csv(csv_path)
        assert len(df) == 4

    def test_empty_snapshot_writes_nothing(self, tmp_path):
        csv_path = tmp_path / "snapshots.csv"
        empty = pd.DataFrame(columns=SNAPSHOT_COLUMNS)
        with patch("scripts.capture_line_snapshots.SNAPSHOTS_CSV", csv_path):
            n = append_snapshot(empty)
        assert n == 0
        assert not csv_path.exists()


class TestComputeLineMovement:
    def test_no_file_returns_empty(self, tmp_path):
        with patch("scripts.capture_line_snapshots.SNAPSHOTS_CSV", tmp_path / "missing.csv"):
            result = compute_line_movement()
        assert result.empty

    def test_single_snapshot_no_movement(self, tmp_path):
        csv_path = tmp_path / "snapshots.csv"
        df = pd.DataFrame({
            "snapshot_ts": ["2026-03-13T10:00:00Z"],
            "game_date": ["2026-03-13"],
            "home_team": ["BOS"],
            "away_team": ["NYK"],
            "spread": [-5.5],
            "home_moneyline": [-220],
            "away_moneyline": [180],
            "total": [218.5],
        })
        df.to_csv(csv_path, index=False)
        with patch("scripts.capture_line_snapshots.SNAPSHOTS_CSV", csv_path):
            result = compute_line_movement()
        assert len(result) == 1
        assert result.iloc[0]["spread_move"] == 0.0
        assert result.iloc[0]["is_reverse_line_movement"] == False

    def test_detects_reverse_line_movement(self, tmp_path):
        """Home is favored (-5.5), line moves toward away (+1.0) -> RLM."""
        csv_path = tmp_path / "snapshots.csv"
        df = pd.DataFrame({
            "snapshot_ts": [
                "2026-03-13T10:00:00Z",
                "2026-03-13T18:00:00Z",
            ],
            "game_date": ["2026-03-13", "2026-03-13"],
            "home_team": ["BOS", "BOS"],
            "away_team": ["NYK", "NYK"],
            "spread": [-5.5, -4.5],  # moved 1 point toward away
            "home_moneyline": [-220, -200],
            "away_moneyline": [180, 170],
            "total": [218.5, 219.0],
        })
        df.to_csv(csv_path, index=False)
        with patch("scripts.capture_line_snapshots.SNAPSHOTS_CSV", csv_path):
            result = compute_line_movement()
        assert len(result) == 1
        row = result.iloc[0]
        assert row["opening_spread"] == -5.5
        assert row["latest_spread"] == -4.5
        assert row["spread_move"] == 1.0
        assert row["is_reverse_line_movement"] == True

    def test_no_rlm_on_small_move(self, tmp_path):
        """Moves less than 0.5 should not flag as RLM."""
        csv_path = tmp_path / "snapshots.csv"
        df = pd.DataFrame({
            "snapshot_ts": [
                "2026-03-13T10:00:00Z",
                "2026-03-13T18:00:00Z",
            ],
            "game_date": ["2026-03-13", "2026-03-13"],
            "home_team": ["BOS", "BOS"],
            "away_team": ["NYK", "NYK"],
            "spread": [-5.5, -5.5],  # no movement
            "home_moneyline": [-220, -220],
            "away_moneyline": [180, 180],
            "total": [218.5, 218.5],
        })
        df.to_csv(csv_path, index=False)
        with patch("scripts.capture_line_snapshots.SNAPSHOTS_CSV", csv_path):
            result = compute_line_movement()
        assert result.iloc[0]["is_reverse_line_movement"] == False

    def test_filter_by_game_date(self, tmp_path):
        csv_path = tmp_path / "snapshots.csv"
        df = pd.DataFrame({
            "snapshot_ts": [
                "2026-03-13T10:00:00Z",
                "2026-03-14T10:00:00Z",
            ],
            "game_date": ["2026-03-13", "2026-03-14"],
            "home_team": ["BOS", "LAL"],
            "away_team": ["NYK", "GSW"],
            "spread": [-5.5, 2.0],
            "home_moneyline": [-220, 110],
            "away_moneyline": [180, -130],
            "total": [218.5, 225.0],
        })
        df.to_csv(csv_path, index=False)
        with patch("scripts.capture_line_snapshots.SNAPSHOTS_CSV", csv_path):
            result = compute_line_movement(game_date="2026-03-13")
        assert len(result) == 1
        assert result.iloc[0]["home_team"] == "BOS"

    def test_n_snapshots_counted(self, tmp_path):
        csv_path = tmp_path / "snapshots.csv"
        df = pd.DataFrame({
            "snapshot_ts": [
                "2026-03-13T10:00:00Z",
                "2026-03-13T13:00:00Z",
                "2026-03-13T16:00:00Z",
                "2026-03-13T18:00:00Z",
            ],
            "game_date": ["2026-03-13"] * 4,
            "home_team": ["BOS"] * 4,
            "away_team": ["NYK"] * 4,
            "spread": [-5.5, -5.0, -4.5, -4.0],
            "home_moneyline": [-220, -210, -200, -190],
            "away_moneyline": [180, 175, 170, 165],
            "total": [218.5, 219.0, 219.5, 220.0],
        })
        df.to_csv(csv_path, index=False)
        with patch("scripts.capture_line_snapshots.SNAPSHOTS_CSV", csv_path):
            result = compute_line_movement()
        assert result.iloc[0]["n_snapshots"] == 4
