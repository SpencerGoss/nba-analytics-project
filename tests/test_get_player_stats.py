"""Tests for src/data/get_player_stats.py — player stats fetcher.

Mocks the NBA API boundary (fetch_with_retry) so tests run without network.
"""

import os
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from src.data.get_player_stats import get_player_stats_all_seasons


@pytest.fixture
def sample_player_df():
    """Minimal DataFrame resembling LeagueDashPlayerStats output."""
    return pd.DataFrame({
        "PLAYER_ID": [201939, 203507],
        "PLAYER_NAME": ["Stephen Curry", "Giannis Antetokounmpo"],
        "TEAM_ID": [1610612744, 1610612749],
        "TEAM_ABBREVIATION": ["GSW", "MIL"],
        "GP": [70, 68],
        "PTS": [1820, 2040],
        "AST": [420, 380],
        "REB": [350, 780],
    })


@patch("src.data.get_player_stats.time.sleep")
@patch("src.data.get_player_stats.fetch_with_retry")
def test_single_season_saves_csv(mock_fetch, mock_sleep, sample_player_df, tmp_path):
    """Single successful season produces a CSV file."""
    mock_fetch.return_value = {"success": True, "data": sample_player_df, "error": None}

    with patch("src.data.get_player_stats.os.makedirs"):
        with patch("builtins.print"):
            # Patch the CSV write path
            with patch.object(sample_player_df, "to_csv") as mock_csv:
                mock_fetch.return_value = {"success": True, "data": sample_player_df, "error": None}
                get_player_stats_all_seasons(start_year=2023, end_year=2023)

    mock_fetch.assert_called_once()
    mock_sleep.assert_called_once_with(1)


@patch("src.data.get_player_stats.time.sleep")
@patch("src.data.get_player_stats.fetch_with_retry")
def test_failed_fetch_skips_season(mock_fetch, mock_sleep, tmp_path):
    """Failed API call skips the season without crashing."""
    mock_fetch.return_value = {"success": False, "data": None, "error": "timeout"}

    with patch("src.data.get_player_stats.os.makedirs"):
        with patch("builtins.print"):
            get_player_stats_all_seasons(start_year=2023, end_year=2023)

    mock_fetch.assert_called_once()


@patch("src.data.get_player_stats.time.sleep")
@patch("src.data.get_player_stats.fetch_with_retry")
def test_multiple_seasons_loop(mock_fetch, mock_sleep, sample_player_df):
    """Multiple seasons each trigger a fetch."""
    mock_fetch.return_value = {"success": True, "data": sample_player_df, "error": None}

    with patch("src.data.get_player_stats.os.makedirs"):
        with patch("builtins.print"):
            get_player_stats_all_seasons(start_year=2020, end_year=2023)

    assert mock_fetch.call_count == 4
    assert mock_sleep.call_count == 4


@patch("src.data.get_player_stats.time.sleep")
@patch("src.data.get_player_stats.fetch_with_retry")
def test_season_string_format(mock_fetch, mock_sleep, sample_player_df):
    """Season label passed to fetch_with_retry uses 'YYYY-YY' format."""
    calls = []

    def capture_call(fn, label, **kwargs):
        calls.append(label)
        return {"success": True, "data": sample_player_df, "error": None}

    mock_fetch.side_effect = capture_call

    with patch("src.data.get_player_stats.os.makedirs"):
        with patch("builtins.print"):
            get_player_stats_all_seasons(start_year=2023, end_year=2024)

    assert "2023-24" in calls
    assert "2024-25" in calls


@patch("src.data.get_player_stats.time.sleep")
@patch("src.data.get_player_stats.fetch_with_retry")
def test_creates_output_directory(mock_fetch, mock_sleep, sample_player_df):
    """Output directory data/raw/player_stats is created."""
    mock_fetch.return_value = {"success": True, "data": sample_player_df, "error": None}

    with patch("src.data.get_player_stats.os.makedirs") as mock_mkdir:
        with patch("builtins.print"):
            get_player_stats_all_seasons(start_year=2023, end_year=2023)

    mock_mkdir.assert_called_with("data/raw/player_stats", exist_ok=True)


@patch("src.data.get_player_stats.time.sleep")
@patch("src.data.get_player_stats.fetch_with_retry")
def test_mixed_success_and_failure(mock_fetch, mock_sleep, sample_player_df):
    """Mix of successful and failed fetches handles gracefully."""
    mock_fetch.side_effect = [
        {"success": True, "data": sample_player_df, "error": None},
        {"success": False, "data": None, "error": "rate limited"},
        {"success": True, "data": sample_player_df, "error": None},
    ]

    with patch("src.data.get_player_stats.os.makedirs"):
        with patch("builtins.print"):
            get_player_stats_all_seasons(start_year=2021, end_year=2023)

    assert mock_fetch.call_count == 3
