"""Tests for src/data/get_team_stats.py — team stats fetcher.

Mocks the NBA API boundary (fetch_with_retry) so tests run without network.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from src.data.get_team_stats import get_team_stats_all_seasons


@pytest.fixture
def sample_team_df():
    """Minimal DataFrame resembling LeagueDashTeamStats output."""
    return pd.DataFrame({
        "TEAM_ID": [1610612738, 1610612747],
        "TEAM_NAME": ["Boston Celtics", "Los Angeles Lakers"],
        "TEAM_ABBREVIATION": ["BOS", "LAL"],
        "GP": [82, 82],
        "W": [64, 47],
        "L": [18, 35],
        "W_PCT": [0.780, 0.573],
        "PTS": [9350, 8940],
        "OFF_RATING": [118.2, 114.5],
        "DEF_RATING": [109.1, 112.3],
    })


@patch("src.data.get_team_stats.time.sleep")
@patch("src.data.get_team_stats.fetch_with_retry")
def test_single_season_fetches_and_saves(mock_fetch, mock_sleep, sample_team_df):
    """Single season calls fetch_with_retry once."""
    mock_fetch.return_value = {"success": True, "data": sample_team_df, "error": None}

    with patch("src.data.get_team_stats.os.makedirs"):
        with patch("builtins.print"):
            get_team_stats_all_seasons(start_year=2023, end_year=2023)

    mock_fetch.assert_called_once()
    mock_sleep.assert_called_once_with(1)


@patch("src.data.get_team_stats.time.sleep")
@patch("src.data.get_team_stats.fetch_with_retry")
def test_failed_fetch_skips(mock_fetch, mock_sleep):
    """Failed API call skips without error."""
    mock_fetch.return_value = {"success": False, "data": None, "error": "timeout"}

    with patch("src.data.get_team_stats.os.makedirs"):
        with patch("builtins.print"):
            get_team_stats_all_seasons(start_year=2023, end_year=2023)

    mock_fetch.assert_called_once()


@patch("src.data.get_team_stats.time.sleep")
@patch("src.data.get_team_stats.fetch_with_retry")
def test_multi_season_loop(mock_fetch, mock_sleep, sample_team_df):
    """Three seasons produce three fetch calls."""
    mock_fetch.return_value = {"success": True, "data": sample_team_df, "error": None}

    with patch("src.data.get_team_stats.os.makedirs"):
        with patch("builtins.print"):
            get_team_stats_all_seasons(start_year=2021, end_year=2023)

    assert mock_fetch.call_count == 3
    assert mock_sleep.call_count == 3


@patch("src.data.get_team_stats.time.sleep")
@patch("src.data.get_team_stats.fetch_with_retry")
def test_season_format_in_label(mock_fetch, mock_sleep, sample_team_df):
    """Season labels use 'YYYY-YY' format."""
    labels = []

    def capture(fn, label, **kwargs):
        labels.append(label)
        return {"success": True, "data": sample_team_df, "error": None}

    mock_fetch.side_effect = capture

    with patch("src.data.get_team_stats.os.makedirs"):
        with patch("builtins.print"):
            get_team_stats_all_seasons(start_year=2022, end_year=2023)

    assert labels == ["2022-23", "2023-24"]


@patch("src.data.get_team_stats.time.sleep")
@patch("src.data.get_team_stats.fetch_with_retry")
def test_creates_output_directory(mock_fetch, mock_sleep, sample_team_df):
    """Output directory data/raw/team_stats is created."""
    mock_fetch.return_value = {"success": True, "data": sample_team_df, "error": None}

    with patch("src.data.get_team_stats.os.makedirs") as mock_mkdir:
        with patch("builtins.print"):
            get_team_stats_all_seasons(start_year=2023, end_year=2023)

    mock_mkdir.assert_called_with("data/raw/team_stats", exist_ok=True)


@patch("src.data.get_team_stats.time.sleep")
@patch("src.data.get_team_stats.fetch_with_retry")
def test_mixed_results(mock_fetch, mock_sleep, sample_team_df):
    """Handles mix of success/failure without crashing."""
    mock_fetch.side_effect = [
        {"success": False, "data": None, "error": "503"},
        {"success": True, "data": sample_team_df, "error": None},
    ]

    with patch("src.data.get_team_stats.os.makedirs"):
        with patch("builtins.print"):
            get_team_stats_all_seasons(start_year=2022, end_year=2023)

    assert mock_fetch.call_count == 2
