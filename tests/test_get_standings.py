"""Tests for src/data/get_standings.py -- standings fetcher.

Mocks the NBA API boundary (fetch_with_retry) so tests run without network.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from src.data.get_standings import get_standings


@pytest.fixture
def sample_standings_df():
    """Minimal DataFrame resembling LeagueStandingsV3 output."""
    return pd.DataFrame({
        "TeamID": [1610612738, 1610612747],
        "TeamCity": ["Boston", "Los Angeles"],
        "TeamName": ["Celtics", "Lakers"],
        "TeamSlug": ["celtics", "lakers"],
        "Conference": ["East", "West"],
        "ConferenceRecord": ["36-16", "28-24"],
        "WINS": [64, 47],
        "LOSSES": [18, 35],
        "WinPCT": [0.780, 0.573],
        "Record": ["64-18", "47-35"],
    })


@patch("src.data.get_standings.time.sleep")
@patch("src.data.get_standings.fetch_with_retry")
def test_single_season_fetches_and_saves(mock_fetch, mock_sleep, sample_standings_df):
    """Single season calls fetch_with_retry once and sleeps."""
    mock_fetch.return_value = {"success": True, "data": sample_standings_df, "error": None}

    with patch("src.data.get_standings.os.makedirs"):
        with patch("builtins.print"):
            get_standings(start_year=2023, end_year=2023)

    mock_fetch.assert_called_once()
    mock_sleep.assert_called_once_with(1)


@patch("src.data.get_standings.time.sleep")
@patch("src.data.get_standings.fetch_with_retry")
def test_failed_fetch_skips(mock_fetch, mock_sleep):
    """Failed API call skips without error."""
    mock_fetch.return_value = {"success": False, "data": None, "error": "timeout"}

    with patch("src.data.get_standings.os.makedirs"):
        with patch("builtins.print"):
            get_standings(start_year=2023, end_year=2023)

    mock_fetch.assert_called_once()


@patch("src.data.get_standings.time.sleep")
@patch("src.data.get_standings.fetch_with_retry")
def test_multi_season_loop(mock_fetch, mock_sleep, sample_standings_df):
    """Four seasons produce four fetch calls."""
    mock_fetch.return_value = {"success": True, "data": sample_standings_df, "error": None}

    with patch("src.data.get_standings.os.makedirs"):
        with patch("builtins.print"):
            get_standings(start_year=2020, end_year=2023)

    assert mock_fetch.call_count == 4
    assert mock_sleep.call_count == 4


@patch("src.data.get_standings.time.sleep")
@patch("src.data.get_standings.fetch_with_retry")
def test_season_format_in_label(mock_fetch, mock_sleep, sample_standings_df):
    """Season labels use 'YYYY-YY' format."""
    labels = []

    def capture(fn, label, **kwargs):
        labels.append(label)
        return {"success": True, "data": sample_standings_df, "error": None}

    mock_fetch.side_effect = capture

    with patch("src.data.get_standings.os.makedirs"):
        with patch("builtins.print"):
            get_standings(start_year=2022, end_year=2023)

    assert labels == ["2022-23", "2023-24"]


@patch("src.data.get_standings.time.sleep")
@patch("src.data.get_standings.fetch_with_retry")
def test_creates_output_directory(mock_fetch, mock_sleep, sample_standings_df):
    """Output directory data/raw/standings is created."""
    mock_fetch.return_value = {"success": True, "data": sample_standings_df, "error": None}

    with patch("src.data.get_standings.os.makedirs") as mock_mkdir:
        with patch("builtins.print"):
            get_standings(start_year=2023, end_year=2023)

    mock_mkdir.assert_called_with("data/raw/standings", exist_ok=True)


@patch("src.data.get_standings.time.sleep")
@patch("src.data.get_standings.fetch_with_retry")
def test_csv_filename_format(mock_fetch, mock_sleep, sample_standings_df):
    """CSV filename strips the dash from season string (e.g. standings_202324.csv)."""
    mock_fetch.return_value = {"success": True, "data": sample_standings_df, "error": None}

    with patch("src.data.get_standings.os.makedirs"):
        with patch("builtins.print"):
            with patch.object(sample_standings_df, "to_csv") as mock_csv:
                get_standings(start_year=2023, end_year=2023)

    mock_csv.assert_called_once_with(
        "data/raw/standings/standings_202324.csv", index=False
    )


@patch("src.data.get_standings.time.sleep")
@patch("src.data.get_standings.fetch_with_retry")
def test_mixed_results(mock_fetch, mock_sleep, sample_standings_df):
    """Handles mix of success/failure without crashing."""
    mock_fetch.side_effect = [
        {"success": False, "data": None, "error": "503"},
        {"success": True, "data": sample_standings_df, "error": None},
        {"success": True, "data": sample_standings_df, "error": None},
    ]

    with patch("src.data.get_standings.os.makedirs"):
        with patch("builtins.print"):
            get_standings(start_year=2021, end_year=2023)

    assert mock_fetch.call_count == 3


@patch("src.data.get_standings.time.sleep")
@patch("src.data.get_standings.fetch_with_retry")
def test_century_boundary_season_format(mock_fetch, mock_sleep, sample_standings_df):
    """Season spanning century boundary (1999-2000) uses correct format '1999-00'."""
    labels = []

    def capture(fn, label, **kwargs):
        labels.append(label)
        return {"success": True, "data": sample_standings_df, "error": None}

    mock_fetch.side_effect = capture

    with patch("src.data.get_standings.os.makedirs"):
        with patch("builtins.print"):
            get_standings(start_year=1999, end_year=1999)

    assert labels == ["1999-00"]
