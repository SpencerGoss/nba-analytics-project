"""
Tests for src/data/get_lineup_data.py

Focuses on:
  - _fetch_team_lineups: returns empty DF on API failure, filters by MIN_GAMES_PLAYED
  - _fetch_team_lineups: renames columns and inserts team identifiers
  - get_lineup_data: saves CSV with expected columns and integer season code
  - get_lineup_data: skips seasons with no data (no CSV written)
"""

import os
import sys
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.get_lineup_data import (
    _fetch_team_lineups,
    get_lineup_data,
    MIN_GAMES_PLAYED,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_api_frame(gp_values: list[int]) -> pd.DataFrame:
    """Return a synthetic TeamDashLineups frame[1]-style DataFrame."""
    rows = []
    for i, gp in enumerate(gp_values):
        rows.append({
            "GROUP_NAME": f"Player{i+1} - Player{i+2} - Player{i+3} - Player{i+4} - Player{i+5}",
            "GP": gp,
            "MIN": 120.0 * gp,
            "NET_RATING": 5.0 + i,
            "OFF_RATING": 110.0 + i,
            "DEF_RATING": 105.0 + i,
            "TS_PCT": 0.55,
            "AST_RATIO": 18.0,
            "OREB_PCT": 0.25,
            "DREB_PCT": 0.75,
            "REB_PCT": 0.50,
        })
    return pd.DataFrame(rows)


# ── _fetch_team_lineups ───────────────────────────────────────────────────────


class TestFetchTeamLineups:

    def test_returns_empty_on_api_failure(self):
        """When fetch_with_retry reports failure, return empty DataFrame."""
        failed_result = {"success": False, "error": "timeout", "data": None}
        with patch(
            "src.data.get_lineup_data.fetch_with_retry",
            return_value=failed_result,
        ):
            result = _fetch_team_lineups(
                team_id=1610612747, team_abbr="LAL", season="2024-25"
            )
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_filters_lineups_below_min_games_played(self):
        """Lineups with GP < MIN_GAMES_PLAYED must be excluded from output."""
        raw = _make_api_frame(gp_values=[MIN_GAMES_PLAYED - 1, MIN_GAMES_PLAYED, 10])
        success_result = {"success": True, "data": raw}
        with patch(
            "src.data.get_lineup_data.fetch_with_retry",
            return_value=success_result,
        ):
            result = _fetch_team_lineups(
                team_id=1610612747, team_abbr="LAL", season="2024-25"
            )
        # Only rows with GP >= MIN_GAMES_PLAYED should be kept
        assert len(result) == 2
        assert "gp" in result.columns
        assert result["gp"].min() >= MIN_GAMES_PLAYED

    def test_output_columns_renamed_and_team_identifiers_inserted(self):
        """Output must have lowercase column names and team_id/team_abbreviation."""
        raw = _make_api_frame(gp_values=[10, 8])
        success_result = {"success": True, "data": raw}
        with patch(
            "src.data.get_lineup_data.fetch_with_retry",
            return_value=success_result,
        ):
            result = _fetch_team_lineups(
                team_id=1610612747, team_abbr="LAL", season="2024-25"
            )

        assert "team_id" in result.columns
        assert "team_abbreviation" in result.columns
        assert "group_name" in result.columns
        assert "net_rating" in result.columns
        assert result["team_id"].iloc[0] == 1610612747
        assert result["team_abbreviation"].iloc[0] == "LAL"
        # Raw uppercase columns must not appear
        assert "GROUP_NAME" not in result.columns
        assert "NET_RATING" not in result.columns

    def test_returns_empty_when_api_returns_empty_frame(self):
        """When the API returns an empty DataFrame, return empty without error."""
        success_result = {"success": True, "data": pd.DataFrame()}
        with patch(
            "src.data.get_lineup_data.fetch_with_retry",
            return_value=success_result,
        ):
            result = _fetch_team_lineups(
                team_id=1610612747, team_abbr="LAL", season="2024-25"
            )
        assert result.empty


# ── get_lineup_data ───────────────────────────────────────────────────────────


class TestGetLineupData:

    def test_saves_csv_with_integer_season_column(self, tmp_path):
        """
        When lineup data is fetched, the saved CSV must contain a 'season'
        column with integer values (e.g. 202425, not '2024-25').
        """
        raw = _make_api_frame(gp_values=[10, 7])
        success_result = {"success": True, "data": raw}

        mock_teams = [
            {"id": 1610612747, "abbreviation": "LAL"},
        ]

        with (
            patch("src.data.get_lineup_data._get_all_team_ids", return_value=mock_teams),
            patch("src.data.get_lineup_data.fetch_with_retry", return_value=success_result),
            patch("src.data.get_lineup_data.time.sleep"),
            patch("src.data.get_lineup_data.RAW_LINEUPS_DIR", str(tmp_path)),
        ):
            get_lineup_data(start_year=2024, end_year=2024)

        output_file = tmp_path / "lineup_data_202425.csv"
        assert output_file.exists(), "Expected lineup_data_202425.csv to be saved"

        df = pd.read_csv(output_file)
        assert "season" in df.columns
        assert pd.api.types.is_integer_dtype(df["season"]) or df["season"].iloc[0] == 202425
        assert int(df["season"].iloc[0]) == 202425

    def test_no_csv_written_when_all_teams_return_empty(self, tmp_path):
        """When all teams return empty data, no CSV file should be written."""
        failed_result = {"success": False, "error": "no data", "data": None}

        mock_teams = [
            {"id": 1610612747, "abbreviation": "LAL"},
            {"id": 1610612738, "abbreviation": "BOS"},
        ]

        with (
            patch("src.data.get_lineup_data._get_all_team_ids", return_value=mock_teams),
            patch("src.data.get_lineup_data.fetch_with_retry", return_value=failed_result),
            patch("src.data.get_lineup_data.time.sleep"),
            patch("src.data.get_lineup_data.RAW_LINEUPS_DIR", str(tmp_path)),
        ):
            get_lineup_data(start_year=2024, end_year=2024)

        csv_files = list(tmp_path.glob("lineup_data_*.csv"))
        assert len(csv_files) == 0, "No CSV should be written when all teams fail"

    def test_output_csv_has_required_columns(self, tmp_path):
        """Saved CSV must contain the documented output schema columns."""
        raw = _make_api_frame(gp_values=[10])
        success_result = {"success": True, "data": raw}

        mock_teams = [{"id": 1610612747, "abbreviation": "LAL"}]

        with (
            patch("src.data.get_lineup_data._get_all_team_ids", return_value=mock_teams),
            patch("src.data.get_lineup_data.fetch_with_retry", return_value=success_result),
            patch("src.data.get_lineup_data.time.sleep"),
            patch("src.data.get_lineup_data.RAW_LINEUPS_DIR", str(tmp_path)),
        ):
            get_lineup_data(start_year=2024, end_year=2024)

        output_file = tmp_path / "lineup_data_202425.csv"
        df = pd.read_csv(output_file)

        required_cols = [
            "season", "team_id", "team_abbreviation", "group_name",
            "gp", "min", "net_rating",
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"


    def test_multiple_teams_all_included(self, tmp_path):
        """When two teams return data, both appear in the output CSV."""
        raw = _make_api_frame(gp_values=[10])
        success_result = {"success": True, "data": raw}

        mock_teams = [
            {"id": 1610612747, "abbreviation": "LAL"},
            {"id": 1610612738, "abbreviation": "BOS"},
        ]

        with (
            patch("src.data.get_lineup_data._get_all_team_ids", return_value=mock_teams),
            patch("src.data.get_lineup_data.fetch_with_retry", return_value=success_result),
            patch("src.data.get_lineup_data.time.sleep"),
            patch("src.data.get_lineup_data.RAW_LINEUPS_DIR", str(tmp_path)),
        ):
            get_lineup_data(start_year=2024, end_year=2024)

        output_file = tmp_path / "lineup_data_202425.csv"
        df = pd.read_csv(output_file)
        teams_found = set(df["team_abbreviation"].tolist())
        assert "LAL" in teams_found
        assert "BOS" in teams_found

    def test_all_below_min_gp_no_csv_written(self, tmp_path):
        """When all lineups are below MIN_GAMES_PLAYED, no CSV should be written."""
        raw = _make_api_frame(gp_values=[MIN_GAMES_PLAYED - 1, MIN_GAMES_PLAYED - 2])
        success_result = {"success": True, "data": raw}

        mock_teams = [{"id": 1610612747, "abbreviation": "LAL"}]

        with (
            patch("src.data.get_lineup_data._get_all_team_ids", return_value=mock_teams),
            patch("src.data.get_lineup_data.fetch_with_retry", return_value=success_result),
            patch("src.data.get_lineup_data.time.sleep"),
            patch("src.data.get_lineup_data.RAW_LINEUPS_DIR", str(tmp_path)),
        ):
            get_lineup_data(start_year=2024, end_year=2024)

        csv_files = list(tmp_path.glob("lineup_data_*.csv"))
        assert len(csv_files) == 0

    def test_multi_season_writes_separate_files(self, tmp_path):
        """get_lineup_data for 2 seasons should write 2 separate CSV files."""
        raw = _make_api_frame(gp_values=[10])
        success_result = {"success": True, "data": raw}

        mock_teams = [{"id": 1610612747, "abbreviation": "LAL"}]

        with (
            patch("src.data.get_lineup_data._get_all_team_ids", return_value=mock_teams),
            patch("src.data.get_lineup_data.fetch_with_retry", return_value=success_result),
            patch("src.data.get_lineup_data.time.sleep"),
            patch("src.data.get_lineup_data.RAW_LINEUPS_DIR", str(tmp_path)),
        ):
            get_lineup_data(start_year=2023, end_year=2024)

        csv_files = list(tmp_path.glob("lineup_data_*.csv"))
        assert len(csv_files) == 2
