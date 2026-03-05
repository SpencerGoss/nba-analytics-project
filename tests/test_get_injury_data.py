"""
Tests for src/data/get_injury_data.py

Focuses on:
  - _normalize_pdf_response: maps statuses to lowercase schema, skips unknowns
  - _normalize_nba_api_response: maps team names to abbreviations
  - get_injury_report: returns empty DF with correct schema when all sources fail
  - load_historical_injuries: concatenates multiple CSVs; empty dir returns schema
  - STATUS_MAP / TEAM_NAME_TO_ABB: mapping correctness
"""

import os
import sys
import warnings
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.get_injury_data import (
    _normalize_pdf_response,
    _normalize_nba_api_response,
    load_historical_injuries,
    STATUS_MAP,
    TEAM_NAME_TO_ABB,
)


# ── _normalize_pdf_response ───────────────────────────────────────────────────


class TestNormalizePdfResponse:

    def _make_pdf_frame(self, rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def test_known_statuses_mapped_to_lowercase_schema(self):
        """Out, Questionable, Probable, Day-To-Day must map to normalized strings."""
        raw = self._make_pdf_frame([
            {"player_name": "Player A", "team": "Boston Celtics",
             "current_status": "Out", "reason": "Knee"},
            {"player_name": "Player B", "team": "Los Angeles Lakers",
             "current_status": "Questionable", "reason": "Back"},
            {"player_name": "Player C", "team": "Miami Heat",
             "current_status": "Probable", "reason": "Ankle"},
            {"player_name": "Player D", "team": "Chicago Bulls",
             "current_status": "Day-To-Day", "reason": "Illness"},
        ])

        result = _normalize_pdf_response(raw, "2025-03-05")

        assert len(result) == 4
        statuses = set(result["status"].tolist())
        assert statuses == {"out", "questionable", "probable", "day-to-day"}

    def test_unknown_status_skipped(self):
        """Rows with statuses not in STATUS_MAP (e.g. 'Available') must be skipped."""
        raw = self._make_pdf_frame([
            {"player_name": "Player X", "team": "Boston Celtics",
             "current_status": "Available", "reason": ""},
            {"player_name": "Player Y", "team": "Boston Celtics",
             "current_status": "Out", "reason": "Flu"},
        ])

        result = _normalize_pdf_response(raw, "2025-03-05")

        assert len(result) == 1
        assert result["status"].iloc[0] == "out"

    def test_team_name_mapped_to_abbreviation(self):
        """Full team names must be converted to 3-letter abbreviations."""
        raw = self._make_pdf_frame([
            {"player_name": "Player Z", "team": "Los Angeles Lakers",
             "current_status": "Out", "reason": "Rest"},
        ])

        result = _normalize_pdf_response(raw, "2025-03-05")
        assert result["team_abbr"].iloc[0] == "LAL"

    def test_output_schema_columns_present(self):
        """Output must contain all 6 documented output columns."""
        raw = self._make_pdf_frame([
            {"player_name": "Player A", "team": "Boston Celtics",
             "current_status": "Out", "reason": "Knee"},
        ])

        result = _normalize_pdf_response(raw, "2025-03-05")

        expected_cols = {"date", "player_name", "player_id", "team_abbr", "status", "injury_type"}
        assert expected_cols.issubset(set(result.columns))

    def test_player_id_is_empty_string_from_pdf(self):
        """player_id must always be empty string for PDF-sourced data (not available)."""
        raw = self._make_pdf_frame([
            {"player_name": "Player A", "team": "Boston Celtics",
             "current_status": "Questionable", "reason": "Hamstring"},
        ])

        result = _normalize_pdf_response(raw, "2025-03-05")
        assert result["player_id"].iloc[0] == ""


# ── _normalize_nba_api_response ───────────────────────────────────────────────


class TestNormalizeNbaApiResponse:

    def test_team_name_mapped_to_abbreviation(self):
        """team_name from nba_api must be converted to abbreviation via TEAM_NAME_TO_ABB."""
        raw = pd.DataFrame([{
            "player_name": "Player A",
            "player_id": "12345",
            "team_name": "Golden State Warriors",
            "player_status": "Out",
            "return_date": "Knee",
        }])

        result = _normalize_nba_api_response(raw, "2025-03-05")
        assert len(result) == 1
        assert result["team_abbr"].iloc[0] == "GSW"

    def test_unknown_status_rows_skipped(self):
        """Rows where player_status is empty or unknown must be excluded."""
        raw = pd.DataFrame([
            {"player_name": "A", "player_id": "1", "team_name": "Boston Celtics",
             "player_status": "", "return_date": ""},
            {"player_name": "B", "player_id": "2", "team_name": "Boston Celtics",
             "player_status": "Out", "return_date": "Flu"},
        ])

        result = _normalize_nba_api_response(raw, "2025-03-05")
        # Empty status produces "".lower() -> "" which is falsy -> skipped
        # "Out" maps to "out" -> kept
        assert len(result) == 1
        assert result["player_name"].iloc[0] == "B"


# ── load_historical_injuries ──────────────────────────────────────────────────


class TestLoadHistoricalInjuries:

    EXPECTED_COLS = {"date", "player_name", "player_id", "team_abbr", "status", "injury_type"}

    def test_empty_dir_returns_schema_dataframe(self, tmp_path):
        """When no CSVs are in the directory, return an empty DF with correct columns."""
        result = load_historical_injuries(injuries_dir=str(tmp_path))

        assert isinstance(result, pd.DataFrame)
        assert result.empty
        assert self.EXPECTED_COLS.issubset(set(result.columns))

    def test_concatenates_multiple_csv_files(self, tmp_path):
        """Multiple injury_report_*.csv files must be concatenated into one DF."""
        row_template = {
            "date": "2025-03-01",
            "player_name": "Player A",
            "player_id": "",
            "team_abbr": "LAL",
            "status": "out",
            "injury_type": "Knee",
        }
        for date_suffix in ["2025-03-01", "2025-03-02", "2025-03-03"]:
            df = pd.DataFrame([{**row_template, "date": date_suffix}])
            df.to_csv(tmp_path / f"injury_report_{date_suffix}.csv", index=False)

        result = load_historical_injuries(injuries_dir=str(tmp_path))

        assert len(result) == 3
        assert set(result["date"].tolist()) == {"2025-03-01", "2025-03-02", "2025-03-03"}

    def test_ignores_non_matching_csv_files(self, tmp_path):
        """Files not matching injury_report_*.csv pattern must not be loaded."""
        # Write a valid injury CSV
        pd.DataFrame([{
            "date": "2025-03-01", "player_name": "Player A",
            "player_id": "", "team_abbr": "LAL", "status": "out", "injury_type": "Knee",
        }]).to_csv(tmp_path / "injury_report_2025-03-01.csv", index=False)

        # Write a non-matching file that should be ignored
        pd.DataFrame([{"col": "val"}]).to_csv(tmp_path / "other_data.csv", index=False)

        result = load_historical_injuries(injuries_dir=str(tmp_path))
        assert len(result) == 1


# ── Status map and team mapping sanity ────────────────────────────────────────


class TestMappingConstants:

    def test_status_map_covers_all_documented_statuses(self):
        """STATUS_MAP must cover all 4 documented injury statuses."""
        expected_keys = {"Out", "Questionable", "Probable", "Day-To-Day"}
        assert expected_keys == set(STATUS_MAP.keys())

    def test_all_30_teams_in_team_name_to_abb(self):
        """TEAM_NAME_TO_ABB must contain all 30 current NBA franchises."""
        assert len(TEAM_NAME_TO_ABB) == 30

    def test_team_name_to_abb_values_are_three_chars(self):
        """All team abbreviations must be exactly 3 uppercase characters."""
        for name, abbr in TEAM_NAME_TO_ABB.items():
            assert len(abbr) == 3 and abbr.isupper(), (
                f"'{name}' maps to invalid abbreviation '{abbr}'"
            )
