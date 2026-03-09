"""
Tests for src/data/get_balldontlie.py

Focuses on:
  - get_balldontlie_injuries: always returns empty DF with expected schema
  - get_balldontlie_stats: returns empty DF with schema when API key missing
  - get_balldontlie_stats: parses paginated records into correct output columns
  - _fetch_all_pages: handles HTTP 401, empty data, and cursor-based pagination
  - get_balldontlie_teams: returns empty DF when API key missing
"""

import os
import sys
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.get_balldontlie import (
    get_balldontlie_injuries,
    get_balldontlie_stats,
    get_balldontlie_teams,
    _fetch_all_pages,
    _build_headers,
)


# ── get_balldontlie_injuries ──────────────────────────────────────────────────


class TestGetBalldontlieInjuries:

    def test_always_returns_empty_dataframe(self):
        """injuries endpoint does not exist — must always return empty DataFrame."""
        result = get_balldontlie_injuries()
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_returned_schema_has_expected_columns(self):
        """Empty DataFrame must carry the 4 documented schema columns."""
        result = get_balldontlie_injuries()
        expected_cols = {"player_name", "team_abbr", "status", "description"}
        assert expected_cols == set(result.columns)

    def test_no_network_call_made(self):
        """Calling get_balldontlie_injuries must not make any HTTP request."""
        with patch("src.data.get_balldontlie.requests.get") as mock_get:
            get_balldontlie_injuries()
        mock_get.assert_not_called()


# ── get_balldontlie_stats ─────────────────────────────────────────────────────


class TestGetBalldontlieStats:

    def test_returns_empty_df_with_schema_when_no_api_key(self):
        """Without an API key, must return empty DataFrame with all schema columns."""
        with patch("src.data.get_balldontlie._get_api_key", return_value=None):
            result = get_balldontlie_stats(season=2024)

        assert isinstance(result, pd.DataFrame)
        assert result.empty
        expected_cols = {
            "game_id", "date", "season", "home_team_id", "home_team_abbr",
            "visitor_team_id", "visitor_team_abbr", "home_team_score",
            "visitor_team_score", "status", "period", "time",
        }
        assert expected_cols == set(result.columns)

    def test_parses_game_records_into_correct_columns(self):
        """When API returns records, output must contain all documented columns."""
        fake_records = [
            {
                "id": 101,
                "date": "2024-10-22",
                "season": 2024,
                "home_team": {"id": 1, "abbreviation": "LAL"},
                "visitor_team": {"id": 2, "abbreviation": "BOS"},
                "home_team_score": 110,
                "visitor_team_score": 105,
                "status": "Final",
                "period": 4,
                "time": "",
            }
        ]

        with (
            patch("src.data.get_balldontlie._get_api_key", return_value="fake-key"),
            patch("src.data.get_balldontlie._fetch_all_pages", return_value=fake_records),
        ):
            result = get_balldontlie_stats(season=2024)

        assert len(result) == 1
        assert result["game_id"].iloc[0] == 101
        assert result["home_team_abbr"].iloc[0] == "LAL"
        assert result["visitor_team_abbr"].iloc[0] == "BOS"
        assert result["home_team_score"].iloc[0] == 110

    def test_returns_empty_df_when_api_returns_no_records(self):
        """When _fetch_all_pages returns [], result must be an empty DataFrame."""
        with (
            patch("src.data.get_balldontlie._get_api_key", return_value="fake-key"),
            patch("src.data.get_balldontlie._fetch_all_pages", return_value=[]),
        ):
            result = get_balldontlie_stats(season=2024)

        assert isinstance(result, pd.DataFrame)
        assert result.empty


# ── get_balldontlie_teams ─────────────────────────────────────────────────────


class TestGetBalldontlieTeams:

    def test_returns_empty_df_when_no_api_key(self):
        """Without API key, teams function returns empty DataFrame."""
        with patch("src.data.get_balldontlie._get_api_key", return_value=None):
            result = get_balldontlie_teams()

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_parses_team_records_into_correct_columns(self):
        """Team records must be mapped to the documented output schema."""
        fake_records = [
            {
                "id": 1,
                "abbreviation": "LAL",
                "city": "Los Angeles",
                "conference": "West",
                "division": "Pacific",
                "full_name": "Los Angeles Lakers",
                "name": "Lakers",
            }
        ]

        with (
            patch("src.data.get_balldontlie._get_api_key", return_value="fake-key"),
            patch("src.data.get_balldontlie._fetch_all_pages", return_value=fake_records),
        ):
            result = get_balldontlie_teams()

        assert len(result) == 1
        expected_cols = {"team_id", "abbreviation", "city", "conference", "division", "full_name", "name"}
        assert expected_cols.issubset(set(result.columns))
        assert result["abbreviation"].iloc[0] == "LAL"


# ── _fetch_all_pages ──────────────────────────────────────────────────────────


class TestFetchAllPages:

    def _mock_response(self, data: list, next_cursor=None, status_code: int = 200):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = {
            "data": data,
            "meta": {"next_cursor": next_cursor},
        }
        resp.raise_for_status = MagicMock()
        return resp

    def test_single_page_no_cursor_returns_all_records(self):
        """Single page with no next_cursor must return all records."""
        fake_records = [{"id": 1}, {"id": 2}]
        mock_resp = self._mock_response(data=fake_records, next_cursor=None)

        with (
            patch("src.data.get_balldontlie.requests.get", return_value=mock_resp),
            patch("src.data.get_balldontlie.time.sleep"),
        ):
            result = _fetch_all_pages(
                endpoint="https://api.balldontlie.io/v1/games",
                params={"per_page": 100},
                headers=_build_headers("fake-key"),
            )

        assert result == fake_records

    def test_cursor_pagination_follows_next_cursor(self):
        """When next_cursor is present, must fetch subsequent pages."""
        page1_records = [{"id": 1}]
        page2_records = [{"id": 2}]

        resp1 = MagicMock()
        resp1.status_code = 200
        resp1.json.return_value = {"data": page1_records, "meta": {"next_cursor": "abc"}}
        resp1.raise_for_status = MagicMock()

        resp2 = MagicMock()
        resp2.status_code = 200
        resp2.json.return_value = {"data": page2_records, "meta": {"next_cursor": None}}
        resp2.raise_for_status = MagicMock()

        with (
            patch("src.data.get_balldontlie.requests.get", side_effect=[resp1, resp2]),
            patch("src.data.get_balldontlie.time.sleep"),
        ):
            result = _fetch_all_pages(
                endpoint="https://api.balldontlie.io/v1/games",
                params={"per_page": 100},
                headers=_build_headers("fake-key"),
            )

        assert result == page1_records + page2_records

    def test_http_401_returns_empty_list(self):
        """A 401 Unauthorized response must return an empty list (not raise)."""
        import requests as req_lib

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        http_err = req_lib.exceptions.HTTPError(response=mock_resp)
        mock_resp.raise_for_status.side_effect = http_err

        with (
            patch("src.data.get_balldontlie.requests.get", return_value=mock_resp),
            patch("src.data.get_balldontlie.time.sleep"),
        ):
            result = _fetch_all_pages(
                endpoint="https://api.balldontlie.io/v1/games",
                params={"per_page": 100},
                headers=_build_headers("bad-key"),
            )

        assert result == []


# ── _build_headers ────────────────────────────────────────────────────────────


class TestBuildHeaders:

    def test_authorization_key_set(self):
        """Authorization header must equal the provided API key."""
        headers = _build_headers("my-secret-key")
        assert headers["Authorization"] == "my-secret-key"

    def test_accept_header_is_json(self):
        """Accept header must be application/json."""
        headers = _build_headers("any-key")
        assert headers["Accept"] == "application/json"

    def test_returns_dict(self):
        """Return value must be a plain dict."""
        result = _build_headers("k")
        assert isinstance(result, dict)

    def test_both_keys_present(self):
        """Returned dict must contain exactly Authorization and Accept."""
        headers = _build_headers("key")
        assert "Authorization" in headers
        assert "Accept" in headers

    def test_different_keys_passed_through(self):
        """Each unique key string is stored verbatim in Authorization."""
        for key in ["abc", "Bearer token123", "a" * 64]:
            h = _build_headers(key)
            assert h["Authorization"] == key

    def test_empty_string_key_stored(self):
        """An empty string API key is stored as-is (caller's responsibility)."""
        headers = _build_headers("")
        assert headers["Authorization"] == ""
