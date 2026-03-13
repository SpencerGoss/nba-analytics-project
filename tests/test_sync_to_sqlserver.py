"""Tests for sync_to_sqlserver.py — helper functions (no SQL Server required)."""

from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.sync_to_sqlserver import (
    _clean_value,
    _coerce_dates,
    _build_create_sql,
    _conn_string,
    _sql_type,
    _table_name_from_path,
)


# ---------------------------------------------------------------------------
# _clean_value
# ---------------------------------------------------------------------------

class TestCleanValue:
    def test_none_returns_none(self):
        assert _clean_value(None) is None

    def test_float_nan_returns_none(self):
        assert _clean_value(float("nan")) is None

    def test_pd_na_returns_none(self):
        assert _clean_value(pd.NA) is None

    def test_np_nan_returns_none(self):
        assert _clean_value(np.nan) is None

    def test_numpy_int64_returns_python_int(self):
        val = np.int64(42)
        result = _clean_value(val)
        assert result == 42
        assert type(result) is int

    def test_numpy_float64_returns_python_float(self):
        val = np.float64(3.14)
        result = _clean_value(val)
        assert abs(result - 3.14) < 1e-10
        assert type(result) is float

    def test_regular_string_passes_through(self):
        assert _clean_value("BOS") == "BOS"

    def test_regular_int_passes_through(self):
        assert _clean_value(42) == 42

    def test_regular_float_passes_through(self):
        assert _clean_value(3.14) == 3.14

    def test_pd_nat_returns_none(self):
        assert _clean_value(pd.NaT) is None


# ---------------------------------------------------------------------------
# _coerce_dates
# ---------------------------------------------------------------------------

class TestCoerceDates:
    def test_coerces_game_date_column(self):
        df = pd.DataFrame({"game_date": ["2026-03-13", "2026-03-14"]})
        result = _coerce_dates(df)
        assert pd.api.types.is_datetime64_any_dtype(result["game_date"])

    def test_coerces_created_at_column(self):
        df = pd.DataFrame({"created_at": ["2026-03-13 10:00:00"]})
        result = _coerce_dates(df)
        assert pd.api.types.is_datetime64_any_dtype(result["created_at"])

    def test_ignores_non_date_columns(self):
        df = pd.DataFrame({"team": ["BOS"], "pts": [110]})
        result = _coerce_dates(df)
        assert result["team"].dtype == object
        assert result["pts"].dtype in (np.int64, np.int32, int)

    def test_handles_mixed_format_dates(self):
        df = pd.DataFrame({"game_date": ["2026-03-13", "2026-03-14 00:00:00"]})
        result = _coerce_dates(df)
        assert pd.api.types.is_datetime64_any_dtype(result["game_date"])
        assert not result["game_date"].isna().any()


# ---------------------------------------------------------------------------
# _sql_type
# ---------------------------------------------------------------------------

class TestSqlType:
    def test_int64_maps_to_bigint(self):
        assert _sql_type("int64") == "BIGINT"

    def test_float64_maps_to_float(self):
        assert _sql_type("float64") == "FLOAT"

    def test_object_maps_to_nvarchar(self):
        assert _sql_type("object") == "NVARCHAR(512)"

    def test_bool_maps_to_bit(self):
        assert _sql_type("bool") == "BIT"

    def test_unknown_dtype_defaults_to_nvarchar(self):
        assert _sql_type("category") == "NVARCHAR(512)"


# ---------------------------------------------------------------------------
# _table_name_from_path
# ---------------------------------------------------------------------------

class TestTableNameFromPath:
    def test_csv_to_table_name(self):
        p = Path("data/processed/player_stats.csv")
        assert _table_name_from_path(p, "processed") == "processed_player_stats"

    def test_different_prefix(self):
        p = Path("data/odds/game_lines.csv")
        assert _table_name_from_path(p, "odds") == "odds_game_lines"


# ---------------------------------------------------------------------------
# _conn_string
# ---------------------------------------------------------------------------

class TestConnString:
    def test_default_database_is_master(self):
        cs = _conn_string()
        assert "DATABASE=master" in cs
        assert "ODBC Driver 17" in cs

    def test_custom_database(self):
        cs = _conn_string("nba_analytics")
        assert "DATABASE=nba_analytics" in cs

    def test_trusted_connection(self):
        cs = _conn_string()
        assert "Trusted_Connection=yes" in cs


# ---------------------------------------------------------------------------
# _build_create_sql
# ---------------------------------------------------------------------------

class TestBuildCreateSql:
    def test_generates_create_table(self):
        df = pd.DataFrame({
            "team": ["BOS"],
            "pts": [110],
            "win_pct": [0.65],
        })
        sql = _build_create_sql("test_table", df)
        assert "CREATE TABLE [test_table]" in sql
        assert "[team] NVARCHAR(512)" in sql
        assert "[pts] BIGINT" in sql
        assert "[win_pct] FLOAT" in sql
        assert "IF NOT EXISTS" in sql

    def test_all_columns_nullable(self):
        df = pd.DataFrame({"a": [1], "b": ["x"]})
        sql = _build_create_sql("t", df)
        assert sql.count("NULL") == 2  # each column has NULL
