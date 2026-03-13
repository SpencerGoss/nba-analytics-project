"""Tests for scripts/builder_helpers.py — shared dashboard builder utilities."""
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.builder_helpers import (
    load_team_names,
    record_str,
    games_behind,
    safe_float,
    write_json,
    load_json,
)


# ---------------------------------------------------------------------------
# record_str
# ---------------------------------------------------------------------------

def test_record_str_basic():
    assert record_str(50, 32) == "50-32"


def test_record_str_zeros():
    assert record_str(0, 0) == "0-0"


# ---------------------------------------------------------------------------
# games_behind
# ---------------------------------------------------------------------------

def test_games_behind_leader():
    assert games_behind(50, 20, 50, 20) == 0.0


def test_games_behind_one_back():
    assert games_behind(50, 20, 49, 21) == 1.0


def test_games_behind_half():
    assert games_behind(50, 20, 50, 21) == 0.5


def test_games_behind_ten():
    assert games_behind(60, 10, 50, 20) == 10.0


# ---------------------------------------------------------------------------
# safe_float
# ---------------------------------------------------------------------------

def test_safe_float_normal():
    assert safe_float(3.14) == 3.14


def test_safe_float_string():
    assert safe_float("42.5") == 42.5


def test_safe_float_none():
    assert safe_float(None) is None


def test_safe_float_none_with_default():
    assert safe_float(None, default=0.0) == 0.0


def test_safe_float_nan():
    assert safe_float(float("nan")) is None


def test_safe_float_nan_with_default():
    assert safe_float(float("nan"), default=-1.0) == -1.0


def test_safe_float_bad_string():
    assert safe_float("not_a_number") is None


def test_safe_float_rounding():
    assert safe_float(3.14159, decimals=2) == 3.14


def test_safe_float_rounding_with_default():
    assert safe_float("bad", default=0.0, decimals=3) == 0.0


def test_safe_float_int():
    assert safe_float(42) == 42.0


def test_safe_float_pd_na():
    assert safe_float(pd.NA) is None


# ---------------------------------------------------------------------------
# load_team_names
# ---------------------------------------------------------------------------

def test_load_team_names_from_csv(tmp_path):
    csv = tmp_path / "teams.csv"
    csv.write_text("abbreviation,full_name\nBOS,Boston Celtics\nLAL,Los Angeles Lakers\n")
    result = load_team_names(csv)
    assert result["BOS"] == "Boston Celtics"
    assert result["LAL"] == "Los Angeles Lakers"


def test_load_team_names_missing_csv_falls_back(tmp_path):
    missing = tmp_path / "nonexistent.csv"
    result = load_team_names(missing)
    # Should fall back to config.TEAM_ABBREV_TO_FULL
    assert len(result) == 30
    assert "BOS" in result


# ---------------------------------------------------------------------------
# write_json / load_json
# ---------------------------------------------------------------------------

def test_write_and_load_json(tmp_path):
    data = {"key": "value", "nested": [1, 2, 3]}
    out = tmp_path / "sub" / "test.json"
    write_json(data, out)
    assert out.exists()
    loaded = load_json(out)
    assert loaded == data


def test_load_json_missing_file(tmp_path):
    result = load_json(tmp_path / "nope.json")
    assert result == {}


def test_write_json_compact(tmp_path):
    out = tmp_path / "compact.json"
    write_json({"a": 1}, out)
    raw = out.read_text()
    assert " " not in raw  # compact separators, no spaces
