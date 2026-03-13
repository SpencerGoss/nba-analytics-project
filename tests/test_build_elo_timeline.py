"""Tests for scripts/build_elo_timeline.py."""

import json
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def elo_csv(tmp_path):
    """Create a minimal elo_ratings.csv for testing."""
    data = {
        "team_abbreviation": ["OKC"] * 6 + ["BOS"] * 4,
        "game_date": [
            "2025-10-22", "2025-10-25", "2025-10-28",
            "2025-10-31", "2025-11-03", "2025-11-06",
            "2025-10-23", "2025-10-26", "2025-10-29", "2025-11-01",
        ],
        "season": [202526] * 10,
        "elo_pre": [1500, 1510, 1520, 1530, 1540, 1550, 1490, 1500, 1510, 1520],
        "elo_pre_fast": [1500, 1515, 1525, 1540, 1555, 1560, 1490, 1505, 1515, 1525],
        "elo_momentum": [0, 10, 5, 10, 15, 10, 0, 10, 5, 10],
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "elo_ratings.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def out_json(tmp_path):
    return tmp_path / "elo_timeline.json"


def _patch_and_build(monkeypatch, elo_csv_path, out_json_path):
    """Patch module-level paths and run build."""
    import scripts.build_elo_timeline as mod

    monkeypatch.setattr(mod, "ELO_CSV", elo_csv_path)
    monkeypatch.setattr(mod, "OUT_JSON", out_json_path)
    return mod.build_elo_timeline()


def test_basic_output_structure(monkeypatch, elo_csv, out_json):
    result = _patch_and_build(monkeypatch, elo_csv, out_json)
    assert "teams" in result
    assert "OKC" in result["teams"]
    assert "BOS" in result["teams"]


def test_sampling_reduces_data_points(monkeypatch, elo_csv, out_json):
    """With SAMPLE_EVERY_N=3, 6 OKC games -> sampled indices 0,3 + last(5) = 3 points."""
    result = _patch_and_build(monkeypatch, elo_csv, out_json)
    okc = result["teams"]["OKC"]
    # 6 games, sample every 3rd: indices 0,3, plus last(5) -> 3 unique points
    assert len(okc) == 3


def test_entry_fields(monkeypatch, elo_csv, out_json):
    result = _patch_and_build(monkeypatch, elo_csv, out_json)
    entry = result["teams"]["OKC"][0]
    assert "date" in entry
    assert "elo" in entry
    assert "fast_elo" in entry
    assert "elo_momentum" in entry
    assert isinstance(entry["elo"], float)


def test_missing_csv_returns_empty(monkeypatch, tmp_path, out_json):
    missing = tmp_path / "nonexistent.csv"
    result = _patch_and_build(monkeypatch, missing, out_json)
    assert result == {"teams": {}}


def test_empty_season_returns_empty(monkeypatch, tmp_path, out_json):
    """CSV exists but no rows for current season."""
    data = {"team_abbreviation": ["OKC"], "game_date": ["2023-10-22"],
            "season": [202324], "elo_pre": [1500]}
    df = pd.DataFrame(data)
    csv_path = tmp_path / "elo.csv"
    df.to_csv(csv_path, index=False)
    result = _patch_and_build(monkeypatch, csv_path, out_json)
    assert result["teams"] == {}


def test_writes_json_file(monkeypatch, elo_csv, out_json):
    _patch_and_build(monkeypatch, elo_csv, out_json)
    assert out_json.exists()
    with open(out_json) as f:
        data = json.load(f)
    assert "teams" in data


def test_no_fast_elo_column(monkeypatch, tmp_path, out_json):
    """When elo_pre_fast is missing, fast_elo should be None."""
    data = {"team_abbreviation": ["OKC"] * 3,
            "game_date": ["2025-10-22", "2025-10-25", "2025-10-28"],
            "season": [202526] * 3, "elo_pre": [1500, 1510, 1520]}
    csv_path = tmp_path / "elo.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)
    result = _patch_and_build(monkeypatch, csv_path, out_json)
    entry = result["teams"]["OKC"][0]
    assert entry["fast_elo"] is None
    assert entry["elo_momentum"] is None
