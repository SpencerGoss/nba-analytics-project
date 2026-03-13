"""Tests for scripts/pipeline_runner.py.

Tests the builder registry, mode filtering, phase ordering, resume logic,
and dry-run behavior without running actual subprocess calls.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import scripts.pipeline_runner as pr


# ---------------------------------------------------------------------------
# Builder registry
# ---------------------------------------------------------------------------

def test_builders_registry_not_empty():
    assert len(pr.BUILDERS) > 0


def test_all_builders_have_required_keys():
    required = {"phase", "script", "label", "modes"}
    for name, info in pr.BUILDERS.items():
        missing = required - set(info.keys())
        assert not missing, f"Builder '{name}' missing keys: {missing}"


def test_all_builder_scripts_exist():
    for name, info in pr.BUILDERS.items():
        script_path = pr.SCRIPTS_DIR / info["script"]
        assert script_path.exists(), f"Builder '{name}' script missing: {script_path}"


def test_valid_modes_in_builders():
    valid_modes = {"full", "injuries_odds", "pretip"}
    for name, info in pr.BUILDERS.items():
        for mode in info["modes"]:
            assert mode in valid_modes, f"Builder '{name}' has invalid mode: {mode}"


def test_phase_numbers_are_integers():
    for name, info in pr.BUILDERS.items():
        assert isinstance(info["phase"], int), f"Builder '{name}' phase is not int"
        assert 1 <= info["phase"] <= 10, f"Builder '{name}' phase out of range"


# ---------------------------------------------------------------------------
# Mode filtering
# ---------------------------------------------------------------------------

def test_full_mode_includes_all_phases():
    full_builders = {n: i for n, i in pr.BUILDERS.items() if "full" in i["modes"]}
    phases = {i["phase"] for i in full_builders.values()}
    assert len(phases) >= 3, "Full mode should span at least 3 phases"


def test_pretip_mode_includes_picks():
    pretip = {n for n, i in pr.BUILDERS.items() if "pretip" in i["modes"]}
    assert "build_picks" in pretip
    assert "fetch_odds" in pretip


def test_injuries_odds_mode_includes_injuries():
    inj = {n for n, i in pr.BUILDERS.items() if "injuries_odds" in i["modes"]}
    assert "build_injuries" in inj
    assert "fetch_odds" in inj


def test_odds_runs_before_picks():
    """fetch_odds phase must be lower than build_picks phase."""
    odds_phase = pr.BUILDERS["fetch_odds"]["phase"]
    picks_phase = pr.BUILDERS["build_picks"]["phase"]
    assert odds_phase < picks_phase, "Odds must run before picks"


def test_outcomes_run_first():
    """backfill_outcomes should be phase 1 (runs before everything)."""
    assert pr.BUILDERS["backfill_outcomes"]["phase"] == 1


# ---------------------------------------------------------------------------
# Pipeline runner (dry-run, no subprocess)
# ---------------------------------------------------------------------------

def test_dry_run_returns_all_dry_run_statuses(tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(pr, "PROJECT_ROOT", tmp_path)
    (tmp_path / "dashboard" / "data").mkdir(parents=True, exist_ok=True)
    result = pr.run_pipeline(mode="full", dry_run=True)
    assert "results" in result
    for r in result["results"]:
        assert r["status"] == "dry_run"


def test_single_builder_dry_run(tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(pr, "PROJECT_ROOT", tmp_path)
    (tmp_path / "dashboard" / "data").mkdir(parents=True, exist_ok=True)
    result = pr.run_pipeline(mode="full", dry_run=True, single_builder="build_standings")
    assert len(result["results"]) == 1
    assert result["results"][0]["name"] == "build_standings"


def test_unknown_builder_returns_error():
    result = pr.run_pipeline(mode="full", single_builder="nonexistent_builder")
    assert "error" in result


# ---------------------------------------------------------------------------
# Run builder (unit test)
# ---------------------------------------------------------------------------

def test_run_builder_dry_run():
    import logging
    logger = logging.getLogger("test")
    info = pr.BUILDERS["build_standings"]
    result = pr.run_builder("build_standings", info, dry_run=True, logger=logger)
    assert result["status"] == "dry_run"
    assert result["duration"] == 0


def test_run_builder_missing_script():
    import logging
    logger = logging.getLogger("test")
    info = {"phase": 1, "script": "does_not_exist.py", "label": "test", "modes": ["full"]}
    result = pr.run_builder("fake", info, dry_run=False, logger=logger)
    assert result["status"] == "missing"


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def test_save_and_load_state(tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "STATE_FILE", tmp_path / "state.json")
    state = {"mode": "full", "results": [{"name": "x", "status": "success"}]}
    pr.save_state(state)
    loaded = pr.load_state()
    assert loaded["mode"] == "full"
    assert loaded["results"][0]["name"] == "x"


def test_load_state_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "STATE_FILE", tmp_path / "nonexistent.json")
    loaded = pr.load_state()
    assert loaded == {}


# ---------------------------------------------------------------------------
# Health report
# ---------------------------------------------------------------------------

def test_dry_run_writes_report(tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "STATE_FILE", tmp_path / "state.json")
    # Prevent writing to actual dashboard/data/
    monkeypatch.setattr(pr, "PROJECT_ROOT", tmp_path)
    (tmp_path / "dashboard" / "data").mkdir(parents=True, exist_ok=True)

    result = pr.run_pipeline(mode="full", dry_run=True)
    full_count = len([n for n, i in pr.BUILDERS.items() if "full" in i["modes"]])
    assert result["summary"]["skipped"] == full_count
    assert result["summary"]["failed"] == 0
