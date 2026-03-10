"""Tests for scripts/build_meta.py"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_meta import build_meta, _load_model_version, DEFAULT_MODEL_VERSION


class TestBuildMeta:
    def test_returns_dict(self):
        result = build_meta()
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        result = build_meta()
        assert "exported_at" in result
        assert "db_exists" in result
        assert "model_version" in result
        assert "season" in result
        assert "sample_data" in result

    def test_exported_at_is_iso_string(self):
        result = build_meta()
        # Should parse without error
        dt = datetime.fromisoformat(result["exported_at"])
        assert dt.tzinfo is not None  # timezone-aware

    def test_exported_at_is_recent(self):
        before = datetime.now(timezone.utc)
        result = build_meta()
        after = datetime.now(timezone.utc)
        dt = datetime.fromisoformat(result["exported_at"])
        assert before <= dt <= after

    def test_db_exists_is_bool(self):
        result = build_meta()
        assert isinstance(result["db_exists"], bool)

    def test_sample_data_is_false(self):
        result = build_meta()
        assert result["sample_data"] is False

    def test_season_is_string(self):
        result = build_meta()
        assert isinstance(result["season"], str)
        assert len(result["season"]) > 0

    def test_model_version_is_string(self):
        result = build_meta()
        assert isinstance(result["model_version"], str)
        assert len(result["model_version"]) > 0


class TestLoadModelVersion:
    def test_returns_string(self):
        version = _load_model_version()
        assert isinstance(version, str)

    def test_returns_default_when_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "scripts.build_meta.METADATA_PATH", tmp_path / "nonexistent.json"
        )
        version = _load_model_version()
        assert version == DEFAULT_MODEL_VERSION

    def test_reads_version_from_file(self, tmp_path, monkeypatch):
        metadata = {"model_version": "my_model_v9.9"}
        meta_file = tmp_path / "game_outcome_metadata.json"
        meta_file.write_text(json.dumps(metadata))
        monkeypatch.setattr("scripts.build_meta.METADATA_PATH", meta_file)
        version = _load_model_version()
        assert version == "my_model_v9.9"

    def test_uses_default_when_key_missing(self, tmp_path, monkeypatch):
        metadata = {"other_key": "value"}
        meta_file = tmp_path / "game_outcome_metadata.json"
        meta_file.write_text(json.dumps(metadata))
        monkeypatch.setattr("scripts.build_meta.METADATA_PATH", meta_file)
        version = _load_model_version()
        assert version == DEFAULT_MODEL_VERSION

    def test_handles_malformed_json(self, tmp_path, monkeypatch):
        meta_file = tmp_path / "game_outcome_metadata.json"
        meta_file.write_text("{ not valid json }")
        monkeypatch.setattr("scripts.build_meta.METADATA_PATH", meta_file)
        version = _load_model_version()
        assert version == DEFAULT_MODEL_VERSION


class TestBuildMetaIntegration:
    def test_writes_valid_json(self, tmp_path, monkeypatch):
        out = tmp_path / "meta.json"
        monkeypatch.setattr("scripts.build_meta.OUT_JSON", out)
        from scripts.build_meta import main
        main()
        assert out.exists()
        data = json.loads(out.read_text())
        assert "exported_at" in data
        assert data["sample_data"] is False

    def test_two_calls_advance_timestamp(self):
        r1 = build_meta()
        r2 = build_meta()
        t1 = datetime.fromisoformat(r1["exported_at"])
        t2 = datetime.fromisoformat(r2["exported_at"])
        assert t2 >= t1
