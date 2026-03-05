"""
Tests for src/features/lineup_features.py

Focuses on:
  - build_lineup_features: output schema has all documented columns
  - build_lineup_features: season column is integer dtype (project convention)
  - build_lineup_features: filters lineups below MIN_GP threshold
  - build_lineup_features: aggregates correctly (top1, top3, avg weighted by minutes)
  - build_lineup_features: raises FileNotFoundError when no data source exists
  - build_lineup_features: saves output CSV to the configured path
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features.lineup_features import build_lineup_features, MIN_GP


# ── Helpers ───────────────────────────────────────────────────────────────────

REQUIRED_OUTPUT_COLS = [
    "season", "team_id", "team_abbreviation",
    "top1_lineup_net_rtg", "top3_lineup_net_rtg", "avg_lineup_net_rtg",
    "lineup_net_rtg_std", "best_off_rating", "best_def_rating", "n_lineups",
]


def _make_lineup_csv(tmp_path, rows: list[dict]) -> str:
    """Write synthetic lineup rows to a temp CSV and return the path."""
    df = pd.DataFrame(rows)
    path = str(tmp_path / "lineup_data.csv")
    df.to_csv(path, index=False)
    return path


def _base_lineup_row(
    season: int = 202425,
    team_id: int = 1610612747,
    team_abbreviation: str = "LAL",
    group_name: str = "P1 - P2 - P3 - P4 - P5",
    gp: int = 10,
    min: float = 200.0,
    net_rating: float = 5.0,
    off_rating: float = 110.0,
    def_rating: float = 105.0,
) -> dict:
    return {
        "season": season,
        "team_id": team_id,
        "team_abbreviation": team_abbreviation,
        "group_name": group_name,
        "gp": gp,
        "min": min,
        "net_rating": net_rating,
        "off_rating": off_rating,
        "def_rating": def_rating,
    }


# ── Output schema ─────────────────────────────────────────────────────────────


class TestOutputSchema:

    def test_required_columns_present(self, tmp_path):
        """Output DataFrame must contain all documented output columns."""
        rows = [_base_lineup_row()]
        path = _make_lineup_csv(tmp_path, rows)
        # Redirect output to tmp_path to avoid writing to real data/features/
        import src.features.lineup_features as lf
        original_output = lf.OUTPUT_PATH
        lf.OUTPUT_PATH = str(tmp_path / "lineup_team_features.csv")

        try:
            result = build_lineup_features(lineup_csv_path=path)
        finally:
            lf.OUTPUT_PATH = original_output

        for col in REQUIRED_OUTPUT_COLS:
            assert col in result.columns, f"Missing required column: {col}"

    def test_season_column_is_integer_dtype(self, tmp_path):
        """
        season column must be integer dtype (e.g. 202425), not string.
        Project convention: integer season codes.
        """
        rows = [_base_lineup_row(season=202324)]
        path = _make_lineup_csv(tmp_path, rows)
        import src.features.lineup_features as lf
        original_output = lf.OUTPUT_PATH
        lf.OUTPUT_PATH = str(tmp_path / "lineup_team_features.csv")

        try:
            result = build_lineup_features(lineup_csv_path=path)
        finally:
            lf.OUTPUT_PATH = original_output

        assert np.issubdtype(result["season"].dtype, np.integer), (
            f"season must be integer dtype, got {result['season'].dtype}"
        )
        assert int(result["season"].iloc[0]) == 202324


# ── MIN_GP filter ─────────────────────────────────────────────────────────────


class TestMinGpFilter:

    def test_lineups_below_min_gp_excluded(self, tmp_path):
        """Lineups with GP < MIN_GP must not appear in the output."""
        rows = [
            _base_lineup_row(gp=MIN_GP - 1, net_rating=20.0, group_name="Below - Threshold"),
            _base_lineup_row(gp=MIN_GP, net_rating=5.0, group_name="At - Threshold"),
            _base_lineup_row(gp=MIN_GP + 5, net_rating=8.0, group_name="Above - Threshold"),
        ]
        path = _make_lineup_csv(tmp_path, rows)
        import src.features.lineup_features as lf
        original_output = lf.OUTPUT_PATH
        lf.OUTPUT_PATH = str(tmp_path / "lineup_team_features.csv")

        try:
            result = build_lineup_features(lineup_csv_path=path)
        finally:
            lf.OUTPUT_PATH = original_output

        # n_lineups should only count the 2 qualifying rows (not the below-threshold one)
        assert result["n_lineups"].iloc[0] == 2

    def test_all_below_min_gp_returns_empty(self, tmp_path):
        """When all lineups are below MIN_GP, result must be an empty DataFrame."""
        rows = [_base_lineup_row(gp=MIN_GP - 1)]
        path = _make_lineup_csv(tmp_path, rows)
        import src.features.lineup_features as lf
        original_output = lf.OUTPUT_PATH
        lf.OUTPUT_PATH = str(tmp_path / "lineup_team_features.csv")

        try:
            result = build_lineup_features(lineup_csv_path=path)
        finally:
            lf.OUTPUT_PATH = original_output

        assert result.empty


# ── Aggregation correctness ───────────────────────────────────────────────────


class TestAggregations:

    def test_top1_is_max_net_rating(self, tmp_path):
        """top1_lineup_net_rtg must be the maximum net_rating across qualifying lineups."""
        rows = [
            _base_lineup_row(gp=10, net_rating=12.0, group_name="Best"),
            _base_lineup_row(gp=8, net_rating=5.0, group_name="Middle"),
            _base_lineup_row(gp=6, net_rating=-2.0, group_name="Worst"),
        ]
        path = _make_lineup_csv(tmp_path, rows)
        import src.features.lineup_features as lf
        original_output = lf.OUTPUT_PATH
        lf.OUTPUT_PATH = str(tmp_path / "lineup_team_features.csv")

        try:
            result = build_lineup_features(lineup_csv_path=path)
        finally:
            lf.OUTPUT_PATH = original_output

        assert result["top1_lineup_net_rtg"].iloc[0] == pytest.approx(12.0)

    def test_best_def_rating_is_minimum(self, tmp_path):
        """
        best_def_rating must be the LOWEST def_rating (lower = better defense).
        """
        rows = [
            _base_lineup_row(gp=10, def_rating=100.0, group_name="Best defense"),
            _base_lineup_row(gp=8, def_rating=112.0, group_name="Worst defense"),
        ]
        path = _make_lineup_csv(tmp_path, rows)
        import src.features.lineup_features as lf
        original_output = lf.OUTPUT_PATH
        lf.OUTPUT_PATH = str(tmp_path / "lineup_team_features.csv")

        try:
            result = build_lineup_features(lineup_csv_path=path)
        finally:
            lf.OUTPUT_PATH = original_output

        assert result["best_def_rating"].iloc[0] == pytest.approx(100.0)

    def test_avg_lineup_net_rtg_is_minutes_weighted(self, tmp_path):
        """
        avg_lineup_net_rtg must be weighted by minutes, not a simple mean.
        Lineup A: net_rating=10, min=300 (heavy usage)
        Lineup B: net_rating=0,  min=100
        Expected weighted avg = (10*300 + 0*100) / 400 = 7.5
        """
        rows = [
            _base_lineup_row(gp=15, net_rating=10.0, min=300.0, group_name="Heavy"),
            _base_lineup_row(gp=5, net_rating=0.0, min=100.0, group_name="Light"),
        ]
        path = _make_lineup_csv(tmp_path, rows)
        import src.features.lineup_features as lf
        original_output = lf.OUTPUT_PATH
        lf.OUTPUT_PATH = str(tmp_path / "lineup_team_features.csv")

        try:
            result = build_lineup_features(lineup_csv_path=path)
        finally:
            lf.OUTPUT_PATH = original_output

        expected_weighted_avg = (10.0 * 300.0 + 0.0 * 100.0) / 400.0
        assert result["avg_lineup_net_rtg"].iloc[0] == pytest.approx(expected_weighted_avg, abs=0.01)

    def test_multiple_teams_produce_separate_rows(self, tmp_path):
        """Each (season, team_id) combination must produce exactly one output row."""
        rows = [
            _base_lineup_row(team_id=1, team_abbreviation="LAL", gp=10),
            _base_lineup_row(team_id=2, team_abbreviation="BOS", gp=10),
        ]
        path = _make_lineup_csv(tmp_path, rows)
        import src.features.lineup_features as lf
        original_output = lf.OUTPUT_PATH
        lf.OUTPUT_PATH = str(tmp_path / "lineup_team_features.csv")

        try:
            result = build_lineup_features(lineup_csv_path=path)
        finally:
            lf.OUTPUT_PATH = original_output

        assert len(result) == 2
        team_ids = set(result["team_id"].tolist())
        assert team_ids == {1, 2}


# ── Error handling ────────────────────────────────────────────────────────────


class TestErrorHandling:

    def test_raises_file_not_found_when_no_data_source(self, tmp_path):
        """FileNotFoundError must be raised when the path doesn't exist and no raw files exist."""
        nonexistent_path = str(tmp_path / "does_not_exist.csv")

        import src.features.lineup_features as lf
        # Also patch RAW_GLOB to point somewhere with no files
        original_glob = lf.RAW_GLOB
        lf.RAW_GLOB = str(tmp_path / "raw_lineups" / "lineup_data_*.csv")
        original_output = lf.OUTPUT_PATH
        lf.OUTPUT_PATH = str(tmp_path / "lineup_team_features.csv")

        try:
            with pytest.raises(FileNotFoundError):
                build_lineup_features(lineup_csv_path=nonexistent_path)
        finally:
            lf.RAW_GLOB = original_glob
            lf.OUTPUT_PATH = original_output


# ── Output file written ───────────────────────────────────────────────────────


class TestOutputFile:

    def test_csv_saved_to_output_path(self, tmp_path):
        """build_lineup_features must save a CSV at the configured OUTPUT_PATH."""
        rows = [_base_lineup_row()]
        path = _make_lineup_csv(tmp_path, rows)

        import src.features.lineup_features as lf
        output_path = str(tmp_path / "lineup_team_features.csv")
        original_output = lf.OUTPUT_PATH
        lf.OUTPUT_PATH = output_path

        try:
            build_lineup_features(lineup_csv_path=path)
        finally:
            lf.OUTPUT_PATH = original_output

        assert os.path.exists(output_path), "Output CSV must be saved to OUTPUT_PATH"
        saved = pd.read_csv(output_path)
        assert len(saved) > 0
