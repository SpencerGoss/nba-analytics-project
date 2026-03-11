"""
Tests for src/features/hustle_features.py
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.features.hustle_features import (
    build_hustle_features,
    _extract_season_code,
    _compute_hustle_index,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS = [
    "season", "team_id", "contested_shots", "deflections",
    "screen_assists", "loose_balls_recovered", "charges_drawn",
    "box_outs", "hustle_index",
]

TEAMS_PER_SEASON = 30


def _make_fake_hustle_csv(tmpdir: str, season_code: int, n_teams: int = 30) -> str:
    """Create a fake hustle stats CSV with realistic column names."""
    rng = np.random.default_rng(seed=season_code)
    rows = []
    for i in range(n_teams):
        rows.append({
            "TEAM_ID": 1610612737 + i,
            "TEAM_NAME": f"Team_{i}",
            "MIN": 3000 + rng.integers(-200, 200),
            "CONTESTED_SHOTS": int(rng.integers(2000, 3500)),
            "CONTESTED_SHOTS_2PT": int(rng.integers(1200, 2200)),
            "CONTESTED_SHOTS_3PT": int(rng.integers(800, 1300)),
            "DEFLECTIONS": int(rng.integers(800, 1500)),
            "CHARGES_DRAWN": int(rng.integers(10, 40)),
            "SCREEN_ASSISTS": int(rng.integers(400, 700)),
            "SCREEN_AST_PTS": int(rng.integers(900, 1500)),
            "OFF_LOOSE_BALLS_RECOVERED": int(rng.integers(100, 200)),
            "DEF_LOOSE_BALLS_RECOVERED": int(rng.integers(100, 200)),
            "LOOSE_BALLS_RECOVERED": int(rng.integers(200, 400)),
            "PCT_LOOSE_BALLS_RECOVERED_OFF": round(rng.uniform(0.4, 0.6), 3),
            "PCT_LOOSE_BALLS_RECOVERED_DEF": round(rng.uniform(0.4, 0.6), 3),
            "OFF_BOXOUTS": int(rng.integers(50, 150)),
            "DEF_BOXOUTS": int(rng.integers(250, 400)),
            "BOX_OUTS": int(rng.integers(300, 550)),
            "PCT_BOX_OUTS_OFF": round(rng.uniform(0.1, 0.3), 3),
            "PCT_BOX_OUTS_DEF": round(rng.uniform(0.7, 0.9), 3),
        })
    df = pd.DataFrame(rows)
    filepath = os.path.join(tmpdir, f"team_hustle_stats_{season_code}.csv")
    df.to_csv(filepath, index=False)
    return filepath


@pytest.fixture
def fake_hustle_dir():
    """Create a temp directory with 3 seasons of fake hustle data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_fake_hustle_csv(tmpdir, 202324, n_teams=30)
        _make_fake_hustle_csv(tmpdir, 202425, n_teams=30)
        _make_fake_hustle_csv(tmpdir, 202526, n_teams=30)
        yield tmpdir


# ---------------------------------------------------------------------------
# Tests: _extract_season_code
# ---------------------------------------------------------------------------

class TestExtractSeasonCode:
    def test_valid_filename(self):
        assert _extract_season_code("team_hustle_stats_202526.csv") == 202526

    def test_full_path(self):
        assert _extract_season_code("/some/path/team_hustle_stats_201920.csv") == 201920

    def test_invalid_filename(self):
        assert _extract_season_code("some_other_file.csv") is None

    def test_windows_path(self):
        assert _extract_season_code("C:\\data\\team_hustle_stats_202324.csv") == 202324


# ---------------------------------------------------------------------------
# Tests: build_hustle_features
# ---------------------------------------------------------------------------

class TestBuildHustleFeatures:
    def test_output_has_expected_columns(self, fake_hustle_dir):
        df = build_hustle_features(data_dir=fake_hustle_dir, output_path=None)
        assert list(df.columns) == EXPECTED_COLUMNS

    def test_season_codes_are_integers(self, fake_hustle_dir):
        df = build_hustle_features(data_dir=fake_hustle_dir, output_path=None)
        assert df["season"].dtype in (np.int64, np.int32, int)
        assert set(df["season"].unique()) == {202324, 202425, 202526}

    def test_hustle_index_not_all_nan(self, fake_hustle_dir):
        df = build_hustle_features(data_dir=fake_hustle_dir, output_path=None)
        assert df["hustle_index"].notna().any(), "hustle_index should not be all NaN"
        # With 30 teams per season, all should be non-NaN
        assert df["hustle_index"].notna().all(), "hustle_index should have no NaN values"

    def test_approximately_30_teams_per_season(self, fake_hustle_dir):
        df = build_hustle_features(data_dir=fake_hustle_dir, output_path=None)
        for season in df["season"].unique():
            n_teams = len(df[df["season"] == season])
            assert 28 <= n_teams <= 30, (
                f"Season {season} has {n_teams} teams, expected ~30"
            )

    def test_no_duplicate_season_team_pairs(self, fake_hustle_dir):
        df = build_hustle_features(data_dir=fake_hustle_dir, output_path=None)
        dupes = df.duplicated(subset=["season", "team_id"])
        assert not dupes.any(), "Found duplicate (season, team_id) pairs"

    def test_empty_directory_returns_empty_dataframe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = build_hustle_features(data_dir=tmpdir, output_path=None)
            assert df.empty
            assert list(df.columns) == EXPECTED_COLUMNS

    def test_hustle_index_mean_near_zero(self, fake_hustle_dir):
        """Within each season, z-scored hustle_index should average near zero."""
        df = build_hustle_features(data_dir=fake_hustle_dir, output_path=None)
        for season in df["season"].unique():
            season_mean = df.loc[df["season"] == season, "hustle_index"].mean()
            assert abs(season_mean) < 0.01, (
                f"Season {season} hustle_index mean={season_mean:.4f}, expected ~0"
            )

    def test_saves_csv_when_output_path_given(self, fake_hustle_dir):
        with tempfile.TemporaryDirectory() as outdir:
            outpath = os.path.join(outdir, "hustle_features.csv")
            df = build_hustle_features(data_dir=fake_hustle_dir, output_path=outpath)
            assert os.path.exists(outpath)
            saved = pd.read_csv(outpath)
            assert len(saved) == len(df)

    def test_hustle_stats_are_positive(self, fake_hustle_dir):
        """All raw hustle stats should be positive counts."""
        df = build_hustle_features(data_dir=fake_hustle_dir, output_path=None)
        stat_cols = [
            "contested_shots", "deflections", "screen_assists",
            "loose_balls_recovered", "charges_drawn", "box_outs",
        ]
        for col in stat_cols:
            assert (df[col] >= 0).all(), f"{col} has negative values"


# ---------------------------------------------------------------------------
# Tests: _compute_hustle_index
# ---------------------------------------------------------------------------

class TestComputeHustleIndex:
    def test_adds_hustle_index_column(self):
        df = pd.DataFrame({
            "season": [202526] * 5,
            "team_id": range(5),
            "contested_shots": [100, 200, 300, 400, 500],
            "deflections": [50, 100, 150, 200, 250],
            "screen_assists": [30, 60, 90, 120, 150],
            "loose_balls_recovered": [20, 40, 60, 80, 100],
            "charges_drawn": [5, 10, 15, 20, 25],
            "box_outs": [40, 80, 120, 160, 200],
        })
        result = _compute_hustle_index(df)
        assert "hustle_index" in result.columns

    def test_higher_stats_give_higher_index(self):
        df = pd.DataFrame({
            "season": [202526] * 3,
            "team_id": [1, 2, 3],
            "contested_shots": [100, 300, 500],
            "deflections": [50, 150, 250],
            "screen_assists": [30, 90, 150],
            "loose_balls_recovered": [20, 60, 100],
            "charges_drawn": [5, 15, 25],
            "box_outs": [40, 120, 200],
        })
        result = _compute_hustle_index(df)
        # Team with highest raw stats should have highest hustle_index
        assert result.loc[result["team_id"] == 3, "hustle_index"].values[0] > \
               result.loc[result["team_id"] == 1, "hustle_index"].values[0]


# ---------------------------------------------------------------------------
# Tests: integration with real data (skipped if files missing)
# ---------------------------------------------------------------------------

REAL_DATA_DIR = "data/raw/team_hustle_stats/"


@pytest.mark.skipif(
    not os.path.isdir(REAL_DATA_DIR),
    reason="Real hustle stats data not available",
)
class TestRealData:
    def test_real_data_loads(self):
        df = build_hustle_features(data_dir=REAL_DATA_DIR, output_path=None)
        assert not df.empty
        assert list(df.columns) == EXPECTED_COLUMNS

    def test_real_data_team_count(self):
        df = build_hustle_features(data_dir=REAL_DATA_DIR, output_path=None)
        for season in df["season"].unique():
            n_teams = len(df[df["season"] == season])
            assert 15 <= n_teams <= 30, (
                f"Season {season} has {n_teams} teams"
            )
