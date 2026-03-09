"""
Tests for pure helper functions in src/models/player_performance_model.py

Covers:
  - get_feature_cols: excludes raw game stats, metadata, and non-numeric cols;
                      prioritizes rolling/form features when present
  - _split_train_validation: season-based split, index-fallback when empty
  - _extract_importance: coef_ path, feature_importances_ path, zero fallback
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.player_performance_model import (
    VALIDATION_SEASON,
    get_feature_cols,
    _split_train_validation,
    _extract_importance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_player_df(n: int = 20, include_roll_cols: bool = True) -> pd.DataFrame:
    """Minimal player-game DataFrame for feature-col tests."""
    rng = np.random.default_rng(42)
    data: dict = {
        # metadata — always excluded
        "player_id": range(n),
        "player_name": ["Player"] * n,
        "team_id": [1610612747] * n,
        "team_abbreviation": ["LAL"] * n,
        "game_id": [f"G{i:03d}" for i in range(n)],
        "game_date": ["2025-01-01"] * n,
        "matchup": ["LAL vs. GSW"] * n,
        "season": [202425] * n,
        # raw game stats — always excluded
        "pts": rng.integers(5, 40, n).astype(float),
        "reb": rng.integers(0, 15, n).astype(float),
        "ast": rng.integers(0, 12, n).astype(float),
        "stl": rng.integers(0, 4, n).astype(float),
        "blk": rng.integers(0, 4, n).astype(float),
        "tov": rng.integers(0, 6, n).astype(float),
        "min": rng.uniform(10, 40, n),
        "fgm": rng.integers(2, 15, n).astype(float),
        "fga": rng.integers(5, 25, n).astype(float),
        "fg_pct": rng.uniform(0.3, 0.65, n),
        "fg3m": rng.integers(0, 6, n).astype(float),
        "fg3a": rng.integers(0, 10, n).astype(float),
        "fg3_pct": rng.uniform(0.2, 0.5, n),
        "ftm": rng.integers(0, 8, n).astype(float),
        "fta": rng.integers(0, 10, n).astype(float),
        "ft_pct": rng.uniform(0.6, 1.0, n),
        "plus_minus": rng.integers(-20, 20, n).astype(float),
        "win": rng.integers(0, 2, n).astype(float),
        "wl": ["W", "L"] * (n // 2),
        # string — always excluded
        "string_col": ["x"] * n,
    }
    if include_roll_cols:
        data.update({
            "pts_roll10": rng.uniform(10, 30, n),
            "ast_roll10": rng.uniform(2, 10, n),
            "usg_pct": rng.uniform(0.15, 0.35, n),
            "ts_pct": rng.uniform(0.5, 0.65, n),
            "form_delta_pts": rng.uniform(-5, 5, n),
            "season_avg_pts": rng.uniform(10, 30, n),
        })
    return pd.DataFrame(data)


def _make_multi_season_df(n_rows: int = 40) -> pd.DataFrame:
    """Multi-season DataFrame for split tests."""
    rng = np.random.default_rng(7)
    season_codes = ["202122", "202223", "202324", "202425"]
    seasons = [season_codes[i % len(season_codes)] for i in range(n_rows)]
    return pd.DataFrame({
        "season": seasons,
        "pts_roll10": rng.uniform(10, 30, n_rows),
        "pts": rng.uniform(5, 40, n_rows),  # target
    })


# ---------------------------------------------------------------------------
# get_feature_cols
# ---------------------------------------------------------------------------

class TestGetFeatureCols:
    def test_excludes_raw_game_stats(self):
        df = _make_player_df()
        result = get_feature_cols(df, target="pts")
        for raw_stat in ("pts", "reb", "ast", "stl", "blk", "tov", "min",
                         "fgm", "fga", "fg_pct", "fg3m", "fg3a", "fg3_pct",
                         "ftm", "fta", "ft_pct", "plus_minus", "win", "wl"):
            assert raw_stat not in result, f"Raw stat should be excluded: {raw_stat}"

    def test_excludes_metadata(self):
        df = _make_player_df()
        result = get_feature_cols(df, target="pts")
        for meta in ("player_id", "player_name", "team_id", "team_abbreviation",
                     "game_id", "game_date", "matchup", "season"):
            assert meta not in result, f"Metadata col should be excluded: {meta}"

    def test_excludes_string_columns(self):
        df = _make_player_df()
        result = get_feature_cols(df, target="pts")
        assert "string_col" not in result

    def test_includes_rolling_features(self):
        df = _make_player_df(include_roll_cols=True)
        result = get_feature_cols(df, target="pts")
        assert "pts_roll10" in result
        assert "ast_roll10" in result

    def test_includes_usage_and_efficiency(self):
        df = _make_player_df(include_roll_cols=True)
        result = get_feature_cols(df, target="pts")
        assert "usg_pct" in result
        assert "ts_pct" in result

    def test_includes_form_delta(self):
        df = _make_player_df(include_roll_cols=True)
        result = get_feature_cols(df, target="pts")
        assert "form_delta_pts" in result

    def test_includes_season_avg(self):
        df = _make_player_df(include_roll_cols=True)
        result = get_feature_cols(df, target="pts")
        assert "season_avg_pts" in result

    def test_returns_sorted_list(self):
        df = _make_player_df()
        result = get_feature_cols(df, target="pts")
        assert isinstance(result, list)
        assert result == sorted(result)

    def test_fallback_when_no_priority_cols(self):
        """Without any priority-keyword cols, returns all remaining numeric cols."""
        df = pd.DataFrame({
            "player_id": [1, 2],
            "season": [202425, 202425],
            "game_date": ["2025-01-01", "2025-01-02"],
            "some_numeric": [1.0, 2.0],
            "pts": [20.0, 25.0],  # raw stat — excluded
        })
        result = get_feature_cols(df, target="pts")
        assert "some_numeric" in result
        assert "pts" not in result


# ---------------------------------------------------------------------------
# _split_train_validation
# ---------------------------------------------------------------------------

class TestSplitTrainValidation:
    def test_subtrain_is_before_validation_season(self):
        df = _make_multi_season_df()
        subtrain, valid = _split_train_validation(df)
        if not valid.empty:
            subtrain_seasons = set(subtrain["season"].astype(str))
            assert all(s < VALIDATION_SEASON for s in subtrain_seasons)

    def test_valid_contains_only_validation_season(self):
        df = _make_multi_season_df()
        _, valid = _split_train_validation(df)
        if not valid.empty:
            assert set(valid["season"].astype(str)) == {VALIDATION_SEASON}

    def test_fallback_when_validation_season_absent(self):
        """If validation season is not in data, falls back to 85/15 index split."""
        df = pd.DataFrame({
            "season": ["200001"] * 20,  # no VALIDATION_SEASON
            "pts_roll10": np.random.randn(20),
            "pts": np.random.randn(20),
        })
        subtrain, valid = _split_train_validation(df)
        # Should produce non-empty splits
        assert not subtrain.empty
        assert not valid.empty

    def test_no_overlap_between_splits(self):
        df = _make_multi_season_df()
        subtrain, valid = _split_train_validation(df)
        subtrain_idx = set(subtrain.index)
        valid_idx = set(valid.index)
        assert not (subtrain_idx & valid_idx), "Train/valid rows must not overlap"

    def test_subtrain_plus_valid_covers_relevant_seasons(self):
        """Subtrain (< VALIDATION_SEASON) + valid (== VALIDATION_SEASON) covers all
        rows whose season <= VALIDATION_SEASON. Rows after VALIDATION_SEASON are
        intentionally excluded (reserved as future test data)."""
        df = _make_multi_season_df()
        subtrain, valid = _split_train_validation(df)
        relevant = df[df["season"].astype(str) <= VALIDATION_SEASON]
        assert len(subtrain) + len(valid) == len(relevant)


# ---------------------------------------------------------------------------
# _extract_importance
# ---------------------------------------------------------------------------

class TestExtractImportance:
    def _make_fitted_ridge(self, feat_cols: list) -> Pipeline:
        """Train a Ridge model on random data."""
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((50, len(feat_cols))), columns=feat_cols)
        y = rng.standard_normal(50)
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ])
        pipe.fit(X, y)
        return pipe

    def _make_fitted_gb(self, feat_cols: list) -> Pipeline:
        """Train a GradientBoosting model on random data."""
        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.standard_normal((50, len(feat_cols))), columns=feat_cols)
        y = rng.standard_normal(50)
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("model", GradientBoostingRegressor(n_estimators=10, random_state=42)),
        ])
        pipe.fit(X, y)
        return pipe

    def test_coef_path_returns_series(self):
        feat_cols = ["feat_a", "feat_b", "feat_c"]
        pipe = self._make_fitted_ridge(feat_cols)
        result = _extract_importance(pipe, feat_cols)
        assert isinstance(result, pd.Series)

    def test_coef_path_sorted_descending(self):
        feat_cols = ["feat_a", "feat_b", "feat_c"]
        pipe = self._make_fitted_ridge(feat_cols)
        result = _extract_importance(pipe, feat_cols)
        values = result.values
        assert list(values) == sorted(values, reverse=True)

    def test_coef_path_uses_absolute_values(self):
        feat_cols = ["feat_a", "feat_b", "feat_c"]
        pipe = self._make_fitted_ridge(feat_cols)
        result = _extract_importance(pipe, feat_cols)
        assert (result >= 0).all(), "Importance values should be non-negative (abs of coef_)"

    def test_feature_importances_path_returns_series(self):
        feat_cols = ["feat_a", "feat_b", "feat_c"]
        pipe = self._make_fitted_gb(feat_cols)
        result = _extract_importance(pipe, feat_cols)
        assert isinstance(result, pd.Series)

    def test_feature_importances_path_sorted_descending(self):
        feat_cols = ["feat_a", "feat_b", "feat_c"]
        pipe = self._make_fitted_gb(feat_cols)
        result = _extract_importance(pipe, feat_cols)
        values = result.values
        assert list(values) == sorted(values, reverse=True)

    def test_result_length_matches_feat_cols(self):
        feat_cols = ["feat_a", "feat_b", "feat_c", "feat_d"]
        pipe = self._make_fitted_ridge(feat_cols)
        result = _extract_importance(pipe, feat_cols)
        assert len(result) == len(feat_cols)
