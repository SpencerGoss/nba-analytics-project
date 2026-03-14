"""Tests for src/models/ats_model.py.

Tests the helper functions (get_ats_feature_cols, _validate_null_rates,
_clone_pipeline) and the predict_ats function with mock artifacts.
Does NOT test train_ats_model end-to-end (requires real CSV data).

Note: pickle is used intentionally here — it's the project standard for
sklearn model serialization. All pickle operations use trusted local data.
"""
import json
import os
import numpy as np
import pandas as pd
import pickle
import pytest

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.models.ats_model import (
    get_ats_feature_cols,
    _validate_null_rates,
    _clone_pipeline,
    predict_ats,
)


# -- Fixtures -----------------------------------------------------------------

@pytest.fixture
def sample_ats_df():
    """DataFrame mimicking game_ats_features.csv structure."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "game_id": range(n),
        "season": [202425] * n,
        "game_date": pd.date_range("2025-01-01", periods=n),
        "home_team": ["BOS"] * n,
        "away_team": ["LAL"] * n,
        "home_win": np.random.randint(0, 2, n),
        "covers_spread": np.random.randint(0, 2, n).astype(float),
        # diff columns
        "diff_elo": np.random.randn(n),
        "diff_net_rtg": np.random.randn(n),
        "diff_off_rtg": np.random.randn(n),
        # schedule cols
        "home_days_rest": np.random.randint(1, 5, n).astype(float),
        "away_days_rest": np.random.randint(1, 5, n).astype(float),
        "home_is_back_to_back": np.random.randint(0, 2, n).astype(float),
        "away_is_back_to_back": np.random.randint(0, 2, n).astype(float),
        "season_month": np.random.randint(1, 7, n).astype(float),
        # injury cols
        "home_missing_minutes": np.random.uniform(0, 50, n),
        "away_missing_minutes": np.random.uniform(0, 50, n),
        # ATS-specific cols
        "spread": np.random.uniform(-10, 10, n),
        "home_implied_prob": np.random.uniform(0.3, 0.7, n),
        "away_implied_prob": np.random.uniform(0.3, 0.7, n),
        # ATS v2 cols
        "home_ats_record_5g": np.random.uniform(0, 1, n),
        "away_ats_record_5g": np.random.uniform(0, 1, n),
        "diff_ats_record_5g": np.random.randn(n),
    })


@pytest.fixture
def dummy_ats_model(sample_ats_df):
    """A fitted pipeline for ATS prediction."""
    feat_cols = get_ats_feature_cols(sample_ats_df)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("clf", GradientBoostingClassifier(n_estimators=10, max_depth=2, random_state=42)),
    ])
    pipe.fit(sample_ats_df[feat_cols], sample_ats_df["covers_spread"])
    return pipe, feat_cols


# -- get_ats_feature_cols tests -----------------------------------------------

class TestGetAtsFeatureCols:
    def test_returns_list(self, sample_ats_df):
        cols = get_ats_feature_cols(sample_ats_df)
        assert isinstance(cols, list)
        assert len(cols) > 0

    def test_excludes_target_and_ids(self, sample_ats_df):
        cols = get_ats_feature_cols(sample_ats_df)
        excluded = {"covers_spread", "game_id", "season", "game_date", "home_team", "away_team", "home_win"}
        for col in excluded:
            assert col not in cols, f"{col} should be excluded"

    def test_includes_diff_columns(self, sample_ats_df):
        cols = get_ats_feature_cols(sample_ats_df)
        diff_cols = [c for c in cols if c.startswith("diff_")]
        assert len(diff_cols) >= 3

    def test_includes_ats_specific_cols(self, sample_ats_df):
        cols = get_ats_feature_cols(sample_ats_df)
        assert "spread" in cols
        assert "home_implied_prob" in cols
        assert "away_implied_prob" in cols

    def test_includes_schedule_cols(self, sample_ats_df):
        cols = get_ats_feature_cols(sample_ats_df)
        assert "home_days_rest" in cols
        assert "away_days_rest" in cols

    def test_sorted_output(self, sample_ats_df):
        cols = get_ats_feature_cols(sample_ats_df)
        assert cols == sorted(cols)

    def test_no_string_columns(self, sample_ats_df):
        """Only numeric columns should be selected."""
        sample_ats_df["string_col"] = "text"
        cols = get_ats_feature_cols(sample_ats_df)
        assert "string_col" not in cols


# -- _validate_null_rates tests -----------------------------------------------

class TestValidateNullRates:
    def test_no_nulls_passes(self, sample_ats_df):
        cols = get_ats_feature_cols(sample_ats_df)
        _validate_null_rates(sample_ats_df, cols)

    def test_all_null_column_raises(self, sample_ats_df):
        sample_ats_df["diff_elo"] = np.nan
        cols = get_ats_feature_cols(sample_ats_df)
        with pytest.raises(ValueError, match="null threshold"):
            _validate_null_rates(sample_ats_df, cols)

    def test_partial_nulls_below_threshold_passes(self, sample_ats_df):
        sample_ats_df.loc[sample_ats_df.index[:50], "diff_elo"] = np.nan
        cols = get_ats_feature_cols(sample_ats_df)
        _validate_null_rates(sample_ats_df, cols)

    def test_custom_threshold(self, sample_ats_df):
        sample_ats_df.loc[sample_ats_df.index[:60], "diff_elo"] = np.nan
        cols = get_ats_feature_cols(sample_ats_df)
        with pytest.raises(ValueError):
            _validate_null_rates(sample_ats_df, cols, threshold=0.50)


# -- _clone_pipeline tests ---------------------------------------------------

class TestClonePipeline:
    def test_returns_pipeline(self):
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("clf", GradientBoostingClassifier(n_estimators=10, random_state=42)),
        ])
        cloned = _clone_pipeline(pipe)
        assert isinstance(cloned, Pipeline)

    def test_clone_is_unfitted(self):
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randint(0, 2, 50)
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("clf", GradientBoostingClassifier(n_estimators=10, random_state=42)),
        ])
        pipe.fit(X, y)
        cloned = _clone_pipeline(pipe)
        assert not hasattr(cloned.named_steps["clf"], "estimators_")

    def test_clone_preserves_params(self):
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("clf", GradientBoostingClassifier(n_estimators=42, max_depth=5, random_state=42)),
        ])
        cloned = _clone_pipeline(pipe)
        assert cloned.named_steps["clf"].n_estimators == 42
        assert cloned.named_steps["clf"].max_depth == 5


# -- predict_ats tests --------------------------------------------------------

class TestPredictAts:
    def _save_artifacts(self, tmp_path, model, feat_cols, threshold=0.50):
        """Helper: persist model artifacts to tmp_path."""
        with open(tmp_path / "ats_model.pkl", "wb") as f:
            pickle.dump(model, f)
        with open(tmp_path / "ats_model_features.pkl", "wb") as f:
            pickle.dump(feat_cols, f)
        meta = {"threshold": threshold}
        with open(tmp_path / "ats_model_metadata.json", "w") as f:
            json.dump(meta, f)

    def test_returns_dataframe(self, tmp_path, sample_ats_df, dummy_ats_model):
        model, feat_cols = dummy_ats_model
        self._save_artifacts(tmp_path, model, feat_cols)
        result = predict_ats(sample_ats_df, str(tmp_path))
        assert isinstance(result, pd.DataFrame)
        assert "covers_spread_prob" in result.columns
        assert "covers_spread_pred" in result.columns

    def test_probabilities_in_range(self, tmp_path, sample_ats_df, dummy_ats_model):
        model, feat_cols = dummy_ats_model
        self._save_artifacts(tmp_path, model, feat_cols)
        result = predict_ats(sample_ats_df, str(tmp_path))
        assert (result["covers_spread_prob"] >= 0).all()
        assert (result["covers_spread_prob"] <= 1).all()

    def test_predictions_are_binary(self, tmp_path, sample_ats_df, dummy_ats_model):
        model, feat_cols = dummy_ats_model
        self._save_artifacts(tmp_path, model, feat_cols)
        result = predict_ats(sample_ats_df, str(tmp_path))
        assert set(result["covers_spread_pred"].unique()).issubset({0, 1})

    def test_preserves_index(self, tmp_path, sample_ats_df, dummy_ats_model):
        model, feat_cols = dummy_ats_model
        self._save_artifacts(tmp_path, model, feat_cols)
        sample_ats_df.index = range(100, 200)
        result = predict_ats(sample_ats_df, str(tmp_path))
        assert list(result.index) == list(range(100, 200))

    def test_missing_model_raises(self, tmp_path, sample_ats_df):
        with pytest.raises(FileNotFoundError, match="ATS model artifact not found"):
            predict_ats(sample_ats_df, str(tmp_path))

    def test_missing_columns_handled(self, tmp_path, sample_ats_df, dummy_ats_model):
        """Missing feature columns become NaN and are imputed."""
        model, feat_cols = dummy_ats_model
        self._save_artifacts(tmp_path, model, feat_cols)
        partial_df = sample_ats_df.drop(columns=["diff_elo"])
        result = predict_ats(partial_df, str(tmp_path))
        assert len(result) == len(partial_df)

    def test_custom_threshold_from_metadata(self, tmp_path, sample_ats_df, dummy_ats_model):
        model, feat_cols = dummy_ats_model

        # High threshold -> more predictions should be 0
        self._save_artifacts(tmp_path, model, feat_cols, threshold=0.99)
        result_high = predict_ats(sample_ats_df, str(tmp_path))

        # Low threshold -> more predictions should be 1
        self._save_artifacts(tmp_path, model, feat_cols, threshold=0.01)
        result_low = predict_ats(sample_ats_df, str(tmp_path))

        assert result_low["covers_spread_pred"].sum() >= result_high["covers_spread_pred"].sum()
