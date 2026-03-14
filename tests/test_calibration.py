"""Tests for src/models/calibration.py.

Tests the calibration wrappers (_CalibratedWrapper, _PlattWrapper),
helper functions (_expected_calibration_error, _bin_calibration_stats),
and model loading utilities. Does NOT test the full run_calibration_analysis
pipeline (requires trained model artifacts).

Note: pickle is used here intentionally — it's how sklearn models are
serialized in this project. All pickle operations use trusted local data.
"""
import numpy as np
import pandas as pd
import pickle
import tempfile
import os
import pytest

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

from src.models.calibration import (
    _CalibratedWrapper,
    _PlattWrapper,
    _expected_calibration_error,
    _bin_calibration_stats,
    _load_model,
    _load_features,
)


# -- Fixtures -----------------------------------------------------------------

@pytest.fixture
def dummy_model():
    """A fitted sklearn pipeline that acts as a base model."""
    np.random.seed(42)
    X = np.random.randn(200, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=10, max_depth=2, random_state=42)),
    ])
    pipe.fit(X, y)
    return pipe


@pytest.fixture
def fitted_isotonic(dummy_model):
    """An IsotonicRegression fitted on dummy model predictions."""
    np.random.seed(42)
    X_cal = np.random.randn(100, 5)
    y_cal = (X_cal[:, 0] + X_cal[:, 1] > 0).astype(int)
    raw_probs = dummy_model.predict_proba(X_cal)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_probs, y_cal)
    return iso


@pytest.fixture
def fitted_platt(dummy_model):
    """A LogisticRegression (Platt scaler) fitted on dummy model predictions."""
    np.random.seed(42)
    X_cal = np.random.randn(100, 5)
    y_cal = (X_cal[:, 0] + X_cal[:, 1] > 0).astype(int)
    raw_probs = dummy_model.predict_proba(X_cal)[:, 1]
    lr = LogisticRegression(solver="lbfgs", max_iter=1000)
    lr.fit(raw_probs.reshape(-1, 1), y_cal)
    return lr


@pytest.fixture
def test_X():
    np.random.seed(99)
    return np.random.randn(50, 5)


# -- _CalibratedWrapper tests ------------------------------------------------

class TestCalibratedWrapper:
    def test_predict_proba_shape(self, dummy_model, fitted_isotonic, test_X):
        wrapper = _CalibratedWrapper(dummy_model, fitted_isotonic)
        proba = wrapper.predict_proba(test_X)
        assert proba.shape == (50, 2)

    def test_predict_proba_sums_to_one(self, dummy_model, fitted_isotonic, test_X):
        wrapper = _CalibratedWrapper(dummy_model, fitted_isotonic)
        proba = wrapper.predict_proba(test_X)
        sums = proba.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)

    def test_predict_proba_in_range(self, dummy_model, fitted_isotonic, test_X):
        wrapper = _CalibratedWrapper(dummy_model, fitted_isotonic)
        proba = wrapper.predict_proba(test_X)
        assert (proba >= 0).all()
        assert (proba <= 1).all()

    def test_predict_returns_binary(self, dummy_model, fitted_isotonic, test_X):
        wrapper = _CalibratedWrapper(dummy_model, fitted_isotonic)
        preds = wrapper.predict(test_X)
        assert set(preds).issubset({0, 1})

    def test_predict_matches_proba_threshold(self, dummy_model, fitted_isotonic, test_X):
        wrapper = _CalibratedWrapper(dummy_model, fitted_isotonic)
        proba = wrapper.predict_proba(test_X)[:, 1]
        preds = wrapper.predict(test_X)
        expected = (proba >= 0.5).astype(int)
        np.testing.assert_array_equal(preds, expected)

    def test_calibration_method_attribute(self, dummy_model, fitted_isotonic):
        wrapper = _CalibratedWrapper(dummy_model, fitted_isotonic)
        assert wrapper.calibration_method == "isotonic"

    def test_pickleable(self, dummy_model, fitted_isotonic, test_X):
        """Pickle round-trip produces identical predictions (required for model persistence)."""
        wrapper = _CalibratedWrapper(dummy_model, fitted_isotonic)
        proba_before = wrapper.predict_proba(test_X)
        data = pickle.dumps(wrapper)
        restored = pickle.loads(data)
        proba_after = restored.predict_proba(test_X)
        np.testing.assert_array_equal(proba_before, proba_after)


# -- _PlattWrapper tests -----------------------------------------------------

class TestPlattWrapper:
    def test_predict_proba_shape(self, dummy_model, fitted_platt, test_X):
        wrapper = _PlattWrapper(dummy_model, fitted_platt)
        proba = wrapper.predict_proba(test_X)
        assert proba.shape == (50, 2)

    def test_predict_proba_sums_to_one(self, dummy_model, fitted_platt, test_X):
        wrapper = _PlattWrapper(dummy_model, fitted_platt)
        proba = wrapper.predict_proba(test_X)
        sums = proba.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)

    def test_predict_proba_in_range(self, dummy_model, fitted_platt, test_X):
        wrapper = _PlattWrapper(dummy_model, fitted_platt)
        proba = wrapper.predict_proba(test_X)
        assert (proba >= 0).all()
        assert (proba <= 1).all()

    def test_predict_returns_binary(self, dummy_model, fitted_platt, test_X):
        wrapper = _PlattWrapper(dummy_model, fitted_platt)
        preds = wrapper.predict(test_X)
        assert set(preds).issubset({0, 1})

    def test_calibration_method_attribute(self, dummy_model, fitted_platt):
        wrapper = _PlattWrapper(dummy_model, fitted_platt)
        assert wrapper.calibration_method == "platt"

    def test_pickleable(self, dummy_model, fitted_platt, test_X):
        """Pickle round-trip produces identical predictions (required for model persistence)."""
        wrapper = _PlattWrapper(dummy_model, fitted_platt)
        proba_before = wrapper.predict_proba(test_X)
        data = pickle.dumps(wrapper)
        restored = pickle.loads(data)
        proba_after = restored.predict_proba(test_X)
        np.testing.assert_array_equal(proba_before, proba_after)


# -- _expected_calibration_error tests ----------------------------------------

class TestExpectedCalibrationError:
    def test_perfect_calibration(self):
        """ECE should be near 0 for a well-calibrated model."""
        np.random.seed(42)
        n = 10000
        y_prob = np.random.uniform(0, 1, n)
        y_true = (np.random.uniform(0, 1, n) < y_prob).astype(float)
        ece = _expected_calibration_error(y_true, y_prob, n_bins=10)
        assert ece < 0.03

    def test_terrible_calibration(self):
        """Always predicting 0.9 when truth is ~50/50 should give high ECE."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
        y_prob = np.full(10, 0.9)
        ece = _expected_calibration_error(y_true, y_prob, n_bins=10)
        assert ece > 0.3

    def test_returns_float(self):
        y_true = np.array([0, 1, 0, 1], dtype=float)
        y_prob = np.array([0.3, 0.7, 0.4, 0.6])
        ece = _expected_calibration_error(y_true, y_prob)
        assert isinstance(ece, float)

    def test_ece_bounded(self):
        """ECE must be between 0 and 1."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100).astype(float)
        y_prob = np.random.uniform(0, 1, 100)
        ece = _expected_calibration_error(y_true, y_prob)
        assert 0 <= ece <= 1

    def test_empty_bins_handled(self):
        """All predictions in one bin should still compute ECE."""
        y_true = np.array([0, 1, 0, 1], dtype=float)
        y_prob = np.array([0.51, 0.52, 0.53, 0.54])
        ece = _expected_calibration_error(y_true, y_prob, n_bins=10)
        assert isinstance(ece, float)


# -- _bin_calibration_stats tests ---------------------------------------------

class TestBinCalibrationStats:
    def test_returns_dataframe(self):
        y_true = np.array([0, 1, 0, 1, 0, 1], dtype=float)
        y_prob = np.array([0.1, 0.3, 0.5, 0.7, 0.2, 0.8])
        result = _bin_calibration_stats(y_true, y_prob)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        y_true = np.array([0, 1, 0, 1], dtype=float)
        y_prob = np.array([0.2, 0.7, 0.3, 0.8])
        result = _bin_calibration_stats(y_true, y_prob)
        expected_cols = {"bin_low", "bin_high", "n_games", "mean_pred", "actual_rate", "gap"}
        assert set(result.columns) == expected_cols

    def test_n_games_sums_to_total(self):
        np.random.seed(42)
        n = 100
        y_true = np.random.randint(0, 2, n).astype(float)
        y_prob = np.random.uniform(0, 1, n)
        result = _bin_calibration_stats(y_true, y_prob)
        assert result["n_games"].sum() == n

    def test_gap_equals_pred_minus_actual(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 50).astype(float)
        y_prob = np.random.uniform(0, 1, 50)
        result = _bin_calibration_stats(y_true, y_prob)
        for _, row in result.iterrows():
            expected_gap = row["mean_pred"] - row["actual_rate"]
            assert abs(row["gap"] - expected_gap) < 1e-10


# -- _load_model / _load_features tests ---------------------------------------

class TestLoadModel:
    def test_missing_model_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Trained model not found"):
            _load_model(str(tmp_path))

    def test_loads_pickled_model(self, tmp_path, dummy_model):
        """Loads a trusted local model artifact (pickle is the project standard for sklearn)."""
        path = tmp_path / "game_outcome_model.pkl"
        with open(path, "wb") as f:
            pickle.dump(dummy_model, f)
        loaded = _load_model(str(tmp_path))
        assert hasattr(loaded, "predict_proba")


class TestLoadFeatures:
    def test_missing_features_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _load_features(str(tmp_path))

    def test_loads_feature_list(self, tmp_path):
        """Loads a trusted local feature list (pickle is the project standard for sklearn)."""
        features = ["feat_a", "feat_b", "feat_c"]
        path = tmp_path / "game_outcome_features.pkl"
        with open(path, "wb") as f:
            pickle.dump(features, f)
        loaded = _load_features(str(tmp_path))
        assert loaded == features
