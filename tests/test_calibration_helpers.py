"""
Tests for pure helper functions in src/models/calibration.py

Covers:
  - _expected_calibration_error: ECE computation
  - _bin_calibration_stats: per-bin calibration diagnostics
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.calibration import (
    _expected_calibration_error,
    _bin_calibration_stats,
)


# ---------------------------------------------------------------------------
# _expected_calibration_error
# ---------------------------------------------------------------------------

class TestExpectedCalibrationError:
    def test_perfect_calibration_is_zero(self):
        """When predicted prob == actual rate in each bin, ECE = 0."""
        # 100 games: probabilities 0.7, outcomes 1 (perfectly calibrated in 1 bin)
        n = 100
        y_prob = np.full(n, 0.7)
        y_true = np.array([1] * 70 + [0] * 30)
        ece = _expected_calibration_error(y_true, y_prob, n_bins=10)
        assert ece == pytest.approx(0.0, abs=0.01)

    def test_worst_calibration_approaches_one(self):
        """All predictions 0.9 but all outcomes are 0 -> large ECE."""
        n = 100
        y_prob = np.full(n, 0.9)
        y_true = np.zeros(n)
        ece = _expected_calibration_error(y_true, y_prob, n_bins=10)
        # |0.9 - 0.0| * 1.0 = 0.9
        assert ece == pytest.approx(0.9, abs=0.02)

    def test_ece_in_0_1_range(self):
        """ECE must always be in [0, 1]."""
        rng = np.random.default_rng(42)
        n = 200
        y_prob = rng.uniform(0, 1, n)
        y_true = (rng.random(n) > 0.5).astype(int)
        ece = _expected_calibration_error(y_true, y_prob, n_bins=10)
        assert 0.0 <= ece <= 1.0

    def test_returns_float(self):
        y_prob = np.array([0.6, 0.7, 0.8])
        y_true = np.array([1, 1, 0])
        result = _expected_calibration_error(y_true, y_prob, n_bins=5)
        assert isinstance(result, float)

    def test_empty_bins_not_counted(self):
        """Bins with no predictions should not affect ECE."""
        # All predictions in [0.6, 0.7] range with 10 bins
        y_prob = np.array([0.65] * 10)
        y_true = np.array([1] * 6 + [0] * 4)  # 60% actual
        ece_tight = _expected_calibration_error(y_true, y_prob, n_bins=10)
        # ECE = |0.65 - 0.60| = 0.05 (all weight in 1 bin)
        assert ece_tight == pytest.approx(0.05, abs=0.01)


# ---------------------------------------------------------------------------
# _bin_calibration_stats
# ---------------------------------------------------------------------------

class TestBinCalibrationStats:
    def test_returns_dataframe(self):
        y_prob = np.array([0.3, 0.5, 0.7, 0.8])
        y_true = np.array([0, 1, 1, 1])
        result = _bin_calibration_stats(y_true, y_prob, n_bins=10)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns_present(self):
        y_prob = np.array([0.3, 0.5, 0.7, 0.8])
        y_true = np.array([0, 1, 1, 1])
        result = _bin_calibration_stats(y_true, y_prob, n_bins=10)
        for col in ("bin_low", "bin_high", "n_games", "mean_pred", "actual_rate", "gap"):
            assert col in result.columns, f"Missing column: {col}"

    def test_n_games_sums_to_total(self):
        """Total n_games across bins should equal the input length."""
        n = 50
        y_prob = np.linspace(0.1, 0.9, n)
        y_true = (np.random.default_rng(7).random(n) > 0.5).astype(int)
        result = _bin_calibration_stats(y_true, y_prob, n_bins=10)
        assert result["n_games"].sum() == n

    def test_mean_pred_in_bin_range(self):
        """mean_pred for each bin must be within [bin_low, bin_high]."""
        n = 100
        rng = np.random.default_rng(0)
        y_prob = rng.uniform(0, 1, n)
        y_true = (rng.random(n) > 0.5).astype(int)
        result = _bin_calibration_stats(y_true, y_prob, n_bins=10)
        for _, row in result.iterrows():
            assert row["bin_low"] <= row["mean_pred"] <= row["bin_high"] + 0.001

    def test_actual_rate_in_0_1(self):
        """actual_rate (win rate) must be in [0, 1] for all bins."""
        y_prob = np.array([0.4, 0.5, 0.6, 0.6, 0.7])
        y_true = np.array([0, 0, 1, 1, 1])
        result = _bin_calibration_stats(y_true, y_prob, n_bins=5)
        assert (result["actual_rate"] >= 0.0).all()
        assert (result["actual_rate"] <= 1.0).all()

    def test_gap_equals_mean_pred_minus_actual_rate(self):
        """gap = mean_pred - actual_rate for each bin."""
        y_prob = np.array([0.6, 0.65, 0.62])
        y_true = np.array([1, 1, 0])
        result = _bin_calibration_stats(y_true, y_prob, n_bins=5)
        for _, row in result.iterrows():
            expected_gap = row["mean_pred"] - row["actual_rate"]
            assert row["gap"] == pytest.approx(expected_gap, abs=1e-9)

    def test_empty_bins_excluded_from_output(self):
        """Bins with no predictions must not appear in output."""
        # All predictions in [0.6, 0.7] with 10 bins -> only 1 bin populated
        y_prob = np.array([0.65] * 10)
        y_true = np.array([1] * 6 + [0] * 4)
        result = _bin_calibration_stats(y_true, y_prob, n_bins=10)
        # Should be exactly 1 non-empty bin
        assert len(result) == 1
