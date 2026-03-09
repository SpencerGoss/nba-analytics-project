"""
Tests for pure helper functions in src/models/game_outcome_model.py

Covers:
  - validate_feature_null_rates: raises on high-null columns, passes otherwise
  - get_feature_cols: column selection logic (diff_, schedule, injury context)
  - _best_threshold: finds threshold that maximizes accuracy on binary labels
  - _season_splits: expanding-window CV split structure
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.game_outcome_model import (
    TARGET,
    MIN_TRAIN_SEASONS_FOR_TUNING,
    validate_feature_null_rates,
    get_feature_cols,
    _best_threshold,
    _season_splits,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(extra_cols: dict | None = None) -> pd.DataFrame:
    """Minimal matchup DataFrame for feature-col tests."""
    data = {
        TARGET: [1, 0] * 5,
        "game_id": [f"G{i:03d}" for i in range(10)],
        "season": ["202425"] * 10,
        "game_date": ["2025-01-01"] * 10,
        "home_team": ["LAL"] * 10,
        "away_team": ["GSW"] * 10,
        # diff_ features
        "diff_net_rtg_roll5": np.random.randn(10),
        "diff_win_pct_roll10": np.random.randn(10),
        # schedule context
        "home_days_rest": np.random.randint(1, 5, 10).astype(float),
        "away_days_rest": np.random.randint(1, 5, 10).astype(float),
        # injury context
        "home_missing_minutes": np.random.randn(10),
        "away_missing_minutes": np.random.randn(10),
        "home_star_player_out": np.random.randint(0, 2, 10).astype(float),
        "away_star_player_out": np.random.randint(0, 2, 10).astype(float),
        # should be excluded
        "string_col": ["x"] * 10,
    }
    if extra_cols:
        data.update(extra_cols)
    return pd.DataFrame(data)


def _make_multi_season_df(n_seasons: int = 8, rows_per_season: int = 5) -> pd.DataFrame:
    """Build a DataFrame with n_seasons distinct season codes."""
    season_codes = [f"{2010 + i}{11 + i:02d}" for i in range(n_seasons)]
    rows = []
    for code in season_codes:
        for _ in range(rows_per_season):
            rows.append({
                "season": code,
                "diff_net_rtg": np.random.randn(),
                TARGET: np.random.randint(0, 2),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# validate_feature_null_rates
# ---------------------------------------------------------------------------

class TestValidateFeatureNullRates:
    def test_raises_when_column_exceeds_threshold(self):
        df = pd.DataFrame({
            "feat_a": [np.nan] * 10 + [1.0] * 2,   # ~83% null
            "feat_b": [1.0] * 12,
        })
        with pytest.raises(ValueError):
            validate_feature_null_rates(df, ["feat_a", "feat_b"], threshold=0.50)

    def test_passes_when_null_below_threshold(self):
        df = pd.DataFrame({
            "feat_a": [1.0, np.nan] + [1.0] * 8,   # 10% null
            "feat_b": [1.0] * 10,
        })
        # Should not raise
        validate_feature_null_rates(df, ["feat_a", "feat_b"], threshold=0.95)

    def test_all_null_raises(self):
        df = pd.DataFrame({"feat": [np.nan] * 10})
        with pytest.raises(ValueError):
            validate_feature_null_rates(df, ["feat"], threshold=0.50)

    def test_zero_nulls_passes(self):
        df = pd.DataFrame({"feat": [1.0, 2.0, 3.0]})
        validate_feature_null_rates(df, ["feat"], threshold=0.01)

    def test_error_message_contains_column_name(self):
        df = pd.DataFrame({"bad_col": [np.nan] * 10})
        with pytest.raises(ValueError, match="bad_col"):
            validate_feature_null_rates(df, ["bad_col"], threshold=0.50)

    def test_exactly_at_threshold_raises(self):
        """Null rate == threshold should raise (>= threshold)."""
        df = pd.DataFrame({"feat": [np.nan, np.nan, 1.0, 1.0]})  # 50% null
        with pytest.raises(ValueError):
            validate_feature_null_rates(df, ["feat"], threshold=0.50)


# ---------------------------------------------------------------------------
# get_feature_cols
# ---------------------------------------------------------------------------

class TestGetFeatureCols:
    def test_excludes_target(self):
        df = _make_df()
        result = get_feature_cols(df)
        assert TARGET not in result

    def test_excludes_metadata(self):
        df = _make_df()
        result = get_feature_cols(df)
        for meta in ("game_id", "season", "game_date", "home_team", "away_team"):
            assert meta not in result, f"Metadata col should be excluded: {meta}"

    def test_includes_diff_cols(self):
        df = _make_df()
        result = get_feature_cols(df)
        assert "diff_net_rtg_roll5" in result
        assert "diff_win_pct_roll10" in result

    def test_includes_schedule_context(self):
        df = _make_df()
        result = get_feature_cols(df)
        assert "home_days_rest" in result
        assert "away_days_rest" in result

    def test_includes_injury_context(self):
        df = _make_df()
        result = get_feature_cols(df)
        assert "home_missing_minutes" in result
        assert "away_missing_minutes" in result
        assert "home_star_player_out" in result
        assert "away_star_player_out" in result

    def test_excludes_string_columns(self):
        df = _make_df()
        result = get_feature_cols(df)
        assert "string_col" not in result

    def test_returns_list(self):
        df = _make_df()
        result = get_feature_cols(df)
        assert isinstance(result, list)

    def test_fallback_when_no_diff_cols(self):
        """Without diff_ cols, all numeric non-metadata columns are returned."""
        df = pd.DataFrame({
            TARGET: [1, 0],
            "game_id": ["001", "002"],
            "season": [202425, 202425],
            "game_date": ["2025-01-01", "2025-01-02"],
            "home_team": ["LAL", "BOS"],
            "away_team": ["GSW", "NYK"],
            "some_numeric": [1.0, 2.0],
        })
        result = get_feature_cols(df)
        assert "some_numeric" in result
        assert TARGET not in result

    def test_sorted_output(self):
        """Result should be sorted (sorted set of diff + context cols)."""
        df = _make_df()
        result = get_feature_cols(df)
        assert result == sorted(result)


# ---------------------------------------------------------------------------
# _best_threshold
# ---------------------------------------------------------------------------

class TestBestThreshold:
    def test_returns_tuple(self):
        y_true = pd.Series([1, 0, 1, 0, 1, 1, 0, 0, 1, 0])
        proba = np.array([0.7, 0.3, 0.8, 0.2, 0.6, 0.9, 0.4, 0.1, 0.75, 0.35])
        result = _best_threshold(y_true, proba)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_threshold_in_range(self):
        """Threshold must be between 0.35 and 0.65."""
        y_true = pd.Series([1, 0] * 10)
        proba = np.random.default_rng(42).uniform(0.3, 0.7, 20)
        t, acc = _best_threshold(y_true, proba)
        assert 0.35 <= t <= 0.65

    def test_accuracy_in_0_1_range(self):
        y_true = pd.Series([1, 0] * 5)
        proba = np.array([0.8, 0.2] * 5)
        _, acc = _best_threshold(y_true, proba)
        assert 0.0 <= acc <= 1.0

    def test_perfect_prediction_finds_high_accuracy(self):
        """With perfectly separable data, best accuracy should be 1.0."""
        y_true = pd.Series([1, 1, 1, 1, 0, 0, 0, 0])
        proba = np.array([0.9, 0.8, 0.85, 0.95, 0.1, 0.15, 0.2, 0.05])
        _, acc = _best_threshold(y_true, proba)
        assert acc == pytest.approx(1.0, abs=0.01)

    def test_threshold_is_float(self):
        y_true = pd.Series([1, 0] * 5)
        proba = np.array([0.6, 0.4] * 5)
        t, _ = _best_threshold(y_true, proba)
        assert isinstance(t, float)

    def test_default_threshold_is_reasonable(self):
        """Even with random data, returns a threshold (at least 0.5 default)."""
        rng = np.random.default_rng(99)
        y_true = pd.Series(rng.integers(0, 2, 20))
        proba = rng.uniform(0, 1, 20)
        t, acc = _best_threshold(y_true, proba)
        assert 0.35 <= t <= 0.65
        assert 0.0 <= acc <= 1.0


# ---------------------------------------------------------------------------
# _season_splits
# ---------------------------------------------------------------------------

class TestSeasonSplits:
    def test_splits_not_empty(self):
        df = _make_multi_season_df(n_seasons=8)
        splits = _season_splits(df)
        assert len(splits) > 0

    def test_no_leakage_train_before_valid(self):
        """All train seasons must precede the validation season."""
        df = _make_multi_season_df(n_seasons=9)
        splits = _season_splits(df)
        for tr, va, label in splits:
            if label == "date_fallback":
                continue
            train_seasons = set(tr["season"].astype(str))
            valid_seasons = set(va["season"].astype(str))
            assert not (train_seasons & valid_seasons)
            assert max(train_seasons) < min(valid_seasons)

    def test_fallback_when_too_few_seasons(self):
        """Single-season data falls back to index-based split."""
        df = pd.DataFrame({
            "season": ["202425"] * 20,
            "diff_net_rtg": np.random.randn(20),
            TARGET: np.random.randint(0, 2, 20),
        })
        splits = _season_splits(df)
        assert len(splits) == 1
        _, _, label = splits[0]
        assert label == "date_fallback"

    def test_split_count_increases_with_more_seasons(self):
        df_few = _make_multi_season_df(n_seasons=6)
        df_many = _make_multi_season_df(n_seasons=10)
        splits_few = _season_splits(df_few)
        splits_many = _season_splits(df_many)
        assert len(splits_many) > len(splits_few)

    def test_each_split_has_three_elements(self):
        df = _make_multi_season_df(n_seasons=8)
        splits = _season_splits(df)
        for split in splits:
            assert len(split) == 3  # (train, valid, label)

    def test_no_empty_train_or_valid(self):
        df = _make_multi_season_df(n_seasons=8)
        splits = _season_splits(df)
        for tr, va, _ in splits:
            assert not tr.empty
            assert not va.empty
