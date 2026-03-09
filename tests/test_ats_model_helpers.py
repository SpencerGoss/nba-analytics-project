"""
Tests for pure helper functions in src/models/ats_model.py

Covers:
  - get_ats_feature_cols: column selection logic
  - _ats_season_splits: expanding-window validation split structure
  - _validate_null_rates: raises on high-null columns
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ats_model import (
    TARGET,
    MIN_TRAIN_SEASONS,
    get_ats_feature_cols,
    _ats_season_splits,
    _validate_null_rates,
)


# ---------------------------------------------------------------------------
# get_ats_feature_cols
# ---------------------------------------------------------------------------

def _make_ats_df(extra_cols: dict | None = None) -> pd.DataFrame:
    """Minimal DataFrame matching ATS training schema."""
    data = {
        TARGET: [1, 0] * 5,
        "game_id": [f"G{i:03d}" for i in range(10)],
        "season": ["202425"] * 10,
        "game_date": ["2025-01-01"] * 10,
        "home_team": ["LAL"] * 10,
        "away_team": ["GSW"] * 10,
        "home_win": [1, 0] * 5,
        # Feature cols
        "diff_net_rtg_roll5": np.random.randn(10),
        "diff_win_pct_roll10": np.random.randn(10),
        "home_days_rest": np.random.randint(1, 5, 10).astype(float),
        "away_days_rest": np.random.randint(1, 5, 10).astype(float),
        "spread": np.random.randn(10),
        "home_implied_prob": np.random.uniform(0.4, 0.7, 10),
        "away_implied_prob": np.random.uniform(0.3, 0.6, 10),
        "string_col": ["x"] * 10,  # should be excluded
    }
    if extra_cols:
        data.update(extra_cols)
    return pd.DataFrame(data)


class TestGetAtsFeatureCols:
    def test_excludes_target(self):
        df = _make_ats_df()
        result = get_ats_feature_cols(df)
        assert TARGET not in result

    def test_excludes_metadata_cols(self):
        df = _make_ats_df()
        result = get_ats_feature_cols(df)
        for meta in ("game_id", "season", "game_date", "home_team", "away_team", "home_win"):
            assert meta not in result, f"Metadata col should be excluded: {meta}"

    def test_includes_diff_cols(self):
        df = _make_ats_df()
        result = get_ats_feature_cols(df)
        assert "diff_net_rtg_roll5" in result
        assert "diff_win_pct_roll10" in result

    def test_includes_market_signals(self):
        df = _make_ats_df()
        result = get_ats_feature_cols(df)
        for col in ("spread", "home_implied_prob", "away_implied_prob"):
            assert col in result, f"ATS market signal should be included: {col}"

    def test_excludes_string_columns(self):
        df = _make_ats_df()
        result = get_ats_feature_cols(df)
        assert "string_col" not in result

    def test_returns_list(self):
        df = _make_ats_df()
        result = get_ats_feature_cols(df)
        assert isinstance(result, list)

    def test_includes_schedule_cols(self):
        df = _make_ats_df()
        result = get_ats_feature_cols(df)
        assert "home_days_rest" in result
        assert "away_days_rest" in result


# ---------------------------------------------------------------------------
# _ats_season_splits
# ---------------------------------------------------------------------------

def _make_multi_season_df(n_seasons: int = 6, rows_per_season: int = 5) -> pd.DataFrame:
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


class TestAtsSeason_Splits:
    def test_splits_not_empty(self):
        df = _make_multi_season_df(n_seasons=6)
        splits = _ats_season_splits(df, min_train=4)
        assert len(splits) > 0

    def test_no_leakage_train_before_valid(self):
        """Train seasons must all come before the validation season."""
        df = _make_multi_season_df(n_seasons=7)
        splits = _ats_season_splits(df, min_train=4)
        for tr, va, label in splits:
            if label == "date_fallback":
                continue
            train_seasons = set(tr["season"].astype(str))
            valid_seasons = set(va["season"].astype(str))
            assert not (train_seasons & valid_seasons), "Train/valid seasons must not overlap"
            assert max(train_seasons) < min(valid_seasons), "All train seasons must be before valid"

    def test_fallback_when_too_few_seasons(self):
        """With only 1 season, splits fall back to index-based split."""
        df = pd.DataFrame({
            "season": ["202425"] * 20,
            "diff_net_rtg": np.random.randn(20),
            TARGET: np.random.randint(0, 2, 20),
        })
        splits = _ats_season_splits(df, min_train=4)
        assert len(splits) == 1
        _, _, label = splits[0]
        assert label == "date_fallback"

    def test_split_count_increases_with_more_seasons(self):
        df_few = _make_multi_season_df(n_seasons=5)
        df_many = _make_multi_season_df(n_seasons=8)
        splits_few = _ats_season_splits(df_few, min_train=4)
        splits_many = _ats_season_splits(df_many, min_train=4)
        assert len(splits_many) > len(splits_few)


# ---------------------------------------------------------------------------
# _validate_null_rates
# ---------------------------------------------------------------------------

class TestValidateNullRates:
    def test_raises_when_null_exceeds_threshold(self):
        df = pd.DataFrame({
            "feat_a": [np.nan] * 10 + [1.0] * 2,   # ~83% null
            "feat_b": [1.0] * 12,
        })
        with pytest.raises(ValueError):
            _validate_null_rates(df, ["feat_a", "feat_b"], threshold=0.50)

    def test_passes_when_null_below_threshold(self):
        df = pd.DataFrame({
            "feat_a": [1.0, np.nan, 3.0] + [1.0] * 7,  # 10% null
            "feat_b": [1.0] * 10,
        })
        # Should not raise
        _validate_null_rates(df, ["feat_a", "feat_b"], threshold=0.95)

    def test_all_null_raises(self):
        df = pd.DataFrame({"feat": [np.nan] * 10})
        with pytest.raises(ValueError):
            _validate_null_rates(df, ["feat"], threshold=0.50)

    def test_zero_nulls_passes(self):
        df = pd.DataFrame({"feat": [1.0, 2.0, 3.0]})
        _validate_null_rates(df, ["feat"], threshold=0.01)
