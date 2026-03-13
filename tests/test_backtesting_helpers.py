"""
Tests for pure helper functions in src/models/backtesting.py

Covers:
  - _sorted_seasons: sorts season values from a DataFrame column
  - _get_feature_cols_game: excludes metadata columns, keeps numeric features
  - _get_feature_cols_player: excludes raw game stats and metadata columns
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.pipeline import Pipeline

from src.models.backtesting import (
    _sorted_seasons,
    _get_feature_cols_game,
    _get_feature_cols_player,
    _build_gb_classifier,
    _build_gb_regressor,
    TARGET_CLASS,
)


# ---------------------------------------------------------------------------
# _sorted_seasons
# ---------------------------------------------------------------------------

class TestSortedSeasons:
    def test_returns_sorted_list(self):
        df = pd.DataFrame({"season": [202425, 202324, 202223]})
        result = _sorted_seasons(df)
        assert result == sorted(result)

    def test_returns_unique_values(self):
        df = pd.DataFrame({"season": [202425, 202425, 202324]})
        result = _sorted_seasons(df)
        assert len(result) == len(set(result))

    def test_returns_int_values(self):
        df = pd.DataFrame({"season": [202425]})
        result = _sorted_seasons(df)
        assert all(isinstance(s, (int, type(result[0]))) for s in result)

    def test_custom_col(self):
        df = pd.DataFrame({"my_season": [202425, 202324]})
        result = _sorted_seasons(df, col="my_season")
        assert len(result) == 2

    def test_single_season(self):
        df = pd.DataFrame({"season": [202425]})
        result = _sorted_seasons(df)
        assert result == [202425]

    def test_empty_df_returns_empty_list(self):
        df = pd.DataFrame({"season": []})
        result = _sorted_seasons(df)
        assert result == []


# ---------------------------------------------------------------------------
# _get_feature_cols_game
# ---------------------------------------------------------------------------

class TestGetFeatureColsGame:
    def _make_df(self, extra_cols: dict | None = None) -> pd.DataFrame:
        """Make a DataFrame with standard metadata + numeric feature cols."""
        data = {
            TARGET_CLASS: [1, 0],
            "game_id": ["001", "002"],
            "season": [202425, 202425],
            "game_date": ["2026-01-01", "2026-01-02"],
            "home_team": ["LAL", "BOS"],
            "away_team": ["GSW", "NYK"],
            "net_rating": [5.0, -2.0],
            "win_pct": [0.6, 0.55],
        }
        if extra_cols:
            data.update(extra_cols)
        return pd.DataFrame(data)

    def test_excludes_target_column(self):
        df = self._make_df()
        result = _get_feature_cols_game(df)
        assert TARGET_CLASS not in result

    def test_excludes_game_id(self):
        df = self._make_df()
        result = _get_feature_cols_game(df)
        assert "game_id" not in result

    def test_excludes_season(self):
        df = self._make_df()
        result = _get_feature_cols_game(df)
        assert "season" not in result

    def test_excludes_game_date(self):
        df = self._make_df()
        result = _get_feature_cols_game(df)
        assert "game_date" not in result

    def test_excludes_team_columns(self):
        df = self._make_df()
        result = _get_feature_cols_game(df)
        assert "home_team" not in result
        assert "away_team" not in result

    def test_includes_numeric_features(self):
        df = self._make_df()
        result = _get_feature_cols_game(df)
        assert "net_rating" in result
        assert "win_pct" in result

    def test_excludes_string_columns(self):
        df = self._make_df({"team_name": ["Lakers", "Celtics"]})
        result = _get_feature_cols_game(df)
        assert "team_name" not in result

    def test_returns_list(self):
        df = self._make_df()
        result = _get_feature_cols_game(df)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# _get_feature_cols_player
# ---------------------------------------------------------------------------

class TestGetFeatureColsPlayer:
    def _make_player_df(self, target: str = "pts") -> pd.DataFrame:
        return pd.DataFrame({
            "player_id": [1], "player_name": ["A"], "team_id": [1],
            "team_abbreviation": ["LAL"], "game_id": ["001"],
            "game_date": ["2026-01-01"], "season": [202425],
            "matchup": ["LAL vs GSW"], "wl": ["W"],
            # raw game stats (should be excluded)
            "pts": [25.0], "reb": [5.0], "ast": [7.0],
            "min": [35.0], "fga": [20.0], "fg_pct": [0.5],
            # rolling features (should be included)
            "pts_roll10": [23.0], "reb_roll10": [4.8],
        })

    def test_excludes_raw_game_stats(self):
        df = self._make_player_df(target="pts")
        result = _get_feature_cols_player(df, target="pts")
        for raw in ["pts", "reb", "ast", "min", "fga"]:
            assert raw not in result, f"Raw stat '{raw}' should be excluded"

    def test_excludes_player_metadata(self):
        df = self._make_player_df(target="pts")
        result = _get_feature_cols_player(df, target="pts")
        for meta in ["player_id", "player_name", "team_id", "game_id", "game_date"]:
            assert meta not in result

    def test_includes_rolling_features(self):
        df = self._make_player_df(target="pts")
        result = _get_feature_cols_player(df, target="pts")
        assert "pts_roll10" in result
        assert "reb_roll10" in result

    def test_returns_list(self):
        df = self._make_player_df(target="pts")
        result = _get_feature_cols_player(df, target="pts")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# _build_gb_classifier
# ---------------------------------------------------------------------------

class TestBuildGbClassifier:
    def test_returns_pipeline(self):
        pipe = _build_gb_classifier()
        assert isinstance(pipe, Pipeline)

    def test_has_imputer_step(self):
        pipe = _build_gb_classifier()
        assert "imputer" in pipe.named_steps

    def test_has_clf_step(self):
        pipe = _build_gb_classifier()
        assert "clf" in pipe.named_steps

    def test_can_fit_and_predict(self):
        import numpy as np
        pipe = _build_gb_classifier()
        X = np.random.default_rng(0).standard_normal((50, 4))
        y = np.random.default_rng(0).integers(0, 2, 50)
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert len(preds) == 50
        assert set(preds).issubset({0, 1})

    def test_predict_proba_available(self):
        import numpy as np
        pipe = _build_gb_classifier()
        X = np.random.default_rng(1).standard_normal((50, 3))
        y = np.random.default_rng(1).integers(0, 2, 50)
        pipe.fit(X, y)
        probas = pipe.predict_proba(X)
        assert probas.shape == (50, 2)
        assert ((probas >= 0) & (probas <= 1)).all()


# ---------------------------------------------------------------------------
# _build_gb_regressor
# ---------------------------------------------------------------------------

class TestBuildGbRegressor:
    def test_returns_pipeline(self):
        pipe = _build_gb_regressor()
        assert isinstance(pipe, Pipeline)

    def test_has_imputer_step(self):
        pipe = _build_gb_regressor()
        assert "imputer" in pipe.named_steps

    def test_has_model_step(self):
        pipe = _build_gb_regressor()
        assert "model" in pipe.named_steps

    def test_can_fit_and_predict(self):
        import numpy as np
        pipe = _build_gb_regressor()
        X = np.random.default_rng(2).standard_normal((60, 4))
        y = np.random.default_rng(2).standard_normal(60)
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert len(preds) == 60
        assert all(isinstance(p, float) for p in preds)

    def test_handles_nan_inputs_via_imputer(self):
        import numpy as np
        pipe = _build_gb_regressor()
        rng = np.random.default_rng(3)
        X = rng.standard_normal((60, 4))
        X[5, 0] = np.nan  # inject NaN — imputer should handle it
        y = rng.standard_normal(60)
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert len(preds) == 60
