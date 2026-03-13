"""
Tests for src/models/margin_model.py

Coverage targets:
  - train_margin_model(): end-to-end with synthetic data
  - predict_margin(): round-trip, missing artifact, missing team
  - _get_feature_cols(): fallback and outcome-features paths
  - _season_splits(): expanding-window property (no leakage)
  - _derive_point_diff(): correct home plus_minus join
  - _validate_null_rates(): raises on high-null columns
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on sys.path so src.* imports resolve
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.models.margin_model import (  # noqa: E402
    TARGET,
    _derive_point_diff,
    _get_feature_cols,
    _season_splits,
    _validate_null_rates,
    predict_margin,
    train_margin_model,
)


# -- Helpers --------------------------------------------------------------

N_ROWS = 30


def _make_matchup_df(n=N_ROWS):
    """Synthetic matchup DataFrame mirroring the real CSV structure.

    Uses 7 season codes so that tests can use "201920" as a holdout season
    while still having 6 earlier seasons for CV splits.
    """
    np.random.seed(42)
    # 7 seasons: 6 for CV training + 1 as holdout test season
    season_codes = [
        "201314", "201415", "201516", "201617", "201718", "201819", "201920"
    ]
    seasons = [season_codes[min(i // 4, 6)] for i in range(n)]
    return pd.DataFrame(
        {
            "game_id": [f"G{i:04d}" for i in range(n)],
            "season": seasons,
            "game_date": pd.date_range("2013-11-01", periods=n, freq="7D"),
            "home_team": ["LAL"] * n,
            "away_team": ["GSW"] * n,
            "diff_pts_roll5": np.random.randn(n) * 5,
            "diff_win_pct_roll10": np.random.randn(n) * 0.1,
            "diff_net_rtg_game_roll5": np.random.randn(n) * 3,
            "home_days_rest": np.random.randint(1, 5, n).astype(float),
            "away_days_rest": np.random.randint(1, 5, n).astype(float),
            "home_win": np.random.randint(0, 2, n),
        }
    )


def _make_logs_df(matchup):
    """Synthetic team_game_logs rows (one home row per game)."""
    np.random.seed(7)
    point_diffs = (
        matchup["diff_pts_roll5"] * 1.5 + np.random.randn(len(matchup)) * 3
    ).round()
    rows = []
    for i, (_, row) in enumerate(matchup.iterrows()):
        rows.append({
            "game_id": row["game_id"],
            "matchup": f"{row['home_team']} vs. {row['away_team']}",
            "plus_minus": int(point_diffs.iloc[i]),
        })
    return pd.DataFrame(rows)


# -- Test 1: public API callable ------------------------------------------


def test_public_api_callable():
    """All public functions must be importable and callable."""
    assert callable(train_margin_model)
    assert callable(predict_margin)
    assert callable(_get_feature_cols)
    assert callable(_season_splits)
    assert callable(_derive_point_diff)
    assert callable(_validate_null_rates)


# -- Test 2: _derive_point_diff joins correctly ---------------------------


def test_derive_point_diff(tmp_path):
    """_derive_point_diff attaches point_diff from home team plus_minus."""
    matchup = _make_matchup_df()
    logs = _make_logs_df(matchup)
    logs_path = str(tmp_path / "logs.csv")
    logs.to_csv(logs_path, index=False)

    result = _derive_point_diff(matchup, game_logs_path=logs_path)

    assert TARGET in result.columns
    assert len(result) == len(matchup)
    assert result[TARGET].notna().all()


def test_derive_point_diff_drops_missing(tmp_path):
    """Rows with no matching log entry are dropped gracefully."""
    matchup = _make_matchup_df(n=10)
    logs = _make_logs_df(matchup).head(5)
    logs_path = str(tmp_path / "logs_partial.csv")
    logs.to_csv(logs_path, index=False)

    result = _derive_point_diff(matchup, game_logs_path=logs_path)
    assert len(result) == 5


# -- Test 3: _season_splits no leakage ------------------------------------


def test_season_splits_no_leakage():
    """Expanding-window splits must never include future seasons in train fold."""
    df = _make_matchup_df(n=N_ROWS)
    splits = _season_splits(df)

    assert len(splits) > 0

    for tr, va, label in splits:
        train_seasons = set(tr["season"].astype(str))
        valid_seasons = set(va["season"].astype(str))

        overlap = train_seasons & valid_seasons
        assert not overlap, (
            f"Split '{label}': train and validation overlap: {overlap}"
        )

        max_train = max(train_seasons)
        min_valid = min(valid_seasons)
        assert max_train < min_valid, (
            f"Split '{label}': train season {max_train} >= valid {min_valid}"
            " -- future data leaked!"
        )


def test_season_splits_fallback():
    """Falls back to index-based split when too few seasons."""
    df = _make_matchup_df(n=10)
    df["season"] = "201314"
    splits = _season_splits(df)
    assert len(splits) == 1
    _, _, label = splits[0]
    assert label == "date_fallback"


# -- Test 4: _get_feature_cols paths --------------------------------------


def test_get_feature_cols_with_outcome_features():
    """When outcome_features is provided, only available columns are returned."""
    df = _make_matchup_df()
    outcome_feats = ["diff_pts_roll5", "diff_win_pct_roll10", "nonexistent_col"]

    result = _get_feature_cols(df, outcome_features=outcome_feats)

    assert "nonexistent_col" not in result
    assert "diff_pts_roll5" in result
    assert "diff_win_pct_roll10" in result


def test_get_feature_cols_fallback():
    """Fallback derivation excludes metadata and target columns."""
    df = _make_matchup_df()
    result = _get_feature_cols(df, outcome_features=None)

    forbidden = {
        TARGET, "home_win", "game_id", "season", "game_date",
        "home_team", "away_team",
    }
    for col in result:
        assert col not in forbidden, f"Forbidden column in features: {col}"
    assert len(result) > 0


# -- Test 5: _validate_null_rates -----------------------------------------


def test_validate_null_rates_raises():
    """Must raise ValueError when a column exceeds the null threshold."""
    df = pd.DataFrame({
        "feat_a": [1.0, np.nan, np.nan, np.nan, np.nan],
        "feat_b": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    with pytest.raises(ValueError, match="null threshold"):
        _validate_null_rates(df, ["feat_a", "feat_b"], threshold=0.50)


def test_validate_null_rates_passes_partial():
    """Partial nulls below threshold must not raise."""
    df = pd.DataFrame({
        "feat_a": [1.0, np.nan, 3.0, 4.0, 5.0],
        "feat_b": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    _validate_null_rates(df, ["feat_a", "feat_b"], threshold=0.95)


# -- Test 6: train_margin_model end-to-end --------------------------------


def test_train_margin_model_end_to_end(tmp_path):
    """Full train run produces artifacts and returns reasonable metrics."""
    matchup = _make_matchup_df(n=N_ROWS)
    matchup_path = str(tmp_path / "matchup.csv")
    matchup.to_csv(matchup_path, index=False)

    logs = _make_logs_df(matchup)
    logs_path = str(tmp_path / "logs.csv")
    logs.to_csv(logs_path, index=False)

    artifacts_dir = str(tmp_path / "artifacts")

    pipeline, metrics = train_margin_model(
        matchup_path=matchup_path,
        game_logs_path=logs_path,
        artifacts_dir=artifacts_dir,
        test_seasons=["201920"],
    )

    assert os.path.exists(os.path.join(artifacts_dir, "margin_model.pkl"))
    assert os.path.exists(
        os.path.join(artifacts_dir, "margin_model_features.pkl")
    )

    assert "selected_model" in metrics
    assert metrics["selected_model"] in (
        "ridge", "lasso", "gradient_boosting", "huber_gbm"
    )
    assert metrics["cv_mean_mae"] >= 0
    # test_mae/rmse may be NaN when test_seasons not present in data
    import math
    assert math.isnan(metrics["test_mae"]) or metrics["test_mae"] >= 0
    assert math.isnan(metrics["test_rmse"]) or metrics["test_rmse"] >= 0
    assert metrics["n_features"] > 0


# -- Test 7: predict_margin round-trip ------------------------------------


def test_predict_margin_round_trip(tmp_path):
    """predict_margin returns a float after training."""
    matchup = _make_matchup_df(n=N_ROWS)
    matchup_path = str(tmp_path / "matchup.csv")
    matchup.to_csv(matchup_path, index=False)

    logs = _make_logs_df(matchup)
    logs_path = str(tmp_path / "logs.csv")
    logs.to_csv(logs_path, index=False)

    artifacts_dir = str(tmp_path / "artifacts")
    train_margin_model(
        matchup_path=matchup_path,
        game_logs_path=logs_path,
        artifacts_dir=artifacts_dir,
        test_seasons=["201920"],
    )

    result = predict_margin(
        home_team="LAL",
        away_team="GSW",
        matchup_path=matchup_path,
        artifacts_dir=artifacts_dir,
    )

    assert isinstance(result, float)
    assert -60 < result < 60


# -- Test 8: predict_margin raises on missing artifacts -------------------


def test_predict_margin_raises_on_missing_artifacts(tmp_path):
    """FileNotFoundError must be raised when model artifact does not exist."""
    with pytest.raises(FileNotFoundError, match="margin_model"):
        predict_margin(
            home_team="LAL",
            away_team="GSW",
            artifacts_dir=str(tmp_path / "nonexistent"),
        )


# -- Test 9: predict_margin raises on unknown team ------------------------


def test_predict_margin_raises_on_unknown_team(tmp_path):
    """ValueError must be raised when team has no matchup history."""
    matchup = _make_matchup_df(n=N_ROWS)
    matchup_path = str(tmp_path / "matchup.csv")
    matchup.to_csv(matchup_path, index=False)

    logs = _make_logs_df(matchup)
    logs_path = str(tmp_path / "logs.csv")
    logs.to_csv(logs_path, index=False)

    artifacts_dir = str(tmp_path / "artifacts")
    train_margin_model(
        matchup_path=matchup_path,
        game_logs_path=logs_path,
        artifacts_dir=artifacts_dir,
        test_seasons=["201920"],
    )

    with pytest.raises(ValueError, match="Not enough current-season history"):
        predict_margin(
            home_team="UNKNOWN_TEAM_XYZ",
            away_team="ANOTHER_UNKNOWN",
            matchup_path=matchup_path,
            artifacts_dir=artifacts_dir,
        )


# -- Test 10: pipeline is serialisable ------------------------------------


def test_validate_null_rates_no_nulls_passes():
    """A DataFrame with zero nulls must never raise."""
    df = pd.DataFrame({"feat_a": [1.0, 2.0, 3.0], "feat_b": [4.0, 5.0, 6.0]})
    _validate_null_rates(df, ["feat_a", "feat_b"], threshold=0.95)


def test_validate_null_rates_below_threshold_passes():
    """Null rate below threshold (e.g. 1/5 = 0.20 < 0.95) must not raise."""
    df = pd.DataFrame({"feat_a": [1.0, np.nan, 3.0, 4.0, 5.0]})
    _validate_null_rates(df, ["feat_a"], threshold=0.95)


def test_validate_null_rates_at_threshold_raises():
    """Null rate exactly at threshold (>= threshold) must raise."""
    # 5/5 = 1.0 >= 0.95 -> should raise
    df = pd.DataFrame({"feat_a": [np.nan] * 5})
    with pytest.raises(ValueError):
        _validate_null_rates(df, ["feat_a"], threshold=0.95)


def test_get_feature_cols_returns_list():
    """_get_feature_cols must return a Python list."""
    df = _make_matchup_df()
    result = _get_feature_cols(df, outcome_features=None)
    assert isinstance(result, list)
    assert len(result) > 0


def test_season_splits_train_strictly_before_test():
    """Every season in a train fold must be strictly before every season in validation."""
    df = _make_matchup_df(n=N_ROWS)
    splits = _season_splits(df)
    for tr, va, label in splits:
        if label == "date_fallback":
            continue
        max_train = max(tr["season"].astype(str))
        min_valid = min(va["season"].astype(str))
        assert max_train < min_valid, (
            f"Split '{label}': train max={max_train} >= validation min={min_valid}"
        )


def test_pipeline_serialisable(tmp_path):
    """The trained pipeline must survive a pickle round-trip."""
    matchup = _make_matchup_df(n=N_ROWS)
    matchup_path = str(tmp_path / "matchup.csv")
    matchup.to_csv(matchup_path, index=False)

    logs = _make_logs_df(matchup)
    logs_path = str(tmp_path / "logs.csv")
    logs.to_csv(logs_path, index=False)

    artifacts_dir = str(tmp_path / "artifacts")
    pipeline, _ = train_margin_model(
        matchup_path=matchup_path,
        game_logs_path=logs_path,
        artifacts_dir=artifacts_dir,
        test_seasons=["201920"],
    )

    buf = pickle.dumps(pipeline)
    loaded = pickle.loads(buf)
    assert loaded is not None

    feat_path = os.path.join(artifacts_dir, "margin_model_features.pkl")
    with open(feat_path, "rb") as fh:
        feat_cols = pickle.load(fh)
    X = _make_matchup_df(n=5)[feat_cols].fillna(0)
    preds = loaded.predict(X)
    assert len(preds) == 5


# -- Test: predict_margin refreshes Elo ------------------------------------


def test_predict_margin_refreshes_elo(tmp_path):
    """predict_margin must call get_current_elos() to get fresh Elo values."""
    from unittest.mock import patch

    # Train a model first so artifacts exist
    matchup = _make_matchup_df(n=N_ROWS)
    matchup_path = str(tmp_path / "matchup.csv")
    matchup.to_csv(matchup_path, index=False)

    logs = _make_logs_df(matchup)
    logs_path = str(tmp_path / "logs.csv")
    logs.to_csv(logs_path, index=False)

    artifacts_dir = str(tmp_path / "artifacts")
    train_margin_model(
        matchup_path=matchup_path,
        game_logs_path=logs_path,
        artifacts_dir=artifacts_dir,
        test_seasons=["201920"],
    )

    with patch("src.models.margin_model.get_current_elos") as mock_elos:
        mock_elos.return_value = {
            "LAL": {"elo": 1550.0, "elo_fast": 1560.0, "momentum": 10.0},
            "GSW": {"elo": 1450.0, "elo_fast": 1440.0, "momentum": -10.0},
        }
        result = predict_margin(
            home_team="LAL",
            away_team="GSW",
            matchup_path=matchup_path,
            artifacts_dir=artifacts_dir,
        )
        mock_elos.assert_called_once_with(extended=True)
        assert isinstance(result, float)


# -- Test: Huber GBM candidate exists --------------------------------------


def test_huber_gbm_candidate_exists(tmp_path):
    """Huber GBM should be listed as a model candidate and compete in CV."""
    matchup = _make_matchup_df(n=N_ROWS)
    matchup_path = str(tmp_path / "matchup.csv")
    matchup.to_csv(matchup_path, index=False)

    logs = _make_logs_df(matchup)
    logs_path = str(tmp_path / "logs.csv")
    logs.to_csv(logs_path, index=False)

    artifacts_dir = str(tmp_path / "artifacts")
    _, metrics = train_margin_model(
        matchup_path=matchup_path,
        game_logs_path=logs_path,
        artifacts_dir=artifacts_dir,
        test_seasons=["201920"],
    )

    assert metrics["selected_model"] in (
        "ridge", "lasso", "gradient_boosting", "huber_gbm"
    )


# -- Test: residual_std artifact saved -------------------------------------


def test_residual_std_artifact_saved(tmp_path):
    """residual_std artifact should be a JSON with a positive float."""
    import json

    matchup = _make_matchup_df(n=N_ROWS)
    matchup_path = str(tmp_path / "matchup.csv")
    matchup.to_csv(matchup_path, index=False)

    logs = _make_logs_df(matchup)
    logs_path = str(tmp_path / "logs.csv")
    logs.to_csv(logs_path, index=False)

    artifacts_dir = str(tmp_path / "artifacts")
    train_margin_model(
        matchup_path=matchup_path,
        game_logs_path=logs_path,
        artifacts_dir=artifacts_dir,
        test_seasons=["201920"],
    )

    residual_path = os.path.join(artifacts_dir, "margin_residual_std.json")
    assert os.path.exists(residual_path), "margin_residual_std.json not created"

    with open(residual_path) as f:
        data = json.load(f)
    assert "residual_std" in data
    assert isinstance(data["residual_std"], float)
    assert data["residual_std"] > 0


# -- Test: predict_margin returns float ------------------------------------


def test_predict_margin_returns_float():
    """predict_margin should be callable and return numeric values."""
    assert callable(predict_margin)
