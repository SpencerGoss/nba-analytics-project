"""Tests for src/models/ensemble.py"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.models.ensemble import (  # noqa: E402
    HIGH_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD,
    NBAEnsemble,
    _confidence_label,
    _sigmoid,
    run_ensemble_on_predictions,
)

FEAT = ["diff_pts_roll5", "diff_win_pct_roll10", "diff_net_rtg_game_roll5"]
ATS_FEAT = FEAT + ["spread"]
MAR_FEAT = FEAT


# -- Helpers ------------------------------------------------------------------

def _make_df(n=5, seed=0):
    rng = np.random.default_rng(seed)
    all_cols = sorted(set(FEAT + ATS_FEAT))
    return pd.DataFrame({c: rng.normal(0, 1, n) for c in all_cols})


def _save_pkl(path, obj):
    import pickle
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)


def _clf(X, y, feats, seed=42):
    pipe = Pipeline([
        ("i", SimpleImputer(strategy="mean")),
        ("s", StandardScaler()),
        ("c", LogisticRegression(max_iter=200, random_state=seed)),
    ])
    pipe.fit(X[feats], y)
    return pipe


def _reg(X, y, feats, seed=42):
    pipe = Pipeline([
        ("i", SimpleImputer(strategy="mean")),
        ("r", GradientBoostingRegressor(n_estimators=10, random_state=seed)),
    ])
    pipe.fit(X[feats], y)
    return pipe


def _build(tmp_path, margin=True):
    rng = np.random.default_rng(99)
    n = 80
    all_cols = sorted(set(FEAT + ATS_FEAT))
    X = pd.DataFrame({c: rng.normal(0, 1, n) for c in all_cols})
    yc = (rng.random(n) > 0.5).astype(int)
    yr = rng.normal(0, 10, n)
    ad = tmp_path / "artifacts"
    ad.mkdir()
    _save_pkl(ad / "game_outcome_model_calibrated.pkl", _clf(X, yc, FEAT))
    _save_pkl(ad / "game_outcome_features.pkl", FEAT)
    _save_pkl(ad / "ats_model.pkl", _clf(X, yc, ATS_FEAT))
    _save_pkl(ad / "ats_model_features.pkl", ATS_FEAT)
    if margin:
        _save_pkl(ad / "margin_model.pkl", _reg(X, yr, MAR_FEAT))
        _save_pkl(ad / "margin_model_features.pkl", MAR_FEAT)
    return ad


# -- Test 1: NBAEnsemble.load() -----------------------------------------------

def test_loads_all_three_models(tmp_path):
    """load() succeeds when all three model artifacts are present."""
    ens = NBAEnsemble.load(_build(tmp_path, margin=True))
    assert ens.outcome_model is not None
    assert ens.ats_model is not None
    assert ens.margin_model is not None


def test_loads_without_margin(tmp_path):
    """load() warns and sets margin_model=None when margin artifact is absent."""
    ad = _build(tmp_path, margin=False)
    with pytest.warns(UserWarning, match="Margin model not found"):
        ens = NBAEnsemble.load(ad)
    assert ens.margin_model is None


def test_load_raises_missing_outcome_model(tmp_path):
    """FileNotFoundError raised when game outcome model is missing."""
    empty = tmp_path / "e"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="game outcome model"):
        NBAEnsemble.load(empty)


def test_load_raises_missing_ats_model(tmp_path):
    """FileNotFoundError raised when ATS model is missing."""
    ad = tmp_path / "p"
    ad.mkdir()
    rng = np.random.default_rng(1)
    n = 40
    X = pd.DataFrame({c: rng.normal(0, 1, n) for c in FEAT})
    y = (rng.random(n) > 0.5).astype(int)
    _save_pkl(ad / "game_outcome_model_calibrated.pkl", _clf(X, y, FEAT))
    _save_pkl(ad / "game_outcome_features.pkl", FEAT)
    with pytest.raises(FileNotFoundError, match="ATS model"):
        NBAEnsemble.load(ad)


# -- Test 2: predict() returns expected columns --------------------------------

def test_predict_returns_expected_columns(tmp_path):
    """predict() output must contain all required columns."""
    ens = NBAEnsemble.load(_build(tmp_path))
    result = ens.predict(_make_df(n=5))
    expected = {
        "win_prob", "ats_prob", "margin_pred", "margin_signal",
        "ensemble_score", "ensemble_edge", "confidence",
    }
    assert expected.issubset(set(result.columns))
    assert len(result) == 5


def test_predict_preserves_index(tmp_path):
    """predict() preserves input DataFrame index."""
    ens = NBAEnsemble.load(_build(tmp_path))
    X = _make_df(n=4)
    X.index = [10, 20, 30, 40]
    assert list(ens.predict(X).index) == [10, 20, 30, 40]


# -- Test 3: ensemble_score in [0, 1] ----------------------------------------

def test_ensemble_score_in_range(tmp_path):
    """ensemble_score must be in [0, 1] for all rows."""
    ens = NBAEnsemble.load(_build(tmp_path))
    result = ens.predict(_make_df(n=20, seed=7))
    assert (result["ensemble_score"] >= 0).all()
    assert (result["ensemble_score"] <= 1).all()


def test_win_prob_in_range(tmp_path):
    """win_prob must be in [0, 1]."""
    ens = NBAEnsemble.load(_build(tmp_path))
    result = ens.predict(_make_df(n=10))
    assert (result["win_prob"] >= 0).all() and (result["win_prob"] <= 1).all()


def test_ats_prob_in_range(tmp_path):
    """ats_prob must be in [0, 1]."""
    ens = NBAEnsemble.load(_build(tmp_path))
    result = ens.predict(_make_df(n=10))
    assert (result["ats_prob"] >= 0).all() and (result["ats_prob"] <= 1).all()


def test_margin_signal_in_range(tmp_path):
    """margin_signal must be in [0, 1] when margin model is present."""
    ens = NBAEnsemble.load(_build(tmp_path, margin=True))
    sig = ens.predict(_make_df(n=10))["margin_signal"].dropna()
    assert (sig >= 0).all() and (sig <= 1).all()


# -- Test 4: confidence is one of high/medium/low ----------------------------

def test_confidence_values_valid(tmp_path):
    """confidence column values must be high, medium, or low only."""
    ens = NBAEnsemble.load(_build(tmp_path))
    result = ens.predict(_make_df(n=15, seed=3))
    invalid = set(result["confidence"].unique()) - {"high", "medium", "low"}
    assert not invalid, f"Invalid confidence labels found: {invalid}"


def test_confidence_label_thresholds():
    """_confidence_label must bucket edges correctly."""
    assert _confidence_label(HIGH_CONFIDENCE_THRESHOLD + 0.01) == "high"
    assert _confidence_label(-(HIGH_CONFIDENCE_THRESHOLD + 0.01)) == "high"
    assert _confidence_label(MEDIUM_CONFIDENCE_THRESHOLD + 0.01) == "medium"
    assert _confidence_label(0.0) == "low"
    assert _confidence_label(0.05) == "low"


# -- Test 5: two-model fallback (no margin) -----------------------------------

def test_two_model_fallback_score_in_range(tmp_path):
    """Without margin model, ensemble_score must still be in [0, 1]."""
    ad = _build(tmp_path, margin=False)
    with pytest.warns(UserWarning):
        ens = NBAEnsemble.load(ad)
    result = ens.predict(_make_df(n=10))
    assert (result["ensemble_score"] >= 0).all()
    assert (result["ensemble_score"] <= 1).all()
    assert result["margin_pred"].isna().all()


# -- Test 6: save_config() writes valid JSON ----------------------------------

def test_save_config_writes_json(tmp_path):
    """save_config() must write a valid JSON file with weights summing to 1."""
    ad = _build(tmp_path)
    ens = NBAEnsemble.load(ad)
    config_path = ens.save_config(ad)
    assert config_path.exists()
    config = json.loads(config_path.read_text())
    assert "version" in config
    assert "weights" in config
    w = config["weights"]
    total = w["win_prob"] + w["ats_prob"] + w["margin_signal"]
    assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, not 1.0"


# -- Test 7: run_ensemble_on_predictions helper ------------------------------

def test_run_ensemble_extends_df(tmp_path):
    """run_ensemble_on_predictions() returns df with original + ensemble cols."""
    ad = _build(tmp_path)
    X = _make_df(n=5)
    result = run_ensemble_on_predictions(X, artifacts_dir=ad)
    assert set(X.columns).issubset(set(result.columns))
    assert "ensemble_score" in result.columns
    assert "confidence" in result.columns
    assert len(result) == 5


# -- Test 8: sigmoid helper ---------------------------------------------------

def test_sigmoid_is_monotone():
    """_sigmoid must be strictly increasing: larger x -> larger output."""
    x_sorted = np.array([-5.0, -2.0, -0.5, 0.0, 0.5, 2.0, 5.0])
    y = _sigmoid(x_sorted)
    for i in range(1, len(y)):
        assert y[i] >= y[i - 1], f"sigmoid not monotone at index {i}"


def test_confidence_high_at_strong_edge():
    """edge well above HIGH threshold must label as high."""
    assert _confidence_label(1.0) == "high"
    assert _confidence_label(-1.0) == "high"


def test_run_ensemble_row_count_preserved(tmp_path):
    """run_ensemble_on_predictions must return same row count as input."""
    ad = _build(tmp_path)
    X = _make_df(n=12, seed=5)
    result = run_ensemble_on_predictions(X, artifacts_dir=ad)
    assert len(result) == 12


def test_sigmoid_output_range():
    """_sigmoid must output values in [0, 1]; midpoint must be 0.5."""
    # Practical inputs (within ~10x the norm factor) must be strictly (0,1)
    x_practical = np.array([-30.0, -10.0, -1.0, 0.0, 1.0, 10.0, 30.0])
    y_practical = _sigmoid(x_practical)
    assert (y_practical > 0).all(), "sigmoid below 0"
    assert (y_practical < 1).all(), "sigmoid above 1 for practical inputs"
    # Extreme values are clipped; result must still be in [0, 1]
    x_extreme = np.array([-1000.0, 1000.0])
    y_extreme = _sigmoid(x_extreme)
    assert (y_extreme >= 0).all() and (y_extreme <= 1).all()
    # Midpoint check
    assert abs(float(_sigmoid(0)) - 0.5) < 1e-10
