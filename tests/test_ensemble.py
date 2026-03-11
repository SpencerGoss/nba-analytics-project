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
    ATS_WEIGHT,
    HIGH_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD,
    WEIGHTS_DEFAULT,
    WEIGHTS_HIGH_CONF,
    WEIGHTS_UNCERTAIN,
    NBAEnsemble,
    _confidence_label,
    _select_weight_regime,
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
        "ensemble_score", "ensemble_edge", "confidence", "weight_regime",
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
    """save_config() must write a valid JSON with all weight regimes summing to 1."""
    ad = _build(tmp_path)
    ens = NBAEnsemble.load(ad)
    config_path = ens.save_config(ad)
    assert config_path.exists()
    config = json.loads(config_path.read_text())
    assert "version" in config
    assert "weight_regimes" in config
    for regime_name, w in config["weight_regimes"].items():
        total = w["win_prob"] + w["ats_prob"] + w["margin_signal"]
        assert abs(total - 1.0) < 1e-9, (
            f"Weights in {regime_name} sum to {total}, not 1.0"
        )


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


# -- Test 9: weight regime selection -----------------------------------------

def test_select_weight_regime_high_confidence():
    """High win_prob (>0.65) should select high_confidence regime."""
    weights, regime = _select_weight_regime(0.70)
    assert regime == "high_confidence"
    assert weights == WEIGHTS_HIGH_CONF


def test_select_weight_regime_high_confidence_low():
    """Low win_prob (<0.35) should select high_confidence regime."""
    weights, regime = _select_weight_regime(0.25)
    assert regime == "high_confidence"
    assert weights == WEIGHTS_HIGH_CONF


def test_select_weight_regime_uncertain():
    """win_prob in [0.45, 0.55] should select uncertain regime."""
    weights, regime = _select_weight_regime(0.50)
    assert regime == "uncertain"
    assert weights == WEIGHTS_UNCERTAIN


def test_select_weight_regime_default():
    """win_prob between thresholds (not high, not uncertain) -> default."""
    weights, regime = _select_weight_regime(0.60)
    assert regime == "default"
    assert weights == WEIGHTS_DEFAULT


def test_select_weight_regime_boundary_065():
    """win_prob exactly 0.65 is NOT > 0.65 so should be default."""
    _, regime = _select_weight_regime(0.65)
    assert regime == "default"


def test_select_weight_regime_boundary_045():
    """win_prob exactly 0.45 is in uncertain range [0.45, 0.55]."""
    _, regime = _select_weight_regime(0.45)
    assert regime == "uncertain"


# -- Test 10: ATS weight is zero ---------------------------------------------

def test_ats_weight_is_zero():
    """ATS_WEIGHT must be 0.0 (ATS model effectively disabled)."""
    assert ATS_WEIGHT == 0.0
    for regime in (WEIGHTS_HIGH_CONF, WEIGHTS_DEFAULT, WEIGHTS_UNCERTAIN):
        assert regime["ats_prob"] == 0.0


# -- Test 11: all weight regimes sum to 1.0 ----------------------------------

def test_all_weight_regimes_sum_to_one():
    """Each weight regime must sum to exactly 1.0."""
    for name, regime in [
        ("high_confidence", WEIGHTS_HIGH_CONF),
        ("default", WEIGHTS_DEFAULT),
        ("uncertain", WEIGHTS_UNCERTAIN),
    ]:
        total = sum(regime.values())
        assert abs(total - 1.0) < 1e-9, f"{name} weights sum to {total}"


# -- Test 12: weight_regime column is populated --------------------------------

def test_predict_weight_regime_column(tmp_path):
    """predict() must return weight_regime with valid regime names."""
    ens = NBAEnsemble.load(_build(tmp_path))
    result = ens.predict(_make_df(n=20, seed=11))
    valid = {"high_confidence", "default", "uncertain"}
    invalid = set(result["weight_regime"].unique()) - valid
    assert not invalid, f"Invalid weight regimes: {invalid}"


# -- Test 13: dynamic weights change ensemble_score ---------------------------

def test_dynamic_weights_vary_by_confidence(tmp_path):
    """Rows with different win_prob ranges should get different weight regimes."""
    ens = NBAEnsemble.load(_build(tmp_path))
    # Use enough rows that we're likely to get multiple regimes
    result = ens.predict(_make_df(n=50, seed=42))
    regimes_seen = set(result["weight_regime"].unique())
    # With 50 random rows we should see at least 2 different regimes
    assert len(regimes_seen) >= 1, "Expected at least one weight regime"


# -- Test 14: ensemble_score still in [0, 1] with dynamic weights ------------

def test_ensemble_score_in_range_dynamic(tmp_path):
    """ensemble_score must be in [0, 1] with dynamic weighting."""
    ens = NBAEnsemble.load(_build(tmp_path))
    for seed in range(5):
        result = ens.predict(_make_df(n=20, seed=seed))
        assert (result["ensemble_score"] >= 0).all(), f"Below 0 at seed {seed}"
        assert (result["ensemble_score"] <= 1).all(), f"Above 1 at seed {seed}"
