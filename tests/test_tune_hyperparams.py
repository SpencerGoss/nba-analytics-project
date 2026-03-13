"""
Tests for scripts/tune_hyperparams.py

Design:
- Tests import cleanly and exercise helper functions.
- Full Optuna study is never run -- tests use n_trials=1 with a tiny synthetic
  DataFrame, or mock the objective directly.
- A tiny 30-row DataFrame with 3 "seasons" (10 rows each) lets season_splits()
  produce real splits without touching any real data files.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ── Ensure PROJECT_ROOT is on sys.path ─────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Module under test ──────────────────────────────────────────────────────────
import scripts.tune_hyperparams as tht  # noqa: E402


# ── Fixtures ───────────────────────────────────────────────────────────────────

FEAT_COLS = ["diff_pts", "diff_reb", "diff_ast", "diff_tov", "diff_fg_pct"]
N_SEASONS = 3
ROWS_PER_SEASON = 10
SEASONS = ["201314", "201415", "201516"]


def _make_tiny_df(n_seasons: int = N_SEASONS, rows_per_season: int = ROWS_PER_SEASON) -> pd.DataFrame:
    """Build a minimal DataFrame that mimics matchup feature structure.

    Uses 3 past seasons so season_splits() can produce at least one split
    (MIN_TRAIN_SEASONS_FOR_TUNING=6 would require more; the fallback 85/15
    split is exercised instead for small datasets).
    """
    rng = np.random.default_rng(42)
    seasons = SEASONS[:n_seasons]
    records = []
    base_date = pd.Timestamp("2013-10-01")
    for i, season in enumerate(seasons):
        for j in range(rows_per_season):
            day_offset = i * 200 + j
            records.append({
                "season": season,
                "game_date": base_date + pd.Timedelta(days=day_offset),
                "home_team": "LAL",
                "away_team": "GSW",
                "home_win": int(rng.integers(0, 2)),
                **{c: float(rng.standard_normal()) for c in FEAT_COLS},
            })
    df = pd.DataFrame(records)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df.sort_values("game_date").reset_index(drop=True)


@pytest.fixture()
def tiny_df():
    return _make_tiny_df()


# ── Test 1: Module imports cleanly ─────────────────────────────────────────────

def test_module_imports():
    """All public symbols used by tests are present after import."""
    assert callable(tht.load_data)
    assert callable(tht.load_feature_cols)
    assert callable(tht.derive_feature_cols)
    assert callable(tht.season_splits)
    assert callable(tht.expanding_window_auc)
    assert callable(tht.make_objective)
    assert callable(tht.run_study)
    assert callable(tht.retrain_with_best_params)
    assert callable(tht.parse_args)
    assert callable(tht.main)


# ── Test 2: derive_feature_cols returns diff_ columns ─────────────────────────

def test_derive_feature_cols_picks_diff_cols(tiny_df):
    """derive_feature_cols() prefers diff_ columns over raw numeric cols."""
    cols = tht.derive_feature_cols(tiny_df)
    assert len(cols) > 0
    for c in cols:
        assert c.startswith("diff_"), f"Unexpected non-diff column: {c}"
    # All FEAT_COLS should appear
    for fc in FEAT_COLS:
        assert fc in cols, f"Expected feature column '{fc}' not found"


def test_derive_feature_cols_excludes_target_and_metadata(tiny_df):
    """derive_feature_cols() never includes target or metadata columns."""
    cols = tht.derive_feature_cols(tiny_df)
    excluded = {"home_win", "game_id", "season", "game_date", "home_team", "away_team"}
    for c in cols:
        assert c not in excluded, f"Excluded column '{c}' should not be in feature list"


# ── Test 3: season_splits produces valid non-overlapping splits ────────────────

def test_season_splits_fallback_for_small_dataset(tiny_df):
    """With only 3 seasons (below MIN_TRAIN_SEASONS_FOR_TUNING=6), the fallback
    85/15 date split is used instead of season splits."""
    splits = tht.season_splits(tiny_df)
    assert len(splits) >= 1, "At least one split should be produced"
    for tr, va, label in splits:
        assert len(tr) > 0, f"Split '{label}' has empty training set"
        assert len(va) > 0, f"Split '{label}' has empty validation set"
        # Training and validation rows must not overlap
        tr_idx = set(tr.index)
        va_idx = set(va.index)
        assert tr_idx.isdisjoint(va_idx), f"Split '{label}' has overlapping train/val rows"


def test_season_splits_no_future_leakage():
    """Expanding window: val game_dates must always be later than all train game_dates."""
    # Use enough seasons to trigger real season-based splits
    rng = np.random.default_rng(0)
    seasons = [str(201314 + i) for i in range(8)]
    records = []
    base = pd.Timestamp("2013-10-01")
    for i, season in enumerate(seasons):
        for j in range(10):
            records.append({
                "season": season,
                "game_date": base + pd.Timedelta(days=i * 200 + j),
                "home_win": int(rng.integers(0, 2)),
                "diff_pts": float(rng.standard_normal()),
            })
    df = pd.DataFrame(records).sort_values("game_date").reset_index(drop=True)

    splits = tht.season_splits(df)
    assert len(splits) >= 1
    for tr, va, label in splits:
        max_train_date = tr["game_date"].max()
        min_val_date = va["game_date"].min()
        assert min_val_date > max_train_date, (
            f"Split '{label}': validation starts at {min_val_date} but "
            f"training ends at {max_train_date} -- data leakage detected"
        )


# ── Test 4: expanding_window_auc returns a valid float ────────────────────────

def test_expanding_window_auc_returns_float_in_range(tiny_df):
    """expanding_window_auc() returns a float in [0, 1] for a simple pipeline."""
    from sklearn.dummy import DummyClassifier
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([("clf", DummyClassifier(strategy="most_frequent"))])
    auc = tht.expanding_window_auc(pipeline, tiny_df, FEAT_COLS)
    assert isinstance(auc, float), f"Expected float, got {type(auc)}"
    assert 0.0 <= auc <= 1.0, f"AUC {auc} is outside [0, 1]"


def test_expanding_window_auc_empty_df_returns_zero():
    """expanding_window_auc() returns 0.0 when splits cannot be formed."""
    from sklearn.dummy import DummyClassifier
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([("clf", DummyClassifier())])
    empty_df = pd.DataFrame(columns=["season", "game_date", "home_win"] + FEAT_COLS)
    auc = tht.expanding_window_auc(pipeline, empty_df, FEAT_COLS)
    assert auc == 0.0


# ── Test 5: make_objective returns a callable objective ───────────────────────

def test_make_objective_returns_callable(tiny_df):
    """make_objective() returns a function that accepts an Optuna trial."""
    if not tht._LGBM_AVAILABLE:
        pytest.skip("LightGBM not installed")

    objective = tht.make_objective(tiny_df, FEAT_COLS, "lgbm")
    assert callable(objective)


def test_make_objective_raises_for_missing_library(tiny_df):
    """make_objective() raises RuntimeError when the requested library is absent."""
    with patch.object(tht, "_LGBM_AVAILABLE", False):
        with pytest.raises(RuntimeError, match="LightGBM"):
            tht.make_objective(tiny_df, FEAT_COLS, "lgbm")

    with patch.object(tht, "_XGB_AVAILABLE", False):
        with pytest.raises(RuntimeError, match="XGBoost"):
            tht.make_objective(tiny_df, FEAT_COLS, "xgb")


# ── Test 6: make_objective callable returns float in [0, 1] ───────────────────

def test_make_objective_lgbm_one_trial(tiny_df):
    """Run 1 LGBM trial end-to-end with a tiny dataset."""
    if not tht._LGBM_AVAILABLE or not tht._OPTUNA_AVAILABLE:
        pytest.skip("LightGBM or Optuna not installed")

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    objective = tht.make_objective(tiny_df, FEAT_COLS, "lgbm")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)
    assert 0.0 <= study.best_value <= 1.0


def test_make_objective_xgb_one_trial(tiny_df):
    """Run 1 XGB trial end-to-end with a tiny dataset."""
    if not tht._XGB_AVAILABLE or not tht._OPTUNA_AVAILABLE:
        pytest.skip("XGBoost or Optuna not installed")

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    objective = tht.make_objective(tiny_df, FEAT_COLS, "xgb")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)
    assert 0.0 <= study.best_value <= 1.0


# ── Test 7: run_study returns expected types ───────────────────────────────────

def test_run_study_returns_params_and_auc(tiny_df):
    """run_study() returns (dict, float) with model_type key."""
    if not tht._LGBM_AVAILABLE or not tht._OPTUNA_AVAILABLE:
        pytest.skip("LightGBM or Optuna not installed")

    best_params, best_auc = tht.run_study(tiny_df, FEAT_COLS, "lgbm", n_trials=1)

    assert isinstance(best_params, dict)
    assert "model_type" in best_params
    assert best_params["model_type"] == "lgbm"
    assert isinstance(best_auc, float)
    assert 0.0 <= best_auc <= 1.0


# ── Test 8: retrain_with_best_params produces a working Pipeline ──────────────

def test_retrain_with_best_params_saves_artifact_and_predicts(tiny_df, tmp_path):
    """retrain_with_best_params() fits a pipeline, saves a .pkl file, and
    the returned pipeline produces valid probabilities on the training data."""
    if not tht._LGBM_AVAILABLE:
        pytest.skip("LightGBM not installed")

    fake_params = {
        "model_type": "lgbm",
        "lgbm_n_estimators": 10,
        "lgbm_max_depth": 3,
        "lgbm_learning_rate": 0.1,
        "lgbm_num_leaves": 15,
        "lgbm_subsample": 0.8,
        "lgbm_colsample_bytree": 0.8,
        "lgbm_min_child_samples": 5,
        "lgbm_reg_alpha": 0.01,
        "lgbm_reg_lambda": 0.01,
    }

    pipeline = tht.retrain_with_best_params(
        tiny_df, FEAT_COLS, fake_params, artifacts_dir=str(tmp_path)
    )

    # Artifact file created
    out_path = tmp_path / "game_outcome_model_hpo.pkl"
    assert out_path.exists(), "game_outcome_model_hpo.pkl was not created"

    # Returned pipeline predicts valid probabilities
    assert hasattr(pipeline, "predict_proba"), "Pipeline must have predict_proba"
    proba = pipeline.predict_proba(tiny_df[FEAT_COLS])
    assert proba.shape == (len(tiny_df), 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)


def test_retrain_with_best_params_xgb_saves_artifact(tiny_df, tmp_path):
    """retrain_with_best_params() works for XGBoost model_type."""
    if not tht._XGB_AVAILABLE:
        pytest.skip("XGBoost not installed")

    fake_params = {
        "model_type": "xgb",
        "xgb_n_estimators": 10,
        "xgb_max_depth": 3,
        "xgb_learning_rate": 0.1,
        "xgb_subsample": 0.8,
        "xgb_colsample_bytree": 0.8,
        "xgb_min_child_weight": 1,
        "xgb_reg_alpha": 0.01,
        "xgb_reg_lambda": 0.01,
        "xgb_gamma": 0.0,
    }

    tht.retrain_with_best_params(tiny_df, FEAT_COLS, fake_params, artifacts_dir=str(tmp_path))
    out_path = tmp_path / "game_outcome_model_hpo.pkl"
    assert out_path.exists()


def test_retrain_with_best_params_unknown_type_raises(tiny_df, tmp_path):
    """retrain_with_best_params() raises ValueError for unknown model_type."""
    with pytest.raises(ValueError, match="Unknown model_type"):
        tht.retrain_with_best_params(
            tiny_df, FEAT_COLS, {"model_type": "svm"}, artifacts_dir=str(tmp_path)
        )


# ── Test 9: load_feature_cols returns None gracefully when pkl missing ─────────

def test_load_feature_cols_missing_returns_none(tmp_path):
    """load_feature_cols() returns None when the pkl file does not exist."""
    result = tht.load_feature_cols(str(tmp_path / "nonexistent.pkl"))
    assert result is None


# ── Test 10: main() JSON output has required keys (mocked study) ───────────────

def test_main_produces_json_with_required_keys(tiny_df, tmp_path):
    """main() saves best_hpo_params.json containing the required top-level keys."""
    if not tht._LGBM_AVAILABLE or not tht._OPTUNA_AVAILABLE:
        pytest.skip("LightGBM or Optuna not installed")

    # Write a tiny CSV in a temp location
    csv_path = str(tmp_path / "features.csv")
    tiny_df.to_csv(csv_path, index=False)

    artifacts_dir = str(tmp_path / "artifacts")

    # Patch load_feature_cols to return FEAT_COLS (no pkl needed)
    with patch.object(tht, "load_feature_cols", return_value=FEAT_COLS):
        payload = tht.main([
            "--trials", "1",
            "--no-retrain",
            "--matchup-path", csv_path,
            "--artifacts-dir", artifacts_dir,
        ])

    required_keys = {"best_model_type", "best_auc", "best_params", "all_results"}
    assert required_keys <= set(payload.keys()), (
        f"JSON payload missing keys: {required_keys - set(payload.keys())}"
    )

    json_path = os.path.join(artifacts_dir, "best_hpo_params.json")
    assert os.path.exists(json_path), "best_hpo_params.json was not created"

    with open(json_path) as f:
        saved = json.load(f)
    assert required_keys <= set(saved.keys())


# ── Test 11: parse_args defaults and overrides ─────────────────────────────────

def test_parse_args_defaults():
    """parse_args() returns correct defaults when no args given."""
    args = tht.parse_args([])
    assert args.n_trials == tht.DEFAULT_N_TRIALS
    assert args.no_retrain is False


def test_parse_args_overrides():
    """parse_args() respects --trials and --no-retrain flags."""
    args = tht.parse_args(["--trials", "7", "--no-retrain"])
    assert args.n_trials == 7
    assert args.no_retrain is True

    args2 = tht.parse_args(["-n", "3"])
    assert args2.n_trials == 3


# ── Test 12: no data leakage -- season ordering strictly enforced ──────────────

def test_season_splits_train_always_precedes_val():
    """In every split, all training seasons sort before the validation season."""
    rng = np.random.default_rng(99)
    seasons = [201314 + i for i in range(8)]
    records = []
    for i, season in enumerate(seasons):
        for j in range(15):
            records.append({
                "season": season,
                "game_date": pd.Timestamp("2013-10-01") + pd.Timedelta(days=i * 200 + j),
                "home_win": int(rng.integers(0, 2)),
                "diff_pts": float(rng.standard_normal()),
            })
    df = pd.DataFrame(records).sort_values("game_date").reset_index(drop=True)

    splits = tht.season_splits(df)
    assert len(splits) >= 1

    for tr, va, label in splits:
        if label == "date_fallback":
            # date_fallback splits by index, not season — overlap is expected
            continue
        tr_seasons = set(tr["season"].astype(int).unique())
        va_seasons = set(va["season"].astype(int).unique())
        # No season appears in both train and val
        assert tr_seasons.isdisjoint(va_seasons), (
            f"Split '{label}': season overlap between train {tr_seasons} and val {va_seasons}"
        )
        # Every training season must sort before every validation season
        max_tr_season = max(tr_seasons)
        min_va_season = min(va_seasons)
        assert max_tr_season < min_va_season, (
            f"Split '{label}': training season {max_tr_season} is not "
            f"before validation season {min_va_season}"
        )
