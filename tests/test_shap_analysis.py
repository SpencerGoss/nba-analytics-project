import pytest
import numpy as np
import os

# Skip all tests if shap not installed
shap = pytest.importorskip("shap")


def test_module_importable():
    from src.models.shap_analysis import run_shap_analysis, get_top_shap_features
    assert callable(run_shap_analysis)
    assert callable(get_top_shap_features)


def test_get_top_shap_features_empty():
    """Returns empty list when no analysis has been saved yet."""
    from src.models.shap_analysis import get_top_shap_features
    # Point to nonexistent path
    features = get_top_shap_features(artifact_dir="nonexistent_dir_12345")
    assert isinstance(features, list)
    assert len(features) == 0


def test_compute_shap_values_small():
    """SHAP values should be computable on a small synthetic dataset."""
    from src.models.shap_analysis import compute_shap_values
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((50, 5)), columns=[f"f{i}" for i in range(5)])
    y = (X["f0"] > 0).astype(int)

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=10, max_depth=2)),
    ])
    pipe.fit(X, y)

    shap_df = compute_shap_values(pipe, X)
    assert shap_df.shape == X.shape
    assert list(shap_df.columns) == list(X.columns)


def test_top_features_from_shap_values():
    """Should rank features by mean absolute SHAP value."""
    from src.models.shap_analysis import top_features_from_shap
    import pandas as pd

    shap_df = pd.DataFrame({
        "big_feature": [1.0, -2.0, 1.5, -1.0],
        "small_feature": [0.1, -0.1, 0.05, -0.05],
        "medium_feature": [0.5, -0.5, 0.3, -0.3],
    })
    ranked = top_features_from_shap(shap_df, n=3)
    assert len(ranked) == 3
    assert ranked[0][0] == "big_feature"  # highest mean abs
    assert ranked[1][0] == "medium_feature"
    assert ranked[2][0] == "small_feature"


def test_save_and_load_shap_results():
    """Save/load round-trip should preserve data."""
    from src.models.shap_analysis import save_shap_results, get_top_shap_features
    import tempfile

    results = [("feat_a", 0.5), ("feat_b", 0.3), ("feat_c", 0.1)]
    with tempfile.TemporaryDirectory() as tmpdir:
        save_shap_results(results, artifact_dir=tmpdir)
        loaded = get_top_shap_features(artifact_dir=tmpdir)
        assert len(loaded) == 3
        assert loaded[0][0] == "feat_a"
        assert abs(loaded[0][1] - 0.5) < 0.001
