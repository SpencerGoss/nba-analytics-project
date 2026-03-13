"""SHAP analysis for game outcome model feature attribution."""
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = str(PROJECT_ROOT / "models" / "artifacts" / "shap_analysis")


def _require_shap():
    if not _SHAP_AVAILABLE:
        raise ImportError(
            "shap package not installed. Install with: pip install shap"
        )


def compute_shap_values(
    pipeline,
    X: pd.DataFrame,
    max_samples: int = 500,
) -> pd.DataFrame:
    """
    Compute SHAP values for a scikit-learn Pipeline with a tree-based classifier.

    Extracts the classifier from the pipeline, transforms X through preprocessing
    steps, then runs TreeExplainer.

    Returns DataFrame of SHAP values with same columns as X.
    """
    _require_shap()

    # Unwrap calibration wrapper if present (e.g., _PlattWrapper)
    if hasattr(pipeline, "base_model"):
        pipeline = pipeline.base_model

    # Extract preprocessing steps and classifier
    steps = list(pipeline.named_steps.keys())
    clf_name = steps[-1]
    clf = pipeline.named_steps[clf_name]

    # Transform X through all steps except the classifier
    X_transformed = X.copy()
    for step_name in steps[:-1]:
        step = pipeline.named_steps[step_name]
        X_transformed = pd.DataFrame(
            step.transform(X_transformed),
            columns=X.columns,
            index=X.index,
        )

    # Sample if too large
    if len(X_transformed) > max_samples:
        X_transformed = X_transformed.sample(n=max_samples, random_state=42)

    # SHAP TreeExplainer
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_transformed.values)

    # For binary classification, shap_values may be a list [class0, class1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class

    return pd.DataFrame(shap_values, columns=X.columns, index=X_transformed.index)


def top_features_from_shap(
    shap_df: pd.DataFrame,
    n: int = 20,
) -> list[tuple[str, float]]:
    """
    Rank features by mean absolute SHAP value.

    Returns list of (feature_name, mean_abs_shap) sorted descending.
    """
    mean_abs = shap_df.abs().mean().sort_values(ascending=False)
    return [(name, float(val)) for name, val in mean_abs.head(n).items()]


def save_shap_results(
    ranked_features: list[tuple[str, float]],
    artifact_dir: str = DEFAULT_ARTIFACT_DIR,
) -> None:
    """Save ranked SHAP features to JSON."""
    os.makedirs(artifact_dir, exist_ok=True)
    path = os.path.join(artifact_dir, "shap_top_features.json")
    data = [{"feature": name, "mean_abs_shap": val} for name, val in ranked_features]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_top_shap_features(
    artifact_dir: str = DEFAULT_ARTIFACT_DIR,
) -> list[tuple[str, float]]:
    """Load previously saved SHAP feature rankings."""
    path = os.path.join(artifact_dir, "shap_top_features.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    return [(item["feature"], item["mean_abs_shap"]) for item in data]


def run_shap_analysis(
    model_path: Optional[str] = None,
    features_path: Optional[str] = None,
    data_path: Optional[str] = None,
    artifact_dir: str = DEFAULT_ARTIFACT_DIR,
    max_samples: int = 500,
) -> list[tuple[str, float]]:
    """
    Full SHAP analysis pipeline: load model + data, compute SHAP, save results.

    This is the main entry point for running SHAP analysis on the trained model.
    """
    _require_shap()
    import joblib

    if model_path is None:
        model_path = str(
            PROJECT_ROOT / "models" / "artifacts" / "game_outcome_model_calibrated.pkl"
        )
    if features_path is None:
        features_path = str(
            PROJECT_ROOT / "models" / "artifacts" / "game_outcome_features.pkl"
        )
    if data_path is None:
        data_path = str(
            PROJECT_ROOT / "data" / "features" / "game_matchup_features.csv"
        )

    # Load model and features
    model = joblib.load(model_path)
    feat_cols = joblib.load(features_path)

    # Load data
    df = pd.read_csv(data_path)
    X = df[feat_cols]

    # Compute SHAP values
    shap_df = compute_shap_values(model, X, max_samples=max_samples)

    # Rank and save
    ranked = top_features_from_shap(shap_df)
    save_shap_results(ranked, artifact_dir=artifact_dir)

    return ranked
