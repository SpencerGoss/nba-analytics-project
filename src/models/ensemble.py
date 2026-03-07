"""NBA Ensemble Model
====================
Blend game_outcome (win probability), ATS (cover probability), and margin
(point differential) predictions into a unified ensemble score.

ensemble_score = 0.5 * win_prob + 0.3 * ats_prob + 0.2 * margin_signal
margin_signal  = sigmoid(margin_pred / 15)
ensemble_edge  = ensemble_score - 0.5  (positive means lean home)
confidence     = high/medium/low based on |ensemble_edge| thresholds

The ensemble wraps three individual model pkl files and has no pkl of its own.
Configuration is saved as ensemble_config.json.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

ARTIFACTS_DIR = Path(PROJECT_ROOT) / "models" / "artifacts"

WIN_PROB_WEIGHT = 0.5
ATS_PROB_WEIGHT = 0.3
MARGIN_SIGNAL_WEIGHT = 0.2
HIGH_CONFIDENCE_THRESHOLD = 0.15
MEDIUM_CONFIDENCE_THRESHOLD = 0.08
MARGIN_NORM_FACTOR = 15.0
ENSEMBLE_VERSION = "1.0.0"


def _sigmoid(x):
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _confidence_label(edge):
    """Map |ensemble_edge| to high/medium/low string."""
    abs_edge = abs(edge)
    if abs_edge > HIGH_CONFIDENCE_THRESHOLD:
        return "high"
    if abs_edge > MEDIUM_CONFIDENCE_THRESHOLD:
        return "medium"
    return "low"


def _load_artifact(path):
    """Load a serialised sklearn pipeline from path. Project standard format."""
    with open(path, "rb") as fh:
        return pickle.load(fh)


class NBAEnsemble:
    """Blend NBA prediction models into a unified ensemble score.

    Weights: win_prob=0.5, ats_prob=0.3, margin_signal=0.2
    margin_signal = sigmoid(margin_pred / 15)
    ensemble_edge = ensemble_score - 0.5  (positive -> lean home)
    confidence: high if |edge|>0.15, medium if >0.08, low otherwise
    """

    def __init__(
        self,
        outcome_model,
        outcome_feats,
        ats_model,
        ats_feats,
        margin_model=None,
        margin_feats=None,
    ):
        self.outcome_model = outcome_model
        self.outcome_feats = outcome_feats
        self.ats_model = ats_model
        self.ats_feats = ats_feats
        self.margin_model = margin_model
        self.margin_feats = margin_feats
        self.weights = {
            "win_prob": WIN_PROB_WEIGHT,
            "ats_prob": ATS_PROB_WEIGHT,
            "margin_signal": MARGIN_SIGNAL_WEIGHT,
        }

    @classmethod
    def load(cls, artifacts_dir=ARTIFACTS_DIR):
        """Load model artifacts from artifacts_dir.

        Outcome + ATS models are required. Margin is optional (warns if absent).

        Raises:
            FileNotFoundError: If required outcome or ATS artifacts are missing.
        """
        artifacts_dir = Path(artifacts_dir)

        cal_path = artifacts_dir / "game_outcome_model_calibrated.pkl"
        raw_path = artifacts_dir / "game_outcome_model.pkl"
        feat_path = artifacts_dir / "game_outcome_features.pkl"

        if cal_path.exists():
            outcome_model = _load_artifact(cal_path)
        elif raw_path.exists():
            warnings.warn(
                "Calibrated outcome model not found; using uncalibrated model.",
                UserWarning,
                stacklevel=2,
            )
            outcome_model = _load_artifact(raw_path)
        else:
            raise FileNotFoundError(
                f"No game outcome model found in {str(artifacts_dir)!r}. "
                "Run: python src/models/game_outcome_model.py"
            )

        if not feat_path.exists():
            raise FileNotFoundError(
                f"game_outcome_features.pkl not found in {str(artifacts_dir)!r}."
            )
        outcome_feats = _load_artifact(feat_path)

        ats_path = artifacts_dir / "ats_model.pkl"
        ats_feat_path = artifacts_dir / "ats_model_features.pkl"

        if not ats_path.exists():
            raise FileNotFoundError(
                f"ATS model artifact not found at {str(ats_path)!r}. "
                "Run: python src/models/ats_model.py"
            )
        ats_model = _load_artifact(ats_path)
        ats_feats = _load_artifact(ats_feat_path) if ats_feat_path.exists() else []

        margin_model = None
        margin_feats = None
        margin_path = artifacts_dir / "margin_model.pkl"
        margin_feat_path = artifacts_dir / "margin_model_features.pkl"

        if margin_path.exists():
            margin_model = _load_artifact(margin_path)
            if margin_feat_path.exists():
                margin_feats = _load_artifact(margin_feat_path)
        else:
            warnings.warn(
                "Margin model not found; ensemble uses win_prob+ats_prob only. "
                "Run: python src/models/margin_model.py to add margin signal.",
                UserWarning,
                stacklevel=2,
            )

        return cls(
            outcome_model=outcome_model,
            outcome_feats=outcome_feats,
            ats_model=ats_model,
            ats_feats=ats_feats,
            margin_model=margin_model,
            margin_feats=margin_feats,
        )

    def predict(self, X):
        """Compute ensemble predictions for each row in X.

        Args:
            X: DataFrame with feature columns. Missing columns become NaN and
               are handled by each sub-model's internal SimpleImputer.

        Returns:
            DataFrame with columns: win_prob, ats_prob, margin_pred,
            margin_signal, ensemble_score, ensemble_edge, confidence.
        """
        n = len(X)

        X_outcome = X.reindex(columns=self.outcome_feats)
        win_prob = self.outcome_model.predict_proba(X_outcome)[:, 1]

        X_ats = X.reindex(columns=self.ats_feats) if self.ats_feats else X
        ats_prob = self.ats_model.predict_proba(X_ats)[:, 1]

        if self.margin_model is not None:
            X_margin = (
                X.reindex(columns=self.margin_feats) if self.margin_feats else X
            )
            margin_pred = self.margin_model.predict(X_margin)
            margin_signal = _sigmoid(margin_pred / MARGIN_NORM_FACTOR)
        else:
            margin_pred = np.full(n, np.nan)
            margin_signal = np.full(n, np.nan)

        if self.margin_model is not None:
            ensemble_score = (
                WIN_PROB_WEIGHT * win_prob
                + ATS_PROB_WEIGHT * ats_prob
                + MARGIN_SIGNAL_WEIGHT * margin_signal
            )
        else:
            total_w = WIN_PROB_WEIGHT + ATS_PROB_WEIGHT
            w_win = WIN_PROB_WEIGHT / total_w
            w_ats = ATS_PROB_WEIGHT / total_w
            ensemble_score = w_win * win_prob + w_ats * ats_prob

        ensemble_edge = ensemble_score - 0.5
        confidence = [_confidence_label(float(e)) for e in ensemble_edge]

        return pd.DataFrame(
            {
                "win_prob": win_prob,
                "ats_prob": ats_prob,
                "margin_pred": margin_pred,
                "margin_signal": margin_signal,
                "ensemble_score": ensemble_score,
                "ensemble_edge": ensemble_edge,
                "confidence": confidence,
            },
            index=X.index,
        )

    def save_config(self, artifacts_dir=ARTIFACTS_DIR):
        """Save ensemble weights and version info to ensemble_config.json.

        The ensemble wraps individual model pkl files; it has no pkl of its own.
        Returns path to the saved JSON file.
        """
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "version": ENSEMBLE_VERSION,
            "created_at": datetime.now().isoformat(),
            "weights": {
                "win_prob": WIN_PROB_WEIGHT,
                "ats_prob": ATS_PROB_WEIGHT,
                "margin_signal": MARGIN_SIGNAL_WEIGHT,
            },
            "margin_norm_factor": MARGIN_NORM_FACTOR,
            "confidence_thresholds": {
                "high": HIGH_CONFIDENCE_THRESHOLD,
                "medium": MEDIUM_CONFIDENCE_THRESHOLD,
            },
            "margin_model_present": self.margin_model is not None,
            "outcome_feature_count": len(self.outcome_feats),
            "ats_feature_count": len(self.ats_feats),
            "margin_feature_count": (
                len(self.margin_feats) if self.margin_feats else 0
            ),
        }

        config_path = artifacts_dir / "ensemble_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"Ensemble config saved -> {config_path}")
        return config_path


def run_ensemble_on_predictions(predictions_df, artifacts_dir=ARTIFACTS_DIR):
    """Run ensemble over a predictions DataFrame. Returns df with ensemble cols."""
    ensemble = NBAEnsemble.load(artifacts_dir)
    scores = ensemble.predict(predictions_df)
    result = predictions_df.copy()
    for col in scores.columns:
        result[col] = scores[col].values
    return result


if __name__ == "__main__":
    print("Loading NBAEnsemble...")
    ens = NBAEnsemble.load()
    config_path = ens.save_config()
    print(f"  Margin model present: {ens.margin_model is not None}")
    print(f"  Outcome features: {len(ens.outcome_feats)}")
    print(f"  ATS features: {len(ens.ats_feats)}")
    print(f"  Config saved -> {config_path}")
