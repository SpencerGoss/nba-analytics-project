"""NBA Ensemble Model
====================
Blend game_outcome (win probability), ATS (cover probability), and margin
(point differential) predictions into a unified ensemble score.

Uses confidence-dependent weighting: when win probability is decisive the
win model gets more weight; when uncertain the margin model gets more.

margin_signal  = sigmoid(margin_pred / 15)
ensemble_edge  = ensemble_score - 0.5  (positive means lean home)
confidence     = high/medium/low based on |ensemble_edge| thresholds

Weight regimes (win / ats / margin):
  high-confidence (win_prob >0.65 or <0.35): 0.75 / 0.0 / 0.25
  default         (in between):               0.65 / 0.0 / 0.35
  uncertain       (win_prob 0.45-0.55):       0.55 / 0.0 / 0.45

The ensemble wraps three individual model pkl files and has no pkl of its own.
Configuration is saved as ensemble_config.json.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

ARTIFACTS_DIR = Path(PROJECT_ROOT) / "models" / "artifacts"

# --- Weight regime constants -------------------------------------------------
# ATS model (AUC ~0.557) is near-random.  Set to 0.0 to remove noise from the
# blend.  Keep the code path so ATS can be re-enabled if the model improves.
ATS_WEIGHT = 0.0

# High-confidence regime: win_prob > 0.65 or < 0.35
WEIGHTS_HIGH_CONF = {"win_prob": 0.75, "ats_prob": ATS_WEIGHT, "margin_signal": 0.25}
# Default regime: 0.35 <= win_prob <= 0.65, excluding uncertain band
WEIGHTS_DEFAULT = {"win_prob": 0.65, "ats_prob": ATS_WEIGHT, "margin_signal": 0.35}
# Uncertain regime: 0.45 <= win_prob <= 0.55
WEIGHTS_UNCERTAIN = {"win_prob": 0.55, "ats_prob": ATS_WEIGHT, "margin_signal": 0.45}

# Backward-compat aliases (sum of non-ATS weights for each regime)
WIN_PROB_WEIGHT = WEIGHTS_DEFAULT["win_prob"]
ATS_PROB_WEIGHT = ATS_WEIGHT
MARGIN_SIGNAL_WEIGHT = WEIGHTS_DEFAULT["margin_signal"]

# Confidence-label thresholds (on ensemble_edge)
HIGH_CONFIDENCE_THRESHOLD = 0.15
MEDIUM_CONFIDENCE_THRESHOLD = 0.08
MARGIN_NORM_FACTOR = 15.0
ENSEMBLE_VERSION = "2.0.0"

# Win-prob thresholds that determine the weight regime
_WIN_PROB_HIGH_UPPER = 0.65  # above this -> high-confidence regime
_WIN_PROB_HIGH_LOWER = 0.35  # below this -> high-confidence regime
_WIN_PROB_UNCERTAIN_UPPER = 0.55  # at or below -> uncertain regime
_WIN_PROB_UNCERTAIN_LOWER = 0.45  # at or above -> uncertain regime


def _sigmoid(x):
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _select_weight_regime(win_prob):
    """Choose weight regime based on a single win probability value.

    Returns:
        Tuple of (weights_dict, regime_name).
    """
    if win_prob > _WIN_PROB_HIGH_UPPER or win_prob < _WIN_PROB_HIGH_LOWER:
        return WEIGHTS_HIGH_CONF, "high_confidence"
    if _WIN_PROB_UNCERTAIN_LOWER <= win_prob <= _WIN_PROB_UNCERTAIN_UPPER:
        return WEIGHTS_UNCERTAIN, "uncertain"
    return WEIGHTS_DEFAULT, "default"


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

    Uses confidence-dependent weighting that shifts weight between win-prob
    and margin models depending on how decisive the win probability is.

    ATS model (AUC ~0.557) is loaded but receives 0 weight (ATS_WEIGHT=0.0).
    Set ATS_WEIGHT > 0 to re-enable once the model improves.

    margin_signal = sigmoid(margin_pred / 15)
    ensemble_edge = ensemble_score - 0.5  (positive -> lean home)
    confidence: high if |edge|>0.15, medium if >0.08, low otherwise
    """

    # Class-level weight attributes for easy external access / tuning
    W_WIN = WIN_PROB_WEIGHT
    W_ATS = ATS_PROB_WEIGHT
    W_MARGIN = MARGIN_SIGNAL_WEIGHT

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

        Outcome model is required. ATS is loaded only when ATS_WEIGHT > 0.
        Margin is optional (warns if absent).

        Raises:
            FileNotFoundError: If required outcome artifacts are missing, or
                ATS artifacts are missing when ATS_WEIGHT > 0.
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

        if ATS_WEIGHT > 0:
            if not ats_path.exists():
                raise FileNotFoundError(
                    f"ATS model artifact not found at {str(ats_path)!r}. "
                    "Run: python src/models/ats_model.py"
                )
            ats_model = _load_artifact(ats_path)
            ats_feats = _load_artifact(ats_feat_path) if ats_feat_path.exists() else []
        else:
            ats_model = None
            ats_feats = []
            logger.info("ATS_WEIGHT=0.0 -- skipping ATS model load")

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

        Uses confidence-dependent weighting: high-confidence win probs get
        more win-model weight; uncertain probs shift weight to margin model.

        Args:
            X: DataFrame with feature columns. Missing columns become NaN and
               are handled by each sub-model's internal SimpleImputer.

        Returns:
            DataFrame with columns: win_prob, ats_prob, margin_pred,
            margin_signal, ensemble_score, ensemble_edge, confidence,
            weight_regime.
        """
        n = len(X)

        X_outcome = X.reindex(columns=self.outcome_feats)
        win_prob = self.outcome_model.predict_proba(X_outcome)[:, 1]

        if self.ats_model is not None and ATS_WEIGHT > 0:
            X_ats = X.reindex(columns=self.ats_feats) if self.ats_feats else X
            ats_prob = self.ats_model.predict_proba(X_ats)[:, 1]
        else:
            ats_prob = np.full(n, 0.5)  # Neutral when ATS disabled

        if self.margin_model is not None:
            X_margin = (
                X.reindex(columns=self.margin_feats) if self.margin_feats else X
            )
            margin_pred = self.margin_model.predict(X_margin)
            margin_signal = _sigmoid(margin_pred / MARGIN_NORM_FACTOR)
        else:
            margin_pred = np.full(n, np.nan)
            margin_signal = np.full(n, np.nan)

        # --- Dynamic weighting per row ---
        ensemble_score = np.empty(n)
        weight_regimes = []
        regime_counts = {"high_confidence": 0, "default": 0, "uncertain": 0}

        for i in range(n):
            wp_i = float(win_prob[i])
            weights, regime = _select_weight_regime(wp_i)
            weight_regimes.append(regime)
            regime_counts[regime] += 1

            if self.margin_model is not None:
                score = (
                    weights["win_prob"] * wp_i
                    + weights["ats_prob"] * float(ats_prob[i])
                    + weights["margin_signal"] * float(margin_signal[i])
                )
            else:
                # No margin model -- redistribute among win + ats only
                total_w = weights["win_prob"] + weights["ats_prob"]
                if total_w > 0:
                    score = (
                        (weights["win_prob"] / total_w) * wp_i
                        + (weights["ats_prob"] / total_w) * float(ats_prob[i])
                    )
                else:
                    score = wp_i
            ensemble_score[i] = score

        # Clamp to [0, 1] (should already be there, but guard)
        ensemble_score = np.clip(ensemble_score, 0.0, 1.0)

        logger.info(
            "Ensemble weight regimes: %d high_confidence, %d default, %d uncertain",
            regime_counts["high_confidence"],
            regime_counts["default"],
            regime_counts["uncertain"],
        )

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
                "weight_regime": weight_regimes,
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
            "weight_regimes": {
                "high_confidence": WEIGHTS_HIGH_CONF,
                "default": WEIGHTS_DEFAULT,
                "uncertain": WEIGHTS_UNCERTAIN,
            },
            "ats_weight": ATS_WEIGHT,
            "win_prob_thresholds": {
                "high_upper": _WIN_PROB_HIGH_UPPER,
                "high_lower": _WIN_PROB_HIGH_LOWER,
                "uncertain_upper": _WIN_PROB_UNCERTAIN_UPPER,
                "uncertain_lower": _WIN_PROB_UNCERTAIN_LOWER,
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
