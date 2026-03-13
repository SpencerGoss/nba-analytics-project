"""Conformal prediction for guaranteed coverage intervals.

Provides distribution-free prediction intervals: given a desired coverage
(e.g., 90%), calibrate_conformal computes a quantile from held-out residuals
such that future predictions have >= coverage probability of containing
the true value.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def calibrate_conformal(residuals: np.ndarray, coverage: float = 0.90) -> float:
    """Compute conformal quantile from calibration residuals.

    Args:
        residuals: Array of (actual - predicted) values from held-out data.
        coverage: Desired coverage probability (e.g., 0.90 for 90%).

    Returns:
        Conformal quantile (half-width of prediction interval).
    """
    abs_residuals = np.abs(residuals)
    n = len(abs_residuals)
    q = np.ceil((n + 1) * coverage) / n
    return float(np.quantile(abs_residuals, min(q, 1.0)))


def conformal_interval(prediction: float, quantile: float) -> dict:
    """Build prediction interval from conformal quantile.

    Args:
        prediction: Point prediction (e.g., 22.5 points).
        quantile: Conformal quantile from calibrate_conformal().

    Returns:
        Dict with lower, upper, width.
    """
    return {
        "lower": round(prediction - quantile, 1),
        "upper": round(prediction + quantile, 1),
        "width": round(2 * quantile, 1),
    }


def save_conformal_quantiles(
    quantiles: dict[str, float],
    artifacts_dir: str = "models/artifacts",
) -> None:
    """Save conformal quantiles to JSON."""
    path = Path(artifacts_dir) / "conformal_quantiles.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(quantiles, f, indent=2)


def load_conformal_quantiles(
    artifacts_dir: str = "models/artifacts",
) -> dict[str, float]:
    """Load conformal quantiles from JSON."""
    path = Path(artifacts_dir) / "conformal_quantiles.json"
    with open(path) as f:
        return json.load(f)
