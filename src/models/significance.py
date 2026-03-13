"""Statistical significance tests for model accuracy and betting ROI."""

import numpy as np
from scipy import stats


def accuracy_significance(
    wins: int,
    total: int,
    null_prob: float = 0.5,
    alpha: float = 0.05,
) -> dict:
    """
    Binomial test for model accuracy significance.

    Tests H0: accuracy = null_prob vs H1: accuracy > null_prob (one-sided).
    Returns p-value, significance flag, and Wilson confidence interval.
    """
    # One-sided binomial test (scipy >= 1.7 API)
    result = stats.binomtest(wins, total, null_prob, alternative="greater")
    p_value = result.pvalue

    # Wilson score confidence interval (better than Wald for proportions)
    observed = wins / total
    z = stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / total
    center = (observed + z**2 / (2 * total)) / denom
    spread = (
        z
        * np.sqrt((observed * (1 - observed) + z**2 / (4 * total)) / total)
        / denom
    )

    return {
        "observed_accuracy": observed,
        "null_prob": null_prob,
        "p_value": float(p_value),
        "significant": bool(p_value < alpha),
        "ci_lower": float(center - spread),
        "ci_upper": float(center + spread),
        "n_games": total,
    }


def sample_size_needed(
    target_accuracy: float = 0.55,
    null_accuracy: float = 0.50,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """
    Calculate sample size needed to detect a given accuracy improvement.

    Uses normal approximation to the binomial for a one-proportion z-test.
    """
    p1 = target_accuracy
    p0 = null_accuracy

    z_alpha = stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)

    n = (
        (z_alpha * np.sqrt(p0 * (1 - p0)) + z_beta * np.sqrt(p1 * (1 - p1)))
        / (p1 - p0)
    ) ** 2

    return int(np.ceil(n))


def roi_significance(
    pnls: np.ndarray,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
) -> dict:
    """
    Bootstrap test for ROI significance.

    Tests whether mean ROI is significantly different from 0.
    """
    rng = np.random.default_rng(42)
    n = len(pnls)
    total_wagered = np.sum(np.abs(pnls))
    mean_roi = float(np.sum(pnls) / total_wagered) if total_wagered > 0 else 0.0

    # Bootstrap
    boot_rois = []
    for _ in range(n_bootstrap):
        sample = rng.choice(pnls, size=n, replace=True)
        wagered = np.sum(np.abs(sample))
        boot_roi = float(np.sum(sample) / wagered) if wagered > 0 else 0.0
        boot_rois.append(boot_roi)

    boot_rois = np.array(boot_rois)
    ci_lower = float(np.percentile(boot_rois, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_rois, 100 * (1 - alpha / 2)))

    # p-value: proportion of bootstrap samples <= 0
    p_value = float(np.mean(boot_rois <= 0))

    return {
        "mean_roi": mean_roi,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "significant": ci_lower > 0,
        "n_bets": n,
    }


def model_comparison(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    actuals: np.ndarray,
) -> dict:
    """
    McNemar's test comparing two models' predictions.

    Tests whether models have significantly different error rates.
    """
    correct_a = preds_a == actuals
    correct_b = preds_b == actuals

    # Contingency: a_right_b_wrong vs a_wrong_b_right
    a_right_b_wrong = int(np.sum(correct_a & ~correct_b))
    a_wrong_b_right = int(np.sum(~correct_a & correct_b))

    # McNemar's test (with continuity correction)
    if a_right_b_wrong + a_wrong_b_right == 0:
        p_value = 1.0
    else:
        stat = (
            (abs(a_right_b_wrong - a_wrong_b_right) - 1) ** 2
            / (a_right_b_wrong + a_wrong_b_right)
        )
        p_value = float(1 - stats.chi2.cdf(stat, df=1))

    return {
        "model_a_acc": float(np.mean(correct_a)),
        "model_b_acc": float(np.mean(correct_b)),
        "a_right_b_wrong": a_right_b_wrong,
        "a_wrong_b_right": a_wrong_b_right,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }
