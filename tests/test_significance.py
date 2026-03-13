import pytest
import numpy as np


def test_binomial_test_significant():
    """67.5% on 1000 games should be highly significant vs 50%."""
    from src.models.significance import accuracy_significance
    result = accuracy_significance(wins=675, total=1000, null_prob=0.5)
    assert result["p_value"] < 0.001
    assert result["significant"] is True


def test_binomial_test_not_significant():
    """51% on 100 games should NOT be significant."""
    from src.models.significance import accuracy_significance
    result = accuracy_significance(wins=51, total=100, null_prob=0.5)
    assert result["significant"] is False


def test_confidence_interval_contains_point():
    from src.models.significance import accuracy_significance
    result = accuracy_significance(wins=675, total=1000, null_prob=0.5)
    assert result["ci_lower"] <= 0.675 <= result["ci_upper"]


def test_ats_significance():
    """55% ATS on 500 bets — check if significant."""
    from src.models.significance import accuracy_significance
    result = accuracy_significance(wins=275, total=500, null_prob=0.5)
    # 55% on 500 should be significant (p ~ 0.013)
    assert result["p_value"] < 0.05
    assert result["significant"] is True


def test_ats_significance_small_sample():
    """55% on 50 bets should NOT be significant."""
    from src.models.significance import accuracy_significance
    result = accuracy_significance(wins=28, total=50, null_prob=0.5)
    assert result["significant"] is False


def test_sample_size_needed():
    """Calculate bets needed to confirm 55% ATS edge."""
    from src.models.significance import sample_size_needed
    n = sample_size_needed(target_accuracy=0.55, null_accuracy=0.5, alpha=0.05, power=0.80)
    assert isinstance(n, int)
    assert 300 < n < 1000  # should be around 385


def test_roi_significance():
    """Test ROI significance with bootstrap."""
    from src.models.significance import roi_significance
    pnls = np.array([1.0, -1.1, 1.0, 1.0, -1.1, 1.0, -1.1, 1.0, 1.0, -1.1])
    result = roi_significance(pnls, n_bootstrap=1000)
    assert "mean_roi" in result
    assert "ci_lower" in result
    assert "ci_upper" in result
    assert isinstance(result["mean_roi"], float)


def test_model_comparison():
    """Compare two models' predictions."""
    from src.models.significance import model_comparison
    preds_a = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 1])
    preds_b = np.array([1, 0, 0, 1, 0, 1, 0, 0, 1, 1])
    actuals = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
    result = model_comparison(preds_a, preds_b, actuals)
    assert "p_value" in result
    assert "model_a_acc" in result
    assert "model_b_acc" in result
