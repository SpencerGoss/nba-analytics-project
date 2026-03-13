import numpy as np
import pytest


def test_conformal_calibration():
    from src.models.conformal import calibrate_conformal
    residuals = np.array([1.0, -2.0, 0.5, 3.0, -1.5, 2.0, -0.5, 1.5, -3.0, 0.0])
    quantile = calibrate_conformal(residuals, coverage=0.90)
    assert quantile > 0
    # Should be >= most absolute residuals for 90% coverage
    assert quantile >= 2.0


def test_conformal_interval():
    from src.models.conformal import conformal_interval
    interval = conformal_interval(prediction=22.0, quantile=5.0)
    assert interval["lower"] == 17.0
    assert interval["upper"] == 27.0
    assert interval["width"] == 10.0


def test_conformal_coverage_guarantee():
    """Conformal intervals should cover at least the requested rate on calibration data."""
    from src.models.conformal import calibrate_conformal, conformal_interval
    np.random.seed(42)
    # Simulate predictions and actuals
    preds = np.random.normal(20, 3, 100)
    actuals = preds + np.random.normal(0, 4, 100)
    residuals = actuals - preds

    quantile = calibrate_conformal(residuals, coverage=0.90)

    # Check coverage on calibration set
    covered = sum(1 for p, a in zip(preds, actuals)
                  if conformal_interval(p, quantile)["lower"] <= a <= conformal_interval(p, quantile)["upper"])
    coverage = covered / len(preds)
    assert coverage >= 0.88  # Allow small tolerance


def test_conformal_zero_prediction():
    from src.models.conformal import conformal_interval
    interval = conformal_interval(prediction=0.0, quantile=3.0)
    assert interval["lower"] == -3.0
    assert interval["upper"] == 3.0


def test_save_and_load_quantiles():
    """Should be able to save and load conformal quantiles."""
    from src.models.conformal import calibrate_conformal, save_conformal_quantiles, load_conformal_quantiles
    import tempfile

    np.random.seed(42)
    residuals = {"pts": np.random.normal(0, 3, 50), "reb": np.random.normal(0, 2, 50)}

    quantiles = {}
    for stat, res in residuals.items():
        quantiles[stat] = calibrate_conformal(res, coverage=0.90)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_conformal_quantiles(quantiles, tmpdir)
        loaded = load_conformal_quantiles(tmpdir)
        assert abs(loaded["pts"] - quantiles["pts"]) < 0.01
        assert abs(loaded["reb"] - quantiles["reb"]) < 0.01
