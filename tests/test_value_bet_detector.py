"""
Tests for get_strong_value_bets() in src/models/value_bet_detector.py

TDD RED phase: These tests are written before the implementation exists.
They will fail on ImportError until get_strong_value_bets() is added.
"""

import pytest
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_bet(edge_magnitude, is_value_bet=True):
    """Create a minimal bet dict matching the run_value_bet_scan() schema."""
    return {
        "home_team": "LAL",
        "away_team": "GSW",
        "game_date": "2025-01-15",
        "season": 202425,
        "model_win_prob": 0.60,
        "market_implied_prob": 0.60 - edge_magnitude,
        "edge": edge_magnitude,
        "edge_magnitude": edge_magnitude,
        "is_value_bet": is_value_bet,
        "bet_side": "home",
    }


SAMPLE_BETS = [
    _make_bet(0.03),   # below 0.08 threshold
    _make_bet(0.06),   # below 0.08 threshold
    _make_bet(0.09),   # above threshold -> strong
    _make_bet(0.12),   # above threshold -> strong
    _make_bet(0.04),   # below threshold
]


# ---------------------------------------------------------------------------
# Test 1: Import succeeds and function is callable
# ---------------------------------------------------------------------------

def test_import_and_callable():
    """get_strong_value_bets must be importable and callable."""
    from src.models.value_bet_detector import get_strong_value_bets
    assert callable(get_strong_value_bets)


# ---------------------------------------------------------------------------
# Test 2: Threshold filtering returns correct subset
# ---------------------------------------------------------------------------

def test_threshold_filter():
    """get_strong_value_bets(0.08) returns only bets with edge_magnitude > 0.08."""
    from src.models.value_bet_detector import get_strong_value_bets

    with patch(
        "src.models.value_bet_detector.run_value_bet_scan",
        return_value=SAMPLE_BETS,
    ):
        result = get_strong_value_bets(strong_threshold=0.08)

    assert len(result) == 2, f"Expected 2 strong bets, got {len(result)}"
    for bet in result:
        assert bet["edge_magnitude"] > 0.08, (
            f"Bet with edge_magnitude={bet['edge_magnitude']} should not appear"
        )


# ---------------------------------------------------------------------------
# Test 3: Results are sorted by edge_magnitude descending
# ---------------------------------------------------------------------------

def test_sort_order():
    """Results are sorted strongest-edge-first (descending by edge_magnitude)."""
    from src.models.value_bet_detector import get_strong_value_bets

    with patch(
        "src.models.value_bet_detector.run_value_bet_scan",
        return_value=SAMPLE_BETS,
    ):
        result = get_strong_value_bets(strong_threshold=0.08)

    assert len(result) >= 2
    assert result[0]["edge_magnitude"] >= result[1]["edge_magnitude"], (
        "First result should have the largest edge_magnitude"
    )
    assert result[0]["edge_magnitude"] == pytest.approx(0.12)
    assert result[1]["edge_magnitude"] == pytest.approx(0.09)


# ---------------------------------------------------------------------------
# Test 4: Empty result when nothing meets threshold
# ---------------------------------------------------------------------------

def test_empty_when_no_strong_bets():
    """Returns empty list when no bets exceed strong_threshold."""
    from src.models.value_bet_detector import get_strong_value_bets

    low_bets = [_make_bet(0.02), _make_bet(0.04), _make_bet(0.07)]
    with patch(
        "src.models.value_bet_detector.run_value_bet_scan",
        return_value=low_bets,
    ):
        result = get_strong_value_bets(strong_threshold=0.08)

    assert result == [], f"Expected [], got {result}"


# ---------------------------------------------------------------------------
# Test 5: Custom threshold respected
# ---------------------------------------------------------------------------

def test_custom_threshold():
    """Custom strong_threshold argument is respected."""
    from src.models.value_bet_detector import get_strong_value_bets

    with patch(
        "src.models.value_bet_detector.run_value_bet_scan",
        return_value=SAMPLE_BETS,
    ):
        # With threshold=0.05, all bets with edge_magnitude > 0.05 qualify
        result_05 = get_strong_value_bets(strong_threshold=0.05)
        # Expected: 0.06, 0.09, 0.12 -> 3 bets
        assert len(result_05) == 3, f"Expected 3 bets at threshold=0.05, got {len(result_05)}"

        # With threshold=0.10, only 0.12 qualifies
        result_10 = get_strong_value_bets(strong_threshold=0.10)
        assert len(result_10) == 1, f"Expected 1 bet at threshold=0.10, got {len(result_10)}"
        assert result_10[0]["edge_magnitude"] == pytest.approx(0.12)


# ---------------------------------------------------------------------------
# Test 6: Schema check on returned dicts
# ---------------------------------------------------------------------------

def test_return_schema():
    """Each returned dict contains required value-bet schema keys."""
    from src.models.value_bet_detector import get_strong_value_bets

    with patch(
        "src.models.value_bet_detector.run_value_bet_scan",
        return_value=SAMPLE_BETS,
    ):
        result = get_strong_value_bets(strong_threshold=0.08)

    required_keys = {"edge_magnitude", "is_value_bet", "bet_side", "model_win_prob", "market_implied_prob"}
    for bet in result:
        missing = required_keys - set(bet.keys())
        assert not missing, f"Bet dict missing keys: {missing}"


# ---------------------------------------------------------------------------
# Test 7: Composite score formula correctness
# ---------------------------------------------------------------------------

def test_composite_score_formula():
    from src.models.value_bet_detector import (
        _score_bets_with_ats, COMPOSITE_EDGE_WEIGHT, COMPOSITE_ATS_WEIGHT,
    )
    import pandas as pd, tempfile, os
    from unittest.mock import patch, MagicMock
    bets = [_make_bet(0.10)]
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [[0.4, 0.65]]
    ats_rows = [{"home_team": "LAL", "away_team": "GSW", "game_date": "2025-01-15",
                 "spread": -3.5, "home_implied_prob": 0.55, "away_implied_prob": 0.45}]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        pd.DataFrame(ats_rows).to_csv(f, index=False)
        tmp_path = f.name
    try:
        feat_cols = ["spread", "home_implied_prob", "away_implied_prob"]
        with patch("src.models.value_bet_detector._load_ats_model",
                   return_value=(mock_model, feat_cols)):
            result = _score_bets_with_ats(bets, ats_features_path=tmp_path)
    finally:
        os.unlink(tmp_path)
    assert len(result) == 1
    b = result[0]
    expected = COMPOSITE_EDGE_WEIGHT * 0.10 + COMPOSITE_ATS_WEIGHT * (0.65 - 0.5)
    assert b["ats_prob"] == pytest.approx(0.65, abs=0.001)
    assert b["composite_score"] == pytest.approx(round(expected, 4), abs=0.0001)
    assert b["ats_model_used"] is True


# ---------------------------------------------------------------------------
# Test 8: ATS model fallback when artifact missing (cold start)
# ---------------------------------------------------------------------------

def test_ats_model_missing_falls_back():
    from src.models.value_bet_detector import _score_bets_with_ats, COMPOSITE_EDGE_WEIGHT
    from unittest.mock import patch
    bets = [_make_bet(0.10)]
    with patch("src.models.value_bet_detector._load_ats_model",
               side_effect=FileNotFoundError("no file")):
        result = _score_bets_with_ats(bets)
    assert len(result) == 1
    b = result[0]
    assert b["ats_prob"] is None
    assert b["ats_model_used"] is False
    expected = round(COMPOSITE_EDGE_WEIGHT * 0.10, 4)
    assert b["composite_score"] == pytest.approx(expected, abs=0.0001)


# ---------------------------------------------------------------------------
# Test 9: Composite path filters on composite_score, not raw edge
# ---------------------------------------------------------------------------

def test_composite_path_filters_on_composite_score():
    from src.models.value_bet_detector import get_strong_value_bets
    from unittest.mock import patch
    bet_high_edge = {**_make_bet(0.12), "ats_prob": 0.4, "composite_score": 0.03, "ats_model_used": True}
    bet_low_edge  = {**_make_bet(0.09), "ats_prob": 0.7, "composite_score": 0.08, "ats_model_used": True}
    scored = [bet_high_edge, bet_low_edge]
    with (patch("src.models.value_bet_detector.run_value_bet_scan", return_value=SAMPLE_BETS),
          patch("src.models.value_bet_detector._score_bets_with_ats", return_value=scored)):
        result = get_strong_value_bets(strong_threshold=0.08, composite_threshold=0.05)
    assert len(result) == 1
    assert result[0]["edge_magnitude"] == pytest.approx(0.09)
    assert result[0]["composite_score"] == pytest.approx(0.08)


# ---------------------------------------------------------------------------
# Test 10: Composite path sorts by composite_score descending
# ---------------------------------------------------------------------------

def test_composite_path_sorted_by_composite_score():
    from src.models.value_bet_detector import get_strong_value_bets
    from unittest.mock import patch
    bet_a = {**_make_bet(0.12), "ats_prob": 0.6, "composite_score": 0.12, "ats_model_used": True}
    bet_b = {**_make_bet(0.09), "ats_prob": 0.7, "composite_score": 0.09, "ats_model_used": True}
    scored = [bet_b, bet_a]  # deliberately wrong order
    with (patch("src.models.value_bet_detector.run_value_bet_scan", return_value=SAMPLE_BETS),
          patch("src.models.value_bet_detector._score_bets_with_ats", return_value=scored)):
        result = get_strong_value_bets(strong_threshold=0.08, composite_threshold=0.04)
    assert len(result) == 2
    assert result[0]["composite_score"] >= result[1]["composite_score"]
    assert result[0]["composite_score"] == pytest.approx(0.12)


# ---------------------------------------------------------------------------
# no_vig_prob
# ---------------------------------------------------------------------------

class TestNoVigProb:
    def test_even_money_both_sides(self):
        """Standard -110/-110 spread: after vig removal both are 50%."""
        from src.models.value_bet_detector import no_vig_prob
        home, away = no_vig_prob(-110, -110)
        assert home == pytest.approx(0.5, abs=0.001)
        assert away == pytest.approx(0.5, abs=0.001)

    def test_sums_to_one(self):
        """No-vig probs must always sum to 1.0."""
        from src.models.value_bet_detector import no_vig_prob
        for (home_ml, away_ml) in [(-150, 130), (-200, 170), (110, -130), (-110, -110)]:
            h, a = no_vig_prob(home_ml, away_ml)
            assert h + a == pytest.approx(1.0, abs=0.001)

    def test_favourite_has_higher_prob(self):
        """Heavy home favourite should yield home_prob > away_prob."""
        from src.models.value_bet_detector import no_vig_prob
        h, a = no_vig_prob(-300, 250)
        assert h > a

    def test_none_returns_nan(self):
        """None moneyline inputs should produce NaN outputs."""
        import math
        from src.models.value_bet_detector import no_vig_prob
        h, a = no_vig_prob(None, -110)
        assert math.isnan(h) and math.isnan(a)

    def test_nan_returns_nan(self):
        import math
        from src.models.value_bet_detector import no_vig_prob
        h, a = no_vig_prob(float("nan"), -110)
        assert math.isnan(h) and math.isnan(a)

    def test_positive_away_underdog(self):
        """Away underdog (+150) should have probability less than 50%."""
        from src.models.value_bet_detector import no_vig_prob
        h, a = no_vig_prob(-180, 150)
        assert h > 0.5
        assert a < 0.5

    def test_probabilities_in_range(self):
        """All valid outputs must be in (0, 1)."""
        from src.models.value_bet_detector import no_vig_prob
        for ml_pair in [(-110, -110), (-200, 170), (130, -150)]:
            h, a = no_vig_prob(*ml_pair)
            assert 0.0 < h < 1.0
            assert 0.0 < a < 1.0


# ---------------------------------------------------------------------------
# _compute_kelly_fraction
# ---------------------------------------------------------------------------

class TestComputeKellyFraction:
    def test_positive_edge_home_side(self):
        """Model 65%, market 50% -> positive kelly."""
        from src.models.value_bet_detector import _compute_kelly_fraction
        bet = {"model_win_prob": 0.65, "market_implied_prob": 0.50, "bet_side": "home"}
        result = _compute_kelly_fraction(bet)
        assert result > 0.0

    def test_zero_when_no_edge(self):
        """Model prob == market prob -> Kelly is 0."""
        from src.models.value_bet_detector import _compute_kelly_fraction
        bet = {"model_win_prob": 0.55, "market_implied_prob": 0.55, "bet_side": "home"}
        result = _compute_kelly_fraction(bet)
        assert result == 0.0

    def test_zero_when_model_prob_missing(self):
        from src.models.value_bet_detector import _compute_kelly_fraction
        bet = {"model_win_prob": None, "market_implied_prob": 0.55, "bet_side": "home"}
        assert _compute_kelly_fraction(bet) == 0.0

    def test_zero_when_market_prob_missing(self):
        from src.models.value_bet_detector import _compute_kelly_fraction
        bet = {"model_win_prob": 0.65, "market_implied_prob": None, "bet_side": "home"}
        assert _compute_kelly_fraction(bet) == 0.0

    def test_zero_when_market_degenerate(self):
        from src.models.value_bet_detector import _compute_kelly_fraction
        bet = {"model_win_prob": 0.65, "market_implied_prob": 1.0, "bet_side": "home"}
        assert _compute_kelly_fraction(bet) == 0.0

    def test_away_side_flips_probs(self):
        """Away bet should flip both probabilities for Kelly calculation."""
        from src.models.value_bet_detector import _compute_kelly_fraction
        # Away win prob = 1 - 0.35 = 0.65, market away = 1 - 0.50 = 0.50
        bet = {"model_win_prob": 0.35, "market_implied_prob": 0.50, "bet_side": "away"}
        result = _compute_kelly_fraction(bet)
        assert result > 0.0

    def test_half_kelly_scale(self):
        """Default scale is 0.5 (half Kelly)."""
        from src.models.value_bet_detector import _compute_kelly_fraction
        p, q = 0.65, 0.50
        b = (1 - q) / q
        full_kelly = (p * b - (1 - p)) / b
        expected = round(0.5 * full_kelly, 4)
        bet = {"model_win_prob": p, "market_implied_prob": q, "bet_side": "home"}
        assert _compute_kelly_fraction(bet) == pytest.approx(expected, abs=0.001)

    def test_result_is_non_negative(self):
        """Result must always be >= 0."""
        from src.models.value_bet_detector import _compute_kelly_fraction
        bet = {"model_win_prob": 0.30, "market_implied_prob": 0.70, "bet_side": "home"}
        assert _compute_kelly_fraction(bet) >= 0.0

