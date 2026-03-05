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
    assert b["ats_prob"] == 0.5
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

