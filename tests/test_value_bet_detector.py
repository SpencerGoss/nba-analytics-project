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
