"""Shared odds utility functions.

Centralizes devigging, conversion, and EV calculations previously
duplicated across fetch_odds.py, value_bet_detector.py, and build_value_bets.py.
"""
from __future__ import annotations


def american_to_decimal(american: int | float) -> float:
    """Convert American odds to decimal odds."""
    if american == 0:
        raise ValueError("American odds cannot be 0 (invalid odds value)")
    if american >= 100:
        return (american / 100) + 1
    return (100 / abs(american)) + 1


def american_to_implied_prob(american: int | float | None) -> float | None:
    """Convert American odds to implied probability (vig-inclusive)."""
    if american is None:
        return None
    if american == 0:
        raise ValueError("American odds cannot be 0 (invalid odds value)")
    if american >= 100:
        return 100 / (american + 100)
    return abs(american) / (abs(american) + 100)


def no_vig_odds_ratio(
    home_ml: int | float | None,
    away_ml: int | float | None,
) -> tuple[float | None, float | None]:
    """Remove vig using multiplicative normalization.

    Returns (home_prob, away_prob) summing to 1.0, or (None, None) if
    either input is None.
    """
    if home_ml is None or away_ml is None:
        return (None, None)
    home_dec = american_to_decimal(home_ml)
    away_dec = american_to_decimal(away_ml)
    home_imp = 1 / home_dec
    away_imp = 1 / away_dec
    total = home_imp + away_imp
    return (home_imp / total, away_imp / total)


def expected_value(model_prob: float, market_prob: float) -> float | None:
    """EV = (model_prob / market_prob) - 1.

    Positive means the model sees more value than the market.
    Returns None if market_prob is zero or negative.
    """
    if market_prob <= 0:
        return None
    return (model_prob / market_prob) - 1
