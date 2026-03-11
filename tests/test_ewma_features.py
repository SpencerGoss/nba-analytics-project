"""Tests for EWMA (exponentially weighted moving average) features."""

import numpy as np
import pandas as pd
import pytest

from src.features.team_game_features import _ewma_shift, EWMA_SPANS, EWMA_STATS


def _make_team_df(n=25, team_id=1, pts_base=100, net_rtg_base=3.0):
    """Synthetic single-team game log sorted by date."""
    np.random.seed(42)
    dates = pd.date_range("2025-10-20", periods=n, freq="2D")
    return pd.DataFrame({
        "team_id": team_id,
        "game_date": dates,
        "pts": pts_base + np.random.randint(-15, 15, size=n),
        "net_rtg_game": net_rtg_base + np.random.randn(n) * 5,
        "off_rtg_game": 110 + np.random.randn(n) * 4,
        "def_rtg_game": 108 + np.random.randn(n) * 4,
        "wl": np.random.choice(["W", "L"], size=n),
    })


# ── Leakage ───────────────────────────────────────────────────────────────────

def test_ewma_shift_no_leakage():
    """EWMA at row N must not include row N's own value (shift-1)."""
    df = _make_team_df(10)
    # Spike row 5 to a huge value
    df.loc[df.index[5], "pts"] = 999
    result = _ewma_shift(df, "pts", span=7)
    # Row 5's EWMA should NOT reflect 999 — only rows before it
    assert result.iloc[5] < 200, "EWMA at row 5 should not see the spike at row 5"
    # Row 6 SHOULD start reflecting it
    assert result.iloc[6] > result.iloc[4], "Row 6 EWMA should reflect the spike"


# ── Reactivity ────────────────────────────────────────────────────────────────

def test_ewma_span7_more_reactive():
    """Shorter span (7) reacts faster to a sudden shift than longer span (15)."""
    df = _make_team_df(20)
    # First 10 games: pts ~100, then jump to 150
    df.loc[df.index[10:], "pts"] = 150
    ewma7 = _ewma_shift(df, "pts", span=7)
    ewma15 = _ewma_shift(df, "pts", span=15)
    # At row 15, span-7 should be closer to 150 than span-15
    assert ewma7.iloc[15] > ewma15.iloc[15], "span=7 should react faster"


# ── Column existence ──────────────────────────────────────────────────────────

def test_ewma_columns_exist():
    """Verify expected EWMA column names are generated from config."""
    expected = set()
    for span in EWMA_SPANS:
        for src_col, prefix in EWMA_STATS.items():
            expected.add(f"{prefix}_ewma{span}")
    # Must include the key columns referenced in matchup context_cols
    for col in ["net_rtg_ewma7", "net_rtg_ewma15", "pts_ewma7", "pts_ewma15"]:
        assert col in expected, f"{col} not in generated EWMA column set"


# ── NaN behaviour ─────────────────────────────────────────────────────────────

def test_ewma_no_nans_after_warmup():
    """After sufficient rows, EWMA values should not be NaN."""
    df = _make_team_df(20)
    result = _ewma_shift(df, "pts", span=7)
    # Row 0 is NaN (no prior data), but rows 2+ should be filled
    assert result.iloc[2:].notna().all(), "EWMA should have no NaN after warmup"


def test_ewma_first_row_is_nan():
    """First row has no prior games so EWMA should be NaN."""
    df = _make_team_df(5)
    result = _ewma_shift(df, "pts", span=7)
    assert pd.isna(result.iloc[0])


# ── Value ranges ──────────────────────────────────────────────────────────────

def test_ewma_values_reasonable_pts():
    """EWMA of pts should be in a reasonable NBA range (50-200)."""
    df = _make_team_df(30, pts_base=110)
    result = _ewma_shift(df, "pts", span=7)
    valid = result.dropna()
    assert (valid > 50).all() and (valid < 200).all(), "pts EWMA out of range"


def test_ewma_values_reasonable_net_rtg():
    """EWMA of net_rtg should be within (-30, 30) for typical data."""
    df = _make_team_df(30, net_rtg_base=2.0)
    result = _ewma_shift(df, "net_rtg_game", span=15)
    valid = result.dropna()
    assert (valid > -30).all() and (valid < 30).all()


# ── Determinism ───────────────────────────────────────────────────────────────

def test_ewma_deterministic():
    """Same input produces same output."""
    df = _make_team_df(15)
    r1 = _ewma_shift(df.copy(), "pts", span=7)
    r2 = _ewma_shift(df.copy(), "pts", span=7)
    pd.testing.assert_series_equal(r1, r2)


# ── Multiple stats ────────────────────────────────────────────────────────────

def test_ewma_all_configured_stats_computable():
    """_ewma_shift works for every column listed in EWMA_STATS."""
    df = _make_team_df(15)
    for src_col in EWMA_STATS:
        result = _ewma_shift(df, src_col, span=7)
        assert len(result) == len(df)
        assert result.iloc[1:].notna().any(), f"All NaN for {src_col}"
