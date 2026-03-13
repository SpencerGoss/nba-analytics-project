"""
Tests for predict_game() in src/models/game_outcome_model.py

Covers:
  - Current-season filtering: predict_game() must not use stale cross-season data
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_predict_game_uses_current_season_row():
    """predict_game must filter to latest season, not use stale cross-season data.

    diff_elo is 37.3% feature importance — using a row from a prior season
    with stale Elo values would corrupt ~3 of 6 daily predictions.
    """
    from src.models import game_outcome_model

    source = inspect.getsource(game_outcome_model.predict_game)
    # Must either filter by season or sort by date and take last row
    has_season_filter = "current_season" in source or "latest_season" in source
    has_date_sort = "sort_values" in source and ("iloc[-1]" in source or "tail(1)" in source)
    assert has_season_filter or has_date_sort, (
        "predict_game() does not filter to current season — may use stale cross-season data"
    )


def test_synthesize_matchup_row_uses_current_season():
    """_synthesize_matchup_row must also filter to current season as fallback."""
    from src.models import game_outcome_model

    source = inspect.getsource(game_outcome_model._synthesize_matchup_row)
    assert "current_season" in source or "latest_season" in source, (
        "_synthesize_matchup_row() does not filter to current season"
    )
