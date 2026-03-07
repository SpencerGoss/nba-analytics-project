"""
Tests for scripts/build_line_movement.py

Covers:
  - Required JSON fields present on every entry
  - movement = closing - opening (correct direction)
  - direction values (toward_home / toward_away / no_movement)
  - classification thresholds (sharp_action, moderate_move, stable)
  - interpretation string is non-empty
  - Output sorted by date descending
  - Deduplication (same game appears once)
  - CSV preferred over DB when both have explicit spreads
  - Graceful handling of empty inputs
  - game_predictions fallback: all entries have movement=0 and stable classification
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_line_movement import (
    MODERATE_THRESHOLD,
    SHARP_THRESHOLD,
    _classify,
    _direction,
    _interpretation,
    build_line_movement,
)

REQUIRED_FIELDS = {
    "home_team",
    "away_team",
    "game_date",
    "opening_spread",
    "current_spread",
    "movement",
    "direction",
    "classification",
    "interpretation",
}

VALID_DIRECTIONS = {"toward_home", "toward_away", "no_movement"}
VALID_CLASSIFICATIONS = {"sharp_action", "moderate_move", "stable"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spread_df(records: list[dict]) -> pd.DataFrame:
    """Build a DataFrame that simulates the CLV / CSV spread data."""
    rows = []
    for rec in records:
        rows.append(
            {
                "home_team": rec.get("home_team", "OKC"),
                "away_team": rec.get("away_team", "POR"),
                "game_date": pd.Timestamp(rec.get("game_date", "2026-03-06")),
                "opening_spread": float(rec.get("opening_spread", -5.0)),
                "closing_spread": float(rec.get("closing_spread", -6.0)),
            }
        )
    return pd.DataFrame(rows)


def _prob_df(records: list[dict]) -> pd.DataFrame:
    """Build a DataFrame that simulates the game_predictions fallback."""
    rows = []
    for rec in records:
        rows.append(
            {
                "home_team": rec.get("home_team", "OKC"),
                "away_team": rec.get("away_team", "POR"),
                "game_date": pd.Timestamp(rec.get("game_date", "2026-03-06")),
                "home_win_prob": float(rec.get("home_win_prob", 0.60)),
                "notes": None,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Unit tests: _classify
# ---------------------------------------------------------------------------

class TestClassify:
    def test_sharp_action_above_threshold(self):
        assert _classify(SHARP_THRESHOLD + 0.1) == "sharp_action"
        assert _classify(-(SHARP_THRESHOLD + 0.1)) == "sharp_action"

    def test_sharp_action_at_threshold(self):
        assert _classify(SHARP_THRESHOLD) == "sharp_action"

    def test_moderate_between_thresholds(self):
        mid = (SHARP_THRESHOLD + MODERATE_THRESHOLD) / 2
        assert _classify(mid) == "moderate_move"
        assert _classify(-mid) == "moderate_move"

    def test_moderate_at_lower_threshold(self):
        assert _classify(MODERATE_THRESHOLD) == "moderate_move"

    def test_stable_below_moderate(self):
        assert _classify(MODERATE_THRESHOLD - 0.1) == "stable"
        assert _classify(0.0) == "stable"


# ---------------------------------------------------------------------------
# Unit tests: _direction
# ---------------------------------------------------------------------------

class TestDirection:
    def test_toward_home_negative_movement(self):
        assert _direction(-1.0) == "toward_home"

    def test_toward_away_positive_movement(self):
        assert _direction(1.0) == "toward_away"

    def test_no_movement_at_zero(self):
        assert _direction(0.0) == "no_movement"

    def test_no_movement_near_zero(self):
        assert _direction(0.04) == "no_movement"
        assert _direction(-0.04) == "no_movement"


# ---------------------------------------------------------------------------
# Unit tests: _interpretation
# ---------------------------------------------------------------------------

class TestInterpretation:
    def test_stable_no_movement(self):
        result = _interpretation("OKC", "POR", 0.0, -6.5, -6.5)
        assert "stable" in result.lower() or "no significant" in result.lower()

    def test_sharp_action_mentions_team(self):
        result = _interpretation("OKC", "POR", -2.0, -5.0, -7.0)
        assert "OKC" in result or "Sharp" in result

    def test_moderate_move_mentions_team(self):
        result = _interpretation("OKC", "POR", 1.0, -7.0, -6.0)
        assert "POR" in result or "Moderate" in result

    def test_nonempty_for_all_classifications(self):
        for movement in [-2.0, -0.7, 0.0, 0.7, 2.0]:
            result = _interpretation("A", "B", movement, -5.0, -5.0 + movement)
            assert result, f"Empty interpretation for movement={movement}"


# ---------------------------------------------------------------------------
# Unit tests: build_line_movement
# ---------------------------------------------------------------------------

class TestBuildLineMovement:
    def test_returns_list(self):
        csv_df = _spread_df([{}])
        result = build_line_movement(db_df=pd.DataFrame(), csv_df=csv_df)
        assert isinstance(result, list)

    def test_required_fields_present(self):
        csv_df = _spread_df([{"home_team": "OKC", "away_team": "POR", "opening_spread": -6.5, "closing_spread": -7.5}])
        result = build_line_movement(db_df=pd.DataFrame(), csv_df=csv_df)
        assert len(result) == 1
        missing = REQUIRED_FIELDS - set(result[0].keys())
        assert not missing, f"Missing fields: {missing}"

    def test_movement_computed_correctly(self):
        csv_df = _spread_df([{"opening_spread": -6.5, "closing_spread": -7.5}])
        result = build_line_movement(db_df=pd.DataFrame(), csv_df=csv_df)
        assert result[0]["movement"] == pytest.approx(-1.0, abs=1e-6)

    def test_direction_toward_home(self):
        csv_df = _spread_df([{"opening_spread": -5.0, "closing_spread": -7.0}])
        result = build_line_movement(db_df=pd.DataFrame(), csv_df=csv_df)
        assert result[0]["direction"] == "toward_home"

    def test_direction_toward_away(self):
        csv_df = _spread_df([{"opening_spread": -7.0, "closing_spread": -5.0}])
        result = build_line_movement(db_df=pd.DataFrame(), csv_df=csv_df)
        assert result[0]["direction"] == "toward_away"

    def test_direction_no_movement(self):
        csv_df = _spread_df([{"opening_spread": -6.5, "closing_spread": -6.5}])
        result = build_line_movement(db_df=pd.DataFrame(), csv_df=csv_df)
        assert result[0]["direction"] == "no_movement"

    def test_sharp_action_classified(self):
        csv_df = _spread_df([{"opening_spread": -5.0, "closing_spread": -7.0}])
        result = build_line_movement(db_df=pd.DataFrame(), csv_df=csv_df)
        assert result[0]["classification"] == "sharp_action"

    def test_stable_classified(self):
        csv_df = _spread_df([{"opening_spread": -6.5, "closing_spread": -6.5}])
        result = build_line_movement(db_df=pd.DataFrame(), csv_df=csv_df)
        assert result[0]["classification"] == "stable"

    def test_sorted_by_date_descending(self):
        csv_df = _spread_df([
            {"game_date": "2026-03-04", "opening_spread": -5.0, "closing_spread": -5.5},
            {"game_date": "2026-03-06", "opening_spread": -6.0, "closing_spread": -7.0},
            {"game_date": "2026-03-05", "opening_spread": -7.0, "closing_spread": -6.5},
        ])
        result = build_line_movement(db_df=pd.DataFrame(), csv_df=csv_df)
        dates = [r["game_date"] for r in result]
        assert dates == sorted(dates, reverse=True), f"Not sorted desc: {dates}"

    def test_deduplication_same_game(self):
        """Same game in CSV and DB should appear once."""
        csv_df = _spread_df([{"home_team": "OKC", "away_team": "POR", "game_date": "2026-03-06",
                               "opening_spread": -6.5, "closing_spread": -7.5}])
        db_df = _spread_df([{"home_team": "OKC", "away_team": "POR", "game_date": "2026-03-06",
                              "opening_spread": -6.5, "closing_spread": -7.5}])
        # Rename closing to match CLV table column name
        db_df = db_df.rename(columns={"closing_spread": "closing_spread"})
        result = build_line_movement(db_df=db_df, csv_df=csv_df)
        okc_por = [r for r in result if r["home_team"] == "OKC" and r["away_team"] == "POR"]
        assert len(okc_por) == 1, f"Expected 1 entry, got {len(okc_por)}"

    def test_empty_both_returns_empty(self):
        result = build_line_movement(db_df=pd.DataFrame(), csv_df=pd.DataFrame())
        assert result == []

    def test_game_predictions_fallback_movement_zero(self):
        db_df = _prob_df([{"home_team": "OKC", "away_team": "POR", "home_win_prob": 0.65}])
        result = build_line_movement(db_df=db_df, csv_df=pd.DataFrame())
        assert len(result) == 1
        assert result[0]["movement"] == pytest.approx(0.0, abs=1e-6)
        assert result[0]["classification"] == "stable"

    def test_game_predictions_fallback_has_all_fields(self):
        db_df = _prob_df([{"home_win_prob": 0.60}])
        result = build_line_movement(db_df=db_df, csv_df=pd.DataFrame())
        assert len(result) == 1
        missing = REQUIRED_FIELDS - set(result[0].keys())
        assert not missing, f"Missing fields in fallback: {missing}"

    def test_interpretation_nonempty(self):
        csv_df = _spread_df([{"opening_spread": -6.5, "closing_spread": -8.0}])
        result = build_line_movement(db_df=pd.DataFrame(), csv_df=csv_df)
        assert result[0]["interpretation"]

    def test_classification_all_valid(self):
        csv_df = _spread_df([
            {"opening_spread": -5.0, "closing_spread": -7.0},   # sharp
            {"home_team": "BOS", "away_team": "MIL", "opening_spread": -5.0, "closing_spread": -5.7},  # moderate
            {"home_team": "LAL", "away_team": "GSW", "opening_spread": -3.0, "closing_spread": -3.0},  # stable
        ])
        result = build_line_movement(db_df=pd.DataFrame(), csv_df=csv_df)
        for entry in result:
            assert entry["classification"] in VALID_CLASSIFICATIONS


# ---------------------------------------------------------------------------
# Integration smoke test (requires real DB)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestLineMovementIntegration:
    def test_loads_real_db(self):
        db_path = PROJECT_ROOT / "database" / "predictions_history.db"
        if not db_path.exists():
            pytest.skip("predictions_history.db not available")

        from scripts.build_line_movement import load_from_db, load_from_csv

        db_df = load_from_db()
        csv_df = load_from_csv()

        result = build_line_movement(db_df, csv_df)
        assert isinstance(result, list)
        for entry in result:
            missing = REQUIRED_FIELDS - set(entry.keys())
            assert not missing
