"""Tests for predict_cli.py — argument parsing and ATS handler logic."""

from __future__ import annotations

import argparse
import sys
import os
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

# Ensure project root on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.predict_cli import parse_args, _handle_ats


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_game_command(self):
        with patch("sys.argv", ["cli", "game", "--home", "BOS", "--away", "LAL"]):
            args = parse_args()
        assert args.command == "game"
        assert args.home == "BOS"
        assert args.away == "LAL"
        assert args.date is None

    def test_game_with_date(self):
        with patch("sys.argv", ["cli", "game", "--home", "BOS", "--away", "LAL", "--date", "2026-03-13"]):
            args = parse_args()
        assert args.date == "2026-03-13"

    def test_player_command(self):
        with patch("sys.argv", ["cli", "player", "--name", "LeBron James"]):
            args = parse_args()
        assert args.command == "player"
        assert args.name == "LeBron James"

    def test_ats_command(self):
        with patch("sys.argv", ["cli", "ats", "--home", "BOS", "--away", "LAL", "--spread", "-3.5"]):
            args = parse_args()
        assert args.command == "ats"
        assert args.spread == -3.5
        assert args.home_ml is None
        assert args.away_ml is None

    def test_ats_with_moneylines(self):
        with patch("sys.argv", ["cli", "ats", "--home", "BOS", "--away", "LAL",
                                 "--spread", "-3.5", "--home-ml", "-150", "--away-ml", "130"]):
            args = parse_args()
        assert args.home_ml == -150.0
        assert args.away_ml == 130.0

    def test_value_bet_command_defaults(self):
        with patch("sys.argv", ["cli", "value-bet"]):
            args = parse_args()
        assert args.command == "value-bet"
        assert args.live is False
        assert args.threshold == 0.05

    def test_value_bet_with_options(self):
        with patch("sys.argv", ["cli", "value-bet", "--live", "--threshold", "0.10"]):
            args = parse_args()
        assert args.live is True
        assert args.threshold == 0.10

    def test_missing_command_raises(self):
        with patch("sys.argv", ["cli"]):
            with pytest.raises(SystemExit):
                parse_args()

    def test_game_missing_home_raises(self):
        with patch("sys.argv", ["cli", "game", "--away", "LAL"]):
            with pytest.raises(SystemExit):
                parse_args()

    def test_ats_missing_spread_raises(self):
        with patch("sys.argv", ["cli", "ats", "--home", "BOS", "--away", "LAL"]):
            with pytest.raises(SystemExit):
                parse_args()


# ---------------------------------------------------------------------------
# _handle_ats — matchup file missing
# ---------------------------------------------------------------------------

class TestHandleAts:
    def test_missing_matchup_file_returns_error(self, tmp_path):
        """When matchup features CSV doesn't exist, return error dict."""
        args = argparse.Namespace(home="BOS", away="LAL", spread=-3.5,
                                  home_ml=None, away_ml=None)
        fake_path = str(tmp_path / "missing.csv")
        with patch("src.models.predict_cli.os.path.join", return_value=fake_path):
            result = _handle_ats(args)
        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_no_history_returns_error(self, tmp_path):
        """When no history exists for the matchup, return error dict."""
        # Create a minimal CSV with a different matchup
        csv_path = tmp_path / "game_matchup_features.csv"
        df = pd.DataFrame({
            "home_team": ["MIA"],
            "away_team": ["ORL"],
            "game_date": ["2026-01-01"],
            "season": [202526],
            "diff_elo": [50.0],
        })
        df.to_csv(csv_path, index=False)

        args = argparse.Namespace(home="BOS", away="LAL", spread=-3.5,
                                  home_ml=None, away_ml=None)
        with patch("src.models.predict_cli.os.path.join", return_value=str(csv_path)), \
             patch("src.models.predict_cli.os.path.exists", return_value=True), \
             patch("src.models.game_outcome_model._get_current_season_code", return_value=202526), \
             patch("src.models.game_outcome_model._synthesize_matchup_row", return_value=None):
            result = _handle_ats(args)
        assert "error" in result
        assert "not enough history" in result["error"].lower()
