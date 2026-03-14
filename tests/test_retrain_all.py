"""Tests for retrain_all.py — step runner and pipeline orchestration."""

from __future__ import annotations

import subprocess
import sys
import os
from unittest.mock import patch, MagicMock, ANY

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.retrain_all import run_step, main


# ---------------------------------------------------------------------------
# run_step
# ---------------------------------------------------------------------------

class TestRunStep:
    def test_success_returns_true(self):
        result = MagicMock()
        result.returncode = 0
        with patch("subprocess.run", return_value=result) as mock_run:
            assert run_step("Test step", ["echo", "hello"]) is True
            mock_run.assert_called_once_with(
                ["echo", "hello"], capture_output=False, text=True, env=ANY
            )

    def test_failure_returns_false(self):
        result = MagicMock()
        result.returncode = 1
        with patch("subprocess.run", return_value=result):
            assert run_step("Failing step", ["false"]) is False

    def test_nonzero_exit_code(self):
        result = MagicMock()
        result.returncode = 137
        with patch("subprocess.run", return_value=result):
            assert run_step("Killed step", ["killed"]) is False


# ---------------------------------------------------------------------------
# main — pipeline orchestration
# ---------------------------------------------------------------------------

class TestMain:
    def _mock_run(self, returncode=0):
        result = MagicMock()
        result.returncode = returncode
        return result

    def test_all_steps_succeed(self):
        """When all 6 steps pass, main completes without sys.exit."""
        with patch("subprocess.run", return_value=self._mock_run(0)) as mock_run:
            main()
        # 6 steps should produce 6 subprocess.run calls
        assert mock_run.call_count == 6

    def test_step1_failure_exits(self):
        """When step 1 fails, pipeline exits immediately."""
        with patch("subprocess.run", return_value=self._mock_run(1)):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_step3_failure_exits(self):
        """When step 3 fails (model training), pipeline exits after step 2 succeeded."""
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            # Steps 1, 2 succeed; step 3 fails
            result.returncode = 1 if call_count == 3 else 0
            return result

        with patch("subprocess.run", side_effect=side_effect):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
        assert call_count == 3  # stopped at step 3

    def test_steps_run_in_order(self):
        """Verify the 6 steps run in the expected order."""
        calls = []

        def capture_call(cmd, **kwargs):
            # Extract the meaningful part of the command
            cmd_str = " ".join(str(c) for c in cmd)
            calls.append(cmd_str)
            result = MagicMock()
            result.returncode = 0
            return result

        with patch("subprocess.run", side_effect=capture_call):
            main()

        assert len(calls) == 6
        # Step 1: lineup features
        assert "lineup_features" in calls[0]
        # Step 2: team_game_features
        assert "team_game_features" in calls[1]
        # Step 3: game_outcome_model
        assert "game_outcome_model" in calls[2]
        # Step 4: calibration
        assert "calibration" in calls[3]
        # Step 5: margin_model
        assert "margin_model" in calls[4]
        # Step 6: ats_model
        assert "ats_model" in calls[5]
