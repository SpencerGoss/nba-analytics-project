"""
Tests for src/data/api_client.py

Covers:
  - fetch_with_retry: success on first call
  - fetch_with_retry: retries on exception, succeeds on N-th attempt
  - fetch_with_retry: returns failure dict after all retries exhausted
  - fetch_with_retry: returns correct data shape
  - fetch_with_retry: error message captured in result
  - fetch_with_retry: sleep is called between retries
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.api_client import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY,
    fetch_with_retry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok_fn(data=None):
    """Returns a fetch function that always succeeds."""
    if data is None:
        data = pd.DataFrame({"x": [1, 2, 3]})
    return lambda: data


def _fail_fn(exc_type=RuntimeError, msg="boom"):
    """Returns a fetch function that always raises."""
    def fn():
        raise exc_type(msg)
    return fn


def _fail_then_succeed(n_failures: int, data=None):
    """Raise n_failures times, then return data."""
    if data is None:
        data = pd.DataFrame({"v": [99]})
    call_count = [0]
    def fn():
        call_count[0] += 1
        if call_count[0] <= n_failures:
            raise RuntimeError(f"failure #{call_count[0]}")
        return data
    return fn


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------

class TestFetchWithRetrySuccess:
    def test_success_first_attempt_returns_success_true(self):
        result = fetch_with_retry(_ok_fn(), label="test", retry_delay=0)
        assert result["success"] is True

    def test_success_data_returned(self):
        df = pd.DataFrame({"col": [1, 2]})
        result = fetch_with_retry(_ok_fn(df), label="test", retry_delay=0)
        assert result["data"].equals(df)

    def test_success_error_is_none(self):
        result = fetch_with_retry(_ok_fn(), label="test", retry_delay=0)
        assert result["error"] is None

    def test_result_has_required_keys(self):
        result = fetch_with_retry(_ok_fn(), label="test", retry_delay=0)
        for key in ("success", "data", "error"):
            assert key in result, f"Missing key: {key}"

    def test_no_sleep_on_first_success(self):
        with patch("src.data.api_client.time.sleep") as mock_sleep:
            fetch_with_retry(_ok_fn(), label="test", retry_delay=1)
        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# Retry behavior
# ---------------------------------------------------------------------------

class TestFetchWithRetryRetries:
    def test_fails_once_then_succeeds(self):
        fn = _fail_then_succeed(n_failures=1)
        with patch("src.data.api_client.time.sleep"):
            result = fetch_with_retry(fn, label="test", max_retries=3, retry_delay=0)
        assert result["success"] is True

    def test_sleep_called_between_retries(self):
        fn = _fail_then_succeed(n_failures=2)
        with patch("src.data.api_client.time.sleep") as mock_sleep:
            fetch_with_retry(fn, label="test", max_retries=3, retry_delay=5)
        # Should sleep twice (after failure 1 and failure 2, not after success)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(5)

    def test_all_retries_exhausted_returns_failure(self):
        fn = _fail_fn()
        with patch("src.data.api_client.time.sleep"):
            result = fetch_with_retry(fn, label="test", max_retries=3, retry_delay=0)
        assert result["success"] is False
        assert result["data"] is None

    def test_error_message_captured_in_result(self):
        fn = _fail_fn(msg="network timeout")
        with patch("src.data.api_client.time.sleep"):
            result = fetch_with_retry(fn, label="test", max_retries=2, retry_delay=0)
        assert "network timeout" in result["error"]

    def test_max_retries_1_no_sleep(self):
        """With max_retries=1, a single failure should not trigger any sleep."""
        fn = _fail_fn()
        with patch("src.data.api_client.time.sleep") as mock_sleep:
            result = fetch_with_retry(fn, label="test", max_retries=1, retry_delay=5)
        mock_sleep.assert_not_called()
        assert result["success"] is False

    def test_max_retries_2_sleeps_once_on_failure(self):
        """max_retries=2: fail twice -> sleep called once (not after last attempt)."""
        fn = _fail_fn()
        with patch("src.data.api_client.time.sleep") as mock_sleep:
            fetch_with_retry(fn, label="test", max_retries=2, retry_delay=3)
        assert mock_sleep.call_count == 1


# ---------------------------------------------------------------------------
# Default parameter values
# ---------------------------------------------------------------------------

class TestDefaultParameters:
    def test_default_max_retries_constant(self):
        assert isinstance(DEFAULT_MAX_RETRIES, int)
        assert DEFAULT_MAX_RETRIES >= 1

    def test_default_retry_delay_constant(self):
        assert isinstance(DEFAULT_RETRY_DELAY, (int, float))
        assert DEFAULT_RETRY_DELAY > 0
