"""Tests for scripts/build_live_scores.py pure utility functions."""
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_live_scores import (
    _period_label,
    _format_clock,
)


# ─── _period_label ────────────────────────────────────────────────────────────

def test_period_label_first():
    assert _period_label(1) == "1st"


def test_period_label_second():
    assert _period_label(2) == "2nd"


def test_period_label_third():
    assert _period_label(3) == "3rd"


def test_period_label_fourth():
    assert _period_label(4) == "4th"


def test_period_label_ot():
    assert _period_label(5) == "OT"


def test_period_label_double_ot():
    assert _period_label(6) == "OT2"


def test_period_label_triple_ot():
    assert _period_label(7) == "OT3"


def test_period_label_returns_string():
    for p in range(1, 8):
        assert isinstance(_period_label(p), str)


# ─── _format_clock ────────────────────────────────────────────────────────────

def test_format_clock_iso_standard():
    # NBA live API format: "PT04M22.00S"
    assert _format_clock("PT04M22.00S") == "4:22"


def test_format_clock_iso_zero_secs():
    assert _format_clock("PT12M00.00S") == "12:00"


def test_format_clock_iso_single_digit_mins():
    assert _format_clock("PT01M30.00S") == "1:30"


def test_format_clock_already_formatted():
    # Already "4:22" passes through unchanged
    result = _format_clock("4:22")
    assert result == "4:22"


def test_format_clock_empty_string():
    assert _format_clock("") == ""


def test_format_clock_none():
    assert _format_clock(None) == ""


def test_format_clock_iso_full_minute():
    assert _format_clock("PT10M00.00S") == "10:00"


def test_format_clock_seconds_padded():
    # Seconds should be zero-padded
    result = _format_clock("PT05M05.00S")
    assert result == "5:05"
