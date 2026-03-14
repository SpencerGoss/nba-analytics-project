"""Tests for scripts/fetch_player_positions.py — position mapping logic.

Tests the height parser, APG lookup, and smart position mapping without
hitting the NBA API.
"""

import pytest

from scripts.fetch_player_positions import (
    _parse_height_inches,
    _map_positions,
)


# ── Height parsing ───────────────────────────────────────────────────────────

class TestParseHeightInches:
    def test_standard_height(self):
        assert _parse_height_inches("6-6") == 78

    def test_short_player(self):
        assert _parse_height_inches("5-9") == 69

    def test_tall_player(self):
        assert _parse_height_inches("7-4") == 88

    def test_empty_string(self):
        assert _parse_height_inches("") is None

    def test_none_input(self):
        assert _parse_height_inches(None) is None

    def test_whitespace_only(self):
        assert _parse_height_inches("  ") is None

    def test_invalid_format(self):
        assert _parse_height_inches("six-two") is None

    def test_single_number(self):
        assert _parse_height_inches("6") is None

    def test_six_feet_even(self):
        assert _parse_height_inches("6-0") == 72

    def test_strips_whitespace(self):
        assert _parse_height_inches(" 6-6 ") == 78


# ── Position mapping: Guards ────────────────────────────────────────────────

class TestMapPositionsGuard:
    def test_short_guard_is_pg(self):
        """G, 6'2" -> PG regardless of APG."""
        assert _map_positions("G", 74, 3.0) == "PG"

    def test_guard_6_4_is_pg(self):
        """G, 6'4" (76") -> PG."""
        assert _map_positions("G", 76, 2.0) == "PG"

    def test_mid_guard_high_apg_is_pg_sg(self):
        """G, 6'5" (77"), APG>=5 -> PG, SG (e.g., SGA)."""
        assert _map_positions("G", 77, 6.6) == "PG, SG"

    def test_mid_guard_low_apg_is_sg(self):
        """G, 6'6" (78"), APG<5 -> SG (e.g., Reaves)."""
        assert _map_positions("G", 78, 4.8) == "SG"

    def test_tall_guard_is_sg(self):
        """G, 6'7"+ (79") -> SG."""
        assert _map_positions("G", 79, 7.0) == "SG"

    def test_curry_pg(self):
        """Stephen Curry: G, 6'2" -> PG."""
        assert _map_positions("G", 74, 5.1) == "PG"

    def test_sga_pg_sg(self):
        """SGA: G, 6'6", 6.6 APG -> PG, SG."""
        assert _map_positions("G", 78, 6.6) == "PG, SG"


# ── Position mapping: Guard-Forward ──────────────────────────────────────────

class TestMapPositionsGuardForward:
    def test_gf_is_sg_sf(self):
        """G-F -> SG, SF (e.g., Jaylen Brown)."""
        assert _map_positions("G-F", 78, 3.5) == "SG, SF"

    def test_gf_ignores_apg(self):
        """G-F -> SG, SF regardless of assist rate."""
        assert _map_positions("G-F", 78, 8.0) == "SG, SF"


# ── Position mapping: Forward-Guard ──────────────────────────────────────────

class TestMapPositionsForwardGuard:
    def test_fg_high_apg_is_pg_sg(self):
        """F-G, APG>=5 -> PG, SG (e.g., Luka)."""
        assert _map_positions("F-G", 78, 8.2) == "PG, SG"

    def test_fg_low_apg_is_sg_sf(self):
        """F-G, APG<5 -> SG, SF (e.g., Tatum)."""
        assert _map_positions("F-G", 80, 4.6) == "SG, SF"

    def test_luka_pg_sg(self):
        """Luka Doncic: F-G, 6'7", 8.2 APG -> PG, SG."""
        assert _map_positions("F-G", 79, 8.2) == "PG, SG"

    def test_tatum_sg_sf(self):
        """Jayson Tatum: F-G, 6'8", 4.6 APG -> SG, SF."""
        assert _map_positions("F-G", 80, 4.6) == "SG, SF"


# ── Position mapping: Forwards ───────────────────────────────────────────────

class TestMapPositionsForward:
    def test_short_forward_is_sf(self):
        """F, 6'7" (79") -> SF."""
        assert _map_positions("F", 79, 2.0) == "SF"

    def test_mid_forward_is_sf_pf(self):
        """F, 6'8"-6'9" (80-81") -> SF, PF."""
        assert _map_positions("F", 80, 3.0) == "SF, PF"
        assert _map_positions("F", 81, 3.0) == "SF, PF"

    def test_tall_forward_is_pf(self):
        """F, 6'10"+ (82") -> PF."""
        assert _map_positions("F", 82, 2.0) == "PF"

    def test_lebron_sf_pf(self):
        """LeBron James: F, 6'9" -> SF, PF."""
        assert _map_positions("F", 81, 7.1) == "SF, PF"


# ── Position mapping: Centers ────────────────────────────────────────────────

class TestMapPositionsCenters:
    def test_fc_is_pf_c(self):
        """F-C -> PF, C (e.g., Anthony Davis)."""
        assert _map_positions("F-C", 82, 2.0) == "PF, C"

    def test_cf_is_c_pf(self):
        """C-F -> C, PF."""
        assert _map_positions("C-F", 83, 1.5) == "C, PF"

    def test_center_is_c(self):
        """C -> C (e.g., Jokic)."""
        assert _map_positions("C", 83, 9.0) == "C"

    def test_jokic_c(self):
        """Nikola Jokic: C, 6'11" -> C."""
        assert _map_positions("C", 83, 9.8) == "C"

    def test_ad_pf_c(self):
        """Anthony Davis: F-C, 6'10" -> PF, C."""
        assert _map_positions("F-C", 82, 2.3) == "PF, C"


# ── Edge cases ───────────────────────────────────────────────────────────────

class TestMapPositionsEdgeCases:
    def test_empty_label(self):
        assert _map_positions("", 78, 5.0) == ""

    def test_none_label(self):
        assert _map_positions(None, 78, 5.0) == ""

    def test_lowercase_label(self):
        """Labels are normalized to uppercase."""
        assert _map_positions("g", 74, 5.0) == "PG"

    def test_unknown_label_starting_with_g(self):
        """Unknown label starting with G falls back to SG."""
        assert _map_positions("G-C", 80, 5.0) == "SG"

    def test_unknown_label_starting_with_f(self):
        """Unknown label starting with F falls back to SF."""
        assert _map_positions("F-X", 80, 5.0) == "SF"

    def test_unknown_label_starting_with_c(self):
        """Unknown label starting with C falls back to C."""
        assert _map_positions("C-X", 80, 5.0) == "C"

    def test_none_height_defaults(self):
        """None height defaults to 78 inches (6'6")."""
        # G with default height 78 and APG<5 -> SG
        assert _map_positions("G", None, 4.0) == "SG"

    def test_primary_is_first_position(self):
        """Primary position is the first in the comma-separated list."""
        positions = _map_positions("F-G", 78, 8.0)
        primary = positions.split(",")[0].strip()
        assert primary == "PG"

    def test_apg_boundary_exactly_5(self):
        """APG exactly at 5 triggers the high-assist path."""
        assert _map_positions("G", 77, 5.0) == "PG, SG"
        assert _map_positions("F-G", 78, 5.0) == "PG, SG"
