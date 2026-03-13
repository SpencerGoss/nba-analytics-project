import pytest
from datetime import date


def test_current_season_is_integer():
    from src.config import get_current_season
    season = get_current_season()
    assert isinstance(season, int)
    assert season > 200000


def test_current_season_format():
    """Season code should be 6 digits like 202526."""
    from src.config import get_current_season
    season = get_current_season()
    s = str(season)
    assert len(s) == 6
    # Last 2 digits = first 4 digits' last 2 + 1
    start = int(s[:4])
    end = int(s[4:])
    assert end == (start + 1) % 100


def test_conference_teams():
    from src.config import EAST_TEAMS, WEST_TEAMS, ALL_TEAMS
    assert len(EAST_TEAMS) == 15
    assert len(WEST_TEAMS) == 15
    assert len(ALL_TEAMS) == 30
    assert "BOS" in EAST_TEAMS
    assert "LAL" in WEST_TEAMS
    assert set(EAST_TEAMS) & set(WEST_TEAMS) == set()  # no overlap


def test_modern_era_start():
    from src.config import MODERN_ERA_START
    assert isinstance(MODERN_ERA_START, int)
    assert MODERN_ERA_START == 201314


def test_calibration_season():
    from src.config import CALIBRATION_SEASON
    assert CALIBRATION_SEASON == 202122


def test_paths():
    from src.config import PROJECT_ROOT, ARTIFACTS_DIR, DATA_DIR
    from pathlib import Path
    assert isinstance(PROJECT_ROOT, Path)
    assert isinstance(ARTIFACTS_DIR, Path)
    assert isinstance(DATA_DIR, Path)
    assert ARTIFACTS_DIR == PROJECT_ROOT / "models" / "artifacts"


def test_api_throttle():
    from src.config import NBA_API_DELAY
    assert NBA_API_DELAY >= 0.6


def test_team_name_mapping():
    from src.config import TEAM_ABBREV_TO_FULL
    assert TEAM_ABBREV_TO_FULL["BOS"] == "Boston Celtics"
    assert TEAM_ABBREV_TO_FULL["LAL"] == "Los Angeles Lakers"
    assert len(TEAM_ABBREV_TO_FULL) == 30
