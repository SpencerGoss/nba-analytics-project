"""Shared constants for the NBA analytics project.

Centralizes values that were previously hardcoded across multiple files:
seasons, teams, paths, API settings, model parameters.
"""
from datetime import date
from pathlib import Path


# -- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
ODDS_DIR = DATA_DIR / "odds"
DASHBOARD_DATA_DIR = PROJECT_ROOT / "dashboard" / "data"
LOGS_DIR = PROJECT_ROOT / "logs"


# -- Seasons ---------------------------------------------------------------
MODERN_ERA_START = 201314       # Earliest season for full feature coverage
CALIBRATION_SEASON = 202122     # Permanently held out from CV
LINEUP_START_SEASON = 201516    # Lineup features are 0.0 before this


def get_current_season() -> int:
    """Derive current NBA season code from today's date.

    Oct+ = new season start. Returns integer like 202526.
    """
    today = date.today()
    start_year = today.year if today.month >= 10 else today.year - 1
    end_year = start_year + 1
    return int(f"{start_year}{str(end_year)[-2:]}")


def get_current_season_id() -> int:
    """Derive season_id used by player_game_logs.csv.

    Format: prefix '2' + start year. E.g. 202526 season -> 22025.
    """
    today = date.today()
    start_year = today.year if today.month >= 10 else today.year - 1
    return int(f"2{start_year}")


# -- Teams -----------------------------------------------------------------
EAST_TEAMS = [
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DET", "IND",
    "MIA", "MIL", "NYK", "ORL", "PHI", "TOR", "WAS",
]
WEST_TEAMS = [
    "DAL", "DEN", "GSW", "HOU", "LAC", "LAL", "MEM", "MIN",
    "NOP", "OKC", "PHX", "POR", "SAC", "SAS", "UTA",
]
ALL_TEAMS = sorted(EAST_TEAMS + WEST_TEAMS)

# Division mappings
EAST_DIVISIONS: dict[str, list[str]] = {
    "Atlantic":  ["BOS", "BKN", "NYK", "PHI", "TOR"],
    "Central":   ["CHI", "CLE", "DET", "IND", "MIL"],
    "Southeast": ["ATL", "CHA", "MIA", "ORL", "WAS"],
}
WEST_DIVISIONS: dict[str, list[str]] = {
    "Northwest": ["DEN", "MIN", "OKC", "POR", "UTA"],
    "Pacific":   ["GSW", "LAC", "LAL", "PHX", "SAC"],
    "Southwest": ["DAL", "HOU", "MEM", "NOP", "SAS"],
}

TEAM_ABBREV_TO_FULL = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}


# -- API Settings ----------------------------------------------------------
NBA_API_DELAY = 0.6             # Seconds between nba_api requests
PINNACLE_LEAGUE_ID = 487        # NBA league ID for Pinnacle API


# -- Model Defaults --------------------------------------------------------
DEFAULT_KELLY_FRACTION = 0.25   # Quarter Kelly
MAX_KELLY_FRACTION = 0.05       # 5% cap on any single bet
VALUE_BET_THRESHOLD = 0.03      # Minimum edge to flag as value bet
ATS_WEIGHT = 0.0                # ATS model disabled in ensemble

# Confidence tier thresholds
BEST_BET_EDGE = 0.08
SOLID_PICK_EDGE = 0.04
LEAN_EDGE = 0.02
