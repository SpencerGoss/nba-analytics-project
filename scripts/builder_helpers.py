"""Shared helpers for dashboard builder scripts.

Consolidates functions that were duplicated across 4+ builder files.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEAMS_CSV = PROJECT_ROOT / "data" / "processed" / "teams.csv"


def load_team_names(teams_csv: Path = TEAMS_CSV) -> dict[str, str]:
    """Return {abbreviation: full_name} from teams.csv.

    Falls back to config.TEAM_ABBREV_TO_FULL if CSV is missing.
    """
    if teams_csv.exists():
        df = pd.read_csv(teams_csv, usecols=["abbreviation", "full_name"])
        return dict(zip(df["abbreviation"], df["full_name"]))
    # Fallback to config constants
    from src.config import TEAM_ABBREV_TO_FULL
    return dict(TEAM_ABBREV_TO_FULL)


def record_str(wins: int, losses: int) -> str:
    """Format a win-loss record as 'W-L'."""
    return f"{wins}-{losses}"


def games_behind(
    leader_wins: int, leader_losses: int,
    team_wins: int, team_losses: int,
) -> float:
    """Standard GB formula: ((leader_W - team_W) + (team_L - leader_L)) / 2."""
    return ((leader_wins - team_wins) + (team_losses - leader_losses)) / 2.0


def safe_float(
    val: Any,
    default: float | None = None,
    decimals: int | None = None,
) -> float | None:
    """Convert value to float safely, returning default on failure.

    Parameters
    ----------
    val : Any
        Value to convert.
    default : float | None
        Returned when conversion fails or val is NaN. None by default.
    decimals : int | None
        If provided, round result to this many decimal places.
    """
    if val is None:
        return default
    try:
        f = float(val)
    except (TypeError, ValueError):
        return default
    if pd.isna(f):
        return default
    if decimals is not None:
        f = round(f, decimals)
    return f


def write_json(data: dict | list, out_path: Path) -> None:
    """Write compact JSON to out_path, creating parent dirs as needed."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, separators=(",", ":"), default=str)


def load_json(path: Path) -> dict | list:
    """Load JSON from path, returning empty dict if missing."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
