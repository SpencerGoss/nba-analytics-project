"""
NBA Injury Data Fetcher
=========================
Fetches the current NBA pre-game injury report and saves dated snapshots to
data/raw/injuries/ for daily archival.

INFERENCE PATH ONLY -- this module fetches live/recent injury data for
daily updates. It must never be used for historical training features.
The training path uses the game-log-based proxy in
src/features/injury_proxy.py (FR-4.4).

Design notes:
  - Primary source: NBA official PDF injury reports
    (https://ak-static.cms.nba.com/referee/injury/Injury-Report_DATE_TIME.pdf)
  - The nba_api LeagueInjuryReport endpoint is not available in all nba_api
    versions; this script uses the PDF approach with common time slots.
  - Each run saves a dated CSV to data/raw/injuries/injury_report_YYYYMMDD.csv
    overwriting any file from the same date (one canonical snapshot per day).
  - load_historical_injuries() concatenates all saved CSVs for analysis.

Output columns:
    date         -- game date (YYYY-MM-DD)
    player_name  -- player full name
    player_id    -- NBA player ID (str, may be empty if not resolvable)
    team_abbr    -- 3-letter team abbreviation (e.g. "BOS", "LAL")
    status       -- one of: out, questionable, probable, day-to-day
    injury_type  -- reason/description from the report (free text)

Usage:
    from src.data.get_injury_data import get_injury_report
    df = get_injury_report()

    # Accumulate all saved snapshots:
    from src.data.get_injury_data import load_historical_injuries
    history = load_historical_injuries()

    python src/data/get_injury_data.py
"""

import io
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import logging

log = logging.getLogger(__name__)

# Allow running as a script from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# -- Code path boundary (FR-4.4) -----------------------------------------------
# This module is INFERENCE PATH ONLY.
# NEVER import src.features.injury_proxy.build_injury_proxy_features here.
# The training path is src/features/injury_proxy.py.
_CODE_PATH = "INFERENCE"

# ── Constants ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_INJURIES_DIR = PROJECT_ROOT / "data" / "raw" / "injuries"

# NBA official PDF URL pattern — multiple time slots published per day
NBA_INJURY_PDF_PATTERN = (
    "https://ak-static.cms.nba.com/referee/injury/"
    "Injury-Report_{date}_{time}.pdf"
)

# Time slots to try, in reverse order (most recent first)
PDF_TIME_SLOTS = ["06_00PM", "05_00PM", "01_30PM", "11_00AM", "06_00AM"]

# Columns as published in the NBA injury PDF
PDF_COLUMNS = [
    "game_date", "game_time", "matchup", "team",
    "player_name", "current_status", "reason",
]

# Statuses to include (map to normalized output status values)
STATUS_MAP = {
    "Out":          "out",
    "Questionable": "questionable",
    "Probable":     "probable",
    "Day-To-Day":   "day-to-day",
}

# Well-known team name -> abbreviation mapping (NBA PDF uses full names)
TEAM_NAME_TO_ABB = {
    "Atlanta Hawks":          "ATL",
    "Boston Celtics":         "BOS",
    "Brooklyn Nets":          "BKN",
    "Charlotte Hornets":      "CHA",
    "Chicago Bulls":          "CHI",
    "Cleveland Cavaliers":    "CLE",
    "Dallas Mavericks":       "DAL",
    "Denver Nuggets":         "DEN",
    "Detroit Pistons":        "DET",
    "Golden State Warriors":  "GSW",
    "Houston Rockets":        "HOU",
    "Indiana Pacers":         "IND",
    "Los Angeles Clippers":   "LAC",
    "Los Angeles Lakers":     "LAL",
    "Memphis Grizzlies":      "MEM",
    "Miami Heat":             "MIA",
    "Milwaukee Bucks":        "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans":   "NOP",
    "New York Knicks":        "NYK",
    "Oklahoma City Thunder":  "OKC",
    "Orlando Magic":          "ORL",
    "Philadelphia 76ers":     "PHI",
    "Phoenix Suns":           "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings":       "SAC",
    "San Antonio Spurs":      "SAS",
    "Toronto Raptors":        "TOR",
    "Utah Jazz":              "UTA",
    "Washington Wizards":     "WAS",
}


# ── PDF fetcher ────────────────────────────────────────────────────────────────

def _fetch_pdf(date_str: str) -> pd.DataFrame:
    """
    Download and parse the NBA official injury report PDF for the given date.

    Tries PDF_TIME_SLOTS in reverse order (most recent first) until one
    succeeds. Parses in memory (PDF never written to disk).

    Args:
        date_str: Date in YYYY-MM-DD format.

    Returns:
        Raw DataFrame from the PDF, or empty DataFrame if all slots fail.
    """
    try:
        import pdfplumber
    except ImportError:
        log.info("  [get_injury_data] pdfplumber not installed. "
            "Install with: pip install pdfplumber>=0.10.0")
        return pd.DataFrame()

    for time_slot in PDF_TIME_SLOTS:
        url = NBA_INJURY_PDF_PATTERN.format(date=date_str, time=time_slot)
        time.sleep(1)  # polite rate limit between attempts
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()

            rows = []
            with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
                for page in pdf.pages:
                    table = page.extract_table()
                    if table and len(table) > 1:
                        rows.extend(table[1:])  # skip header row per page

            if not rows:
                log.info(f"  [get_injury_data] PDF at {time_slot} had no extractable rows.")
                continue

            df = pd.DataFrame(rows, columns=PDF_COLUMNS)
            log.info(f"  [get_injury_data] PDF ({time_slot}): "
                f"parsed {len(df)} rows for {date_str}")
            return df

        except requests.exceptions.HTTPError as http_err:
            code = http_err.response.status_code if http_err.response else "unknown"
            if code in (403, 404):
                continue  # this time slot not published yet
            log.error(f"  [get_injury_data] HTTP error at {time_slot}: {http_err}")
            continue
        except Exception as exc:
            log.error(f"  [get_injury_data] Parse error at {time_slot}: {exc}")
            continue

    log.error(f"  [get_injury_data] All PDF time slots failed for {date_str}. "
        "May be off-season or report not yet published.")
    return pd.DataFrame()


# ── nba_api fallback ───────────────────────────────────────────────────────────

def _fetch_nba_api() -> pd.DataFrame:
    """
    Attempt to fetch injury report via nba_api LeagueInjuryReport endpoint.

    Note: LeagueInjuryReport is not available in all nba_api versions.
    Returns empty DataFrame if the endpoint is unavailable or the call fails.
    """
    try:
        from nba_api.stats.endpoints import leagueinjuryreport

        time.sleep(1)
        report = leagueinjuryreport.LeagueInjuryReport(
            league_id="00",
            season_type="Regular Season",
        ).get_data_frames()[0]

        if report.empty:
            return pd.DataFrame()

        report.columns = [c.lower() for c in report.columns]
        log.info(f"  [get_injury_data] nba_api: fetched {len(report)} injury entries.")
        return report

    except ImportError:
        return pd.DataFrame()
    except Exception as exc:
        log.error(f"  [get_injury_data] nba_api fetch failed: {exc}")
        return pd.DataFrame()


# ── Public interface ───────────────────────────────────────────────────────────

def get_injury_report(
    season_year: int | None = None,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Fetch today's NBA injury report and save a dated snapshot to
    data/raw/injuries/.

    Tries (in order):
    1. nba_api LeagueInjuryReport endpoint (if available)
    2. NBA official PDF injury report (multiple time slots)

    Output schema:
        date         -- game date (YYYY-MM-DD)
        player_name  -- full player name
        player_id    -- always empty string (not resolvable from PDF)
        team_abbr    -- 3-letter abbreviation, or raw team name if unmapped
        status       -- normalized: out | questionable | probable | day-to-day
        injury_type  -- reason text from the report

    Args:
        season_year:  Ignored (kept for API compatibility — injury reports are
                      always current-day snapshots). Pass None or current year.
        output_dir:   Directory to save CSV (default: data/raw/injuries/).

    Returns:
        DataFrame with the output schema above. Empty DataFrame on total failure.
    """
    save_dir = Path(output_dir) if output_dir else RAW_INJURIES_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    today_str = datetime.now().strftime("%Y-%m-%d")
    log.info(f"[get_injury_data] Fetching NBA injury report for {today_str}...")

    # -- Try nba_api first (may not be available in all versions)
    raw = _fetch_nba_api()

    if not raw.empty and "player_name" in raw.columns:
        # Normalize nba_api response to output schema
        df = _normalize_nba_api_response(raw, today_str)
    else:
        # Fall back to PDF
        if raw.empty:
            log.info("[get_injury_data] Falling back to PDF source...")
        raw_pdf = _fetch_pdf(today_str)
        if raw_pdf.empty:
            warnings.warn(
                "[get_injury_data] Could not fetch injury report from any source. "
                "Returning empty DataFrame.",
                RuntimeWarning,
                stacklevel=2,
            )
            return pd.DataFrame(
                columns=["date", "player_name", "player_id", "team_abbr",
                         "status", "injury_type"]
            )
        df = _normalize_pdf_response(raw_pdf, today_str)

    if df.empty:
        log.info(f"  [get_injury_data] No actionable injury entries for {today_str}.")
        return df

    # -- Save dated snapshot (one file per day, overwrite if re-run same day)
    out_path = save_dir / f"injury_report_{today_str}.csv"
    df.to_csv(out_path, index=False)
    log.info(f"  [get_injury_data] Saved {len(df)} rows -> {out_path}")

    return df


def _normalize_nba_api_response(raw: pd.DataFrame, date_str: str) -> pd.DataFrame:
    """Normalize a nba_api LeagueInjuryReport DataFrame to output schema."""
    rows = []
    for _, row in raw.iterrows():
        status_raw = str(row.get("player_status", "")).strip()
        status_norm = STATUS_MAP.get(status_raw, status_raw.lower())
        if not status_norm:
            continue
        team_name = str(row.get("team_name", row.get("team", ""))).strip()
        rows.append({
            "date":        date_str,
            "player_name": str(row.get("player_name", "")).strip(),
            "player_id":   str(row.get("player_id", "")).strip(),
            "team_abbr":   TEAM_NAME_TO_ABB.get(team_name, team_name),
            "status":      status_norm,
            "injury_type": str(row.get("return_date", row.get("reason", ""))).strip(),
        })
    return pd.DataFrame(rows)


def _normalize_pdf_response(raw: pd.DataFrame, date_str: str) -> pd.DataFrame:
    """Normalize a PDF-parsed injury report DataFrame to output schema."""
    rows = []
    for _, row in raw.iterrows():
        status_raw = str(row.get("current_status", "")).strip()
        status_norm = STATUS_MAP.get(status_raw)
        if status_norm is None:
            continue  # skip statuses not in STATUS_MAP (e.g. "Available")
        team_name = str(row.get("team", "")).strip()
        rows.append({
            "date":        date_str,
            "player_name": str(row.get("player_name", "")).strip(),
            "player_id":   "",  # not available from PDF
            "team_abbr":   TEAM_NAME_TO_ABB.get(team_name, team_name),
            "status":      status_norm,
            "injury_type": str(row.get("reason", "")).strip(),
        })
    return pd.DataFrame(rows)


def load_historical_injuries(
    injuries_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load and concatenate all saved injury report CSVs from data/raw/injuries/.

    Returns a DataFrame with columns:
        date, player_name, player_id, team_abbr, status, injury_type

    Each row represents one player's injury status for one day.
    If no files are found, returns an empty DataFrame.

    Args:
        injuries_dir: Directory to search for injury_report_*.csv files.
                      Defaults to data/raw/injuries/.
    """
    search_dir = Path(injuries_dir) if injuries_dir else RAW_INJURIES_DIR
    csv_files = sorted(search_dir.glob("injury_report_*.csv"))

    if not csv_files:
        log.info(f"  [get_injury_data] No injury CSVs found in {search_dir}")
        return pd.DataFrame(
            columns=["date", "player_name", "player_id", "team_abbr",
                     "status", "injury_type"]
        )

    parts = []
    for path in csv_files:
        try:
            part = pd.read_csv(path, dtype=str)
            parts.append(part)
        except Exception as exc:
            log.error(f"  [get_injury_data] Could not read {path}: {exc}")

    if not parts:
        return pd.DataFrame(
            columns=["date", "player_name", "player_id", "team_abbr",
                     "status", "injury_type"]
        )

    combined = pd.concat(parts, ignore_index=True)
    log.info(f"  [get_injury_data] Loaded {len(combined)} rows from "
        f"{len(parts)} injury report CSVs.")
    return combined


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = get_injury_report()
    if df.empty:
        log.info("No injury report data available (may be off-season or no games today).")
    else:
        log.info(f"\nFetched {len(df)} injury report entries:")
        log.info(df.to_string(index=False))
