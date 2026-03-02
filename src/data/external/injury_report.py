"""
NBA Pre-Game Injury Report Fetcher
====================================
Fetches the official NBA pre-game injury report for live inference.

INFERENCE PATH ONLY -- never import or call this module from
build_injury_proxy_features() or any training code path.
The training path uses historical game-log-based proxy from
src/features/injury_proxy.py. These two code paths must NEVER share inputs
(FR-4.4).

Usage:
    from src.data.external.injury_report import get_todays_nba_injury_report
    df = get_todays_nba_injury_report()
"""

import io
import os
import sys
import time
import warnings
from datetime import datetime, timedelta

import pandas as pd
import requests

# Allow running as a script from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# -- Code path boundary (FR-4.4) -----------------------------------------------
# This module is INFERENCE PATH ONLY.
# NEVER import src.features.injury_proxy.build_injury_proxy_features here.
# NEVER use historical game log data for injury features in this module.
# The training path is src/features/injury_proxy.py.
_CODE_PATH = "INFERENCE"

# ── Constants ─────────────────────────────────────────────────────────────────

OUTPUT_DIR = "data/raw/external/injury_reports/"

# PDF URL pattern — try in reverse order (most recent to earliest) per Pitfall 3
# in 03-RESEARCH.md: multiple reports published per day; no index to enumerate them.
NBA_INJURY_URL_PATTERN = (
    "https://ak-static.cms.nba.com/referee/injury/"
    "Injury-Report_{date}_{time}.pdf"
)

# Common time slots in reverse order (try most recent first)
PDF_TIME_SLOTS = ["06_00PM", "05_00PM", "01_30PM", "11_00AM", "06_00AM"]

# Expected columns in the injury report PDF
INJURY_COLUMNS = [
    "game_date", "game_time", "matchup", "team",
    "player_name", "current_status", "reason",
]

# Statuses that indicate a player's availability is in question
RELEVANT_STATUSES = ["Out", "Questionable", "Probable"]


# ── Date guard (FR-4.4 enforcement) ───────────────────────────────────────────

def _assert_recent_date(date_str: str) -> None:
    """
    Guard against using this fetcher for historical training data (FR-4.4).

    Raises ValueError if the given date is more than 2 days in the past.
    This makes it structurally impossible to accidentally use this inference-path
    module to backfill training features.
    """
    target = datetime.strptime(date_str, "%Y-%m-%d")
    if (datetime.now() - target).days > 2:
        raise ValueError(
            f"injury_report.py is for live inference only (FR-4.4). "
            f"Date {date_str} is more than 2 days ago. "
            f"Use build_injury_proxy_features() for historical training data."
        )


# ── Primary source: nba_api LeagueInjuryReport ────────────────────────────────

def _fetch_via_nba_api() -> pd.DataFrame:
    """
    Fetch today's injury report via nba_api LeagueInjuryReport endpoint.

    This is the primary source — no URL guessing required (per Open Question 3
    in 03-RESEARCH.md and the recommendation from the plan).

    Returns:
        DataFrame with normalized lowercase columns, filtered to relevant statuses.
        Returns empty DataFrame if the endpoint fails.
    """
    try:
        from nba_api.stats.endpoints import leagueinjuryreport

        time.sleep(1)  # polite to the NBA API
        report = leagueinjuryreport.LeagueInjuryReport(
            league_id="00",
            season_type="Regular Season",
        ).get_data_frames()[0]

        if report.empty:
            print("  [injury_report] nba_api returned empty report (off-season or no games).")
            return pd.DataFrame()

        # Normalize column names to lowercase
        report.columns = [c.lower() for c in report.columns]

        # Filter to actionable statuses
        filtered = report[report["player_status"].isin(RELEVANT_STATUSES)].copy()
        print(f"  [injury_report] nba_api: fetched {len(filtered)} entries "
              f"(Out/Questionable/Probable).")
        return filtered

    except Exception as exc:
        print(f"  [injury_report] nba_api fetch failed: {exc}")
        return pd.DataFrame()


# ── Fallback: PDF parsing via pdfplumber ──────────────────────────────────────

def _fetch_via_pdf(date_str: str) -> pd.DataFrame:
    """
    Fetch injury report from NBA's published PDF (fallback when API fails).

    Tries common time slots in reverse order (most recent first) until one
    succeeds. Downloads and parses in memory via io.BytesIO — PDF is never
    written to disk (anti-pattern per 03-RESEARCH.md).

    Args:
        date_str: Date in YYYY-MM-DD format.

    Returns:
        DataFrame with normalized columns, filtered to relevant statuses.
        Returns empty DataFrame if all time slots return 4xx.
    """
    try:
        import pdfplumber
    except ImportError:
        print("  [injury_report] pdfplumber not installed; PDF fallback unavailable.")
        print("  Install with: pip install pdfplumber>=0.10.0")
        return pd.DataFrame()

    for time_slot in PDF_TIME_SLOTS:
        url = NBA_INJURY_URL_PATTERN.format(date=date_str, time=time_slot)
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()

            # Parse PDF in memory — never write to disk
            rows = []
            with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
                for page in pdf.pages:
                    table = page.extract_table()
                    if table and len(table) > 1:
                        # Skip header row on each page
                        rows.extend(table[1:])

            if not rows:
                print(f"  [injury_report] PDF at {time_slot} had no extractable rows.")
                continue

            df = pd.DataFrame(rows, columns=INJURY_COLUMNS)
            # Normalize column names (already lowercase, but ensure no spaces)
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]

            filtered = df[df["current_status"].isin(RELEVANT_STATUSES)].copy()
            # Rename current_status -> player_status for API consistency
            filtered = filtered.rename(columns={"current_status": "player_status"})

            print(f"  [injury_report] PDF ({time_slot}): fetched {len(filtered)} entries.")
            return filtered

        except requests.exceptions.HTTPError as http_err:
            status_code = http_err.response.status_code if http_err.response else "unknown"
            if status_code in (403, 404):
                # Expected when this time slot hasn't been published yet
                continue
            print(f"  [injury_report] PDF HTTP error at {time_slot}: {http_err}")
            continue
        except Exception as exc:
            print(f"  [injury_report] PDF parse error at {time_slot}: {exc}")
            continue

    print(f"  [injury_report] All PDF time slots failed for {date_str}. "
          f"May be off-season or report not yet published.")
    return pd.DataFrame()


# ── Public interface ───────────────────────────────────────────────────────────

def get_todays_nba_injury_report(
    output_dir: str = OUTPUT_DIR,
    save_snapshot: bool = True,
) -> pd.DataFrame:
    """
    Fetch current NBA official pre-game injury report.

    INFERENCE PATH ONLY. Never use for training feature construction.
    See build_injury_proxy_features() in src/features/injury_proxy.py for
    the historical proxy used in training (FR-4.4).

    Strategy:
    1. Try nba_api LeagueInjuryReport endpoint (primary -- no URL guessing)
    2. If API fails, try PDF fallback with common time slots in reverse order
    3. Return structured DataFrame or empty DataFrame on total failure

    Args:
        output_dir:    Directory to save CSV snapshot (created if missing).
        save_snapshot: Whether to save a dated CSV snapshot for archival.

    Returns:
        DataFrame with columns: player_name, team, player_status, reason, game_date
        Filtered to Out, Questionable, Probable statuses.
        Returns empty DataFrame if both sources fail.
    """
    today_str = datetime.now().strftime("%Y-%m-%d")
    print(f"[injury_report] Fetching NBA injury report for {today_str}...")

    # Try primary source: nba_api endpoint
    df = _fetch_via_nba_api()

    # Fallback: PDF parsing
    if df.empty:
        print("[injury_report] Falling back to PDF source...")
        df = _fetch_via_pdf(today_str)

    if df.empty:
        warnings.warn(
            "[injury_report] Could not fetch injury report from any source. "
            "Returning empty DataFrame.",
            RuntimeWarning,
            stacklevel=2,
        )
        return pd.DataFrame()

    # Add game_date column if not already present (nba_api may include it)
    if "game_date" not in df.columns:
        df["game_date"] = today_str

    # Save snapshot for archival
    if save_snapshot:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        snapshot_name = f"injury_report_{timestamp}.csv"
        os.makedirs(output_dir, exist_ok=True)
        snapshot_path = os.path.join(output_dir, snapshot_name)
        df.to_csv(snapshot_path, index=False)
        print(f"[injury_report] Snapshot saved -> {snapshot_path}")

    return df


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = get_todays_nba_injury_report()
    if df.empty:
        print("No injury report data available (may be off-season or no games today)")
    else:
        print(f"Fetched {len(df)} injury report entries:")
        print(df.to_string(index=False))
