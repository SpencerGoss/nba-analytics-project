"""
Sportsbook odds refresh for the daily pipeline.

Called by update.py as part of the daily update sequence. Delegates to
scripts/fetch_odds.py which handles the full API call, data formatting,
and model-vs-odds comparison file assembly.

Returns True on success, False if the refresh was skipped (e.g., no API
key configured) or failed — so update.py can log and continue rather than
aborting the entire pipeline.
"""

import logging
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

_FETCH_SCRIPT = Path(__file__).resolve().parent.parent.parent / "scripts" / "fetch_odds.py"


def refresh_odds_data() -> bool:
    """
    Run the daily sportsbook odds refresh.

    Calls scripts/fetch_odds.py as a subprocess so that a missing API key
    or network error never raises an exception into update.py.

    Returns:
        True  — odds files were written successfully
        False — refresh was skipped or failed (not a fatal error)
    """
    if not _FETCH_SCRIPT.exists():
        log.warning("scripts/fetch_odds.py not found — odds refresh skipped.")
        return False

    result = subprocess.run(
        [sys.executable, str(_FETCH_SCRIPT)],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        log.info("Odds refresh completed successfully.")
        # Echo the script's output so it appears in update.py logs
        if result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                log.info(f"  [odds] {line}")
        return True
    else:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        log.warning(
            f"Odds refresh failed (exit code {result.returncode}). "
            f"Details: {(stderr or stdout)[:300]}"
        )
        return False
