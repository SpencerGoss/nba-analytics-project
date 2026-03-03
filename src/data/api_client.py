"""
Shared NBA API client utilities.

Centralizes the retry logic, HTTP headers, and rate-limit configuration
that was previously duplicated across every get_*.py data fetcher script.

Usage:
    from src.data.api_client import fetch_with_retry, HEADERS, configure_logging

    result = fetch_with_retry(
        lambda: some_endpoint(headers=HEADERS, timeout=60).get_data_frames()[0],
        label="2024-25",
    )
    if result["success"]:
        df = result["data"]
    else:
        print(result["error"])
"""

import logging
import time

import pandas as pd

logger = logging.getLogger(__name__)

# ── HTTP headers for stats.nba.com ────────────────────────────────────────────

HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nba.com/",
    "Connection": "keep-alive",
    "Origin": "https://www.nba.com",
}

# ── Default rate-limit configuration ─────────────────────────────────────────

DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 10  # seconds between retry attempts


# ── Logging helper ────────────────────────────────────────────────────────────

def configure_logging(level=logging.INFO):
    """
    Set up basic logging for data-fetcher scripts.

    Call once at the top of a standalone script's ``if __name__ == "__main__"``
    block so that log messages from fetch_with_retry appear on the console.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ── Core retry function ──────────────────────────────────────────────────────

def fetch_with_retry(fetch_fn, label, max_retries=DEFAULT_MAX_RETRIES,
                     retry_delay=DEFAULT_RETRY_DELAY):
    """
    Execute *fetch_fn* with automatic retries on failure.

    Parameters
    ----------
    fetch_fn : callable
        Zero-argument callable that returns a ``pd.DataFrame`` on success.
    label : str
        Human-readable label for log messages (e.g. a season string).
    max_retries : int
        Maximum number of attempts before giving up.
    retry_delay : int | float
        Seconds to sleep between retry attempts.

    Returns
    -------
    dict
        ``{"success": True,  "data": pd.DataFrame, "error": None}`` on success.
        ``{"success": False, "data": None, "error": str}`` after all retries fail.
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            data = fetch_fn()
            return {"success": True, "data": data, "error": None}
        except Exception as exc:
            last_error = str(exc)
            if attempt < max_retries - 1:
                logger.warning(
                    "Attempt %d failed for %s: %s. Retrying in %ds...",
                    attempt + 1, label, exc, retry_delay,
                )
                time.sleep(retry_delay)
            else:
                logger.error(
                    "All %d retries failed for %s: %s. Skipping.",
                    max_retries, label, exc,
                )

    return {"success": False, "data": None, "error": last_error}
