"""
BallDontLie API client for NBA game results and player data.

BallDontLie v1 (https://www.balldontlie.io/api/v1/) was free with no API key
but has been sunset as of 2024. The current API (v1 at api.balldontlie.io)
requires an API key from https://www.balldontlie.io.

Set your API key in the environment:
    export BALLDONTLIE_API_KEY=your_key_here

Or add to .env:
    BALLDONTLIE_API_KEY=your_key_here

Available endpoints (v2 with API key):
    GET /teams           -- team directory
    GET /players         -- player directory
    GET /games           -- game results by season
    GET /stats           -- player game stats

NOTE: /injuries endpoint does NOT exist in BallDontLie v1 or v2.
Injury data is fetched from the NBA official PDF reports via get_injury_data.py.

Usage:
    from src.data.get_balldontlie import get_balldontlie_injuries, get_balldontlie_stats
    injuries_df = get_balldontlie_injuries()  # returns empty DF (not available)
    stats_df = get_balldontlie_stats(season=2024)
"""

import logging
import os
import time

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

BALLDONTLIE_BASE_URL = "https://api.balldontlie.io/v1"
BALLDONTLIE_API_KEY_ENV = "BALLDONTLIE_API_KEY"

# Rate limiting: free tier allows 60 requests/minute
REQUEST_DELAY_SECONDS = 1.1


def _get_api_key() -> str | None:
    """Return API key from environment, or None if not set."""
    # Try dotenv if available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    return os.environ.get(BALLDONTLIE_API_KEY_ENV)


def _build_headers(api_key: str) -> dict:
    """Build request headers including Authorization."""
    return {
        "Authorization": api_key,
        "Accept": "application/json",
    }


# ── Pagination helper ─────────────────────────────────────────────────────────

def _fetch_all_pages(
    endpoint: str,
    params: dict,
    headers: dict,
    max_pages: int = 500,
) -> list[dict]:
    """
    Fetch all pages from a paginated BallDontLie endpoint.

    Parameters
    ----------
    endpoint : str
        Full URL for the endpoint.
    params : dict
        Query parameters (per_page, filters, etc.).
    headers : dict
        Request headers including Authorization.
    max_pages : int
        Safety limit on pages to fetch (prevents infinite loops).

    Returns
    -------
    list[dict]
        All records from all pages, or empty list on failure.
    """
    all_records = []
    cursor = None
    page_count = 0

    while page_count < max_pages:
        page_params = dict(params)
        if cursor is not None:
            page_params["cursor"] = cursor

        try:
            time.sleep(REQUEST_DELAY_SECONDS)
            resp = requests.get(endpoint, params=page_params, headers=headers, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response else "unknown"
            if status == 401:
                logger.error(
                    "BallDontLie API: unauthorized. "
                    "Set %s environment variable with your API key from "
                    "https://www.balldontlie.io",
                    BALLDONTLIE_API_KEY_ENV,
                )
            else:
                logger.error("BallDontLie API HTTP error: %s", exc)
            return []
        except Exception as exc:
            logger.error("BallDontLie API request failed: %s", exc)
            return []

        records = payload.get("data", [])
        all_records.extend(records)
        page_count += 1

        # Cursor-based pagination: next_cursor signals more pages
        meta = payload.get("meta", {})
        next_cursor = meta.get("next_cursor")
        if next_cursor is None:
            break
        cursor = next_cursor

    return all_records


# ── Public interface ──────────────────────────────────────────────────────────

def get_balldontlie_injuries() -> pd.DataFrame:
    """
    Attempt to fetch injury data from BallDontLie API.

    NOTE: BallDontLie does NOT provide an /injuries endpoint in any API version.
    Injury data is available from the NBA official PDF injury reports.
    See src/data/get_injury_data.py for the injury report fetcher.

    Returns
    -------
    pd.DataFrame
        Always returns an empty DataFrame with the expected schema.
        Use get_injury_data.get_injury_report() for real injury data.
    """
    logger.warning(
        "BallDontLie does not provide an injuries endpoint. "
        "Use src/data/get_injury_data.get_injury_report() instead."
    )
    return pd.DataFrame(
        columns=["player_name", "team_abbr", "status", "description"]
    )


def get_balldontlie_stats(season: int) -> pd.DataFrame:
    """
    Fetch game results from BallDontLie API for a given season.

    Fetches all games for the specified season year. A BallDontLie API key
    is required — set BALLDONTLIE_API_KEY in your environment or .env file.

    Parameters
    ----------
    season : int
        Season start year (e.g. 2024 for the 2024-25 season).

    Returns
    -------
    pd.DataFrame
        Game records with columns:
            game_id, date, season, home_team_id, home_team_abbr,
            visitor_team_id, visitor_team_abbr, home_team_score,
            visitor_team_score, status, period, time

        Returns empty DataFrame if API key is missing or request fails.
    """
    api_key = _get_api_key()
    if not api_key:
        logger.error(
            "BallDontLie API key not found. "
            "Set %s in your environment or .env file. "
            "Get a free key at https://www.balldontlie.io",
            BALLDONTLIE_API_KEY_ENV,
        )
        return pd.DataFrame(
            columns=[
                "game_id", "date", "season", "home_team_id", "home_team_abbr",
                "visitor_team_id", "visitor_team_abbr", "home_team_score",
                "visitor_team_score", "status", "period", "time",
            ]
        )

    headers = _build_headers(api_key)
    endpoint = f"{BALLDONTLIE_BASE_URL}/games"
    params = {
        "seasons[]": season,
        "per_page": 100,
    }

    logger.info("Fetching BallDontLie games for season %d...", season)
    records = _fetch_all_pages(endpoint, params, headers)

    if not records:
        logger.warning("No games returned from BallDontLie for season %d", season)
        return pd.DataFrame()

    rows = []
    for game in records:
        home_team = game.get("home_team", {})
        visitor_team = game.get("visitor_team", {})
        rows.append({
            "game_id":            game.get("id"),
            "date":               game.get("date"),
            "season":             game.get("season"),
            "home_team_id":       home_team.get("id"),
            "home_team_abbr":     home_team.get("abbreviation"),
            "visitor_team_id":    visitor_team.get("id"),
            "visitor_team_abbr":  visitor_team.get("abbreviation"),
            "home_team_score":    game.get("home_team_score"),
            "visitor_team_score": game.get("visitor_team_score"),
            "status":             game.get("status"),
            "period":             game.get("period"),
            "time":               game.get("time"),
        })

    df = pd.DataFrame(rows)
    logger.info(
        "BallDontLie: fetched %d games for season %d", len(df), season
    )
    return df


def get_balldontlie_teams() -> pd.DataFrame:
    """
    Fetch the full NBA team directory from BallDontLie API.

    Parameters
    ----------
    None

    Returns
    -------
    pd.DataFrame
        Team records with columns: team_id, abbreviation, city, conference,
        division, full_name, name. Returns empty DataFrame if API key is missing.
    """
    api_key = _get_api_key()
    if not api_key:
        logger.error(
            "BallDontLie API key not found. Set %s in your environment.",
            BALLDONTLIE_API_KEY_ENV,
        )
        return pd.DataFrame()

    headers = _build_headers(api_key)
    endpoint = f"{BALLDONTLIE_BASE_URL}/teams"
    params = {"per_page": 100}

    logger.info("Fetching BallDontLie team directory...")
    records = _fetch_all_pages(endpoint, params, headers)

    if not records:
        return pd.DataFrame()

    rows = [
        {
            "team_id":      t.get("id"),
            "abbreviation": t.get("abbreviation"),
            "city":         t.get("city"),
            "conference":   t.get("conference"),
            "division":     t.get("division"),
            "full_name":    t.get("full_name"),
            "name":         t.get("name"),
        }
        for t in records
    ]

    df = pd.DataFrame(rows)
    logger.info("BallDontLie: fetched %d teams", len(df))
    return df


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    print("Testing BallDontLie API client...")
    print()

    # Check API key
    key = _get_api_key()
    if key:
        print(f"API key found: ...{key[-4:]}")
        teams_df = get_balldontlie_teams()
        if not teams_df.empty:
            print(f"Teams: {len(teams_df)} records")
            print(teams_df.head(3).to_string(index=False))
        print()
        stats_df = get_balldontlie_stats(season=2024)
        if not stats_df.empty:
            print(f"Games (2024-25): {len(stats_df)} records")
            print(stats_df.head(3).to_string(index=False))
    else:
        print(
            f"No API key found ({BALLDONTLIE_API_KEY_ENV} not set).\n"
            "Get a free key at https://www.balldontlie.io and add to .env:\n"
            f"  {BALLDONTLIE_API_KEY_ENV}=your_key_here"
        )

    print()
    injuries_df = get_balldontlie_injuries()
    print(f"Injuries endpoint: {len(injuries_df)} rows (not available — expected)")
