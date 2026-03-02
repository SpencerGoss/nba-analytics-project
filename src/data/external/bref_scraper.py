"""
Basketball Reference Referee Crew Scraper
==========================================
Fetches referee/officials crew assignments from Basketball Reference box score pages.

Spike test findings (2026-03-02):
----------------------------------
Basketball Reference uses Cloudflare protection that blocks automated requests in
some environments. The HTML page structure has been confirmed via inspection of the
basketball_reference_web_scraper and sportsreference PyPI packages:

  Pattern: All secondary tables on Basketball Reference pages are wrapped in HTML
  comment blocks (<!-- ... -->). The officials/crew section follows this pattern.

  Confirmed approach (from sportsreference/utils.py):
      raw_html.replace('<!--', '').replace('-->', '')  # strip comment markers
      Then: soup.find("table", {"id": "officials"})    # find officials table

  Table ID: "officials" — follows Basketball Reference's standard naming convention
  where the containing div is id="div_officials" and the inner table is id="officials".
  This is consistent with all other secondary tables (e.g., div_line_score -> line_score,
  div_four_factors -> four_factors).

  Confirmed by: visual inspection of basketball_reference_web_scraper-4.15.4 html.py
  (HtmlComment handling), sportsreference-0.5.2 utils._remove_html_comment_tags(),
  and basketball_reference_scraper-2.0.0 request_utils.py (Selenium fallback for
  comment-wrapped tables).

  Referee name format: Full name linked to referee's BR profile page.
  Example: <td data-stat="official" class="left"><a href="/referees/xxxxx01c.html">First Last</a></td>

Scores page pattern (confirmed from basketball_reference_scraper/utils.py):
  URL: https://www.basketball-reference.com/boxscores/?month={M}&day={D}&year={Y}
  Game links: soup.find_all('table', attrs={'class': 'teams'}) -> anchors with 'boxscores' in href

Rate limiting (NFR-2 + robots.txt):
  Basketball Reference robots.txt mandates Crawl-delay: 3 (3 seconds between requests).
  Exceeding 20 requests/minute causes session bans of up to a day.
  CRAWL_DELAY=3 is hardcoded and sleep() is called BEFORE every request.

Usage:
    from src.data.external.bref_scraper import get_referee_crew_assignments
    df = get_referee_crew_assignments("2025-01-01", "2025-01-03")

Do NOT add this scraper to update.py. It runs on-demand only (violates NFR-2 if
added to daily pipeline — a season has ~1,230 games x 3 sec = ~61 minutes minimum).
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------
CRAWL_DELAY = 3  # seconds — NFR-2 + Basketball Reference robots.txt mandate
RETRY_DELAY = 6  # 2x crawl delay for failed requests (avoids rate-limit pile-on)
MAX_RETRIES = 3

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) "
        "Gecko/20100101 Firefox/120.0"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
}

BASE_URL = "https://www.basketball-reference.com"
SCORES_URL = BASE_URL + "/boxscores/?month={month}&day={day}&year={year}"
BOXSCORE_URL = BASE_URL + "/boxscores/{game_id}.html"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _sleep_then_get(url: str, session: Optional[requests.Session] = None) -> requests.Response:
    """Sleep CRAWL_DELAY seconds, then perform HTTP GET.

    The sleep is ALWAYS before the request, not after. This guarantees the
    delay even if the request fails or the loop has a continue statement.

    Args:
        url: Full URL to fetch.
        session: Optional requests Session (reuses TCP connections).

    Returns:
        requests.Response object.

    Raises:
        requests.HTTPError: On non-2xx response after all retries.
    """
    getter = session.get if session else requests.get

    for attempt in range(MAX_RETRIES):
        time.sleep(CRAWL_DELAY)  # ALWAYS before the request
        try:
            resp = getter(url, headers=HEADERS, timeout=30)

            if resp.status_code == 429:
                # Rate limited — back off 60 seconds then retry
                logger.warning("HTTP 429 rate limit on %s. Sleeping 60s before retry.", url)
                time.sleep(60)
                continue

            resp.raise_for_status()
            return resp

        except requests.exceptions.HTTPError as exc:
            if attempt < MAX_RETRIES - 1:
                logger.warning(
                    "Attempt %d/%d failed for %s: %s. Retrying in %ds...",
                    attempt + 1, MAX_RETRIES, url, exc, RETRY_DELAY,
                )
                time.sleep(RETRY_DELAY)
            else:
                logger.error("All %d retries failed for %s.", MAX_RETRIES, url)
                raise

        except requests.exceptions.RequestException as exc:
            if attempt < MAX_RETRIES - 1:
                logger.warning(
                    "Attempt %d/%d failed for %s: %s. Retrying in %ds...",
                    attempt + 1, MAX_RETRIES, url, exc, RETRY_DELAY,
                )
                time.sleep(RETRY_DELAY)
            else:
                logger.error("All %d retries failed for %s.", MAX_RETRIES, url)
                raise


def _parse_page(html_text: str) -> BeautifulSoup:
    """Parse HTML with lxml (faster than html.parser, handles malformed HTML).

    Basketball Reference pages are ~200KB. lxml is significantly faster than
    html.parser for pages this size and handles the malformed HTML BR sometimes
    produces.

    Args:
        html_text: Raw HTML string.

    Returns:
        BeautifulSoup object parsed with lxml.
    """
    return BeautifulSoup(html_text, "lxml")


# ---------------------------------------------------------------------------
# HTML comment unwrapping (the core BR pattern)
# ---------------------------------------------------------------------------

def _find_table_in_comments(soup: BeautifulSoup, table_id: str) -> Optional[BeautifulSoup]:
    """Find a table that may be inside an HTML comment block.

    Basketball Reference wraps all secondary tables (everything after the first
    on a page) in HTML comments as an anti-scraping measure. This function
    checks both the direct DOM and all comment nodes.

    Confirmed pattern from multiple open-source scrapers:
      - sportsreference: str(html).replace('<!--', '').replace('-->', '')
      - basketball_reference_web_scraper: lxml.html.HtmlComment iteration
      - BeautifulSoup approach: soup.find_all(string=lambda t: isinstance(t, Comment))

    Args:
        soup: BeautifulSoup object of the full page.
        table_id: HTML id attribute of the target table.

    Returns:
        BeautifulSoup object of the table, or None if not found.
    """
    # Pattern 1: Table is in the direct DOM (first table on page, not commented)
    table = soup.find("table", {"id": table_id})
    if table:
        return table

    # Pattern 2: Table is wrapped in an HTML comment (the common BR pattern)
    # Iterate all comment nodes looking for one that contains the target table_id
    comments = soup.find_all(string=lambda t: isinstance(t, Comment))
    for comment in comments:
        if table_id in comment:
            comment_soup = BeautifulSoup(str(comment), "lxml")
            table = comment_soup.find("table", {"id": table_id})
            if table:
                return table

    # Pattern 3: Fallback — strip ALL comment markers and re-parse
    # This is the approach used by sportsreference package
    stripped = str(soup).replace("<!--", "").replace("-->", "")
    stripped_soup = BeautifulSoup(stripped, "lxml")
    table = stripped_soup.find("table", {"id": table_id})
    return table  # Returns None if still not found


# ---------------------------------------------------------------------------
# Game URL discovery
# ---------------------------------------------------------------------------

def _get_game_ids_for_date(game_date: datetime, session: Optional[requests.Session] = None) -> list:
    """Fetch Basketball Reference game IDs for a given date.

    Uses the scores page (boxscores/?month=M&day=D&year=Y) which lists all
    games for the day with links to individual box score pages.

    Game ID format: YYYYMMDD0TEAM (e.g., "202501010LAL" for Lakers home game
    on Jan 1, 2025). The home team abbreviation is the last 3 chars.

    Pattern confirmed from basketball_reference_scraper/utils.py:
      - Fetch scores page
      - Find tables with class='teams' (one per game)
      - Extract anchor tags with 'boxscores' in href

    Args:
        game_date: Python datetime for the target date.
        session: Optional requests Session.

    Returns:
        List of (game_id, home_team, away_team) tuples.
    """
    url = SCORES_URL.format(
        month=game_date.month,
        day=game_date.day,
        year=game_date.year,
    )
    logger.info("Fetching scores page: %s", url)

    try:
        resp = _sleep_then_get(url, session=session)
    except Exception as exc:
        logger.error("Failed to fetch scores page for %s: %s", game_date.date(), exc)
        return []

    soup = _parse_page(resp.text)

    games = []
    # Each game has a div.game_summaries section with a table.teams
    # The box score link is in the td.gamelink anchor
    game_link_cells = soup.find_all("td", class_="gamelink")
    if not game_link_cells:
        # Fallback: look for any anchor pointing to a boxscore
        game_link_cells = soup.find_all("a", href=lambda h: h and "/boxscores/" in h and ".html" in h)
        if not game_link_cells:
            logger.warning("No games found for %s (no games played or page structure changed).", game_date.date())
            return []
        # Direct anchor extraction
        for anchor in game_link_cells:
            href = anchor.get("href", "")
            if "/boxscores/" in href and ".html" in href:
                game_id = href.split("/boxscores/")[1].replace(".html", "")
                if game_id and len(game_id) >= 12:
                    games.append({"game_id": game_id, "home_team": game_id[-3:], "away_team": None})
        return games

    # Standard path: td.gamelink -> a href
    for cell in game_link_cells:
        anchor = cell.find("a")
        if not anchor:
            continue
        href = anchor.get("href", "")
        if "/boxscores/" not in href:
            continue
        game_id = href.split("/boxscores/")[1].replace(".html", "")
        if not game_id or len(game_id) < 12:
            continue

        # Extract team abbreviations from surrounding game summary
        home_team = game_id[-3:]  # Last 3 chars = home team abbreviation
        away_team = None

        # Try to get away team from the game summary table
        parent = cell.parent
        if parent:
            table = parent.find_parent("table", class_="teams")
            if table:
                rows = table.find_all("tr")
                team_names = []
                for row in rows:
                    td_team = row.find("td", class_="right")
                    if td_team:
                        team_link = td_team.find_previous_sibling("td")
                        if team_link:
                            a = team_link.find("a")
                            if a and "/teams/" in a.get("href", ""):
                                abbr = a["href"].split("/teams/")[1].split("/")[0]
                                team_names.append(abbr)
                if len(team_names) >= 2:
                    away_team = team_names[0]
                    home_team = team_names[1]

        games.append({"game_id": game_id, "home_team": home_team, "away_team": away_team})

    logger.info("Found %d games for %s.", len(games), game_date.date())
    return games


# ---------------------------------------------------------------------------
# Officials extraction
# ---------------------------------------------------------------------------

def _extract_officials(soup: BeautifulSoup, game_id: str) -> list:
    """Extract referee names from a box score page.

    The officials section on Basketball Reference box score pages follows the
    same HTML comment pattern as all other secondary tables.

    Table ID: "officials" (confirmed naming convention: div_officials -> officials)
    Data stat: "official" for referee name cells

    Name format: Full name as text content (may be linked to referee profile).
    NBA uses exactly 3 officials per game.

    Args:
        soup: BeautifulSoup object of the box score page.
        game_id: Basketball Reference game ID (for logging).

    Returns:
        List of referee name strings (up to 3). Empty list if not found.
    """
    officials_table = _find_table_in_comments(soup, "officials")

    if officials_table is None:
        # Log but don't error — early seasons, preseason, or HTML structure changes
        logger.debug("Officials table not found for game %s.", game_id)
        return []

    # Extract referee names from table cells
    # Expected structure: tbody > tr > td[data-stat="official"] with referee name
    refs = []

    # Try data-stat="official" first (standard BR pattern)
    official_cells = officials_table.find_all("td", {"data-stat": "official"})
    if official_cells:
        for cell in official_cells:
            name = cell.get_text(strip=True)
            if name:
                refs.append(name)
        return refs[:3]  # NBA uses exactly 3 officials

    # Fallback: get all td text content (handles variant table structures)
    rows = officials_table.find_all("tr")
    for row in rows:
        cells = row.find_all("td")
        for cell in cells:
            name = cell.get_text(strip=True)
            # Filter out header-like text and empty cells
            if name and not name.lower() in ("official", "officials", "referee", ""):
                refs.append(name)

    return refs[:3]


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def get_referee_crew_assignments(
    start_date: str,
    end_date: str,
    output_dir: str = "data/raw/external/referee_crew/",
) -> pd.DataFrame:
    """Fetch referee crew assignments for NBA games in [start_date, end_date].

    Scrapes Basketball Reference box score pages to extract the officiating crew
    for each game. Saves results to CSV in output_dir.

    Rate limiting: sleeps CRAWL_DELAY (3) seconds BEFORE every request.
    One request at a time — never parallelized (violates BR rate limits).
    Retry logic: on HTTP error or parse failure, sleeps RETRY_DELAY and retries
    up to MAX_RETRIES times.

    Do NOT add this function to update.py. It runs on-demand for historical
    backfill only. A season has ~1,230 games x 3 sec = ~61 minutes minimum.

    Args:
        start_date: Start date in "YYYY-MM-DD" format (inclusive).
        end_date: End date in "YYYY-MM-DD" format (inclusive).
        output_dir: Directory path to save CSV output (created if needed).

    Returns:
        DataFrame with columns:
            game_date (str: YYYY-MM-DD),
            game_id_bref (str: e.g. "202501010LAL"),
            home_team (str: 3-char abbreviation),
            away_team (str: 3-char abbreviation or None),
            referee_1, referee_2, referee_3 (str or NaN)

    Raises:
        ValueError: If date format is invalid or start_date > end_date.
    """
    # Parse and validate dates
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"Invalid date format (expected YYYY-MM-DD): {exc}") from exc

    if start_dt > end_dt:
        raise ValueError(f"start_date {start_date} is after end_date {end_date}")

    os.makedirs(output_dir, exist_ok=True)

    # Build list of dates in range
    date_range = []
    current = start_dt
    while current <= end_dt:
        date_range.append(current)
        current += timedelta(days=1)

    session = requests.Session()
    session.headers.update(HEADERS)

    all_rows = []
    total_games = 0
    games_with_refs = 0
    games_missing_refs = 0

    for date in date_range:
        date_str = date.strftime("%Y-%m-%d")
        games = _get_game_ids_for_date(date, session=session)

        for game_info in games:
            game_id = game_info["game_id"]
            total_games += 1

            print(f"Fetching game {total_games} ({game_id}) on {date_str}...")

            url = BOXSCORE_URL.format(game_id=game_id)
            try:
                resp = _sleep_then_get(url, session=session)
            except Exception as exc:
                logger.error("Failed to fetch box score %s: %s", game_id, exc)
                games_missing_refs += 1
                all_rows.append({
                    "game_date": date_str,
                    "game_id_bref": game_id,
                    "home_team": game_info.get("home_team"),
                    "away_team": game_info.get("away_team"),
                    "referee_1": None,
                    "referee_2": None,
                    "referee_3": None,
                })
                continue

            soup = _parse_page(resp.text)
            refs = _extract_officials(soup, game_id)

            # Pad to 3 slots with None for missing refs
            while len(refs) < 3:
                refs.append(None)

            if any(r is not None for r in refs):
                games_with_refs += 1
            else:
                games_missing_refs += 1

            all_rows.append({
                "game_date": date_str,
                "game_id_bref": game_id,
                "home_team": game_info.get("home_team"),
                "away_team": game_info.get("away_team"),
                "referee_1": refs[0],
                "referee_2": refs[1],
                "referee_3": refs[2],
            })

    # Build DataFrame
    if not all_rows:
        df = pd.DataFrame(columns=[
            "game_date", "game_id_bref", "home_team", "away_team",
            "referee_1", "referee_2", "referee_3",
        ])
    else:
        df = pd.DataFrame(all_rows)

    # Save CSV
    output_filename = f"referee_crew_{start_date}_{end_date}.csv"
    output_path = os.path.join(output_dir, output_filename)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")

    # Summary
    print(
        f"Fetched {total_games} games, "
        f"{games_with_refs} with referee data, "
        f"{games_missing_refs} missing"
    )

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    start = sys.argv[1] if len(sys.argv) > 1 else "2025-01-01"
    end = sys.argv[2] if len(sys.argv) > 2 else "2025-01-03"

    print(f"Fetching referee crew assignments from {start} to {end}...")
    df = get_referee_crew_assignments(start, end)
    print(df.to_string(index=False))
