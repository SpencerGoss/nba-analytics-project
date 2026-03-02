# Phase 3: External Data Layer - Research

**Researched:** 2026-03-02
**Domain:** Web scraping (Basketball Reference), PDF parsing (NBA injury reports), feature integration
**Confidence:** MEDIUM — stack is confirmed, but the exact HTML selector for Basketball Reference officials requires a live spike test (the STATE.md blocker is real and still open)

---

## Summary

Phase 3 adds two external data sources to the pipeline: Basketball Reference referee crew assignments (scraped from box score pages) and the official NBA pre-game injury report (parsed from a publicly available PDF). Both follow the existing `src/data/get_*.py` module pattern.

The referee scraper is the riskier of the two. Basketball Reference wraps most of its secondary tables in HTML comments, and the officials section of a box score page has no confirmed public documentation of its CSS selector. The STATE.md blocker explicitly requires a spike test on 10 real game pages before committing to the full scrape. Rate limits are strict: 20 requests/minute maximum, 3-second crawl-delay specified in robots.txt — and the requirement (NFR-2) explicitly mandates the 3-second delay. This means the referee scrape is IO-bound and slow by design.

The injury report is cleaner. The NBA publishes structured PDFs at a predictable URL pattern (`ak-static.cms.nba.com/referee/injury/Injury-Report_{DATE}_{TIME}.pdf`) with confirmed columns: Game Date, Game Time, Matchup, Team, Player Name, Current Status, Reason. The `pdfplumber` library extracts these tables without Java dependencies (unlike `tabula-py`). Historical data is available back to the 2021-22 season. The `nba_api`'s `LeagueInjuryReport` endpoint also works for the current live report and is already partially implemented in `src/features/injury_proxy.py`'s `get_todays_injury_report()`.

The critical architectural constraint for this phase is FR-4.4: training and inference injury code paths must never share inputs. The training path uses `build_injury_proxy_features()` from game logs. The inference path uses live PDF or `nba_api` endpoint data. These are two separate Python functions that must never call each other.

**Primary recommendation:** Spike the Basketball Reference scraper first (plan 03-01) to confirm the officials section HTML selector before writing the feature engineering code (plan 03-02). Build the injury fetcher (03-03) and enforce code path separation (03-04) in parallel with or after the referee feature work.

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| FR-4.1 | Scrape referee crew assignments from Basketball Reference box score pages | BeautifulSoup4 + lxml parses bref HTML; spike test needed to confirm officials section selector; HTML comment unwrapping technique confirmed as required pattern for secondary tables |
| FR-4.2 | Compute referee foul rate (FTA/game, pace impact) as rolling feature | Rolling window pattern identical to existing ROLL_STATS in team_game_features.py; crew rolling stats join on (game_id, referee_ids); stored in data/raw/external/ then joined at feature assembly time |
| FR-4.3 | Integrate official NBA pre-game injury report for inference | PDF at ak-static.cms.nba.com/referee/injury/; pdfplumber extracts structured table; nba_api LeagueInjuryReport already partially implemented in injury_proxy.py |
| FR-4.4 | Training path: historical proxy from game logs; inference path: live report — never mix | Two separate functions, never calling each other; training path = build_injury_proxy_features(); inference path = get_todays_injury_report() + apply_live_injuries(); framing confirmed in existing code |
| FR-7.2 | External scrapers callable as modules, listed in PIPELINE.md | Module pattern: src/data/external/bref_scraper.py and src/data/external/injury_report.py; PIPELINE.md External Data Scrapers section already has placeholder for these modules |
| NFR-2 | Daily pipeline < 15 min; Basketball Reference 3-second delay between requests | 3-second delay is both required (NFR-2) and enforced by robots.txt; scraper is run on-demand for historical backfill, not in daily update.py pipeline (would violate NFR-2 if added there) |
| NFR-4 | All data sources free/public | Basketball Reference is free/public; NBA injury report PDFs are publicly accessible without auth |
</phase_requirements>

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| beautifulsoup4 | 4.14.3 (latest, Nov 2025) | Parse Basketball Reference HTML | Industry standard for HTML parsing; already listed as required in REQUIREMENTS.md; handles comment-encoded tables via `bs4.Comment` |
| lxml | latest stable (pip install lxml) | Fast HTML parser for BeautifulSoup | Recommended parser by BS4 docs ("Very fast"); C extension, fastest option for Sports Reference page volume |
| requests | >=2.28.0 | HTTP GET for box score pages | Already in requirements.txt; used by all existing get_*.py scrapers |
| pdfplumber | 0.11.x (Jan 2026 latest) | Parse NBA injury report PDFs | No Java dependency (unlike tabula-py); pure Python + pdfminer; extract_tables() handles structured PDFs well |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| nba_api (LeagueInjuryReport) | >=1.4.0 | Live injury report via API endpoint | Fallback or supplement to PDF parsing; already partially implemented in injury_proxy.py; no separate install needed |
| time (stdlib) | - | Rate limit delays between requests | `time.sleep(3)` for Basketball Reference crawl delay |
| io (stdlib) | - | In-memory PDF download (BytesIO) | Avoid writing PDF to disk; download and parse in memory |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| pdfplumber | tabula-py | tabula-py requires Java (JRE/JDK 8+); adds a system dependency that breaks on machines without Java; pdfplumber is pure Python |
| pdfplumber | nbainjuries package | nbainjuries uses tabula-py internally (Java dep); also abstracts away the URL pattern we need to control for historical backfill |
| requests + BS4 | basketball_reference_scraper (vishaalagartha) | Third-party package abstracts HTML structure but adds maintenance risk if package goes stale; the officials section is not implemented in existing packages; custom scraper gives full control |
| requests + BS4 | Selenium | Dynamic rendering not required for static HTML box score pages; Selenium adds complexity and browser dependency |

**Installation:**
```bash
pip install beautifulsoup4 lxml pdfplumber
```

Add to `requirements.txt`:
```
beautifulsoup4>=4.12.0
lxml>=4.9.0
pdfplumber>=0.10.0
```

---

## Architecture Patterns

### Recommended Project Structure
```
src/
├── data/
│   └── external/                  # new subdirectory for external scrapers
│       ├── __init__.py
│       ├── bref_scraper.py        # Basketball Reference referee scraper (FR-4.1)
│       └── injury_report.py       # NBA pre-game injury report fetcher (FR-4.3)
├── features/
│   └── injury_proxy.py            # existing — training path ONLY; inference helpers stay here
data/
├── raw/
│   └── external/                  # new subdirectory for external raw files
│       ├── referee_crew/          # referee_crew_YYYYMMDD.csv per game date batch
│       └── injury_reports/        # injury_report_YYYYMMDD_HHMM.csv per snapshot
```

### Pattern 1: Basketball Reference Scraper (with HTML Comment Unwrapping)

**What:** Basketball Reference wraps secondary tables (everything after the first on a page) in HTML comments. The box score officials section is likely embedded this way. The spike test in 03-01 must confirm the selector, but the unwrapping pattern is confirmed.

**When to use:** Whenever scraping a secondary table on any Basketball Reference page.

**Example:**
```python
# Source: verified technique from multiple community scrapers (MEDIUM confidence)
import requests
from bs4 import BeautifulSoup, Comment
import time

BASE_URL = "https://www.basketball-reference.com/boxscores/{game_id}.html"
CRAWL_DELAY = 3  # required by robots.txt AND NFR-2

def fetch_boxscore_page(game_id: str) -> BeautifulSoup:
    url = BASE_URL.format(game_id=game_id)
    time.sleep(CRAWL_DELAY)  # ALWAYS sleep before each request
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")

def extract_officials(soup: BeautifulSoup) -> list[str]:
    """
    Extract referee names from box score page.

    SPIKE TEST REQUIRED: The selector below is the HYPOTHESIZED pattern.
    Run 03-01 to confirm before using in production.
    """
    # Pattern 1: Officials may be in a direct table element
    # HYPOTHESIS (unconfirmed): table id="officials" or similar
    officials_table = soup.find("table", {"id": "officials"})
    if officials_table:
        return [row.get_text(strip=True)
                for row in officials_table.find_all("td")]

    # Pattern 2: Officials section may be in HTML comments (common BR pattern)
    # Search all comment nodes for one containing "Officials"
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for comment in comments:
        if "Official" in comment:
            comment_soup = BeautifulSoup(comment, "lxml")
            table = comment_soup.find("table")
            if table:
                return [td.get_text(strip=True)
                        for td in table.find_all("td")]

    # Pattern 3: Officials as plain text in div#content near bottom
    # inspect page source manually during spike test if patterns 1+2 fail
    return []
```

### Pattern 2: NBA Injury Report PDF Fetcher

**What:** NBA publishes official injury reports as PDFs at a predictable URL. The PDF has a structured table with columns: Game Date, Game Time, Matchup, Team, Player Name, Current Status, Reason.

**When to use:** For inference path only — never call this from build_injury_proxy_features().

**Example:**
```python
# Source: URL pattern confirmed via live search results (MEDIUM confidence)
# Column structure confirmed via nbainjuries package README (MEDIUM confidence)
import io
import requests
import pdfplumber
import pandas as pd

NBA_INJURY_URL_PATTERN = (
    "https://ak-static.cms.nba.com/referee/injury/"
    "Injury-Report_{date}_{time}.pdf"
)
# date format: YYYY-MM-DD, time format: e.g. 06_00PM

INJURY_COLUMNS = [
    "game_date", "game_time", "matchup", "team",
    "player_name", "current_status", "reason"
]

def fetch_injury_report_pdf(date_str: str, time_str: str) -> pd.DataFrame:
    """
    Fetch and parse the official NBA injury report PDF.
    INFERENCE PATH ONLY — never call from build_injury_proxy_features().
    """
    url = NBA_INJURY_URL_PATTERN.format(date=date_str, time=time_str)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    rows = []
    with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                # skip header row, collect data rows
                rows.extend(table[1:])

    df = pd.DataFrame(rows, columns=INJURY_COLUMNS)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    return df[df["current_status"].isin(["Out", "Questionable", "Probable"])]
```

### Pattern 3: Module Interface (matching get_*.py convention)

**What:** All new external scrapers must be callable as modules per FR-7.2 and the PIPELINE.md module pattern.

**When to use:** Every new scraper in this phase.

**Example:**
```python
# src/data/external/bref_scraper.py — module-level function signature
def get_referee_crew_assignments(
    start_date: str,
    end_date: str,
    output_dir: str = "data/raw/external/referee_crew/",
) -> pd.DataFrame:
    """
    Fetch referee crew assignments for games in [start_date, end_date].
    Saves to data/raw/external/referee_crew/referee_crew_YYYYMMDD.csv
    Respects 3-second delay between requests (NFR-2, robots.txt).

    Usage:
        from src.data.external.bref_scraper import get_referee_crew_assignments
        df = get_referee_crew_assignments("2026-01-01", "2026-02-28")
    """

# src/data/external/injury_report.py — module-level function signature
def get_todays_nba_injury_report(
    output_dir: str = "data/raw/external/injury_reports/",
) -> pd.DataFrame:
    """
    Fetch the current NBA official pre-game injury report.
    INFERENCE PATH ONLY. Never use for training feature construction.
    """
```

### Anti-Patterns to Avoid

- **Mixing injury code paths:** Never import `get_todays_nba_injury_report` from inside `build_injury_proxy_features()`. These are the two code paths FR-4.4 mandates stay separate. Add a module-level docstring comment to each function warning against this.
- **Adding bref_scraper to daily update.py:** The referee scraper takes ~3 seconds/game. A season has ~1,230 games. Historical backfill takes hours. Do not add this to the daily pipeline. It runs on-demand only (violates NFR-2 if added to the 15-minute pipeline).
- **Saving PDFs to disk:** Download and parse in memory via `io.BytesIO`. Disk storage of PDFs is wasteful; save only the parsed CSV rows.
- **Fetching page and sleeping AFTER:** Always `time.sleep(CRAWL_DELAY)` BEFORE the request. This guarantees the delay even if the request fails or the loop has a continue statement.
- **Using `html.parser` instead of `lxml`:** lxml is significantly faster and handles malformed HTML better. Basketball Reference pages are large (~200KB); the difference compounds over thousands of pages.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| PDF table extraction | Custom PDF text parser | pdfplumber's `extract_table()` | PDF coordinate extraction handles page margins, multi-row cells, header rows; hand-rolling fails on font encoding issues |
| HTML parsing | regex-based HTML extraction | BeautifulSoup with lxml | Regex fails on nested tags, comment-encoded tables, malformed HTML — all present on Basketball Reference |
| HTTP retry logic | Custom retry loop | Existing `fetch_with_retry()` in get_game_log.py pattern | Already implemented and tested in the project; copy the pattern, don't rewrite |
| Rate limiting | Custom sleep logic | `time.sleep(3)` before every request | The simple solution is correct here; do not over-engineer |

**Key insight:** Basketball Reference's comment-encoded tables are the primary complexity. BeautifulSoup's `Comment` class is the right tool. Everything else is standard HTTP + pandas.

---

## Common Pitfalls

### Pitfall 1: HTML Comments on Basketball Reference (CONFIRMED BLOCKER)
**What goes wrong:** Tables after the first on a Basketball Reference page are wrapped in HTML comments as an anti-scraping measure. `soup.find("table", id="officials")` returns `None` even when the data exists.
**Why it happens:** Sports Reference wraps secondary content in `<!-- -->` comment blocks. Regular BeautifulSoup traversal skips comments by default.
**How to avoid:** Use `from bs4 import Comment` and iterate: `soup.find_all(string=lambda t: isinstance(t, Comment))`. For each comment containing relevant content, re-parse it: `BeautifulSoup(comment, "lxml")`.
**Warning signs:** `soup.find(...)` returns `None` but you can see the data when viewing page source in browser.

### Pitfall 2: Rate Limit Jail (HIGH RISK)
**What goes wrong:** Exceeding 20 requests/minute on Sports Reference causes a session ban of up to a day (per their policy page).
**Why it happens:** A 3-second delay allows 20 requests/minute exactly. Any additional requests (retries, parallel calls, etc.) push over the limit.
**How to avoid:** Hard-code `CRAWL_DELAY = 3` as a module constant. Never parallelize Basketball Reference requests. Add retry delays of `CRAWL_DELAY * 2` (6 seconds) before retrying a failed request to avoid piling on.
**Warning signs:** HTTP 429 response code from Basketball Reference.

### Pitfall 3: NBA Injury PDF URL Discovery
**What goes wrong:** The PDF URL includes both a date AND a time component (e.g., `Injury-Report_2026-01-01_05_00PM.pdf`). Multiple reports are published per day (6 AM, 11 AM, 1 PM, 5 PM, 6 PM, etc.). There is no API to enumerate available report times.
**Why it happens:** The NBA publishes rolling updates throughout the day; each update gets a new URL.
**How to avoid:** For inference (current day): try a list of known common time patterns in reverse order (6 PM → 5 PM → 1 PM → ...) until one succeeds. For historical use: the `nba_api` `LeagueInjuryReport` endpoint provides a consistent interface without URL guessing.
**Warning signs:** HTTP 403 or 404 when trying to fetch a specific PDF time slot.

### Pitfall 4: Training/Inference Code Path Leakage (FR-4.4 VIOLATION)
**What goes wrong:** A well-meaning developer adds live injury report data to the feature assembly loop "for completeness," causing the model to train on data that was only available at game time — not before.
**Why it happens:** The injury proxy and live report both output similar DataFrames. It's tempting to unify them.
**How to avoid:** Keep them in separate modules. Add a docstring warning to each function. In `injury_report.py`, add an assertion that raises if called with a historical game date more than 2 days old.
**Warning signs:** Feature assembly starts taking longer than expected; `data/features/` files grow unexpectedly.

### Pitfall 5: Missing `__init__.py` in `src/data/external/`
**What goes wrong:** `from src.data.external.bref_scraper import get_referee_crew_assignments` raises `ModuleNotFoundError`.
**Why it happens:** Python requires `__init__.py` in every directory in the import path.
**How to avoid:** Create `src/data/external/__init__.py` (empty file is fine).
**Warning signs:** `ModuleNotFoundError` during import.

### Pitfall 6: Referee Feature Rolling Window — No Historical Data Before Scrape Date
**What goes wrong:** The referee foul-rate rolling feature is NaN for all games before the first scrape date, which may be 2024-25 or later. Training data (2013-14+) will have NaN referee features for most seasons.
**Why it happens:** Basketball Reference data is publicly available going back to 1988-89, but we haven't scraped it yet. The scrape is the bottleneck.
**How to avoid:** The rolling referee feature should default to NaN (not zero) for pre-scrape games. The model's SimpleImputer handles this via mean imputation. Acknowledge in the feature docstring that referee features have sparse historical coverage. Do NOT backfill with zeros — that would inject false signal.
**Warning signs:** After adding referee features, model accuracy drops; check null rates by season.

---

## Code Examples

### HTML Comment Extraction (Confirmed Pattern)
```python
# Source: multiple verified community scrapers (MEDIUM confidence)
# This is the confirmed technique for any secondary table on Basketball Reference
from bs4 import BeautifulSoup, Comment
import requests

def get_soup(url: str) -> BeautifulSoup:
    time.sleep(3)  # ALWAYS before request
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")

def find_table_in_comments(soup: BeautifulSoup, table_id: str):
    """Find a table that may be inside an HTML comment block."""
    # First try direct find (first table on page is not commented)
    table = soup.find("table", id=table_id)
    if table:
        return table
    # Search HTML comment nodes
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment_soup = BeautifulSoup(str(comment), "lxml")
        table = comment_soup.find("table", id=table_id)
        if table:
            return table
    return None
```

### pdfplumber PDF Table Extraction
```python
# Source: pdfplumber PyPI documentation (HIGH confidence)
import io
import pdfplumber
import requests
import pandas as pd

def parse_injury_pdf(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    rows = []
    with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table and len(table) > 1:
                rows.extend(table[1:])  # skip header row per page
    return pd.DataFrame(rows)
```

### nba_api LeagueInjuryReport (Already in injury_proxy.py)
```python
# Source: existing src/features/injury_proxy.py get_todays_injury_report()
# This is already implemented — don't rewrite it; extend it in injury_report.py
from nba_api.stats.endpoints import leagueinjuryreport
import time

def get_live_injury_report() -> pd.DataFrame:
    time.sleep(1)
    report = leagueinjuryreport.LeagueInjuryReport(
        league_id="00",
        season_type="Regular Season",
    ).get_data_frames()[0]
    report.columns = [c.lower() for c in report.columns]
    return report[report["player_status"].isin(["Out", "Questionable"])]
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| tabula-py for PDF extraction | pdfplumber | 2019-2020 | Eliminates Java dependency; pdfplumber is pure Python and handles multi-page tables without JVM setup |
| Selenium for Basketball Reference | requests + BS4 Comment | 2018-2019 | Box score HTML is static (not JS-rendered); Selenium adds unnecessary browser overhead for this use case |
| Widgets.sports-reference.com endpoint | Direct page scraping | Pre-2022 | Old endpoint is defunct; some community scrapers still reference it — ignore those examples |

**Deprecated/outdated:**
- `widgets.sports-reference.com`: The old Sports Reference data widget endpoint no longer works. Any scraping examples referencing this URL are stale.
- `tabula-py` for NBA injury PDFs: Still functional but introduces a Java JRE dependency. `pdfplumber` is the modern replacement.

---

## Open Questions

1. **What is the exact HTML element containing officials on a Basketball Reference box score?**
   - What we know: Basketball Reference wraps secondary tables in HTML comments; the officials data is on the box score page; the selectors are not publicly documented.
   - What's unclear: Is it `table#officials`, plain text in a div, or something else entirely? The spike test in plan 03-01 will determine this.
   - Recommendation: The spike test on 10 real game pages (per STATE.md blocker) is the blocking first task. Do not write the rolling referee feature until the selector is confirmed.

2. **How far back does historical referee data need to go for meaningful rolling features?**
   - What we know: Training data starts 2013-14. Basketball Reference has referee data back to 1988-89. A 20-game rolling window needs ~20 prior games to populate.
   - What's unclear: Is it worth backfilling all seasons, or just 2013-14 forward? Backfilling 10+ seasons at 3 seconds/game means ~3,700 games = ~3 hours.
   - Recommendation: Backfill 2013-14 forward to match training data start. Document the backfill runtime in PIPELINE.md. Add a `backfill_referee_data.py` script for one-time use.

3. **NBA injury PDF URL time slot discovery for current day**
   - What we know: URLs include a time component (05_00PM, 06_00PM, etc.) that varies by day. Multiple snapshots per day.
   - What's unclear: Is there a canonical "latest report" URL or index page with all PDFs listed?
   - Recommendation: Try `nba_api`'s `LeagueInjuryReport` endpoint first (already works, no URL guessing). Use PDF fallback only if the API endpoint fails. The existing `get_todays_injury_report()` in injury_proxy.py already implements the API path.

4. **Where does referee feature integration fit in the feature assembly pipeline?**
   - What we know: team_game_features.py joins several sources; referee data is keyed on game_id; rolling window requires knowing which referee crew worked which game before the current one.
   - What's unclear: Does rolling referee foul rate belong in team_game_features.py or a separate referee_features.py?
   - Recommendation: Separate `src/features/referee_features.py` module, joined into matchup features at the same level as injury_proxy. Keeps team_game_features.py from becoming a monolith.

---

## Validation Architecture

> The config.json does not contain a `nyquist_validation` key. This section is included as the workflow.research is true; validation architecture is relevant given the spike-test requirement.

Phase 3 has no automated test suite in the current project (no `tests/` directory, no pytest config found). Given that FR-4.4 (training/inference path separation) is a correctness invariant that is easy to break silently, the following manual verification steps are the minimum acceptable gates:

### Phase Gates (Manual Verification)
| Req ID | Behavior | Verification |
|--------|----------|--------------|
| FR-4.1 | bref_scraper.py runs for a date range, saves CSVs, respects 3-second delay | Run scraper on 10 game dates, verify CSV output in data/raw/external/referee_crew/, verify runtime ~30s for 10 games |
| FR-4.2 | Referee foul-rate feature present in matchup features | `assert "ref_fta_per_game_roll10" in game_matchup_features.columns` |
| FR-4.3 | injury_report.py fetches and saves structured status | Run on today's date, verify CSV with correct columns |
| FR-4.4 | Training/inference paths never share inputs | Grep: `build_injury_proxy_features` must not call `get_todays_nba_injury_report` and vice versa |
| FR-7.2 | Modules callable via import | `python -c "from src.data.external.bref_scraper import get_referee_crew_assignments; print('ok')"` |

---

## Sources

### Primary (HIGH confidence)
- pdfplumber PyPI page — confirmed current version (0.11.x, Jan 2026), extract_table() API
- Beautiful Soup 4 official documentation — find_all(string=lambda t: isinstance(t, Comment)) pattern, lxml parser recommendation
- Sports Reference rate limit policy page (sports-reference.com/bot-traffic.html) — confirmed 20 req/min limit, up-to-day ban on violation
- Basketball Reference robots.txt — confirmed 3-second crawl-delay

### Secondary (MEDIUM confidence)
- NBA injury report PDF URL pattern (`ak-static.cms.nba.com/referee/injury/`) — confirmed via multiple live PDF links in search results; column names confirmed via nbainjuries README
- nba_api LeagueInjuryReport endpoint — confirmed working, already in src/features/injury_proxy.py
- Basketball Reference HTML comment table pattern — confirmed by multiple community scrapers and Medium articles; `from bs4 import Comment` technique is consistent across sources
- Basketball Reference referee register pages — verified live at /referees/ and /referees/2025_register.html (FTA, PF, FGA per game stats per referee available at season level)

### Tertiary (LOW confidence)
- Exact HTML selector for officials section on box score pages — NOT confirmed; only hypothesis that it follows the comment-table pattern; REQUIRES spike test
- Whether bref scraper for officials data is in direct HTML or comment — unconfirmed without running a test request

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — beautifulsoup4, lxml, pdfplumber, requests all confirmed with current versions; installation commands verified
- Architecture: MEDIUM — module pattern confirmed from existing codebase; directory structure is a reasonable extension; integration point into matchup features is logical but needs confirmation during implementation
- Pitfalls: HIGH — rate limit policy is official documentation; HTML comment pattern is cross-verified by multiple sources; training/inference path separation is specified in existing code; PDF URL time-slot issue is observable from live search results
- Officials HTML selector: LOW — the single remaining unknown; blocked on spike test (this is the STATE.md blocker)

**Research date:** 2026-03-02
**Valid until:** 2026-04-02 (30 days) — Basketball Reference HTML structure changes infrequently but does happen; verify selectors if more than a month passes before implementation
