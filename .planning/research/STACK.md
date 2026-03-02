# Technology Stack — Additional Libraries for ATS & Data Enrichment

**Project:** NBA Game Outcome Prediction & ATS Betting Analytics
**Researched:** 2026-03-01
**Scope:** Additive only — what to ADD to the existing stack for the active milestone
**Existing stack (not re-researched):** Python 3.14, pandas, numpy, scikit-learn, shap, matplotlib, nba_api, requests, python-dotenv

---

## What This Research Covers

The existing pipeline is functional. This research covers ONLY the additional
libraries and data sources needed for:

1. External data scraping (Basketball Reference, NBA injury reports)
2. ATS/spread modeling (new prediction target alongside win probability)
3. Feature engineering improvements (pace, rest, travel, lineup)
4. Web-ready model output serialization (JSON, not just pickle)

---

## Recommended Additions

### 1. Web Scraping — Basketball Reference

**Use: `beautifulsoup4` + `lxml` (direct HTML scraping)**

**Confidence: HIGH** — These are the standard, mature scraping tools for sports-reference.com.

| Library | Pin Version | Purpose | Why |
|---------|-------------|---------|-----|
| beautifulsoup4 | `>=4.12.0` | Parse Basketball Reference HTML tables | Standard parser, stable API, handles sports-reference table structure cleanly |
| lxml | `>=4.9.0` | Fast HTML/XML parser backend for BeautifulSoup | Significantly faster than html.parser; sports-reference pages are large and lxml handles malformed HTML well |

**Why not `sportsipy` / `sportsreference` packages:**
The `sportsipy` library wraps sports-reference.com but as of 2024-25 it is minimally maintained, has broken endpoints due to sports-reference HTML changes, and adds an abstraction layer that makes debugging harder when scraped selectors break. Direct `requests` + `beautifulsoup4` scraping is more reliable for a project that already uses `requests`. The existing `requests` package covers the HTTP layer.

**Why not `selenium` or `playwright`:**
Basketball Reference renders tables as static HTML — no JavaScript execution required. Selenium/Playwright add browser overhead with no benefit. Reserve these for truly JS-rendered pages.

**Sports-reference rate limiting:** Basketball Reference enforces a soft rate limit (roughly 20 requests/minute). Use `time.sleep(3)` between page fetches. The existing `fetch_with_retry()` pattern in the project is directly applicable.

**What to scrape from Basketball Reference:**
- Advanced team stats per game (pace, eFG%, TOV%, ORB%, FT/FGA) — available back to 1946-47
- Referee crew assignments per game — available in game box score pages
- Rest days / back-to-back schedule — derivable from game schedule pages
- Four Factors data — directly relevant to ATS model feature engineering

---

### 2. NBA Official Injury Reports

**Use: `nba_api` (already installed) — `LeagueInjuryReport` endpoint**

**Confidence: MEDIUM** — The `LeagueInjuryReport` endpoint exists in nba_api as of v1.4.x. The existing `injury_proxy.py` file already calls it in `get_todays_injury_report()`. The issue is that this endpoint is *current-only* (today's report), not historical.

**For historical injury data:** No free, clean, structured source exists. Options:

| Approach | Confidence | Notes |
|----------|------------|-------|
| nba_api `LeagueInjuryReport` | HIGH | Current day only; already partially wired in codebase |
| Basketball Reference game logs (played/DNP) | HIGH | "Did Not Play" entries in box scores are reliable historical proxy; already implemented via `injury_proxy.py` logic |
| NBA.com injury report PDF archive | LOW | PDFs exist at nba.com but parsing is fragile; not recommended |
| RotoWire / ESPN historical injuries | LOW | No free API; scraping ToS is ambiguous |

**Recommendation:** The `injury_proxy.py` approach (infer from game logs) IS the right historical solution. The fix is debugging why the features arrive null in the matchup table, not finding a new data source. For live/upcoming games, the existing `get_todays_injury_report()` function is the right tool — it just needs to be wired into `predict_cli.py`.

**No new library needed here.** The bug is in feature pipeline wiring, not the data source.

---

### 3. ATS Model — New Prediction Target

**Use: existing scikit-learn (already installed) — no new ML library needed**

**Confidence: HIGH**

The ATS target (did home team cover the spread?) is a binary classification problem identical in structure to win probability. The existing sklearn Pipeline, GradientBoostingClassifier, and walk-forward backtesting code handles this without new libraries.

**What IS needed for ATS modeling:**

| Addition | Library | Why |
|----------|---------|-----|
| Spread/line as input feature | The Odds API (already integrated) | Vegas closing line is the single strongest ATS predictor; using it as a feature (not just comparison) unlocks value-bet identification |
| Kelly Criterion bet sizing | Pure Python math, no library | Kelly formula is a two-line calculation; no library needed |
| ROI / Sharpe tracking | pandas (already installed) | Backtest ATS performance metrics are standard pandas aggregations |

**No new ML library is justified.** XGBoost or LightGBM are commonly cited for sports betting models, but:
- The existing GradientBoostingClassifier at 65%+ accuracy is already competitive
- LightGBM/XGBoost would add a new dependency and training paradigm for marginal gains on a dataset of ~10,000 games (2014-present)
- If accuracy stalls after feature improvements, revisit XGBoost then — not before

---

### 4. Feature Engineering — Pace, Rest, Travel

**Use: `geopy` for travel distance calculations**

**Confidence: MEDIUM** — geopy is the standard Python library for geodesic distance calculations between city coordinates.

| Library | Pin Version | Purpose | Why |
|---------|-------------|---------|-----|
| geopy | `>=2.4.0` | Calculate travel distance between NBA arenas | Cross-country back-to-backs (e.g., Miami → Portland) have measurable fatigue effects; distance is a better signal than back-to-back binary flag alone |

**Arena coordinates:** NBA arena lat/lon is static data — hardcode a dict of 30 arenas (arena_name → lat/lon). This avoids any geocoding API calls at runtime.

**Why not a travel API or geocoding service:** Arena locations don't change (rarely). Static dict is simpler, faster, and has no API dependency or rate limits.

**Rest and schedule features** (back-to-back flag, days_rest, games_last_7_days): These are derivable from the existing `team_game_logs` data using pandas date arithmetic. The existing `team_game_features.py` already has `games_last_5_days` and `games_last_7_days`. Travel distance is the only signal genuinely requiring a new library.

**Pace and Three-Point Era features:** Derivable from existing data — team_game_logs has `fg3a`, `fga`, possession estimates from `team_stats_advanced`. No new library needed; these are pandas calculations.

---

### 5. JSON Serialization — Web-Ready Outputs

**Use: Python standard library `json` module (no new library)**

**Confidence: HIGH**

The existing pickle artifacts need a JSON-serializable companion for each prediction run. The standard `json` module handles this. For numpy/pandas types that don't serialize by default, a small custom encoder is all that's needed:

```python
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
```

**No external library needed.** Libraries like `orjson` offer faster serialization but the output volume here is small (one JSON file per prediction run). Adding a dependency to solve a non-problem is unjustified.

**What JSON outputs to produce:**
- `predictions/game_predictions_YYYYMMDD.json` — per-game win probability + ATS prediction + value bet flag
- `predictions/model_metadata.json` — model version, training seasons, feature list, accuracy metrics
- `predictions/player_projections_YYYYMMDD.json` — pts/reb/ast projections for tonight

These feed a future web layer without requiring that layer to exist now.

---

### 6. Sportsbook Lines as Model Features

**Use: The Odds API (already integrated) — extend existing `fetch_odds.py`**

**Confidence: HIGH**

The current integration fetches odds for comparison only (output to `model_vs_odds.csv`). To use Vegas lines as model input features, the spread and moneyline need to be stored historically and joined to the feature table at training time.

**What's needed:** A historical odds cache — store each day's fetched lines to `data/odds/historical_lines.csv`. This means the existing `fetch_odds.py` needs an append-to-history mode alongside the daily-refresh mode.

**No new library.** This is a data management change to the existing fetch_odds.py script.

**Implied line conversion:** Moneyline → implied probability conversion is a 3-line formula:
```python
def american_to_implied_prob(american_odds: int) -> float:
    if american_odds > 0:
        return 100 / (american_odds + 100)
    return abs(american_odds) / (abs(american_odds) + 100)
```
No library needed.

---

## Full Additions List

| Library | Version | Purpose | Confidence | Install |
|---------|---------|---------|------------|---------|
| beautifulsoup4 | `>=4.12.0` | Scrape Basketball Reference HTML tables | HIGH | `pip install beautifulsoup4` |
| lxml | `>=4.9.0` | Fast HTML parser backend for BeautifulSoup | HIGH | `pip install lxml` |
| geopy | `>=2.4.0` | Arena-to-arena travel distance calculations | MEDIUM | `pip install geopy` |

**Total new libraries: 3**

Everything else (ATS model, JSON serialization, injury features, pace features, rest features, odds-as-features) is achievable with the existing stack via code changes only.

---

## Alternatives Considered and Rejected

| Category | Rejected Option | Why Rejected |
|----------|----------------|--------------|
| Scraping | `sportsipy` / `sportsreference` package | Minimally maintained as of 2024-25; breaks with sports-reference HTML changes; direct BeautifulSoup more reliable |
| Scraping | `selenium` / `playwright` | Basketball Reference is static HTML; browser overhead unjustified |
| ML | `xgboost` / `lightgbm` | Dataset is ~10k games; existing GBM at 65%+ is already competitive; revisit only if accuracy stalls post feature-fix |
| ML | `pytorch` / `tensorflow` | Neural nets are not justified for tabular sports data at this scale; interpretability is harder; training is slower locally |
| Serialization | `orjson` | Faster than stdlib json but prediction volume is tiny; dependency not justified |
| Injury data | RotoWire / ESPN scraping | ToS ambiguous; no free API; game log proxy (existing approach) is the right historical solution |
| Travel | Geocoding API (Google, Nominatim) | Arena locations are static; hardcoded dict is simpler and has no API rate limits |

---

## Installation

```bash
# Activate virtual environment first
pip install beautifulsoup4>=4.12.0 lxml>=4.9.0 geopy>=2.4.0
```

Add to `requirements.txt`:
```
# External data scraping
beautifulsoup4>=4.12.0
lxml>=4.9.0

# Travel distance calculations
geopy>=2.4.0
```

---

## Confidence Assessment

| Area | Confidence | Rationale |
|------|------------|-----------|
| BeautifulSoup4 + lxml for Basketball Reference | HIGH | Standard, mature tools; no alternatives are better for static HTML scraping |
| nba_api LeagueInjuryReport for current games | MEDIUM | Endpoint exists in v1.4.x per existing codebase; historical limitation is inherent to the API |
| sklearn for ATS model (no new ML library) | HIGH | ATS is binary classification; existing pipeline is appropriate |
| geopy for travel distance | MEDIUM | Library is correct tool; arena coordinate dict approach is sound but not verified against current geopy 2.4.x API |
| JSON stdlib for serialization | HIGH | Standard library; no uncertainty |
| Odds-as-features via existing The Odds API | HIGH | Data already flows in; historical cache is a pipeline extension, not a new integration |

---

## What NOT to Add (and Why)

**Do not add a time-series ML library (Prophet, statsmodels, sktime):** NBA game prediction is not a forecasting problem — it's a classification problem with temporal ordering. Walk-forward backtesting already handles the temporal structure correctly.

**Do not add a database ORM (SQLAlchemy, tortoise-orm):** The project writes to SQLite directly via pandas `to_sql()`. Adding an ORM for a local SQLite file is over-engineering.

**Do not add a data validation library (Great Expectations, pandera):** Valuable long-term but out of scope for this milestone. The bug in injury feature nulls is a logic error, not a validation gap.

**Do not add an async HTTP library (aiohttp, httpx):** The NBA API rate limit (1 req/sec) means parallelizing requests would immediately trigger throttling. Sequential requests with delays is the correct approach.

---

## Sources

**Confidence note:** Web search tools were unavailable during this research session. All findings are based on:
- Direct inspection of the existing codebase (HIGH confidence for project-specific claims)
- Training knowledge through August 2025 (MEDIUM confidence for library ecosystem claims)
- Official documentation known as of training cutoff (MEDIUM confidence for version pins)

**Library version pins should be verified before implementing:**
```bash
pip index versions beautifulsoup4
pip index versions lxml
pip index versions geopy
```

**Known limitation:** sportsipy/sportsreference maintenance status was assessed from training knowledge (last known state: declining maintenance as of early 2025). Verify current status at https://pypi.org/project/sportsipy/ before definitively ruling it out.
