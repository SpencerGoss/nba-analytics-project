# External Integrations

**Analysis Date:** 2025-03-01

## APIs & External Services

**NBA Stats API:**
- Service: stats.nba.com (official NBA statistics endpoint)
- What it's used for: Fetch all historical and current season player/team statistics, game logs, standings, and tracking data
- SDK/Client: `nba_api>=1.4.0` (Python wrapper)
- Authentication: Custom User-Agent headers mimicking web browser requests
  - User-Agent, Referer, Accept headers configured in data modules
  - Rate limiting: 1-2 second delays between requests to avoid throttling
- Retry mechanism: 3 attempts with 10-second backoff for network failures
- Modules: `src/data/get_*.py` (20+ data fetching scripts)
- Examples:
  - `src/data/get_player_stats.py` - Player stats via leaguedashplayerstats endpoint
  - `src/data/get_game_log.py` - Team/player game logs via leaguegamelog endpoint
  - `src/data/get_standings.py` - Season standings via leaguestandingsv3 endpoint

**Pinnacle Guest API:**
- Service: https://guest.api.arcadia.pinnacle.com/0.1
- What it's used for: Daily refresh of sportsbook moneylines, spreads, and player prop odds for comparison with model predictions
- Authentication: No authentication required (free, keyless guest endpoint)
- NBA league ID: 487
- Integration module: `scripts/fetch_odds.py`
- Invoked by: `src/data/get_odds.py` (subprocess wrapper, non-fatal on network error)
- Output: Generates `data/odds/game_lines.csv`, `data/odds/player_props.csv`, `data/odds/model_vs_odds.csv`
- Odds format: american
- Team mapping: Full team names → 3-letter NBA abbreviations (30 teams, mapping in fetch_odds.py lines 70-100)
- Feature flags: Configurable thresholds for flagging significant model-vs-odds gaps (PROP_FLAG_GAP=1.5 units, WINPROB_FLAG_PP=0.05)

## Data Storage

**Databases:**
- Not used. Project uses file-based CSV storage only
  - Memory/design note: Database integration mentioned in earlier README versions but not implemented
  - All data in `data/raw/` (seasonal CSV files) and `data/processed/` (consolidated CSVs)
  - Feature tables cached in `data/features/`
  - Model artifacts in `models/artifacts/` (pickle files)

**File Storage:**
- Local filesystem only
- No cloud storage integrations

**Caching:**
- CSV files in `data/processed/` serve as processed data cache
- Feature tables in `data/features/` cache derived features
- Model pickle files in `models/artifacts/` cache trained models

## Authentication & Identity

**Auth Provider:**
- None. Project uses no API keys for odds data (NBA Stats requires no auth; Pinnacle guest API requires no auth)

**API Key Management:**
- Pinnacle guest API requires no key
- BALLDONTLIE_API_KEY stored in `.env` file at project root (if used)
- Environment file is `.gitignore`d (never committed)

## Monitoring & Observability

**Error Tracking:**
- Custom error logging to `logs/pipeline_errors.log`
- Timestamp + script name + error message format
- Logged by: `update.py`, `backfill.py` error handlers

**Logs:**
- `logs/pipeline_errors.log` - Pipeline failures
- `logs/update.log` - Daily update script output (via Windows Task Scheduler)
- Console output during manual script execution

**Status Monitoring:**
- No external monitoring service
- Return codes: Exit code 1 on failure, 0 on success
- Odds refresh is graceful: non-fatal failure, pipeline continues

## CI/CD & Deployment

**Hosting:**
- Local machine (Windows 11 Pro) or compatible Linux/Mac environment
- Data files remain local (no cloud sync)

**CI Pipeline:**
- None configured. Manual execution or Windows Task Scheduler

**Scheduling:**
- Windows Task Scheduler: Daily at 7:00 AM
- Batch file: `scripts/run_update.bat` (wrapper for `python update.py`)
- Fallback: Manual execution of `python update.py`

## Environment Configuration

**Required env vars:**
- Pinnacle - No API key required for daily odds refresh

**Optional env vars:**
- None others configured currently

**Secrets location:**
- `.env` file (project root)
- Never committed to git
- No live odds API key needed (Pinnacle guest API is keyless)

## Data Source Details

**NBA Stats API Coverage:**
- Historical data range varies by endpoint:
  - Player/team regular season stats: 1996-97+
  - Clutch/scoring stats: 2000-01+ (tracking actually starts 2004-05)
  - Hustle/tracking: 2015-16+
  - Game logs: 1946-47+ (sparse early coverage)
  - Standings: 1979-80+ (1990-91 missing — API returned empty)
  - Player/team master (biographical): All-time

**Season Code Format:**
- Integer format: e.g., 202425 (not "2024-25")
- Auto-detected current season in update.py based on month:
  - October or later → current year
  - Before October → previous year

**Update Frequency:**
- Daily via `update.py` - refreshes current season only
- Historical backfill via `backfill.py` - covers 1946-2025 (one-time or occasional)

## Webhooks & Callbacks

**Incoming:**
- None. Pipeline is unidirectional (fetch only)

**Outgoing:**
- None. No external webhooks triggered by this pipeline

## Data Flow & Integration Points

**Fetch → Process → Train → Predict:**

1. **Fetch Phase** (`src/data/*.py`)
   - Calls nba_api endpoints
   - Saves raw CSVs to `data/raw/[dataset]/`
   - Triggered by: `update.py` (current season) or `backfill.py` (historical)

2. **Preprocess Phase** (`src/processing/preprocessing.py`)
   - Reads raw seasonal CSVs
   - Combines, cleans, renames columns
   - Outputs consolidated CSVs to `data/processed/`
   - Called by: `update.py` after fetching

3. **Feature Engineering** (`src/features/*.py`)
   - Reads processed CSVs and model training data
   - Builds derived features (game matchups, player stats, injury proxies, era labels)
   - Outputs to `data/features/`
   - Called by: `src/models/train_all_models.py`

4. **Model Training** (`src/models/*.py`)
   - Reads features and processed data
   - Trains three prediction models: game_outcome, player_performance, playoff_odds
   - Saves to `models/artifacts/` (pickle + metadata)
   - Run: `python src/models/train_all_models.py`

5. **Odds Refresh** (`scripts/fetch_odds.py`)
   - Called as subprocess by `src/data/get_odds.py`
   - Fetches Pinnacle guest API (https://guest.api.arcadia.pinnacle.com/0.1, NBA league 487) → `data/odds/game_lines.csv`
   - Reads model predictions from `models/artifacts/`
   - Produces `data/odds/model_vs_odds.csv` for comparison
   - Non-fatal; pipeline continues on network error

---

*Integration audit: 2025-03-01*
