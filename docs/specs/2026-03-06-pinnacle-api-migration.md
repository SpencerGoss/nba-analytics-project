# Spec: Pinnacle API Migration (Replace The Odds API)
Date: 2026-03-06
Status: COMPLETE

## Goal
Replace all references to The Odds API (expired key, 500 req/month quota) with the
Pinnacle guest API (free, keyless, no quota, sharper lines) across the entire project.

## Non-Goals
- Player props from Pinnacle (endpoint structure differs; stub returns empty DataFrame)
- Historical odds backfill via Pinnacle (still uses Kaggle CSV)
- Any changes to model training, feature engineering, or prediction pipeline
- Dashboard UI changes

## Pinnacle Guest API (verified working 2026-03-06)
- Base URL: `https://guest.api.arcadia.pinnacle.com/0.1`
- Auth: None required
- NBA league ID: 487, sport ID: 4
- Endpoint 1: `GET /leagues/487/matchups` — upcoming games + participants
- Endpoint 2: `GET /leagues/487/markets/straight` — moneyline + spread prices
- Rate limit: unknown; add 1 req/sec sleep to be safe
- Response structure:
  - Matchup parent: `{id, startTime, participants: [{alignment: "home"|"away", name: "Sacramento Kings"}]}`
  - Market: `{matchupId, type: "moneyline"|"spread", period: 0, prices: [{designation: "home"|"away", price: -248, points: 4.5}]}`
  - Filter: only matchups where participants have alignment=home/away (not neutral = futures)

## Files to Change

### 1. `scripts/fetch_odds.py` — Full API rewrite
- Remove: `API_KEY`, `ODDS_API_KEY` env check, `BASE_URL`/`SPORT`/`REGION`/`ODDS_FMT`/`DATE_FMT`
- Add: `PINNACLE_BASE`, `LEAGUE_ID = 487`
- Rename: `get_odds_api()` -> `get_pinnacle()` (no auth header, just requests.get)
- Rewrite: `fetch_game_lines()` — use `/leagues/487/matchups` + `/leagues/487/markets/straight`
- Stub: `fetch_player_props()` — return empty DataFrame (no-op, Pinnacle props differ)
- Keep unchanged: `load_model_game_projections()`, `load_model_player_projections()`,
  `build_model_vs_odds()`, `main()`

### 2. `src/data/get_odds.py` — Remove ODDS_API_KEY log references
- Remove: ODDS_API_KEY string references in log.warning messages
- Update: log messages to say "Pinnacle API" instead of "ODDS_API_KEY"

### 3. `src/models/value_bet_detector.py` — Remove quota guard + key check
- Remove: `check_remaining_quota()` function entirely (Pinnacle has no quota)
- Remove: `QuotaError` exception class
- Update: `run_value_bet_scan()` — remove ODDS_API_KEY check and quota check block
- Update: docstrings that mention "The Odds API" or "ODDS_API_KEY"
- Keep: `use_live_odds` parameter and historical fallback logic (unchanged)

### 4. `update.py` — Remove ODDS_API_KEY from env check
- Remove: `ODDS_API_KEY` from the `missing` env var list (~line 167-168)

### 5. `.env` — Remove expired key
- Remove: `ODDS_API_KEY=9644650bfc066e63611e1eea878c8130` line

### 6. `.env.example` — Update docs
- Remove: `ODDS_API_KEY=your_odds_api_key_here` line and its comment

### 7. `src/models/predict_cli.py` — Update help text
- Update: `--live` argument help string to say "Pinnacle API" instead of "The Odds API (requires ODDS_API_KEY)"

## Acceptance Criteria
- [ ] `python scripts/fetch_odds.py` runs without error and writes non-empty `game_lines.csv`
- [ ] `game_lines.csv` has columns: date, home_team, away_team, home_moneyline, away_moneyline, spread
- [ ] Team names are correctly mapped to 3-letter abbreviations
- [ ] No ODDS_API_KEY reference remains in any Python file
- [ ] No sys.exit(1) on startup (no API key guard)
- [ ] `value_bet_detector.py` import succeeds; `check_remaining_quota` removed
- [ ] `update.py` does not list ODDS_API_KEY as a required env var
- [ ] `.env` and `.env.example` have no ODDS_API_KEY
- [ ] `python -m pytest tests/ -q` passes (145 tests, same baseline)

## Implementation Tasks
- [x] Task 1: Rewrite `scripts/fetch_odds.py` (Pinnacle API client + fetch_game_lines)
- [x] Task 2: Update `src/data/get_odds.py` (remove ODDS_API_KEY log strings)
- [x] Task 3: Update `src/models/value_bet_detector.py` (remove QuotaError, check_remaining_quota, ODDS_API_KEY refs)
- [x] Task 4: Update `update.py` + `.env` + `.env.example` + `predict_cli.py`
- [x] Task 5: Verify live run + run full test suite (145 passing, 7 game lines fetched live)

## Open Questions
- None. All resolved via live API testing on 2026-03-06.
