# Handoff — NBA Analytics Project

_Last updated: 2026-03-06_

## What Was Built

**Pinnacle API Migration — COMPLETE**

Replaced the expired The Odds API (401 errors, 500 req/month quota) with the Pinnacle guest API (free, keyless, no quota, sharper lines) across the entire project.

### Files changed
- `scripts/fetch_odds.py` — full rewrite: new `get_pinnacle()` client, `fetch_game_lines()` via Pinnacle, `fetch_player_props()` stubbed
- `src/models/value_bet_detector.py` — removed `QuotaError`, `check_remaining_quota()`, ODDS_API_KEY guard
- `src/data/get_odds.py` — removed ODDS_API_KEY log branch
- `update.py` — removed ODDS_API_KEY from env check
- `.env` / `.env.example` — ODDS_API_KEY deleted
- `src/models/predict_cli.py` — updated --live help text
- Docs: `ARCHITECTURE.md`, `CONTEXT.md`, `PIPELINE.md`, `nba-domain.md`, `INTEGRATIONS.md`
- Spec: `docs/specs/2026-03-06-pinnacle-api-migration.md`

## Current State

- Raw data: fresh (Mar 5 2026)
- Calibrated model: operational
- ATS model: operational
- Tests: **145 passing**, 0 failing
- `predictions_history.db`: 9 rows (healthy)
- `data/odds/game_lines.csv`: 7 rows (live Pinnacle data, Mar 6 2026)
- `data/odds/model_vs_odds.csv`: 7 rows (model vs Pinnacle lines)
- Branch: `main` — committed as `feat(odds): replace The Odds API with Pinnacle guest API`

## Pinnacle API Details (for reference)
- Base URL: `https://guest.api.arcadia.pinnacle.com/0.1`
- No auth required
- NBA league ID: 487
- Matchups: `GET /leagues/487/matchups` — filter to parentId=None, participants with alignment=home/away
- Markets: `GET /leagues/487/markets/straight` — type=moneyline|spread, period=0, prices with designation=home|away
- Team names: same full names as The Odds API (ODDS_TEAM_TO_ABB mapping reused)

## What's Next

### v3.0 — Web Dashboard Polish
The odds pipeline is now fully operational. Next milestone is polishing the web dashboard:
1. Wire `game_lines.csv` and `model_vs_odds.csv` into the dashboard display
2. Show today's Pinnacle lines alongside model win probabilities
3. Surface value bets (flagged rows from model_vs_odds.csv)
4. Dashboard UI design/polish using `frontend-design` skill

### Known Stubs / Future Work
- `fetch_player_props()` is a no-op stub — Pinnacle player props use a different endpoint
  structure; implement when needed for player prop value bet detection
- `database/nba.db` — leave as empty legacy artifact; pipeline is CSV-based

## Key Decisions
- Pinnacle over The Odds API: free, keyless, no quota, sharper lines (better signal)
- Player props stubbed rather than partially implemented — cleaner than broken code
- `format="mixed"` is the standard for all `pd.to_datetime()` calls on game_date columns
