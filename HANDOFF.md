# Handoff — NBA Analytics Project

_Last updated: 2026-03-05_

## What Was Built

**Investigated pipeline state + decided on odds source migration**

- Confirmed `predictions_history.db` is healthy — 9 predictions written for Mar 5 games by `update.py` Step 6 (no action needed)
- Investigated `database/nba.db` (0 bytes): confirmed legacy artifact from Feb 2026 early dev; no current code uses it; pipeline is entirely CSV-based. Corrected stale docs in ARCHITECTURE.md and `.claude/rules/nba-domain.md`.
- Ran `scripts/fetch_odds.py`: ODDS_API_KEY returns 401 (expired/invalid). Model pipeline inside works correctly.
- Researched free/legal odds alternatives; **decided to replace The Odds API with Pinnacle API** — free, keyless, no quota, sharper lines (better for value-bet model).

## Current State

- Raw data: fresh (Mar 5 2026)
- `team_game_features.csv`: 136,452 rows x 118 cols (fresh)
- `game_matchup_features.csv`: 291 cols (fresh)
- Calibrated model: `game_outcome_model_calibrated.pkl` — operational
- ATS model: `ats_model.pkl` — operational
- Tests: 145 passing, 0 failing
- `predictions_history.db`: 9 rows (healthy)
- `data/odds/`: game_lines.csv and model_vs_odds.csv are empty (fetch_odds.py blocked by expired key)

## What's Next (immediate priority)

### Pinnacle API Migration

Replace all Odds API references with Pinnacle across the entire project:

1. **Audit all files** referencing `ODDS_API_KEY`, `the-odds-api.com`, or The Odds API:
   - `scripts/fetch_odds.py` — main script to rewrite
   - `.env` and `.env.example` — remove `ODDS_API_KEY`
   - `docs/odds_integration_notes.md` (if exists)
   - Any other references

2. **Rewrite `scripts/fetch_odds.py`** to use Pinnacle API:
   - Base URL: `https://api.pinnacle.com/v1/`
   - No API key required for odds reads
   - Map Pinnacle sport IDs (NBA = 4) and team IDs to project abbreviations
   - Fetch game lines (moneyline + spread) and replace current fetch_game_lines()
   - Keep same output format: `data/odds/game_lines.csv`, `model_vs_odds.csv`

3. **Remove `ODDS_API_KEY`** from `.env`, `.env.example`, and the startup check in `fetch_odds.py`

4. **Test** that `fetch_odds.py` runs without error and produces populated CSVs

5. **After v2.0 odds migration** → v3.0 web dashboard polish

## Failed Approaches

- The Odds API free tier: key expired; replacement key would work but quota is 500 req/month (limiting)
- Action Network: unofficial endpoints, risk of breaking without notice
- Chose Pinnacle: free, keyless, widely used in betting analytics community, sharper lines

## Key Decisions

- `database/nba.db` — leave as empty legacy artifact; do not populate; pipeline is CSV-based
- Pinnacle API over The Odds API: no key, no quota, better signal for value-bet detection
- `format="mixed"` is the standard for all `pd.to_datetime()` calls on game_date columns
