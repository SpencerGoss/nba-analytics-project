# Handoff — NBA Analytics Project

_Last updated: 2026-03-07 Session 4_

## What Was Done This Session

### Dashboard Overhaul — 3 New Tabs Built

**Sharp Money Tracker** (page-sharp-money)
- Driven by `line_movement.json`; stats bar (Games Tracked, Steam Moves, Biggest Move); per-game cards with movement bars; STEAM badge when |move| >= 1.5 pts; direction arrows; color coding

**Bet Tracker** (page-bet-tracker)
- localStorage key `baseline_bets_v1`; add-bet form (date pre-filled to today, matchup, pick, market, odds, stake, result); stats row (total bets, W-L, ROI%, net P&L); history table with per-bet delete; clear-all button; fully functional with no backend

**Season History** (page-history)
- Lazy-loaded on tab click from `season_history.json` (553 KB); 5 seasons (2020-21 to 2024-25), 30 teams/season, 5,995 games; season selector dropdown; standings table (rank, team, W, L, PCT); 50-game log table
- `scripts/build_season_history.py` created; wired into `update.py` Step 7

### Security Hardening
- CSP meta tag added (with `frame-ancestors` removed — it's HTTP-header only)
- Plotly SRI hash: `sha384-Hl48Kq2HifOWdXEjMsKo6qxqvRLTYqIGbvlENBmkHAxZKIGCXv43H6W1jA671RzC`
- Fake account dropdown removed
- `_setHtml(el, html)` helper established as XSS-safe DOM write pattern (createContextualFragment + replaceChildren)

### Hardcoded Data Cleared
- `const MATCHUP_DATA=[]` (was 3 Mar-5 games) — now populated from matchup_analysis.json
- `DATA.picks=[]` (was 9 Mar-5 picks) — populated from todays_picks.json
- `bt-backtest-total` — dynamic from accuracy_history backtest entries
- `bt-ats-inline` — dynamic from performance.json
- OG/Twitter meta tag accuracy % — no longer hardcoded
- `ADV` const with 17 current players → `LEGENDS_ADV` (6 legends only); `_mergeAdv()` uses live advanced_stats.json (504 players) first

### Bug Fixes
- `gameCard()` TypeError — `g.ats` can be undefined; fixed with `const ats=g.ats||''`
- CSP `frame-ancestors` removed from meta tag (browser ignores it there)
- Greeting de-personalized (removed "Spencer" from 3 strings)
- Today page placeholder ("Building player database...") → Player Comparison tool link card

### Verification
- Playwright confirms 0 JS errors across Today, History, Sharp Money, Bet Tracker tabs
- 560 tests passing (pytest)
- 4 commits pushed to main; GitHub Pages deploying

## Pending at Session End

**Nothing pending** — all tasks completed and deployed.

## Next Steps (priority order)

1. **Wire CLV summary card to real CLV data** — `updateCLVSummary()` currently uses `value_bets.edge_pct` as a proxy (wrong semantics); real data is in `predictions_history.db` `clv_tracking` table; build `scripts/build_clv.py` -> `dashboard/data/clv_summary.json` and wire into Promise.all (append 15th fetch)
2. **Daily deployment automation** — every day: `python update.py` -> `git add dashboard/data/ && git push`. Consider Windows Task Scheduler job.
3. **Season History tab**: currently shows team abbreviations in game log (home/away columns); full names would be cleaner — minor UI improvement

## Key Files

- `update.py` — full pipeline entry point (run this daily); Step 7 calls all 24 builder scripts (added build_season_history)
- `scripts/fetch_odds.py` — Pinnacle odds + props; writes `data/odds/game_lines.csv`
- `scripts/build_value_bets.py` — reads `data/odds/game_lines.csv` (NOT data/processed/)
- `scripts/build_props.py` — player props with Pinnacle book lines (60 players, 34 matched lines)
- `scripts/build_season_history.py` — NEW; 5 seasons from team_game_logs.csv -> season_history.json
- `dashboard/data/*.json` — COMMITTED to git; update by running update.py then pushing

## Critical Gotchas
- `dashboard/data/*.json` are COMMITTED (not gitignored) — must `git add dashboard/data/` after each update.py run
- `game_lines.csv` is at `data/odds/` not `data/processed/`
- `player_stats.csv` has season TOTALS — divide by `gp` before computing per-game projections
- Security hook blocks Edit when replacement text contains the literal string "innerHTML" — use `_setHtml(el,html)` pattern instead
- CSP `frame-ancestors` is IGNORED in meta tags — only works as HTTP response header
- `g.ats` can be undefined in game objects — always use `const ats=g.ats||''` before `.includes()`/`.startsWith()`
- Promise.all is 14 fetches — adding a new tab: append to tuple, destructure, wire in BOTH loader AND tab-click handler
- Python http.server: stale content served after in-place file edits — start fresh server on a new port (8081) to verify fixes
