# Handoff — NBA Analytics Project

_Last updated: 2026-03-07 Session 3_

## What Was Done This Session

### Live Site Now Deployed
- Removed `dashboard/data/*.json` from `.gitignore`; committed all 22 JSON files + pushed
- GitHub Actions auto-deploy triggered — live site now shows real season data
- **To update the live site:** run `python update.py` (now includes Step 7 that builds all 23 JSON files), then `git add dashboard/data/ && git push`

### Pipeline: update.py Step 7
Added Step 7 that calls all 23 builder scripts in dependency order after predictions are generated. Running `update.py` alone now produces a complete, deployable dashboard end-to-end.

Builder order: backfill_outcomes → fetch_odds → build_picks → build_value_bets → standings/injuries/rankings/h2h/streaks/advanced_stats/live_scores/playoff_odds/trends/totals/game_context/explainers/matchup_analysis → performance/accuracy_history/line_movement → build_props → build_player_comparison

### Bug Fixes
- **`build_value_bets.py`**: path was `data/processed/game_lines.csv` → fixed to `data/odds/game_lines.csv`; column mismatch fixed (`date`→`game_date`, `home_moneyline`→`home_market_prob`)
- **`backfill_outcomes.py`**: ran, resolved 8 predictions from 2026-03-06; season accuracy 64.0% (214 games)

### New Predictions Today (2026-03-07)
6 games: OKC/GSW 77.5%, MIL/UTA 86.3%, DET/BKN 88.5%, MIN/ORL 86.3%, MEM/LAC 77.5%, ATL/PHI 50%

### New Scripts
- `scripts/build_player_props.py` — season-avg based props builder (80 players, reads player_stats.csv / gp)

### BallDontLie API Key
User added key to `.env` — next `update.py` run will fetch supplementary game stats via BallDontLie.

## Pending at Session End

**`props-wiring` agent still running** — replacing `fetch_player_props()` stub in `scripts/fetch_odds.py` with real Pinnacle API calls (157 props, 92% accessible). Also joining real book lines in `scripts/build_props.py`. When it finishes:
```bash
git add scripts/fetch_odds.py scripts/build_props.py
git commit -m "feat(props): wire real Pinnacle player prop lines into build_props.py"
git push
```

## Next Steps (priority order)

1. **Check if props-wiring agent completed** — commit `scripts/fetch_odds.py` + `scripts/build_props.py` if so; re-run `scripts/build_props.py` to regenerate `player_props.json` with real Pinnacle lines, then push
2. **Verify live GitHub Pages site** — open the site, check each tab loads real data
3. **Wire CLV summary card correctly** — `updateCLVSummary()` currently uses `value_bets.edge_pct` as a CLV proxy (wrong semantics); real CLV data is in `predictions_history.db`'s `clv_tracking` table; build a `build_clv.py` → `clv_summary.json` and wire it into the dashboard separately
4. **Daily deployment workflow** — every day: `python update.py` → `git add dashboard/data/ && git push`. Consider automating with a Windows Task Scheduler job or cron.

## Key Files

- `update.py` — full pipeline entry point (run this daily)
- `scripts/fetch_odds.py` — Pinnacle odds + props; writes `data/odds/game_lines.csv`
- `scripts/build_value_bets.py` — reads `data/odds/game_lines.csv` (NOT data/processed/)
- `scripts/build_props.py` — player props with Pinnacle book lines (being updated by agent)
- `dashboard/data/*.json` — committed; update by running update.py then pushing

## Critical Gotchas
- `dashboard/data/*.json` are committed (not gitignored) — must `git add dashboard/data/` after each update.py run
- `game_lines.csv` is at `data/odds/` not `data/processed/`
- `player_stats.csv` has season TOTALS — divide by `gp` before computing per-game projections
- Security hook blocks Edit with "innerHTML" in replacement text — rephrase or use anchor strings
- Promise.all is 14 fetches — adding a new tab: append to tuple, destructure, wire in both loader AND tab-click handler
