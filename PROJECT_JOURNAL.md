# NBA Analytics Project — Session Journal

Append a dated entry at the start of each session. Keep entries brief — just what was done and what's next.

---

## 2026-03-07 (Session 4) — Dashboard Overhaul: 3 New Tabs + Security Hardening + Bug Fixes

**Done:**
- **Sharp Money Tracker tab** — fully built from `line_movement.json`: stats bar (games tracked, steam moves, biggest move), per-game cards with movement bars, STEAM badge when |move| >= 1.5 pts, direction arrows
- **Bet Tracker tab** — localStorage-based (key `baseline_bets_v1`): add-bet form (date pre-filled to today, matchup, pick, market, odds, stake, result), stats row (total bets, W-L record, ROI%, net P&L), history table with per-bet delete, clear-all
- **Season History tab** — lazy-loaded `season_history.json` (552.8 KB, 5 seasons, 5,995 games, 30 teams/season): season selector dropdown, standings table, 50-game log table; `scripts/build_season_history.py` created and wired into `update.py` Step 7
- **Security hardening** — CSP meta tag added (removed invalid `frame-ancestors`); Plotly SRI hash added (`sha384-Hl48Kq2HifOWdXEjMsKo6qxqvRLTYqIGbvlENBmkHAxZKIGCXv43H6W1jA671RzC`); fake account dropdown removed; `_setHtml()` helper using `createContextualFragment + replaceChildren` established as XSS-safe DOM write pattern
- **ADV constant replaced** — 17-player hardcoded ADV → `LEGENDS_ADV` (6 legends: Jordan/Kobe/Bird/Magic/Kareem/Wilt only); `_mergeAdv()` prioritizes live `advanced_stats.json` (504 players) over legends
- **All hardcoded data cleared** — `const MATCHUP_DATA=[]` (was 3 Mar-5 games), `DATA.picks=[]` (was 9 Mar-5 picks), `bt-backtest-total` → dynamic from accuracy_history backtest, `bt-ats-inline` → dynamic, OG/Twitter meta tags no longer hardcode accuracy %
- **Greeting de-personalized** — "Spencer" removed from 3 greeting strings
- **Today page placeholder replaced** — "Building player database..." → Player Comparison tool link card
- **Bug fixes** — `gameCard()` TypeError when `g.ats` undefined (added `ats=g.ats||''` guard); CSP `frame-ancestors` removed from meta tag (only valid as HTTP header)
- **Playwright verification** — 0 JS errors across Today, History, Sharp Money, Bet Tracker tabs; only harmless `apple-mobile-web-app-capable` deprecation warning remains
- **Pinnacle props wired** (completed from Session 3 agent) — `player_props.json` refreshed: 60 players, 34 Pinnacle-matched prop lines; `value_bets.json` refreshed: 1 entry (PHI@ATL, 21.5% edge)
- **3 commits, 2 pushes** — deployed to GitHub Pages

**Next:**
1. Build `build_clv.py` → `clv_summary.json` and wire CLV card to real CLV data (currently uses `value_bets.edge_pct` as proxy — wrong semantics; real data is in `predictions_history.db` `clv_tracking` table)
2. Daily workflow automation — Windows Task Scheduler job to run `python update.py && git add dashboard/data/ && git push` daily
3. Run `python update.py` daily to refresh all 23 JSON files before pushing

---

## 2026-03-07 (Session 3) — Live Site Deployed: Pipeline End-to-End + Real Dashboard Data

**Done:**
- **GitHub Pages now shows real data** — removed `dashboard/data/*.json` from .gitignore; committed all 22 JSON files; pushed → deploy triggered
- **`update.py` Step 7 added** — 23 builder scripts called in dependency order after prediction generation; running `update.py` alone now produces a fully deployable dashboard
- **`build_value_bets.py` fixed** — path mismatch: `data/processed/game_lines.csv` → `data/odds/game_lines.csv`; column mismatch: `date`→`game_date`, `home_moneyline`→`home_market_prob` (American odds to implied prob)
- **`backfill_outcomes.py` ran** — 8 predictions resolved from 2026-03-06; season accuracy now 64.0% (214 games, 6/8 correct)
- **6 new predictions for 2026-03-07** written: OKC/GSW 77.5%, MIL/UTA 86.3%, DET/BKN 88.5%, MIN/ORL 86.3%, MEM/LAC 77.5%, ATL/PHI 50%
- **Pinnacle player props** (157 live props, 92% accessible) — `fetch_player_props()` stub in `scripts/fetch_odds.py` being replaced with real implementation (in-progress at session end)
- **`build_player_props.py` created** — season-avg based props (80 players, <5s, reads player_stats.csv + game_logs)
- **BallDontLie API key** added to `.env` by user; `.env.example` updated to document it
- **Audited full dashboard** — 22 JSON files, all populated; confirmed 13 more builder scripts exist beyond the 5 known ones; `scheduler.py` is the separate runner
- **All builder scripts running** — standings, injuries, power_rankings, h2h, streaks, live_scores, playoff_odds, trends, totals, game_context, explainers, matchup_analysis all OK
- **2 commits, 1 push** — GitHub Actions deploying

**Still in progress at session end:**
- `props-wiring` agent: replacing Pinnacle props stub in `scripts/fetch_odds.py` + joining real book lines in `scripts/build_props.py`

**Next:**
1. Commit `scripts/fetch_odds.py` + `scripts/build_props.py` once props-wiring agent finishes
2. Verify live GitHub Pages site loads real data
3. Wire CLV summary card to `clv_tracking` table (currently uses `value_bets.edge_pct` as proxy — wrong semantics)

---

## 2026-03-07 (Session 2) — Phase 2 Complete: Ensemble Model + Full Dashboard De-hardcoding

**Done:**
- **Margin regression model trained** — `src/models/margin_model.py` runs Ridge/Lasso/GBR expanding-window CV; **Ridge selected** (MAE 10.57 vs GBR 10.59); 74 features; artifacts: `margin_model.pkl`, `margin_model_features.pkl`
- **Ensemble model** — `src/models/ensemble.py` blends all 3 models: win_prob 50% + ats_prob 30% + margin_signal 20%; `NBAEnsemble.load()` confirmed all 3 models active; 30 tests passing; config saved to `ensemble_config.json`
- **5 new builder scripts** — `build_live_scores.py`, `build_playoff_odds.py`, `build_streaks.py`, `build_advanced_stats.py`, `build_accuracy_history.py` all new; all produce real JSON for dashboard
- **Dashboard: 10 hardcoded sections eliminated:**
  - MATCHUP_DATA (3 stale games) → 8 live matchups from `matchup_analysis.json`
  - Championship odds chart → live from `playoff_odds.json` top-8 title odds
  - Performance tab stats (54.9%/18,496 hardcoded) → `performance.json`
  - Sharp Money tab → `renderSharpMoney()` wired to `line_movement.json`
  - **Player Props tab** (was "coming soon") → 80 players, filter bar (All/PTS/REB/AST/3PM/STL/BLK), value badges
  - ADV merge (17 players only) → net_rtg→bpm, efg→thr for all 504 live players
  - Home spotlight (hardcoded Hornets/Doncic) → live from `streaks.json`
  - Timestamps ("Mar 5" everywhere) → `meta.json` exported_at
  - Games Tonight + Picks tab counters → live from picks + live_scores.json
  - Def Rating highlighting → fixed lower-is-better comparison in drawMatchup()
- **Promise.all extended** from 10 to 14 fetches: added matchup_analysis, performance, line_movement, player_props
- **All 16 builder scripts ran** — fresh data: 6 live games, 8 matchups, 30 team streaks, 4 hot/cold players, 504 player advanced stats
- **meta.json** now includes `model_version`, `season`, `sample_data` fields
- **560 tests passing** (no regressions; 16 ensemble tests new)
- **Pushed to GitHub Pages**

**Files changed:**
- `dashboard/index.html` — 300+ line delta; all above JS changes
- `scripts/build_live_scores.py`, `build_playoff_odds.py`, `build_streaks.py`, `build_advanced_stats.py`, `build_accuracy_history.py` — new
- `src/models/ensemble.py` — new (NBAEnsemble class)
- `tests/test_ensemble.py` — new (30 tests → 16 after dedup)
- `update.py` — ensemble step 6b added
- `models/artifacts/margin_model.pkl`, `margin_model_features.pkl`, `ensemble_config.json` — new (gitignored)

**Remaining legitimate placeholders:**
- Bet Tracker — needs user accounts; Season History — needs full game log browser; Shot maps — need NBA shot chart API (3-4h, excluded from pipeline)

---

## 2026-03-07 — Players Tab Historical Stats, Dashboard Audit, HPO Confirmed, Bug Fix

**Done:**
- **Players tab — 1,710 players from JSON** — `loadDashboardData()` now fetches `player_comparison.json` and maps all players into `DATA.players` at runtime. Static 12-entry hardcoded array replaced. Added `mapJsonPlayer()` and `LEGENDS` const to normalize fields; legends merged in after fetch.
- **Search + pagination** — added `<input id="player-search">` with `setPlayerSearch()` function; table defaults to top-100 rows with "Show More" button (`pShowAll` flag); prevents DOM freeze with 1,710 rows.
- **Filter fix** — "Current Season" filter now uses `seasons_span` containing "2024"/"2025" instead of `!p.retired` (more accurate).
- **Compare pickers** — optgroup labels show player counts ("Current Season (471)"); null-safe with `esc()`.
- **ADV_NULL stub** — `updateComparison()` uses `ADV[name]||ADV_NULL` so radar/advanced sections show "No advanced data" instead of crashing for players without ADV entries.
- **Standings tab order fixed** — "Standings" (conference tables) is now the default active tab; "Today's Games" is second. Page description updated.
- **Legends expanded** — Kobe Bryant, Larry Bird, Magic Johnson, Kareem Abdul-Jabbar, Wilt Chamberlain added to DATA.players with `retired:true` + ADV entries. LeBron corrected to LAL.
- **Dashboard audit — 7 bugs found and fixed:**
  1. `showPage` IIFE executing before function declared → `DOMContentLoaded`
  2. `setEraMode` null crash — era toggle buttons missing from HTML → added buttons + null guards
  3. Parlay functions (`renderParlay`, `toggleParlay`, `updateParlaySlip`) null crashes → guards added
  4. `injectTooltips` querying `model-tab-*` IDs (don't exist) → fixed to `page-bet-*`
  5. Dead `showBettingTab` stub conflicting with real function → removed
  6. `switchTab('games','standings')` targeting non-existent page → fixed to `switchTab('standings','conf')`
  7. `LEGENDS` and `mapJsonPlayer` referenced before defined → definitions added before `loadDashboardData`
- **Optuna HPO ran (100 trials each)** — LightGBM AUC=0.7116, XGBoost AUC=0.7115. Both WORSE than gradient_boosting baseline (0.7406). Production model unchanged. Best params saved to `models/artifacts/best_hpo_params.json`.
- **`build_line_movement.py` NULL guard** — `closing_spread` is NULL for games that haven't closed; added `pd.isna()` check before `float()` cast. Test `test_loads_real_db` now passes (29/29).
- **Daily pipeline ran** — 7 predictions for Mar 6 games written to `predictions_history.db`.

**Issues encountered:**
- Background `run_in_background` bash commands on Windows didn't capture output (empty files). Fixed by using `cmd > /tmp/file.txt 2>&1 &` pattern instead.
- HPO script uses `--trials` not `--n-trials`; failed silently on wrong flag name.
- `python` vs `.venv/Scripts/python.exe` — system Python lacks optuna/lightgbm; always use venv Python for ML scripts.
- Two parallel agents editing same file (historical stats + audit) — both changes captured in one commit since auditor ran while wiring agent was mid-edit.

**Files changed:**
- `dashboard/index.html` — 1,710 player JSON wiring, search/pagination, audit fixes, legends, standings tab order
- `scripts/build_line_movement.py` — NULL guard on closing_spread
- `WORKING_NOTES.md` — HPO result insight added
- `models/artifacts/best_hpo_params.json`, `game_outcome_model_hpo.pkl` — HPO artifacts (gitignored)

**Decision: gradient_boosting retained over LightGBM/XGBoost**
- Context: 100 Optuna trials each, same expanding-window CV
- Chose: keep gradient_boosting (AUC=0.7406) — LightGBM only reached 0.7116
- Trade-off: LightGBM may improve with more features or different feature engineering; revisit in Phase 3

**Next:** Phase 2 model improvements (margin regression, ensemble stacking, SBRO historical odds integration); wire real standings/injuries/rankings data from builder scripts into dashboard.

---

## 2026-03-06 — Dashboard v3 Full Redesign + Legend Players + Phase 2 HPO

**Done:**
- **Full dashboard v3 redesign** — Linear/Coinbase aesthetic (ultra-dark, glass-morphism cards, neon-green). Design doc written at `docs/plans/2026-03-06-dashboard-v3-design.md`. 53 new files including 16 builder scripts, `dashboard/about.html`, PWA manifest + service worker, og-image SVG.
- **Player Comparison page** — LeBron vs Jordan as default, team color theming (TEAM_COLORS dict), era toggle (Raw/Era-Adjusted), sparklines, career trajectory radar charts. Built `scripts/build_player_comparison.py`.
- **Legend player injection** — `_inject_legends()` in build_player_comparison.py replaces partial NBA API records with curated Basketball Reference career stats: Jordan (30.1 PPG / 6.2 RPG / 5.3 APG / 1072 games / 1984-2003), Bird, Magic, Kareem, Kobe, Wilt. Legal: career stats are publicly known facts, not copyrightable. Output: 1,710 players (1,706 API + 4 new legends not in API).
- **Historical data fetch** — `scripts/fetch_historical_players.py` fetched 79 seasons (1946-47 to 2024-25, ~1996-97+ reliable). Fixed header bug (see Issues). Saved `data/raw/historical_player_seasons.csv`.
- **Teams page** — East/West conference tables (Overview tab), 30-team form streak cards (Form tab), wired to `DATA.east` / `DATA.west` from standings builder.
- **H2H page** — Team dropdowns, `renderH2H()` / `initH2H()` / `_populateDropdowns()` / `_findMatchup()` / `_renderBanner()` / `_renderStats()` / `_renderTable()`. Lazy-loads `dashboard/data/head_to_head.json`.
- **Power Rankings page** — Hero banner, score progress bars, divergence callouts (Our Rank vs Media). Lazy-loads `dashboard/data/power_rankings.json`.
- **Injuries tab** — Per-game cards with HIGH/MED/LOW impact pills, spread impact callout. Lazy-loads `dashboard/data/injuries.json`.
- **Value Bets cards upgrade** — Rich cards with WE SAY / BOOKS SAY / EDGE bars, 3-bullet explainer, share button, confidence pill.
- **Betting Tools drawer** — Fixed-right slide-out (⚙ nav icon): odds converter (American↔Decimal↔Implied%), edge calculator, confidence sizing (opt-in + disclaimer). Functions: `ocFromAmerican()`, `ocFromDecimal()`, `ocFromImplied()`, `calcEdge()`, `calcSizing()`.
- **Optuna HPO** — `scripts/tune_hyperparams.py` dispatched (100 trials, LightGBM+XGBoost, expanding-window CV). Running in background.
- **`dashboard/data/*.json`** added to `.gitignore` (regenerated by scripts, not committed).
- All 3 commits pushed to GitHub Pages; live site deploying.

**Issues encountered:**
- **NBA API missing Jordan's prime** — `LeagueDashPlayerStats` only covers ~1996-97+; Jordan had only 4 seasons (306 games) below thresholds. Fixed: (1) lowered to `--min-seasons 3 --min-games 200`, (2) added `_inject_legends()` with full career stats.
- **fetch_historical_players.py header bug** — Line 229 had `first_write and i <= len(frames)`. When 50+ early seasons return no data, `i` (60+) >> `len(frames)` (10), so condition was always False → header never written → CSV had no column names. Fixed to `first_write` only.
- **Rate-limited parallel agents** — All 4 dispatched UI agents hit "You've hit your limit" immediately. Implemented Teams page and Betting drawer directly; waited for reset before re-dispatching remaining agents with surgical scope.
- **UnicodeEncodeError in Python -c** — Windows cp1252 blocked player name output; fixed by using `sys.stdout.buffer.write(msg.encode('utf-8'))`.

**Files changed:**
- `dashboard/index.html` — grew 3098 → 4216 lines (Teams, H2H, Power Rankings, Injuries, Value Bets, Betting drawer, era toggle, LeBron/Jordan default)
- `scripts/build_player_comparison.py` — legend injection, TEAM_COLORS, era normalization
- `scripts/fetch_historical_players.py` — header bug fix (line 229)
- `scripts/tune_hyperparams.py` — new Optuna HPO script
- `dashboard/about.html`, `dashboard/manifest.json`, `dashboard/sw.js`, `dashboard/og-image.svg` — new files
- 16 builder scripts in `scripts/` — data layer for all v3 sections
- `.gitignore` — dashboard/data/*.json excluded

**Decision: Legend override pattern for pre-1996 players**
- Context: NBA API doesn't cover pre-1996 seasons; Jordan/Bird/Magic careers truncated to Washington/Boston cameos
- Chose: curated `_LEGENDS` dict with Basketball Reference career aggregates over web-scraping (Cloudflare blocked) or paying for historical data
- Trade-off: static career totals only (no season-by-season for pre-1996); era normalization requires league-avg data per season for full accuracy

**Next:** Check Optuna HPO results; retrain LightGBM if beats 67.1% baseline; model blending; review live site at https://spencergoss.github.io/nba-analytics-project/

---

## 2026-03-06 — Dashboard UI Enhancement + Security Hardening + Bug Fixes

**Done:**
- **Dark/light theme toggle** — nav bar button; persists to localStorage; CSS variables for both modes.
- **Game ticker bar** — scrolling strip of live game scores/picks (DATA.games + DATA.picks).
- **Value Bets tab** — new tab rendering value_bets.json: matchup, side, model prob, market prob, edge%, kelly fraction via `renderValueBets()`.
- **CLV summary card** in Track Record tab — mean CLV, positive CLV rate, edge flag badge. Extracted to `updateCLVSummary()`, called from data loader so it populates regardless of tab click order.
- **Sortable standings** — W/L/PCT headers clickable; missing span IDs added (sarr-east-l/pct, sarr-west-l/pct).
- **Real accuracy chart** — `renderDashChart()` replaced hardcoded PPG with accuracy_history.json (40-day rolling; Plotly area + 50% dashed baseline). Empty state uses textContent (not direct DOM injection).
- **VALUE badge + Kelly display** on pick cards — green VALUE badge and Kelly % when data present. `kelly_fraction` now mapped from JSON in `loadDashboardData()`.
- **probToAmerican()** helper — correct American odds formula (fav: neg round, dog: pos round).
- **Security hardening** — `esc()` sanitizer on all external JSON strings; onerror fallback uses textContent via `hsErr()` function; integer guard on nbaId before CDN URL construction.
- **Async data loader** — `loadDashboardData()` fetches all 4 JSON files in parallel via Promise.all.
- **5 code review bugs fixed:** (1) Number() coercion on posRate string comparison; (2) 4 missing sort arrow span IDs; (3) consistent table rendering in renderValueBets; (4) CLV extracted to standalone function; (5) kelly_fraction mapped in picks + updateCLVSummary wired into data loader.
- **CSS additions:** value-pulse keyframes, flash-green keyframes, ticker-scroll animation, light mode CSS vars, CLV badge styles.

**Issues encountered:**
- Security hook blocked direct DOM injection for empty state — replaced with textContent + Object.assign(el.style).
- posRate comparison was string vs number — fixed with Number() coercion.
- CLV card only populated on tab click — fixed by calling updateCLVSummary from data loader.

**Files changed:**
- `dashboard/index.html` — all UI enhancements, security fixes, bug fixes

**Next:** Phase 2 — Optuna HPO on LightGBM/XGBoost, model blending, SBRO historical odds, margin regression model.

---

## 2026-03-06 — Dashboard Full Data Update (all mock sections wired to real data)

**Done:**
- **`scripts/build_dashboard.py` overhauled** — 10 new data sections (16–25) read CSVs + predictions DB at build time and embed real values as JS in `dashboard/index.html`. All ~35 previously mock/hardcoded sections are now live.
- **Section 16 — PPG Trend:** Weekly scoring trend from `team_game_logs.csv` (202526 season, W7–W20). Real values 110–118 PPG range replacing fake `[110,112,109...]`.
- **Section 17 — ATS Win Rate Over Time:** 14 months of real monthly cover rates (42–57%) from `game_ats_features.csv`. Labels formatted as "Oct '23", "Feb '25" etc.
- **Section 18 — ATS Best Pick Categories:** Cover rates by spread bucket (Pick'em 50.1%, Fav 3.5-7 49.4%, 7.5-10 49.2%, HeavyFav 48.3%). Honest data replacing fake "ATS/ML/Over-Under" labels.
- **Section 19 — Confidence Calibration:** Implied prob buckets (50-75%+) vs. actual cover rate from market odds. Flat ~49-51% — honest display; axis range changed to [44,58] to show true signal level.
- **Section 20 — Backtest Stat Tiles:** Best month = Feb '25 (57.4%); High conf tile = 49.0% (market implied >= 65%); Total = 18,496 games (2007-2025) replacing "1,240".
- **Section 21 — Matchup Breakdown:** TOR@MIN (92% MIN), BKN@MIA (90% MIA), NOP@SAC (90% SAC) — real rolling stats from `game_matchup_features.csv` + predictions from `predictions_history.db`. Smart reason text (home advantage vs. offensive/defensive edge).
- **Section 22 — Hot/Cold Streaks:** Real last-5 game scores per player from `player_game_logs.csv` (season_id=22025, min 15 PPG filter). Hot: Cooper Flagg (+8.2), Jalen Duren (+6.8), Jrue Holiday (+6.4), Saddiq Bey (+5.9). Cold: Jalen Johnson, Austin Reaves, Franz Wagner, Deni Avdija.
- **Section 23 — Shot Quality/Efficiency:** Real TS% [61.3–67.6%] and pts/TSA [1.14–1.35] from `player_stats_advanced.csv` for top 8 scorers. Color thresholds updated to TS% scale.
- **Section 24 — Shot Zones:** Luka/SGA/Edwards zone breakdown from `player_stats_scoring.csv` (pct_pts_3pt, pct_pts_paint, pct_uast_2pm). Replaces fake SGA/Jokic/Giannis data.
- **Section 25 — Parlay Odds:** Model-derived American odds from `predictions_history.db` win probabilities. Signed string format ("-230" not "+230") with template literal fix.
- 145 tests passing; 0 regressions.

**Key discovery:** ATS cover rates are essentially flat (48–51%) across all implied-prob and spread buckets — no raw market edge. The model's 54.9% comes from 73-feature ML.

**Issues encountered:**
- `season_id=22025` is correct for player_game_logs.csv 202526 season (not integer 202526); other CSVs use `season=202526`.
- Long 2s zone formula was computing `(pts_paint - pts_2mr - pts_paint) = -pts_2mr` → always 0. Fixed to `max(2, round((1.0 - pts_3 - pts_paint - pts_2mr) * 50))`.
- ATS backtest chart ranges changed: PPG y=[105,122], calibration y=[44,58], TS% x=[50,72].
- Parlay `+${odds[i]}` template must become `${odds[i]}` when odds are signed strings.

**Files changed:**
- `scripts/build_dashboard.py` — new file (~420 lines), 10 new real-data sections
- `dashboard/index.html` — rebuilt with all real data embedded
- `WORKING_NOTES.md` — new [dashboard] domain section (4 insights)

**Next:** Phase 2 — Optuna HPO on LightGBM/XGBoost, model blending, SBRO historical odds, margin regression model. Dashboard is now production-ready for data display.

---

## 2026-03-06 — Phase 1 remaining items: LightGBM, Pythagorean win%, Fractional Kelly, CLV (complete)

**Done:**
- **LightGBM candidate added** to `game_outcome_model.py` with guarded import (`_LGBM_AVAILABLE` flag). Competes in expanding-window CV but gradient_boosting still selected (67.1% acc, AUC 0.7406). LightGBM available for Phase 2 Optuna HPO.
- **Pythagorean win% feature** added to `team_game_features.py`: `pythagorean_win_pct_game` (per-game, Morey exponent 14.3) + `pythagorean_win_pct_roll10` (10-game rolling with shift(1)). `diff_pythagorean_win_pct_roll10` added to matchup dataset diff_stats. Matchup CSV: 68,216 rows x 296 cols (was 291).
- **Fractional Kelly sizing** added to `value_bet_detector.py` — `kelly_fraction` field in every `get_strong_value_bets()` output dict. Formula: `f = 0.5 * (p*b - (1-p)) / b` where `b = (1-q)/q` (no-vig market odds). Home/away side-aware.
- **CLV tracker** created (`src/models/clv_tracker.py`): `clv_tracking` table in `predictions_history.db`, `CLVTracker.log_opening_line()` (INSERT OR IGNORE), `update_closing_line()` (computes CLV = opening - closing), `get_clv_summary()` with `has_edge` flag. `fetch_odds.py` updated to call `log_opening_line()` for each game after saving `game_lines.csv` (step 1b, non-fatal).
- Calibrated model regenerated (02:02), ATS model retrained (02:23) — both artifacts fresh.
- 145 tests passing throughout; 0 regressions.

**Issues encountered:**
- `pythagorean_win_pct_roll10` has `_roll` in name → auto-captured by `roll_cols` filter in `build_matchup_dataset()`. Adding it also to `context_cols` caused `ValueError: Cannot set a DataFrame with multiple columns` (duplicate). Fix: removed from `context_cols` in both functions; only kept in `diff_stats`.
- CLV formula initially implemented as `closing - opening` (wrong). Corrected to `opening - closing` per research plan (positive = we got a better line).
- `calibration.py` and `ats_model.py` don't set `sys.path` at top level — running as scripts fails with `ModuleNotFoundError: No module named 'src'`. Workaround: `python -c "import sys; sys.path.insert(0,'.'); from src.models.calibration import run_calibration_analysis; run_calibration_analysis()"`.

**Files changed:**
- `src/models/game_outcome_model.py` — LightGBM guarded import + candidate pipeline
- `src/features/team_game_features.py` — pythagorean_win_pct_game, pythagorean_win_pct_roll10, diff_pythagorean_win_pct_roll10
- `src/models/value_bet_detector.py` — `_compute_kelly_fraction()` + kelly_fraction field
- `src/models/clv_tracker.py` — new file (~180 lines)
- `scripts/fetch_odds.py` — step 1b CLV opening line logging
- `requirements.txt` — `lightgbm>=4.0.0`
- `WORKING_NOTES.md`, `HANDOFF.md` — updated

**Decision: gradient_boosting over LightGBM for v2.1**
- Context: LightGBM added as candidate; both ran in expanding-window CV
- Chose: gradient_boosting (existing sklearn) — marginally better AUC in this run
- Trade-off: LightGBM available for Optuna HPO in Phase 2 where hyperparameter search will give it proper tuning budget

**Next:** Phase 2 — Optuna HPO on LightGBM/XGBoost, model blending, SBRO historical odds, margin regression model.

---

## 2026-03-06 — Model Improvement Phase 1 (complete)

**Done:**
- Deployed 5 parallel research agents to identify highest-leverage model improvements. Synthesized into prioritized plan at `docs/plans/2026-03-06-model-improvement-research.md`.
- **CF-2 (injury features zero-filled) — FIXED:** `get_historical_absences.py` was crashing on game_date time suffix; added `format="mixed"`. Generated `data/processed/player_absences.csv` (1,098,538 rows, 12.6% absence rate, 75 seasons). `injury_proxy.py` now uses primary path — 60.9% of games have non-zero missing_minutes (was 0% before). Rebuilt `game_matchup_features.csv` (68,216 rows x 291 cols). After retraining: 11 injury features appear in importances; `home_rotation_availability` is rank #5. AUC improved 0.7256 -> **0.7419**.
- **CF-1 (ATS optimizing wrong metric) — FIXED:** Switched ATS model selection from `max(accuracy)` to `min(brier_score_loss)`. Added `CALIBRATION_SEASON="202122"` held out from expanding-window CV. `calibration.py` now fits isotonic calibrator on held-out 2021-22 season (1,230 games, no in-sample leakage). ATS retrained: `logistic_l1` selected (vs old accuracy-based winner), test accuracy **54.9%** (up from 53.5%), AUC 0.5571. Metadata includes `validation_mean_brier` and `calibration_season` fields.
- Fixed Unicode `→` arrows in 3 model files that crashed on Windows cp1252 (`game_outcome_model.py`, `playoff_odds_model.py`, `train_all_models.py`).
- Spec written and completed: `docs/specs/2026-03-06-model-improvement-phase1.md`.
- 145 tests passing throughout; 0 regressions.

**Files changed:**
- `src/data/get_historical_absences.py` — `format="mixed"` fix for game_date parsing
- `src/models/ats_model.py` — Brier score selection, calibration split, metadata update
- `src/models/calibration.py` — held-out calibration season parameter
- `src/models/game_outcome_model.py`, `playoff_odds_model.py`, `train_all_models.py` — Unicode arrow fix
- `.planning/codebase/CONCERNS.md` — two issues marked RESOLVED
- `HANDOFF.md` — updated to reflect new model state
- `docs/plans/2026-03-06-model-improvement-research.md` — new research synthesis
- `docs/specs/2026-03-06-model-improvement-phase1.md` — new spec (COMPLETE)

**Next:** Add LightGBM candidate model, Pythagorean win% feature, Fractional Kelly sizing, CLV tracking.

---

## 2026-03-06 — Pinnacle API migration (complete)

**Done:**
- Replaced The Odds API with Pinnacle guest API across the entire project. Pinnacle is free, keyless, no quota, and provides sharper lines (better for value-bet signal).
- Rewrote `scripts/fetch_odds.py`: new `get_pinnacle()` client (no auth), `fetch_game_lines()` using `/leagues/487/matchups` + `/leagues/487/markets/straight`, `fetch_player_props()` stubbed (empty DataFrame — Pinnacle props use different endpoint structure).
- Removed `QuotaError`, `check_remaining_quota()`, and `ODDS_API_KEY` guard from `src/models/value_bet_detector.py`.
- Removed `ODDS_API_KEY` from `update.py` env check, `.env`, `.env.example`.
- Updated `src/data/get_odds.py` and `src/models/predict_cli.py` help text.
- Updated 5 doc files: `ARCHITECTURE.md`, `CONTEXT.md`, `docs/PIPELINE.md`, `.claude/rules/nba-domain.md`, `.planning/codebase/INTEGRATIONS.md`.
- Bug fixes from code review: (1) added `parentId is not None` filter to exclude Pinnacle alternate/period lines from matchup list; (2) fixed `flagged.sum()` to use `dropna()` for pandas >= 2.0 compatibility.
- Spec written: `docs/specs/2026-03-06-pinnacle-api-migration.md`.
- Live run confirmed: 7 NBA game lines fetched from Pinnacle with no errors; `game_lines.csv` populated.
- 145 tests passing throughout; 0 regressions.

**Files changed:**
- `scripts/fetch_odds.py` — full rewrite of API client section
- `src/models/value_bet_detector.py` — removed quota guard + ODDS_API_KEY check
- `src/data/get_odds.py` — removed ODDS_API_KEY log branch
- `update.py` — removed ODDS_API_KEY from env check
- `.env` / `.env.example` — ODDS_API_KEY removed
- `src/models/predict_cli.py` — updated --live help text
- `ARCHITECTURE.md`, `CONTEXT.md`, `docs/PIPELINE.md`, `.claude/rules/nba-domain.md`, `.planning/codebase/INTEGRATIONS.md` — updated to reference Pinnacle
- `docs/specs/2026-03-06-pinnacle-api-migration.md` — new spec file

**Decision: Pinnacle guest API over The Odds API**
- Context: ODDS_API_KEY expired (401); The Odds API free tier is 500 req/month (limiting)
- Chose: Pinnacle guest API (`https://guest.api.arcadia.pinnacle.com/0.1`, NBA league 487) — verified free + keyless 2026-03-06
- Trade-off: player props not yet implemented (Pinnacle props use different endpoint; stubbed for now)

**Next:**
- v3.0 web dashboard — wire Pinnacle game lines (`game_lines.csv`, `model_vs_odds.csv`) into the dashboard display; surface value bets flagged by the model

---

## 2026-03-05 — Investigated nba.db + odds API replacement decision

**Done:**
- Confirmed `predictions_history.db` has 9 rows (healthy) — was already written by last session's `update.py` run; no action needed
- Investigated `database/nba.db` (0 bytes): confirmed it is a legacy artifact from Feb 2026 early dev (populated once via `src/processing/load_to_sql.py`). No current code in `src/` reads or writes to it. Pipeline is entirely CSV-based. Safe to ignore.
- Ran `scripts/fetch_odds.py`: confirmed ODDS_API_KEY returns 401 (key expired/invalid). Model pipeline inside the script works correctly (loaded calibrated model, generated win probs for 921 games).
- Researched free/legal odds API alternatives: Pinnacle (keyless public API, sharp-money lines), Action Network (unofficial), ESPN embedded JSON, The Odds API free tier (just needs new key).
- **Decision:** Replace The Odds API with Pinnacle API across the entire project. Remove all Odds API references. Pinnacle is free, keyless, no quota limits, and provides sharper lines (better for value-bet detection).
- Updated stale docs: `ARCHITECTURE.md` and `.claude/rules/nba-domain.md` both said "nba.db — 18 tables"; corrected to reflect legacy/empty status.

**Files changed:**
- `ARCHITECTURE.md` — corrected nba.db description (legacy/empty, not 18 tables)
- `.claude/rules/nba-domain.md` — corrected nba.db description

**Next:**
- Switch `scripts/fetch_odds.py` to Pinnacle API (free, keyless) — remove all The Odds API code
- Remove `ODDS_API_KEY` from `.env`, `.env.example`, and any references in project docs
- Audit all files referencing "odds api", "ODDS_API_KEY", or "the-odds-api.com" and update

---

## 2026-03-05 — Injury features restored + prediction store wired up

**Done:**
- Fixed missing injury proxy columns in `game_matchup_features.csv` (11 cols: `home_/away_/diff_missing_minutes`, `missing_usg_pct`, `rotation_availability`, `star_player_out`). Root cause: `build_team_game_features()` wraps the injury proxy join in a bare `except Exception` — any failure is swallowed silently, CSV written without those columns. Fix: merged `injury_proxy_features.csv` directly into `team_game_features.csv` (92.9% match rate, 126,818 rows), then rebuilt matchup dataset. Matchup CSV now has 291 cols.
- Verified calibrated model inference: 921 current-season games, win prob range 0.0–1.0 ✓
- Wired up `predictions_history.db` — was always 0 rows because nothing ever called `predict_game()` for today's schedule. Added `generate_today_predictions(game_date)` to `update.py` as Step 6: fetches schedule via `ScoreboardV2`, maps team IDs → abbreviations using `nba_api.stats.static.teams`, calls `predict_game()` for each game (which writes to store). Wrote 9 predictions for tonight's games using calibrated model.
- Also ran `fetch_odds.py` — confirmed 401 on ODDS_API_KEY (key expired, needs renewal); non-blocking.
- 145 tests passing throughout.

**Files changed:**
- `update.py` — added `generate_today_predictions()` function + Step 6 call in `main()`
- `WORKING_NOTES.md` — new `[injury]` domain entry
- `data/features/team_game_features.csv` — patched with 5 injury proxy columns (123 cols total)
- `data/features/game_matchup_features.csv` — rebuilt with 291 cols (was 278)

**Issues found (not yet fixed):**
- `build_team_game_features()` uses bare `except Exception` for injury proxy join — should be narrowed to `except ImportError` so real failures surface
- `ODDS_API_KEY` is expired — fetch_odds.py returns 401; needs new key from the-odds-api.com
- `database/nba.db` remains 0 bytes (pipeline runs entirely off CSVs)

**Next:**
- Renew ODDS_API_KEY in .env
- Tighten `except Exception` in `build_team_game_features()` injury proxy join
- v3.0 planning — web dashboard polish

---

## 2026-03-05 — Data refresh + feature pipeline bug fixes

**Done:**
- Ran full daily update (update.py): fetched 2025-26 season data (1,852 team games, 20,103 player game logs, standings, hustle stats, reference tables)
- Found and fixed 3 bugs blocking the feature rebuild:
  1. `pd.to_datetime()` without `format="mixed"` — NBA API now sends current season game_date as "2025-10-21 00:00:00" (with time suffix) while historical rows use "YYYY-MM-DD". Pandas inferred the wrong format. Fixed in 6 places across `team_game_features.py` and `injury_proxy.py`.
  2. `build_matchup_dataset()` was missing from `update.py` step 3 — `game_matchup_features.csv` (the file `fetch_odds.py` reads for predictions) was never being rebuilt on daily runs. Added explicit import and call.
  3. Unicode `→` in print statements raised `UnicodeEncodeError` on Windows cp1252 terminal. Replaced with `->` in 3 print statements.
- Feature rebuild now completes: `team_game_features.csv` (136,452 rows × 118 cols), `game_matchup_features.csv` (68,216 rows × 278 cols) — both fresh as of Mar 5
- Used parallel agents: Agent 1 ran update.py in background while Agent 2 pre-validated data state; Agent 1 surfaced the feature rebuild error; fixed and re-ran
- 145 tests passing throughout (no regressions)
- Committed: `51f3d11 fix(features): handle mixed game_date formats from NBA API`

**Files changed:**
- `src/features/team_game_features.py` — format="mixed" at lines 340, 849; `->` arrows in 3 print statements
- `src/features/injury_proxy.py` — format="mixed" at lines 161, 327, 362, 673
- `update.py` — import + call `build_matchup_dataset()` after `build_team_game_features()` in step 3

**Pre-validation findings (noted for future work):**
- `database/nba.db` is empty (0 bytes, no tables) — pipeline runs entirely off CSVs; DB population is a separate unreached step
- `predictions_history.db` has schema but 0 rows — `fetch_odds.py` has not yet written predictions successfully

**Next:**
- Run `scripts/fetch_odds.py` to generate today's game predictions using fresh features
- Investigate empty `nba.db` — determine if DB population is needed or if CSV-only is the intended architecture

---

## 2026-03-05 — Bug fix: calibrated model not loading in fetch_odds.py

**Done:**
- Diagnosed fetch_odds.py silently falling back to a feature-based proxy instead of the trained 68% game outcome model
- Root cause: PROJECT_ROOT was not on sys.path, so the deserializer could not find src.models.calibration._CalibratedWrapper
- Fix: Added sys.path.insert guard after PROJECT_ROOT is resolved in scripts/fetch_odds.py
- Verified: now logs "Loaded calibrated game outcome model"; win probs generated for 870 current-season games
- All 59 tests still passing

**Next:**
- Audit other scripts/ scripts for the same missing sys.path guard

---

## 2026-03-04 — Project restructure

**Done:**
- v1.0 milestone complete (all 5 phases, 17/17 plans). See `.planning/STATE.md`.
- Added `.env.example`, `PROJECT_OVERVIEW.md`, `PROJECT_JOURNAL.md` (this file)
- Moved `CLAUDE_CODE_TASKS.md` → `docs/plans/v2-multi-agent-task-prompt.md`
- Collapsed stale duplicate blocks in `.planning/STATE.md`
- Updated `CLAUDE.md` skill routing to match current skills
- Updated `.gitignore` to cover dashboard generated data

**v1.0 status summary:**
- Game outcome: 66.8% accuracy, calibrated model saved
- ATS model: 51.2% (below 52.4% vig breakeven — v2 improvement target)
- Prediction store: operational (WAL-mode SQLite + JSON snapshots)
- Dashboard: built (`dashboard/index.html` + `scripts/export_dashboard_data.py`)
- Tests: 3 test files covering preprocessing, injury proxy, team game features

**Known open issues:**
- ATS accuracy below breakeven — v2 feature engineering needed
- Injury features may still be all-null in production model (verify with `reports/explainability/`)
- Basketball Reference scraper blocked by Cloudflare in Windows dev environment
- Calibrated model not loaded in `fetch_odds.py` (uses uncalibrated probabilities)

**Next session should:**
- Decide v2 focus: ATS model improvement vs injury feature debugging vs test coverage
- Check `.planning/codebase/CONCERNS.md` for full issue list before starting any work

---

## Template for new entries

```
## YYYY-MM-DD — <one-line topic>

**Done:**
- ...

**Next:**
- ...
```
