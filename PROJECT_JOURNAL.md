# NBA Analytics Project — Session Journal

Append a dated entry at the start of each session. Keep entries brief — just what was done and what's next.

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
