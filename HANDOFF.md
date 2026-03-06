# Handoff — NBA Analytics Project

_Last updated: 2026-03-05_

## What Was Built

**Data refresh + feature pipeline bug fixes**

- Ran `update.py` — refreshed 2025-26 raw data (1,852 team games, 20,103 player game logs, standings, hustle stats)
- Fixed `pd.to_datetime()` bug in `team_game_features.py` (lines 340, 849) and `injury_proxy.py` (lines 161, 327, 362, 673) — NBA API sends current season game_date as "YYYY-MM-DD 00:00:00", historical as "YYYY-MM-DD"; `format="mixed"` required
- Fixed 3 Unicode `->` arrows in print statements (Windows cp1252 UnicodeEncodeError on `->`)
- Added `build_matchup_dataset()` call to `update.py` step 3 — it was never being called, so `game_matchup_features.csv` was silently stale on every daily run
- Rebuilt `team_game_features.csv` (136,452 rows x 118 cols) and `game_matchup_features.csv` (68,216 rows x 278 cols) — both fresh as of Mar 5
- Updated `CLAUDE.md` (100 lines), `WORKING_NOTES.md` (Core Insights synthesized to 10 bullets), `PROJECT_JOURNAL.md`, `PROJECT_OVERVIEW.md`

**Commits this session:**
- `51f3d11` — fix(features): handle mixed game_date formats from NBA API
- `71333bc` — docs(session): update journal, working memory, and CLAUDE.md

## Current State

- Raw data: fresh (Mar 5 2026)
- `team_game_features.csv`: fresh (136,452 rows)
- `game_matchup_features.csv`: fresh (68,216 rows x 278 cols)
- Calibrated model: `game_outcome_model_calibrated.pkl` — Mar 5
- ATS model: `ats_model.pkl` — Mar 4
- Tests: 145 passing, 0 failing
- Pipeline: `update.py` now fully correct end-to-end

## Failed Approaches

- Attempted full parallelism for `update.py` — NBA API enforces 1 req/sec throttling, so only genuine parallelism was: update.py in background while a second agent pre-validated data state

## Autonomous Decisions (Flag for Review)

- Trimmed Python and Project Meta skill routing sections from CLAUDE.md to hit 100-line limit — those rows are still available via `AGENTS.md`
- Used `format="mixed"` (not `format="ISO8601"`) for `pd.to_datetime()` — handles heterogeneous historical/current-season date formats

## What's Left

1. **Run `scripts/fetch_odds.py`** — generate today's predictions using freshly rebuilt features; verify predictions are written to `predictions_history.db`
2. **Investigate empty `nba.db`** — `database/nba.db` is 0 bytes with no tables; determine if DB population is needed or CSV-only is the intended architecture
3. **`predictions_history.db` has 0 rows** — verify `fetch_odds.py` writes predictions successfully after running it
4. **v3.0 planning** — web dashboard polish is the next milestone per CLAUDE.md

## Where Stuck

No blockers. Pipeline is healthy. Logical next action is running `scripts/fetch_odds.py`.

## Key Decisions

- `format="mixed"` is now the standard for all `pd.to_datetime()` calls on game_date columns — never use bare `pd.to_datetime(col)` without it
- `build_matchup_dataset()` must always be called after `build_team_game_features()` in any pipeline entry point (update.py, backfill.py)
