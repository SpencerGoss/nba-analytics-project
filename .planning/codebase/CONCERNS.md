# Codebase Concerns

**Analysis Date:** 2026-03-05

## Tech Debt

**Duplicated Error Handling and Logging: RESOLVED**
- **RESOLVED (2026-03-05):** `src/data/api_client.py` created with shared `fetch_with_retry()`, `HEADERS`, and retry config. All 19 data-fetching scripts now import from it. No script defines its own retry logic.

**Shot Chart API Bottleneck: PARTIALLY RESOLVED**
- Issue: `src/data/get_shot_chart.py` makes one API call per player per season (~11,000 calls, 3-4 hours runtime)
- **RESOLVED (2026-03-08):** Added per-season incremental skip — if `data/raw/shot_chart/shot_chart_{season_code}.csv` already exists, that season is skipped entirely. Re-runs only fetch missing seasons.
- Remaining: Still a one-time 3-4h cost for full historical fetch; excluded from daily pipeline by design.

**Preprocessing Always Rebuilds All CSVs: RESOLVED**
- **RESOLVED (2026-03-08):** `_needs_rebuild(raw_path, out_path)` helper added to `preprocessing.py`. Skips rebuild when output CSV exists and is newer than raw input. `_coerce_int_col()` helper replaces 24 silent `.astype(int)` calls with `pd.to_numeric(errors='coerce')` + explicit WARN log.

**Inconsistent Data Type Coercion: RESOLVED**
- **RESOLVED (2026-03-08):** `_coerce_int_col(df, col)` helper added to `src/processing/preprocessing.py`. All player_id, game_id, team_id, season, from_season, to_season columns now use coerce-with-logging instead of silent `.astype(int)`. 24 call sites converted.

## Known Bugs

**Stale SHAP Explainability Reports: RESOLVED**
- **RESOLVED (2026-03-05):** `src/models/model_explainability.py` refactored with graceful SHAP fallback to sklearn permutation importance. All reports regenerated against current model schema. SHAP is now a soft optional dependency with `SHAP_AVAILABLE` guard.

**Calibration Artifacts Saved But Never Loaded: RESOLVED**
- Symptoms: `models/artifacts/game_outcome_model_calibrated.pkl` exists but no production code ever loads it
- Files: `src/models/calibration.py` line 377-380 saves calibrated model; `scripts/fetch_odds.py` and `src/models/predict_cli.py` never import it
- Impact: Calibration work is wasted; `fetch_odds.py` outputs uncalibrated probabilities to sportsbook comparison file
- **RESOLVED (commit 942d909, 2026-03-05):** Calibrated model is now loaded first in `fetch_odds.py` and `src/models/value_bet_detector.py`.

**Missing Injury Features in Game Outcome Model: RESOLVED**
- **RESOLVED (2026-03-06):** `src/data/get_historical_absences.py` now uses `format="mixed"` for game_date parsing (was crashing). `data/processed/player_absences.csv` generated (1,098,538 rows, 12.6% absence rate). `injury_proxy.py` now uses primary path (126,818 team-game rows, 60.9% with missing minutes). `game_matchup_features.csv` rebuilt. After retraining: 11 injury features appear in `game_outcome_importances.csv` — `home_rotation_availability` is rank #5 (0.027), `diff_missing_usg_pct` rank #10 (0.015).

**Player Backtest Stops in 2015-16: RESOLVED**
- **RESOLVED (2026-03-05 audit):** `reports/backtest_player_pts.csv` verified to cover 2001-02 through 2025-26 (30 seasons). Player features CSV has data for all seasons 1996-97 to 2025-26. Loop boundary in `backtesting.py` is correct — `PLAYER_BACKTEST_START = "200001"` with no upper bound.

**Minor V1 Model Calibration Data Leakage: RESOLVED**
- **RESOLVED (2026-03-06):** `calibration.py` `run_calibration_analysis()` now accepts `calibration_season="202122"` parameter. When season is in the training pool and has >=100 games, isotonic regression is fit on held-out predictions from that season instead of in-sample. `ats_model.py` reserves `202122` as a dedicated calibration split (excluded from expanding-window CV) and saves `models/artifacts/ats_calibration_split.json` for downstream use.

## Security Considerations

**Environment Variables Not Validated at Startup: RESOLVED**
- **RESOLVED (2026-03-05):** `update.py` now calls `_check_env_vars()` as the first statement in `main()`. Loads `.env` via python-dotenv, warns clearly for each missing key (`ODDS_API_KEY`, `BALLDONTLIE_API_KEY`) without halting the pipeline.

**API Headers Hardcoded Across Scripts: RESOLVED**
- **RESOLVED (2026-03-05):** `HEADERS` dict now defined once in `src/data/api_client.py` and imported by all data scripts. Still a single User-Agent fingerprint — but now centrally managed (one edit vs 19 files).

**Error Logs May Contain Sensitive Data: MITIGATED**
- **MITIGATED (2026-03-05):** `.gitignore` confirmed to include `logs/` and `*.log` — log files will never be committed. Remaining recommendation: sanitize error messages to exclude full URL/payload stack traces (low priority, v3.0 scope).

## Performance Bottlenecks

**Shot Chart API Ingestion (3-4 hours): PARTIALLY RESOLVED**
- **RESOLVED (2026-03-08):** Per-season incremental skip added — already-fetched seasons are skipped on re-run. One-time cost is still 3-4h; subsequent runs are instant for fetched seasons.
- Remaining improvement: per-player incrementalism within a season (skip players already in CSV).

**Preprocessing Reprocesses Entire History Daily: RESOLVED**
- **RESOLVED (2026-03-08):** `_needs_rebuild()` helper skips seasons where output CSV is already up-to-date. See Tech Debt section above.

**Duplicate Column Cleaning Across All Tables:**
- Problem: `clean_columns()` called 25+ times per preprocessing run; does string operations on 100,000+ rows
- Files: `src/processing/preprocessing.py` lines 21-31
- Impact: Negligible for current scale (~25MB processed data) but will bottleneck if schema expands or raw ingestion speeds increase
- Improvement path: Cache cleaned column names per prefix; vectorize string operations if adding >50 more raw data sources

**Game Outcome Model Backtest Accuracy Degrades Post-2014: PARTIALLY RESOLVED**
- **RESOLVED (2026-03-13):** (a) Modern era filter active (2013-14+, excludes bubble seasons), (b) diff_pace_game_roll20 + diff_four_factors_roll20 added — improved accuracy 67.5->67.9%, AUC 0.7422->0.7455, (c) injury features confirmed active (rank #5-12).
- Remaining: Post-2014 gap is partially explained by declining home court advantage (60.3% pre-2014 -> 55.3% post-2022). Modern NBA games are inherently harder to predict — this is a data characteristic, not a model flaw.

## Fragile Areas

**Data Fetcher Scripts Are Tightly Coupled to nba_api Endpoint Structure:**
- Files: `src/data/get_player_stats.py` lines 46-50, `src/data/get_team_stats.py` lines 45-51, etc. (all use hardcoded endpoint args)
- Why fragile: If nba_api changes endpoint signatures or returns different column names, all 15+ scripts fail simultaneously
- Safe modification: (a) Create a wrapper for each endpoint that translates between our schema and nba_api's response, (b) add CSV validation in preprocessing to catch schema mismatches early and log which endpoint drifted
- Test coverage: Gaps — no unit tests for data fetchers; only integration tests implicitly via successful CSV generation

**Preprocessing Column Rename Mapping Is Hardcoded:**
- Files: `src/processing/preprocessing.py` lines 70-76, 174-181 (rename dicts)
- Why fragile: If NBA endpoint returns a new column, preprocessing silently includes it in mixed case (not cleaned); if they rename an existing column, preprocessing uses old name and job fails
- Safe modification: (a) Add schema validation at start of preprocessing — compute expected columns from config, compare to actual input, (b) generate the rename dict from a configuration file rather than hardcoding
- Test coverage: Gaps — no schema validation tests; only tested via end-to-end runs

**Error Recovery Relies on Silent Continuation:**
- Files: `src/data/get_player_stats.py` line 54-55 (continue on None); `update.py` line 126-127 (odds refresh optional)
- Why fragile: `fetch_with_retry()` returns `None` on all failures; calling code has no way to distinguish "API timeout" from "invalid season" from "malformed response"
- Safe modification: (a) Return structured result object `{success: bool, data: pd.DataFrame | None, error: str}` instead of just `None or DataFrame`, (b) log error type when continuing, (c) fail loudly if critical endpoint returns None
- Test coverage: Gaps — no unit tests for retry logic or error paths; only catch-all exception handler in `update.py`

**Feature Engineering Pipeline Has Silent Data Joins:**
- Files: `src/features/injury_proxy.py` (377 lines), `src/features/team_game_features.py` (673 lines)
- Why fragile: Multiple inner joins on (`game_id`, `team_id`, `season`) can silently drop rows if join keys mismatch; no row count assertions after joins
- Safe modification: (a) Add assertions after every join: `assert len(before_join) == len(after_join) or len(after_join) == 0` with descriptive error, (b) log join statistics (how many rows matched), (c) add unit tests with synthetic data
- Test coverage: Gaps — no tests for feature engineering; difficult to debug missing features in downstream model

**Custom NumPy GBM Implementation: RESOLVED**
- `src/models/numpy_gbm.py` was deleted (2026-03-05) — 700 lines of dead code that was never imported anywhere.
- All production models now use `sklearn.ensemble.GradientBoostingClassifier` and related sklearn estimators.

## Scaling Limits

**API Rate Limiting and Throttling:**
- Current capacity: ~10-15 API calls per daily update run (~2-5 minutes); shot chart fetch takes 3-4 hours
- Limit: nba_api throttles at 1 request per ~1-2 seconds per session
- Scaling path: (a) Implement request batching/pipelining where nba_api supports it (most endpoints don't), (b) cache raw CSV files for 24h to avoid re-fetches if job runs twice, (c) separate shot chart into monthly task with its own scheduler

**Database Growth (SQLite):**
- Current capacity: `database/nba.db` contains 18 tables, 60+ years of data; fits in single file
- Limit: SQLite file locking becomes problematic if >10GB; row counts in `player_game_logs` (~1.5M rows) and `team_game_logs` (~50k rows) are manageable
- Scaling path: (a) Add query indexing on foreign keys and date ranges if query performance degrades, (b) partition historical (1946-2000) from modern (2001-2026) data if storage grows beyond 1GB, (c) migrate to PostgreSQL if concurrent query load exceeds 2-3 simultaneous connections

**CSV Processing in Memory:**
- Current capacity: `preprocessing.py` concatenates all 25 seasons of player/team stats into single DataFrames; each ~100MB
- Limit: Reading 25 seasons + 3 retries × 25 seasons for different stat types = ~600MB peak memory; acceptable on modern machines but risks OOM on resource-constrained environments
- Scaling path: (a) Process one season at a time and append to output CSV incrementally, (b) use `dask.dataframe` for out-of-core processing if adding future data sources

**Feature Engineering Job Runtime:**
- Current capacity: `src/features/team_game_features.py` (673 lines) takes ~5-10 minutes to compute rolling features over 80 years of data
- Limit: If feature set expands 10x or data window extends to play-by-play granularity, could exceed 1-hour threshold
- Scaling path: (a) Vectorize rolling window operations (use `rolling()` instead of loops), (b) cache intermediate results (e.g., 10-game rolling averages), (c) parallelize by season or team

## Dependencies at Risk

**nba_api Package Maintenance Status:**
- Risk: `nba_api` is a community-maintained wrapper around NBA.com JSON endpoints; if NBA changes APIs or blocks unofficial clients, package breaks
- Impact: All 15+ data-fetching scripts fail simultaneously; no data pipeline update until fix is available
- Migration plan: (a) Document fallback endpoint URLs in case nba_api breaks, (b) fork nba_api and maintain internally if breaking changes occur frequently, (c) investigate official NBA Stats API alternatives (e.g., SportsRadar)

**scikit-learn Version Compatibility: RESOLVED**
- **RESOLVED (2026-03-05):** `requirements.txt` now pins `scikit-learn==1.8.0` (exact installed version). Model artifacts will not silently break on version upgrades.

**SHAP Dependency Optional But Model Explainability Depends On It: RESOLVED**
- **RESOLVED (2026-03-05):** `model_explainability.py` refactored with `try/except ImportError` guard and `SHAP_AVAILABLE` flag. Falls back to sklearn's permutation importance when shap is absent. No hard failure.

## Missing Critical Features

**Shot Chart Data Unbuilt:**
- Problem: `shot_chart` table is planned and code exists (`src/data/get_shot_chart.py`) but dataset not generated
- Blocks: Shot chart visualizations, heat map analysis, shot clustering features
- Current status: In docs as TODO; not run as part of daily pipeline due to 3-4 hour runtime
- Documentation: `src/processing/preprocessing.py` line 329 acknowledges it's optional and skipped

**Calibration Outputs Not Integrated: RESOLVED**
- Calibrated model (`game_outcome_model_calibrated.pkl`) is now loaded first in `scripts/fetch_odds.py` and `src/models/value_bet_detector.py` (fixed 2026-03-05, commit 942d909).
- sys.path guard added to ensure model wrapper class is importable during deserialization.

**Player Model Prediction Confidence Intervals: RESOLVED**
- **RESOLVED (2026-03-13):** Conformal prediction intervals implemented with distribution-free 90% coverage guarantees (PTS +/-7.93, REB +/-2.96, AST +/-2.14, 3PM +/-2.30). Quantile regression (p25/p50/p75) with monotonicity enforcement.

**Minutes Projection for Player Model: RESOLVED**
- **RESOLVED (2026-03-13):** Two-stage architecture: Stage 1 GBM with Huber loss predicts minutes (MAE 5.03), Stage 2 uses per-36 rates scaled by predicted minutes. Isolates playing-time signal from skill.

## Test Coverage Gaps

**No Unit Tests for Data Fetchers: MOSTLY RESOLVED**
- **RESOLVED (2026-03-05):** `tests/test_get_balldontlie.py`, `tests/test_get_injury_data.py`, `tests/test_get_lineup_data.py` added (31 tests covering retry logic, pagination, schema contracts, and error paths).
- **RESOLVED (2026-03-13):** `tests/test_get_player_stats.py` (6 tests), `tests/test_get_team_stats.py` (6 tests), `tests/test_fetch_player_positions.py` (41 tests) added — mocked API boundary tests.
- Remaining gap: `get_standings.py` nba_api caller still lacks unit tests.

**No Unit Tests for Preprocessing: RESOLVED**
- **RESOLVED (2026-03-05):** `tests/test_preprocessing.py` has 35 tests covering `clean_columns`, `load_season_folder`, `load_season_files`, `get_stale_seasons`, `merge_incremental`, `_season_label`, type coercion patterns, and duplicate removal.

**No Unit Tests for Feature Engineering: RESOLVED**
- **RESOLVED (2026-03-05):** `tests/test_injury_proxy.py` (9 tests: absent rotation detection, star player flag, staleness window, output schema, join key types) and `tests/test_team_game_features.py` (28 tests: parse_home_away, rolling mean shift, rolling win pct, haversine, no-leakage integration test) cover all critical paths. Explicit `shift(1)` leakage test verifies no data leakage regresses.

**No Unit Tests for Custom NumPy GBM: RESOLVED**
- `src/models/numpy_gbm.py` was deleted (2026-03-05). All models use sklearn. No custom GBM to test.

**Integration Tests Only; No End-to-End Validation: RESOLVED**
- **RESOLVED (2026-03-05):** `tests/test_data_integrity.py` added with 17 tests: game_id uniqueness, season integer dtype, home_win valid values, (game_id, player_id) pair uniqueness, rolling feature leakage spot-check. All tests skip cleanly on fresh clones via `pytest.mark.skipif` guards.

## Critical: Stale Feature Bug in predict_game() (2026-03-11): RESOLVED

**RESOLVED (2026-03-12):** `predict_game()` now filters for current-season exact matchups only (line 714-718). When no current-season matchup exists, `_synthesize_matchup_row()` builds a fresh row from each team's most recent game, injects current Elo from `get_current_elos()`, and recomputes all `diff_*` columns. Same fix applied to `predict_cli.py` and `build_picks.py`.

**SEVERITY: HIGH -- predictions for 3 of 6 games today are materially wrong** (original severity, now fixed)

### Root Cause

`predict_game()` in `src/models/game_outcome_model.py` (line 626-628) uses the **most recent exact historical matchup** between two specific teams, regardless of how old it is. For teams that haven't played each other yet this season, it falls back to **last season's features** -- including last season's Elo ratings, win percentages, rolling stats, and cumulative records.

The model does NOT synthesize a current-season feature row from each team's latest stats when the exact matchup is stale. It literally uses the feature values frozen at the time of their last meeting.

### Affected Predictions (2026-03-11)

| Game | Prediction | Features From | Key Problem |
|------|-----------|---------------|-------------|
| **CHA @ SAC** | SAC 87% (WRONG) | 2025-02-24 (last season) | Uses SAC Elo 1465 / CHA Elo 1200 / diff_elo +265. **Reality: SAC Elo 1252 / CHA Elo 1639 / diff should be -387.** SAC is 16-50 (worst in West), CHA is 33-33. Prediction is inverted. |
| **TOR @ NOP** | NOP 69% win (SUSPICIOUS) | 2024-11-27 (last season) | Uses NOP Elo 1381 / TOR Elo 1309 / diff_elo +73. **Reality: NOP Elo 1408 / TOR Elo 1519 / diff should be -111.** NOP is 21-45, TOR is rebuilding but ~30+ wins. Direction may be correct (NOP has home court) but magnitude is inflated by stale features. |
| **HOU @ DEN** | DEN 61% (STALE) | 2025-12-20 (early this season) | Uses DEN Elo 1735 / HOU Elo 1647 / diff_elo +88. **Reality: DEN Elo 1556 / HOU Elo 1602 / diff should be -46.** DEN was much stronger in December; has since fallen. Prediction direction might flip with current features. |

Three games use current-season features and are reasonable:
- CLE @ ORL: features from 2026-01-24 (this season) -- reasonable
- MIN @ LAC: features from 2026-02-26 (this season) -- reasonable
- NYK @ UTA: features from 2024-11-23 (last season) but both teams' relative strength is similar, so prediction (NYK 96%) is directionally correct

### Why diff_elo Matters So Much

`diff_elo` is the #1 feature at 37.3% importance in the game outcome model. When this value is wrong by 500+ Elo points (as in CHA @ SAC: model uses +265, reality is -387), the prediction is catastrophically wrong.

### The Fallback Path (lines 630-647) Is Also Problematic

When there is no exact matchup, `predict_game()` synthesizes a row from each team's latest home/away row separately and recomputes diff_ columns. This fallback is **better** than the exact-match path for cross-season predictions because it uses each team's most recent game. However, it has its own issues:
- It copies `away_*` columns from a different opponent context
- diff_ columns are recomputed from home/away raw values, but Elo columns (home_elo, away_elo, diff_elo) are NOT recomputed -- they come from the copied rows where one team was playing a different opponent

### Proposed Fix

**Option A (quick fix):** In `predict_game()`, when the exact matchup row is from a previous season (different `season` value), force the fallback path instead. This ensures current-season stats are used for each team.

**Option B (proper fix):** Always synthesize a fresh feature row for future-game predictions:
1. Take the most recent home row for the home team (from current season)
2. Take the most recent away row for the away team (from current season)
3. Copy home_* and away_* columns from their respective rows
4. Recompute ALL diff_ columns (including diff_elo using current Elo ratings from `get_current_elos()`)
5. This ensures predictions always reflect the teams' current form

**Option C (best fix):** Create a `predict_future_game()` function that:
1. Loads current Elo ratings from `elo_ratings.csv` (latest per team)
2. Loads each team's most recent team_game_features row
3. Constructs a matchup row from scratch using current features
4. Never relies on historical exact-match rows for future games

### Impact Assessment

- Every cross-season matchup prediction is potentially wrong
- Teams that improved or declined significantly between seasons are most affected
- SAC (went from .500 to .242) and CHA (went from .254 to .500) show the most dramatic reversal
- This bug has been present since the prediction system was built -- it affects all historical predictions for first-time seasonal matchups

### Related Files

- `src/models/game_outcome_model.py` lines 604-676 (`predict_game()`)
- `src/models/predict_cli.py` lines 60-105 (`_handle_ats()` -- same pattern)
- `scripts/build_picks.py` lines 139-175 (`_build_margin_lookup()` -- same pattern for margin model)
- `src/features/elo.py` line 206-226 (`get_current_elos()` -- provides correct current Elos)
- `data/features/game_matchup_features.csv` -- the feature source

---

*Concerns audit: 2026-03-01*
*Stale feature bug analysis: 2026-03-11*
