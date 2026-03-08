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

**Game Outcome Model Backtest Accuracy Degrades Post-2014:**
- Problem: Model achieves 67-69% accuracy on 2005-2015 data but only 64-66% on 2016-2026 data
- Files: `src/models/game_outcome_model.py`, `reports/backtest_game_outcome.csv`
- Cause: Modern 3-point era introduces higher variance (more outlier games); model trained on mixed eras
- Improvement path: (a) Filter training to modern era only (2014+) per Proposal 5 in `docs/model_advisor_notes.md`, (b) add pace and 3-point efficiency rolling averages, (c) verify injury features are active

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

**Player Model Prediction Confidence Intervals:**
- Problem: Player points/rebounds/assists projections are point estimates with no uncertainty
- Blocks: Cannot distinguish high-confidence from low-confidence projections; flagging logic treats all disagreements equally
- Current status: Not implemented; listed as Proposal 8 in `docs/model_advisor_notes.md`
- Documentation: Feature described in model advisor notes; no ticket created

**Minutes Projection for Player Model:**
- Problem: Player models predict points/rebounds/assists but not minutes; minutes allocation is volatile and affects accuracy
- Blocks: Cannot adjust for load management; may systematically overestimate players coming off injury
- Current status: Not implemented; listed as Proposal 9 in `docs/model_advisor_notes.md`
- Documentation: Feature described in notes; no implementation started

## Test Coverage Gaps

**No Unit Tests for Data Fetchers: PARTIALLY RESOLVED**
- **RESOLVED (2026-03-05):** `tests/test_get_balldontlie.py`, `tests/test_get_injury_data.py`, `tests/test_get_lineup_data.py` added (31 tests covering retry logic, pagination, schema contracts, and error paths).
- Remaining gap: `src/data/get_player_stats.py`, `get_team_stats.py`, `get_standings.py` nba_api callers still lack unit tests (they require nba_api mocking or live network).

**No Unit Tests for Preprocessing: RESOLVED**
- **RESOLVED (2026-03-05):** `tests/test_preprocessing.py` has 35 tests covering `clean_columns`, `load_season_folder`, `load_season_files`, `get_stale_seasons`, `merge_incremental`, `_season_label`, type coercion patterns, and duplicate removal.

**No Unit Tests for Feature Engineering: RESOLVED**
- **RESOLVED (2026-03-05):** `tests/test_injury_proxy.py` (9 tests: absent rotation detection, star player flag, staleness window, output schema, join key types) and `tests/test_team_game_features.py` (28 tests: parse_home_away, rolling mean shift, rolling win pct, haversine, no-leakage integration test) cover all critical paths. Explicit `shift(1)` leakage test verifies no data leakage regresses.

**No Unit Tests for Custom NumPy GBM: RESOLVED**
- `src/models/numpy_gbm.py` was deleted (2026-03-05). All models use sklearn. No custom GBM to test.

**Integration Tests Only; No End-to-End Validation: RESOLVED**
- **RESOLVED (2026-03-05):** `tests/test_data_integrity.py` added with 17 tests: game_id uniqueness, season integer dtype, home_win valid values, (game_id, player_id) pair uniqueness, rolling feature leakage spot-check. All tests skip cleanly on fresh clones via `pytest.mark.skipif` guards.

---

*Concerns audit: 2026-03-01*
