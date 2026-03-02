# Roadmap: NBA Analytics — Model Enhancement & ATS Betting

## Overview

Starting from a working but imperfect NBA prediction pipeline, this milestone closes three gaps: broken foundations that silently degrade predictions, missing data sources that unlock new predictive signal, and the ATS betting model that is the core business goal. Each phase delivers a verifiable capability that the next phase depends on. The critical path runs Foundation → Modern Features → External Data → ATS Model, with Rest/Schedule features running after external data patterns are established.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Foundation & Outputs** - Fix broken features, wire calibrated model, and start accumulating prediction history
- [x] **Phase 2: Modern Era Features** - Add pace-normalized efficiency metrics and restrict training to the modern NBA (completed 2026-03-02)
- [ ] **Phase 3: External Data Layer** - Build scrapers for Basketball Reference and live injury reports
- [ ] **Phase 4: Rest & Schedule Features** - Add travel distance, back-to-back fatigue, and season-context features
- [ ] **Phase 5: ATS Model** - Build the ATS classifier, value-bet detector, and full backtest harness

## Phase Details

### Phase 1: Foundation & Outputs
**Goal**: The model produces predictions backed by working features, calibrated probabilities, and a persistent history store — removing all known silent failures before any new features are added
**Depends on**: Nothing (first phase)
**Requirements**: FR-1.1, FR-1.2, FR-1.3, FR-6.1, FR-6.2, FR-6.3, FR-6.4, FR-6.5, FR-7.1, FR-7.2, FR-7.3, FR-7.4, NFR-1, NFR-2, NFR-3
**Success Criteria** (what must be TRUE):
  1. Running `predict_cli.py` on any upcoming game produces non-null values for `missing_minutes` and `star_player_out` in the feature vector
  2. The calibrated model artifact (`game_outcome_model_calibrated.pkl`) is loaded at inference — confidence outputs are calibrated probabilities, not raw scores
  3. Running feature assembly with any column exceeding 95% null raises an explicit error with the column name before training begins
  4. A prediction record (game, predicted winner, probabilities, model metadata) is written to `predictions_history.db` after every `predict_cli.py` run, and a JSON snapshot is written to the output directory
  5. Each pipeline stage (`fetch`, `preprocess`, `features`, `train`, `predict`) can be invoked independently with documented inputs and outputs; `update.py` calls each as a one-line module import
**Plans**: 4 plans

Plans:
- [x] 01-01: Fix injury proxy join and null-guard feature assembly
- [x] 01-02: Wire calibrated model into predict_cli inference path
- [x] 01-03: Build prediction store and JSON export layer
- [x] 01-04: Document and clean up pipeline stage boundaries

### Phase 2: Modern Era Features
**Goal**: The game outcome model is trained exclusively on modern NBA data (2014+) and uses pace-normalized efficiency metrics that exist in the database but were never wired into the feature pipeline
**Depends on**: Phase 1
**Requirements**: FR-2.1, FR-2.2, FR-2.3, FR-2.4, FR-2.5, NFR-1
**Success Criteria** (what must be TRUE):
  1. The game outcome model training script filters to seasons 2013-14 through 2023-24, explicitly excluding 2019-20 bubble and 2020-21 shortened seasons
  2. The feature matrix for any game includes rolling ORtg, DRtg, net rating, eFG%, TS%, and pace drawn from `team_stats_advanced` — all with `.shift(1)` applied before `.rolling()` to prevent lookahead
  3. Turnover rate is computed as turnovers per possession (not raw count) and appears as a rolling feature in the assembled feature table
  4. The Four Factors differential composite (eFG%, TOV%, ORB%, FT rate) exists as a matchup feature column in the assembled training data
  5. Model accuracy on the 2014+ holdout set meets or exceeds the accuracy on the full historical dataset, confirming era filtering improves rather than hurts performance
**Plans**: 3 plans

Plans:
- [x] 02-01: Wire advanced stats table into team feature pipeline (ORtg, DRtg, pace, eFG%, TS%)
- [ ] 02-02: Add Four Factors differential and pace-normalized turnover rate
- [ ] 02-03: Restrict training era to 2014+ and validate accuracy improvement

### Phase 3: External Data Layer
**Goal**: The pipeline can fetch referee crew assignments from Basketball Reference and official NBA pre-game injury report status — each following the existing `src/data/get_*.py` module pattern
**Depends on**: Phase 1
**Requirements**: FR-4.1, FR-4.2, FR-4.3, FR-4.4, FR-7.2, NFR-2, NFR-4
**Success Criteria** (what must be TRUE):
  1. Running `src/data/external/bref_scraper.py` fetches referee crew assignments for a given date range, saves to `data/raw/external/`, and respects a 3-second delay between requests
  2. A referee foul-rate rolling feature (FTA/game, pace impact per crew) is present in the assembled feature table for games after the scrape date range
  3. Running `src/data/external/injury_report.py` fetches the current NBA pre-game injury report and saves structured status (Questionable/Probable/Out) per player
  4. The inference path uses live injury report status; the training path uses historical injury proxy from game logs — these are separate code paths that never share inputs
  5. All new external scrapers are callable as modules (matching `src/data/get_*.py` pattern) and listed in the pipeline reference document
**Plans**: 4 plans

Plans:
- [ ] 03-01: Spike and build Basketball Reference referee scraper
- [ ] 03-02: Build referee foul-rate feature from scraped crew data
- [ ] 03-03: Build NBA pre-game injury report fetcher
- [ ] 03-04: Enforce training/inference injury code path separation

### Phase 4: Rest & Schedule Features
**Goal**: The feature pipeline captures fatigue signals — travel distance, back-to-back games, and season-segment context — for every game in the training set and in live inference
**Depends on**: Phase 2, Phase 3
**Requirements**: FR-3.1, FR-3.2, FR-3.3, FR-3.4, NFR-1, NFR-2
**Success Criteria** (what must be TRUE):
  1. The assembled feature table includes `days_rest` (integer days since last game) for both home and away teams, with back-to-back correctly identified as 1
  2. The assembled feature table includes `travel_miles` (geodesic distance between consecutive arenas) computed via geopy using a hardcoded arena coordinate dictionary
  3. A `cross_country_travel` binary flag (timezone change indicator) is present as a feature column
  4. A `season_month` integer feature (1-12) is present for every game, capturing season-segment context (opening, mid-season, playoff race)
  5. All four rest/schedule features are computed from existing game log data without any new API calls — no rate limit exposure
**Plans**: 3 plans

Plans:
- [ ] 04-01: Build days-rest and back-to-back features from game logs
- [ ] 04-02: Build travel distance and timezone-change features using geopy
- [ ] 04-03: Add season-segment context and integrate all schedule features into training pipeline

### Phase 5: ATS Model
**Goal**: A separate ATS classifier predicts whether a team covers the spread, value bets are identified when model probability diverges from market-implied odds, and results are backtested against closing lines over 500+ games
**Depends on**: Phase 2, Phase 3, Phase 4
**Requirements**: FR-5.1, FR-5.2, FR-5.3, FR-5.4, FR-5.5, NFR-1, NFR-3
**Success Criteria** (what must be TRUE):
  1. `game_ats_features.csv` exists as a separate file from `game_matchup_features.csv` and includes Vegas spread and implied moneyline probability as input columns — the win-probability model's training data contains neither
  2. An ATS classifier trained on `game_ats_features.csv` with expanding-window validation produces a `covers_spread` prediction for each upcoming game alongside the win-probability prediction
  3. Running the value-bet detector on any day's upcoming games outputs a list of games where the model's win probability differs from market-implied probability by more than the configured threshold
  4. Historical odds data is backfilled from The Odds API within free-tier quota limits — the quota audit and response-header check run before any batch backfill begins
  5. The ATS backtest report shows ROI, Closing Line Value, and hit rate computed against closing lines over at least 500 games — and the backtest script refuses to report results on fewer than 500 games
**Plans**: 4 plans

Plans:
- [ ] 05-01: Build ATS feature table with spread and implied probability inputs
- [ ] 05-02: Train ATS classifier with expanding validation splits
- [ ] 05-03: Build value-bet identification and odds backfill pipeline
- [ ] 05-04: Build ATS backtest harness with CLV and ROI reporting

## Progress

**Execution Order:**
Phases execute in dependency order: 1 → 2 → 3 → 4 → 5 (Phases 2 and 3 can overlap; Phase 4 needs both; Phase 5 needs all)

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation & Outputs | 4/4 | Complete | 2026-03-02 |
| 2. Modern Era Features | 3/3 | Complete   | 2026-03-02 |
| 3. External Data Layer | 0/4 | Not started | - |
| 4. Rest & Schedule Features | 0/3 | Not started | - |
| 5. ATS Model | 0/4 | Not started | - |
