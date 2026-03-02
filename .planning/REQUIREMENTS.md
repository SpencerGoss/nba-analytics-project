# Requirements — NBA Analytics Model Enhancement

**Milestone:** v2 — Model Enhancement & ATS Betting
**Date:** 2026-03-01
**Status:** Active

---

## Success Criteria

The milestone is complete when:

1. Game outcome model achieves **>66% accuracy on 2014+ holdout seasons** (currently 64-66%)
2. ATS model identifies value bets that are **profitable over 500+ game backtest** against closing lines
3. Prediction history is **stored and accumulating** in web-ready format (JSON + SQLite)
4. All pipeline stages are **self-contained and documented** — each runs independently with clear inputs/outputs

---

## Functional Requirements

### FR-1: Fix Broken Features

- **FR-1.1:** Fix injury proxy join so `missing_minutes` and `star_player_out` features reach the model with non-null values
- **FR-1.2:** Wire calibrated model artifact (`game_outcome_model_calibrated.pkl`) into `predict_cli.py` inference path
- **FR-1.3:** Validate all feature columns have <95% null rate before model training; fail loudly if exceeded

### FR-2: Modern Era Features

- **FR-2.1:** Add pace-normalized efficiency features from `team_stats_advanced` table: ORtg, DRtg, net rating, eFG%, TS% as rolling averages (5/10/20 game windows)
- **FR-2.2:** Add pace (possessions/game) as rolling feature
- **FR-2.3:** Add turnover rate per possession (not raw count) as rolling feature
- **FR-2.4:** Add Four Factors differential composite (eFG%, TOV%, ORB%, FT rate) as matchup feature
- **FR-2.5:** Restrict game outcome model training to 2014+ seasons; exclude 2019-20 bubble and 2020-21 shortened seasons

### FR-3: Rest & Schedule Features

- **FR-3.1:** Compute days-between-games for each team (back-to-back = 1 day)
- **FR-3.2:** Compute travel distance between consecutive game arenas using static arena coordinate dict + geopy
- **FR-3.3:** Add timezone change flag (cross-country travel indicator)
- **FR-3.4:** Add season-segment context (month of season as feature)

### FR-4: External Data Integration

- **FR-4.1:** Scrape referee crew assignments from Basketball Reference box score pages
- **FR-4.2:** Compute referee crew foul rate (FTA/game, pace impact) as rolling feature
- **FR-4.3:** Integrate official NBA pre-game injury reports (Questionable/Probable/Out status) for inference
- **FR-4.4:** Maintain two injury code paths: historical proxy for training, live report for inference — never mix

### FR-5: ATS Model

- **FR-5.1:** Create separate `game_ats_features.csv` that extends matchup features WITH Vegas spread and implied moneyline probability
- **FR-5.2:** Train ATS classifier (target: `covers_spread` binary) using expanding validation splits
- **FR-5.3:** Build value-bet identification: flag games where win-probability model disagrees with implied odds by configurable threshold
- **FR-5.4:** Backfill historical odds data from The Odds API (respect free tier quota limits)
- **FR-5.5:** Backtest ATS model against closing lines over 500+ games; report ROI, CLV, and hit rate

### FR-6: Prediction Store & Web-Ready Outputs

- **FR-6.1:** Create append-only SQLite prediction store (`predictions_history.db`) with game predictions, player projections, and model metadata
- **FR-6.2:** Enable WAL mode on prediction store for concurrent read/write
- **FR-6.3:** Export daily JSON snapshot of predictions for web consumption
- **FR-6.4:** Store prediction results with timestamps for historical accuracy tracking
- **FR-6.5:** Serialize model metadata (feature importances, thresholds, training dates) as JSON alongside pickle artifacts

### FR-7: Pipeline Organization

- **FR-7.1:** Each pipeline stage (fetch, preprocess, features, train, predict) runs independently with documented inputs/outputs
- **FR-7.2:** External data scrapers follow existing `src/data/get_*.py` module pattern
- **FR-7.3:** `update.py` remains thin — each capability is one-line call to its module
- **FR-7.4:** Document pipeline stage order, dependencies, and expected runtime in a pipeline reference

---

## Non-Functional Requirements

### NFR-1: Data Integrity

- All rolling features use `.shift(1)` before `.rolling()` to prevent lookahead bias
- Assert `result.shape[0] > 0` after every DataFrame join
- Log null rates per column post-feature-assembly

### NFR-2: Performance

- Daily update pipeline completes in <15 minutes (current: ~10 min)
- Model training completes in <15 minutes on local machine
- Basketball Reference scraping respects 3-second delay between requests

### NFR-3: Web Readiness

- All model outputs serializable to JSON
- Prediction history queryable via SQLite
- No architecture decisions that block a future web frontend

### NFR-4: Free Data Only

- All data sources are free/public (NBA API, Basketball Reference, The Odds API free tier, official NBA injury reports)
- No paid subscriptions or premium API tiers required

---

## Out of Scope

- Web application / dashboard (future milestone)
- Real-time in-game predictions
- Player prop betting model
- Shot chart ingestion (3-4 hour runtime)
- Unit test coverage (separate effort)
- Mobile app
- Custom Elo system
- Play-by-play features
- Social media / NLP signals

---

## Dependencies

| Requirement | Depends On |
| ----------- | ---------- |
| FR-2 (efficiency features) | FR-1 (broken features fixed first) |
| FR-4 (external data) | New libraries: beautifulsoup4, lxml |
| FR-3 (travel distance) | New library: geopy |
| FR-5 (ATS model) | FR-1 + FR-2 (stable feature pipeline), FR-6.1 (prediction store) |
| FR-5.4 (odds backfill) | Audit of Odds API free tier historical depth |
| FR-4.1 (referee scrape) | Spike test of Basketball Reference HTML selectors |

---

## Traceability

| Requirement | Phase | Status |
| ----------- | ----- | ------ |
| FR-1.1 | Phase 1: Foundation & Outputs | Pending |
| FR-1.2 | Phase 1: Foundation & Outputs | Pending |
| FR-1.3 | Phase 1: Foundation & Outputs | Pending |
| FR-6.1 | Phase 1: Foundation & Outputs | Pending |
| FR-6.2 | Phase 1: Foundation & Outputs | Pending |
| FR-6.3 | Phase 1: Foundation & Outputs | Pending |
| FR-6.4 | Phase 1: Foundation & Outputs | Pending |
| FR-6.5 | Phase 1: Foundation & Outputs | Pending |
| FR-7.1 | Phase 1: Foundation & Outputs | Pending |
| FR-7.2 | Phase 1: Foundation & Outputs | Pending |
| FR-7.3 | Phase 1: Foundation & Outputs | Pending |
| FR-7.4 | Phase 1: Foundation & Outputs | Pending |
| NFR-1 | Phase 1: Foundation & Outputs (primary) | Pending |
| NFR-2 | Phase 1: Foundation & Outputs (primary) | Pending |
| NFR-3 | Phase 1: Foundation & Outputs (primary) | Pending |
| FR-2.1 | Phase 2: Modern Era Features | Pending |
| FR-2.2 | Phase 2: Modern Era Features | Pending |
| FR-2.3 | Phase 2: Modern Era Features | Pending |
| FR-2.4 | Phase 2: Modern Era Features | Pending |
| FR-2.5 | Phase 2: Modern Era Features | Pending |
| FR-4.1 | Phase 3: External Data Layer | Pending |
| FR-4.2 | Phase 3: External Data Layer | Pending |
| FR-4.3 | Phase 3: External Data Layer | Pending |
| FR-4.4 | Phase 3: External Data Layer | Pending |
| NFR-4 | Phase 3: External Data Layer (primary) | Pending |
| FR-3.1 | Phase 4: Rest & Schedule Features | Pending |
| FR-3.2 | Phase 4: Rest & Schedule Features | Pending |
| FR-3.3 | Phase 4: Rest & Schedule Features | Pending |
| FR-3.4 | Phase 4: Rest & Schedule Features | Pending |
| FR-5.1 | Phase 5: ATS Model | Pending |
| FR-5.2 | Phase 5: ATS Model | Pending |
| FR-5.3 | Phase 5: ATS Model | Pending |
| FR-5.4 | Phase 5: ATS Model | Pending |
| FR-5.5 | Phase 5: ATS Model | Pending |

---

Requirements derived from research synthesis (.planning/research/SUMMARY.md)
