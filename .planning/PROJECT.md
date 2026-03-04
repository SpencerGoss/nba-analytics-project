# NBA Analytics — Model Enhancement & Pipeline Organization

## What This Is

An NBA game prediction system that predicts game outcomes and identifies value bets against sportsbook spreads. Built on a data pipeline that fetches from the NBA Stats API, engineers features, and trains ML models — focused on the modern era (2014+). The system also includes player performance projections (pts/reb/ast) and playoff odds simulation.

## Core Value

Identify games where the model's win probability meaningfully disagrees with Vegas lines — producing profitable against-the-spread picks over a full NBA season.

## Current Milestone: v2.0 — Data Expansion & Model Intelligence

**Goal:** Push game outcome accuracy to 68%+ and ATS above vig breakeven (52.4%+) by adding real lineup/injury data, new free data sources, and ensemble model improvements.

**Target features:**
- Kaggle historical injury dataset (replace proxy with actual player absence records)
- pbpstats lineup tracking (who is actually on the floor — the strongest prediction signal)
- BallDontLie API (structured injury + stats backup)
- Calibrated model wired into all production inference paths
- Retrained game outcome + ATS models with new feature set
- ATS ensemble above vig breakeven

## Requirements

### Validated

- ✓ NBA Stats API data pipeline (20+ endpoints, daily refresh via `update.py`) — v1.0
- ✓ Raw → processed CSV preprocessing with column normalization — v1.0
- ✓ Team game feature engineering (rolling windows, strength of schedule, matchup differentials) — v1.0
- ✓ Player feature engineering (rolling stats, season priors, opponent context) — v1.0
- ✓ Game outcome classifier (66.8% accuracy on 2023-25 holdout) — v1.0
- ✓ Player performance regressors (pts/reb/ast per target) — v1.0
- ✓ Playoff odds Monte Carlo simulation — v1.0
- ✓ Walk-forward backtesting with expanding validation splits — v1.0
- ✓ Sportsbook odds integration via The Odds API — v1.0
- ✓ ATS model with value-bet detection (51.2% raw, +1.28% ROI on filtered) — v1.0
- ✓ Prediction store (WAL SQLite + JSON snapshots) — v1.0
- ✓ Static web dashboard (Chart.js) — v1.0
- ✓ Shared API client (retry logic centralized) — v1.0 gap closure
- ✓ Incremental preprocessing (mtime-based) — v1.0 gap closure
- ✓ Data integrity validation framework — v1.0 gap closure

### Active (v2.0)

- [ ] Integrate Kaggle NBA injury dataset as actual absence features (replace injury proxy)
- [ ] Add pbpstats lineup tracking data (on-court lineup strength features)
- [ ] Add BallDontLie API as data source (structured injuries + stats)
- [ ] Wire calibrated model into fetch_odds.py (uncalibrated model used there now)
- [ ] Retrain game outcome model with new injury + lineup features (target: 68%+)
- [ ] Push ATS model above 52.4% vig breakeven (currently 51.4% holdout)
- [ ] Add lineup net rating differentials as matchup features
- [ ] Ensemble/stack game outcome + ATS models for value-bet detection

### Out of Scope

- Web application — future v3.0 milestone
- Real-time in-game predictions — pregame only
- Player prop betting model — defer until ATS model is profitable
- Shot chart ingestion — 3-4 hour runtime, not worth it yet
- Mobile app — future milestone

### Out of Scope

- Web application / dashboard — future milestone, but current architecture should not block it
- Real-time in-game predictions — not in scope (pregame only)
- Player prop betting model — defer until game outcome ATS model is proven
- Shot chart ingestion — 3-4 hour runtime, separate concern from model accuracy
- Mobile app — future milestone
- Unit test coverage — important but separate effort from model improvement

## Context

**Existing codebase:** Brownfield project with working pipeline. See `.planning/codebase/` for full analysis.

**Current model performance (v1.0 final):**
- Game outcome: 66.8% accuracy on 2023-24/2024-25 holdout
- ATS: 51.2% raw (51.41% in v2 experiments), +1.28% ROI on value-bet filtered — below 52.4% vig breakeven
- Injury features flow correctly through pipeline (ranked 11-21 in SHAP) but use proxy estimates not real absence data
- Lineup data not used at all — strongest predictor of actual game outcomes

**Known gaps carried to v2.0:**
- Calibrated model not wired into fetch_odds.py (uses uncalibrated probabilities)
- ATS 1.27pp below vig breakeven
- Injury features use rolling-average proxy, not actual player availability records
- No lineup/rotation data (pbpstats can provide this)

**Key signal sources not yet tapped:**
- Official NBA injury reports (strongest predictor of game outcomes)
- Referee tendencies (pace, foul rate by crew)
- Lineup/rotation data (who's actually playing together)
- Travel/schedule fatigue (cross-country back-to-backs)
- Vegas lines as features (market consensus is highly informative)

**Future milestone context:**
The next major milestone after model improvement will be a web platform (data exploration + daily predictions + model transparency). Current work should produce web-friendly outputs (JSON predictions, stored results, queryable data) without building the web layer itself.

## Constraints

- **Data sources**: Free/public sources only (NBA API, Basketball Reference, official injury reports, The Odds API free tier)
- **Training era**: Modern era focus (2014+ seasons) for game outcome model
- **Compute**: Local machine (Windows 11 Pro), no cloud compute — training must complete in reasonable time
- **API rate limits**: NBA Stats API throttles at ~1 req/sec; The Odds API has monthly request limits on free tier
- **Architecture**: Must remain web-ready — model outputs stored as JSON, prediction history persisted, data queryable via SQLite

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Modern era only (2014+) for training | Post-2014 NBA plays fundamentally differently (pace, 3PT volume); mixed-era training hurts modern accuracy | — Pending |
| ATS model alongside win probability | Beating the spread requires different signal than predicting winners; separate model allows targeted optimization | — Pending |
| Vegas lines as features | Market consensus is the strongest public predictor; using it as input rather than just comparison target unlocks value-bet identification | — Pending |
| Keep pipeline stages separate | User prefers clear, self-contained stages over monolithic orchestration; easier to debug and experiment | — Pending |
| JSON-serializable outputs | Future web platform needs to consume predictions; designing for this now avoids costly refactor later | — Pending |
| Free data sources only | No paid data subscriptions; Basketball Reference scraping, NBA injury reports, and The Odds API free tier | — Pending |

---
*Last updated: 2026-03-04 after v2.0 milestone start*
