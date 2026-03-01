# NBA Analytics — Model Enhancement & Pipeline Organization

## What This Is

An NBA game prediction system that predicts game outcomes and identifies value bets against sportsbook spreads. Built on a data pipeline that fetches from the NBA Stats API, engineers features, and trains ML models — focused on the modern era (2014+). The system also includes player performance projections (pts/reb/ast) and playoff odds simulation.

## Core Value

Identify games where the model's win probability meaningfully disagrees with Vegas lines — producing profitable against-the-spread picks over a full NBA season.

## Requirements

### Validated

- ✓ NBA Stats API data pipeline (20+ endpoints, daily refresh via `update.py`) — existing
- ✓ Raw → processed CSV preprocessing with column normalization — existing
- ✓ Team game feature engineering (rolling windows, strength of schedule, matchup differentials) — existing
- ✓ Player feature engineering (rolling stats, season priors, opponent context) — existing
- ✓ Game outcome classifier (GradientBoosting/RandomForest, ~65% accuracy) — existing
- ✓ Player performance regressors (pts/reb/ast per target) — existing
- ✓ Playoff odds Monte Carlo simulation — existing
- ✓ Walk-forward backtesting with expanding validation splits — existing
- ✓ Sportsbook odds integration via The Odds API — existing
- ✓ CLI prediction interface (`predict_cli.py`) — existing
- ✓ Historical backfill pipeline (`backfill.py`) — existing
- ✓ Scheduled daily updates via Windows Task Scheduler — existing

### Active

- [ ] Fix injury proxy features (missing_minutes, star_player_out currently broken/all-null)
- [ ] Add pace and 3-point era features (rolling pace, 3PT rate, possessions)
- [ ] Add rest and schedule features (back-to-backs, travel distance, days between games)
- [ ] Integrate official NBA injury reports (questionable/probable/out status)
- [ ] Scrape external data sources (Basketball Reference advanced stats, referee assignments, lineup data)
- [ ] Add ATS (against-the-spread) model as new prediction target
- [ ] Integrate sportsbook odds/lines as model features (not just comparison)
- [ ] Value bet identification system (flag games where model disagrees with Vegas)
- [ ] Focus training on modern era (2014+) to match current NBA style
- [ ] Store prediction results for historical accuracy tracking (web-ready)
- [ ] Serialize model outputs as JSON (not just pickle) for future web consumption
- [ ] Organize pipeline stages so each is clean and self-contained
- [ ] Document pipeline stage order and dependencies clearly

### Out of Scope

- Web application / dashboard — future milestone, but current architecture should not block it
- Real-time in-game predictions — not in scope (pregame only)
- Player prop betting model — defer until game outcome ATS model is proven
- Shot chart ingestion — 3-4 hour runtime, separate concern from model accuracy
- Mobile app — future milestone
- Unit test coverage — important but separate effort from model improvement

## Context

**Existing codebase:** Brownfield project with working pipeline. See `.planning/codebase/` for full analysis.

**Current model performance:**
- Game outcome: ~65-66% accuracy overall, drops to 64% on post-2014 data
- Backtest shows accuracy degradation in modern 3-point era
- Injury features are defined in code but silently absent/null — player availability has zero effect on current predictions

**Known issues (from CONCERNS.md):**
- Injury proxy features broken (all-null, never reach the model)
- Calibrated model saved but never loaded in production
- Player backtest stops at 2015-16 (missing 10 years of data)
- Preprocessing rebuilds all CSVs daily (inefficient but functional)
- No unit tests across the pipeline

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
*Last updated: 2026-03-01 after initialization*
