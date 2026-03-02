# Research Summary

**Project:** NBA Analytics — Model Enhancement & Pipeline Organization
**Domain:** Sports ML pipeline, ATS betting analytics, NBA game prediction
**Researched:** 2026-03-01
**Confidence:** HIGH

---

## Key Findings

Top findings ranked by impact on roadmap decisions:

1. **Fix broken foundations before adding anything new.** Injury features are defined, engineered, and passed to the model — but are silently all-null due to a join key mismatch. The calibrated model artifact is saved but never loaded. Both bugs make current predictions worse than they need to be. Any new features built on top of this broken state will be evaluated against a degraded baseline. Fix first.

2. **The training data era is the most important hyperparameter.** Training on 1996-2026 actively hurts modern accuracy. The model achieves 67-69% on 2005-2015 data and only 64-66% on the modern era it is actually deployed to predict. Filtering to 2014+ is not a nice-to-have — it is the highest-leverage single change available before adding any new features.

3. **Pace-normalized efficiency metrics (ORtg, DRtg, net rating) are the most important missing features.** These are table stakes in every serious NBA prediction model and are NOT in the current feature pipeline — even though the underlying data (team_stats_advanced) already exists in the database. This is a data wiring gap, not a data acquisition problem.

4. **Vegas lines must be treated as a model input for the ATS model, not just a comparison target.** The win-probability model should remain blind to lines. The ATS model requires the spread as an input feature. These are separate models with separate feature tables. Conflating them destroys the value-bet signal.

5. **Start storing prediction history now, not at the web milestone.** Prediction accuracy can only be tracked for games predicted after the store is created. Every day without a prediction store is a day of lost calibration history. The outputs layer is low complexity and should be built in Phase 1 alongside bug fixes.

6. **Only 3 new libraries are justified.** The existing stack handles everything else: ATS modeling (sklearn), JSON serialization (stdlib), injury data (existing nba_api + proxy logic), pace/rest features (pandas). New additions: `beautifulsoup4` + `lxml` for Basketball Reference scraping, `geopy` for travel distance.

7. **Lookahead bias and training/inference feature drift are the two most likely silent failure modes.** Both produce inflated backtests that collapse in live deployment. They must be guarded against architecturally — not just at code-review time.

8. **The ATS model backtest requires closing line evaluation and 500+ games.** Opening line backtests are misleading. Any ATS result reported against fewer than 500 games cannot be distinguished from noise. Build the evaluation harness correctly from the start or ATS results will be meaningless.

---

## Recommended Stack Additions

The existing stack (Python, pandas, numpy, scikit-learn, shap, nba_api, requests) covers all modeling needs. Three targeted additions only:

| Library | Version | Purpose | Rationale |
|---------|---------|---------|-----------|
| `beautifulsoup4` | `>=4.12.0` | Scrape Basketball Reference HTML tables | Standard, mature tool; direct scraping is more reliable than `sportsipy` wrapper (minimally maintained as of 2025) |
| `lxml` | `>=4.9.0` | Fast HTML parser backend for BeautifulSoup | Significantly faster than `html.parser`; sports-reference pages are large |
| `geopy` | `>=2.4.0` | Calculate arena-to-arena travel distance | Geodesic distance is a better fatigue signal than binary back-to-back flag; arena coordinates are static (hardcode dict, no API calls) |

**What NOT to add:** XGBoost/LightGBM (existing GBM at 65%+ is competitive on 10k games; revisit only if accuracy stalls), neural nets (not justified for tabular sports data), ORMs (SQLite via pandas is sufficient), async HTTP (NBA API rate limits make parallel requests counterproductive).

Install: `pip install beautifulsoup4>=4.12.0 lxml>=4.9.0 geopy>=2.4.0`

---

## Feature Priority Matrix

### Table Stakes (must have for ATS model to work correctly)

| Feature | Status | Source |
|---------|--------|--------|
| Injury proxy fix (`missing_minutes`, `star_player_out`) | BROKEN — fix, not new build | Existing `injury_proxy.py` |
| Calibrated model wired to production inference | BROKEN — wire, not new build | Existing `calibration.py` artifact |
| ORtg / DRtg / net rating rolling | MISSING — data in DB, needs wiring | `team_stats_advanced` table |
| Pace (possessions/game) rolling | MISSING — data in DB, needs wiring | `team_stats_advanced` table |
| eFG% / TS% rolling | MISSING — data in DB, needs wiring | `team_stats_advanced` table |
| Turnover rate (per possession, not raw count) | PARTIAL — exists as raw count | `team_game_logs` table |
| Vegas moneyline → implied probability (as feature) | MISSING — odds exist, not as input | `data/odds/game_lines.csv` |

### High-Value (likely to improve accuracy meaningfully)

| Feature | Complexity | Source |
|---------|-----------|--------|
| Travel distance + timezone change for rest features | Medium | `geopy` + static arena dict |
| Referee crew foul rate (pace, FTA/game) | Medium | Basketball Reference scrape |
| Four Factors differential (eFG%, TOV%, ORB%, FT rate) | Medium | Derived from existing stats |
| Historical ATS cover rate per team (rolling 20-game) | Medium | Requires historical spread backfill |
| Season-segment context (month of season) | Low | Derived from game date |
| First-game-back flag after injury absence | Medium | Derived from injury proxy + game logs |

### Nice-to-Have (add if time permits)

| Feature | Complexity | Notes |
|---------|-----------|-------|
| Official pre-game NBA injury report (Questionable/Out status) | High | Separate code path from historical proxy required |
| Head-to-head ATS history | Medium | Requires historical spread data |
| Line movement (open to close delta) | High | Requires multiple Odds API snapshots per day |

### Anti-Features (explicitly do not build)

| Anti-Feature | Reason |
|-------------|--------|
| Play-by-play / shot chart features | 11k API call pipeline, 3-4 hours runtime; eFG% rolling is sufficient proxy |
| Social media / injury news NLP | Noisy, real-time scraping required; official injury report is authoritative |
| Full franchise historical ATS record (>3 seasons) | Roster changes invalidate 10-year ATS patterns; use rolling 20-game only |
| Clutch stats as primary features | Data starts 2004-05; truncates training data with low SHAP return |
| In-game / quarter-level win probability | Out of scope; project is pregame only |
| Custom Elo system | Rolling net rating is a functional Elo proxy; new system adds complexity without proven gain |
| Salary / contract year effects | Noise at game level; injury proxy captures availability better |
| Player prop model features (beyond pts/reb/ast) | Defer until game outcome ATS model is proven |

---

## Architecture Recommendations

### How the Pipeline Should Evolve

The current five-layer ETL+ML pipeline is well-structured. Three gaps need to close:

1. **External data has no home.** Create `src/data/external/` for scrapers (Basketball Reference, injury reports) with matching `data/raw/external/` storage. Mirror the existing `src/data/get_*.py` module pattern — one file per source, `fetch_*()` and `save_*()` functions.

2. **ATS model needs a separate feature table.** `game_matchup_features.csv` feeds the win-probability model (no odds inputs). A new `game_ats_features.csv` extends it with sportsbook spread and implied probability — exclusively for the ATS model. This architectural boundary enforces Rule: the win-probability model must never receive Vegas lines as input (destroys the value-bet signal).

3. **Outputs must become a persistent layer.** Predictions go to stdout today. Create `src/outputs/prediction_store.py` (append-only SQLite) and `src/outputs/json_export.py` (daily JSON snapshot). Start accumulating prediction history immediately — this is the web platform's primary data source.

### Build Order (Hard Dependencies)

```
Phase 1 — Foundation fixes (must come first)
  Fix injury_proxy.py join alignment
  Wire calibrated model into predict_cli.py
  Add ORtg/DRtg/pace/eFG% to team feature pipeline
  Create prediction_store.py + json_export.py (start accumulating history)
  Restrict training to 2014+ era

Phase 2 — External data layer (unlocks richer features)
  src/data/external/bref_scraper.py (referee data, Four Factors)
  src/data/external/injury_report.py (live pre-game reports)
  src/processing/external_preprocessing.py

Phase 3 — ATS model (requires Phase 1 feature pipeline stability)
  src/features/odds_features.py → game_ats_features.csv
  src/models/ats_model.py (target: covers_spread)
  src/models/value_bets.py (flags model vs market disagreement)
  Historical odds backfill (quota-guarded)

Phase 4 — Differentiators (external data from Phase 2 required)
  Travel distance features (geopy + arena dict)
  Referee foul-rate features
  Four Factors differential composite
  Season-segment / load management context
```

**Critical path:** Phase 1 → Phase 2 → Phase 3. Phase 1 output infrastructure (prediction store) runs in parallel and should start immediately.

### Key Architectural Rules

- **Separate win-probability model and ATS model feature tables.** Spread as input is ATS-only.
- **Two injury code paths, not one.** Historical proxy (game log gaps) for training; live official report for inference. Never mix these — mixing creates lookahead leakage.
- **Keep `update.py` thin.** Each new capability lives in its own module; `update.py` calls it in one line. A monolithic 300-line orchestrator is the anti-pattern.
- **Enable SQLite WAL mode** on `predictions_history.db` at creation to allow web reads concurrent with update writes.

---

## Critical Pitfalls to Avoid

**1. Lookahead bias in rolling features** (HIGH confidence)
Rolling averages that accidentally include the current game produce inflated backtests (68-72%) that collapse in production. Prevention: sort by date, use `.shift(1)` before `.rolling()`, add date assertions that features for game on day D use only rows where date < D. Add a synthetic fixture test.

**2. Silent null features invalidating model inputs** (HIGH confidence, active bug)
Injury features compile without errors but arrive all-null at the model due to join key mismatch. The imputer fills with column mean, making the feature invisible. Prevention: assert `result.shape[0] > 0` after every join; log null rates per column post-assembly; flag any column >95% null as an error.

**3. Backtesting ATS against opening line instead of closing line** (HIGH confidence)
Opening-line backtests show fake profitability. The closing line is what you actually bet against. Prevention: enforce closing-line evaluation in the backtesting harness from day one; track Closing Line Value (CLV); do not report ATS results until compared to closing lines over 500+ games.

**4. Training on mixed eras (1996–2026)** (HIGH confidence, active issue)
Pre-2014 NBA pace and 3PT patterns dilute modern signal. Model achieves 67-69% on old data, 64-66% on modern — the old data is not helping. Prevention: filter training to 2014+ as the first change before any new features; explicitly exclude 2019-20 bubble and 2020-21 shortened seasons.

**5. Uncalibrated probabilities flowing into value-bet logic** (HIGH confidence, active bug)
`game_outcome_model_calibrated.pkl` is saved but `predict_cli.py` never loads it. Overconfident raw probabilities inflate apparent edge. Prevention: wire the calibrated model into all inference paths before any ATS model development; add an assertion that raises loudly if the calibrated artifact is missing; verify with reliability diagrams.

---

## Unresolved Questions

| Question | How to Handle |
|----------|--------------|
| **Historical spread data availability.** The Odds API free tier has limited historical depth. How many seasons of closing lines are available via free tier backfill? | Audit at start of Phase 3: run a small backfill test, check response headers for available date range before committing to ATS model design |
| **Injury proxy join key.** Exactly why do injury features arrive null? Is it game_id format mismatch, date offset, or empty source data? | Trace `injury_proxy.py` join logic before writing any new injury feature code — the fix may be one line or may require restructuring |
| **Referee data completeness on Basketball Reference.** Referee crew assignments exist in box score pages, but scraping them at scale requires validating the selector against current HTML structure. | Spike scrape 10 games before committing to referee feature in roadmap |
| **geopy 2.4.x API shape.** Travel distance approach is sound but the specific geopy API calls haven't been verified against the current version. | Verify with `pip show geopy` and a two-line test before building schedule_features.py |
| **Closing line value in backtests.** Do sufficient closing-line snapshots exist in `data/odds/` from the current Odds API integration to run a meaningful ATS backtest? | Check `data/odds/game_lines.csv` row count and date range before Phase 3 planning |

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Additive only; all recommendations based on direct codebase inspection; only 3 new libraries, all mature |
| Features | HIGH | Table stakes grounded in established NBA analytics (Dean Oliver Four Factors); active bugs confirmed by CONCERNS.md |
| Architecture | HIGH | Entirely derived from codebase inspection; no speculative claims; patterns mirror existing code structure |
| Pitfalls | HIGH | Top 5 pitfalls all have direct codebase evidence or are high-confidence domain fundamentals; no web search needed |

**Overall confidence: HIGH**

All four research files were grounded primarily in direct codebase analysis rather than web sources. The main uncertainty is operational (API rate limits, current HTML structure of Basketball Reference, exact version of geopy API) — not architectural or modeling.

---

## Sources

### Primary (HIGH confidence — direct codebase analysis)
- `src/features/team_game_features.py`, `src/features/injury_proxy.py`, `src/features/player_features.py` — feature inventory and broken state confirmed
- `src/models/calibration.py`, `src/models/predict_cli.py` — calibration artifact/inference disconnect confirmed
- `src/models/game_outcome_model.py`, `update.py`, `scripts/fetch_odds.py` — architecture current state
- `.planning/codebase/CONCERNS.md` — active bug list (injury nulls, calibration wire-up, player backtest cutoff)
- `docs/model_advisor_notes.md` — Proposals 1-10 (pace features, referee data, minutes projection)
- `database/nba.db` schema — 18 tables; `team_stats_advanced` confirmed to contain ORtg, DRtg, Pace, eFG%, TS%

### Secondary (MEDIUM confidence — training knowledge through August 2025)
- Dean Oliver, "Basketball on Paper" (2004) — Four Factors framework; ORtg/DRtg definitions; foundational NBA analytics literature
- Sports betting ML pattern library — closing line value methodology; ATS sample size requirements; era regime change effects
- The Odds API documentation — rate limits and monthly quota structure for free tier
- `beautifulsoup4`, `lxml`, `geopy` library documentation — version pins and API shapes

### Tertiary (LOW confidence — needs runtime verification)
- Basketball Reference HTML table selector structure — stable but should be verified against live pages before scraping
- `sportsipy`/`sportsreference` package current maintenance status — assessed from training knowledge; verify at https://pypi.org/project/sportsipy/ before ruling out

---

*Research completed: 2026-03-01*
*Ready for roadmap: yes*
