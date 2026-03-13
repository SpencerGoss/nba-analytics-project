# NBA Analytics Project Overhaul — Design Spec

## Summary

Comprehensive overhaul of the NBA analytics project based on systematic 24-area review. Core goals: fix critical prediction bugs, redesign betting architecture for market-specific outputs, build a player prop prediction system, automate the pipeline, and improve the dashboard UX. User's primary interest is player prop betting.

---

## Phase 1: Critical Bug Fixes (P0)

### 1.1 Stale Feature Bug in predict_game()

**Problem:** Cross-season matchups use last season's Elo (can be 500+ pts wrong). `diff_elo` is 37.3% importance — 3 of 6 daily predictions affected.

**Fix in `src/models/ensemble.py` predict_game():**
- Always synthesize fresh feature rows with current-season data
- Filter matchup DataFrame to `latest_season` only (same pattern already applied in `margin_model.py` predict_margin())
- Call `get_current_elos()` and inject fresh Elo values into the synthesized row
- Recompute all `diff_*` columns from fresh home/away stats

### 1.2 predict_margin() Missing Elo Refresh

**Problem:** Unlike predict_game(), margin model does NOT call `get_current_elos()`. Uses stale Elo from most recent CSV row.

**Fix in `src/models/margin_model.py` predict_margin():**
- Add Elo refresh after row construction (lines 360-384)
- Import and call `get_current_elos()` from `src/features/elo.py`
- Overwrite `diff_elo`, `diff_elo_fast`, `diff_elo_momentum` with fresh values

### 1.3 fillna(0) Preempting Pipeline Imputer

**Problem:** `fillna(0)` in `margin_model.py:386`, `value_bet_detector.py:299,366,595`, and `player_performance_model.py` replaces NaN with 0 before the pipeline's `SimpleImputer(strategy="mean")` ever runs. Training uses mean-imputed values but inference uses 0 — systematic bias.

**Fix:** Remove all `.fillna(0)` calls on feature DataFrames at inference time. Let the pipeline's fitted `SimpleImputer` handle NaN with training-set means.

Affected files (audit all prediction paths):
- `src/models/margin_model.py` line 386: `.fillna(0)` -> `.reindex(columns=feat_cols)`
- `src/models/value_bet_detector.py` lines 299, 366, 595: remove `.fillna(0)`
- `src/models/player_performance_model.py`: remove `.fillna(0)` at inference
- `src/models/game_outcome_model.py`: check inference path for `.fillna(0)`
- Any other model file with `.fillna(0)` on feature DataFrames — run project-wide grep before implementing

### 1.4 ATS Model Still Loaded Despite Weight=0

**Problem:** `ensemble.py` loads and runs `ats_model.pkl` predictions even though `ATS_WEIGHT=0.0`. Wastes compute and risks silent failure if artifact missing.

**Fix in `src/models/ensemble.py`:**
- Guard ATS loading/prediction behind `if ATS_WEIGHT > 0`
- Set `ats_prob = 0.5` (neutral) when skipped
- Remove ATS from model disagreement checks

### 1.5 Season Code Type Inconsistency

**Problem:** Multiple files compare season codes as strings (`"202324" >= "202425"` fails lexicographically). Affects `margin_model.py`, `game_outcome_model.py`, `team_game_features.py`.

**Fix:** Standardize all season comparisons to integer: `int(str(season))`. Audit all files that filter by season. Key locations:
- `margin_model.py`: `MODERN_ERA_START`, `TEST_SEASONS`, `EXCLUDED_SEASONS` comparisons
- `game_outcome_model.py`: same pattern
- `team_game_features.py`: season filtering
- `value_bet_detector.py`: season filtering

### 1.6 Known Issues from CLAUDE.md

These are documented known issues that must be addressed:
- **Lineup features missing for 2025-26** — lineup_data.csv may not have current season data. Verify and backfill.
- **ATS features stop at 2024-25** — if ATS model is re-enabled in future, features need updating. Low priority while ATS_WEIGHT=0.
- **`player_absences.csv` written to `data/raw/`** — violates hard rule ("Never modify data/raw/"). Move write target to `data/processed/`.
- **`fetch_odds.py:322` hardcoded date `"2025-10-01"`** — should derive from current season start.
- **Row-count assertions after joins** — silent data drops when join keys don't match. Add `.shape[0]` checks after critical merges in feature engineering.

---

## Phase 2: Betting Architecture Redesign

### 2.1 BettingRouter (replaces NBAEnsemble for betting)

**Problem:** Current ensemble blends win probability + sigmoid(margin/15) into a single score. This destroys market-specific information — a blended 62% means nothing for spread betting.

**New architecture — `src/models/betting_router.py`:**

```
BettingRouter
  |- moneyline_output() -> calibrated P(home_win) from game_outcome_model
  |- spread_output()    -> P(cover) via normal CDF on (pred_margin - spread) / residual_std
  |- total_output()     -> P(over) via normal CDF (future: when total model exists)
  |- props_output()     -> player stat predictions with intervals (Phase 4)
```

**Key design decisions:**
1. **No blending for spread bets** — margin model used standalone via `P(cover) = norm.cdf((pred_margin - spread) / residual_std)`
2. **residual_std replaces MARGIN_NORM_FACTOR** — compute actual prediction error std (~10-11) from model training residuals, store in artifacts
3. **Model disagreement = no bet** — when win_prob says home wins but margin_pred says away wins, Kelly=0 (currently averaged, hiding the conflict)
4. **Calibrate after blending** (if any blending needed), not before — per ICLR 2021 research
5. **NBAEnsemble kept for backward compatibility** — BettingRouter wraps it, adds market-specific outputs. Phase 3.4 ensemble improvements (weight tuning, temperature scaling) modify the wrapped NBAEnsemble internals — these changes flow through to BettingRouter's moneyline output automatically. Spread output is unaffected (uses margin model directly).

**Interface:**
```python
class BettingRouter:
    def __init__(self, artifacts_dir="models/artifacts"):
        self.ensemble = NBAEnsemble(artifacts_dir)
        self.residual_std = self._load_residual_std()

    def moneyline(self, home, away, date=None) -> dict:
        """Returns {prob, confidence_tier, edge_vs_market, ev}"""

    def spread(self, home, away, spread, date=None) -> dict:
        """Returns {cover_prob, confidence_tier, edge, ev}"""

    def props(self, player_id, stat, line, date=None) -> dict:
        """Returns {over_prob, median, p25, p75, confidence_tier}
        NOTE: Stub returning NotImplementedError in Phase 2.
        Wired to prop models in Phase 4."""

    def confidence_tier(self, edge, model_agreement) -> str:
        """Returns 'Best Bet' | 'Solid Pick' | 'Lean' | 'Skip'"""
```

### 2.2 Confidence Tiers (Strict)

**User requirement:** Only "Best Bet" when truly confident. No jargon. Kelly/bet sizing is opt-in only.

**Tier definitions (calibrated on validation data):**

| Tier | Criteria | Expected Frequency |
|------|----------|--------------------|
| **Best Bet** | Edge > 8%, models agree on direction, historical win rate > 65% at this edge level (validated via walk-forward backtest from Phase 3.6) | ~5-10% of picks |
| **Solid Pick** | Edge > 4%, models agree on direction | ~20-25% of picks |
| **Lean** | Edge > 2%, slight model agreement or one model neutral | ~30-35% of picks |
| **Skip** | Edge < 2% OR models disagree on direction | ~30-40% of picks |

**Display rules:**
- Default view: tier label + plain English reason ("Strong edge, all models agree")
- No Kelly fractions, no "ensemble edge", no "Brier score" in primary display
- Tooltips for advanced users: show edge %, model probabilities, EV
- Opt-in toggle: "Show bet sizing" reveals Kelly fraction and suggested unit size
- Kelly tiers when enabled: 25% Kelly (high confidence), 12.5% (medium), 0% (disagreement)

**Bootstrapping on day 1:** Before walk-forward backtest data exists, use edge thresholds only (no historical win rate requirement). Tighten to full criteria after 100+ tracked games validate the tier boundaries.

### 2.3 Value Bet Detector Fixes

**Bug fixes (5):**
1. **fillna(0)** — remove (covered in 1.3)
2. **Elo refresh missing in live mode** (lines 340-366) — add `get_current_elos()` call
3. **Kelly uses fair odds instead of actual payout odds** (lines 430-469) — fix formula: `b = (decimal_odds - 1)`, not `b = model_prob / (1 - model_prob)`
4. **No Kelly cap** — add max 5% of bankroll regardless of Kelly output
5. **COMPOSITE_ATS_WEIGHT=0.4** (line 480) — hard-set default to 0.0 in code (currently env-configurable via `os.getenv("COMPOSITE_ATS_WEIGHT", "0.4")`; change default from "0.4" to "0.0" so ATS noise is excluded without requiring env var)

**Improvements:**
1. **EV calculation** — `EV = (model_prob / market_prob) - 1` instead of raw edge for bet ranking
2. **Odds-ratio devigging** — replace multiplicative no-vig method (~0.5pp more accurate for 2-outcome)
3. **VALUE_BET_THRESHOLD** — lower from 0.05 to 0.03 (pros use 2-3%)
4. **confidence_tier column** — add to output matching the new strict tiers
5. **CLV tracking per tier** — validate that Best Bet actually has higher CLV than Lean

### 2.4 CLV Tracker Fixes

- `conn.total_changes` -> `cursor.rowcount` (line 102, unreliable metric)
- Add moneyline CLV alongside spread CLV
- Track CLV by confidence tier (Best Bet / Solid Pick / Lean / Skip)
- `datetime.utcnow()` -> `datetime.now(timezone.utc)` (deprecated)
- Weekly CLV reporting: grouped by odds range, confidence tier, bet type
- Alert if CLV drops below 52% over 30-game window
- **Fix NULL closing_spread root cause:** `backfill_closing_lines()` in update.py Step 3b must run daily — if pipeline skips a day, closing lines are lost permanently (Pinnacle has no historical API). Pipeline automation (Phase 5) ensures this never misses.
- **Dependency:** CLV-by-tier tracking requires confidence tiers from Phase 2.2 to exist first. Implement basic CLV fixes (rowcount, utcnow, moneyline CLV) immediately; add tier-based tracking after 2.2 ships.

---

## Phase 3: Model Improvements

### 3.1 SHAP Analysis (Before Any Changes)

Run SHAP on current game outcome model to understand true feature contributions before modifying anything. Use `scientific-skills:shap` plugin.

**Output:** SHAP summary plot, top-20 features by mean |SHAP|, interaction effects. Save to `models/artifacts/shap_analysis/`.

### 3.2 Feature Engineering

**New orthogonal features** (informed by SHAP results):
- Opponent-adjusted net rating (net_rtg vs opponent avg)
- Form acceleration (EWMA slope — improving vs declining)
- Clutch net rating differential
- Pace mismatch (high-pace vs low-pace)
- Scoring concentration (Herfindahl index of player scoring)
- Schedule density quality (strength of schedule over last 10)
- Roster stability (games with same starting 5)

**Feature pruning:** Remove multicollinear features identified by SHAP (not just correlation).

### 3.3 Margin Model Improvements

- **Huber Loss (delta=3-5)** — replace MSE/MAE, de-weight garbage time blowouts
- **Pace interaction features** — `pace_ratio * elo_diff`, `pace_ratio * net_rtg_diff`
- **Four Factors differential** — eFG%, TOV%, OREB%, FT Rate diffs (a composite Four Factors feature was added in session 2026-03-11; verify scope and enhance if needed)
- **Segmented MAE tracking** — spread<3, 3-7, >7 separately
- **Store residual_std in artifacts** — needed for BettingRouter spread output
- **Late-season tanking exclusion** — exclude games where team >10 losses behind 8th seed

### 3.4 Ensemble Improvements

- **Grid search weights** — with ATS_WEIGHT=0, this is effectively 1-dimensional (game_outcome_weight determines margin_weight = 1 - game_outcome_weight). ~20 candidate values at 0.05 increments. If ATS is ever re-enabled, expands to 2D grid.
- **Grid search MARGIN_NORM_FACTOR** — test [5,10,15,20,25,30], pick lowest Brier
- **Data-driven confidence thresholds** — replace arbitrary 0.15/0.08 with validation-optimized values
- **Disagreement-based weighting** — models agree -> higher confidence; disagree -> more uncertain
- **Temperature scaling** — add as 3rd calibration option alongside Platt/Isotonic

### 3.5 Training Window Experiments

Test accuracy with different training windows:
- 2013+ (current) vs 2016+ vs 2018+
- Hypothesis: older data adds noise as play style has shifted
- Use walk-forward backtest to validate

### 3.6 Walk-Forward Backtest

Use `quantitative-trading:backtesting-frameworks` to validate that model edge translates to actual betting profit. Minimum 200+ games simulated with realistic vig.

### 3.7 Statistical Significance Testing

Use `scientific-skills:statsmodels` to test:
- Is 67.5% accuracy significantly different from 50% baseline? (yes, but quantify CI)
- Is ATS 55% real or noise? (need p-value)
- 600+ bets to confirm 55% win rate at p<0.05

---

## Phase 4: Player Prop System (User's Primary Interest)

### 4.1 Architecture: Two-Stage Prediction

**Stage 1: Minutes model**
- Separate model predicting minutes played
- Minutes explains ~65% of stat variance (R^2=0.65)
- Features: role (starter/bench), team pace, opponent pace, B2B, blowout risk, injury context
- Blowout risk adjustment: `expected_minutes = base * (1 - blowout_prob * 0.30)`
  - `blowout_prob` derived from the game spread (from `data/odds/game_lines.csv`): logistic function mapping abs(spread) to blowout probability, calibrated on historical data (games where starters sat 4th quarter)
  - 7pt spread = 20% blowout prob -> -6% minutes
  - 10pt spread = 35% -> -10%
  - 14pt spread = 55% -> -16%

**Stage 2: Per-minute stat models** (one per stat: PTS, REB, AST, 3PM)
- Predict per-36 rate, then scale by predicted minutes
- Features (priority order):
  - Tier 1 (80%+ variance): usage rate, FGA + eFG%, opponent DvP rank, EWMA recent form, pace adjustment
  - Tier 2 (15-20% more): B2B penalty, travel impact, teammate injury usage boost, blowout risk, home/away split, H2H rolling avg
  - Tier 3 (refinement): foul-proneness, recovery curve, load management flag

### 4.2 Quantile Regression

Predict 25th/50th/75th percentiles, not just mean. Enables:
- "80% chance 18-26 points" instead of just "22 points"
- Asymmetric over/under decisions
- Pinball loss function for training

### 4.3 Conformal Prediction Intervals

Distribution-free, finite-sample guarantee:
- Split 70% train / 30% calibration
- Compute residual quantiles on calibration set
- If 90% coverage set, true value falls in interval 90% of the time

### 4.4 Recency vs Season Weighting

- Default: 60% season avg + 40% L10 (EWMA alpha=0.15)
- Role change (trade, injury return): 40% season + 60% L5 (alpha=0.25)
- Early season (<30 games): 75% position avg + 25% player data
- Hot streak cap: max 50% recent weight

### 4.5 Model Candidates

- XGBoost (primary): n_estimators=500, max_depth=5, lr=0.05, subsample=0.8, objective=reg:huber
- LightGBM (secondary)
- Quantile Regression: 3 models (alpha=0.25, 0.50, 0.75)
- Ridge (baseline)
- Blend with weights inversely proportional to validation error

### 4.6 Pipeline Integration

**Training scripts:**
- `src/models/player_minutes_model.py` — minutes prediction model
- `src/models/player_stat_models.py` — PTS/REB/AST/3PM per-minute models (all in one file, shared training loop)

**Artifacts (saved to `models/artifacts/`):**
- `player_minutes_model.pkl`, `player_minutes_features.pkl`
- `player_pts_model.pkl`, `player_reb_model.pkl`, `player_ast_model.pkl`, `player_3pm_model.pkl`
- `player_stat_features.pkl` (shared feature list across stat models)
- `player_stat_residual_quantiles.pkl` (conformal prediction calibration residuals)

**update.py integration:**
- Step 4b (after model training): train/retrain prop models if new player_game_logs data
- Step 7 (builders): `build_props.py` calls prop models to generate `dashboard/data/props.json`

**Retraining cadence:** Weekly (Monday 4AM run), not daily — player stat distributions change slowly. Daily runs use existing model artifacts.

**BettingRouter wiring (Phase 4 completes Phase 2):**
- `BettingRouter.props()` loads player stat models + minutes model
- Returns `{over_prob, median, p25, p75, confidence_tier}` for any player/stat/line combo

### 4.7 Existing Data (Underutilized)

Already in the project:
- `player_game_logs.csv`: 1946-2026, ~1.5M rows
- `player_stats_advanced.csv`: per-36, pace adj, possessions
- `player_hustle_stats.csv`: contested shots, deflections
- `player_stats_clutch.csv`: close-game performance
- `player_stats_scoring.csv`: paint/3PT/FT distribution
- `player_positions.csv`: position for DvP matching
- `player_absences.csv`: injury/recovery tracking
- `lineup_data.csv`: rotation patterns, starter vs bench
- `player_props_lines.csv`: sportsbook lines (ground truth for evaluation)

---

## Phase 5: Pipeline Automation

### 5.1 Windows Task Scheduler

Three daily runs:
- **4:00 AM ET** — Full pipeline (post-game): fetch data, rebuild features, retrain if needed, generate predictions
- **11:30 AM ET** — Injuries + odds: fetch injury reports, update odds, rebuild picks
- **6:30 PM ET** — Pre-tip final: last odds refresh, final predictions, deploy dashboard

### 5.2 Pipeline Health Reporting

- `pipeline_report.json` generated after every run
- Dashboard health badge (green/yellow/red based on data staleness)
- Log to `logs/` directory with structured JSON logging (replace 170+ print() calls)

### 5.3 Infrastructure Improvements

- **Exponential backoff** in all API clients (replace flat 10s retries)
- **Row-count validation** on fetched data (reject partial fetches)
- **Resume capability** — track which steps completed, restart from failure point
- **Data quality gates** — validate pipeline outputs before they reach models (use `data-engineering:data-quality-frameworks`)

### 5.4 Builder Consolidation

- **Single builder registry** — extract from update.py Step 7, eliminate scheduler.py's duplicate list
- **7-phase dependency graph** with parallel execution within phases
- **Fix execution order bug:** build_performance runs before build_accuracy_history (dependency violation)
- **Fix scheduler.py reference** — verify `scripts/fetch_live_scores.py` path is correct (may reference wrong filename vs `build_live_scores.py`)

---

## Phase 6: Dashboard Improvements

### 6.1 Performance (High Impact)

1. **Tiered JSON loading** — only load Today tab data on page open; defer other tabs until activated. Cuts initial payload ~60-70%.
2. **Promise.allSettled** — replace Promise.all to prevent total failure when one JSON 404s. Show error state for failed tabs, render the rest.
3. **Paginate Players table** — 50 rows per page instead of 450+ at once
4. **Loading progress indicator** — "Loading 12/20..." counter during initial load
5. **Cache-busting** — add `?v=YYYYMMDD` param to JSON URLs
6. **Purge Plotly charts** on tab switch — call `Plotly.purge()` to prevent memory leaks
7. **Lazy load images** — add `loading="lazy"` to headshot img tags

### 6.2 Betting Tools UX

1. **Strict confidence tiers** in Picks tab — Best Bet / Solid Pick / Lean / Skip with calibrated thresholds
2. **"Best Bets" summary** — top 3 highest-edge plays prominent at top of Betting Tools
3. **Bankroll management** in Bet Tracker — starting bankroll field, current bankroll, ROI%
4. **Bet Tracker data persistence** — JSON import/export (localStorage is device-specific)
5. **Fix Sharp Money tab** — disclaim CLV NULL data or fix pipeline
6. **Props enhancements** — hit-rate history ("Over 25.5 in 7 of last 10"), trend arrows, correlation warnings
7. **Kelly opt-in toggle** — hidden by default, shows bet sizing when enabled
8. **Player detail -> Props deep link** — "View Props" button in player detail

### 6.3 General UX

1. **Pause ticker on hover** — `animation-play-state: paused` on :hover
2. **Persist filter state** in sessionStorage across tab switches
3. **Actionable empty states** — "Predictions generated daily at 10am ET" instead of "No picks available"
4. **H2H pre-populate** from today's matchups, one-click from game card

### 6.4 Code Quality

- Extract inline `style="color:..."` to 15-20 CSS utility classes
- Extract magic numbers to JS constants
- Long-term: split 8,889-line index.html into modules (low priority — works, no build step on GitHub Pages)

### 6.5 Accessibility

- Add `scope="col"` to table th elements
- Add `role="button"` and `tabindex="0"` to clickable divs
- Add secondary cues beyond color (icons/patterns for color-blind users)
- Add `aria-labels` to confidence meters
- Add focus trapping in modals

---

## Phase 7: Code Cleanup & Infrastructure

### 7.1 Dead Code Removal

Delete (with their test files):
- `scripts/build_dashboard.py`
- `scripts/export_dashboard_data.py`
- `scripts/generate_sample_dashboard_data.py`
- `scripts/package_for_colab.py`
- `scripts/build_player_props.py` (deprecated duplicate of `build_props.py`)
- `src/data/get_player_bio_stats.py` (never called in update.py)

### 7.2 File Size Reduction

Split `src/features/team_game_features.py` (1,385 lines, 2.3x over limit):
- `rolling_features.py` — EWMA, rolling windows
- `matchup_builder.py` — home/away merge, diff columns
- `home_away_parsing.py` — game log parsing
- `interaction_features.py` — cross-matchup interactions

### 7.3 Configuration Consolidation

Create `src/config.py` for shared constants:
- `TEST_SEASONS`, `MODERN_ERA_START`, `EXCLUDED_SEASONS`
- Conference/division team mappings (duplicated in build_standings.py and build_playoff_odds.py)
- `CURRENT_SEASON` derived from data (fix hardcoded `202526` in build_playoff_odds.py)

### 7.4 Odds Pipeline Consolidation

Three separate odds fetchers with different schemas:
- `src/data/get_odds.py`
- `src/data/get_historical_odds.py`
- `scripts/fetch_odds.py`

Consolidate devigging logic into one utility. Standardize on odds-ratio method for 2-outcome markets.

### 7.5 Logging

Replace 170+ `print()` calls across data fetchers with Python `logging` module:
- Structured JSON logging
- Configurable log levels (DEBUG for development, WARNING for production)
- Separate log files per pipeline run in `logs/`

### 7.6 CI/CD

- **Repurpose `daily_deploy.yml`** as CI-only: remove builders/commit/push, keep test runner. Trigger on push AND PR.
- **Add pyright** to CI pipeline
- Keep `deploy-pages.yml` as-is (working correctly)

### 7.7 Test Coverage

- Add integration test: 3-season mini-pipeline (ingest -> features -> train -> predict)
- Add tests for: NBAEnsemble.predict(), value_bet_detector live mode, prediction_store write/read
- Delete stale `test_build_player_props.py` with the dead script
- After feature overhaul, audit all test assertions for stale feature names

---

## Phase 8: Data & External Integrations

### 8.1 Line Movement Self-Capture

Log Pinnacle spreads every 2-3 hours on game days:
- 10AM (opening), 1PM, 4PM, 6PM (pre-tip)
- Store in `data/odds/line_snapshots.csv`
- Compute opening-to-closing movement
- Detect reverse line movement (strongest sharp money signal)

### 8.2 The Odds API (Future)

Free tier: 500 credits/month. Provides:
- Multi-book odds comparison
- Historical odds from mid-2020
- Player props from multiple books
- Decision: defer until prop system is built and validated

### 8.3 SQLite Unified Store (Future)

Replace scattered CSVs with SQLite as single source of truth:
- All raw data in normalized tables
- All processed data in materialized views
- Dashboard builders query SQLite directly
- Keep SQL Server sync for warehouse/analytics

---

## Implementation Order

```
Phase 1: Critical Bug Fixes          [1-2 sessions]
  |
  v
Phase 2: Betting Architecture        [2-3 sessions]
  |
  v
Phase 3: Model Improvements          [3-4 sessions]
  |   (SHAP first, then features, then retrain)
  v
Phase 4: Player Prop System           [6-8 sessions]
  |   (user's primary interest, biggest new capability — minutes model + 4 stat models + quantile regression + conformal intervals + pipeline integration)
  v
Phase 5: Pipeline Automation          [2-3 sessions]
  |
  v
Phase 6: Dashboard Improvements       [3-4 sessions]
  |   (can partially overlap with Phase 4-5)
  v
Phase 7: Code Cleanup                 [2-3 sessions]
  |   (can partially overlap with anything)
  v
Phase 8: External Integrations        [ongoing]
```

**Quick wins** that can be done anytime:
- Promise.allSettled (Phase 6.1)
- Dead code deletion (Phase 7.1)
- ATS weight guard (Phase 1.4)
- fillna(0) removal (Phase 1.3)
- Ticker pause on hover (Phase 6.3)

---

## Success Criteria

1. **Predictions use fresh data** — no cross-season stale features in any prediction path
2. **BettingRouter provides market-specific outputs** — separate probabilities for ML, spread, props
3. **Confidence tiers are strict** — "Best Bet" wins at 65%+ rate, verified by CLV tracking
4. **Player prop predictions exist** — PTS/REB/AST/3PM with quantile intervals
5. **Pipeline runs unattended** — 3x daily via Task Scheduler, health badge on dashboard
6. **Dashboard loads fast** — <3s initial load (tiered loading), no total failure on missing JSON
7. **CLV > 56%** over 100+ tracked games (minimum viability for real money)
8. **No jargon in default display** — plain English confidence labels, Kelly opt-in only
