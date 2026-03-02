# Domain Pitfalls: NBA Prediction & ATS Betting Analytics

**Domain:** NBA game prediction, against-the-spread betting models, sports ML pipelines
**Researched:** 2026-03-01
**Confidence:** HIGH (grounded in codebase audit + domain knowledge; WebSearch unavailable this session)

---

## Critical Pitfalls

Mistakes that cause rewrites, silent model failures, or systematically wrong predictions.

---

### Pitfall 1: Lookahead Bias in Rolling Features

**What goes wrong:** Rolling averages, form metrics, and season totals are computed using data that would not have been available at game time. For example, a 10-game rolling average for game N accidentally includes game N itself, or post-game box scores are merged with pre-game prediction windows.

**Why it happens:** Pandas `.rolling()` with `min_periods=1` over an unsorted or incorrectly indexed DataFrame silently includes future rows. Date-based joins between game logs and feature tables frequently have off-by-one errors (game day vs. game day+1 for when stats are published).

**Consequences:** Backtests show inflated accuracy (68-72%) that evaporates in live deployment. The model appears to learn patterns that don't exist, giving false confidence in the pipeline.

**Warning signs:**
- Backtest accuracy is suspiciously high (>70%) on a dataset where Vegas already prices at 60-62% correct)
- Feature importance shows raw box score stats (points scored in last game) as top predictors
- Model accuracy collapses immediately when deployed live

**Prevention:**
- Sort by date before every rolling operation; use `.shift(1)` before `.rolling()` to exclude current game
- Add explicit date assertions: rolling feature for game on date D must only use rows where date < D
- Test with a synthetic fixture: inject a known outcome on day X, verify it does not appear in features for games on day X

**Phase to address:** Feature engineering phase (any phase that touches `team_game_features.py` or `player_features.py`). Add assertions to feature pipeline before model training phases.

**Confidence:** HIGH — this is the most common ML sports prediction error and directly applies to the rolling window code in `src/features/team_game_features.py`.

---

### Pitfall 2: Silent Feature Nulls Invalidating Model Inputs

**What goes wrong:** Feature columns are defined in code and passed to the model, but are silently all-null (or near-null) due to broken upstream joins. The model's imputer fills them with the column mean, making the feature invisible — it carries zero signal and the model trains as if it doesn't exist.

**Why it happens:** Inner joins in the feature pipeline silently drop rows when join keys don't match. Injury proxy features, for example, require game_id to exist in both the game log table and the injury source; if the injury source is empty or uses a different game_id format, the join produces an empty result.

**Consequences:** This is exactly what happened to `home_missing_minutes` and `home_star_player_out`. Player availability — the single strongest predictor of game outcome — has zero effect on predictions. The model "works" but is fundamentally blind to the most important variable.

**Warning signs:**
- Feature does not appear in model importances file despite being in the feature list
- Column exists in the CSV but all values are 0.0 or NaN
- No error is raised during feature engineering — it "succeeds" silently

**Prevention:**
- After every join in feature engineering, assert: `assert result.shape[0] > 0, "Join produced empty result: check key alignment"`
- Log null rates per column after feature assembly: any column with >50% nulls should raise a warning, >95% nulls should raise an error
- Add a feature audit step that compares the model's feature list against importance rankings: any feature with zero importance after training is a red flag
- Specific fix for injury proxy: verify the game_id format used in `injury_proxy.py` matches `team_game_features.py` before joining

**Phase to address:** Injury feature fix phase. Also applies to any phase that adds new features — establish the null-check pattern before the first new feature is merged.

**Confidence:** HIGH — documented bug in CONCERNS.md with direct evidence of the failure mode.

---

### Pitfall 3: Backtesting ATS Without Closing Line Value

**What goes wrong:** An ATS model is evaluated on whether it went 54%+ against the opening spread. But opening lines move significantly before tip-off. A model that beats the opener by picking line movement (i.e., what sharps already moved) provides no real edge — you cannot get opener prices in live deployment.

**Why it happens:** ATS backtests default to using the opening line because it's available historically. Closing lines are harder to source but represent what you'd actually bet against.

**Consequences:** Backtest shows profitable ATS record (+8% ROI). In live betting, the line has already moved against every pick; actual ROI is -3% (juice bleeds you).

**Warning signs:**
- ATS model performs much better against opening lines than closing lines in backtests
- Model consistently picks the same direction as line movement (confirms it's tracking sharp money, not finding edge)
- The Odds API provides closing lines — if model never compares to closing line, the backtest is incomplete

**Prevention:**
- Always evaluate ATS model against CLOSING line, not opening line
- Track "closing line value" (CLV): did the line move in your direction after the pick? Consistent positive CLV is evidence of real edge
- Store both opening and closing odds via The Odds API (historical data); compare pick outcomes against both
- Accept that beating closing line at >52.4% (break-even against -110 juice) for 200+ games is the minimum evidence bar for real edge

**Phase to address:** ATS model development phase. Must be enforced in the backtesting harness before declaring any ATS model results.

**Confidence:** HIGH — fundamental sports betting principle, directly relevant to the ATS model being planned.

---

### Pitfall 4: Training on Mixed Eras Destroys Modern Accuracy

**What goes wrong:** The model trains on 1996-2026 data (all available seasons). Pre-2014 NBA played at slower pace, lower 3PT volume, and fundamentally different style. The model learns patterns from an era that no longer exists, diluting its ability to recognize modern signals.

**Why it happens:** Including more data usually improves ML models, so the intuition is "more is better." In sports, however, regime changes (the 3-point revolution post-2014) invalidate older data as a signal source for current games.

**Consequences:** Confirmed in this codebase — model achieves 67-69% accuracy on 2005-2015 data, 64-66% on 2016-2026. The older data is not helping; it is actively hurting.

**Warning signs:**
- Walk-forward backtest shows accuracy declining in more recent years despite more training data being available
- Feature importances show "season" or "year" as top predictors (model learned era differences, not basketball patterns)
- Model performs differently on playoff vs. regular season games (regime sensitivity)

**Prevention:**
- Filter training data to 2014+ for game outcome model (already identified in PROJECT.md as a pending decision)
- Use walk-forward validation with expanding window starting from 2014, not 1996
- Optionally: test training on 2014-2019 vs. 2014-2026 to verify that 2020+ load management era is compatible with 2014-era patterns
- When adding historical data for a specific feature (e.g., referee tendencies back to 2010), use it as a feature but keep the training window at 2014+

**Phase to address:** Model retraining phase. This is the first change to make before any new features are added — the baseline must be correct.

**Confidence:** HIGH — documented in CONCERNS.md with specific accuracy numbers; confirmed by PROJECT.md key decisions.

---

### Pitfall 5: Using Vegas Lines as Targets Instead of Features

**What goes wrong:** The model is trained to predict game outcomes (win/loss or point differential), then its predictions are compared to Vegas lines after the fact to find disagreements. This is less powerful than using the Vegas line as an input feature and training an ATS-specific model directly.

A related mistake: treating the Vegas line as a validation oracle ("our model agrees with Vegas 80% of the time, so it's good"). Vegas is a better predictor than almost any public model — if your model just rediscovers the line, it has no edge.

**Why it happens:** It's natural to first predict outcomes, then compare to betting lines. Using lines as features feels circular ("we're using Vegas to beat Vegas"), but this misunderstands the value: the line is public information that the model should be able to leverage, not hide from.

**Consequences:** A model that ignores Vegas lines must rediscover consensus wisdom from raw box scores. Its "edge" detections are mostly noise. A model that uses the line as an input can focus on finding systematic inefficiencies — game situations where the market misprices risk.

**Warning signs:**
- Model's win probability closely tracks 1 - implied probability from the spread (model is just re-deriving the line)
- "Value bets" flagged by comparing model output to line turn out profitable only at 51-52% — within noise for <500 game samples
- Model's top features are things like home court and win percentage (what the line already prices)

**Prevention:**
- Build a dedicated ATS model with spread as an input feature: `model.predict(features + [current_spread]) → cover_probability`
- Features should include: spread at open, spread at close, line movement direction/magnitude (sharp signal), and market implied probability
- Train ATS model on binary target: did home team cover? (1/0)
- Keep win-probability model and ATS model separate — they optimize different objectives

**Phase to address:** ATS model development phase. The feature engineering for the ATS model must include odds/line data from the start.

**Confidence:** HIGH — established sports analytics principle; directly referenced in PROJECT.md key decisions.

---

### Pitfall 6: Calibration Artifacts Disconnected from Production

**What goes wrong:** Calibrated model probabilities are computed and saved, but uncalibrated predictions are served in production. The sportsbook comparison function compares uncalibrated win probabilities (often overconfident) against implied probabilities from the spread, producing systematically biased "value bet" signals.

**Why it happens:** Calibration is done as a post-processing step and saved separately. Unless the inference path explicitly loads the calibrated model, the default (uncalibrated) model is used.

**Consequences:** The model may report 72% confidence on a pick where true probability is 60%. This inflates apparent edge — a pick looks like +12% value when it's actually +2% (within noise). Real-money decisions based on this are overconfident.

**Warning signs:**
- Win probability outputs cluster near 0.7+ for picks; few predictions in the 0.55-0.65 "uncertain" range
- Reliability diagram shows predicted probabilities consistently higher than observed frequencies
- `models/artifacts/game_outcome_model_calibrated.pkl` exists but `predict_cli.py` never imports it

**Prevention:**
- Wire calibrated model into all inference paths before any ATS model development begins; otherwise the ATS model trains on uncalibrated probability inputs
- Add an assertion in `predict_cli.py` that raises an error if the calibrated artifact is missing (fail loudly rather than silently using the uncalibrated model)
- Use proper held-out calibration set (not the training data) per the data leakage note in CONCERNS.md
- Periodically check reliability diagrams: bin predictions into deciles, verify observed frequency matches predicted probability

**Phase to address:** Pipeline cleanup phase — fix before model retraining begins. Calibration must be correct before new ATS model is layered on top.

**Confidence:** HIGH — documented bug in CONCERNS.md with file-level evidence.

---

## Moderate Pitfalls

Mistakes that degrade model quality or create maintenance burden without causing complete failure.

---

### Pitfall 7: Injury Data That's Too Late

**What goes wrong:** Injury data is sourced from daily game logs (who played vs. who didn't), which is historical reconstruction of availability. This tells you what happened after the game. Real predictive value comes from pre-game injury reports — who is listed as questionable/out before tip-off.

**Why it happens:** Historical "who played" data is easy to get from nba_api box scores. Official pre-game injury reports require scraping a separate source (NBA.com injury report PDFs or ESPN) with a different schema.

**Consequences:** "Injury features" that use post-game availability (minutes played) are leaky — they use information only available after the game starts. Pre-game injury reports are the correct signal. Confusing these two sources is a common cause of inflated backtest accuracy.

**Warning signs:**
- Injury proxy uses "minutes played last game" as a proxy for "injured this game" — this is post-game data used to predict the prior game
- Backtest accuracy on games where star players were out is higher than expected (model peeked at who actually played)
- When injury proxy features are fixed, accuracy unexpectedly drops (was benefiting from leakage)

**Prevention:**
- Clearly separate: (a) historical availability reconstruction (who played, minutes) for training features, and (b) real-time pre-game injury report for live inference
- For training: use "player did not play last game" as a next-game injury signal — this is pre-game knowledge for game N+1
- For live prediction: scrape official NBA injury report 2 hours before tip-off; do not wait for box score availability
- Add a comment in `injury_proxy.py` explaining which features are pre-game vs. post-game available

**Phase to address:** Injury feature fix phase.

**Confidence:** HIGH — directly applicable to the broken injury proxy features documented in CONCERNS.md.

---

### Pitfall 8: Small Sample Size in ATS Win Rate Claims

**What goes wrong:** An ATS model goes 58% against the spread in backtesting over 200 games. This is reported as "significant edge." At 200 games, the 95% confidence interval for a 58% win rate spans roughly 51-65%. The model's actual edge could be zero.

**Why it happens:** Sports betting ROI math feels convincing ("if I bet $100 per game for 200 games at 58% hit rate, I profit $X"). The sample size required to distinguish real edge from variance is much larger than intuition suggests — typically 500-1000 games for 3-5% edge, more for smaller edges.

**Consequences:** Declaring the ATS model profitable after one season of backtesting leads to overconfidence. Model may be pattern-fitting to specific years' variance, not discovering real structural inefficiency.

**Warning signs:**
- ATS performance varies dramatically season-to-season in backtesting (good year/bad year alternating pattern)
- Performance is concentrated in specific game types (e.g., only home underdogs) — suggestive of overfitting to a segment
- Win rate is close to 54-55% — cannot be distinguished from juice-adjusted break-even with fewer than 500 games

**Prevention:**
- Report ATS results with confidence intervals, not just point estimates: "57.2% ± 4.3% (95% CI, n=320 games)"
- Require 500+ games before any ATS claim is treated as actionable
- Segment results: does the edge hold across multiple seasons? Multiple game types? If edge only appears in one season, assume it's noise
- Do not pathologically optimize ATS model for backtest period — use strict train/validation/test splits where test period is never touched during model development

**Phase to address:** ATS model evaluation phase. Enforce sample size requirements in the backtesting harness.

**Confidence:** HIGH — statistical principle; directly relevant to the scale of NBA seasons (~1200 games/season, meaning multi-season backtests are needed).

---

### Pitfall 9: Feature Engineering Drift Between Training and Inference

**What goes wrong:** Features are computed differently during model training (using full historical data) than during live inference (using only what's available today). For example, a "season average" feature computed at training time includes the full season's data; at inference time mid-season, it's only a partial season average.

**Why it happens:** Training pipelines batch-compute all features over the historical dataset. Inference pipelines compute features on-the-fly with only current data. If these code paths diverge, the model receives different feature distributions at inference time than it was trained on.

**Consequences:** Model accuracy in production is lower than backtest accuracy. The degradation is unpredictable and hard to debug — predictions look reasonable but are systematically biased.

**Warning signs:**
- Live predictions are consistently more extreme (higher/lower) than backtest predictions for equivalent game situations
- Features like "season ranking" or "cumulative stats" differ between training CSV and inference calculation
- No shared feature computation code between training and inference paths

**Prevention:**
- Write feature computation as functions with clear inputs (game_date, lookback_window) and test them in isolation
- Use the same function for both training (iterating over historical dates) and inference (using today's date)
- Add integration test: compute features for a historical game using the inference path and compare to the training CSV — values should match
- Document which features are "as of game date" vs. "full-season" and be explicit in code

**Phase to address:** Pipeline cleanup phase. Before adding new features for the ATS model, establish a consistent feature computation pattern.

**Confidence:** MEDIUM — well-known ML production issue; directly relevant given this project's separate training and inference code paths.

---

### Pitfall 10: Overfitting to NBA Schedule Patterns

**What goes wrong:** The model learns that teams on back-to-backs lose more often — which is true — but then overfits to specific back-to-back patterns (e.g., road back-to-backs in the Western Conference in December) that don't generalize. Schedule features are powerful but have many interacting levels.

**Why it happens:** Schedule fatigue is a real signal but it interacts with team quality, roster depth, and opponent strength. Simple "days rest" features capture only part of the signal. Over-engineered schedule features (city distance, timezone crossings, consecutive road games) can overfit.

**Consequences:** Model shows strong backtest performance with schedule features, but live performance is mediocre. Schedule patterns are partially priced by Vegas already — the edge is smaller than the raw backtest suggests.

**Warning signs:**
- Days rest is a top-5 feature by importance (model may be overfitting to historical schedule irregularities like COVID bubble seasons)
- ATS performance with schedule features is concentrated in specific months or game types
- 2019-20 bubble season data is causing spurious schedule patterns (all games at neutral site, no travel)

**Prevention:**
- Exclude 2019-20 bubble season from training data (or treat it as a special case with indicator variable)
- Exclude 2020-21 shortened/COVID season similarly
- Use coarse schedule features (back-to-back yes/no, days rest 0/1/2/3+) rather than fine-grained distance metrics initially
- Validate schedule feature value out-of-sample before adding to production model

**Phase to address:** Feature engineering phase for rest/schedule features.

**Confidence:** MEDIUM — known issue in sports analytics; COVID seasons specifically cause spurious patterns.

---

### Pitfall 11: The Preprocessing Full-Rebuild Masking Silent Bugs

**What goes wrong:** Preprocessing rebuilds all CSVs from scratch every run. This means if a raw data file is corrupted or has schema drift from a new nba_api version, the error propagates through the entire processed dataset — overwriting all historical processed CSVs with bad data. There's no previous version to roll back to.

**Why it happens:** Full rebuild is simple and guarantees consistency. The risk is only visible when something goes wrong upstream.

**Consequences:** A single bad nba_api response for the current season can silently corrupt 25+ seasons of processed data. The next model training run uses corrupted features without any error.

**Warning signs:**
- Preprocessing completes successfully but model accuracy suddenly drops
- Column counts differ unexpectedly between raw files (nba_api endpoint changed schema)
- Processed CSV row counts don't match historical expectations

**Prevention:**
- Never overwrite existing processed CSVs without first writing to a temp file and validating schema/row counts
- Add schema validation at the start of preprocessing: compare expected column list to actual columns in raw files
- Implement incremental preprocessing (only rebuild current season) to limit blast radius
- Keep a backup of last known good processed CSVs (or use git-tracked processed data snapshots)

**Phase to address:** Pipeline reliability phase (infrastructure/testing focus).

**Confidence:** HIGH — documented in CONCERNS.md; directly relevant to daily scheduled pipeline.

---

## Minor Pitfalls

Common mistakes that cause confusion or wasted effort but are recoverable.

---

### Pitfall 12: Reporting Accuracy Without Context of Vegas Baseline

**What goes wrong:** "Our model achieves 65% accuracy predicting NBA game winners" is reported as a success metric. The actual baseline for a naive model (always pick the home team or always pick the favorite) is often 60-62%. The marginal value of the ML model is 3-5 percentage points, not 65%.

**Why it happens:** Absolute accuracy sounds impressive. Relative improvement over baseline requires knowing what the baseline is.

**Prevention:**
- Always report accuracy relative to: (a) always-home-team baseline, (b) always-pick-favorite baseline, (c) Vegas implied probability baseline
- For ATS specifically: random picking covers at ~50%, break-even against juice is ~52.4% — report against these
- Document these baselines in any backtest report output

**Phase to address:** Any model evaluation phase. Enforce in backtest reporting format.

---

### Pitfall 13: Momentum Features That Are Actually Mean Reversion Signals

**What goes wrong:** "Team won 5 in a row, likely to win again" — momentum features encode recent form as a continuation signal. But NBA research shows moderate mean reversion: teams playing above their talent level regress. The feature direction may be backwards.

**Why it happens:** Sports fans intuitively believe in hot streaks. The ML model will find the correlation that's in the data — which may be mean reversion rather than momentum — but if the feature is engineered as "positive = good recent form," interpretation of the model is misleading.

**Prevention:**
- Include both recent form AND season expectation as separate features; let the model discover which matters more
- Check feature direction in SHAP values: is high recent win rate associated with higher win probability (momentum) or lower (regression)? Either could be true
- Avoid calling features "momentum" in code — name them descriptively ("rolling_win_rate_10g") and let the model decide the direction

**Phase to address:** Feature engineering phase for rolling window features.

---

### Pitfall 14: The Odds API Rate Limit Blowout During Backfill

**What goes wrong:** The Odds API free tier has a monthly request limit (500-2,000 requests depending on plan). Running a historical backfill of odds data can consume the entire monthly quota in a single run, blocking live data access for the rest of the month.

**Why it happens:** Backfill scripts loop over historical dates without checking remaining API quota. The Odds API returns quota usage in response headers but scripts may not read them.

**Prevention:**
- Add a quota check at the start of any odds backfill: read remaining monthly requests from The Odds API `/remaining-requests` header
- Add a hard stop if remaining requests fall below a safety threshold (e.g., 100)
- Backfill odds in small batches (one week at a time) with manual confirmation between batches
- Cache all API responses locally immediately; never re-fetch data that's already been stored

**Phase to address:** Odds integration / ATS data pipeline phase.

---

### Pitfall 15: Custom NumPy GBM Silently Diverging

**What goes wrong:** The hand-rolled GBM in `src/models/numpy_gbm.py` produces predictions that drift outside [0,1] or accumulate numerical errors across many estimators. No assertions exist to catch this.

**Why it happens:** Custom implementations lack the defensive programming of production libraries. Edge cases (empty leaf nodes, zero-variance splits) cause NaN propagation.

**Prevention:**
- Add `assert 0 <= pred <= 1` after every probability prediction
- Compare output distribution against scikit-learn's `GradientBoostingClassifier` on the same dataset monthly — if they diverge by >2%, investigate
- Document why custom implementation exists; if sklearn is available, default to it and reserve numpy_gbm as a fallback

**Phase to address:** Testing/reliability phase.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|----------------|------------|
| Injury feature fix | Silent null propagation (Pitfall 2) — fix compiles without error but features remain null | Add null-rate assertion after join; verify feature appears in model importances after retraining |
| ATS model development | Using opening line instead of closing line in backtest (Pitfall 3) | Enforce closing-line evaluation in backtesting harness before declaring any ATS result |
| Modern era retraining | Forgetting to exclude 2019-20 bubble / 2020-21 shortened seasons (Pitfall 10) | Explicit season exclusion list in training filter; document why |
| Calibration wire-up | Wiring calibrated model into ATS feature pipeline before re-fitting calibrator on correct holdout (Pitfall 6) | Fix calibration data split first; then wire inference path; then train ATS model |
| Rest/schedule features | Lookahead bias: "days rest" computed using game result date instead of game scheduled date (Pitfall 1) | Use scheduled game date as the anchor, not result posting date |
| Vegas line as feature | Using line as target (ATS outcome) AND as a training feature in the same model — this is fine. Using line as a feature to predict raw win probability — introduces circularity, do NOT do this (Pitfall 5) | Keep win-probability model (no line features) separate from ATS model (line as feature) |
| Backfill odds data | Rate limit exhaustion (Pitfall 14) | Batch backfill with quota guard |
| Feature engineering expansion | Training/inference feature drift (Pitfall 9) | Shared feature functions used in both paths |
| Model evaluation reporting | Claiming 65% accuracy without baseline context (Pitfall 12) | Always compute and report home-team baseline and Vegas-implied baseline |
| ATS win rate claim | Sample size too small (Pitfall 8) | Do not report ATS edge until 500+ game test set accumulated |

---

## Sources

- Project codebase audit: `.planning/codebase/CONCERNS.md` (2026-03-01) — direct evidence for Pitfalls 2, 6, 7, 11, 15
- Project specification: `.planning/PROJECT.md` (2026-03-01) — confirms Pitfalls 4, 5, 7
- Domain knowledge (NBA analytics, sports betting ML): training data through August 2025
  - Pitfalls 1, 3, 4, 5, 8, 9, 10, 12, 13 are well-established in sports analytics literature
  - Closing Line Value (CLV) as edge proxy: documented in sports betting analytics community (Joseph Peta, Rufus Peabody, Power Ranking Guru methodology)
  - Era regime change effect (pre/post 3-point revolution): documented in NBA analytics literature (Kirk Goldsberry, Nate Silver / FiveThirtyEight RAPTOR methodology)
- The Odds API rate limits: documented in The Odds API free tier pricing page (verified knowledge through August 2025)

**Note:** WebSearch and WebFetch were unavailable this research session. All domain-specific claims are based on training data knowledge (through August 2025) and the codebase audit. Pitfalls 3, 8, and 10 in particular should be verified against current sports analytics literature before implementing the ATS model evaluation harness.
