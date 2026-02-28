# Model Advisor Notes
**Written:** 2026-02-28
**Author:** Model Advisor Agent (Cycle 1)
**Files read:** `docs/agent_task_plan.md`, `docs/project_overview.md`, `reports/backtest_game_outcome.csv`, `reports/backtest_player_pts.csv`, `reports/explainability/feature_direction_*.csv`, `models/artifacts/*_importances.csv`
**Missing files noted:** `docs/project_master.md` — does not exist yet (first cycle). `docs/debugger_notes.md` — does not exist yet (first cycle). Calibration folder (`reports/calibration/`) exists but is empty — calibration was built but apparently not run against current models.

---

## What I Found Before Writing Proposals

### Game Outcome Model — Backtest Performance

The model has been tested across every season from 2005–06 through the current 2025–26 season. The headline numbers look reasonable — accuracy between 62% and 70%, ROC-AUC between 0.65 and 0.74 — but there is a clear and consistent pattern in the data that needs to be addressed.

The model performs noticeably better in the "Open Court" era (roughly 2005–2015) than it does in the modern "3-Point Revolution" era (2016–present). In the Open Court era, accuracy averaged around 67–69% with ROC-AUC around 0.72–0.74. In the 3-Point Revolution era, accuracy has averaged around 64–66% with ROC-AUC around 0.68–0.72. The worst recent seasons — 2020–21 (COVID bubble), 2022–23, and the current partial season — all fall below 65% accuracy with ROC-AUC below 0.70. This is not random noise. The model was largely designed and tuned on older data, and the modern game's increased variance (driven by three-point volume) is genuinely harder to predict AND may be exposing feature gaps.

The calibration folder is empty, meaning we have no data on whether the predicted probabilities are trustworthy. This is a serious gap for sportsbook comparison. A model can have 66% accuracy and still be a terrible betting tool if its 75% confidence predictions only win 58% of the time.

### Game Outcome Model — Feature Analysis

The two most important features are the 20-game rolling plus/minus differential between teams and the cumulative win percentage differential. Season-level net rating (split into home and away contexts) is 3rd and 4th. These make sense, but together they account for roughly 37% of the model's total importance weight — meaning the top four features do most of the work, and 100+ other features are sharing the remaining 63%.

Several notable gaps:
- **Injury proxy features are absent.** The features `home_missing_minutes`, `away_missing_minutes`, `home_star_player_out`, etc. were built and described in `project_overview.md`, but none of them appear in the model importances file. They are either not being passed into the model at training time, or they have so many missing values that the imputer neutralizes them. Given that player availability is one of the single biggest predictors of game outcome (a sportsbook's line moves 2–3 points when a star sits), this is the most important gap to fix.
- **Three-point features underperform their real-world importance.** The `diff_three_style_mismatch` feature is present (5th in the importances list), which is good. But three-point percentage rolling averages have near-zero SHAP importance across all window lengths. In an era where teams shoot 35+ threes per game, this is suspect — the model may be capturing pace and volume through plus_minus rather than specifically modeling three-point efficiency matchups.
- **Pace is not explicitly modeled.** The modern game is characterized by dramatically higher possession counts than 2005–2015. No `pace_roll` feature exists in the importances. When two fast-paced teams meet, the game is more high-variance and harder to predict — the model has no way to adjust its confidence for these games.

### Player Points Model — Backtest Performance

The player backtest data only runs through the 2015–16 test season. This is a significant gap — we're missing a full decade of NBA evolution, including the explosive three-point era that changed how scorers are used. More importantly, it means we don't know how the model performs against today's sportsbook lines. The MAE throughout the available history is consistently in the 2.4–2.7 range, beating the naive "predict the training mean" baseline by roughly 4 points — which is genuinely good.

However, 2.5 MAE is likely not tight enough for reliable sportsbook edge detection. A typical prop line is set within 1.0–1.5 points of a player's expected output. If our model is off by 2.5 points on average, the disagreements it flags against a sportsbook line will include a lot of noise alongside real edges. The north star here is getting MAE below 2.0 for current-era players — and ideally, having a confidence interval around each projection so we can distinguish high-certainty and low-certainty predictions.

### Player Model SHAP — Inconsistency and Possible Leakage

This is the most urgent finding in this analysis. The SHAP explainability files for the player models tell a different story than the model importance files:

- The SHAP output for the **points model** shows `fantasy_pts` as the single most important feature by a factor of three (abs_mean_shap = 6.4, vs. 0.84 for the second-place feature `dreb`). The current importances file does not include `fantasy_pts`. This means SHAP was computed on a different, earlier version of the model — one that apparently used a feature called `fantasy_pts` that has since been removed.
- The SHAP output for the **rebounds model** shows raw `dreb` (that game's actual defensive rebounds) and raw `oreb` (that game's actual offensive rebounds) as the two dominant features. These are the current game's actual stats, not rolling averages. Predicting rebounds using the same game's rebounds is circular — it is data leakage.
- The SHAP output for the **assists model** also shows `fantasy_pts` as the top feature.

To be clear: the current production model importances file does not contain these raw stats, suggesting the production models were retrained to remove them. But the SHAP reports in `reports/explainability/` are stale — they describe an older, potentially leaky model. The Coder needs to re-run `src/models/run_evaluation.py` to regenerate SHAP against the current models so we have trustworthy explainability data going forward.

---

## Proposals

### Proposal 1: Re-run Evaluation Suite to Regenerate Stale Reports
**What to try:** Run `src/models/run_evaluation.py` in full against the current trained models. The SHAP explainability reports in `reports/explainability/` were generated from a previous model version and show features (`fantasy_pts`, raw `dreb`, raw `oreb`) that are no longer in the current model. Before any further improvements can be evaluated honestly, the reports need to reflect the actual models in production.

**Why it helps:** Nearly every other proposal in this document depends on having accurate explainability data. Without updated SHAP outputs, we can't tell which features are actually doing work, which are deadweight, or whether any leakage survived into the current model. This is also the only way to generate the calibration outputs that are essential for sportsbook comparison.

**Effort:** Low (running a script — a few minutes of compute, maybe an hour to review output).

---

### Proposal 2: Verify Injury Proxy Features Are Reaching the Model
**What to try:** In `src/models/game_outcome_model.py`, print or log the list of columns in the feature DataFrame just before model training. Confirm that `home_missing_minutes`, `away_missing_minutes`, `home_rotation_availability`, `away_rotation_availability`, `home_star_player_out`, `away_star_player_out`, and the corresponding `diff_` columns are present and not all-null. If they are missing or all-null, trace back through `src/features/injury_proxy.py` and `src/features/team_game_features.py` to find where the join is failing. Once confirmed present, check if the model's feature importance list includes them — if not after fixing the join, consider making the injury features mandatory inputs rather than optional.

**Why it helps:** Player availability is the single biggest driver of line movement in sportsbooks. A team missing a 28% usage star should have meaningfully lower win probability. The features were built correctly (the logic in project_overview.md is sound), but they're invisible in the importances file — meaning they either aren't making it into the training data or they're being swamped by null values. Getting these features active is likely the highest-ceiling single improvement available to the game outcome model.

**Effort:** Low to Medium (depending on whether the issue is a missing join or a deeper data gap).

---

### Proposal 3: Run Calibration and Apply Isotonic Regression Wrapper to Game Outcome Model
**What to try:** The calibration module (`src/models/calibration.py`) exists and was built, but the `reports/calibration/` folder is empty. Run the calibration suite against the current game outcome model and review the reliability diagram. If the model is overconfident (curves below the diagonal — predicted 70% but actually wins 62% of those games), apply the isotonic regression wrapper that was described in `project_overview.md` and save the calibrated model alongside the uncalibrated one. Save calibration outputs to `reports/calibration/`.

**Why it helps:** This is the most important step before the sportsbook comparison dashboard goes live. If predicted probabilities aren't calibrated, then our "model says 68%, sportsbook implies 58%" flag could be meaningless — the model might actually agree with the sportsbook after adjustment. Calibration is what makes a probability number trustworthy rather than just directionally suggestive.

**Effort:** Low (running a script; possibly adding a few lines to save the calibrated model wrapper).

---

### Proposal 4: Extend Player Backtest to Current Seasons
**What to try:** The player backtest (`reports/backtest_player_pts.csv`) stops at the 2015–16 test season. The game outcome backtest runs through 2025–26. Update `src/models/backtesting.py` so the player model backtest also rolls forward through every available season. The output should be `reports/backtest_player_pts.csv`, `reports/backtest_player_reb.csv`, and `reports/backtest_player_ast.csv` all covering the full range of available data.

**Why it helps:** We need to know how the player model performs on modern NBA data — the 2020–2025 period with its elevated three-point volume, load management, and different player usage patterns. The current backtest only tells us about players and scoring patterns from 2001 to 2016. If MAE has drifted upward on modern seasons (which is plausible), we need to know that before we launch sportsbook comparisons.

**Effort:** Low (modifying a loop boundary and re-running the script).

---

### Proposal 5: Filter Game Outcome Model Training to Modern Era
**What to try:** Add a training data filter in `src/models/game_outcome_model.py` so that only seasons from 2014–15 onward are used for training (corresponding to `era_num = 6`). Keep the existing full-history version as a comparison. Backtest both versions and compare accuracy in the most recent five seasons. If the modern-only model outperforms the full-history model on recent seasons, make it the default.

**Why it helps:** The model is trained on 80 seasons of NBA data spanning six distinct eras. The foundational era (1946–1979) and the physical/isolation era (1994–2004) have meaningfully different structural relationships between stats — pace, three-point rate, foul frequency. Including that data may be hurting performance on modern games by teaching the model patterns that no longer apply. The era labels were built specifically to enable this kind of filtering. The backtest already shows the model performing better on Open Court era data, which suggests that older data may be confusing more than it's helping.

**Effort:** Medium (adding a filter parameter, retraining, comparing results).

---

### Proposal 6: Add Home/Away Split Averages to Player Points Model
**What to try:** In `src/features/player_features.py`, add home-specific and away-specific rolling scoring averages for each player — `pts_home_avg` (rolling 10-game average in home games only) and `pts_away_avg` (rolling 10-game average in away games only). Add the same for rebounds and assists. These should be shift(1) protected like all other rolling features. Note: the current production model importances already show `pts_home_avg` and `pts_away_avg` in the list (ranked 4th and 6th), which suggests these features exist in `player_pts_importances.csv` but may not be in the SHAP analysis or all player feature scripts. Verify they are present and correctly joined for all players, not just a subset.

**Why it helps:** Many players have significant home/away splits — some players score 4–5 more points per game at home. Sportsbooks account for this. If our model only sees overall rolling averages, it will systematically over-project road players and under-project home players for certain archetypes. Confirming these features are correctly built and populated closes a gap that sportsbooks exploit.

**Effort:** Low (verifying existing features are correct; minor additions if any are missing).

---

### Proposal 7: Add a Pace-Adjusted Scoring Opportunity Feature to Player Points Model
**What to try:** In `src/features/player_features.py`, create a feature `pts_per_100_poss_roll10` — the player's points per 100 possessions in their last 10 games, using their team's possession estimate. Team possessions can be estimated as `(FGA + 0.44 * FTA - OREB + TOV)` from `team_game_logs.csv`. Then join the opposing team's rolling pace (average possessions per game) as `opp_pace_roll10`. These two features together tell the model: "this player scores X points per 100 possessions, and tonight's game is expected to have Y possessions." This is a standard adjustment that all serious sportsbook models use.

**Why it helps:** A player in a fast-paced game has more scoring opportunities than the same player in a slow game. A player who averages 22 points against normal defenses might average 26 in an up-tempo game and 18 in a defensive slog. Rolling raw point totals miss this entirely. Pace-adjusted scoring is one of the most reliable improvements available for point projection accuracy, and it directly improves the model's ability to compare against a sportsbook line that has already been pace-adjusted.

**Effort:** Medium (new feature calculation, joining pace data, retraining and evaluating).

---

### Proposal 8: Add Prediction Confidence Intervals to Player Projections
**What to try:** Modify `src/models/player_performance_model.py` so that in addition to the point estimate, it also outputs a confidence interval for each player projection. The simplest approach: use the player's rolling standard deviation features (`pts_std10`, `pts_std20`, which already exist in the importances) to construct a ±1 standard deviation range around the projection. Output this as `pts_projection_low`, `pts_projection`, `pts_projection_high` in the prediction output. Alternatively, use quantile regression (scikit-learn supports `GradientBoostingRegressor` with `loss='quantile'`) to directly predict the 25th and 75th percentile alongside the median.

**Why it helps:** The sportsbook comparison flag threshold in the project spec is "flag when model projection differs from line by more than 1.5 units." But a 1.5 unit difference means something very different for a consistent player (whose 10-game standard deviation is 3 points) versus a high-variance player (whose standard deviation is 9 points). Without a confidence interval, the flagging logic will over-flag volatile players and under-flag consistent ones. A projection with a confidence interval attached is genuinely usable; a bare point estimate with no uncertainty measure is not. This is also one of the most visible improvements for the website — showing "projected: 22–28 pts (line: 24.5)" is far more informative than just "projected: 25 pts."

**Effort:** Medium (adding quantile outputs and propagating them through to the prediction CLI and comparison file).

---

### Proposal 9: Add a Minutes Projection to the Player Model
**What to try:** Build a lightweight minutes prediction model or use a rolling average approach in `src/features/player_features.py`. Specifically: create `predicted_min` as a weighted average of `min_roll5` (60%) and `min_season_avg` (40%), then adjust it downward if `is_back_to_back = 1` (apply a -1.5 minute penalty based on historical back-to-back averages). Use this `predicted_min` as an input feature to the pts/reb/ast models, replacing the current `min_roll5` and `min_roll20` features. The injury proxy module already has a `rotation_availability` signal that can flag when a player's role might expand due to teammates sitting — this should also feed into the predicted minutes.

**Why it helps:** Playing time is the dominant driver of counting stat totals. The current model uses recent minutes averages as features, but doesn't adjust them forward to tonight's expected context. A player coming off a 36-minute game versus a 24-minute game will have different rolling minute averages, but if his role is stable, both should project similarly. More importantly, if a teammate is out (captured by injury proxy), minutes should go up. Connecting injury proxy signals → minutes adjustment → stat projection is the closest this project can get to how a sportsbook's quant team would model player props.

**Effort:** Medium (modifying the feature pipeline to connect injury proxy data through minutes into the stat projections).

---

### Proposal 10: Add Referee Foul-Rate Context Feature
**What to try:** The NBA publishes official game referee assignments. The `nba_api` package exposes referee data through the `LeagueGameFinder` or box score endpoints — specifically, each game's `OFFICIALS` field includes referee names and IDs. Build a small lookup table in `data/processed/referee_stats.csv` that tracks each referee's historical average fouls-called-per-game over the previous season (shift-1 to prevent leakage). Join this to `game_matchup_features.csv` as `referee_foul_rate_avg` before model training. This is a low/medium effort feature that sportsbooks have been using for years.

**Why it helps:** Referee assignment is a known variable that affects game pace, foul totals, and free throw attempts — all of which correlate with scoring totals and game outcomes. A game with a historically foul-happy referee crew is going to play out differently from a game with a hands-off crew. This is a genuinely independent signal that doesn't overlap with any of the team rolling average features currently in the model. It's the kind of edge a sportsbook would use that this project currently ignores entirely.

**Effort:** Medium (new data pull, processing, joining — but the nba_api endpoint work is new territory and may require some experimentation).

---

## Summary

The biggest single opportunity right now is connecting the injury proxy features to the game outcome model. The project already did the hard work of building a sophisticated proxy for player availability from the game logs — but that work appears to have gotten disconnected from the actual model training. Getting `missing_minutes`, `rotation_availability`, and `star_player_out` into the trained model could be the difference between a model that treats "LeBron sits" as a normal game and one that properly adjusts for it. No sportsbook ignores player availability, and right now this model effectively does. Beyond that, the player projection side of the project needs calibration work before it's ready for sportsbook comparison: the SHAP reports are stale and possibly based on a leaky model version, the backtest stops a decade too early to be meaningful for current predictions, and projections have no confidence intervals attached — making it impossible to tell which disagreements with a sportsbook line actually mean something versus which are just noise.
