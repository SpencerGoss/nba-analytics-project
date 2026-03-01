# Debugger Notes
**Reviewer:** Debugger Agent (Cycle 1)
**Date:** 2026-03-01

---

## Create src/data/get_odds.py
**Status:** Pass

**Plain-language explanation:** Before this fix, the entire daily update pipeline was completely broken. Every time `update.py` tried to run, it crashed immediately because it was trying to use a file (`get_odds.py`) that didn't exist yet. This new file acts as a middleman — it runs the odds-fetching script in a safe, isolated way. If your API key isn't set up, or the internet is down, it just says "couldn't get odds today" and lets the rest of the pipeline keep running (game logs, stats, features, etc. all still update). Think of it like a mail carrier who knocks on the door — if nobody answers, they move on to the next house instead of stopping the whole route.

**Risk check:** No data leakage. No broken logic. The script correctly runs `fetch_odds.py` as a separate process rather than importing it directly, which prevents a known crash that happens when the API key is missing. Path resolution uses `Path(__file__).resolve()` which works regardless of where you run the command from. I ran `refresh_odds_data()` successfully — it returned `True` without errors.

**Notes:** The odds script happened to succeed in my test environment (it found an API key in `.env`). On a fresh install without the key, it will return `False` and log a warning, which is the correct behavior.

---

## Error handling in backfill.py
**Status:** Pass

**Plain-language explanation:** The backfill script is the one you run once to pull years of historical NBA data. Before this change, if something went wrong halfway through (say the NBA's servers stopped responding), the script would just crash with a confusing error message and no record of what happened. Now it catches any failure, writes a timestamped entry to a log file (`logs/pipeline_errors.log`), prints a human-readable message, and exits with an error code that Windows Task Scheduler or any automation tool can detect. The actual backfill logic was moved into its own section to keep things organized — same behavior, just wrapped in a safety net.

**Risk check:** No data leakage concerns. No broken logic. The error handling uses a broad `except Exception` which is appropriate for a top-level safety net — it won't accidentally swallow keyboard interrupts (`KeyboardInterrupt` and `SystemExit` are not subclasses of `Exception`). The `raise SystemExit(1)` ensures the process exits with a non-zero code after logging. Import tested successfully.

**Notes:** The `logs/` directory is created automatically if it doesn't exist (`mkdir(parents=True, exist_ok=True)`). No issues here.

---

## Injury proxy features in game_outcome_model.py
**Status:** Pass

**Plain-language explanation:** The project already had code that builds "injury proxy" features — estimates of how much a team is hurt by missing players (how many minutes are missing, how important those players are, whether a star is out). But those features were being calculated and then completely ignored by the prediction model. It's like hiring a scout, getting their report, and then leaving it unopened on your desk. This change opens the envelope — it tells the model to actually look at those 8 injury-related columns (4 for the home team, 4 for the away team) when making predictions. Player availability is one of the strongest signals for who wins a game, so this could meaningfully improve accuracy.

**Risk check:** This is the most important check. I verified:

1. **Column names match exactly.** The 8 column names listed in `game_outcome_model.py` (`home_missing_minutes`, `away_missing_minutes`, `home_missing_usg_pct`, `away_missing_usg_pct`, `home_rotation_availability`, `away_rotation_availability`, `home_star_player_out`, `away_star_player_out`) match exactly what `injury_proxy.py` produces.
2. **Columns are present in the actual data.** I loaded the real `game_matchup_features.csv` and confirmed all 8 injury columns plus 3 `diff_` injury columns are present and selected by `get_feature_cols()`. Total feature count: 46.
3. **No data leakage.** The injury proxy features are built from rolling 5-game averages of player absence data, computed *before* each game. They represent "who is expected to miss this game based on recent history" — not "who actually missed this game." The current game's outcome is not used.
4. **Null handling is safe.** If `injury_proxy_features.csv` hasn't been built yet, these columns will be missing from the matchup file. The `get_feature_cols()` function only selects columns that actually exist in the dataframe (via the list comprehension over `df.columns`), so the model will simply train without them rather than crashing. If the columns exist but have null values, the model's `SimpleImputer(strategy="mean")` fills them with the column average, which is a neutral value.

**Notes:** The model needs to be retrained (`python src/models/game_outcome_model.py`) for these features to actually take effect. Until then, the saved model artifact still uses the old feature list.

---

## Save calibrated model artifact in calibration.py
**Status:** Warning

**Plain-language explanation:** When the model says "68% chance the home team wins," you'd want that to actually mean home teams win about 68% of the time in those situations. That's called calibration. This project already had code to *measure* calibration, but it wasn't saving the improved (calibrated) version of the model anywhere. So even after running the calibration analysis, the prediction scripts were still using the raw, uncalibrated probabilities. This change fixes that — for older model versions (v1), after calibration is computed, the calibrated model is saved to `game_outcome_model_calibrated.pkl` so that other scripts can load and use it.

**Risk check:** There is a **minor data leakage concern** in the v1 calibration path (line 377-380). When fitting the isotonic calibrator, the code uses the same training data that the model was already trained on. The model has already "seen" this data, so its predictions on it are slightly overconfident. This means the calibrator learns a mapping from slightly-too-confident predictions to actual outcomes, which could cause it to under-correct when applied to truly unseen games. The standard fix is to use a held-out calibration set (separate from both training and test) or cross-validated calibration. **This is not a critical bug** — the calibrated probabilities will still be better than the raw ones — but for maximum trustworthiness, the calibration should ideally use data the model hasn't trained on.

For v2 models (which are already wrapped in `CalibratedClassifierCV` from training), this issue does not apply — the calibration was done during training with proper cross-validation.

**Notes:** No downstream scripts (`fetch_odds.py`, `predict_cli.py`) currently load the `_calibrated.pkl` artifact — I searched the codebase and found no references. This means the calibrated model is being saved but not yet used anywhere. A future cycle should wire it in or it will just sit unused.

---

## Modern era training filter in game_outcome_model.py
**Status:** Pass

**Plain-language explanation:** NBA basketball in 2025 looks very different from NBA basketball in 2001. The 3-point revolution, pace changes, rule changes — a model trained on 25 years of data might be learning patterns from an era that no longer applies. This change adds a simple on/off switch: flip `MODERN_ERA_ONLY` to `True`, and the model will only train on games from 2014-15 onward (roughly the start of the 3-point era). The switch is off by default, so nothing changes unless you deliberately turn it on. It's an experiment to see if "less but more relevant" data beats "more but noisier" data.

**Risk check:** No data leakage. The filter is applied to training data only — it doesn't change anything about how test seasons are selected or evaluated. The season comparison uses string comparison (`>= "201415"`), which works correctly because the season format is `YYYYMM` (e.g., `"201415"`) and sorts lexicographically the same as chronologically. Default behavior is unchanged (`MODERN_ERA_ONLY = False`).

**Notes:** When testing this, be aware that switching to modern-era-only cuts training data significantly (roughly half). The model gets more relevant data but less of it. Worth comparing accuracy on the test seasons both ways before committing to one approach.

---

## Overall Summary

**5 changes reviewed — 4 Pass, 1 Warning.**

The Warning is on the calibration artifact save: the v1 calibration path fits on the same data the model trained on, which slightly weakens the calibration's reliability. Not a showstopper, but worth fixing in a future cycle. Additionally, the calibrated model artifact is saved but not yet loaded by any downstream script — it needs to be wired into `fetch_odds.py` or `predict_cli.py` to actually matter.

All imports tested clean. `refresh_odds_data()` ran successfully. Feature column alignment verified against real data. No data leakage found in the prediction features.
