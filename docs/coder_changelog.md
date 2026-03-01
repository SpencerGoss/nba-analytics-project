# Coder Changelog
**Author:** Coder Agent (Cycle 1)
**Date:** 2026-02-28

---

## Change: Create src/data/get_odds.py
File: `src/data/get_odds.py`
What changed: Created a new module with a `refresh_odds_data()` function that calls `scripts/fetch_odds.py` as a subprocess. Returns `True` on success, `False` on any failure (missing API key, network error, etc.) so `update.py` can continue the pipeline rather than crashing.
Why: `update.py` already imports `from src.data.get_odds import refresh_odds_data` at the top of the file, which means the entire update pipeline was broken with an `ImportError` every time it ran. This was the most critical fix in the batch — it restores `update.py` to working order. The subprocess approach was chosen over a direct import because `scripts/fetch_odds.py` calls `sys.exit(1)` at module load time when the API key is absent, which would kill the parent process if imported directly.
Status: Complete

---

## Change: Error handling in backfill.py
File: `backfill.py`
What changed: Added a `_log_pipeline_error()` helper and wrapped the body of `main()` in a `try/except Exception` block. Errors are logged with a timestamp to `logs/pipeline_errors.log` and the script exits with code 1. The actual backfill logic was moved into `_run_backfill()` so the structure mirrors the error-handling pattern already in `update.py`.
Why: `security_notes.md` (Medium priority issue) confirmed that `backfill.py` had no top-level error handling. A failure mid-backfill (e.g., API rate limit, network drop, disk write error) would crash silently with no record of what went wrong. The fix ensures any failure is recorded in the log file and produces a non-zero exit code that Windows Task Scheduler, cron, or CI can detect.
Status: Complete

---

## Change: Injury proxy features in game_outcome_model.py
File: `src/models/game_outcome_model.py`
What changed: Updated `get_feature_cols()` to include injury proxy context columns in addition to the `diff_*` differential features. The added columns are: `home_missing_minutes`, `away_missing_minutes`, `home_missing_usg_pct`, `away_missing_usg_pct`, `home_rotation_availability`, `away_rotation_availability`, `home_star_player_out`, `away_star_player_out`. These are included alongside the existing rest/schedule context columns.
Why: `model_advisor_notes.md` Proposal 2 identified that the injury proxy features built by `src/features/injury_proxy.py` were not reaching the model. The `get_feature_cols()` function previously only selected `diff_*` columns plus four schedule columns. Binary flags like `home_star_player_out` and `away_star_player_out` will never appear as `diff_*` — they're per-side signals where the absolute value (star is out on the home team) matters, not just the differential. These features are the single highest-ceiling improvement identified by the Model Advisor, since player availability is one of the strongest predictors of game outcome. **Note:** These columns will only have non-null values if `injury_proxy_features.csv` has been built and joined into `game_matchup_features.csv`. The model's `SimpleImputer` will handle null values with the column mean (which is the correct neutral value for these features). A retrain of `game_outcome_model.py` is required for these features to take effect.
Status: Complete

---

## Change: Save calibrated model artifact in calibration.py
File: `src/models/calibration.py`
What changed: When `run_calibration_analysis()` fits an isotonic regression calibrator on a v1 (plain Pipeline) model, it now saves the fitted `CalibratedClassifierCV` wrapper to `models/artifacts/game_outcome_model_calibrated.pkl`. For v2 models (already a `CalibratedClassifierCV`), the loaded artifact is already calibrated and no extra save is needed.
Why: `model_advisor_notes.md` Proposal 3 notes that the calibration module existed but the `reports/calibration/` folder was empty and no calibrated model artifact was saved. The calibration analysis provides trustworthy probability outputs, which is essential for the sportsbook comparison: flagging a game as "model says 68%, book implies 58%" is only meaningful if those 68% predictions actually win ~68% of the time. Without saving the calibrated model, `scripts/fetch_odds.py` and `predict_cli.py` would always use the uncalibrated probabilities. **Note:** `src/models/run_evaluation.py` needs to be run (against the current trained models) to populate `reports/calibration/` and produce the calibrated artifact.
Status: Complete

---

## Change: Modern era training filter in game_outcome_model.py
File: `src/models/game_outcome_model.py`
What changed: Added two constants (`MODERN_ERA_ONLY = False`, `MODERN_ERA_START = "201415"`) and wired them into `train_game_outcome_model()` via a new `modern_era_only` parameter. When `modern_era_only=True`, training is restricted to the 2014-15 season onward (the 3-Point Revolution era) instead of the default 2000-01+ range.
Why: `model_advisor_notes.md` Proposal 5 suggests filtering training to the modern era to reduce noise from structurally different historical eras. The default remains `False` (full history), so no existing behavior changes. To run the comparison described in the proposal: set `MODERN_ERA_ONLY = True`, retrain, run the backtest, compare recent-season accuracy. Decision: implemented as a toggle constant rather than a command-line flag to keep the change minimal and easily reversible. The constant is documented with a comment pointing to the proposal for context.
Status: Complete

---

## Items reviewed but no code change needed

**Large data files tracked in git:** `git ls-files data/ models/artifacts/` returned no results — these paths are already untracked. The `.gitignore` is working correctly. No action needed.

**`*.pkl` and `*.db` in .gitignore:** Both patterns already exist in `.gitignore`. The security agent's note appears to have been based on a pre-existing state that had already been resolved.

**`update.py` error handling:** Already implemented with `log_pipeline_error()` and a top-level `try/except`. No change needed.

**`game_outcome_model.py` syntax error at line 201 (flagged by Odds Agent):** Reviewed the file — no syntax error exists. Line 201 is a standard `print()` statement. The error may have been an artifact of the sandbox environment where the Odds Agent couldn't load the model and attributed the failure to a syntax error. No action needed.

**Proposal 1 (Re-run evaluation suite):** This is a script execution task, not a code change. Run `python src/models/run_evaluation.py` from the project root to regenerate SHAP reports and calibration outputs against the current models. Required before Proposal 3's calibrated model artifact will exist.

**Proposal 4 (Extend player backtest to current seasons):** The backtest code in `backtesting.py` already uses all available seasons dynamically — there is no hardcoded cutoff. The reason the backtest stops at 2015-16 is that `player_game_features.csv` only contains data through that season. The fix is to run the full data pipeline (several hours), not a code change.

**Proposal 6 (Verify home/away split averages):** The `pts_home_avg` and `pts_away_avg` features appear in `player_pts_importances.csv`. Based on this evidence, they exist and are being used. A full verification would require reading the player features CSV column headers, which requires the data pipeline to have been run. No code change made — note for the Debugger to verify against actual data.

**Priority 3 Odds integration (beyond get_odds.py):** `scripts/fetch_odds.py` already exists and is fully implemented. `update.py` already calls `refresh_odds_data()`. The integration is complete at the code level. The only remaining items are operational: add `ODDS_API_KEY` to `.env`, and run the player data pipeline to bring `player_game_features.csv` current so player props comparisons work.
