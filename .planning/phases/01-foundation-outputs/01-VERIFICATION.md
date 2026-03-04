---
phase: 01-foundation-outputs
verified: 2026-03-02T00:00:00Z
status: gaps_found
score: 3/5 success criteria verified
gaps:
  - truth: "Running predict_cli.py on any upcoming game produces non-null values for missing_minutes and star_player_out in the feature vector"
    status: partial
    reason: "injury_proxy_features.csv was rebuilt (77,116/126,720 rows non-zero, 60.9%) but team_game_features.py and build_matchup_dataset() were NOT re-run after the fix. game_matchup_features.csv is dated Feb 27, the injury fix was committed Mar 1. All 68,165 rows in game_matchup_features.csv show 0.0 for missing_minutes and 0 for star_player_out."
    artifacts:
      - path: "data/features/game_matchup_features.csv"
        issue: "Stale — built Feb 27 before injury proxy fix. All injury columns are 0 (non-null but meaningless)."
      - path: "data/features/team_game_features.csv"
        issue: "Stale — built Feb 27 before injury proxy fix."
    missing:
      - "Run: python src/features/team_game_features.py to rebuild team_game_features.csv with fixed injury data"
      - "This automatically triggers build_matchup_dataset() to rebuild game_matchup_features.csv"
  - truth: "The calibrated model artifact (game_outcome_model_calibrated.pkl) is loaded at inference"
    status: failed
    reason: "models/artifacts/game_outcome_model_calibrated.pkl does not exist. calibration.py script exists and is documented; _load_game_outcome_model() correctly falls back to uncalibrated model with a UserWarning. But success criterion 2 states the calibrated artifact IS loaded, not that the fallback warning fires."
    artifacts:
      - path: "models/artifacts/game_outcome_model_calibrated.pkl"
        issue: "File does not exist. calibration.py has not been run since training."
    missing:
      - "Run: python src/models/calibration.py to generate game_outcome_model_calibrated.pkl"
human_verification:
  - test: "Run predict_cli.py after rebuilding features and calibrating"
    expected: "Prediction dict shows non-zero home_missing_minutes and home_star_player_out when players are out; model_artifact shows 'game_outcome_model_calibrated.pkl'"
    why_human: "Requires running feature rebuild (~5-10 min) and calibration (~1-2 min) then making a live prediction"
---

# Phase 1: Foundation & Outputs Verification Report

**Phase Goal:** The model produces predictions backed by working features, calibrated probabilities, and a persistent history store — removing all known silent failures before any new features are added
**Verified:** 2026-03-02
**Status:** GAPS FOUND (2 gaps)
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths (from ROADMAP.md Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| SC-1 | Running `predict_cli.py` produces non-null `missing_minutes` and `star_player_out` in feature vector | PARTIAL | Code fixed and injury_proxy_features.csv rebuilt (60.9% non-zero), but game_matchup_features.csv is stale (all-zero). Root code is correct; data artifact needs rebuild. |
| SC-2 | Calibrated model artifact (`game_outcome_model_calibrated.pkl`) is loaded at inference | FAILED | Artifact does not exist. `_load_game_outcome_model()` correctly falls back with UserWarning, but the calibrated artifact was never generated (calibration.py not run post-training). |
| SC-3 | Running feature assembly with column >95% null raises error with column name | VERIFIED | `validate_feature_null_rates()` exists in `game_outcome_model.py`, raises `ValueError` naming the offending column. Called inside `train_game_outcome_model()` before any `model.fit()`. Confirmed by live test. |
| SC-4 | Prediction record written to `predictions_history.db` after every `predict_cli.py` run; JSON snapshot written | VERIFIED | `predictions_history.db` exists with WAL mode. `write_game_prediction()` and `export_daily_snapshot()` wired into `predict_game()` via non-fatal try/except. End-to-end test passed. |
| SC-5 | Each pipeline stage runs independently with documented inputs/outputs; `update.py` one-line module imports | VERIFIED | `docs/PIPELINE.md` exists (1,094 words, all 6 stages, "Does NOT" section). `update.py` has 18 `src.*` imports and zero subprocess calls to Python scripts. Confirmed by AST analysis. |

**Score:** 3/5 success criteria verified

---

## Required Artifacts

### Plan 01-01: Injury Proxy Fix and Null-Rate Guard

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/features/injury_proxy.py` | game_id and team_id cast to str.strip()/int before return | VERIFIED | Lines 317-318: `result["game_id"] = result["game_id"].astype(str).str.strip()` and `result["team_id"] = result["team_id"].astype(int)`. Algorithm rewritten to use `merge_asof` — 137,736 absent instances detected (was 0). |
| `src/features/team_game_features.py` | Both-sides normalization before injury merge; `n_matched > 0` assert | VERIFIED | Lines 537-540 normalize both sides. Lines 548-552 assert `n_matched > 0` with `if not injury_df.empty:` guard. |
| `src/models/game_outcome_model.py` | `validate_feature_null_rates()` called inside training | VERIFIED | Function at lines 52-80. Called at line 235: `validate_feature_null_rates(df, feat_cols)`, after `get_feature_cols(df)` and before any `Pipeline.fit()`. |

### Plan 01-02: Calibrated Inference and Metadata JSON

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/models/game_outcome_model.py` | `_load_game_outcome_model()` helper preferring calibrated artifact | VERIFIED | Lines 123-152. Checks for `game_outcome_model_calibrated.pkl` first; issues `UserWarning` on fallback; raises `FileNotFoundError` if neither exists. |
| `models/artifacts/game_outcome_metadata.json` | JSON with feature_list, trained_at, top_importances | NOT PRESENT | File does not exist. Metadata serialization code exists in `train_game_outcome_model()` (lines 342-370) and will write on next training run. No training has run since the code change was committed. |
| `models/artifacts/game_outcome_model_calibrated.pkl` | Calibrated artifact for inference | MISSING | `calibration.py` exists and is documented; artifact was never generated. |

### Plan 01-03: Prediction Store and JSON Export

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/outputs/__init__.py` | Package marker | VERIFIED | Exists with docstring. |
| `src/outputs/prediction_store.py` | `init_store()`, `write_game_prediction()` — WAL SQLite writer | VERIFIED | Full implementation: WAL pragma in `_get_connection()`, all schema columns, 3 indexes. Live test: rowid=1, WAL mode confirmed. |
| `src/outputs/json_export.py` | `export_daily_snapshot()` — daily JSON file writer | VERIFIED | Full implementation. Writes `predictions_YYYYMMDD.json` to `data/outputs/`. Live test passed. |
| `database/predictions_history.db` | WAL-mode SQLite with `game_predictions` table | VERIFIED | Exists with WAL mode, `game_predictions` table and `sqlite_sequence` present. |

### Plan 01-04: Pipeline Documentation

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `docs/PIPELINE.md` | All 6 stages with commands, inputs, outputs, runtimes; explicit "Does NOT" section | VERIFIED | 1,094 words. All 13 required content sections present (Stage Order, Daily Data Refresh, Does NOT, Stages 1-6, predictions_history.db, data/features, models/artifacts). |
| `update.py` | Thin orchestrator — all capabilities via one-line module imports | VERIFIED | 18 `src.*` imports, zero subprocess calls to Python scripts (confirmed by AST parse). |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `src/features/injury_proxy.py` | `src/features/team_game_features.py` | left merge on [team_id, game_id] with both-sides normalization | VERIFIED | Lines 537-542 in team_game_features.py. Pattern: normalize → merge → assert. |
| `src/models/game_outcome_model.py` | `validate_feature_null_rates` | called at top of `train_game_outcome_model()` | VERIFIED | Line 235: `validate_feature_null_rates(df, feat_cols)` after feature list resolved, before any `fit()`. |
| `src/models/game_outcome_model.py` | `_load_game_outcome_model()` | `predict_game()` calls it instead of hard-coded pkl path | VERIFIED | Line 401: `model, model_artifact_name = _load_game_outcome_model(artifacts_dir)`. |
| `src/models/game_outcome_model.py` | `src/outputs/prediction_store.py` | `write_game_prediction()` called at bottom of `predict_game()` | VERIFIED | Lines 447-457: non-fatal try/except wraps `write_game_prediction(result)` and `export_daily_snapshot(game_date)`. |
| `src/models/game_outcome_model.py` | `src/outputs/json_export.py` | `export_daily_snapshot()` called after write | VERIFIED | Same try/except block at lines 447-457. |
| `src/models/predict_cli.py` | `predict_game()` | `--date` arg passed as `game_date=args.date` | VERIFIED | Lines 29-32: `--date` argument added. Line 43: `predict_game(args.home, args.away, game_date=args.date)`. |
| `src/models/game_outcome_model.py` | `models/artifacts/game_outcome_metadata.json` | `json.dump()` in `train_game_outcome_model()` | WIRED (code) / NOT PRESENT (artifact) | Lines 342-370 write metadata.json with all required fields using Python builtins. Artifact not present because training has not run since code change. |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| FR-1.1 | 01-01 | Fix injury proxy join so missing_minutes and star_player_out reach model non-null | PARTIAL | Code fixed (injury_proxy.py, team_game_features.py). injury_proxy_features.csv rebuilt with 60.9% non-zero rows. But game_matchup_features.csv (the actual model input) is stale — all injury columns are 0. |
| FR-1.2 | 01-02 | Wire calibrated model into predict_cli inference path | PARTIAL | `_load_game_outcome_model()` wired — code correct. Calibrated artifact does not exist; inference currently uses uncalibrated model (with UserWarning). |
| FR-1.3 | 01-01 | Validate all feature columns have <95% null rate before training; fail loudly | VERIFIED | `validate_feature_null_rates()` raises ValueError with column name. Called in `train_game_outcome_model()`. |
| FR-6.1 | 01-03 | Append-only SQLite prediction store | VERIFIED | `predictions_history.db` exists, `game_predictions` table present. |
| FR-6.2 | 01-03 | Enable WAL mode on prediction store | VERIFIED | `PRAGMA journal_mode=WAL` in `_get_connection()`. Confirmed WAL via live query. |
| FR-6.3 | 01-03 | Export daily JSON snapshot | VERIFIED | `export_daily_snapshot()` writes `data/outputs/predictions_YYYYMMDD.json`. Live test passed. |
| FR-6.4 | 01-03 | Store prediction results with timestamps | VERIFIED | `created_at TEXT NOT NULL` (ISO8601 UTC) in schema. `datetime.utcnow().isoformat()` used. |
| FR-6.5 | 01-02 | Serialize model metadata as JSON alongside pickle artifacts | WIRED (code only) | Serialization code present in `train_game_outcome_model()`. `game_outcome_metadata.json` not present (training not re-run). |
| FR-7.1 | 01-04 | Each pipeline stage runs independently with documented inputs/outputs | VERIFIED | `docs/PIPELINE.md` documents all 6 stages with entry points, inputs, outputs, runtimes. |
| FR-7.2 | 01-04 | External data scrapers follow `src/data/get_*.py` module pattern | VERIFIED | PIPELINE.md section "External Data Scrapers" documents the pattern. Existing scrapers follow it. |
| FR-7.3 | 01-04 | `update.py` remains thin — each capability is one-line module import | VERIFIED | AST confirms 18 `src.*` imports, 0 subprocess calls. |
| FR-7.4 | 01-04 | Document pipeline stage order, dependencies, and expected runtime | VERIFIED | `docs/PIPELINE.md` has Stage Order table with runtimes, dependency graph, and daily-vs-rebuild distinction. |
| NFR-1 | 01-01 | All rolling features use `.shift(1)` before `.rolling()`; assert result.shape > 0 after joins | VERIFIED | `shift(1)` confirmed in `injury_proxy.py` (lines 136-141). `assert n_matched > 0` in `team_game_features.py`. |
| NFR-2 | 01-04 | Daily update completes in <15 minutes | VERIFIED (by doc) | PIPELINE.md states "Estimated runtime: 2-5 min for current-season fetch + preprocessing." Summary confirms update.py pre-dates this phase and runs in ~10 min. |
| NFR-3 | 01-02, 01-03 | All model outputs serializable to JSON; prediction history queryable via SQLite | VERIFIED | JSON metadata code uses Python builtins only. SQLite store queryable. JSON export writes valid JSON. |

---

## Anti-Pattern Scan

Scan performed on all 8 Phase 1 modified files:
- `src/features/injury_proxy.py`
- `src/features/team_game_features.py`
- `src/models/game_outcome_model.py`
- `src/outputs/prediction_store.py`
- `src/outputs/json_export.py`
- `src/outputs/__init__.py`
- `src/models/predict_cli.py`
- `docs/PIPELINE.md`

| File | Pattern | Severity | Impact |
|------|---------|----------|--------|
| (none found) | — | — | — |

No TODO, FIXME, placeholder, return null, or empty implementation patterns found.

---

## Human Verification Required

### 1. Rebuild features and verify injury columns are non-zero

**Test:** Run `python src/features/team_game_features.py` then run `python src/models/predict_cli.py game --home BOS --away LAL --date 2026-03-15`
**Expected:** The returned JSON includes `home_missing_minutes` and `home_star_player_out` with non-zero values for games where players are missing; the prediction row in `predictions_history.db` is populated
**Why human:** Requires ~5-10 min feature rebuild runtime, then inspecting the matchup CSV and a live inference call

### 2. Generate calibrated model artifact and verify it loads

**Test:** Run `python src/models/calibration.py` then run `python src/models/predict_cli.py game --home BOS --away LAL`
**Expected:** No `UserWarning` about calibrated model; `model_artifact` in the JSON output shows `game_outcome_model_calibrated.pkl`
**Why human:** Requires running calibration script (~1-2 min), then live inference check

---

## Gaps Summary

Two gaps block full success criterion achievement:

**Gap 1 (Stale features CSV — SC-1, FR-1.1):** The injury proxy code was fixed correctly (commit 2208360, Mar 1 2026). `injury_proxy_features.csv` was rebuilt immediately and shows 60.9% of team-games with non-zero missing minutes. However, Stage 3 (`python src/features/team_game_features.py`) was not re-run after the fix. `game_matchup_features.csv` (dated Feb 27) still contains all-zero injury columns. The code path is fully correct — this is a one-command fix: `python src/features/team_game_features.py`.

**Gap 2 (Missing calibrated artifact — SC-2, FR-1.2):** The `_load_game_outcome_model()` helper is correctly wired into `predict_game()` and prefers `game_outcome_model_calibrated.pkl`. The `UserWarning` fallback is correctly implemented (confirmed by live test). But `calibration.py` has never been run since the training artifacts were built (Feb 28). The calibrated artifact does not exist, so every inference call currently uses the uncalibrated model with a warning. Fix: `python src/models/calibration.py`.

**Root cause:** Both gaps are data/artifact state issues, not code correctness issues. The code changes are complete and correct. The phase is one short pipeline re-run away from full success criterion achievement.

**What is fully working:**
- Null-rate guard (SC-3, FR-1.3): prevent silent model training on broken features
- Prediction store + JSON export (SC-4, FR-6.1-6.4): WAL SQLite, timestamps, JSON snapshots
- Pipeline documentation (SC-5, FR-7.1-7.4): PIPELINE.md with all 6 stages
- All wiring: `_load_game_outcome_model`, `write_game_prediction`, `export_daily_snapshot`, `--date` arg
- update.py thin orchestrator pattern: 18 src.* imports, 0 subprocess calls

---

*Verified: 2026-03-02*
*Verifier: Claude (gsd-verifier)*
