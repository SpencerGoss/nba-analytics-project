---
phase: 05-ats-model
plan: 01
subsystem: data
tags: [odds, betting, ats, kaggle, pandas, feature-engineering]

# Dependency graph
requires:
  - phase: 04-rest-schedule-features
    provides: game_matchup_features.csv with 272 columns, 68 features
provides:
  - src/data/get_historical_odds.py: Kaggle NBA betting dataset loader, normalizes to 3-letter abbreviations
  - src/features/ats_features.py: build_ats_features() joining matchup + betting odds
  - data/features/game_ats_features.csv: ATS feature table with spread, implied probs, covers_spread
affects:
  - 05-ats-model (plans 02+): ATS classifier training uses game_ats_features.csv
  - future-inference: value-bet detector reads home_implied_prob from this table

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Kaggle NBA betting dataset (nba_2008-2025.csv) as historical odds source"
    - "Two-path file loading: canonical name + fallback filename"
    - "Vectorized no-vig probability computation via multiplicative normalization"
    - "Data separation guard: assert forbidden columns not in matchup features at build time"

key-files:
  created:
    - src/data/get_historical_odds.py
    - src/features/ats_features.py
    - data/features/game_ats_features.csv
  modified: []

key-decisions:
  - "Kaggle dataset uses lowercase short codes (gs, sa, no, utah, wsh) not full names -- KAGGLE_TEAM_TO_ABB maps these to 3-letter project abbreviations"
  - "Season in Kaggle data is end-year integer (2008 = 2007-08) -- converted to project format 200708 via _normalize_season()"
  - "Fallback file lookup: load_and_normalize_odds() checks nba_2008-2025.csv if canonical nba_betting_historical.csv not found"
  - "Vectorized implied prob computation replaces row-by-row loop for 23K rows (performance)"
  - "Data separation guard uses assert at build time: any future contamination of matchup features with spread data raises immediately"
  - "Inner join produces 18,496 rows after excluding anomalous seasons -- well above 5000-row success criterion"

patterns-established:
  - "Pattern: load_and_normalize_odds() as single source of truth for odds data normalization"
  - "Pattern: Inner join on game_date + home_team + away_team (same cross-source key as Phase 3 referee features)"
  - "Pattern: Pushes (id_spread=2) stored as NaN in covers_spread -- excluded from training by dropna()"

requirements-completed: [FR-5.1, FR-5.4, NFR-1]

# Metrics
duration: 18min
completed: 2026-03-02
---

# Phase 5 Plan 01: ATS Feature Table Summary

Kaggle NBA betting dataset (23K games, 2007-2025) joined with matchup features to produce 18,496-row ATS feature table with spread, no-vig implied probabilities, and covers_spread target

## Performance

- **Duration:** 18 min
- **Started:** 2026-03-02T18:41:39Z
- **Completed:** 2026-03-02T19:00:01Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Downloaded Kaggle NBA betting dataset (23,118 games, Oct 2007 - Jun 2025) -- accessible without Kaggle auth via API endpoint
- Built `load_and_normalize_odds()` mapping 30 lowercase short codes (gs, sa, no, utah, wsh) to 3-letter abbreviations
- Built `build_ats_features()` producing 18,496-row ATS feature table via inner join on game_date + home_team + away_team
- Data separation guard verified: game_matchup_features.csv contains zero spread/odds columns
- No-vig implied probabilities correctly sum to 1.0000 (multiplicative normalization removes bookmaker overround)
- Pushes (263 games, id_spread=2) stored as NaN in covers_spread per plan spec

## Task Commits

1. **Task 1: Download Kaggle historical odds and build normalization module** - `3772095` (feat)
2. **Task 2: Build ATS feature table by joining matchup features with odds** - `9881f95` (feat)

**Plan metadata:** (docs commit -- see below)

## Files Created/Modified

- `src/data/get_historical_odds.py` - Loads Kaggle betting CSV, maps team names, computes no-vig probs, normalizes seasons
- `src/features/ats_features.py` - build_ats_features() joins matchup + odds, adds spread/implied_prob/covers_spread, saves CSV
- `data/features/game_ats_features.csv` - 18,496 rows, 276 columns (272 matchup + 4 ATS columns)

## Decisions Made

- **Kaggle team name format discovery:** Dataset uses lowercase short codes (gs, sa, no, utah, wsh) not full names or condensed CamelCase as research predicted. KAGGLE_TEAM_TO_ABB built from inspecting actual unique team values in the downloaded file.
- **Season normalization:** Kaggle stores season as end-year integer (2008 = 2007-08 season). _normalize_season() converts to project format (200708) via `(year-1)(year%100)`.
- **Fallback filename:** Downloaded file is nba_2008-2025.csv; code checks this as fallback if canonical nba_betting_historical.csv not found. Both names persisted in data/raw/odds/ (gitignored).
- **Vectorized implied prob computation:** Row-by-row loop replaced with vectorized numpy computation for 23K rows.
- **Data separation guard at build time:** assert checks matchup columns before join, raises immediately if violated. Future-proofs against accidental contamination.

## Deviations from Plan

### Auto-fixed Issues

### 1. [Rule 1 - Bug] pandas infer_datetime_format keyword removed in newer versions

- **Found during:** Task 1 (testing load_and_normalize_odds)
- **Issue:** `pd.to_datetime(..., infer_datetime_format=True)` raises TypeError in pandas 2.x (parameter removed)
- **Fix:** Removed the `infer_datetime_format=True` keyword argument; pandas 2.x auto-detects format
- **Files modified:** src/data/get_historical_odds.py
- **Verification:** Function runs without error; dates parse correctly to YYYY-MM-DD
- **Committed in:** 3772095 (Task 1 commit)

### 2. [Rule 1 - Bug] KAGGLE_TEAM_TO_ABB needed complete rebuild from actual data

- **Found during:** Task 1 (inspecting downloaded Kaggle file)
- **Issue:** Research predicted CamelCase names (GoldenState, LAClippers); actual dataset uses lowercase short codes (gs, sa, no, utah, wsh)
- **Fix:** Rewrote entire KAGGLE_TEAM_TO_ABB mapping dict based on actual unique team values in the file
- **Files modified:** src/data/get_historical_odds.py
- **Verification:** All 30 teams map correctly; 18,496 rows after join (not 0)
- **Committed in:** 3772095 (Task 1 commit)

---

Total deviations: 2 auto-fixed (1 API compatibility, 1 data format correction)

Impact on plan: Both auto-fixes necessary for correctness. No scope creep. Inner join producing 18,496 rows confirms mapping is correct.

## Issues Encountered

- **Kaggle authentication:** No kaggle.json credentials available. Discovered the dataset is publicly accessible via the Kaggle API v1 endpoint without authentication. Downloaded via urllib.request as a zip, extracted nba_2008-2025.csv.

## User Setup Required

None - the Kaggle dataset was downloaded automatically. The file is saved to `data/raw/odds/` (gitignored). Future environments should re-download if needed; `get_historical_odds.py` prints clear instructions if the file is missing.

## Next Phase Readiness

- `game_ats_features.csv` is ready for ATS model training (Plan 05-02)
- 18,496 rows with valid `covers_spread` targets (18,233 non-push rows for training)
- 15,473 rows with valid implied probabilities (3,298 ESPN-sourced rows post-Jan 2023 have NaN moneyline)
- Data separation guard confirmed: game_matchup_features.csv has no spread/odds columns

## Self-Check: PASSED

All created files verified:

- FOUND: src/data/get_historical_odds.py
- FOUND: src/features/ats_features.py
- FOUND: data/features/game_ats_features.csv
- FOUND: .planning/phases/05-ats-model/05-01-SUMMARY.md

All commits verified:

- FOUND: 3772095 (Task 1)
- FOUND: 9881f95 (Task 2)

---
*Phase: 05-ats-model*
*Completed: 2026-03-02*
