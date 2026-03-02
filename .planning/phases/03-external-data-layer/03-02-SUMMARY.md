---
phase: 03-external-data-layer
plan: 02
subsystem: features
tags: [referee-features, rolling-stats, feature-engineering, anti-leakage, nba-analytics]

# Dependency graph
requires:
  - phase: 03-external-data-layer
    plan: 01
    provides: "bref_scraper.py with get_referee_crew_assignments(); referee_crew/ output dir"
  - "data/processed/team_game_logs.csv — FTA and pace per team-game"
provides:
  - "src/features/referee_features.py with build_referee_features()"
  - "Rolling crew foul-rate features: ref_crew_fta_rate_roll10/20, ref_crew_pace_impact_roll10"
  - "data/features/referee_features.csv (when scrape data exists)"
affects: [src/features/team_game_features.py, data/features/team_game_features.csv, data/features/game_matchup_features.csv]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Melt crew assignments (referee_1/2/3) to long format for per-referee rolling computation"
    - "shift(1) BEFORE rolling() for no-lookahead referee stats (NFR-1)"
    - "NaN (not 0) for pre-scrape seasons -- SimpleImputer handles downstream"
    - "Join on (game_date, home_abbr_for_join) so both home and away rows get same referee data"
    - "Try/except wrapper with match-count print following injury proxy pattern"

key-files:
  created:
    - "src/features/referee_features.py -- build_referee_features() with rolling crew stats"
  modified:
    - "src/features/team_game_features.py -- referee join block + context_cols + diff_stats"

key-decisions:
  - "Join on game_date + home_team abbreviation (not game_id) -- bref uses its own game ID format incompatible with NBA API; home team abbreviation is the reliable cross-source key"
  - "home_abbr_for_join derived column used to join both home and away rows -- is_home==1 uses own abbr; is_home==0 uses opponent_abbr (which equals home team)"
  - "Separate referee_features.py module (not inline in team_game_features.py) per RESEARCH.md Open Question 4 recommendation -- keeps team_game_features.py from becoming a monolith"
  - "ref_crew_fta_rate diff_ added to matchup dataset even though same crew for both teams -- diff will be ~0 for properly joined rows; get_feature_cols() auto-pickup via diff_ prefix is the reason"
  - "Empty-data graceful degradation: when no CSV files exist in referee_crew/, build_referee_features() returns correct empty DataFrame schema; team_game_features.py try/except swallows the empty case"

# Metrics
duration: 3min
completed: 2026-03-02
---

# Phase 3 Plan 02: Referee Foul-Rate Feature Module Summary

**Per-crew rolling FTA/game and pace-impact features (10/20-game windows) from scraped Basketball Reference referee assignments, wired into team_game_features and matchup dataset with NaN propagation for pre-scrape seasons**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-03-02T07:00:41Z
- **Completed:** 2026-03-02T07:03:36Z
- **Tasks:** 2
- **Files modified:** 2 (referee_features.py, team_game_features.py)

## Accomplishments

- Created `src/features/referee_features.py` with `build_referee_features()`:
  - Loads all `referee_crew_*.csv` files from `data/raw/external/referee_crew/`
  - Melts crew assignments (referee_1/2/3) to long format for per-referee rolling
  - Joins FTA and pace from `team_game_logs` on game_date + home team abbreviation
  - Computes rolling 10-game and 20-game FTA averages per referee with `shift(1)` (NFR-1)
  - Averages across 3-ref crew for crew-level: `ref_crew_fta_rate_roll10/20`
  - Computes `ref_crew_pace_impact_roll10`: crew avg poss minus league avg poss (rolling)
  - Returns empty DataFrame gracefully when no scrape data exists yet
- Updated `src/features/team_game_features.py`:
  - Added referee join block after injury proxy (try/except, prints match count)
  - Uses `home_abbr_for_join` derived column to join both home AND away rows
  - Added referee columns to `context_cols` in `build_team_game_features()`
  - Added referee columns to `context_cols` in `build_matchup_dataset()`
  - Added referee to `diff_stats` list for `diff_ref_crew_fta_rate_roll10/20` auto-pickup

## Task Commits

Each task was committed atomically:

1. **Task 1: Build referee foul-rate feature module** - `c4ae352` (feat)
2. **Task 2: Wire referee features into team_game_features and matchup dataset** - `1335e01` (feat)

## Files Created/Modified

- `src/features/referee_features.py` - Full referee feature module (new)
- `src/features/team_game_features.py` - Added join block, context_cols, diff_stats entries

## Decisions Made

- **Join on game_date + home team abbreviation:** Basketball Reference uses its own game_id format (e.g., "202501010LAL") incompatible with NBA API game IDs. The home team 3-char abbreviation is the reliable cross-source join key for modern seasons.

- **home_abbr_for_join derived column:** Each team-game row needs to carry the home team's abbreviation for the join. `is_home==1` rows use own `team_abbreviation`; `is_home==0` rows use `opponent_abbr` (which equals the home team). A single merge handles both cases.

- **Separate referee_features.py module:** Per RESEARCH.md Open Question 4 recommendation -- avoids making team_game_features.py a monolith. The module can be tested and extended independently.

- **NaN propagation (not zero):** Per Pitfall 6 in RESEARCH.md -- referee features are absent for pre-scrape seasons (2013-14 through whenever the backfill runs). Zero would inject false signal. NaN is correct; SimpleImputer in the model handles via mean imputation.

- **diff_ prefix on referee columns:** `diff_ref_crew_fta_rate_roll10/20` are auto-discovered by `get_feature_cols()` in `game_outcome_model.py` via the `startswith("diff_")` filter. No model code changes needed.

## Deviations from Plan

### Auto-fixed Issues

None - plan executed exactly as written.

The join design required one implementation decision (home_abbr_for_join approach) not explicitly in the plan but in line with the plan's "adjust join keys based on 03-01 output format" guidance.

## Issues Encountered

- No live referee CSV data exists in `data/raw/external/referee_crew/` (only `.gitkeep`) because Basketball Reference is Cloudflare-blocked in this environment. The module handles this gracefully -- `build_referee_features()` returns an empty DataFrame with correct column schema; the join block in `team_game_features.py` prints a message and skips. No referee feature columns will appear in the output until the scraper is run from a non-blocked environment.

## User Setup Required

To populate referee features:
1. From a non-Cloudflare-blocked environment (cloud VM, deployed server, etc.), run:
   ```bash
   python -c "from src.data.external.bref_scraper import get_referee_crew_assignments; get_referee_crew_assignments('2013-10-01', '2026-03-01')"
   ```
   Runtime: ~3 hours for 2013-14 through current season (~12 seasons x ~1,230 games x 3 sec/game)
2. Then rebuild features:
   ```bash
   python src/features/team_game_features.py
   ```
3. Referee feature columns will now appear in `data/features/team_game_features.csv` and `data/features/game_matchup_features.csv`

## Self-Check: PASSED

- [x] `src/features/referee_features.py` EXISTS
- [x] `src/features/team_game_features.py` MODIFIED (grep confirms referee columns)
- [x] Commit `c4ae352` EXISTS (feat: build referee foul-rate feature module)
- [x] Commit `1335e01` EXISTS (feat: wire referee features into team_game_features)
- [x] `from src.features.referee_features import build_referee_features` IMPORT OK
- [x] `from src.features.team_game_features import build_team_game_features` IMPORT OK
- [x] `shift(1)` before `rolling()` confirmed in referee_features.py (3 instances)
- [x] No `.fillna(0)` on referee feature columns confirmed
- [x] `ref_crew_fta_rate` grep shows 5 lines in team_game_features.py
- [x] `referee_features` import in try/except block confirmed

---
*Phase: 03-external-data-layer*
*Completed: 2026-03-02*
