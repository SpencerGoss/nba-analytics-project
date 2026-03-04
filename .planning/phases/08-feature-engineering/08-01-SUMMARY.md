---
phase: 08
plan: 01
subsystem: feature-engineering
tags: [lineup, net-rating, matchup-features, phase8]
dependency_graph:
  requires: [phase7/lineup-data]
  provides: [lineup-net-rating-features, matchup-feature-expansion]
  affects: [game_matchup_features.csv, model-training]
tech_stack:
  added: [src/features/lineup_features.py]
  patterns: [left-join-with-fillna, team-season-aggregation, weighted-average]
key_files:
  created:
    - src/features/lineup_features.py
    - data/processed/lineup_data.csv (generated, gitignored)
    - data/features/lineup_team_features.csv (generated, gitignored)
  modified:
    - src/features/team_game_features.py
    - data/features/game_matchup_features.csv (regenerated, gitignored)
decisions:
  - Left join on (season, team_id) so pre-2023-24 seasons get 0.0 fill (neutral)
  - Build lineup features from raw files if processed not available (fallback)
  - MIN_GP=5 threshold filters out lineups with tiny sample size
  - Lineup features are training path only (FR-4.4 boundary respected)
metrics:
  duration: ~15min
  completed: "2026-03-04"
  tasks_completed: 3
  files_created: 1
  files_modified: 1
---

# Phase 8 Plan 01: Lineup Net Rating Features Summary

Added 5-man lineup efficiency aggregates as per-team-season features to the game matchup dataset, providing the model direct access to lineup quality signals for 2023-24 and 2024-25 seasons.

## New Feature Columns Added

### Home/Away pairs (14 columns):
- `home_top1_lineup_net_rtg` / `away_top1_lineup_net_rtg` — best single lineup net rating (min 5 gp)
- `home_top3_lineup_net_rtg` / `away_top3_lineup_net_rtg` — avg net rating of top 3 most-used lineups
- `home_avg_lineup_net_rtg` / `away_avg_lineup_net_rtg` — weighted avg net rating (weighted by minutes)
- `home_lineup_net_rtg_std` / `away_lineup_net_rtg_std` — std dev of lineup net ratings (depth measure)
- `home_best_off_rating` / `away_best_off_rating` — highest offensive rating across lineups
- `home_best_def_rating` / `away_best_def_rating` — lowest defensive rating (best defense)
- `home_n_lineups` / `away_n_lineups` — count of qualifying lineups (gp >= 5)

### Differential columns (5 columns):
- `diff_top1_lineup_net_rtg` — home minus away best lineup net rating
- `diff_top3_lineup_net_rtg` — home minus away top-3 avg net rating
- `diff_avg_lineup_net_rtg` — home minus away weighted avg net rating
- `diff_best_off_rating` — home minus away best offensive rating
- `diff_best_def_rating` — home minus away best defensive rating

**Total new columns: 19**

## Non-Null Rate for Lineup Features

| Season | Rows | Non-Null Rate |
|--------|------|---------------|
| 2023-24 | 1,230 | 100.0% |
| 2024-25 | 1,225 | 100.0% |
| Earlier seasons | 65,710 | 0% (filled with 0.0 as designed) |

## Row Count Before/After

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Rows | 68,165 | 68,165 | 0 (no row drops) |
| Columns | 272 | 291 | +19 |

## Issues Encountered

**Issue 1:** `data/processed/lineup_data.csv` did not exist yet (preprocessing not run).
- Fix: `lineup_features.py` falls back to raw glob `data/raw/lineups/lineup_data_*.csv` automatically.
- Additionally ran a quick manual concat to create the processed file for pipeline completeness.

**Issue 2:** `build_team_game_features()` timed out at 5 min (the `close_playoff_race` computation is O(n^2) on conference standings).
- Fix: Ran only `build_matchup_dataset()` which reads the already-generated `team_game_features.csv` — no rebuild needed since we're only adding lineup joins in the matchup step.

**Issue 3:** Data files (`data/features/`, `data/processed/`) are gitignored.
- Expected behavior. Source code committed; generated CSVs regenerated on demand by running `python src/features/team_game_features.py`.

## Step 4 (availability_rate_10g): Not Implemented

After reviewing `injury_proxy.py`, `rotation_availability` is already computed per-game (not windowed). The task asked to check if a rolling mean already exists — it does not, but adding a 10-game rolling mean of `rotation_availability` would be a new column in `team_game_features.csv`. This is deferred because:
1. Context window constraints required prioritizing FEAT-01 and FEAT-02.
2. The existing `rotation_availability` column already flows through as a raw feature and as a diff column.
3. A rolling mean of it would be low-marginal-value given the injury features are already ranked 11-21 in SHAP.

## Self-Check

- [x] `src/features/lineup_features.py` created
- [x] `src/features/team_game_features.py` modified with lineup join
- [x] `data/features/lineup_team_features.csv` generated (60 team-seasons)
- [x] `data/features/game_matchup_features.csv` regenerated (68,165 rows x 291 cols)
- [x] All 59 tests pass
- [x] Commit hash: bbca3b5
