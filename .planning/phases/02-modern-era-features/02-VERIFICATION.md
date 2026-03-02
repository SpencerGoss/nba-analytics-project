---
phase: 02-modern-era-features
verified: 2026-03-02T05:31:50Z
status: gaps_found
score: 4/5 must-haves verified
re_verification: false
gaps:
  - truth: "Model accuracy on the 2014+ holdout set meets or exceeds the accuracy on the full historical dataset"
    status: partial
    reason: "Modern era model (0.6684) is 0.0070 below full-history GB model (0.6754) on the same 202324/202425 holdout. Documented as a research finding in 02-03-SUMMARY.md (fewer validation splits: 4 vs 19 for modern era). The era filtering code is correct; the accuracy gap is real and persists when controlling for model type."
    artifacts:
      - path: "models/artifacts/game_outcome_metadata.json"
        issue: "test_accuracy=0.6684 vs full-history 0.6754 — SC5 states modern >= full"
    missing:
      - "Either demonstrate modern-era accuracy >= full-history via controlled comparison (same model type, same splits), or accept this as a known limitation and update SC5 wording in ROADMAP.md to reflect the research finding"
---

# Phase 2: Modern Era Features Verification Report

**Phase Goal:** The game outcome model is trained exclusively on modern NBA data (2014+) and uses pace-normalized efficiency metrics that exist in the database but were never wired into the feature pipeline
**Verified:** 2026-03-02T05:31:50Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (from ROADMAP.md Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| SC1 | Model training filters to seasons 2013-14 through 2023-24, explicitly excluding 2019-20 bubble and 2020-21 shortened seasons | VERIFIED | `MODERN_ERA_ONLY=True`, `MODERN_ERA_START="201314"`, `EXCLUDED_SEASONS=["201920","202021"]` in game_outcome_model.py; 2,139 games excluded; training seasons confirmed: 201314-202526 minus test holdout |
| SC2 | Feature matrix includes rolling ORtg, DRtg, net rating, eFG%, TS%, pace from team box scores — all with shift(1) before rolling() | VERIFIED | 30 ADV_ROLL_STATS columns (10 metrics x 3 windows) confirmed in team_game_features.csv (121 cols total); `_rolling_mean_shift()` applies `.shift(1)` before `.rolling()`; ORtg mean=108.09, pace mean=100.67 for 2014+ data (sane ranges) |
| SC3 | Turnover rate computed as turnovers per possession and appears as rolling feature | VERIFIED | `tov_poss_game_roll5/10/20` columns present in team_game_features.csv; Oliver formula: `tov / (fga - oreb + tov + 0.44*fta)`; mean=0.1406 for 2014+ data |
| SC4 | Four Factors differential composite (eFG%, TOV%, ORB%, FT rate) exists as matchup feature | VERIFIED | `diff_four_factors_composite` in game_matchup_features.csv (68,165 non-null); Dean Oliver weights (eFG%=+0.40, TOV%=-0.25, OREB%=+0.20, FTR=+0.15) confirmed in `FOUR_FACTORS_WEIGHTS`; auto-picked by `get_feature_cols()` via `startswith("diff_")` |
| SC5 | Model accuracy on 2014+ holdout meets or exceeds full historical dataset accuracy | PARTIAL | Modern era: 0.6684; Full history GB: 0.6754 (documented in 02-03-SUMMARY.md deviation). Gap = 0.0070. Code is correctly implemented; the accuracy gap is a research finding attributed to fewer validation splits (4 vs 19) when training on modern era only |

**Score: 4/5 truths fully verified** (SC1, SC2, SC3, SC4 pass; SC5 partial)

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/features/team_game_features.py` | ADV_ROLL_STATS, opp_box join, per-game advanced metrics, FOUR_FACTORS_WEIGHTS, _four_factors_composite() | VERIFIED | All present: `ADV_ROLL_STATS` at module level (10 metrics), single `opp_box` self-join, Oliver formula possession estimates, 10 per-game metrics, `FOUR_FACTORS_WEIGHTS` dict, `_four_factors_composite()` defined before `build_matchup_dataset()` |
| `data/features/team_game_features.csv` | 30 advanced rolling columns, no raw per-game metric leakage | VERIFIED | 121 columns total; 30 ADV_ROLL_STATS columns confirmed present with non-null values; raw game metrics (off_rtg_game, etc.) correctly excluded from output |
| `src/models/game_outcome_model.py` | MODERN_ERA_ONLY=True, MODERN_ERA_START="201314", EXCLUDED_SEASONS, excluded_seasons param | VERIFIED | All constants set correctly; `excluded_seasons` parameter wired into `train_game_outcome_model()` with correct filter logic |
| `data/features/game_matchup_features.csv` | diff_four_factors_composite + 13 advanced diff columns, no _x/_y artifacts | VERIFIED | 264 columns; `diff_four_factors_composite` (68,165 non-null); all 13 advanced diff columns present (ORtg/DRtg/net_rtg x 3 windows + pace/eFG%/TS%/tov_poss at roll20); zero _x/_y merge artifacts; existing features (diff_three_style_mismatch, diff_rebounding_edge, etc.) preserved |
| `models/artifacts/game_outcome_model.pkl` | Trained model artifact on modern era data | VERIFIED | File exists; trained model saved |
| `models/artifacts/game_outcome_metadata.json` | train_start_season=201314, modern_era_only=True, excluded_seasons, advanced features in feature_list | VERIFIED | All metadata fields correct; 60 features including 14 advanced efficiency diff columns; test_accuracy=0.6684 |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `build_team_game_features()` opp_box self-join | per-game metric computation block | Single expanded `opp_box` merge providing `opp_pts_raw, opp_fga, opp_fg3m, opp_fgm, opp_oreb, opp_tov, opp_fta, opp_dreb` | WIRED | Exactly 1 `df = df.merge(opp_box, ...)` call confirmed; assert guard `df["opp_fga"].notna().sum() > 0` present |
| per-game metric columns | ADV_ROLL_STATS rolling loop | `_rolling_mean_shift()` with `shift(1)` | WIRED | `adv_group.apply(lambda g, s=stat: _rolling_mean_shift(g, s, window), ...)` pattern confirmed; default-arg capture prevents late-binding closure bug |
| `MODERN_ERA_ONLY = True` | training data filter | `start_season = MODERN_ERA_START if modern_era_only else TRAIN_START_SEASON` | WIRED | Filter applied before train/test split; `df[df["season"].astype(str) >= start_season]` |
| `EXCLUDED_SEASONS` | training data filter | `df[~df["season"].astype(str).isin(excluded_seasons)]` | WIRED | Exclusion logic present after era filter; 2,139 games correctly excluded from 201920/202021 |
| `FOUR_FACTORS_WEIGHTS` | `diff_four_factors_composite` column | `_four_factors_composite(matchup, FOUR_FACTORS_WEIGHTS)` called after diff_stats loop | WIRED | `_four_factors_composite()` defined before `build_matchup_dataset()` (NameError prevention confirmed); called at correct position |
| `off_rtg_game_roll*` columns | `diff_stats` list in `build_matchup_dataset()` | Entries in `diff_stats` list: "off_rtg_game_roll5", "off_rtg_game_roll10", "off_rtg_game_roll20", etc. | WIRED | 13 advanced metric roll columns in `diff_stats`; `diff_` prefix auto-picked by `get_feature_cols()` via `startswith("diff_")` |
| `diff_four_factors_composite` | model feature selection | `get_feature_cols()` `diff_cols = [c for c in numeric_cols if c.startswith("diff_")]` | WIRED | Confirmed: `diff_four_factors_composite` in model's 60-feature list |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| FR-2.1 | 02-01 | ORtg, DRtg, net_rtg, eFG%, TS% as rolling averages (5/10/20 windows) | SATISFIED | 15 columns confirmed in team_game_features.csv: off_rtg_game_roll5/10/20, def_rtg_game_roll5/10/20, net_rtg_game_roll5/10/20, efg_game_roll5/10/20, ts_game_roll5/10/20 |
| FR-2.2 | 02-01 | Pace (possessions/game) as rolling feature | SATISFIED | pace_game_roll5/10/20 confirmed present; mean=100.67 for 2014+ games |
| FR-2.3 | 02-01 | Turnover rate per possession (not raw count) as rolling feature | SATISFIED | tov_poss_game_roll5/10/20 confirmed; computed as `tov / poss_est` via Oliver formula |
| FR-2.4 | 02-02 | Four Factors differential composite (eFG%, TOV%, ORB%, FT rate) as matchup feature | SATISFIED | diff_four_factors_composite in game_matchup_features.csv (68,165 non-null); Dean Oliver weights confirmed |
| FR-2.5 | 02-03 | Restrict model training to 2014+ seasons; exclude 2019-20 bubble and 2020-21 shortened seasons | SATISFIED | MODERN_ERA_ONLY=True, MODERN_ERA_START="201314", EXCLUDED_SEASONS=["201920","202021"] confirmed; code path verified |
| NFR-1 | 02-01, 02-02, 02-03 | All rolling features use shift(1) before rolling(); assert after joins; log null rates | SATISFIED | `_rolling_mean_shift()` applies shift(1); assert guard after opp_box join; null rate diagnostics printed for each ADV_ROLL_STAT roll20 column |

No orphaned requirements: all 6 requirements (FR-2.1, FR-2.2, FR-2.3, FR-2.4, FR-2.5, NFR-1) from the PLAN frontmatter are accounted for and satisfied at the code level.

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/features/team_game_features.py` | — | No TODOs, placeholders, or empty implementations | — | None |
| `src/models/game_outcome_model.py` | — | No TODOs, placeholders, or empty implementations | — | None |

No anti-patterns found in either modified file.

---

## Human Verification Required

### 1. SC5 Accuracy Comparison — Research Finding Acceptance

**Test:** Run `train_game_outcome_model()` with `modern_era_only=True` (default) and compare against `train_game_outcome_model(modern_era_only=False, excluded_seasons=[])` using identical model type and seed.

**Expected:** Modern era accuracy >= full history accuracy, or a decision to accept the ~0.7 pct point gap as acceptable given the non-stationarity argument.

**Why human:** The SUMMARY documents a confirmed accuracy inversion (modern=0.6684, full=0.6754). The code is correct but the ROADMAP SC5 states "meets or exceeds." A human needs to decide: (a) accept the gap as a known limitation of fewer training samples and re-word SC5, or (b) investigate whether additional features or era boundary adjustment closes the gap. The 0.7 pct pt gap is small but SC5 as written is not technically met.

---

## Gaps Summary

**One gap blocks the ROADMAP Success Criterion SC5:**

SC5 states the modern era model must achieve accuracy "meets or exceeds" full historical accuracy. The actual result (modern=0.6684, full=0.6754) inverts this by 0.7 percentage points. All code implementation is correct and the Plan's own verification checks passed (accuracy > 0.60 minimum). The SUMMARY documents this as a research finding — GBM models benefit from more training data (28k vs 10k games), and model selection noise is higher with only 4 validation splits vs 19.

**All code implementation is correct and the gap is a research finding, not a code bug.** The era filtering, advanced metric pipeline, Four Factors composite, and wiring are all verified. The gap is specifically the measurable accuracy comparison claimed in SC5.

**Options for resolution:**
1. Accept the gap and update ROADMAP SC5 to "meets or exceeds 66% on modern-era holdout" (the >60% threshold is met)
2. Investigate whether adding more advanced features in Phase 3/4 closes the gap
3. Accept that era filtering improves generalizability (reduces non-stationarity) even at a small accuracy cost on current holdout seasons

---

_Verified: 2026-03-02T05:31:50Z_
_Verifier: Claude (gsd-verifier)_
