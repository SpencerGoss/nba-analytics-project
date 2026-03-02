---
phase: 05-ats-model
plan: 04
subsystem: models
tags: [ats, backtest, roi, clv, hit-rate, betting, value-bet, -110-vig]

# Dependency graph
requires:
  - phase: 05-ats-model
    plan: 02
    provides: models/artifacts/ats_model.pkl + predict_ats()
  - phase: 05-ats-model
    plan: 03
    provides: detect_value_bets(), edge computation

provides:
  - src/models/ats_backtest.py: ATS backtest harness with ROI, CLV, hit rate reporting
  - reports/ats_backtest.csv: Per-season ATS backtest results (16 seasons)
  - reports/ats_backtest_summary.txt: Human-readable backtest summary

affects:
  - phase-06-inference: Backtest numbers inform production readiness decision
  - daily-value-bet-scan: Edge threshold calibration informed by value-bet backtest

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "compute_roi_flat_110(): flat -110 vig ROI receives bet_correct (pred==actual) not raw covers_spread"
    - "500-game hard guard: ValueError if fewer than 500 non-push games in baseline"
    - "Two-mode backtest: baseline (all games) + value-bet filtered (|edge| > threshold)"
    - "Holdout OOS section: separate reporting for test_seasons not used in training"
    - "CLV documented as 0.0 data limitation: no closing line column in Kaggle dataset"

key-files:
  created:
    - src/models/ats_backtest.py
    - reports/ats_backtest.csv
    - reports/ats_backtest_summary.txt
  modified: []

key-decisions:
  - "bet_correct (pred==actual) passed to compute_roi_flat_110, not raw covers_spread -- ROI measures correctness of bets placed, not whether home team covered"
  - "CLV is 0.0 for all games: Kaggle dataset has single spread column (opening only); documented as limitation in reports"
  - "Baseline backtest uses ALL seasons including training data: in-sample results noted as optimistic; holdout OOS section provides honest estimate"
  - "Value-bet threshold 5pp (0.05 default): 13,170 of 18,233 games qualify; all exceed 500-game minimum"
  - "Avg edge N/A for holdout seasons 202324/202425: 100% NaN implied_prob in those seasons (ESPN-sourced rows post-Jan 2023)"

# Metrics
duration: 22min
completed: 2026-03-02
---

# Phase 5 Plan 04: ATS Backtest Harness Summary

ATS backtest harness built and executed: flat -110 ROI reporting over 18,233 non-push games across 16 seasons, with 500-game guard, value-bet filtered mode (53.05% hit rate), and honest holdout OOS reporting (51.20% hit rate, -2.25% ROI on 2,455 unseen games).

## Performance

- **Duration:** 22 min
- **Started:** 2026-03-02T19:59:37Z
- **Completed:** 2026-03-02T20:21:09Z
- **Tasks:** 2
- **Files created:** 3

## Accomplishments

- Created `src/models/ats_backtest.py` with full backtest harness:
  - `compute_roi_flat_110()`: flat -110 vig ROI with 500-game minimum guard (ValueError on insufficient sample)
  - `compute_clv_spread()`: CLV helper (returns 0.0 when no closing line data; documents limitation)
  - `run_ats_backtest()`: loads ATS model, generates predictions, computes edge/CLV/bet_correct, runs baseline + value-bet filtered modes
  - `write_backtest_reports()`: per-season CSV + human-readable summary with model metadata
- 500-game guard tested and verified (raises ValueError with exact count in message)
- Ran full backtest: 18,233 games, 16 seasons (2007-08 through 2024-25), 263 push rows excluded
- Generated `reports/ats_backtest.csv` and `reports/ats_backtest_summary.txt`

## Task Commits

1. **Task 1: Build ATS backtest harness with ROI, CLV, and hit rate** - `36d2b3f` (feat)
2. **Task 2: Run full backtest and generate reports** - `2beb88b` (feat, includes Rule 1 bug fix)

## Files Created/Modified

- `src/models/ats_backtest.py` - ATS backtest harness: compute_roi_flat_110, compute_clv_spread, run_ats_backtest, write_backtest_reports
- `reports/ats_backtest.csv` - Per-season breakdown: season, n_games, wins, losses, hit_rate, roi, avg_clv, avg_edge
- `reports/ats_backtest_summary.txt` - Human-readable summary: baseline + holdout + value-bet + per-season table + CLV note + model metadata

## Backtest Results Summary

| Mode | Games | Hit Rate | ROI | Notes |
|------|-------|----------|-----|-------|
| Baseline (all seasons) | 18,233 | 52.73% | +0.67% | In-sample; optimistic |
| Value-bet filtered (edge>5%) | 13,170 | 53.05% | +1.28% | In-sample; better subset |
| Holdout OOS (202324/202425) | 2,455 | 51.20% | -2.25% | Honest out-of-sample |

**Vig breakeven: 52.38%**

The holdout OOS result (51.20%) is below vig breakeven (-2.25% ROI), which is the expected and honest result. The ATS model establishes a baseline; the value-bet strategy of combining ATS predictions with win-probability disagreement is the Phase 6 objective.

## Per-Season Highlights

Best season: 200708 (57.11% hit rate, +9.04% ROI) -- earliest season, potentially easiest market
Worst holdout: 202425 (49.22% hit rate, -6.03% ROI) -- most recent, most efficient market

## Decisions Made

- **bet_correct (pred==actual) is the correct ROI input:** ROI measures whether your bet was correct, not whether the home team covered. When the model predicts away covers (pred=0) and away actually covers (actual=0), that is a winning bet. Passing raw `covers_spread` values would incorrectly count away-cover wins as losses.

- **CLV documented as limitation:** The Kaggle dataset stores only one spread column (opening). No separate closing line is available. CLV is 0.0 for all games and the limitation is clearly documented in the summary report. True CLV measurement would require a dataset with both opening and closing lines.

- **Holdout OOS separate from baseline:** The summary report explicitly separates holdout season results (202324/202425) from the baseline (all seasons) to provide an honest estimate of out-of-sample performance. In-sample metrics are labeled accordingly.

- **Avg edge N/A for holdout seasons:** Post-Jan 2023 games in game_ats_features.csv have 100% NaN implied_prob (ESPN-sourced rows without moneyline data). This means edge cannot be computed for the two holdout seasons. This is expected behavior -- value-bet detection requires live odds via The Odds API for current games.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] compute_roi_flat_110 received raw covers_spread instead of bet_correct**

- **Found during:** Task 2 (running first backtest execution)
- **Issue:** The function received `df["covers_spread"]` (0 or 1 = whether home team covered) instead of a Series indicating whether the bet was correct (prediction matched actual). When the model predicts away covers (pred=0) and away actually covers (actual=0), `covers_spread=0` was treated as a "losing" bet, when it should be a win. This produced an incorrect hit rate of 49.4% and negative ROI instead of the correct 52.7% hit rate.
- **Fix:** Added `df["bet_correct"] = (df["covers_spread"] == df["covers_spread_pred"]).astype(int)` and passed this Series to all `compute_roi_flat_110()` calls. Updated docstring to clarify the input semantics.
- **Files modified:** `src/models/ats_backtest.py`
- **Verification:** Hit rate of 52.73% matches expected TP+TN/total (9615/18233); ROI of +0.67% is plausible for in-sample evaluation.
- **Committed in:** `2beb88b` (Task 2 commit)

## Self-Check: PASSED

All created files verified:
- FOUND: src/models/ats_backtest.py
- FOUND: reports/ats_backtest.csv
- FOUND: reports/ats_backtest_summary.txt

All commits verified:
- FOUND: 36d2b3f (Task 1 - ATS backtest harness)
- FOUND: 2beb88b (Task 2 - reports + bug fix)

Per-season CSV: 16 seasons, 18,233 total games (>= 500 guard satisfied)

---
*Phase: 05-ats-model*
*Completed: 2026-03-02*
