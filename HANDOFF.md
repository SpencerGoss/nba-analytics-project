# Handoff — NBA Analytics Project

_Last updated: 2026-03-06_

## What Was Built

**Phase 1 Remaining Items — COMPLETE (v2.2)**

All 4 items from the research plan implemented:

### 1. LightGBM Candidate Model (DONE)
- `src/models/game_outcome_model.py` — guarded import (`_LGBM_AVAILABLE` flag), LightGBM added as `sklearn.Pipeline` candidate in expanding-window CV
- `requirements.txt` — `lightgbm>=4.0.0` added (installed 4.6.0)
- Result: gradient_boosting still selected (67.1% acc, AUC 0.7406); LightGBM competed but didn't win without HPO

### 2. Pythagorean Win% Feature (DONE)
- `src/features/team_game_features.py`:
  - `pythagorean_win_pct_game` — per-game (Morey exponent 14.3, pts clipped at 1)
  - `pythagorean_win_pct_roll10` — 10-game rolling with shift(1), no data leakage
  - `diff_pythagorean_win_pct_roll10` added to `diff_stats` in `build_matchup_dataset()`
- Matchup CSV now 68,216 rows × 296 cols (was 291)
- **Gotcha:** `_roll` in the col name means it's auto-captured by `roll_cols` filter — do NOT add to `context_cols` (causes duplicate column ValueError)

### 3. Fractional Kelly Sizing (DONE)
- `src/models/value_bet_detector.py` — `_compute_kelly_fraction()` helper added
- Formula: `f = 0.5 * (p*b - (1-p)) / b` where `b = (1-q)/q` (no-vig implied odds)
- `kelly_fraction` field now appears in every `get_strong_value_bets()` output dict
- Home/away side-aware (bet_side field used to select correct probability)

### 4. CLV Tracking (DONE)
- `src/models/clv_tracker.py` — new file (~180 lines)
  - `CLVTracker` class; `clv_tracking` table in `predictions_history.db`
  - `log_opening_line()` — INSERT OR IGNORE (idempotent, called at fetch time)
  - `update_closing_line()` — computes `clv = opening_spread - closing_spread`
  - `get_clv_summary()` — returns mean_clv, positive_clv_rate, has_edge flag (n>=10, mean>0, rate>0.5)
- `scripts/fetch_odds.py` — step 1b: logs opening lines after `game_lines.csv` saved (non-fatal)
- **CLV formula:** positive = we got a better line than where market settled (e.g., logged -3.5, closed -5.5 → CLV=+2.0)

## Current State

- Raw data: fresh (Mar 5-6 2026)
- Game outcome model: retrained, AUC=0.7406, test acc=67.1%, gradient_boosting selected
- ATS model: retrained (Brier-optimized), test acc=54.9%, AUC=0.5571, logistic_l1 selected
- Calibrated model: rebuilt (02:02), ATS artifact: 02:23
- Matchup CSV: 68,216 rows × 296 cols
- Tests: **145 passing**, 0 failing
- Branch: `main` (changes to commit)

## Pinnacle API Details (for reference)
- Base URL: `https://guest.api.arcadia.pinnacle.com/0.1`
- No auth required; NBA league ID: 487

## What's Next

### Phase 2 (higher effort)
1. **Optuna HPO** on LightGBM + XGBoost (both now in requirements)
2. **Model blending** — ensemble game outcome + ATS predictions
3. **SBRO historical odds** — backtesting CLV over historical seasons
4. **Margin regression model** — predict point differential, not just win/loss

### Known Stubs
- `fetch_player_props()` is a no-op stub
- `database/nba.db` — empty legacy artifact; pipeline is CSV-based

## Key Decisions
- Fractional Kelly (0.5x scale) — conservative sizing until CLV edge is confirmed over 10+ games
- CLV formula: `opening - closing` (positive = we got a better line)
- LightGBM guarded import preserves functionality even without lightgbm installed
- `_roll` columns: never add to `context_cols` — they're auto-captured by roll_cols filter
