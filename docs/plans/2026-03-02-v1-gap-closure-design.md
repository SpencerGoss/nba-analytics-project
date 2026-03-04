# v1.0 Integration Gap Closure — Design

**Date:** 2026-03-02
**Scope:** Close audit gaps INT-01, INT-02, FLOW-01
**Files modified:** `src/models/train_all_models.py`, `src/models/predict_cli.py`

---

## Problem

The v1.0 milestone audit identified 3 integration gaps:

- **INT-01:** `train_all_models.py` does not call `train_ats_model()` or `run_calibration_analysis()`
- **INT-02:** `predict_cli.py` has no `ats` or `value-bet` subcommand
- **FLOW-01:** Full retrain requires 4 manual commands

## Design

### Change 1: train_all_models.py — Add ATS + Calibration

Add two training steps after the existing three:

1. Game outcomes (existing)
2. Player performance (existing)
3. Playoff odds (existing)
4. **Calibration** — `run_calibration_analysis()` (depends on game outcome model)
5. **ATS model** — `train_ats_model()` (independent)

Import `run_calibration_analysis` from `src.models.calibration` and `train_ats_model` from `src.models.ats_model`. Update header text from "3" to "5" tasks. Add calibration Brier score and ATS accuracy to the run summary.

With the existing `--rebuild-features` flag, this makes `train_all_models.py --rebuild-features` the single retrain command (closing FLOW-01).

### Change 2: predict_cli.py — Add ats and value-bet Subcommands

**`ats` subcommand:**
```
python src/models/predict_cli.py ats --home BOS --away LAL --spread -3.5
```
- Required: `--home`, `--away`, `--spread`
- Optional: `--home-ml`, `--away-ml` (moneylines for implied prob columns)
- Builds a single-row DataFrame from latest matchup features + provided spread/moneylines
- Calls `predict_ats()`, returns JSON with `covers_spread_prob` and `covers_spread_pred`

**`value-bet` subcommand:**
```
python src/models/predict_cli.py value-bet
python src/models/predict_cli.py value-bet --live
python src/models/predict_cli.py value-bet --threshold 0.07
```
- Default: historical mode (no API key needed)
- `--live`: fetch lines from The Odds API
- `--threshold`: override value-bet edge threshold (default 0.05)
- Calls `run_value_bet_scan()`, prints JSON results

## Gaps Closed

| Audit Gap | Fix |
|-----------|-----|
| INT-01 | ATS + calibration wired into train_all_models.py |
| INT-02 | `ats` and `value-bet` subcommands in predict_cli.py |
| FLOW-01 | `train_all_models.py --rebuild-features` = single retrain command |
