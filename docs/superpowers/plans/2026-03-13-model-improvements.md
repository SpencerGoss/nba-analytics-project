# Model Improvements Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve game prediction accuracy and validate betting edge through SHAP analysis, feature engineering, margin model upgrades, ensemble tuning, and walk-forward backtesting.

**Architecture:** SHAP analysis first (understand current model before changing anything), then informed feature engineering, then margin model upgrades (Huber loss, residual_std artifact), then ensemble weight optimization, then walk-forward backtest to validate betting edge translates to profit.

**Tech Stack:** Python 3.14, scikit-learn, shap, scipy, statsmodels, pandas, numpy, pytest

**Spec:** `docs/superpowers/specs/2026-03-13-project-overhaul-design.md` (Phase 3)

**Depends on:** Plan A (Critical Fixes + Betting Architecture) must be complete first.

**Note:** Model artifacts use scikit-learn's pickle serialization (trusted, locally-generated files only). SHAP analysis loads these artifacts for feature attribution.

---

## File Structure

### New Files
| File | Responsibility |
|------|---------------|
| `src/models/shap_analysis.py` | SHAP summary, interaction effects, feature importance ranking |
| `src/models/backtest.py` | Walk-forward betting backtest with realistic vig |
| `src/models/significance.py` | Statistical significance tests for model accuracy and ATS |
| `tests/test_shap_analysis.py` | SHAP analysis tests |
| `tests/test_backtest.py` | Backtest logic tests |
| `tests/test_significance.py` | Significance test tests |

### Modified Files
| File | Changes |
|------|---------|
| `src/models/game_outcome_model.py` | Add CV variance reporting |
| `src/models/margin_model.py` | Switch to Huber loss; save residual_std artifact; add segmented MAE |
| `src/models/calibration.py` | Add temperature scaling as 3rd calibration option |
| `src/models/ensemble.py` | Data-driven weight optimization; data-driven confidence thresholds |
| `src/features/team_game_features.py` | Add orthogonal features (opponent-adjusted net rtg, form acceleration) |

---

## Chunk 1: SHAP Analysis + Statistical Significance

### Task 1: SHAP Analysis on Current Model

Run SHAP on the current game outcome model BEFORE making any changes. Reveals true feature contributions vs GBM's biased importance. Uses `scientific-skills:shap` plugin patterns.

**Files:**
- Create: `src/models/shap_analysis.py`
- Create: `tests/test_shap_analysis.py`
- Reference: `src/models/game_outcome_model.py:100` (get_feature_cols)
- Output: `models/artifacts/shap_analysis/` directory

- [ ] **Step 1: Write failing test**

```python
# tests/test_shap_analysis.py
def test_run_shap_module_exists():
    from src.models.shap_analysis import run_shap_analysis, get_top_shap_features
    assert callable(run_shap_analysis)
    assert callable(get_top_shap_features)

def test_shap_top_features_format():
    from src.models.shap_analysis import get_top_shap_features
    # Returns empty list when no analysis has been saved yet
    features = get_top_shap_features()
    assert isinstance(features, list)
```

- [ ] **Step 2: Run test to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_shap_analysis.py -v`

- [ ] **Step 3: Implement shap_analysis.py**

Create `src/models/shap_analysis.py`:
- `run_shap_analysis()`: loads calibrated model + features, runs TreeExplainer on sampled data, saves top-20 features CSV/JSON, summary plot PNG to `models/artifacts/shap_analysis/`
- `get_top_shap_features()`: loads saved JSON, returns list of (feature_name, mean_abs_shap) tuples
- The model is loaded from trusted local artifacts (`game_outcome_model_calibrated.pkl`) using scikit-learn's standard serialization

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_shap_analysis.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/shap_analysis.py tests/test_shap_analysis.py
git commit -m "feat: add SHAP analysis module for game outcome model

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2: Statistical Significance Testing

Test whether 67.5% accuracy and 55% ATS are statistically significant.

**Files:**
- Create: `src/models/significance.py`
- Create: `tests/test_significance.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_significance.py
import pytest

def test_binomial_test_significant():
    from src.models.significance import accuracy_significance
    result = accuracy_significance(wins=675, total=1000, null_prob=0.5)
    assert result["p_value"] < 0.001
    assert result["significant"] is True

def test_binomial_test_not_significant():
    from src.models.significance import accuracy_significance
    result = accuracy_significance(wins=51, total=100, null_prob=0.5)
    assert result["significant"] is False

def test_confidence_interval():
    from src.models.significance import accuracy_significance
    result = accuracy_significance(wins=675, total=1000, null_prob=0.5)
    assert result["ci_lower"] <= 0.675 <= result["ci_upper"]
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_significance.py -v`

- [ ] **Step 3: Implement significance.py**

Create `src/models/significance.py`:
- `accuracy_significance(wins, total, null_prob, alpha)`: binomial test + CI, returns dict with p_value, significant, ci_lower, ci_upper
- `sample_size_needed(target_accuracy, null_accuracy, alpha, power)`: calculates how many bets needed to confirm edge at given power using statsmodels

- [ ] **Step 4: Run tests and commit**

```bash
.venv/Scripts/python.exe -m pytest tests/test_significance.py -v
git add src/models/significance.py tests/test_significance.py
git commit -m "feat: add statistical significance tests for model accuracy

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Chunk 2: Margin Model + Ensemble Upgrades

### Task 3: Upgrade Margin Model (Huber Loss + residual_std)

Switch from MAE/MSE to Huber Loss and save residual_std artifact for BettingRouter.

**Files:**
- Modify: `src/models/margin_model.py:144-302`
- Test: `tests/test_margin_model.py`

- [ ] **Step 1: Write failing test for residual_std**

```python
# tests/test_margin_model.py — add
def test_train_produces_residual_std_function():
    from src.models.margin_model import train_margin_model
    assert callable(train_margin_model)
    # After training runs, models/artifacts/margin_residual_std.json should exist
```

- [ ] **Step 2: Add Huber GBM candidate to margin model**

In `src/models/margin_model.py` model candidates section (~line 160), add:
```python
candidates["huber_gbm"] = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("model", GradientBoostingRegressor(
        loss="huber", alpha=0.9,
        n_estimators=500, max_depth=3, learning_rate=0.03,
        subsample=0.9, min_samples_leaf=20, max_features=0.7,
        validation_fraction=0.1, n_iter_no_change=15,
    )),
])
```

- [ ] **Step 3: Save residual_std artifact after training**

After model selection and final fit (~line 280), add code to:
- Compute `residual_std = np.std(y_train - best_pipeline.predict(X_train))`
- Save to `models/artifacts/margin_residual_std.json` as `{"residual_std": value}`

- [ ] **Step 4: Add segmented MAE reporting**

After test evaluation, add logging for MAE by spread bucket: tight (0-3), medium (3-7), wide (7+).

- [ ] **Step 5: Run tests and commit**

```bash
.venv/Scripts/python.exe -m pytest tests/test_margin_model.py -v
git add src/models/margin_model.py tests/test_margin_model.py
git commit -m "feat: margin model — Huber loss, residual_std artifact, segmented MAE

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 4: Add Temperature Scaling to Calibration

Add as 3rd calibration option alongside Platt and Isotonic.

**Files:**
- Modify: `src/models/calibration.py:455-489`
- Test: `tests/test_calibration.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_calibration.py — add
def test_temperature_scaling_option():
    from src.models.calibration import _fit_temperature_scaling
    import numpy as np
    probs = np.array([0.3, 0.5, 0.7, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1, 1])
    temp = _fit_temperature_scaling(probs, labels)
    assert 0.5 < temp < 3.0
```

- [ ] **Step 2: Implement _fit_temperature_scaling**

In `src/models/calibration.py`, add function that:
- Takes raw probabilities and labels
- Converts to logits, finds optimal temperature T via `scipy.optimize.minimize_scalar` that minimizes Brier score
- Returns float temperature value

Then add temperature as 3rd candidate in the selection logic (~line 475): fit temperature, compute Brier, select best of {platt, isotonic, temperature}.

- [ ] **Step 3: Run tests and commit**

```bash
.venv/Scripts/python.exe -m pytest tests/test_calibration.py -v
git add src/models/calibration.py tests/test_calibration.py
git commit -m "feat: add temperature scaling as 3rd calibration option

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 5: Ensemble Weight Optimization

Replace hardcoded weights with data-driven grid search.

**Files:**
- Modify: `src/models/ensemble.py:50-57`
- Test: `tests/test_ensemble.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_ensemble.py — add
def test_optimize_weights_valid():
    from src.models.ensemble import optimize_ensemble_weights
    import numpy as np
    win_probs = np.array([0.6, 0.4, 0.7, 0.55])
    margin_preds = np.array([3.0, -2.0, 5.0, 1.0])
    actuals = np.array([1, 0, 1, 1])
    weights = optimize_ensemble_weights(win_probs, margin_preds, actuals)
    assert 0 <= weights["game_outcome"] <= 1
    assert abs(weights["game_outcome"] + weights["margin"] - 1.0) < 0.01
```

- [ ] **Step 2: Implement optimize_ensemble_weights**

Add function to `src/models/ensemble.py`:
- 1D grid search at 0.05 increments: game_outcome_weight + margin_weight = 1.0 (~20 candidates since ATS=0)
- Also searches MARGIN_NORM_FACTOR across [5,10,15,20,25,30]
- Minimizes Brier score on validation predictions
- Returns dict with optimal weights and selected norm factor

- [ ] **Step 3: Run tests and commit**

```bash
.venv/Scripts/python.exe -m pytest tests/test_ensemble.py -v
git add src/models/ensemble.py tests/test_ensemble.py
git commit -m "feat: ensemble weight optimization via Brier-minimizing grid search

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Chunk 3: Walk-Forward Backtest + Feature Engineering

### Task 6: Walk-Forward Betting Backtest

Validate that model edge translates to actual betting profit with realistic vig.

**Files:**
- Create: `src/models/backtest.py`
- Create: `tests/test_backtest.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_backtest.py
import numpy as np

def test_backtest_returns_metrics():
    from src.models.backtest import run_backtest
    result = run_backtest(
        model_probs=np.array([0.6, 0.55, 0.7, 0.52]),
        market_probs=np.array([0.5, 0.5, 0.5, 0.5]),
        actuals=np.array([1, 0, 1, 1]),
        edge_threshold=0.03,
    )
    assert "roi" in result
    assert "win_rate" in result
    assert result["n_bets"] <= 4

def test_backtest_no_bets_below_threshold():
    from src.models.backtest import run_backtest
    result = run_backtest(
        model_probs=np.array([0.51, 0.50]),
        market_probs=np.array([0.50, 0.50]),
        actuals=np.array([1, 0]),
        edge_threshold=0.10,
    )
    assert result["n_bets"] == 0
```

- [ ] **Step 2: Implement backtest.py**

Create `src/models/backtest.py`:
- `run_backtest(model_probs, market_probs, actuals, edge_threshold, vig, unit_size)`:
  - Bets on games where |edge| > threshold
  - Positive edge -> bet home, negative -> bet away
  - Payout at -110 vig (~0.9545 multiplier on wins)
  - Returns: roi, win_rate, n_bets, profit, avg_edge, max_drawdown, total_wagered

- [ ] **Step 3: Run tests and commit**

```bash
.venv/Scripts/python.exe -m pytest tests/test_backtest.py -v
git add src/models/backtest.py tests/test_backtest.py
git commit -m "feat: walk-forward betting backtest with realistic vig

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 7: Add Orthogonal Features

New features capturing information not in the existing model (informed by SHAP).

**Files:**
- Modify: `src/features/team_game_features.py:364-950`
- Test: `tests/test_team_game_features.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_team_game_features.py — add
def test_opponent_adjusted_net_rtg_col():
    from src.features.team_game_features import build_team_game_features
    df = build_team_game_features()
    assert "opp_adj_net_rtg_roll20" in df.columns

def test_form_acceleration_col():
    from src.features.team_game_features import build_team_game_features
    df = build_team_game_features()
    assert "net_rtg_acceleration" in df.columns
```

- [ ] **Step 2: Implement new features**

In `build_team_game_features()` after existing rolling features (~line 500):
- **Opponent-adjusted net rating:** `net_rtg_game_roll20 - league_avg_net_rtg` (shifted)
- **Form acceleration:** `ewm(span=7).mean() - ewm(span=7).mean().shift(5)` (delta of EWMA = slope)

Both must use `shift(1)` before any rolling to prevent data leakage.

- [ ] **Step 3: Run tests and commit**

```bash
.venv/Scripts/python.exe -m pytest tests/test_team_game_features.py -v
git add src/features/team_game_features.py tests/test_team_game_features.py
git commit -m "feat: add opponent-adjusted net rating and form acceleration features

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 8: Full Retrain + Analysis Pipeline

Execute all analysis and retraining in order. **Manual execution task.**

- [ ] **Step 1: Run SHAP analysis on current model**
- [ ] **Step 2: Run significance tests (67.5% accuracy, 55% ATS)**
- [ ] **Step 3: Rebuild features with new orthogonal features**
- [ ] **Step 4: Retrain game outcome model**
- [ ] **Step 5: Re-run calibration**
- [ ] **Step 6: Retrain margin model (now with Huber loss)**
- [ ] **Step 7: Run ensemble weight optimization**
- [ ] **Step 8: Run walk-forward backtest on test season**
- [ ] **Step 9: Compare metrics to baseline (67.5% acc, AUC 0.7422, MAE 10.52)**
- [ ] **Step 10: Commit retrained artifacts**

```bash
git add models/artifacts/
git commit -m "chore: retrain all models with improved features and calibration

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```
