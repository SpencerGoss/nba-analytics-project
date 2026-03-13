# Player Prop System Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-stage player prop prediction system (minutes model -> per-minute stat models) with quantile regression and conformal prediction intervals, enabling over/under predictions for PTS, REB, AST, 3PM.

**Architecture:** Stage 1 predicts minutes played (explains ~65% of stat variance). Stage 2 predicts per-36 stat rates, scaled by predicted minutes. Quantile regression provides 25th/50th/75th percentile predictions. Conformal prediction gives guaranteed coverage intervals. BettingRouter.props() wired in final task.

**Tech Stack:** Python 3.14, scikit-learn, XGBoost, scipy, pandas, numpy, pytest

**Spec:** `docs/superpowers/specs/2026-03-13-project-overhaul-design.md` (Phase 4)

**Depends on:** Plan A (BettingRouter exists with props stub), Plan B (SHAP + retrained models).

**Note:** Model artifacts use scikit-learn's standard serialization for trusted, locally-generated models. All player data is from committed CSV files.

---

## File Structure

### New Files
| File | Responsibility |
|------|---------------|
| `src/models/player_minutes_model.py` | Minutes prediction model (Stage 1) + blowout adjustment |
| `src/models/player_stat_models.py` | PTS/REB/AST/3PM per-minute models (Stage 2) + quantile regression |
| `src/models/conformal.py` | Conformal prediction interval calibration |
| `tests/test_player_prop_features.py` | Player feature engineering tests |
| `tests/test_player_minutes_model.py` | Minutes model tests |
| `tests/test_player_stat_models.py` | Stat model tests |
| `tests/test_conformal.py` | Conformal prediction tests |

### Modified Files
| File | Changes |
|------|---------|
| `src/features/player_features.py` | Add build_player_prop_features() for prop-specific features |
| `src/models/betting_router.py` | Wire props() to actual models (remove NotImplementedError) |
| `scripts/build_props.py` | Call prop models to generate dashboard JSON |
| `update.py` | Add Step 4b for weekly prop model training |

---

## Chunk 1: Player Feature Engineering + Minutes Model

### Task 1: Build Player Prop Feature Engineering

Create features for player-level predictions. All rolling features use shift(1) to prevent data leakage.

**Files:**
- Modify: `src/features/player_features.py`
- Create: `tests/test_player_prop_features.py`
- Data: `data/raw/player_game_logs_*.csv`, `data/raw/player_stats_advanced_*.csv`, `data/processed/lineup_data.csv`

- [ ] **Step 1: Write failing tests for required columns**

```python
# tests/test_player_prop_features.py
def test_build_player_features_cols():
    from src.features.player_features import build_player_prop_features
    df = build_player_prop_features()
    required = ["player_id", "game_date", "minutes", "pts_per36", "reb_per36",
                "ast_per36", "fg3m_per36", "usage_rate_ewma", "is_b2b",
                "is_starter", "blowout_risk"]
    for col in required:
        assert col in df.columns, f"Missing: {col}"

def test_features_shifted():
    from src.features.player_features import build_player_prop_features
    df = build_player_prop_features()
    first_games = df.groupby("player_id").first()
    assert first_games["usage_rate_ewma"].isna().sum() > 0
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_player_prop_features.py -v`

- [ ] **Step 3: Implement build_player_prop_features()**

Add to `src/features/player_features.py`:
- Load player_game_logs, compute per-36 rates
- Tier 1: minutes_ewma, usage_rate_ewma, fga_per36, efg_pct_ewma, pace_adjustment
- Tier 2: is_b2b, is_starter, blowout_risk (logistic on spread), is_home
- Tier 3: fouls_per_min_ewma, season_game_num
- All rolling features: `groupby("player_id").shift(1)` before `.ewm()` or `.rolling()`
- Output: `data/features/player_prop_features.csv`

- [ ] **Step 4: Run tests and commit**

```bash
.venv/Scripts/python.exe -m pytest tests/test_player_prop_features.py -v
git add src/features/player_features.py tests/test_player_prop_features.py
git commit -m "feat: player prop feature engineering — EWMA, B2B, blowout risk

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2: Build Minutes Prediction Model

Stage 1: predict minutes played, with blowout adjustment from game spread.

**Files:**
- Create: `src/models/player_minutes_model.py`
- Create: `tests/test_player_minutes_model.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_player_minutes_model.py
def test_blowout_adjustment():
    from src.models.player_minutes_model import apply_blowout_adjustment
    base = 32.0
    adj_7 = apply_blowout_adjustment(base, spread=7.0)
    adj_14 = apply_blowout_adjustment(base, spread=14.0)
    assert 29 < adj_7 < 32  # 7pt spread: ~6% reduction
    assert adj_14 < adj_7    # Bigger spread = more reduction
    assert adj_14 > 20       # Not absurdly low

def test_train_callable():
    from src.models.player_minutes_model import train_minutes_model
    assert callable(train_minutes_model)

def test_predict_callable():
    from src.models.player_minutes_model import predict_minutes
    assert callable(predict_minutes)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_player_minutes_model.py -v`

- [ ] **Step 3: Implement player_minutes_model.py**

Create `src/models/player_minutes_model.py`:
- `apply_blowout_adjustment(base_minutes, spread)`: logistic blowout_prob from abs(spread), reduces minutes by `blowout_prob * 0.30`
- `train_minutes_model(features_path, artifacts_dir)`: Huber GBM on player_prop_features, expanding-window CV, saves model + features + metadata as JSON
- `predict_minutes(player_id, features, spread, artifacts_dir)`: loads model, predicts, applies blowout adjustment

- [ ] **Step 4: Run tests and commit**

```bash
.venv/Scripts/python.exe -m pytest tests/test_player_minutes_model.py -v
git add src/models/player_minutes_model.py tests/test_player_minutes_model.py
git commit -m "feat: Stage 1 minutes prediction with blowout adjustment

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Chunk 2: Per-Stat Models + Quantile Regression + Conformal

### Task 3: Build Per-Stat Models (PTS, REB, AST, 3PM)

Stage 2: predict per-36 rates, scale by predicted minutes.

**Files:**
- Create: `src/models/player_stat_models.py`
- Create: `tests/test_player_stat_models.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_player_stat_models.py
def test_stat_targets():
    from src.models.player_stat_models import STAT_TARGETS
    assert set(STAT_TARGETS) == {"pts", "reb", "ast", "fg3m"}

def test_train_callable():
    from src.models.player_stat_models import train_stat_models
    assert callable(train_stat_models)

def test_predict_callable():
    from src.models.player_stat_models import predict_player_stat
    assert callable(predict_player_stat)
```

- [ ] **Step 2: Implement player_stat_models.py**

Create `src/models/player_stat_models.py`:
- `STAT_TARGETS = ["pts", "reb", "ast", "fg3m"]`
- `train_stat_models()`: one Huber GBM per stat on per-36 rates
- `predict_player_stat(player_id, stat, features, predicted_minutes)`: per-36 rate * (predicted_minutes / 36)
- Saves: `player_{stat}_model.pkl`, `player_stat_features.pkl`, `player_stat_metadata.json`

- [ ] **Step 3: Run tests and commit**

```bash
.venv/Scripts/python.exe -m pytest tests/test_player_stat_models.py -v
git add src/models/player_stat_models.py tests/test_player_stat_models.py
git commit -m "feat: Stage 2 per-stat models (PTS/REB/AST/3PM)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 4: Add Quantile Regression

Predict 25th/50th/75th percentiles for each stat.

**Files:**
- Modify: `src/models/player_stat_models.py`
- Test: `tests/test_player_stat_models.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_player_stat_models.py — add
def test_quantile_predict_callable():
    from src.models.player_stat_models import predict_player_stat_quantiles
    assert callable(predict_player_stat_quantiles)
```

- [ ] **Step 2: Add quantile models**

In `player_stat_models.py`:
- `train_quantile_models()`: 3 GBR models per stat with `loss="quantile"`, alpha in [0.25, 0.50, 0.75]
- `predict_player_stat_quantiles()`: returns `{"p25": x, "p50": y, "p75": z}` scaled by predicted minutes
- Uses Pinball loss for evaluation

- [ ] **Step 3: Run tests and commit**

```bash
.venv/Scripts/python.exe -m pytest tests/test_player_stat_models.py -v
git add src/models/player_stat_models.py tests/test_player_stat_models.py
git commit -m "feat: quantile regression for player stat predictions (p25/p50/p75)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 5: Conformal Prediction Intervals

Distribution-free coverage guarantees.

**Files:**
- Create: `src/models/conformal.py`
- Create: `tests/test_conformal.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_conformal.py
import numpy as np

def test_conformal_calibration():
    from src.models.conformal import calibrate_conformal
    residuals = np.array([1.0, -2.0, 0.5, 3.0, -1.5, 2.0, -0.5, 1.5, -3.0, 0.0])
    quantile = calibrate_conformal(residuals, coverage=0.90)
    assert quantile > 0

def test_conformal_interval():
    from src.models.conformal import conformal_interval
    interval = conformal_interval(prediction=22.0, quantile=5.0)
    assert interval["lower"] == 17.0
    assert interval["upper"] == 27.0
```

- [ ] **Step 2: Implement conformal.py**

```python
# src/models/conformal.py
"""Conformal prediction for guaranteed coverage intervals."""
import numpy as np

def calibrate_conformal(residuals: np.ndarray, coverage: float = 0.90) -> float:
    """Compute conformal quantile from calibration residuals."""
    abs_residuals = np.abs(residuals)
    n = len(abs_residuals)
    q = np.ceil((n + 1) * coverage) / n
    return float(np.quantile(abs_residuals, min(q, 1.0)))

def conformal_interval(prediction: float, quantile: float) -> dict:
    """Build prediction interval from conformal quantile."""
    return {
        "lower": round(prediction - quantile, 1),
        "upper": round(prediction + quantile, 1),
        "width": round(2 * quantile, 1),
    }
```

- [ ] **Step 3: Run tests and commit**

```bash
.venv/Scripts/python.exe -m pytest tests/test_conformal.py -v
git add src/models/conformal.py tests/test_conformal.py
git commit -m "feat: conformal prediction intervals with guaranteed coverage

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Chunk 3: Pipeline Integration + BettingRouter Wiring

### Task 6: Wire Props into BettingRouter

Replace the NotImplementedError stub with actual model calls.

**Files:**
- Modify: `src/models/betting_router.py`
- Test: `tests/test_betting_router.py`

- [ ] **Step 1: Update props() method**

Replace `raise NotImplementedError` with implementation that:
- Builds player features for the given date
- Calls `predict_minutes()` with blowout adjustment
- Calls `predict_player_stat_quantiles()` for the requested stat
- Computes over_prob relative to the line
- Returns `{over_prob, median, p25, p75, pred_minutes, confidence_tier}`

- [ ] **Step 2: Run tests and commit**

```bash
.venv/Scripts/python.exe -m pytest tests/test_betting_router.py -v
git add src/models/betting_router.py tests/test_betting_router.py
git commit -m "feat: wire player prop models into BettingRouter.props()

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 7: Update Pipeline (build_props.py + update.py)

Integrate prop models into daily pipeline.

**Files:**
- Modify: `scripts/build_props.py`
- Modify: `update.py`

- [ ] **Step 1: Update build_props.py**

Call prop models for each player in today's games. Output: `dashboard/data/props.json` with predictions, intervals, over_prob vs sportsbook lines.

- [ ] **Step 2: Add Step 4b to update.py**

Weekly (Monday) prop model retraining: `train_minutes_model()` + `train_stat_models()` + `train_quantile_models()`.

- [ ] **Step 3: Run tests and commit**

```bash
.venv/Scripts/python.exe -m pytest tests/ -q --tb=short
git add scripts/build_props.py update.py
git commit -m "feat: integrate prop models into daily pipeline

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 8: Train Initial Models and Validate

- [ ] **Step 1: Build player prop features**
- [ ] **Step 2: Train minutes model**
- [ ] **Step 3: Train stat models + quantile models**
- [ ] **Step 4: Calibrate conformal intervals on held-out data**
- [ ] **Step 5: Run full test suite**
- [ ] **Step 6: Commit artifacts**

```bash
git add models/artifacts/player_minutes_* models/artifacts/player_*_model.* models/artifacts/player_stat_*
git commit -m "chore: initial player prop model training artifacts

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```
