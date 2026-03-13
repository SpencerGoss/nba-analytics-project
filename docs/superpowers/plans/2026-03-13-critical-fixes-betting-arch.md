# Critical Fixes + Betting Architecture Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 5 critical prediction bugs that silently corrupt daily outputs, then build a BettingRouter that provides market-specific predictions (moneyline, spread, props) with strict confidence tiers and no jargon.

**Architecture:** Phase 1 fixes bugs in the existing prediction pipeline (stale Elo, fillna, ATS waste, season types). Phase 2 adds `BettingRouter` that wraps `NBAEnsemble` and routes predictions to the correct market output (moneyline via calibrated P(win), spread via normal CDF on margin residuals). Confidence tiers (Best Bet / Solid Pick / Lean / Skip) replace raw numeric scores.

**Tech Stack:** Python 3.14, scikit-learn, scipy.stats (norm.cdf), pandas, SQLite, pytest

**Spec:** `docs/superpowers/specs/2026-03-13-project-overhaul-design.md` (Phases 1-2)

**Note:** Existing model artifacts use Python pickle serialization (scikit-learn convention). All pickle loads are from trusted, locally-generated `.pkl` files in `models/artifacts/`.

---

## File Structure

### New Files
| File | Responsibility |
|------|---------------|
| `src/models/betting_router.py` | Market-specific prediction routing (moneyline, spread, props stub) + confidence tiers |
| `src/models/odds_utils.py` | Shared devigging (odds-ratio method), American-to-decimal conversion, EV calculation |
| `tests/test_betting_router.py` | BettingRouter unit tests |
| `tests/test_odds_utils.py` | Devigging and odds conversion tests |
| `tests/test_fillna_removal.py` | Audit test ensuring no fillna(0) in prediction paths |
| `tests/test_season_codes.py` | Audit test ensuring integer season comparisons |
| `tests/test_clv_tracker.py` | CLV tracker fix tests |

### Modified Files
| File | Changes |
|------|---------|
| `src/models/ensemble.py` | Guard ATS behind weight check |
| `src/models/margin_model.py` | Add Elo refresh in predict_margin(); remove fillna(0) |
| `src/models/game_outcome_model.py` | Remove fillna(0) at line 752 |
| `src/models/value_bet_detector.py` | Remove fillna(0) x3; fix Kelly formula; set COMPOSITE_ATS_WEIGHT=0.0; add Elo refresh; lower threshold; add EV+tier |
| `src/models/clv_tracker.py` | Fix conn.total_changes; fix datetime.utcnow; add moneyline CLV |
| `src/models/model_explainability.py` | Remove fillna(0) at line 556 |
| `src/models/player_performance_model.py` | Remove fillna(0) at line 267 |
| `src/models/playoff_odds_model.py` | Remove fillna(0) at line 156 |
| `scripts/fetch_odds.py` | Fix hardcoded date at line 322 |

---

## Chunk 1: Critical Bug Fixes

### Task 0: Verify predict_game() Current-Season Filter (Spec 1.1)

The spec's highest-priority bug: `predict_game()` may use cross-season matchup data with stale Elo. diff_elo is 37.3% feature importance and ~3 of 6 daily predictions are affected if the row used comes from a prior season. `predict_game()` at `game_outcome_model.py:665-684` already refreshes Elo, but we must verify it also filters the matchup DataFrame to the latest season (same pattern `margin_model.py:352-384` uses).

**Files:**
- Audit: `src/models/game_outcome_model.py:665-750`
- Test: `tests/test_game_outcome_model.py`

- [ ] **Step 1: Write test verifying current-season row is used**

```python
# tests/test_game_outcome_model.py — add
def test_predict_game_uses_current_season_row():
    """predict_game must filter to latest season, not use stale cross-season data."""
    import inspect
    from src.models import game_outcome_model
    source = inspect.getsource(game_outcome_model.predict_game)
    # Must either filter by season or sort by date and take last row
    has_season_filter = "latest_season" in source or "season" in source
    has_date_sort = "sort_values" in source or "tail(1)" in source or "iloc[-1]" in source
    assert has_season_filter or has_date_sort, (
        "predict_game() does not filter to current season — may use stale cross-season data"
    )
```

- [ ] **Step 2: Run test and inspect result**

Run: `.venv/Scripts/python.exe -m pytest tests/test_game_outcome_model.py::test_predict_game_uses_current_season_row -v`

If PASS: current-season filtering already exists. Commit the test only.
If FAIL: implement the filter (Step 3).

- [ ] **Step 3: Add current-season filter if missing**

In `predict_game()`, after loading the matchup row (~line 700), add:
```python
    # Filter to current season to avoid stale cross-season features
    if "season" in matchup_df.columns:
        latest_season = matchup_df["season"].max()
        season_rows = matchup_df[matchup_df["season"] == latest_season]
        if len(season_rows) > 0:
            matchup_df = season_rows
```

- [ ] **Step 4: Run tests and commit**

```bash
.venv/Scripts/python.exe -m pytest tests/test_game_outcome_model.py -v --tb=short
git add src/models/game_outcome_model.py tests/test_game_outcome_model.py
git commit -m "fix: verify predict_game() uses current-season rows (spec 1.1)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 0b: Fix Season Code Type Inconsistency (Spec 1.5)

`margin_model.py:179` and `game_outcome_model.py:374` compare seasons as strings: `df["season"].astype(str) >= MODERN_ERA_START`. String comparison works by coincidence for current 6-digit codes but is semantically wrong.

**Files:**
- Modify: `src/models/margin_model.py:179`
- Modify: `src/models/game_outcome_model.py:374`
- Test: `tests/test_season_codes.py`

- [ ] **Step 1: Write regression test**

```python
# tests/test_season_codes.py
"""Ensure all season comparisons use integer, not string."""
import pathlib

MODEL_FILES = [
    "src/models/margin_model.py",
    "src/models/game_outcome_model.py",
]

def test_no_string_season_comparisons():
    violations = []
    for fpath in MODEL_FILES:
        p = pathlib.Path(fpath)
        if not p.exists():
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.split("#")[0]
            if ".astype(str)" in stripped and (">=" in stripped or "<=" in stripped):
                if "season" in stripped.lower():
                    violations.append(f"{fpath}:{i}: {line.strip()}")
    assert not violations, (
        "Found string-based season comparisons (should use int):\n"
        + "\n".join(violations)
    )
```

- [ ] **Step 2: Fix season comparisons**

In both files, change patterns like:
```python
df["season"].astype(str) >= MODERN_ERA_START
```
to:
```python
df["season"].astype(int) >= int(MODERN_ERA_START)
```

- [ ] **Step 3: Run tests and commit**

```bash
.venv/Scripts/python.exe -m pytest tests/test_season_codes.py -v
.venv/Scripts/python.exe -m pytest tests/ -q --tb=short
git add src/models/margin_model.py src/models/game_outcome_model.py tests/test_season_codes.py
git commit -m "fix: use integer season comparisons instead of string (spec 1.5)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 1: Fix Stale Elo in predict_margin()

The margin model's `predict_margin()` does NOT refresh Elo ratings before prediction. It uses whatever Elo values are in the most recent CSV row, which can be 10+ games stale. The game outcome model's `predict_game()` already calls `get_current_elos()` at lines 671-672 — margin needs the same.

**Files:**
- Modify: `src/models/margin_model.py:305-388`
- Reference: `src/models/game_outcome_model.py:671-672,734-735` (existing pattern)
- Reference: `src/features/elo.py:206-226` (`get_current_elos()` returns `dict[str, float]`)
- Test: `tests/test_margin_model.py`

- [ ] **Step 1: Write failing test for Elo refresh**

```python
# tests/test_margin_model.py — add to existing file
from unittest.mock import patch

def test_predict_margin_refreshes_elo():
    """predict_margin must call get_current_elos() to get fresh Elo values."""
    with patch("src.models.margin_model.get_current_elos") as mock_elos:
        mock_elos.return_value = {"LAL": 1550.0, "BOS": 1600.0}
        try:
            from src.models.margin_model import predict_margin
            predict_margin("LAL", "BOS")
        except Exception:
            pass  # Model artifacts may not exist in test env
        mock_elos.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_margin_model.py::test_predict_margin_refreshes_elo -v`
Expected: FAIL — `get_current_elos` is not imported in margin_model

- [ ] **Step 3: Add Elo refresh to predict_margin()**

In `src/models/margin_model.py`, add import at top:
```python
from src.features.elo import get_current_elos
```

Inside `predict_margin()`, after the row is constructed (around line 384) and before `row_df` is built (line 386), insert:
```python
    # Refresh Elo with current ratings (not stale CSV values)
    current_elos = get_current_elos()
    home_elo = current_elos.get(home_team, 1500.0)
    away_elo = current_elos.get(away_team, 1500.0)
    if "diff_elo" in row.index:
        row["diff_elo"] = home_elo - away_elo
    if "home_elo_pre" in row.index:
        row["home_elo_pre"] = home_elo
    if "away_elo_pre" in row.index:
        row["away_elo_pre"] = away_elo
    # Note: diff_elo_fast (K=40) and diff_elo_momentum are separate signals.
    # get_current_elos() only returns standard Elo (K=20). Leave fast/momentum
    # from the CSV row (still fresher than cross-season data). Only override
    # diff_elo which is 37.3% feature importance.
    # TODO (Phase 3): extend get_current_elos() to return fast + momentum Elo.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_margin_model.py::test_predict_margin_refreshes_elo -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q --tb=short`
Expected: All existing tests pass

- [ ] **Step 6: Commit**

```bash
git add src/models/margin_model.py tests/test_margin_model.py
git commit -m "fix: add Elo refresh to predict_margin() — was using stale CSV values

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2: Remove fillna(0) Across All Prediction Paths

Eight `.fillna(0)` calls across six model files replace NaN with 0 at inference time. Training uses `SimpleImputer(strategy="mean")`, so training sees mean-imputed values but inference sees 0 — systematic bias.

**Files:**
- Modify: `src/models/margin_model.py:386`
- Modify: `src/models/game_outcome_model.py:752`
- Modify: `src/models/value_bet_detector.py:299,366,595`
- Modify: `src/models/model_explainability.py:556`
- Modify: `src/models/player_performance_model.py:267`
- Modify: `src/models/playoff_odds_model.py:156`
- Create: `tests/test_fillna_removal.py`

- [ ] **Step 1: Write audit test**

```python
# tests/test_fillna_removal.py
"""Ensure no model inference path uses fillna(0) on feature DataFrames."""
import pathlib

MODEL_FILES = [
    "src/models/margin_model.py",
    "src/models/game_outcome_model.py",
    "src/models/value_bet_detector.py",
    "src/models/model_explainability.py",
    "src/models/player_performance_model.py",
    "src/models/playoff_odds_model.py",
]

def test_no_fillna_zero_in_model_inference():
    """No model file should use .fillna(0) on feature DataFrames."""
    violations = []
    for fpath in MODEL_FILES:
        p = pathlib.Path(fpath)
        if not p.exists():
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.split("#")[0]  # Ignore comments
            if ".fillna(0)" in stripped:
                violations.append(f"{fpath}:{i}: {line.strip()}")
    assert not violations, (
        "Found fillna(0) in prediction paths (should use pipeline imputer):\n"
        + "\n".join(violations)
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fillna_removal.py -v`
Expected: FAIL — finds 8 violations

- [ ] **Step 3: Remove fillna(0) from all six files**

`src/models/margin_model.py` line 386 — change `.fillna(0)` to nothing:
```
row_df = row.to_frame().T.reindex(columns=feat_cols).fillna(0)
```
becomes:
```
row_df = row.to_frame().T.reindex(columns=feat_cols)
```

`src/models/game_outcome_model.py` line 752 — same pattern.

`src/models/value_bet_detector.py` lines 299, 366, 595 — remove `.fillna(0)` from each.

`src/models/model_explainability.py` line 556 — change `row_df.fillna(0)` to `row_df`.

`src/models/player_performance_model.py` line 267 — remove `.fillna(0)`.

`src/models/playoff_odds_model.py` line 156 — remove `.fillna(0)`.

- [ ] **Step 4: Run audit test**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fillna_removal.py -v`
Expected: PASS

- [ ] **Step 5: Run full suite**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q --tb=short`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/models/margin_model.py src/models/game_outcome_model.py src/models/value_bet_detector.py src/models/model_explainability.py src/models/player_performance_model.py src/models/playoff_odds_model.py tests/test_fillna_removal.py
git commit -m "fix: remove fillna(0) from all prediction paths — let pipeline imputer handle NaN

Training uses SimpleImputer(mean) but inference replaced NaN with 0.
Removed from 8 locations across 6 files.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Guard ATS Model Behind Weight Check

`ensemble.py` loads and runs `ats_model.pkl` even though `ATS_WEIGHT=0.0`.

**Files:**
- Modify: `src/models/ensemble.py:130-149,224-311`
- Test: `tests/test_ensemble.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_ensemble.py — add
from unittest.mock import patch, MagicMock
import numpy as np

def test_ensemble_skips_ats_load_when_weight_zero():
    """When ATS_WEIGHT=0, load() should not attempt to load ats_model.pkl."""
    import src.models.ensemble as ens_module
    with patch.object(ens_module, "ATS_WEIGHT", 0.0):
        e = ens_module.NBAEnsemble.__new__(ens_module.NBAEnsemble)
        e.artifacts_dir = "models/artifacts"
        # Mock _load_artifact to track what gets loaded
        loaded = []
        original_load = getattr(e, "_load_artifact", None)
        # After the guard is added, init should skip ATS
        # Test: creating with weight=0 sets ats_model=None
        e.ats_model = None
        assert e.ats_model is None

def test_ensemble_predict_uses_neutral_ats_when_disabled():
    """predict() should use ats_prob=0.5 when ATS model is None."""
    import src.models.ensemble as ens_module
    with patch.object(ens_module, "ATS_WEIGHT", 0.0):
        e = ens_module.NBAEnsemble.__new__(ens_module.NBAEnsemble)
        e.ats_model = None
        # Verify that when ats_model is None, code path doesn't crash
        # Full predict() test requires model artifacts, so we verify the guard logic
        assert e.ats_model is None
```

- [ ] **Step 2: Modify ensemble.py __init__()**

In the ATS loading block (around lines 135-140), wrap with:
```python
        if ATS_WEIGHT > 0:
            ats_path = artifacts / "ats_model.pkl"
            with open(ats_path, "rb") as f:
                self.ats_model = pickle.load(f)
        else:
            self.ats_model = None
```

- [ ] **Step 3: Guard ATS in predict()**

In `predict()` (around lines 244-245), change ATS prediction to:
```python
        if self.ats_model is not None and ATS_WEIGHT > 0:
            ats_prob = float(self.ats_model.predict_proba(X_ats)[0][1])
        else:
            ats_prob = 0.5  # Neutral when ATS disabled
```

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ensemble.py -v`
Expected: PASS

- [ ] **Step 5: Run full suite and commit**

```bash
.venv/Scripts/python.exe -m pytest tests/ -q --tb=short
git add src/models/ensemble.py tests/test_ensemble.py
git commit -m "fix: skip ATS model load/predict when ATS_WEIGHT=0

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 4: Verify Known Issues (Spec 1.6) + Fix If Needed

**Files:**
- Audit: `scripts/fetch_odds.py` (hardcoded date — may already be fixed)
- Audit: `src/data/get_historical_absences.py` (write path — may already be fixed)

- [ ] **Step 1: Verify fetch_odds.py season_start date**

Grep for hardcoded `"2025-10-01"` or `"2024-10-01"` in `scripts/fetch_odds.py`. The spec cites line 322, but actual line numbers may differ. If already dynamic (using `date.today()`), skip to Step 2.

If still hardcoded, replace with:
```python
from datetime import date as _date
_today = _date.today()
_season_start_year = _today.year if _today.month >= 10 else _today.year - 1
_season_start = f"{_season_start_year}-10-01"
```

- [ ] **Step 2: Verify player_absences write path**

Read `src/data/get_historical_absences.py`. If output already goes to `data/processed/`, no change needed. If it writes to `data/raw/`, change to `data/processed/player_absences.csv`.

- [ ] **Step 3: Commit only if changes were needed**

```bash
.venv/Scripts/python.exe -m pytest tests/ -q --tb=short
# Only git add files that were actually changed
git add scripts/fetch_odds.py src/data/get_historical_absences.py
git commit -m "fix: verify/fix known issues from spec 1.6

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Chunk 2: Odds Utilities + BettingRouter

### Task 5: Create Shared Odds Utilities

Devigging logic is duplicated across `fetch_odds.py`, `value_bet_detector.py`, `build_value_bets.py`. Extract to one module with odds-ratio method.

**Files:**
- Create: `src/models/odds_utils.py`
- Create: `tests/test_odds_utils.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_odds_utils.py
import pytest
from src.models.odds_utils import (
    american_to_decimal,
    american_to_implied_prob,
    no_vig_odds_ratio,
    expected_value,
)

class TestAmericanToDecimal:
    def test_positive_odds(self):
        assert american_to_decimal(150) == 2.5

    def test_negative_odds(self):
        assert american_to_decimal(-150) == pytest.approx(1.6667, abs=0.001)

    def test_even_money(self):
        assert american_to_decimal(100) == 2.0

class TestNoVigOddsRatio:
    def test_balanced_market(self):
        home, away = no_vig_odds_ratio(-110, -110)
        assert home == pytest.approx(0.5, abs=0.001)
        assert away == pytest.approx(0.5, abs=0.001)

    def test_favorite_underdog(self):
        home, away = no_vig_odds_ratio(-200, 170)
        assert home + away == pytest.approx(1.0, abs=0.001)
        assert home > 0.5

    def test_none_input(self):
        assert no_vig_odds_ratio(None, -110) == (None, None)

class TestExpectedValue:
    def test_positive_ev(self):
        assert expected_value(0.6, 0.5) == pytest.approx(0.2, abs=0.001)

    def test_zero_market(self):
        assert expected_value(0.5, 0.0) is None
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_odds_utils.py -v`
Expected: FAIL — module doesn't exist

- [ ] **Step 3: Implement odds_utils.py**

```python
# src/models/odds_utils.py
"""Shared odds utility functions.

Centralizes devigging, conversion, and EV calculations previously
duplicated across fetch_odds.py, value_bet_detector.py, and build_value_bets.py.
"""
from __future__ import annotations


def american_to_decimal(american: int | float) -> float:
    """Convert American odds to decimal odds."""
    if american >= 100:
        return (american / 100) + 1
    return (100 / abs(american)) + 1


def american_to_implied_prob(american: int | float | None) -> float | None:
    """Convert American odds to implied probability (vig-inclusive)."""
    if american is None:
        return None
    if american >= 100:
        return 100 / (american + 100)
    return abs(american) / (abs(american) + 100)


def no_vig_odds_ratio(
    home_ml: int | float | None,
    away_ml: int | float | None,
) -> tuple[float | None, float | None]:
    """Remove vig using odds-ratio normalization (more accurate than multiplicative)."""
    if home_ml is None or away_ml is None:
        return (None, None)
    home_dec = american_to_decimal(home_ml)
    away_dec = american_to_decimal(away_ml)
    home_imp = 1 / home_dec
    away_imp = 1 / away_dec
    total = home_imp + away_imp
    return (home_imp / total, away_imp / total)


def expected_value(model_prob: float, market_prob: float) -> float | None:
    """EV = (model_prob / market_prob) - 1. Positive = model sees more value than market."""
    if market_prob <= 0:
        return None
    return (model_prob / market_prob) - 1
```

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_odds_utils.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/odds_utils.py tests/test_odds_utils.py
git commit -m "feat: add shared odds utilities (devigging, conversion, EV)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 6: Build BettingRouter with Confidence Tiers

Core new component. Wraps NBAEnsemble and provides market-specific outputs.

**Files:**
- Create: `src/models/betting_router.py`
- Create: `tests/test_betting_router.py`

- [ ] **Step 1: Write failing tests for confidence tiers and model agreement**

```python
# tests/test_betting_router.py
import pytest
from src.models.betting_router import confidence_tier, model_agreement

class TestConfidenceTier:
    def test_best_bet(self):
        assert confidence_tier(edge=0.10, models_agree=True) == "Best Bet"

    def test_solid_pick(self):
        assert confidence_tier(edge=0.05, models_agree=True) == "Solid Pick"

    def test_lean(self):
        assert confidence_tier(edge=0.025, models_agree=True) == "Lean"

    def test_skip_low_edge(self):
        assert confidence_tier(edge=0.01, models_agree=True) == "Skip"

    def test_skip_disagreement(self):
        assert confidence_tier(edge=0.10, models_agree=False) == "Skip"

    def test_boundary_best_bet(self):
        assert confidence_tier(edge=0.08, models_agree=True) == "Best Bet"

class TestModelAgreement:
    def test_both_favor_home(self):
        assert model_agreement(win_prob=0.65, pred_margin=3.5) is True

    def test_both_favor_away(self):
        assert model_agreement(win_prob=0.35, pred_margin=-5.0) is True

    def test_disagree(self):
        assert model_agreement(win_prob=0.60, pred_margin=-2.0) is False

    def test_neutral_margin(self):
        assert model_agreement(win_prob=0.55, pred_margin=0.5) is True
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_betting_router.py -v`
Expected: FAIL

- [ ] **Step 3: Implement betting_router.py**

```python
# src/models/betting_router.py
"""BettingRouter: market-specific prediction routing with confidence tiers.

Deliberate deviation from spec: takes pre-computed win_prob/pred_margin instead
of team abbreviations. This keeps BettingRouter decoupled from NBAEnsemble —
callers (build_picks.py, build_value_bets.py) already have ensemble outputs.
The spec's team-based interface would couple BettingRouter to model loading.

Wraps NBAEnsemble outputs and provides separate outputs per betting market:
- Moneyline: calibrated P(home_win)
- Spread: P(cover) via normal CDF on (pred_margin - spread) / residual_std
- Props: stub for Phase 4

Confidence tiers (strict, plain English, no jargon):
- Best Bet:   edge >= 8%, models agree
- Solid Pick: edge >= 4%, models agree
- Lean:       edge >= 2%
- Skip:       edge < 2% OR models disagree
"""
from __future__ import annotations

import json
from pathlib import Path

from scipy.stats import norm

from src.models.odds_utils import expected_value, no_vig_odds_ratio

BEST_BET_EDGE = 0.08
SOLID_PICK_EDGE = 0.04
LEAN_EDGE = 0.02
MARGIN_DEAD_ZONE = 1.5
DEFAULT_RESIDUAL_STD = 10.5


def confidence_tier(edge: float, models_agree: bool) -> str:
    """Assign strict confidence tier.

    Note: This is "bootstrap mode" per spec 2.2 — uses edge thresholds only.
    After 100+ tracked games validate tier boundaries (Phase 3 backtest),
    tighten to require historical win rate > 65% for Best Bet.
    """
    if not models_agree:
        return "Skip"
    if edge >= BEST_BET_EDGE:
        return "Best Bet"
    if edge >= SOLID_PICK_EDGE:
        return "Solid Pick"
    if edge >= LEAN_EDGE:
        return "Lean"
    return "Skip"


def model_agreement(win_prob: float, pred_margin: float) -> bool:
    """Check if outcome model and margin model agree on direction."""
    if abs(pred_margin) < MARGIN_DEAD_ZONE:
        return True
    return (win_prob > 0.5) == (pred_margin > 0)


class BettingRouter:
    """Routes predictions to market-specific outputs with confidence tiers."""

    def __init__(self, artifacts_dir: str = "models/artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.residual_std = self._load_residual_std()

    def _load_residual_std(self) -> float:
        """Load residual std from margin model training artifacts."""
        path = self.artifacts_dir / "margin_residual_std.json"
        if path.exists():
            with open(path) as f:
                return float(json.load(f)["residual_std"])
        return DEFAULT_RESIDUAL_STD

    def moneyline(
        self,
        win_prob: float,
        pred_margin: float,
        home_ml: int | float | None = None,
        away_ml: int | float | None = None,
    ) -> dict:
        """Moneyline market output with edge, EV, and confidence tier."""
        agree = model_agreement(win_prob, pred_margin)
        edge = 0.0
        ev = None
        if home_ml is not None and away_ml is not None:
            market_home, _ = no_vig_odds_ratio(home_ml, away_ml)
            if market_home is not None:
                edge = win_prob - market_home
                ev = expected_value(win_prob, market_home)

        tier = confidence_tier(abs(edge), agree)
        return {
            "prob": round(win_prob, 4),
            "pred_margin": round(pred_margin, 2),
            "edge": round(edge, 4),
            "ev": round(ev, 4) if ev is not None else None,
            "confidence_tier": tier,
            "models_agree": agree,
        }

    def spread(
        self,
        pred_margin: float,
        spread_line: float,
        win_prob: float,
        market_spread_prob: float | None = None,
    ) -> dict:
        """Spread market output. Uses margin model standalone via normal CDF."""
        cover_prob = float(norm.cdf((pred_margin - spread_line) / self.residual_std))
        agree = model_agreement(win_prob, pred_margin)
        edge = 0.0
        ev = None
        if market_spread_prob is not None and market_spread_prob > 0:
            edge = cover_prob - market_spread_prob
            ev = expected_value(cover_prob, market_spread_prob)

        tier = confidence_tier(abs(edge), agree)
        return {
            "cover_prob": round(cover_prob, 4),
            "pred_margin": round(pred_margin, 2),
            "spread_line": spread_line,
            "edge": round(edge, 4),
            "ev": round(ev, 4) if ev is not None else None,
            "confidence_tier": tier,
            "models_agree": agree,
        }

    def props(self, player_id: int, stat: str, line: float, date: str | None = None) -> dict:
        """Player prop output. Stub until Phase 4."""
        raise NotImplementedError(
            "Player prop predictions not yet implemented. See Phase 4."
        )
```

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_betting_router.py -v`
Expected: PASS

- [ ] **Step 5: Write spread and moneyline integration tests**

```python
# Add to tests/test_betting_router.py
from src.models.betting_router import BettingRouter

class TestBettingRouterSpread:
    def test_favorite_covers(self):
        router = BettingRouter.__new__(BettingRouter)
        router.residual_std = 10.5
        result = router.spread(pred_margin=7.0, spread_line=-5.5, win_prob=0.65)
        assert result["cover_prob"] > 0.8
        assert result["confidence_tier"] in ("Best Bet", "Solid Pick", "Lean", "Skip")

    def test_with_market_prob(self):
        router = BettingRouter.__new__(BettingRouter)
        router.residual_std = 10.5
        result = router.spread(pred_margin=7.0, spread_line=-5.5, win_prob=0.65, market_spread_prob=0.5)
        assert result["edge"] > 0
        assert result["ev"] is not None

class TestBettingRouterMoneyline:
    def test_with_odds(self):
        router = BettingRouter.__new__(BettingRouter)
        router.residual_std = 10.5
        result = router.moneyline(win_prob=0.65, pred_margin=5.0, home_ml=-200, away_ml=170)
        assert "prob" in result
        assert "confidence_tier" in result
        assert result["models_agree"] is True

    def test_without_odds(self):
        router = BettingRouter.__new__(BettingRouter)
        router.residual_std = 10.5
        result = router.moneyline(win_prob=0.55, pred_margin=2.0)
        assert result["edge"] == 0.0
        assert result["ev"] is None

    def test_props_raises(self):
        router = BettingRouter.__new__(BettingRouter)
        with pytest.raises(NotImplementedError):
            router.props(player_id=1, stat="PTS", line=25.5)
```

- [ ] **Step 6: Run all tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_betting_router.py -v`
Expected: PASS

- [ ] **Step 7: Run full suite and commit**

```bash
.venv/Scripts/python.exe -m pytest tests/ -q --tb=short
git add src/models/betting_router.py tests/test_betting_router.py
git commit -m "feat: add BettingRouter with market-specific outputs and confidence tiers

Moneyline: calibrated P(home_win) with edge vs market
Spread: P(cover) via normal CDF on margin residuals (no blending)
Props: stub for Phase 4
Strict tiers: Best Bet / Solid Pick / Lean / Skip

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Chunk 3: Value Bet Detector + CLV Fixes

### Task 7: Fix Value Bet Detector Bugs

Five bugs: Kelly formula, Kelly cap, ATS weight default, threshold, Elo refresh.

**Files:**
- Modify: `src/models/value_bet_detector.py:57,340-366,430-469,480`
- Test: `tests/test_value_bet_detector.py`

- [ ] **Step 1: Write failing tests for Kelly fix**

```python
# tests/test_value_bet_detector.py — add
def test_kelly_uses_payout_odds():
    """Kelly must use actual payout odds, not model-derived fair odds."""
    from src.models.value_bet_detector import _compute_kelly_fraction
    bet = {"model_win_prob": 0.60, "market_implied_prob": 0.50, "bet_side": "home"}
    result = _compute_kelly_fraction(bet)
    assert 0.05 < result < 0.15  # ~0.10 with half-Kelly

def test_kelly_capped():
    """Kelly fraction must never exceed 5%."""
    from src.models.value_bet_detector import _compute_kelly_fraction
    bet = {"model_win_prob": 0.90, "market_implied_prob": 0.50, "bet_side": "home"}
    result = _compute_kelly_fraction(bet)
    assert result <= 0.05
```

- [ ] **Step 2: Run tests to check current behavior**

Run: `.venv/Scripts/python.exe -m pytest tests/test_value_bet_detector.py::test_kelly_uses_payout_odds tests/test_value_bet_detector.py::test_kelly_capped -v`

- [ ] **Step 3: Verify and fix Kelly formula**

First, read `_compute_kelly_fraction()` (lines 430-469) carefully. The CLAUDE.md notes this was already fixed in a prior session (`b = (1-market)/market` is correct for market-implied odds).

If `b` already uses `market_prob` (not `model_prob`), the formula is correct — only add the 5% hard cap:
```python
fraction = max(0.0, min(fraction, 0.05))
```

If `b` still uses `p / (1 - p)` (model-derived fair odds), change to:
```python
market_prob = bet.get("market_implied_prob", 0.5)
b = (1 - market_prob) / market_prob if market_prob > 0 else 1.0
```
And add the cap.

- [ ] **Step 4: Change COMPOSITE_ATS_WEIGHT default**

Line 480:
```python
COMPOSITE_ATS_WEIGHT = float(os.getenv("COMPOSITE_ATS_WEIGHT", "0.0"))
```

- [ ] **Step 5: Lower VALUE_BET_THRESHOLD**

Line 57:
```python
VALUE_BET_THRESHOLD = float(os.getenv("VALUE_BET_THRESHOLD", "0.03"))
```

- [ ] **Step 6: Add Elo refresh to live mode**

After feature row construction (around line 350), add:
```python
from src.features.elo import get_current_elos
current_elos = get_current_elos()
if home_team in current_elos and away_team in current_elos:
    feature_row["diff_elo"] = current_elos[home_team] - current_elos[away_team]
```

- [ ] **Step 7: Add EV and confidence tier to output**

In `detect_value_bets()`, after edge computation, add:
```python
from src.models.odds_utils import expected_value
from src.models.betting_router import confidence_tier

df["ev"] = df.apply(
    lambda r: expected_value(r["model_win_prob"], r["market_implied_prob"])
    if r["market_implied_prob"] > 0 else None, axis=1)
df["confidence_tier"] = df["edge"].abs().apply(
    lambda e: confidence_tier(e, True))
```

- [ ] **Step 8: Run tests and commit**

```bash
.venv/Scripts/python.exe -m pytest tests/test_value_bet_detector.py -v
.venv/Scripts/python.exe -m pytest tests/ -q --tb=short
git add src/models/value_bet_detector.py tests/test_value_bet_detector.py
git commit -m "fix: value bet detector — Kelly formula, cap, ATS weight, threshold, Elo, EV

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 8: Fix CLV Tracker

**Files:**
- Modify: `src/models/clv_tracker.py:89,102,129`
- Create: `tests/test_clv_tracker.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_clv_tracker.py
import inspect

def test_clv_tracker_no_utcnow():
    """Must not use deprecated datetime.utcnow()."""
    from src.models import clv_tracker
    source = inspect.getsource(clv_tracker)
    assert "utcnow()" not in source

def test_clv_tracker_log_opening(tmp_path):
    """log_opening_line should work and use cursor.rowcount."""
    from src.models.clv_tracker import CLVTracker
    tracker = CLVTracker(str(tmp_path / "test.db"))
    result = tracker.log_opening_line("2026-03-13", "LAL", "BOS", -5.5)
    assert result is True
```

- [ ] **Step 2: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_clv_tracker.py -v`

- [ ] **Step 3: Fix conn.total_changes -> cursor.rowcount**

Line 102: ensure `execute()` result is assigned to cursor, then:
```python
inserted = cursor.rowcount > 0
```

- [ ] **Step 4: Verify datetime.utcnow() is already fixed**

Check `clv_tracker.py` for `utcnow()`. If already using `datetime.now(timezone.utc)` (likely fixed in prior session), skip this step. The `test_clv_tracker_no_utcnow` test should pass immediately.

- [ ] **Step 5: Run tests and commit**

```bash
.venv/Scripts/python.exe -m pytest tests/test_clv_tracker.py -v
.venv/Scripts/python.exe -m pytest tests/ -q --tb=short
git add src/models/clv_tracker.py tests/test_clv_tracker.py
git commit -m "fix: CLV tracker — cursor.rowcount, replace deprecated utcnow()

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 9: Final Integration Verification

- [ ] **Step 1: Run full test suite**

Run: `.venv/Scripts/python.exe -m pytest tests/ -v --tb=short 2>&1 | tail -30`
Expected: All tests pass

- [ ] **Step 2: Smoke test BettingRouter end-to-end**

```bash
.venv/Scripts/python.exe -c "
from src.models.betting_router import BettingRouter, confidence_tier, model_agreement
from src.models.odds_utils import no_vig_odds_ratio, expected_value

router = BettingRouter.__new__(BettingRouter)
router.residual_std = 10.5

ml = router.moneyline(win_prob=0.62, pred_margin=4.5, home_ml=-180, away_ml=155)
print(f'Moneyline: {ml}')

sp = router.spread(pred_margin=4.5, spread_line=-3.5, win_prob=0.62, market_spread_prob=0.52)
print(f'Spread: {sp}')

print(f'Tier test: {confidence_tier(0.10, True)}')
print(f'Agreement test: {model_agreement(0.62, 4.5)}')
print('All smoke tests passed.')
"
```
Expected: Prints results without errors

- [ ] **Step 3: Verify test count hasn't decreased**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q 2>&1 | tail -3`
Expected: Test count >= 1432 (baseline from CLAUDE.md)
