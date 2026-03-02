# Phase 2: Modern Era Features - Research

**Researched:** 2026-03-01
**Domain:** Pandas feature engineering, NBA advanced metrics (Four Factors, pace-normalized efficiency), scikit-learn model era filtering
**Confidence:** HIGH

---

## Summary

Phase 2 extends the game outcome feature pipeline with pace-normalized efficiency metrics (ORtg, DRtg, net rating, eFG%, TS%, pace, turnover rate per possession) and adds a Four Factors differential composite as a matchup feature. It then restricts the game outcome model's training window to the modern NBA era (2013-14 through 2023-24), excluding the 2019-20 bubble and 2020-21 shortened seasons.

All required metrics can be computed per-game from box score columns already present in `team_game_logs.csv` (fgm, fga, fg3m, fg3a, fta, oreb, dreb, tov, pts). The self-join pattern for opponent box score data is already established in `build_team_game_features()` (used for `opp_pts`, `opp_dreb_game`). The computed per-game values are then rolled with `.shift(1)` before `.rolling()` — the exact pattern the codebase already uses — to prevent lookahead bias. No new library dependencies are needed; everything runs on the existing pandas and scikit-learn stack.

The `game_outcome_model.py` already contains the `MODERN_ERA_ONLY` toggle and `MODERN_ERA_START = '201415'` constant. The primary changes are: (1) set `MODERN_ERA_ONLY = True` as the default, (2) change `MODERN_ERA_START` to `'201314'` to capture 2013-14 per the success criterion, and (3) add an `EXCLUDED_SEASONS` list for the two anomalous seasons. The accuracy comparison (success criterion 5) requires running both full-history and modern-era configurations and asserting modern >= full-history on the 2014+ holdout set.

**Primary recommendation:** Compute all advanced metrics (ORtg, DRtg, eFG%, TS%, pace, tov_per_poss, oreb_pct, ft_rate) per-game from box score in `build_team_game_features()`, roll them identically to existing stats, then assemble the Four Factors composite in `build_matchup_dataset()`. Change `MODERN_ERA_ONLY = True` and `MODERN_ERA_START = '201314'` in `game_outcome_model.py`, add excluded seasons filter.

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| FR-2.1 | Add pace-normalized efficiency features from `team_stats_advanced` table: ORtg, DRtg, net rating, eFG%, TS% as rolling averages (5/10/20 game windows) | All five metrics are computable per-game from box score columns (0% null in 2014+ data). Self-join pattern for DRtg (needs opponent pts/poss) already exists. Roll using existing `_rolling_mean_shift()` helper. |
| FR-2.2 | Add pace (possessions/game) as rolling feature | `avg_poss = (team_poss_est + opp_poss_est) / 2` where `poss_est = FGA - OREB + TOV + 0.44*FTA`. This yields ~101 avg_poss per game, consistent with `team_stats_advanced.pace` (~99). Requires same self-join. |
| FR-2.3 | Add turnover rate per possession (not raw count) as rolling feature | `tov_per_poss = TOV / poss_est`. Zero nulls in 2014+ data. Roll with existing pattern. Distinct from the Four Factors TOV% formula (which uses `FGA + 0.44*FTA + TOV` as denominator). |
| FR-2.4 | Add Four Factors differential composite (eFG%, TOV%, ORB%, FT rate) as matchup feature | Four game-level metrics roll into per-team features; composite assembled in `build_matchup_dataset()` using Oliver weights. One `four_factors_composite` column per matchup row. |
| FR-2.5 | Restrict game outcome model training to 2014+ seasons; exclude 2019-20 bubble and 2020-21 shortened seasons | `MODERN_ERA_ONLY = True`, `MODERN_ERA_START = '201314'`, add `EXCLUDED_SEASONS = ['201920', '202021']` filter. The toggle machinery already exists; change defaults + add exclusion logic. |
| NFR-1 | All rolling features use `.shift(1)` before `.rolling()` to prevent lookahead bias; assert shape > 0 after joins; log null rates | The `_rolling_mean_shift()` helper already enforces shift-1. New features must use this helper or the same idiom. Assert and null-rate checks follow the existing `validate_feature_null_rates()` pattern. |
</phase_requirements>

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pandas | (project version) | Per-game metric computation, self-joins, rolling windows | All feature engineering is already pandas; no new dep needed |
| numpy | (project version) | Possession estimation formula (array ops) | Used throughout feature module |
| scikit-learn | 1.x (post-1.8 breaking change documented) | Model training with era filter | Already used; `_CalibratedWrapper` pattern in place for 1.8+ |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| None needed | — | All Phase 2 work uses existing stack | — |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Game-level box score derivation | Join season-level `team_stats_advanced` directly | Season-level gives one row per team per season — cannot produce per-game rolling features without also having game-level data. Box score derivation is correct for the rolling requirement. |
| Separate rolling function | Extending `ROLL_STATS` list | Some metrics (ORtg, DRtg, oreb_pct) need a self-join before rolling; they cannot be added to `ROLL_STATS` directly. A separate computation block before the roll loop is cleaner. |

**Installation:** No new packages required. All dependencies already present.

---

## Architecture Patterns

### Recommended Project Structure

No new files needed. Modifications confined to three existing files:
```
src/features/team_game_features.py    # add per-game advanced metric computation + rolling
src/models/game_outcome_model.py      # flip MODERN_ERA_ONLY, change start season, add exclusions
```
`build_matchup_dataset()` is in `team_game_features.py` and handles the Four Factors composite.

### Pattern 1: Per-Game Advanced Metric Derivation (Box Score Self-Join)

**What:** Compute ORtg, DRtg, net rating, eFG%, TS%, pace, tov_per_poss, oreb_pct, ft_rate from game box score using the same opponent self-join already used for `opp_pts` and `opp_dreb_game`.

**When to use:** Any time a metric requires opponent data (DRtg, net rating, oreb_pct, pace).

**Example:**
```python
# Source: existing pattern in team_game_features.py (lines ~363-371)
# Self-join for opponent box score columns (extend existing opp_style join)
opp_box = df[[
    "team_abbreviation", "game_id",
    "pts", "fga", "fg3m", "fgm", "oreb", "tov", "fta", "dreb"
]].copy()
opp_box.columns = [
    "opponent_abbr", "game_id",
    "opp_pts_raw", "opp_fga", "opp_fg3m", "opp_fgm",
    "opp_oreb", "opp_tov", "opp_fta", "opp_dreb"
]
df = df.merge(opp_box, on=["opponent_abbr", "game_id"], how="left")

# Possession estimates (Oliver formula)
df["poss_est"]     = df["fga"] - df["oreb"] + df["tov"] + 0.44 * df["fta"]
df["opp_poss_est"] = df["opp_fga"] - df["opp_oreb"] + df["opp_tov"] + 0.44 * df["opp_fta"]
df["avg_poss"]     = (df["poss_est"] + df["opp_poss_est"]) / 2

# Per-game advanced metrics (raw, not yet rolled)
df["off_rtg_game"]    = df["pts"] / df["avg_poss"].replace(0, np.nan) * 100
df["def_rtg_game"]    = df["opp_pts_raw"] / df["avg_poss"].replace(0, np.nan) * 100
df["net_rtg_game"]    = df["off_rtg_game"] - df["def_rtg_game"]
df["pace_game"]       = df["avg_poss"]   # possessions per game (~99-101 in modern era)
df["efg_game"]        = (df["fgm"] + 0.5 * df["fg3m"]) / df["fga"].replace(0, np.nan)
df["ts_game"]         = df["pts"] / (2 * (df["fga"] + 0.44 * df["fta"])).replace(0, np.nan)
df["tov_poss_game"]   = df["tov"] / df["poss_est"].replace(0, np.nan)   # FR-2.3
df["tov_pct_game"]    = df["tov"] / (df["fga"] + 0.44 * df["fta"] + df["tov"]).replace(0, np.nan)  # Four Factors TOV%
df["oreb_pct_game"]   = df["oreb"] / (df["oreb"] + df["opp_dreb"]).replace(0, np.nan)  # needs opp_dreb from join
df["ft_rate_game"]    = df["fta"] / df["fga"].replace(0, np.nan)
```

### Pattern 2: Rolling Advanced Metrics with Shift-1 Leakage Prevention

**What:** Roll per-game advanced metrics using the existing `_rolling_mean_shift()` helper. New metrics go through the same group-apply pattern as existing `ROLL_STATS`.

**When to use:** After computing per-game raw values (pattern 1 above).

**Example:**
```python
# Source: existing pattern in team_game_features.py (lines ~317-330)
ADV_ROLL_STATS = [
    "off_rtg_game", "def_rtg_game", "net_rtg_game", "pace_game",
    "efg_game", "ts_game", "tov_poss_game", "tov_pct_game",
    "oreb_pct_game", "ft_rate_game",
]
adv_group = df.groupby("team_id", group_keys=False)

for window in roll_windows:   # [5, 10, 20]
    for stat in ADV_ROLL_STATS:
        col_name = f"{stat}_roll{window}"
        df[col_name] = adv_group.apply(
            lambda g: _rolling_mean_shift(g, stat, window),
            include_groups=False,
        ).values
```

### Pattern 3: Four Factors Composite in build_matchup_dataset()

**What:** After home/away pivot, compute rolling per-team Four Factors values and then assemble a composite differential using Dean Oliver's weights.

**When to use:** In `build_matchup_dataset()`, after the home/away merge, before saving.

**Example:**
```python
# Source: Dean Oliver "Basketball on Paper" (2004) - Oliver weights
# Note: TOV% coefficient is negative (more turnovers = disadvantage)
FOUR_FACTORS_WEIGHTS = {
    "efg_game_roll20":      +0.40,
    "tov_pct_game_roll20":  -0.25,
    "oreb_pct_game_roll20": +0.20,
    "ft_rate_game_roll20":  +0.15,
}

def _four_factors_composite(matchup: pd.DataFrame, weights: dict) -> pd.Series:
    composite = pd.Series(0.0, index=matchup.index)
    for feat, weight in weights.items():
        h_col = f"home_{feat}"
        a_col = f"away_{feat}"
        if h_col in matchup.columns and a_col in matchup.columns:
            diff = matchup[h_col] - matchup[a_col]
            composite += weight * diff.fillna(0)
    return composite

matchup["four_factors_composite"] = _four_factors_composite(matchup, FOUR_FACTORS_WEIGHTS)
```

### Pattern 4: Era Filtering with Excluded Seasons

**What:** Extend the existing `MODERN_ERA_ONLY` / `MODERN_ERA_START` machinery to also exclude specific anomalous seasons.

**When to use:** In `train_game_outcome_model()` after loading matchup data, before train/test split.

**Example:**
```python
# Source: game_outcome_model.py lines ~222-225 (extend existing pattern)
MODERN_ERA_ONLY     = True           # CHANGED from False
MODERN_ERA_START    = "201314"       # CHANGED from "201415" (per SC1: 2013-14+)
EXCLUDED_SEASONS    = ["201920", "202021"]  # NEW: bubble + shortened

# In train_game_outcome_model():
start_season = MODERN_ERA_START if modern_era_only else TRAIN_START_SEASON
df = df[df["season"].astype(str) >= start_season].copy()
if modern_era_only and excluded_seasons:
    before = len(df)
    df = df[~df["season"].astype(str).isin(excluded_seasons)].copy()
    print(f"  Excluded {before - len(df):,} games from anomalous seasons: {excluded_seasons}")
```

### Anti-Patterns to Avoid

- **Computing rolling features without `.shift(1)`:** The `_rolling_mean_shift()` helper enforces this. If you bypass it by calling `.rolling()` directly, you will include the current game's result in the rolling window — a data leakage bug. All new features MUST use the helper.
- **Adding new metrics to `ROLL_STATS` directly:** DRtg, oreb_pct, and pace require opponent data from a self-join that happens BEFORE the rolling loop. They cannot be in `ROLL_STATS` (which only sees columns available at load time). Compute them in a separate block after the self-join, then roll in a separate ADV_ROLL_STATS loop.
- **Using season-level `team_stats_advanced` for rolling:** `team_stats_advanced` has one row per team per season (end-of-season). Joining this to game-level data gives the same stale value for every game in a season — not a rolling feature. Use box score computation for game-level rolling.
- **Confusing `tov_poss_game` with `tov_pct_game`:** FR-2.3 requires `tov / poss_est` (per-possession rate). The Four Factors TOV% uses `tov / (fga + 0.44*fta + tov)` (Oliver's formula). Both are needed; they are different columns.
- **Merging opp_dreb twice:** The existing code already joins `opp_dreb_game` for the rebounding_edge feature. Reuse the existing join rather than adding a second identical merge.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Possession estimation | Custom formula | Oliver formula: `FGA - OREB + TOV + 0.44*FTA` | Industry standard; produces values consistent with NBA's own pace figures (~99-101 per game) |
| Rolling window with leakage prevention | Custom loop | `_rolling_mean_shift()` already in codebase | The helper is tested and correct. Rolling a custom loop risks off-by-one errors in the shift. |
| Four Factors weighting | Custom weights | Dean Oliver's canonical weights (0.40 / 0.25 / 0.20 / 0.15) | Academic consensus; widely reproduced in NBA analytics literature |
| Era filtering | Custom date logic | Extend existing `MODERN_ERA_ONLY` / `MODERN_ERA_START` flags | The machinery is already there and tested; only constants need to change |

**Key insight:** Nearly every algorithmic primitive needed for Phase 2 already exists in the codebase or is a well-established NBA analytics formula. Phase 2 is primarily wiring, not invention.

---

## Common Pitfalls

### Pitfall 1: Using Opponent Data Before the Self-Join

**What goes wrong:** The self-join that fetches opponent box score columns (`opp_fga`, `opp_oreb`, etc.) must happen BEFORE computing DRtg, net_rtg, oreb_pct, and pace. If the computation block is placed before the merge, these columns will not exist.

**Why it happens:** The rolling stats loop for existing metrics (pts, fg_pct, etc.) runs without a self-join. It's easy to assume the same is true for advanced metrics.

**How to avoid:** Place the opponent box score join block explicitly before the ADV_ROLL_STATS loop. Assert `opp_fga` is in `df.columns` before computing possession estimates.

**Warning signs:** `KeyError: 'opp_fga'` at feature computation time; all advanced metrics returning NaN.

### Pitfall 2: Duplicate Opponent Join Creating Naming Conflicts

**What goes wrong:** The existing code already performs a self-join to fetch `opp_dreb_game` for the `rebounding_edge` feature (lines ~366-371 in `team_game_features.py`). Adding a second join for opponent box score will create duplicate or conflicting column names.

**Why it happens:** Two separate merge operations both trying to add `opp_*` columns.

**How to avoid:** Extend the EXISTING opponent self-join to include all needed columns (`pts`, `fga`, `fg3m`, `fgm`, `oreb`, `tov`, `fta`, `dreb`) in a single merge. Do not add a second self-join for the advanced metrics.

**Warning signs:** Duplicate column name errors on merge; `_x`/`_y` suffix columns appearing.

### Pitfall 3: Wrong MODERN_ERA_START Value

**What goes wrong:** The existing constant `MODERN_ERA_START = '201415'` (2014-15) would exclude the 2013-14 season. Success criterion 1 explicitly states "2013-14 through 2023-24."

**Why it happens:** The era_labels.py defines "3-Point Revolution" as starting 2015-16. A developer might use that boundary (201516) or the existing constant (201415) rather than 201314.

**How to avoid:** Set `MODERN_ERA_START = '201314'` and verify with `df['season'].min()` after the filter prints `201314`. The dataset has 1,230 games in 201314 with 0% null fga.

**Warning signs:** Training data prints `201415` as the start season; missing one full season of modern era data.

### Pitfall 4: Missing `four_factors_composite` Column in `get_feature_cols()`

**What goes wrong:** The new `four_factors_composite` column is computed in `build_matchup_dataset()` and saved to `game_matchup_features.csv`. But `get_feature_cols()` in `game_outcome_model.py` selects columns by `diff_` prefix and explicit lists. If `four_factors_composite` does not start with `diff_` and is not in the explicit set, it will be silently excluded from model training.

**Why it happens:** The new composite column has a custom name, not the `diff_` convention used for all other differential features.

**How to avoid:** Either rename the column to `diff_four_factors_composite` (preferred — keeps consistency with the `diff_` prefix convention used in `diff_stats`) or add it explicitly to the schedule/context columns in `get_feature_cols()`.

**Warning signs:** Column exists in CSV but importances show it was never used (zero importance); `four_factors_composite` not appearing in `feat_cols` list from `get_feature_cols()`.

### Pitfall 5: Accuracy Comparison Uses Different Holdout Sets

**What goes wrong:** Success criterion 5 says "accuracy on 2014+ holdout set meets or exceeds accuracy on the full historical dataset." If the two runs use different test seasons, the comparison is invalid.

**Why it happens:** Full-history run uses the same `TEST_SEASONS = ['202324', '202425']`, which are also modern-era games. But when running full-history, the test set includes 2023-24 and 2024-25 games that were also part of the training distribution for modern-only.

**How to avoid:** The comparison is: run model on full history with test set from modern era only (e.g., all 2014+ games held out as test), compare to modern-era-only model on same test set. Use a fixed holdout (e.g., seasons 202223 and 202324) for both runs. Document the comparison methodology in the code.

**Warning signs:** Comparing different test sets gives misleading results; the "improvement" could just be artifact of different test data.

---

## Code Examples

Verified patterns from existing codebase:

### Existing Shift-1 Rolling Helper
```python
# Source: src/features/team_game_features.py line ~77-88
def _rolling_mean_shift(group: pd.DataFrame, col: str, window: int) -> pd.Series:
    return (
        group[col]
        .shift(1)
        .rolling(window=window, min_periods=1)
        .mean()
    )
```

### Existing Self-Join Pattern (Style Mismatch, Lines ~366-371)
```python
# Source: src/features/team_game_features.py
opp_style = df[["team_abbreviation", "game_id", "three_rate_raw", "dreb"]].copy()
opp_style.columns = [
    "opponent_abbr", "game_id", "opp_three_rate_game", "opp_dreb_game"
]
df = df.merge(opp_style, on=["opponent_abbr", "game_id"], how="left")
```
**For Phase 2:** Extend this join to include `pts`, `fga`, `fg3m`, `fgm`, `oreb`, `tov`, `fta` from the opponent side. Replace the existing narrow join with one that captures all needed opponent box score columns.

### Existing Differential Feature Pattern (build_matchup_dataset, Lines ~650-678)
```python
# Source: src/features/team_game_features.py
diff_stats = ["pts_roll5", "win_pct_roll20", ...]
for stat in diff_stats:
    h_col = f"home_{stat}"
    a_col = f"away_{stat}"
    if h_col in matchup.columns and a_col in matchup.columns:
        matchup[f"diff_{stat}"] = matchup[h_col] - matchup[a_col]
```
**For Phase 2:** Add advanced metric rolled columns to `diff_stats` (e.g., `"off_rtg_game_roll20"`, `"efg_game_roll20"`, etc.). Add `"diff_four_factors_composite"` as a separately-computed composite at the end.

### Existing Era Filter (game_outcome_model.py Lines ~222-225)
```python
# Source: src/models/game_outcome_model.py
start_season = MODERN_ERA_START if modern_era_only else TRAIN_START_SEASON
df = df[df["season"].astype(str) >= start_season].copy()
label = f"modern era only ({MODERN_ERA_START}+)" if modern_era_only else f"from {TRAIN_START_SEASON}"
print(f"  Training data {label}: {len(df):,} games")
```
**For Phase 2:** Add one line after this block: `df = df[~df["season"].astype(str).isin(excluded_seasons)].copy()`

---

## Data Landscape

### Verified Column Availability in team_game_logs.csv (2014+ seasons)

| Column | Null Rate (2014+) | Used For |
|--------|-------------------|----------|
| fgm | 0% | eFG% numerator |
| fga | 0% | eFG% denominator, possession estimate, ft_rate |
| fg3m | 0% | eFG% bonus term |
| fta | 0% | possession estimate, TS%, ft_rate |
| oreb | 0% | possession estimate, oreb_pct |
| dreb | 0% | oreb_pct via opponent join |
| tov | 0% | tov_per_poss, possession estimate, Four Factors TOV% |
| pts | 0% | ORtg numerator, TS% numerator |
| min | 0% | 240 for regulation (5 players * 48 min) |

### Season Inventory (game_matchup_features.csv)

| Season | Games | Notes |
|--------|-------|-------|
| 201314 | 1,230 | Earliest modern era training game (SC1) |
| 201415-201819 | 1,230/season | Full seasons, clean data |
| 201920 | 1,059 | EXCLUDE: bubble (72 games/team, neutral site) |
| 202021 | 1,080 | EXCLUDE: shortened COVID season (72 games/team) |
| 202122-202223 | 1,230/season | Full seasons, clean data |
| 202324 | 1,230 | Test season (SC5 holdout) |
| 202425 | 1,225 | Test season |
| 202526 | 870 | In-progress (current season) |

**Training set (modern era, excluding test + excluded):** ~9,480 games
**Test set (holdout):** ~2,455 games (202324 + 202425)

### team_stats_advanced.csv Role

This table has season-level aggregates (one row per team per season, 1996-97 through 2025-26). It is NOT used for game-level rolling features directly. Its value in Phase 2 is:
- As a formula reference: the column names and definitions (`off_rating`, `def_rating`, `net_rating`, `efg_pct`, `ts_pct`, `pace`, `tm_tov_pct`, `oreb_pct`) confirm what metrics to compute
- As a sanity check: rolling averages of per-game computed metrics should converge toward the season-level values in this table

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| Raw box score counts (pts, tov, reb) | Pace-normalized rates (ORtg per 100 poss, TOV per poss) | Removes pace confounding — a 110 ORtg in 2023 = same efficiency as in 2015 even though pace differs |
| Full historical training (1996+) | Modern era only (2013-14+) | Eliminates structural non-stationarity from pre-analytics eras where game style was fundamentally different |
| Rolling raw tov count | Rolling tov per possession | Faster teams have more raw turnovers by definition; per-poss rate is pace-neutral |
| Independent offensive/defensive stats | Four Factors composite differential | Single composite captures the interaction of all four efficiency dimensions in one predictive signal |

---

## Open Questions

1. **Merging the extended self-join into the existing narrow self-join**
   - What we know: The existing code joins `three_rate_raw` and `dreb` for `opp_style`. Adding `pts`, `fga`, `fg3m`, `fgm`, `oreb`, `tov`, `fta` to this join produces a wider merge.
   - What's unclear: Whether the existing variable name `opp_style` will cause confusion if reused for the broader join.
   - Recommendation: Replace the narrow `opp_style` join with a broader `opp_box` join and rename accordingly. This is a single merge call — no duplication.

2. **`four_factors_composite` vs `diff_four_factors_composite` naming**
   - What we know: All other differential features use the `diff_` prefix (enforced by `get_feature_cols()` which looks for `diff_` columns).
   - What's unclear: Whether the composite should be named `diff_four_factors_composite` or `four_factors_composite`.
   - Recommendation: Use `diff_four_factors_composite` so it is automatically included by `get_feature_cols()` without any changes to that function.

3. **Accuracy comparison methodology for SC5**
   - What we know: SC5 requires modern-era accuracy >= full-history accuracy on the "2014+ holdout set." This means both runs must use the same holdout.
   - What's unclear: The exact holdout used for the comparison (the test seasons 202324/202425 are already modern-era).
   - Recommendation: Use the existing `TEST_SEASONS = ['202324', '202425']` as the fixed holdout for both runs. Run the model twice: once with `MODERN_ERA_ONLY=False` (full history, same test seasons), once with `MODERN_ERA_ONLY=True` (filtered training, same test seasons). Compare test_accuracy from each run.

---

## Validation Architecture

> `workflow.nyquist_validation` is not present in `.planning/config.json` (key absent, treated as false). Validation Architecture section is included for reference but no Wave 0 test infrastructure is required.

The project has no formal unit test framework (no pytest.ini, no test files). Validation is manual: console output inspection + model backtest reports.

### Phase Requirements -> Validation Map

| Req ID | Behavior | Validation Method | File |
|--------|----------|-------------------|------|
| FR-2.1 | Rolling ORtg, DRtg, net_rtg, eFG%, TS% appear in feature table | `df.columns` inspection after `build_team_game_features()` + null rate print | `src/features/team_game_features.py` |
| FR-2.2 | Rolling pace appears in feature table | Column presence check + sanity range check (~85-115 avg_poss range) | `src/features/team_game_features.py` |
| FR-2.3 | tov_per_poss appears as rolling feature | Column presence check + verify value range (expect 0.10-0.20) | `src/features/team_game_features.py` |
| FR-2.4 | `diff_four_factors_composite` exists in matchup CSV | Column presence in `game_matchup_features.csv` after `build_matchup_dataset()` | `src/features/team_game_features.py` |
| FR-2.5 | Training data filtered to 201314+ excluding 201920/202021 | Console print "Training data modern era only (201314+): N games" confirms filter; N should be ~9,480 | `src/models/game_outcome_model.py` |
| NFR-1 | Shift-1 applied before rolling on all new features | Code review: all new metrics use `_rolling_mean_shift()` helper | `src/features/team_game_features.py` |

### Accuracy Comparison (SC5)
```
python src/models/game_outcome_model.py   # with MODERN_ERA_ONLY=True -> record test_accuracy
# Temporarily set MODERN_ERA_ONLY=False, same TEST_SEASONS
python src/models/game_outcome_model.py   # full history -> record test_accuracy
# Assert: modern_accuracy >= full_history_accuracy
```

---

## Sources

### Primary (HIGH confidence)
- Codebase direct inspection — `src/features/team_game_features.py` full read, confirmed self-join pattern at lines 363-371, rolling helper at lines 77-88
- Codebase direct inspection — `src/models/game_outcome_model.py` full read, confirmed `MODERN_ERA_ONLY` flag at lines 47-48, `MODERN_ERA_START='201415'` at line 48
- Data verification — `team_game_logs.csv` column null rates for 2014+ seasons (all relevant columns 0% null, verified by Python execution)
- Data verification — `team_stats_advanced.csv` column inventory and season range (1996-97 through 2025-26, verified)
- Data verification — per-game ORtg/DRtg/eFG%/TS%/pace computed from box score, sanity-checked against `team_stats_advanced` seasonal averages (avg_poss ~101 vs pace ~99 in adv table)

### Secondary (MEDIUM confidence)
- Dean Oliver "Basketball on Paper" (2004) — Four Factors weights (0.40/0.25/0.20/0.15) — widely reproduced and accepted in NBA analytics community
- Oliver possession formula: `FGA - OREB + TOV + 0.44*FTA` — standard in the field, consistent with NBA tracking data outputs

### Tertiary (LOW confidence)
- None.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — confirmed from direct codebase inspection; no new dependencies needed
- Architecture patterns: HIGH — all patterns derived from existing codebase code that already works
- Data availability: HIGH — verified by running Python against actual CSV files; 0% null rates confirmed
- Pitfalls: HIGH — identified from reading existing code and understanding merge/column naming conventions
- Four Factors weights: MEDIUM — canonical Oliver weights, widely reproduced but not re-verified against primary source

**Research date:** 2026-03-01
**Valid until:** 2026-06-01 (stable — pandas/sklearn APIs not expected to change; data pipeline is local)
