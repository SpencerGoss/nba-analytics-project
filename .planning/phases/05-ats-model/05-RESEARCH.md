# Phase 5: ATS Model - Research

**Researched:** 2026-03-02
**Domain:** Sports betting model — ATS binary classifier, value-bet detection, odds backfill, CLV backtest
**Confidence:** HIGH (codebase inspection) / MEDIUM (external data sources)

---

## Summary

Phase 5 builds a complete ATS (Against the Spread) betting pipeline on top of the existing game outcome prediction model. The core requirement is a strict data separation: `game_ats_features.csv` contains Vegas spread and implied probability as model inputs, while `game_matchup_features.csv` (used by the win-probability model) contains neither. Value bets are identified by comparing the win-probability model's output to market-implied probabilities — the spread-aware ATS classifier serves as a second, orthogonal signal.

The single largest discovery in this research is that **The Odds API historical odds endpoint is not available on the free tier** (paid plans only, cost is 10 credits per region per market per call, free tier has 500 monthly credits). This means FR-5.4 "backfill historical odds from The Odds API" cannot be fulfilled by that route for historical data. However, Kaggle hosts a well-maintained public dataset covering NBA betting data from October 2007 through June 2025 (sourced from SportsBookReviewsOnline through Jan 2023, ESPN thereafter), which contains spread, moneyline, and `id_spread` (ATS result) columns. This dataset is the correct historical training source; The Odds API free tier covers only current-season forward-looking lines.

The existing project infrastructure is well-suited: `game_matchup_features.csv` has 15,304 modern-era games (201314+) with 272 features and no spread data. The `fetch_odds.py` script already handles `x-requests-remaining` and `x-requests-used` response headers with quota logging. The `game_outcome_model.py` expanding-window validation pattern (`_season_splits()`) is directly reusable for the ATS classifier.

**Primary recommendation:** Use the Kaggle `nba-betting-data` dataset (free, 2007-2025) as the primary historical odds source for training/backtest. Use The Odds API free tier for current-season forward-looking lines only. Join on `game_date + home_team + away_team` (same cross-source key already used in Phase 3 for referee features). Mirror the game_outcome_model.py architecture for the ATS classifier — same expanding-window splits, same sklearn Pipeline, same artifact persistence pattern.

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| FR-5.1 | Create separate `game_ats_features.csv` extending matchup features WITH Vegas spread and implied moneyline probability | Join `game_matchup_features.csv` with Kaggle betting dataset on `game_date + home_team + away_team`; add `spread`, `home_implied_prob`, `away_implied_prob` (no-vig), `covers_spread` target |
| FR-5.2 | Train ATS classifier (target: `covers_spread`) using expanding validation splits | Mirror `_season_splits()` from `game_outcome_model.py`; GradientBoostingClassifier or LogisticRegression; feature set includes all 68 game-outcome features PLUS spread/implied_prob inputs |
| FR-5.3 | Build value-bet identification: flag games where win-probability model disagrees with implied odds by configurable threshold | Read `game_outcome_model_calibrated.pkl`, compute `model_win_prob - market_implied_prob`; threshold already defined as `WINPROB_FLAG_PP = 0.05` in `fetch_odds.py` |
| FR-5.4 | Backfill historical odds data from The Odds API (respect free tier quota) | Historical API endpoint NOT available on free tier. Use Kaggle dataset for historical. Free tier (500 req/month) reserved for current-season lines. Quota audit = check `x-requests-remaining` header before any batch call. |
| FR-5.5 | Backtest ATS model against closing lines over 500+ games; report ROI, CLV, hit rate; refuse < 500 games | 201314+ has 15,304 modern-era games; Kaggle data covers 2007-2025; 500-game guard is trivially satisfied. CLV = (closing_spread - bet_spread) normalized to expected value. ROI = (net_payout / total_staked) |
| NFR-1 | No lookahead leakage — `.shift(1)` before `.rolling()` | ATS features must use same shift(1) pattern; spread/moneyline are pre-game inputs (not future data), so they are legitimate inputs for historical rows where outcome is known |
| NFR-3 | Reproducible results and predictions | `random_state=42` in all classifiers; deterministic feature assembly; ATS model artifacts saved to `models/artifacts/` alongside win-probability model |
</phase_requirements>

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pandas | existing | Feature table assembly, spread join, backtest reporting | Already in project |
| scikit-learn | existing | ATS classifier (GBM/LR/RF), expanding validation, Pipeline | Already in project; same pattern as game_outcome_model.py |
| numpy | existing | No-vig probability calculation, ROI arithmetic | Already in project |
| requests | existing | The Odds API current-season lines fetch | Already in `fetch_odds.py` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| python-dotenv | existing | Read ODDS_API_KEY from .env | Already in fetch_odds.py |
| pickle | stdlib | ATS model artifact persistence | Same pattern as existing model artifacts |
| json | stdlib | ATS model metadata, threshold config | Same pattern as game_outcome_metadata.json |

### No New Libraries Required
All required libraries already exist in the project. No `pip install` step needed.

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Kaggle CSV for historical odds | The Odds API historical endpoint | Odds API historical requires paid plan; Kaggle dataset is free and covers 2007-2025 |
| sklearn GBM | XGBoost / LightGBM | XGBoost not in project; GBM already proven in game_outcome_model.py |
| Manual CLV formula | Third-party library | CLV is simple arithmetic; no library needed |

---

## Architecture Patterns

### Recommended Project Structure

```
src/
  models/
    ats_model.py              # ATS classifier (mirrors game_outcome_model.py)
    ats_backtest.py           # CLV + ROI backtest harness (mirrors backtesting.py)
  features/
    ats_features.py           # build_ats_features() — joins matchup + betting data
  data/
    get_historical_odds.py    # download/load Kaggle betting CSV + persist to data/raw/odds/
data/
  raw/
    odds/
      nba_betting_historical.csv   # Kaggle dataset (2007-2025)
  features/
    game_ats_features.csv     # output of ats_features.py (separate from matchup features)
models/
  artifacts/
    ats_model.pkl
    ats_model_features.pkl
    ats_model_metadata.json
reports/
  ats_backtest.csv            # per-season ROI, CLV, hit rate
  ats_backtest_summary.txt
```

### Pattern 1: Strict Data Separation (FR-5.1 Guard)

**What:** `game_ats_features.csv` is built by joining `game_matchup_features.csv` with the historical odds data, then adding spread/implied_prob columns. The win-probability model (`game_outcome_model.py`) reads ONLY `game_matchup_features.csv` — never the ATS file.

**When to use:** All phases of ATS model training and inference.

**Example:**
```python
# src/features/ats_features.py

def build_ats_features(
    matchup_path: str = "data/features/game_matchup_features.csv",
    odds_path: str = "data/raw/odds/nba_betting_historical.csv",
    output_path: str = "data/features/game_ats_features.csv",
) -> pd.DataFrame:
    """
    Build ATS feature table by joining matchup features with betting odds.

    IMPORTANT: This file intentionally contains spread and implied probability.
    The win-probability model (game_outcome_model.py) MUST NOT use this file.
    """
    matchup = pd.read_csv(matchup_path)
    odds = load_and_normalize_odds(odds_path)  # normalize team names, dates

    # Join on date + home team + away team (same cross-source key as Phase 3)
    merged = matchup.merge(
        odds[["game_date", "home_team", "away_team",
              "spread", "home_moneyline", "away_moneyline",
              "covers_spread"]],
        on=["game_date", "home_team", "away_team"],
        how="inner",  # only rows with known odds — drops pre-2007 games
    )

    # Compute no-vig implied probabilities (removes bookmaker overround)
    merged["home_implied_prob"] = merged["home_moneyline"].apply(
        _american_to_no_vig_prob_home,
        away_ml=merged["away_moneyline"]
    )
    merged.to_csv(output_path, index=False)
    return merged
```

### Pattern 2: No-Vig Probability Calculation

**What:** Convert raw American moneyline odds to fair (no-vig) implied probability. The raw conversion over-counts probability (sums > 1.0); dividing by the total removes the bookmaker's take.

**Example:**
```python
def american_to_prob(ml: float) -> float:
    """Raw implied probability from American odds (vig-inclusive)."""
    if ml > 0:
        return 100 / (ml + 100)
    else:
        return abs(ml) / (abs(ml) + 100)

def no_vig_prob(home_ml: float, away_ml: float) -> tuple[float, float]:
    """Remove vig using multiplicative method; return (home_prob, away_prob)."""
    raw_home = american_to_prob(home_ml)
    raw_away = american_to_prob(away_ml)
    total = raw_home + raw_away            # overround (e.g., 1.04)
    return raw_home / total, raw_away / total
```

Note: `fetch_odds.py` already has `american_odds_to_implied_prob()` which does the raw conversion. The no-vig step (dividing by total) must be added for ATS features.

### Pattern 3: Expanding-Window Validation (mirrors game_outcome_model.py)

**What:** Reuse `_season_splits()` pattern from `game_outcome_model.py` exactly. Train on seasons 1..N-1, validate on season N, iterate forward.

**Example:**
```python
# In ats_model.py — mirrors _season_splits() from game_outcome_model.py
def _ats_season_splits(df: pd.DataFrame, min_train: int = 4) -> list:
    seasons = sorted(df["season"].astype(str).unique())
    splits = []
    for i in range(max(1, min_train - 1), len(seasons)):
        train_seasons = seasons[:i]
        valid_season  = seasons[i]
        tr = df[df["season"].astype(str).isin(train_seasons)].copy()
        va = df[df["season"].astype(str) == valid_season].copy()
        if not tr.empty and not va.empty:
            splits.append((tr, va, valid_season))
    return splits
```

### Pattern 4: CLV and ROI Computation

**What:** Closing Line Value measures whether our bets beat the final market price. ROI measures profitability of a flat-staking strategy.

**Example:**
```python
def compute_clv(row: dict) -> float:
    """
    CLV = (closing_spread - open_spread) from bettor's perspective.
    Positive CLV = we bet a better number than the closing line.
    For spread bets: CLV = closing_spread_for_our_side - spread_we_bet.
    """
    return row["closing_spread"] - row["opening_spread"]

def compute_roi(bets_df: pd.DataFrame, stake: float = 1.0) -> float:
    """
    Flat-stake ROI for -110 standard spread bets.
    Win pays +stake * (100/110). Loss costs -stake.
    ROI = net_profit / total_staked.
    """
    wins = bets_df["covers_spread"].sum()
    losses = len(bets_df) - wins
    net = wins * (100 / 110) * stake - losses * stake
    total_staked = len(bets_df) * stake
    return net / total_staked if total_staked > 0 else 0.0
```

### Pattern 5: Value-Bet Detection

**What:** Compare calibrated win-probability model output to no-vig market implied probability. Flag when difference exceeds configurable threshold.

**Example:**
```python
# WINPROB_FLAG_PP already defined in fetch_odds.py as 0.05 (5pp)
# For ATS value-bet detector, use same threshold or make configurable

VALUE_BET_THRESHOLD = float(os.getenv("VALUE_BET_THRESHOLD", "0.05"))

def detect_value_bets(
    model_win_prob: float,
    market_implied_prob: float,
    threshold: float = VALUE_BET_THRESHOLD,
) -> bool:
    return abs(model_win_prob - market_implied_prob) > threshold
```

### Pattern 6: Quota Guard Before Batch API Calls

**What:** Read `x-requests-remaining` header before any multi-game batch fetch. The existing `get_odds_api()` in `fetch_odds.py` already logs this header on every call — the batch guard needs to READ it before looping.

**Example:**
```python
def check_quota_before_batch(min_remaining: int = 50) -> int:
    """
    Make a single cheap API call to read remaining quota.
    Returns remaining credits, raises QuotaError if below min_remaining.
    Uses /sports endpoint (cheapest: 1 credit).
    """
    r = requests.get(
        f"{BASE_URL}/sports",
        params={"apiKey": API_KEY},
        timeout=10,
    )
    remaining = int(r.headers.get("x-requests-remaining", 0))
    if remaining < min_remaining:
        raise QuotaError(f"Only {remaining} API credits remain — aborting batch.")
    return remaining
```

### Anti-Patterns to Avoid

- **Using game_ats_features.csv as input to game_outcome_model.py**: Spread is a market-consensus input that would give the win-probability model unfair signal (it would just learn to invert the spread). The DATA SEPARATION must be code-level, not just by convention.
- **Using raw (vig-inclusive) implied probabilities**: Raw conversion of `-110` gives 52.38% per side = 104.76% total. Always remove vig before comparing to model win probability.
- **Computing ATS target after the fact on training rows**: `covers_spread = (home_score - away_score) > spread` is a deterministic computation from known outcomes. This is NOT leakage — spread is a pre-game input, outcome is post-game. The shift(1) / rolling NFR-1 rule applies to rolling features only.
- **Treating 2019-20 bubble / 2020-21 shortened seasons as normal**: These are already excluded from win-probability model training (`EXCLUDED_SEASONS`). Same exclusion should apply to ATS model training.
- **Reporting backtest on < 500 games**: The 500-game guard must be a hard assert/raise in the backtest script, not a warning.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Team name normalization (Kaggle → project abbreviations) | Custom fuzzy matcher | Static dict (same `ODDS_TEAM_TO_ABB` in fetch_odds.py) | 30 fixed NBA teams; dict already exists in project |
| No-vig probability removal | Complex Bayesian adjustment | Multiplicative method (2 lines of math) | Standard formula; 2-outcome markets; no library needed |
| ATS result computation | External lookup | `covers_spread = (home_score - away_score + spread) > 0` | Deterministic from known final score + spread |
| Model validation splits | TimeSeriesSplit from sklearn | Custom `_season_splits()` (already in project) | Season boundaries matter more than uniform time splits; existing pattern works |
| CLV calculation | External library | Arithmetic formula (see Code Examples) | Simple difference of probabilities or spread numbers |

---

## Common Pitfalls

### Pitfall 1: The Odds API Historical Endpoint is Paid-Only
**What goes wrong:** Attempting to backfill historical odds via The Odds API free tier returns 401 or 403. The `/historical_odds` endpoint costs 10 credits per region per market per call — even if it were available, 500 free credits would cover only 50 historical snapshots.
**Why it happens:** Historical data is a premium feature; the free tier is designed only for current live/upcoming games.
**How to avoid:** Use the Kaggle `nba-betting-data-october-2007-to-june-2024` dataset (free, 2007-2025) for all training and backtest data. Use The Odds API free tier ONLY for current-season upcoming game lines.
**Warning signs:** The `x-requests-remaining` header drops to near zero after only a few historical calls.

### Pitfall 2: Team Name Mapping Between Datasets
**What goes wrong:** Kaggle dataset uses full city/team names from SportsBookReviewsOnline (e.g., "GoldenState" or "LA Clippers") while our database uses 3-letter NBA abbreviations (GSW, LAC). The join silently produces zero rows.
**Why it happens:** Two different data sources with different naming conventions.
**How to avoid:** Inspect the Kaggle dataset's `home` and `away` columns before building the join. Extend or reuse `ODDS_TEAM_TO_ABB` from `fetch_odds.py`. Add an assertion: `assert merged.shape[0] > 0` after the join.
**Warning signs:** `game_ats_features.csv` has far fewer rows than expected, or inner join yields 0 rows.

### Pitfall 3: Double-Counting Vig in Value-Bet Comparison
**What goes wrong:** Comparing raw model win probability (vig-free calibrated output) to raw market implied probability (vig-inclusive) systematically biases the gap calculation.
**Why it happens:** `american_odds_to_implied_prob()` in `fetch_odds.py` returns vig-inclusive probabilities. Raw `-110` gives 52.38% not 50%.
**How to avoid:** Always use no-vig implied probability for the comparison column in `game_ats_features.csv`. The `home_implied_prob` and `away_implied_prob` columns must use the two-sided normalization.
**Warning signs:** Value-bet detector flags nearly every game because the market "looks" to have edge on every game.

### Pitfall 4: ATS Target Computation Error
**What goes wrong:** `covers_spread` computed incorrectly, especially push cases (home margin exactly equals spread).
**Why it happens:** Point spread is from home team's perspective in some datasets (negative = home favored) but could be reversed. Push (exact tie against spread) should be excluded from ATS backtest, not counted as a loss.
**How to avoid:** Verify spread sign convention in Kaggle dataset. The Kaggle `id_spread` column already encodes this: 1 = favorite covered, 0 = underdog covered, 2 = push. Use the existing `id_spread` rather than recomputing from final scores.
**Warning signs:** ATS hit rate is exactly 50% on first test (may indicate push/tie rows are not being excluded).

### Pitfall 5: Lookahead via Closing Line as Input Feature
**What goes wrong:** Using closing line spread (known post-bet) as an input feature to the ATS classifier. Opening spread is the correct input — it's available pre-game.
**Why it happens:** Some historical datasets have both opening and closing lines. Closing line is a better predictor but represents information unavailable at bet time.
**How to avoid:** Use opening spread as the ATS model input feature. Use closing spread ONLY in the CLV backtest report (as the comparison target, not a training feature).
**Warning signs:** ATS model achieves suspiciously high accuracy (> 58%).

### Pitfall 6: Sparse Odds Data for Early Seasons
**What goes wrong:** Kaggle dataset starts in 2007-08; project's game_matchup_features.csv has games back to 1946. The inner join for ATS training data limits coverage to 2007-08 onward.
**Why it happens:** Sports betting data is not available for pre-2007 NBA seasons in free public datasets.
**How to avoid:** Expected behavior — ATS model trains on 2007-08 onward (still 1,200+ games per season × 17 seasons = ~17,000 games). 500-game backtest requirement is easily met. Document the ATS model coverage start clearly.

### Pitfall 7: Windows cp1252 Encoding
**What goes wrong:** Print statements with special characters (arrows, em-dashes) crash on Windows.
**Why it happens:** Project runs on Windows cp1252 terminal (documented in STATE.md decisions).
**How to avoid:** Use only ASCII characters in all print statements in new ATS scripts.

---

## Code Examples

### ATS Target Computation from Kaggle Data
```python
# Source: Kaggle nba-betting-data dataset analysis
# id_spread: 1=favorite covered, 0=underdog covered, 2=push
# whos_favored: 'home' or 'away'

def compute_home_covers_spread(row):
    """Return 1 if home team covered spread, 0 if not, NaN if push."""
    if row["id_spread"] == 2:
        return np.nan   # push — exclude from training and backtest
    if row["whos_favored"] == "home":
        return int(row["id_spread"] == 1)  # home was favored and covered
    else:
        return int(row["id_spread"] == 0)  # away was favored, underdog (home) covered
```

### CLV Formula (spread-based)
```python
def compute_clv_spread(bet_spread: float, closing_spread: float, bet_side: str) -> float:
    """
    CLV for spread bet.
    Positive = we got better number than market closed at.

    bet_spread: spread at time of bet (from home team's perspective, negative=home favored)
    closing_spread: final market spread
    bet_side: 'home' or 'away'
    """
    if bet_side == "home":
        # Home team covers more easily with larger spread (more points given)
        return bet_spread - closing_spread  # positive if we got home + more points
    else:
        return closing_spread - bet_spread  # positive if we got away + more points
```

### ROI Computation (flat -110 betting)
```python
def compute_roi_flat_110(covers: pd.Series) -> dict:
    """
    Standard -110 spread bet ROI (flat staking, $110 to win $100).
    Excludes pushes (NaN values).
    """
    valid = covers.dropna()
    if len(valid) < 500:
        raise ValueError(
            f"Backtest requires >= 500 games, got {len(valid)}. "
            "Refusing to report results on insufficient sample."
        )
    wins   = valid.sum()
    losses = len(valid) - wins
    net    = wins * 100 - losses * 110   # dollars on $110/bet staking
    roi    = net / (len(valid) * 110)
    return {
        "n_bets": len(valid),
        "wins":   int(wins),
        "losses": int(losses),
        "hit_rate": float(wins / len(valid)),
        "net_units": float(net / 110),    # normalized to 1 unit = $110
        "roi":       float(roi),
    }
```

### Quota Check Before Batch Fetch
```python
# Based on existing get_odds_api() pattern in fetch_odds.py
def check_remaining_quota() -> int:
    """Read current quota before batch calls. Uses x-requests-remaining header."""
    r = requests.get(
        f"{BASE_URL}/sports",
        params={"apiKey": API_KEY, "all": "false"},
        timeout=10,
    )
    remaining = int(r.headers.get("x-requests-remaining", 0))
    used      = int(r.headers.get("x-requests-used", 0))
    log.info(f"Quota status: used={used} remaining={remaining}")
    return remaining
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Raw vig-inclusive implied prob | No-vig multiplicative method | Standard practice | Accurate edge measurement |
| Single train/test split for ATS | Expanding-window season splits | Standard in project (Phase 2+) | Prevents era leakage |
| Reporting on any sample size | Minimum 500-game guard (hard raise) | FR-5.5 requirement | Prevents overfit reporting |

---

## Key Data Source Discovery

### Kaggle NBA Betting Dataset (PRIMARY for historical odds)
- **URL:** https://www.kaggle.com/datasets/cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024
- **Coverage:** October 30, 2007 through June 2025
- **Columns:** `season, date, regular, playoffs, away, home, score_away, score_home, whos_favored, spread, total, moneyline_away, moneyline_home, id_spread, id_total`
- **Source:** SportsBookReviewsOnline (through Jan 2023), ESPN (from Jan 2023 onward)
- **ATS field:** `id_spread` — 1=favorite covered, 0=underdog covered, 2=push
- **Limitation:** Moneyline absent from ESPN-sourced rows (Jan 2023+). Spread is still present.
- **Cost:** Free, public dataset
- **Games in project modern era overlap (201314+):** ~15,000+ (all 30-team seasons × ~82 games × 11 seasons)

### The Odds API (SECONDARY — current season only)
- **Free tier:** 500 monthly requests total
- **Historical endpoint:** NOT available on free tier (paid plans only, $25+/month)
- **Current-season lines:** Available on free tier — game lines cost ~1-2 credits per call
- **Response headers:** `x-requests-remaining`, `x-requests-used`, `x-requests-last`
- **Existing infrastructure:** `fetch_odds.py` already handles these headers
- **Use in Phase 5:** Current-season upcoming game lines for the value-bet detector only

### SportsBookReviewsOnline (ALTERNATIVE — raw Excel files)
- **URL:** sportsbookreviewsonline.com/scoresoddsarchives/nba/
- **Format:** Excel (.xlsx) per season
- **Coverage:** NBA seasons from approximately 2007-08 onward
- **Columns:** VH (visitor/home), Team, 1H, 2H, Final, Open, Close, ML
- **Note:** Kaggle dataset is already a cleaned/processed version of this source — prefer Kaggle

---

## Open Questions

1. **Kaggle team name format vs project abbreviations**
   - What we know: Kaggle dataset uses full team names from SportsBookReviewsOnline (e.g., "GoldenState", "LAClippers") or possible ESPN format (post Jan 2023)
   - What's unclear: Exact team name strings in `home`/`away` columns — need to inspect the actual CSV
   - Recommendation: Download dataset in 05-01, inspect unique team values, build/extend mapping dict before merge

2. **Opening vs closing spread in Kaggle dataset**
   - What we know: SportsBookReviewsOnline provides opening line; it is unclear if the Kaggle column `spread` is open, close, or a consensus line
   - What's unclear: Whether a separate closing spread column exists
   - Recommendation: Treat `spread` as opening line for ATS model input. For CLV backtest, closing line may need to be fetched from The Odds API (current season only) or approximated from opening.

3. **ATS model feature scope**
   - What we know: All 68 game-outcome features from game_outcome_model.py are valid ATS inputs (they're all pre-game signals). Adding `spread` and `home_implied_prob` as inputs creates the ATS-specific feature set.
   - What's unclear: Whether the model learns to just re-encode the spread signal or finds genuine additional signal from feature interactions
   - Recommendation: Include spread and implied probability as raw features alongside matchup differentials. Let feature importance diagnose whether the model adds value beyond the spread itself.

4. **Calibrated model regeneration status**
   - What we know: STATE.md documents that `game_outcome_model_calibrated.pkl` predates the Phase 4 retrained model. The value-bet detector relies on calibrated probabilities.
   - What's unclear: Whether calibration.py has been re-run after Phase 4 completion
   - Recommendation: 05-03 plan must include running `calibration.py` as a precondition before building the value-bet detector.

---

## Validation Architecture

`workflow.nyquist_validation` not present in `.planning/config.json` — skipping formal validation section.

---

## Sources

### Primary (HIGH confidence — direct codebase inspection)
- `C:/Users/spenc/OneDrive/Desktop/GIT/nba-analytics-project/scripts/fetch_odds.py` — existing odds infrastructure, team mapping, quota headers, `WINPROB_FLAG_PP` threshold
- `C:/Users/spenc/OneDrive/Desktop/GIT/nba-analytics-project/src/models/game_outcome_model.py` — `_season_splits()` pattern, expanding-window validation, artifact persistence
- `C:/Users/spenc/OneDrive/Desktop/GIT/nba-analytics-project/src/models/backtesting.py` — walk-forward backtest pattern, ROI/metrics reporting structure
- `C:/Users/spenc/OneDrive/Desktop/GIT/nba-analytics-project/models/artifacts/game_outcome_metadata.json` — confirmed 68 features, random_forest model, 66.8% accuracy
- `C:/Users/spenc/OneDrive/Desktop/GIT/nba-analytics-project/data/features/game_matchup_features.csv` — confirmed 272 columns, no spread data, 15,304 modern-era games

### Primary (HIGH confidence — official documentation)
- https://the-odds-api.com/liveapi/guides/v4/ — response headers (`x-requests-remaining`, `x-requests-used`, `x-requests-last`), historical endpoint not on free tier confirmed
- https://the-odds-api.com/historical-odds-data/ — historical data costs 10 credits per region per market; available since June 2020; explicitly "paid plans only"

### Secondary (MEDIUM confidence — WebFetch of Kaggle page)
- https://www.kaggle.com/datasets/cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024 — dataset columns, coverage dates, `id_spread` encoding, source attribution confirmed from Kaggle page

### Tertiary (LOW confidence — WebSearch summaries)
- CLV calculation methodology — multiple sports betting education sources agree on multiplicative no-vig method and CLV definition; standard formula not disputed

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already in project, no new dependencies
- Architecture: HIGH — patterns directly derived from existing project code (game_outcome_model.py, backtesting.py, fetch_odds.py)
- Historical data source: MEDIUM — Kaggle dataset confirmed by WebFetch but team name format requires inspection at implementation time
- Pitfalls: HIGH — confirmed from official docs (Odds API free tier) and codebase (vig calculation, Windows encoding)

**Research date:** 2026-03-02
**Valid until:** 2026-04-02 (stable — odds API pricing unlikely to change; Kaggle dataset is static historical)
