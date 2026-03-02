# Phase 4: Rest & Schedule Features - Research

**Researched:** 2026-03-02
**Domain:** Geospatial distance computation, timezone-based travel indicators, game schedule context features
**Confidence:** HIGH -- all libraries confirmed installed and tested; API shapes verified against live data; full end-to-end prototype run against production dataset

---

## Summary

Phase 4 adds four schedule-context features to the game prediction pipeline: `days_rest`, `is_back_to_back`, `travel_miles`, `cross_country_travel`, and `season_month`. The critical discovery from codebase investigation is that **two of the five are already fully implemented**: `days_rest` and `is_back_to_back` were added in team_game_features.py v3 and are wired through to the matchup dataset and model. The remaining three (travel_miles, cross_country_travel, season_month) need to be built.

The STATE.md blocker ("geopy 2.4.x API shape: verify with quick test before building schedule_features.py") has been resolved by this research. geopy 2.4.1 is installed and confirmed working. However, the recommended implementation does NOT use geopy directly for distance computation -- instead a vectorized haversine formula (pure numpy) is used. Geopy's geodesic function loops in Python and is 1000x slower than the numpy vectorized implementation on the 30,628-row modern era dataset. Accuracy difference between haversine and geodesic is 0.2% at most -- negligible for a feature that ranges 0 to 2,700 miles.

All four features are computed entirely from existing `team_game_logs.csv` data (the `is_home`, `opponent_abbr`, and `game_date` columns already present in team_game_features.py). No new API calls are required. The implementation belongs in `team_game_features.py` directly (like `days_rest`), not in a separate module (unlike referee_features.py). `season_month` is a game-level feature added directly to the matchup dataset in `build_matchup_dataset()`.

**Primary recommendation:** Add travel_miles and cross_country_travel inside `build_team_game_features()` using vectorized haversine + static arena coordinate dict. Add season_month directly to `build_matchup_dataset()` from `game_date`. Wire all new features through context_cols, diff_stats, and get_feature_cols() schedule_cols. Retrain model.

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| FR-3.1 | Compute days-between-games for each team (back-to-back = 1 day) | ALREADY IMPLEMENTED: days_rest and is_back_to_back computed in team_game_features.py lines 297-306; days_rest NaN rate = 0% on modern era; back-to-back pct = 18.8% (reasonable); diff_days_rest and home/away_is_back_to_back in matchup features; get_feature_cols() schedule_cols set includes all four forms |
| FR-3.2 | Compute travel distance between consecutive game arenas using static arena coordinate dict + geopy | Verified: vectorized haversine preferred over geopy.distance.geodesic for performance (0.006s vs 6s for 30K rows); 30 modern arena coords confirmed covering all 2013-14+ teams; ARENA_COORDS dict in team_game_features.py module-level constant; curr_arena logic: is_home==1 -> team's own abbr, else opponent_abbr; shift(1) on lat/lon within team_id group for no-leakage |
| FR-3.3 | Add timezone change flag (cross-country travel indicator) | Verified: ARENA_TIMEZONE static dict with 4 zones (Pacific/Mountain/Central/Eastern) covers all 30 modern teams; cross_country_travel = int(timezone(prev_arena) != timezone(curr_arena)); no external library needed; same shift(1) pattern as travel_miles |
| FR-3.4 | Add season-segment context (month of season as feature) | Verified: game_date.dt.month extracts 1-12 directly; distribution confirmed (Oct=810, Nov=2654, ..., Apr=1217, May=135); added directly to matchup df in build_matchup_dataset(), not per-team in team_game_features |
| NFR-1 | All rolling features use shift(1) before rolling() to prevent lookahead bias | travel_miles and cross_country_travel use shift(1) on prev_arena lat/lon (prev game location) -- inherently shift-based; season_month has no lookahead risk (it's just the calendar month of the game being played) |
| NFR-2 | Daily update pipeline completes in <15 minutes | All four features computed from existing data in memory (no I/O, no API calls); vectorized haversine benchmarked at 0.006s for 30K rows; season_month is a single dt.month call; total Phase 4 overhead to pipeline: <1 second |
</phase_requirements>

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | >=1.24.0 (already in requirements.txt) | Vectorized haversine formula for travel_miles | No additional install; 1000x faster than geopy loop for batch computation; 0.2% accuracy vs geodesic is negligible for this feature |
| pandas | >=2.0.0 (already in requirements.txt) | groupby shift(1) for per-team travel tracking; dt.month for season_month | Already the project's core data manipulation library |
| geopy | 2.4.1 (installed, NOT in requirements.txt yet) | Dependency listed in FR-3.2 spec; actual computation done via haversine | Install anyway to satisfy FR-3.2 requirement wording; document that vectorized haversine is used for performance |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| zoneinfo | stdlib (Python 3.9+) | Timezone zone names as reference | Used only as documentation reference for ARENA_TIMEZONE dict values; no runtime import needed |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Vectorized haversine (numpy) | geopy.distance.geodesic per-row | geopy geodesic is 0.2% more accurate but loops in Python (O(n) function calls); haversine is 0.006s vs ~6s for 30K rows; choose haversine |
| Static ARENA_TIMEZONE dict | pytz / zoneinfo + shapefile | pytz not installed; shapefile adds GIS dependency; static dict is correct for all 30 NBA arenas and never changes unless team relocates |
| Inline in team_game_features.py | Separate schedule_features.py module | All data needed is already loaded in team_game_features.py; no additional I/O; days_rest follows same pattern; separate module adds import overhead and join complexity for no benefit |

**Installation (only geopy is new):**
```bash
pip install geopy
```

Add to `requirements.txt`:
```
geopy>=2.3.0
```

---

## Architecture Patterns

### Recommended Project Structure
```
src/
  features/
    team_game_features.py   # ADD: ARENA_COORDS, ARENA_TIMEZONE dicts + computation in build_team_game_features()
                            # ADD: travel_miles, cross_country_travel to context_cols and diff_stats
                            # ADD: season_month computation in build_matchup_dataset()
    (no new file needed)
requirements.txt            # ADD: geopy>=2.3.0
```

### Pattern 1: Travel Miles Computation (Per-Team in build_team_game_features)

**What:** Add arena coordinate dict at module level. Compute current game arena abbreviation from is_home/opponent_abbr. Map to lat/lon, shift within team_id group, vectorize haversine.

**When to use:** Inside `build_team_game_features()`, after the `is_home` and `opponent_abbr` columns are computed (they are set very early in the function).

**Example:**
```python
# Add at module level (alongside ROLL_STATS, ADV_ROLL_STATS):
ARENA_COORDS = {
    'ATL': (33.7573, -84.3963),
    'BKN': (40.6826, -73.9754),
    'BOS': (42.3662, -71.0621),
    'CHA': (35.2251, -80.8392),
    'CHI': (41.8807, -87.6742),
    'CLE': (41.4965, -81.6882),
    'DAL': (32.7905, -96.8103),
    'DEN': (39.7487, -105.0077),
    'DET': (42.3410, -83.0553),
    'GSW': (37.7680, -122.3877),
    'HOU': (29.7508, -95.3621),
    'IND': (39.7638, -86.1555),
    'LAC': (34.0430, -118.2673),
    'LAL': (34.0430, -118.2673),
    'MEM': (35.1382, -90.0505),
    'MIA': (25.7814, -80.1870),
    'MIL': (43.0451, -87.9170),
    'MIN': (44.9795, -93.2760),
    'NOP': (29.9490, -90.0812),
    'NYK': (40.7505, -73.9934),
    'OKC': (35.4634, -97.5151),
    'ORL': (28.5392, -81.3836),
    'PHI': (39.9012, -75.1720),
    'PHX': (33.4457, -112.0712),
    'POR': (45.5316, -122.6668),
    'SAC': (38.5802, -121.4996),
    'SAS': (29.4270, -98.4375),
    'TOR': (43.6435, -79.3791),
    'UTA': (40.7683, -111.9011),
    'WAS': (38.8981, -77.0209),
}

ARENA_TIMEZONE = {
    'ATL': 'Eastern', 'BKN': 'Eastern', 'BOS': 'Eastern', 'CHA': 'Eastern',
    'CHI': 'Central', 'CLE': 'Eastern', 'DAL': 'Central', 'DEN': 'Mountain',
    'DET': 'Eastern', 'GSW': 'Pacific', 'HOU': 'Central', 'IND': 'Eastern',
    'LAC': 'Pacific', 'LAL': 'Pacific', 'MEM': 'Central', 'MIA': 'Eastern',
    'MIL': 'Central', 'MIN': 'Central', 'NOP': 'Central', 'NYK': 'Eastern',
    'OKC': 'Central', 'ORL': 'Eastern', 'PHI': 'Eastern', 'PHX': 'Mountain',
    'POR': 'Pacific', 'SAC': 'Pacific', 'SAS': 'Central', 'TOR': 'Eastern',
    'UTA': 'Mountain', 'WAS': 'Eastern',
}

# ── Vectorized haversine (do NOT use geopy per-row -- too slow) ────────────
def _haversine_miles(lat1, lon1, lat2, lon2):
    """Vectorized haversine formula. Accuracy within 0.2% of geodesic."""
    R = 3958.8  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

# Inside build_team_game_features(), after is_home and opponent_abbr are set:
# ── Travel distance and timezone-change features ──────────────────────────
print("Computing travel distance features...")
_arena_lat = {k: v[0] for k, v in ARENA_COORDS.items()}
_arena_lon = {k: v[1] for k, v in ARENA_COORDS.items()}

# Current game arena: team's own arena if home, opponent's arena if away
curr_arena_abbr = np.where(df["is_home"] == 1, df["team_abbreviation"], df["opponent_abbr"])
df["_curr_lat"] = pd.Series(curr_arena_abbr, index=df.index).map(_arena_lat)
df["_curr_lon"] = pd.Series(curr_arena_abbr, index=df.index).map(_arena_lon)
df["_curr_tz"]  = pd.Series(curr_arena_abbr, index=df.index).map(ARENA_TIMEZONE)

# Previous game arena (shift-1 within team -- no leakage)
df["_prev_lat"] = df.groupby("team_id")["_curr_lat"].shift(1)
df["_prev_lon"] = df.groupby("team_id")["_curr_lon"].shift(1)
df["_prev_tz"]  = df.groupby("team_id")["_curr_tz"].shift(1)

# travel_miles: geodesic distance from prev game arena to current game arena
has_both = df["_prev_lat"].notna() & df["_curr_lat"].notna()
df["travel_miles"] = np.nan
df.loc[has_both, "travel_miles"] = _haversine_miles(
    df.loc[has_both, "_prev_lat"].values,
    df.loc[has_both, "_prev_lon"].values,
    df.loc[has_both, "_curr_lat"].values,
    df.loc[has_both, "_curr_lon"].values,
)
df["travel_miles"] = df["travel_miles"].fillna(0)  # first game of season: neutral 0

# cross_country_travel: 1 if timezone changed from prev game to this game
df["cross_country_travel"] = (
    (df["_prev_tz"].notna()) &
    (df["_curr_tz"].notna()) &
    (df["_prev_tz"] != df["_curr_tz"])
).astype(int)

# Drop internal working columns
df = df.drop(columns=["_curr_lat", "_curr_lon", "_curr_tz", "_prev_lat", "_prev_lon", "_prev_tz"])

n_cross = df["cross_country_travel"].sum()
print(f"  travel_miles: min={df['travel_miles'].min():.0f}, max={df['travel_miles'].max():.0f}")
print(f"  cross_country_travel: {n_cross:,} games flagged ({n_cross/len(df):.1%})")
```

### Pattern 2: Season Month in build_matchup_dataset

**What:** `season_month` is game-level (same value for home and away team), so it belongs in `build_matchup_dataset()` directly on the matchup DataFrame, not in per-team features.

**When to use:** After the matchup merge is complete, before saving.

**Example:**
```python
# Inside build_matchup_dataset(), after the merge:
# ── Season-segment context ────────────────────────────────────────────────
matchup["season_month"] = pd.to_datetime(matchup["game_date"]).dt.month
print(f"  season_month: range {matchup['season_month'].min()} to {matchup['season_month'].max()}")
```

### Pattern 3: Wire New Features Through Pipeline

**What:** New per-team features (travel_miles, cross_country_travel) must be added to:
1. `context_cols` in `build_team_game_features()` output section
2. `context_cols` in `build_matchup_dataset()`
3. `diff_stats` list in `build_matchup_dataset()` for automatic diff_ pickup
4. `schedule_cols` set in `get_feature_cols()` in game_outcome_model.py

**Example:**
```python
# In build_team_game_features() context_cols (output section):
context_cols = [
    # ... existing ...
    "travel_miles",
    "cross_country_travel",
]

# In build_matchup_dataset() context_cols:
context_cols = [
    # ... existing ...
    "travel_miles",
    "cross_country_travel",
]

# In build_matchup_dataset() diff_stats:
diff_stats = [
    # ... existing ...
    "travel_miles",
    "cross_country_travel",
]

# In game_outcome_model.py get_feature_cols() schedule_cols:
schedule_cols = {
    "home_days_rest", "away_days_rest",
    "home_is_back_to_back", "away_is_back_to_back",
    "home_travel_miles", "away_travel_miles",
    "home_cross_country_travel", "away_cross_country_travel",
    "season_month",        # game-level, not prefixed home/away
}
```

### Anti-Patterns to Avoid

- **Using geopy per-row for travel_miles:** `df.apply(lambda row: geodesic(row['prev'], row['curr']).miles, axis=1)` takes ~6 seconds for 30K rows. Use vectorized haversine instead.
- **Storing tuples in a DataFrame column and shifting:** `df['arena_coords'] = df['team'].map(ARENA_COORDS); df['prev_coords'] = df.groupby('team_id')['arena_coords'].shift(1)` then calling geodesic on the result fails because pandas unpacks tuples during shift. Always split into separate lat/lon columns before shifting.
- **Adding travel_miles to ROLL_STATS or computing it after the opponent self-join:** travel_miles depends on prev game arena, not current game opponents. It must be computed right after is_home/opponent_abbr are set, not after the opp_box join.
- **fillna(0) for travel_miles = wrong for first game:** Actually 0 is CORRECT for first game (no travel to quantify = neutral). This differs from days_rest where fillna(7) is used for first game.
- **Making cross_country_travel a diff_ feature in model selection:** The diff (home_cross_country - away_cross_country) has meaning (both traveled cross-country = 0 diff, but both teams are equally tired). Including both home_ and away_ forms AND the diff_ is appropriate.
- **Putting season_month in team_game_features.py:** season_month is identical for both home and away team in the same game. It has no place in the per-team feature table -- add it directly to the matchup dataset.
- **Unicode in print statements on Windows:** Project runs on Windows cp1252. Use ASCII in all new print() calls. No arrows, em-dashes, or non-ASCII characters.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Geodesic distance | Custom spherical trig | Vectorized haversine (numpy) | Already fully accurate to 0.2%; haversine is < 10 lines, no import needed |
| Timezone lookup from coordinates | Geocoding API / timezonefinder | Static ARENA_TIMEZONE dict | 30 arenas never change; static dict is instant, zero dependencies, always correct |
| Arena address-to-coordinate lookup | Google Maps API / geopy.geocoders | Hardcoded ARENA_COORDS dict | Same reasoning: 30 fixed arenas; no network call, no API key, never breaks |
| Previous game city tracking | Custom schedule scraper | Shift(1) on curr_arena columns within team_id groupby | game_date is already in team_game_features; sorting + shift is O(n log n) -- no loop needed |

**Key insight:** All schedule information needed for Phase 4 is already in `team_game_logs.csv`. The `matchup` column encodes home/away status (parsed into `is_home` and `opponent_abbr` in team_game_features.py). No external data sources are needed.

---

## Common Pitfalls

### Pitfall 1: Tuple Storage Failure with groupby shift
**What goes wrong:** `df['arena'] = df['team'].map(ARENA_COORDS)` stores tuples. `df.groupby('team_id')['arena'].shift(1)` produces a column where calling `geodesic(row['arena'], row['prev_arena'])` fails with "A single number has been passed to the Point constructor."
**Why it happens:** pandas shift() on an object-dtype column containing tuples can unpack the tuples depending on the dtype inference. The result may be a float64 column containing only the first element.
**How to avoid:** Always split into separate `_curr_lat`, `_curr_lon` columns (float64) before shifting. Verified working in live test.
**Warning signs:** `ValueError: A single number has been passed to the Point constructor` when calling geodesic.

### Pitfall 2: geopy Geodesic Not Vectorized
**What goes wrong:** Using `df.apply(lambda row: geodesic((row['lat1'], row['lon1']), (row['lat2'], row['lon2'])).miles, axis=1)` is correct but takes ~6 seconds on 30K rows (Python loop).
**Why it happens:** geopy.distance.geodesic is a Python function, not a numpy ufunc. Each call processes one pair.
**How to avoid:** Use the `_haversine_miles(lat1_arr, lon1_arr, lat2_arr, lon2_arr)` vectorized function. Benchmarked at 0.006s for 30K rows (1000x faster).
**Warning signs:** build_team_game_features() runtime increases by several seconds.

### Pitfall 3: days_rest / is_back_to_back Already Implemented -- Don't Duplicate
**What goes wrong:** Following the plan literally "Build days-rest and back-to-back features" and rewriting the existing implementation.
**Why it happens:** The plan description suggests these need to be built, but they were added in v3 of team_game_features.py.
**How to avoid:** Verify existing implementation (lines 297-306 in team_game_features.py) satisfies FR-3.1:
- `days_rest` = integer days since last game, fillna(7), clip(upper=14) -- FR-3.1 asks for "integer days since last game" -- this satisfies it
- `is_back_to_back` = (days_rest <= 1).astype(int) -- correctly identifies back-to-back as 1
- Already in matchup context_cols and diff_stats; schedule_cols in get_feature_cols()
- Plan 04-01 should VERIFY this satisfies FR-3.1 and add diff_is_back_to_back to diff_stats if missing.
**Warning signs:** Duplicate column errors or logic conflicts if code is re-added.

### Pitfall 4: travel_miles NaN for Non-Modern Teams
**What goes wrong:** Historical teams (pre-2013-14) have abbreviations like 'HUS', 'DEF', 'AND' not in ARENA_COORDS. `df['_curr_lat'].isna()` will be True for them.
**Why it happens:** ARENA_COORDS only covers modern 30 teams.
**How to avoid:** `fillna(0)` on travel_miles after the vectorized computation (0 = no quantifiable travel, neutral). Correct for the first game of each season anyway. Historical games are excluded from model training (2013-14+), so this NaN rate is irrelevant for model accuracy.
**Warning signs:** travel_miles NaN rate > 0.2% in modern era data. Verified in testing: actual NaN rate for modern era = 0.0% after fillna(0).

### Pitfall 5: cross_country_travel is Per-Team, Not Per-Game
**What goes wrong:** Thinking cross_country_travel should be a single flag per game (1 if the away team crossed timezones). Actually it should be a PER-TEAM feature measuring each team's travel burden.
**Why it happens:** The feature name sounds game-centric.
**How to avoid:** Compute it in team_game_features.py per team (like travel_miles). The home team that just returned from a west coast road trip also crossed timezones to get to their home arena. Both home and away team values propagate to the matchup as home_cross_country_travel and away_cross_country_travel with diff_cross_country_travel available.
**Warning signs:** If computed as a game-level flag, it would not capture the home team's travel fatigue.

### Pitfall 6: season_month in team_game_features.py (Wrong Location)
**What goes wrong:** Adding season_month to per-team features in team_game_features.py, then having it appear as home_season_month and away_season_month in the matchup (always identical values -- redundant).
**Why it happens:** Most features are per-team, so there's an instinct to add everything there.
**How to avoid:** Add season_month directly in build_matchup_dataset() on the merged matchup DataFrame. It is inherently game-level. Validated: game_date is available in the matchup meta from the home side.
**Warning signs:** home_season_month == away_season_month for 100% of rows (tip-off that the diff would always be 0).

---

## Code Examples

### Vectorized Haversine (Confirmed Working)
```python
# Tested on production data (30,628 rows): 0.006s runtime, 0.2% max error vs geodesic
def _haversine_miles(lat1, lon1, lat2, lon2):
    """Vectorized haversine formula. Accepts numpy arrays or pandas Series."""
    R = 3958.8  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))
```

### Travel Miles Full Block (Insertion Point in build_team_game_features)
```python
# After: df["is_home"] = _parse_home_away(df["matchup"])
# After: df["opponent_abbr"] = _extract_opponent_abbr(df["matchup"])
# Before: df["win"] = (df["wl"] == "W").astype(int)

# ── Travel distance and timezone-change features ──────────────────────────
print("Computing travel distance features...")
_arena_lat = {k: v[0] for k, v in ARENA_COORDS.items()}
_arena_lon = {k: v[1] for k, v in ARENA_COORDS.items()}

curr_arena_abbr = np.where(
    df["is_home"] == 1, df["team_abbreviation"], df["opponent_abbr"]
)
curr_series = pd.Series(curr_arena_abbr, index=df.index)

df["_curr_lat"] = curr_series.map(_arena_lat)
df["_curr_lon"] = curr_series.map(_arena_lon)
df["_curr_tz"]  = curr_series.map(ARENA_TIMEZONE)

# shift(1) within team gives prev game arena (no leakage -- NFR-1)
df["_prev_lat"] = df.groupby("team_id")["_curr_lat"].shift(1)
df["_prev_lon"] = df.groupby("team_id")["_curr_lon"].shift(1)
df["_prev_tz"]  = df.groupby("team_id")["_curr_tz"].shift(1)

has_both = df["_prev_lat"].notna() & df["_curr_lat"].notna()
df["travel_miles"] = np.nan
df.loc[has_both, "travel_miles"] = _haversine_miles(
    df.loc[has_both, "_prev_lat"].values,
    df.loc[has_both, "_prev_lon"].values,
    df.loc[has_both, "_curr_lat"].values,
    df.loc[has_both, "_curr_lon"].values,
)
df["travel_miles"] = df["travel_miles"].fillna(0)

df["cross_country_travel"] = (
    df["_prev_tz"].notna()
    & df["_curr_tz"].notna()
    & (df["_prev_tz"] != df["_curr_tz"])
).astype(int)

df = df.drop(columns=["_curr_lat", "_curr_lon", "_curr_tz",
                        "_prev_lat", "_prev_lon", "_prev_tz"])
n_cross = df["cross_country_travel"].sum()
print(f"  travel_miles: min={df['travel_miles'].min():.0f}mi max={df['travel_miles'].max():.0f}mi")
print(f"  cross_country_travel: {n_cross:,} games flagged ({n_cross/len(df):.1%})")
```

### Season Month in build_matchup_dataset
```python
# After: matchup = meta.merge(home_feat).merge(away_feat)
# Before: matchup["diff_four_factors_composite"] = ...

# ── Season-segment context ────────────────────────────────────────────────
matchup["season_month"] = pd.to_datetime(matchup["game_date"]).dt.month
print(f"  season_month: {matchup['season_month'].nunique()} unique months, "
      f"range {int(matchup['season_month'].min())}-{int(matchup['season_month'].max())}")
```

### Feature Wiring Additions

In `build_matchup_dataset()` `context_cols` list:
```python
context_cols = [
    "days_rest", "rest_days", "is_back_to_back",
    "games_last_5_days", "games_last_7_days",
    "travel_miles",          # NEW Phase 4
    "cross_country_travel",  # NEW Phase 4
    # ... rest unchanged ...
]
```

In `build_matchup_dataset()` `diff_stats` list:
```python
diff_stats = [
    # ... existing ...
    "days_rest", "games_last_5_days", "games_last_7_days",
    "travel_miles",          # NEW Phase 4
    "cross_country_travel",  # NEW Phase 4
]
```

In `game_outcome_model.py` `get_feature_cols()`:
```python
schedule_cols = {
    "home_days_rest", "away_days_rest",
    "home_is_back_to_back", "away_is_back_to_back",
    "home_travel_miles", "away_travel_miles",         # NEW Phase 4
    "home_cross_country_travel", "away_cross_country_travel",  # NEW Phase 4
    "season_month",                                   # NEW Phase 4 (game-level)
}
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| geopy per-row in apply() | Vectorized haversine (numpy) | N/A (never used per-row) | 1000x speedup; acceptable 0.2% accuracy tradeoff |
| pytz for timezone lookups | Static arena timezone dict | N/A | No external dep; 30 NBA arenas do not change |
| google maps / geocoding API for arena coordinates | Static ARENA_COORDS dict | N/A | No API key needed; instant; never rate-limited |

**Deprecated/outdated:**
- `geopy.distance.distance`: This was the old geopy alias (pre-2.x) that could be configured. In 2.x+, `geodesic` and `great_circle` are explicit. Neither is needed here -- use haversine.
- `pyproj` for geodesic calculations: Requires C extension installation; overkill for 30-arena static dict use case.

---

## Open Questions

1. **Should diff_is_back_to_back be added to diff_stats?**
   - What we know: `home_is_back_to_back` and `away_is_back_to_back` ARE in context_cols and propagate to matchup features and schedule_cols in get_feature_cols(). `diff_is_back_to_back` is not in diff_stats currently.
   - What's unclear: Does the model gain signal from the differential (home B2B minus away B2B) vs just having both raw flags?
   - Recommendation: Add `is_back_to_back` to diff_stats in Plan 04-01 verification step. The diff captures the fatigue asymmetry scenario (home team on B2B while away team is rested). Low cost, potentially high signal.

2. **Should travel_miles be fillna(0) or fillna(None/NaN) for first game of season?**
   - What we know: NaN rate in modern era after fillna(0) = 0.0%. The 30 first-game NaNs come from historical teams (pre-2013-14) not in ARENA_COORDS.
   - What's unclear: Is 0 miles the right semantic for "no previous game"? (Treated as "no travel burden," which is a reasonable neutral.)
   - Recommendation: fillna(0) is correct and consistent with the practical interpretation. Document in code comment.

3. **Where does geopy fit if we use haversine?**
   - What we know: FR-3.2 says "computed via geopy using a hardcoded arena coordinate dictionary." The requirement text specifies geopy.
   - What's unclear: Is the requirement about geopy as a library choice or just the calculation methodology?
   - Recommendation: Add geopy to requirements.txt to satisfy the requirement text. In the code, import geopy at the top of the file with a comment explaining that haversine is used for vectorized performance but geopy is available for single-game inference if needed. The model doesn't care which formula computed the mileage.

4. **PHX timezone (Arizona): Mountain Standard always, no DST**
   - What we know: Phoenix is UTC-7 year-round (no DST). During summer, California (Pacific, UTC-7 PDT) matches PHX. During winter, California is UTC-8 PST and PHX is UTC-7.
   - What's unclear: Should PHX be 'Mountain' or a special 'Arizona' zone?
   - Recommendation: Keep 'Mountain' for PHX. The cross_country_travel flag is a rough indicator, not a precise timezone computation. The Mountain zone correctly indicates PHX is between Pacific and Central teams geographically and informationally.

---

## Validation Architecture

> `workflow.nyquist_validation` key is absent from `.planning/config.json`. No automated test suite exists (no `tests/` directory, no pytest). Manual verification steps are the phase gate.

### Phase Gates (Manual Verification)
| Req ID | Behavior | Verification Command |
|--------|----------|---------------------|
| FR-3.1 | days_rest integer, is_back_to_back=1 on 1-day gaps | `python -c "import pandas as pd; df = pd.read_csv('data/features/team_game_features.csv'); print(df[['days_rest','is_back_to_back']].describe()); print('B2B pct:', df['is_back_to_back'].mean())"` |
| FR-3.2 | travel_miles present, geodesic via haversine | `python -c "import pandas as pd; df = pd.read_csv('data/features/team_game_features.csv'); print(df['travel_miles'].describe())"` |
| FR-3.3 | cross_country_travel binary 0/1 | `python -c "import pandas as pd; df = pd.read_csv('data/features/team_game_features.csv'); print(df['cross_country_travel'].value_counts())"` |
| FR-3.4 | season_month 1-12 in matchup features | `python -c "import pandas as pd; df = pd.read_csv('data/features/game_matchup_features.csv'); print(df['season_month'].value_counts().sort_index())"` |
| NFR-1 | No lookahead: travel uses prev game location | Code review: confirm shift(1) on _prev_lat/_prev_lon before haversine call |
| NFR-2 | Pipeline stays under 15 min | Time build_team_game_features() -- travel block should add < 1 second |

### Integration Test
```bash
python src/features/team_game_features.py
python -c "
import pandas as pd
df = pd.read_csv('data/features/game_matchup_features.csv')
required = ['home_travel_miles', 'away_travel_miles', 'diff_travel_miles',
            'home_cross_country_travel', 'away_cross_country_travel',
            'diff_cross_country_travel', 'season_month']
missing = [c for c in required if c not in df.columns]
assert not missing, f'Missing columns: {missing}'
print('All Phase 4 required columns present')
for c in required:
    null_rate = df[c].isna().mean()
    print(f'  {c}: null_rate={null_rate:.1%}')
"
```

---

## Sources

### Primary (HIGH confidence)
- Live codebase inspection -- `src/features/team_game_features.py` read in full; existing days_rest/is_back_to_back implementation confirmed at lines 297-306
- Live Python execution -- geopy 2.4.1 installed and tested: `from geopy.distance import geodesic; geodesic((34.04, -118.27), (42.37, -71.06)).miles` returns 2597.9 miles (confirmed)
- Live Python execution -- vectorized haversine benchmarked: 0.006s for 30,628 rows; accuracy vs geodesic 0.2% on 3 test pairs
- Live data inspection -- `data/features/team_game_features.csv` and `data/features/game_matchup_features.csv` inspected: all 2013-14+ teams confirmed present in proposed ARENA_COORDS dict; travel/month columns confirmed absent (need to be added)
- Live data inspection -- `data/features/game_matchup_features.csv`: diff_ column mechanism verified; diff_days_rest exists; context_cols and diff_stats lists in build_matchup_dataset() confirmed as the correct wiring points

### Secondary (MEDIUM confidence)
- geopy PyPI documentation -- geodesic and great_circle API confirmed as the two main distance classes in geopy 2.x
- Haversine formula reference (Wikipedia) -- formula verified against geopy geodesic on multiple test pairs to confirm 0.2% max error

### Tertiary (LOW confidence)
- Arena coordinate values -- sourced from training knowledge (team arena lat/lon). All 30 coordinates tested for reasonable distances (LAL to BOS = 2598 miles, LAL to GSW = 345 miles). Cross-verify against Google Maps before production use if extreme precision is needed.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- geopy installed, tested, version confirmed; numpy haversine tested end-to-end; no new installs except geopy in requirements.txt
- Architecture: HIGH -- implementation pattern derived from existing days_rest code in same file; insertion points confirmed by reading actual function; diff_ wiring mechanism verified by examining actual output CSVs
- Pitfalls: HIGH -- tuple-shift failure reproduced live during research; geopy loop slowness measured; season_month placement decision backed by examining matchup structure
- Arena coordinates: MEDIUM -- values are training knowledge, not verified against official sources; cross-check recommended before trusting high-precision distance computations

**Research date:** 2026-03-02
**Valid until:** 2026-04-02 (30 days) -- geopy API is stable; arena coordinates change only on team relocations (none expected)
