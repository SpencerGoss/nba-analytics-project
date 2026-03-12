# Player Comparison Tool Overhaul — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the existing player comparison UI with a Trading Card design featuring headshots, tug-of-war stat bars, corrected era adjustment using per-season league averages, radar/trend charts, similar players, and share/export.

**Architecture:** The overhaul is entirely client-side (dashboard/index.html) with two minor build script fixes. The comparison section HTML (lines 1351-1379) and JS functions (lines 4980-5385) are replaced wholesale. Era adjustment switches from hardcoded decade buckets to per-season ratios using `league_by_season` data already in the JSON. Charts use Plotly.js (already loaded).

**Tech Stack:** Vanilla JS, Plotly.js 2.27.0 (already in project), CSS custom properties (aurora theme), NBA CDN for headshots/logos.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `scripts/build_player_comparison.py` | Modify (lines 444, 572-587, 762-768) | Fix `best_season` type, inject legends before index |
| `tests/test_build_player_comparison.py` | Modify | Add tests for `best_season` normalization, legend index inclusion |
| `dashboard/index.html` | Modify (lines 1351-1379 HTML, 4980-5385 JS) | Full comparison UI replacement |
| `dashboard/sw.js` | Modify (line 2) | Add NBA CDN headshot caching |

---

## Chunk 1: Build Script Fixes

### Task 1: Normalize `best_season` field type

The `best_season` field is a string for regular players but a dict for legends. Normalize to always be a string.

**Files:**
- Modify: `scripts/build_player_comparison.py:492-568` (legend dicts)
- Test: `tests/test_build_player_comparison.py`

- [ ] **Step 1: Write failing test for best_season normalization**

```python
# tests/test_build_player_comparison.py — add to existing test file
def test_legend_best_season_is_string():
    """All legend entries must have best_season as a string, not a dict."""
    from scripts.build_player_comparison import _LEGENDS
    for leg in _LEGENDS:
        assert isinstance(leg.get("best_season", ""), str), (
            f"{leg['player_name']} best_season is {type(leg['best_season'])}, expected str"
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_build_player_comparison.py::test_legend_best_season_is_string -v`
Expected: FAIL — MJ's best_season is a dict `{"season_str": "1986-87", ...}`

- [ ] **Step 3: Fix legend best_season fields in build script**

In `scripts/build_player_comparison.py`, change each legend's `best_season` from a dict to a string:

```python
# Line 499: Change from:
"best_season": {"season_str": "1986-87", "pts": 37.1, "reb": 5.2, "ast": 4.6},
# To:
"best_season": "1986-87",

# Line 520: Change from:
"best_season": {"season_str": "1987-88", "pts": 29.9, "reb": 9.3, "ast": 6.1},
# To:
"best_season": "1987-88",

# Line 531: Change from:
"best_season": {"season_str": "1988-89", "pts": 22.5, "reb": 7.9, "ast": 12.8},
# To:
"best_season": "1988-89",

# Line 542: Change from:
"best_season": {"season_str": "1971-72", "pts": 34.8, "reb": 16.6, "ast": 4.6},
# To:
"best_season": "1971-72",

# Line 553: Change from:
"best_season": {"season_str": "2005-06", "pts": 35.4, "reb": 5.3, "ast": 4.5},
# To:
"best_season": "2005-06",

# Line 564: Change from:
"best_season": {"season_str": "1961-62", "pts": 50.4, "reb": 25.7, "ast": 2.4},
# To:
"best_season": "1961-62",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_build_player_comparison.py::test_legend_best_season_is_string -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/build_player_comparison.py tests/test_build_player_comparison.py
git commit -m "fix: normalize legend best_season to string type"
```

---

### Task 2: Inject legends before building player_index.json

Currently legends are injected at line 768 AFTER the index is built at line 762. This means MJ, Wilt, etc. are missing from autocomplete.

**Files:**
- Modify: `scripts/build_player_comparison.py:760-770`
- Test: `tests/test_build_player_comparison.py`

- [ ] **Step 1: Write failing test**

```python
def test_legends_in_player_index():
    """Legends must appear in player_index.json for autocomplete."""
    import json
    from pathlib import Path
    # Run a minimal build if needed, or check existing output
    index_path = Path(__file__).resolve().parent.parent / "dashboard" / "data" / "player_index.json"
    if not index_path.exists():
        pytest.skip("player_index.json not built yet")
    with open(index_path) as f:
        index = json.load(f)
    names = {p["name"] for p in index}
    for legend_name in ["Michael Jordan", "Wilt Chamberlain", "Larry Bird",
                        "Magic Johnson", "Kareem Abdul-Jabbar", "Kobe Bryant"]:
        assert legend_name in names, f"{legend_name} missing from player_index.json"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_build_player_comparison.py::test_legends_in_player_index -v`
Expected: FAIL — legends not in index

- [ ] **Step 3: Reorder legend injection before index build**

In `scripts/build_player_comparison.py`, swap lines 762-768 so legends inject first:

```python
    # Was:
    # player_records = build_player_records(...)   # line 760
    # league_rows = build_league_by_season(league)  # line 761
    # player_index = build_player_index(player_records)  # line 762
    # ...
    # player_records = _inject_legends(player_records)  # line 768

    # Now:
    player_records = build_player_records(enriched, min_seasons, min_career_games, pos_lookup)
    league_rows = build_league_by_season(league)

    print(f"  {len(player_records):,} eligible players")

    # Inject curated legend overrides BEFORE building index so legends appear in autocomplete
    player_records = _inject_legends(player_records)
    print(f"  {len(player_records):,} players after legend injection")

    player_index = build_player_index(player_records)
```

- [ ] **Step 4: Rebuild JSONs and run test**

Run: `.venv/Scripts/python.exe scripts/build_player_comparison.py && .venv/Scripts/python.exe -m pytest tests/test_build_player_comparison.py::test_legends_in_player_index -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/build_player_comparison.py tests/test_build_player_comparison.py dashboard/data/player_comparison.json dashboard/data/player_index.json
git commit -m "fix: inject legends before index build so they appear in autocomplete"
```

---

## Chunk 2: mapJsonPlayer Data Gap Fixes

### Task 3: Add missing fields to mapJsonPlayer + legend career_avgs

The mapped player objects are missing `stl`, `blk`, `fgm`, `fga`, `ft`, `best_season` — needed by trading cards, radar, tug-of-war bars, and similar players. Legends also lack `stl`/`blk` in career_avgs for the same reason.

**Files:**
- Modify: `dashboard/index.html:2625-2678` (mapJsonPlayer + _gpWeightedStats)

- [ ] **Step 1: Add missing stat fields to _gpWeightedStats**

In `_gpWeightedStats` (line 2660-2678), add stl, blk, fgm, fga, ft tracking:

```javascript
function _gpWeightedStats(seasons){
  let ptsS=0,rebS=0,astS=0,stlS=0,blkS=0,fgmS=0,fgaS=0,ftS=0,fgS=0,thrS=0,tsS=0,gpT=0,fgGp=0,thrGp=0,tsGp=0;
  seasons.forEach(s=>{
    const gp=s.gp||0; if(gp<=0)return;
    ptsS+=(s.pts||0)*gp; rebS+=(s.reb||0)*gp; astS+=(s.ast||0)*gp;
    stlS+=(s.stl||0)*gp; blkS+=(s.blk||0)*gp;
    fgmS+=(s.fgm||0)*gp; fgaS+=(s.fga||0)*gp;
    gpT+=gp;
    if(s.fg_pct!=null){fgS+=s.fg_pct*gp;fgGp+=gp;}
    if(s.fg3_pct!=null){thrS+=s.fg3_pct*gp;thrGp+=gp;}
    if(s.ft_pct!=null){ftS+=s.ft_pct*gp;}
    if(s.ts_pct!=null){tsS+=s.ts_pct*gp;tsGp+=gp;}
  });
  return {
    pts: gpT>0?Math.round((ptsS/gpT)*10)/10:0,
    reb: gpT>0?Math.round((rebS/gpT)*10)/10:0,
    ast: gpT>0?Math.round((astS/gpT)*10)/10:0,
    stl: gpT>0?Math.round((stlS/gpT)*10)/10:0,
    blk: gpT>0?Math.round((blkS/gpT)*10)/10:0,
    fgm: gpT>0?Math.round((fgmS/gpT)*10)/10:0,
    fga: gpT>0?Math.round((fgaS/gpT)*10)/10:0,
    fg:  fgGp>0?Math.round((fgS/fgGp)*1000)/10:0,
    thr: thrGp>0?Math.round((thrS/thrGp)*1000)/10:0,
    ft:  fgGp>0?Math.round((ftS/fgGp)*1000)/10:0,
    ts:  tsGp>0?Math.round((tsS/tsGp)*1000)/10:0,
    gp:  gpT,
  };
}
```

- [ ] **Step 2: Add missing fields to mapJsonPlayer return**

In `mapJsonPlayer` (line 2636-2658), add the new fields:

```javascript
  return {
    name:     p.player_name,
    team:     team||'N/A',
    pts:      cStats.pts,
    reb:      cStats.reb,
    ast:      cStats.ast,
    stl:      cStats.stl,
    blk:      cStats.blk,
    fgm:      cStats.fgm,
    fga:      cStats.fga,
    fg:       cStats.fg,
    thr:      cStats.thr,
    ft:       cStats.ft,
    ts:       cStats.ts,
    position: p.position||'',
    positions: p.positions||'',
    position_primary: p.position_primary||'',
    jersey: p.jersey_number||'',
    ini:      playerIni(p.player_name),
    nbaId:    p.player_id>0?p.player_id:null,
    debutYear:debutYear,
    primaryTeam:team||'N/A',
    allTeams: allTeams,
    retired:  !spanStr.includes('2024')&&!spanStr.includes('2025'),
    seasons_span: spanStr,
    best_season: p.best_season||'',
    _seasons: p.seasons||[],
  };
```

- [ ] **Step 3: Handle legends with career_avgs but no seasons**

Legends have `career_avgs` with pts/reb/ast but often no `stl`/`blk` and empty `seasons` array. The `_gpWeightedStats` returns 0 for all stats when seasons is empty. Fix by falling back to `career_avgs`:

In `mapJsonPlayer`, after `const cStats=_gpWeightedStats(p.seasons||[]);` add:

```javascript
  // For legends with empty seasons, fall back to career_avgs
  if(cStats.gp===0&&ca.pts){
    cStats.pts=ca.pts||0; cStats.reb=ca.reb||0; cStats.ast=ca.ast||0;
    cStats.stl=ca.stl||0; cStats.blk=ca.blk||0;
  }
```

- [ ] **Step 4: Commit**

```bash
git add dashboard/index.html
git commit -m "feat: add stl/blk/fgm/fga/ft/best_season to player mapping"
```

---

## Chunk 3: Era Adjustment Rewrite (JS)

### Task 4: Replace hardcoded era adjustment with per-season league_by_season data

**Files:**
- Modify: `dashboard/index.html:5007-5036` (era JS functions)

- [ ] **Step 1: Read current era code to confirm line numbers**

Read `dashboard/index.html` lines 5007-5036 to confirm the exact code block.

- [ ] **Step 2: Replace era adjustment functions**

Replace lines 5007-5036 (from `ERA_LEAGUE_AVG` through `applyEra`) with:

```javascript
// Era adjustment using per-season league averages from player_comparison.json
// league_by_season is loaded into window._LEAGUE_BY_SEASON at comparison JSON load time
let _eraModernRef=null; // cached modern (latest) season averages

function _getLeagueSeason(seasonStr){
  const lbs=window._LEAGUE_BY_SEASON;
  if(!lbs||!lbs.length)return null;
  return lbs.find(s=>s.season===seasonStr)||null;
}

function _getModernRef(){
  if(_eraModernRef)return _eraModernRef;
  const lbs=window._LEAGUE_BY_SEASON;
  if(!lbs||!lbs.length)return null;
  _eraModernRef=lbs[lbs.length-1];
  return _eraModernRef;
}

function eraAdjustSeason(rawVal,stat,seasonStr){
  // Returns era-adjusted value for a single season, or raw if no data
  if(!eraAdjusted)return rawVal;
  if(rawVal==null||isNaN(rawVal))return rawVal;
  const modern=_getModernRef();
  if(!modern)return rawVal;
  const leagueSeason=_getLeagueSeason(seasonStr);
  if(!leagueSeason)return rawVal; // pre-1969 — no data, show raw
  const modernAvg=modern['avg_'+stat];
  const seasonAvg=leagueSeason['avg_'+stat];
  if(!modernAvg||!seasonAvg||seasonAvg===0)return rawVal;
  return Math.round(rawVal*(modernAvg/seasonAvg)*10)/10;
}

function eraAdjustCareer(player,stat){
  // GP-weighted era-adjusted career average
  if(!eraAdjusted)return player[stat]||0;
  const seasons=player._seasons||[];
  if(!seasons.length)return player[stat]||0; // legends with no season data
  let totalWeighted=0,totalGp=0;
  for(const s of seasons){
    const gp=s.gp||0;
    if(gp===0)continue;
    const raw=s[stat]||0;
    const adj=eraAdjustSeason(raw,stat,s.season);
    totalWeighted+=adj*gp;
    totalGp+=gp;
  }
  return totalGp>0?Math.round(totalWeighted/totalGp*10)/10:(player[stat]||0);
}

// Legacy compatibility — applyEra still works for simple cases
function applyEra(val,debutYear,stat,seasonStr){
  if(!eraAdjusted)return val;
  if(seasonStr)return eraAdjustSeason(val,stat||'pts',seasonStr);
  return val; // no season context — return raw (caller should use eraAdjustCareer instead)
}
```

- [ ] **Step 3: Store league_by_season on JSON load**

In the `_loadPlayerComparison()` function (around line 4082-4113), after parsing the JSON, store the league data:

```javascript
// After: const playerJson = await resp.json();
// Add:
if(playerJson.league_by_season){
  window._LEAGUE_BY_SEASON=playerJson.league_by_season;
  _eraModernRef=null; // reset cache
}
```

- [ ] **Step 4: Update era footnote text**

Replace the era footnote (line 1363):

```html
<p class="era-footnote" id="era-footnote">Era-adjusted stats use per-season league averages to normalize counting stats (PTS, REB, AST, STL, BLK) across eras. Ratios (FG%, TS%) are not adjusted. Pre-1969 seasons show raw values (no league data available).</p>
```

- [ ] **Step 5: Verify in browser**

Open `http://localhost:8080`, go to Players > Compare, select LeBron vs MJ, toggle era adjustment on/off. Verify:
- Stats change when toggled
- FG%/3P%/TS% do NOT change
- No JS console errors

- [ ] **Step 6: Commit**

```bash
git add dashboard/index.html
git commit -m "feat: rewrite era adjustment to use per-season league averages"
```

---

## Chunk 3: Trading Card HTML + CSS

### Task 4: Replace comparison HTML structure

Replace the current comparison section (lines 1351-1379) with the new Trading Card layout.

**Files:**
- Modify: `dashboard/index.html:1351-1379` (HTML)

- [ ] **Step 1: Replace comparison HTML**

Replace lines 1351-1379 with the new Trading Card structure:

```html
<div class="tab-content" id="players-tab-compare">
  <!-- Search + Controls Bar -->
  <div class="cmp-controls">
    <div class="g2" style="margin-bottom:12px">
      <div class="card cmp-picker-card"><div class="sl" style="margin-bottom:7px">Player 1</div><input id="picker-a" list="picker-a-list" type="search" placeholder="Search player..." oninput="_debouncePicker()" autocomplete="off" class="cmp-search-input"/><datalist id="picker-a-list"></datalist></div>
      <div class="card cmp-picker-card"><div class="sl" style="margin-bottom:7px">Player 2</div><input id="picker-b" list="picker-b-list" type="search" placeholder="Search player..." oninput="_debouncePicker()" autocomplete="off" class="cmp-search-input"/><datalist id="picker-b-list"></datalist></div>
    </div>
    <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;margin-bottom:14px">
      <div class="cmp-presets">
        <button class="cmp-preset-btn" onclick="setCmpPreset('Michael Jordan','LeBron James')">MJ vs LeBron</button>
        <button class="cmp-preset-btn" onclick="setCmpPreset('Larry Bird','Magic Johnson')">Bird vs Magic</button>
        <button class="cmp-preset-btn" onclick="setCmpPreset('Stephen Curry','Kevin Durant')">Curry vs KD</button>
        <button class="cmp-preset-btn" onclick="swapPickers()">Swap</button>
      </div>
      <div class="era-pill">
        <button id="era-btn-raw" class="active" onclick="setEraMode(false)">Raw</button>
        <button id="era-btn-adj" onclick="setEraMode(true)">Era-Adj</button>
      </div>
    </div>
    <p class="era-footnote" id="era-footnote">Era-adjusted stats use per-season league averages to normalize counting stats (PTS, REB, AST, STL, BLK) across eras. Ratios (FG%, TS%) are not adjusted. Pre-1969 seasons show raw values.</p>
  </div>

  <!-- Empty state -->
  <div id="cmp-prompt" class="card" style="text-align:center;padding:48px;color:var(--t2)"><div style="font-size:13px;font-weight:600;color:var(--t1);margin-bottom:4px">Select two players to compare</div><div style="font-size:12px">Trading cards, stat bars, and charts will appear here</div></div>

  <!-- Comparison content (hidden until both players selected) -->
  <div id="cmp-content" style="display:none">
    <!-- Section 2: Trading Cards -->
    <div class="g2 cmp-cards-row" style="margin-bottom:18px">
      <div id="card-a" class="cmp-trading-card"></div>
      <div id="card-b" class="cmp-trading-card"></div>
    </div>

    <!-- Section 3: Tabbed Stat Comparison -->
    <div class="card" style="margin-bottom:16px;padding:16px 20px">
      <div style="display:flex;gap:6px;margin-bottom:14px;flex-wrap:wrap">
        <button class="tab active" id="cmp-tab-pergame" onclick="setCmpStatTab('pergame')">Per Game</button>
        <button class="tab" id="cmp-tab-efficiency" onclick="setCmpStatTab('efficiency')">Efficiency</button>
        <button class="tab" id="cmp-tab-totals" onclick="setCmpStatTab('totals')">Career Totals</button>
      </div>
      <div id="cmp-stat-bars"></div>
    </div>

    <!-- Section 4: Visualizations -->
    <div class="g2" style="margin-bottom:16px">
      <div class="card"><div class="sh"><div><div class="st">Skill Radar</div><div class="ss">6-axis comparison</div></div></div><div id="radar" style="height:310px"></div></div>
      <div class="card"><div class="sh"><div><div class="st">Career Scoring Trend</div><div class="ss">PPG by season</div></div></div><div id="cmp-career-chart" style="height:280px"></div></div>
    </div>

    <!-- Section 5: Head-to-Head (deferred) -->
    <div id="cmp-h2h" class="card" style="margin-bottom:16px;padding:18px 22px"></div>

    <!-- Section 6: Season-by-Season -->
    <div id="cmp-seasons" style="margin-bottom:16px"></div>

    <!-- Section 7: Similar + Export -->
    <div style="display:flex;gap:12px;align-items:flex-start;flex-wrap:wrap;margin-bottom:16px">
      <div id="cmp-similar" style="flex:1;min-width:280px"></div>
      <div id="cmp-export" style="flex-shrink:0"></div>
    </div>
  </div>
</div>
```

- [ ] **Step 2: Add Trading Card CSS**

Add the following CSS block inside the existing `<style>` section (near other comparison styles):

```css
/* ── Trading Card Comparison ── */
.cmp-picker-card{padding:14px 18px}
.cmp-search-input{width:100%;padding:9px 12px;border-radius:10px;border:1px solid var(--b1);background:var(--bg3);color:var(--t0);font-family:var(--f);font-size:13px;outline:none;transition:border-color 140ms}
.cmp-search-input:focus{border-color:var(--green)}
.cmp-presets{display:flex;gap:6px;flex-wrap:wrap}
.cmp-preset-btn{padding:5px 12px;border-radius:8px;border:1px solid var(--b1);background:transparent;color:var(--t2);font-family:var(--f);font-size:11px;font-weight:600;cursor:pointer;transition:all 140ms}
.cmp-preset-btn:hover{border-color:var(--green);color:var(--t0)}
.cmp-trading-card{border-radius:16px;overflow:hidden;position:relative;min-height:320px;display:flex;flex-direction:column}
.cmp-card-header{padding:20px 22px 14px;position:relative;display:flex;align-items:flex-end;gap:16px;min-height:140px}
.cmp-card-headshot{width:100px;height:76px;object-fit:cover;object-position:top;border-radius:10px;background:rgba(255,255,255,.08);flex-shrink:0}
.cmp-card-info{flex:1}
.cmp-card-name{font-size:18px;font-weight:800;color:#fff;text-shadow:0 1px 4px rgba(0,0,0,.5);line-height:1.15}
.cmp-card-meta{font-size:11px;color:rgba(255,255,255,.8);margin-top:3px;font-weight:500}
.cmp-card-body{padding:16px 22px;background:var(--card);flex:1}
.cmp-card-bigstats{display:flex;gap:18px;margin-bottom:12px}
.cmp-card-bigstat{text-align:center}
.cmp-card-bigstat-val{font-size:22px;font-weight:800;line-height:1}
.cmp-card-bigstat-lbl{font-size:10px;font-weight:600;color:var(--t2);text-transform:uppercase;letter-spacing:.4px;margin-top:2px}
.cmp-card-badges{display:flex;gap:5px;flex-wrap:wrap;margin-top:10px}
.cmp-badge{padding:3px 8px;border-radius:6px;font-size:10px;font-weight:700;letter-spacing:.3px;text-transform:uppercase}
.cmp-card-bestseason{font-size:11px;color:var(--t2);margin-top:8px}

/* Tug-of-war bars */
.tow-row{display:flex;align-items:center;gap:8px;margin-bottom:8px}
.tow-label{width:56px;font-size:11px;font-weight:600;color:var(--t2);text-align:center;flex-shrink:0}
.tow-val{width:48px;font-size:12px;font-weight:700;flex-shrink:0}
.tow-val-l{text-align:right}
.tow-val-r{text-align:left}
.tow-bar-wrap{flex:1;height:18px;display:flex;border-radius:4px;overflow:hidden;background:var(--bg3)}
.tow-bar-l{height:100%;border-radius:4px 0 0 4px;transition:width 600ms cubic-bezier(.25,.8,.25,1)}
.tow-bar-r{height:100%;border-radius:0 4px 4px 0;transition:width 600ms cubic-bezier(.25,.8,.25,1)}
.tow-bar-l.winner{opacity:1}.tow-bar-r.winner{opacity:1}
.tow-bar-l:not(.winner){opacity:.35}.tow-bar-r:not(.winner){opacity:.35}

/* Season accordion */
.cmp-season-toggle{width:100%;padding:12px 18px;border:none;background:var(--card);color:var(--t1);font-family:var(--f);font-size:13px;font-weight:700;cursor:pointer;text-align:left;display:flex;justify-content:space-between;align-items:center;border-radius:12px;margin-bottom:6px}
.cmp-season-toggle:hover{background:var(--bg3)}
.cmp-season-body{display:none;padding:0 4px 12px}
.cmp-season-body.open{display:block}
.cmp-season-tbl{width:100%;font-size:11px;border-collapse:collapse}
.cmp-season-tbl th{padding:6px 8px;text-align:right;color:var(--t2);font-weight:600;border-bottom:1px solid var(--b1)}
.cmp-season-tbl th:first-child,.cmp-season-tbl td:first-child{text-align:left}
.cmp-season-tbl td{padding:5px 8px;text-align:right;color:var(--t1)}

@media(max-width:640px){
  .cmp-cards-row{flex-direction:column}
  .cmp-trading-card{min-height:auto}
  .cmp-card-header{min-height:100px;padding:14px 16px 10px}
  .cmp-card-headshot{width:72px;height:55px}
  .cmp-card-name{font-size:15px}
  .cmp-card-bigstat-val{font-size:18px}
}
```

- [ ] **Step 3: Commit HTML/CSS structure**

```bash
git add dashboard/index.html
git commit -m "feat: trading card HTML structure and CSS for comparison overhaul"
```

---

## Chunk 4: Trading Card Rendering + Tug-of-War Bars

### Task 5: Implement Trading Card render function

**Files:**
- Modify: `dashboard/index.html` (JS section, near line 5092)

- [ ] **Step 1: Add preset and swap helper functions**

Add before the `updateComparison` function:

```javascript
function setCmpPreset(nameA,nameB){
  const pA=document.getElementById('picker-a');
  const pB=document.getElementById('picker-b');
  if(pA)pA.value=nameA;
  if(pB)pB.value=nameB;
  updateComparison();
}
function swapPickers(){
  const pA=document.getElementById('picker-a');
  const pB=document.getElementById('picker-b');
  if(!pA||!pB)return;
  const tmp=pA.value;pA.value=pB.value;pB.value=tmp;
  updateComparison();
}
```

- [ ] **Step 2: Implement renderTradingCard function**

```javascript
function renderTradingCard(elId,p,col,colB){
  const el=document.getElementById(elId);
  if(!el||!p)return;
  const colors=TEAM_COLORS[getPlayerPrimaryTeam(p)]||[col,colB||col];
  const nbaId=p.nbaId||0;
  // Headshot: skip for negative IDs (legends)
  const imgUrl=nbaId>0?'https://cdn.nba.com/headshots/nba/latest/1040x760/'+nbaId+'.png':'';
  const imgHtml=imgUrl?'<img src="'+imgUrl+'" class="cmp-card-headshot" onerror="this.style.display=\'none\'" alt="'+esc(p.name)+'"/>':'<div class="cmp-card-headshot" style="display:flex;align-items:center;justify-content:center;font-size:28px;color:rgba(255,255,255,.4)">'+esc(p.ini||'?')+'</div>';

  // Career stats (era-adjusted if toggle on)
  const pts=eraAdjustCareer(p,'pts');
  const reb=eraAdjustCareer(p,'reb');
  const ast=eraAdjustCareer(p,'ast');

  // Milestone badges
  const badges=[];
  const careerGp=p._seasons?p._seasons.reduce((s,r)=>(s+(r.gp||0)),0):0;
  const careerPts=p._seasons?p._seasons.reduce((s,r)=>(s+(r.pts||0)*(r.gp||0)),0):0;
  if(careerPts>=30000)badges.push({t:'30K Club',c:'#FFD700'});
  else if(careerPts>=25000)badges.push({t:'25K Club',c:'#C0C0C0'});
  else if(careerPts>=20000)badges.push({t:'20K Club',c:'#CD7F32'});
  if(careerGp>=1000)badges.push({t:'1K Games',c:'#4ECDC4'});
  if((p.ts||0)>=0.6)badges.push({t:'60% TS+',c:'#FF6B6B'});

  // Best season
  const bestSeason=p.best_season||'';
  const bestRow=p._seasons?p._seasons.find(s=>s.season===bestSeason):null;
  const bestStr=bestRow?bestSeason+': '+(bestRow.pts||'?')+' PPG':'';

  const badgeHtml=badges.map(b=>'<span class="cmp-badge" style="background:'+b.c+'22;color:'+b.c+';border:1px solid '+b.c+'44">'+b.t+'</span>').join('');

  const posStr=p.positions||p.position_primary||'';
  const jerseyStr=p.jersey?'#'+p.jersey:'';
  const metaParts=[posStr,jerseyStr,p.seasons_span||''].filter(Boolean);

  _setHtml(el,
    '<div class="cmp-card-header" style="background:linear-gradient(135deg,'+colors[0]+','+colors[1]+')">'+
    imgHtml+
    '<div class="cmp-card-info">'+
    '<div class="cmp-card-name">'+esc(p.name)+'</div>'+
    '<div class="cmp-card-meta">'+esc(metaParts.join(' · '))+'</div>'+
    '</div></div>'+
    '<div class="cmp-card-body">'+
    '<div class="cmp-card-bigstats">'+
    '<div class="cmp-card-bigstat"><div class="cmp-card-bigstat-val" style="color:'+colors[0]+'">'+pts+'</div><div class="cmp-card-bigstat-lbl">PTS</div></div>'+
    '<div class="cmp-card-bigstat"><div class="cmp-card-bigstat-val" style="color:'+colors[0]+'">'+reb+'</div><div class="cmp-card-bigstat-lbl">REB</div></div>'+
    '<div class="cmp-card-bigstat"><div class="cmp-card-bigstat-val" style="color:'+colors[0]+'">'+ast+'</div><div class="cmp-card-bigstat-lbl">AST</div></div>'+
    '</div>'+
    (badgeHtml?'<div class="cmp-card-badges">'+badgeHtml+'</div>':'')+
    (bestStr?'<div class="cmp-card-bestseason">Best: '+esc(bestStr)+'</div>':'')+
    '</div>'
  );
}
```

- [ ] **Step 3: Implement tug-of-war bar renderer**

```javascript
let _cmpStatTab='pergame';
function setCmpStatTab(tab){
  _cmpStatTab=tab;
  ['pergame','efficiency','totals'].forEach(t=>{
    const btn=document.getElementById('cmp-tab-'+t);
    if(btn)btn.classList.toggle('active',t===tab);
  });
  _renderCmpStatBars();
}

function _renderCmpStatBars(){
  const el=document.getElementById('cmp-stat-bars');
  if(!el)return;
  const nA=document.getElementById('picker-a').value;
  const nB=document.getElementById('picker-b').value;
  const pA=nA?DATA.players.find(p=>p.name===nA):null;
  const pB=nB?DATA.players.find(p=>p.name===nB):null;
  if(!pA||!pB){_setHtml(el,'');return;}
  const colA=(getPlayerColors(pA)||[NEUTRAL_COLOR])[0];
  const colB=(getPlayerColors(pB)||[NEUTRAL_COLOR_B])[0];

  let rows=[];
  if(_cmpStatTab==='pergame'){
    rows=[
      ['PTS',eraAdjustCareer(pA,'pts'),eraAdjustCareer(pB,'pts')],
      ['REB',eraAdjustCareer(pA,'reb'),eraAdjustCareer(pB,'reb')],
      ['AST',eraAdjustCareer(pA,'ast'),eraAdjustCareer(pB,'ast')],
      ['STL',eraAdjustCareer(pA,'stl'),eraAdjustCareer(pB,'stl')],
      ['BLK',eraAdjustCareer(pA,'blk'),eraAdjustCareer(pB,'blk')],
      ['FGM',pA.fgm||0,pB.fgm||0],
      ['FGA',pA.fga||0,pB.fga||0],
    ];
  }else if(_cmpStatTab==='efficiency'){
    rows=[
      ['TS%',pA.ts||0,pB.ts||0,'%'],
      ['FG%',pA.fg||0,pB.fg||0,'%'],
      ['3P%',pA.thr||0,pB.thr||0,'%'],
      ['FT%',pA.ft||0,pB.ft||0,'%'],
    ];
  }else{
    // Career totals: sum per-game * GP across seasons
    function careerTotal(p,stat){
      const ss=p._seasons||[];
      return Math.round(ss.reduce((s,r)=>{
        const v=eraAdjusted?eraAdjustSeason(r[stat]||0,stat,r.season):(r[stat]||0);
        return s+v*(r.gp||0);
      },0));
    }
    rows=[
      ['PTS',careerTotal(pA,'pts'),careerTotal(pB,'pts')],
      ['REB',careerTotal(pA,'reb'),careerTotal(pB,'reb')],
      ['AST',careerTotal(pA,'ast'),careerTotal(pB,'ast')],
      ['STL',careerTotal(pA,'stl'),careerTotal(pB,'stl')],
      ['BLK',careerTotal(pA,'blk'),careerTotal(pB,'blk')],
      ['GP',pA._seasons?pA._seasons.reduce((s,r)=>s+(r.gp||0),0):0,
           pB._seasons?pB._seasons.reduce((s,r)=>s+(r.gp||0),0):0],
      ['MIN',Math.round(pA._seasons?pA._seasons.reduce((s,r)=>s+(r.min||0)*(r.gp||0),0):0),
             Math.round(pB._seasons?pB._seasons.reduce((s,r)=>s+(r.min||0)*(r.gp||0),0):0)],
    ];
  }

  const html=rows.map(r=>{
    const label=r[0],vA=r[1]||0,vB=r[2]||0,suffix=r[3]||'';
    const total=vA+vB||1;
    const pctA=Math.round(vA/total*100);
    const pctB=100-pctA;
    const winA=vA>vB,winB=vB>vA;
    return '<div class="tow-row">'+
      '<div class="tow-val tow-val-l"'+(winA?' style="color:'+colA+'"':'')+'>'+(suffix==='%'?vA.toFixed(1)+suffix:vA.toLocaleString())+'</div>'+
      '<div class="tow-bar-wrap">'+
      '<div class="tow-bar-l'+(winA?' winner':'')+'" style="width:'+pctA+'%;background:'+colA+'"></div>'+
      '<div class="tow-bar-r'+(winB?' winner':'')+'" style="width:'+pctB+'%;background:'+colB+'"></div>'+
      '</div>'+
      '<div class="tow-val tow-val-r"'+(winB?' style="color:'+colB+'"':'')+'>'+(suffix==='%'?vB.toFixed(1)+suffix:vB.toLocaleString())+'</div>'+
      '<div class="tow-label">'+label+'</div>'+
      '</div>';
  }).join('');
  _setHtml(el,html);
}
```

- [ ] **Step 4: Commit**

```bash
git add dashboard/index.html
git commit -m "feat: trading card rendering + tug-of-war stat bars"
```

---

## Chunk 5: Update main updateComparison() orchestrator

### Task 6: Rewrite updateComparison to use new components

**Files:**
- Modify: `dashboard/index.html:5092-5163`

- [ ] **Step 1: Replace updateComparison function**

Replace the existing `updateComparison` function with:

```javascript
function updateComparison(){
  const nA=document.getElementById('picker-a').value,nB=document.getElementById('picker-b').value;
  const pA=nA?DATA.players.find(p=>p.name===nA):null;
  const pB=nB?DATA.players.find(p=>p.name===nB):null;
  if(!nA||!nB||nA===nB||!pA||!pB){
    document.getElementById('cmp-prompt').style.display='block';
    document.getElementById('cmp-content').style.display='none';return;
  }
  document.getElementById('cmp-prompt').style.display='none';
  document.getElementById('cmp-content').style.display='block';
  const colA=(getPlayerColors(pA)||[NEUTRAL_COLOR])[0];
  const colB=(getPlayerColors(pB)||[NEUTRAL_COLOR_B])[0];

  // Section 2: Trading Cards
  renderTradingCard('card-a',pA,colA);
  renderTradingCard('card-b',pB,colB);

  // Section 3: Tug-of-war stat bars
  _renderCmpStatBars();

  // Section 4: Radar + Career Trend
  renderRadar(pA,pB,null,null,colA,colB);
  renderCareerTrend(nA,nB,colA,colB);

  // Section 5: H2H placeholder
  _renderH2HPlaceholder(pA,pB);

  // Section 6: Season-by-season accordion
  _renderSeasonAccordion(pA,pB,colA,colB);

  // Section 7: Similar + Export
  renderCmpSimilar(pA,pB,null,null,colA,colB);
  _renderExportBtn(pA,pB);
}
```

- [ ] **Step 2: Implement H2H placeholder**

```javascript
function _renderH2HPlaceholder(pA,pB){
  const el=document.getElementById('cmp-h2h');
  if(!el)return;
  // Check for overlapping seasons
  const seasonsA=new Set((pA._seasons||[]).map(s=>s.season));
  const seasonsB=new Set((pB._seasons||[]).map(s=>s.season));
  const overlap=[...seasonsA].filter(s=>seasonsB.has(s));
  const msg=overlap.length>0?
    'These players overlapped for '+overlap.length+' season'+(overlap.length>1?'s':'')+'. Head-to-head data coming in a future update.':
    'These players never overlapped in active seasons.';
  _setHtml(el,'<div style="text-align:center;padding:14px;color:var(--t2);font-size:12px"><span style="font-weight:700;margin-right:6px">Head-to-Head</span>'+esc(msg)+'</div>');
}
```

- [ ] **Step 3: Implement season-by-season accordion**

```javascript
function _toggleSeasonAccordion(id){
  const body=document.getElementById(id);
  if(body)body.classList.toggle('open');
}

function _renderSeasonAccordion(pA,pB,colA,colB){
  const el=document.getElementById('cmp-seasons');
  if(!el)return;
  function makeTable(p,color,idx){
    const seasons=p._seasons||[];
    if(!seasons.length)return '<div class="card" style="margin-bottom:6px"><button class="cmp-season-toggle" onclick="_toggleSeasonAccordion(\'sa-'+idx+'\')"><span style="color:'+color+'">'+esc(p.name)+' Seasons</span><span style="font-size:11px;color:var(--t2)">Career averages only</span></button></div>';
    const hdrs=['Season','Team','GP','MIN','PTS','REB','AST','STL','BLK','FG%','3P%','FT%'];
    const thHtml=hdrs.map(h=>'<th>'+h+'</th>').join('');
    const trHtml=seasons.map(s=>{
      const adjPts=eraAdjustSeason(s.pts||0,'pts',s.season);
      const adjReb=eraAdjustSeason(s.reb||0,'reb',s.season);
      const adjAst=eraAdjustSeason(s.ast||0,'ast',s.season);
      const adjStl=eraAdjustSeason(s.stl||0,'stl',s.season);
      const adjBlk=eraAdjustSeason(s.blk||0,'blk',s.season);
      return '<tr>'+
        '<td style="text-align:left;font-weight:600">'+(s.season||'')+'</td>'+
        '<td style="text-align:left">'+(s.team||'')+'</td>'+
        '<td>'+(s.gp||'')+'</td>'+
        '<td>'+(s.min||'')+'</td>'+
        '<td style="font-weight:700;color:'+color+'">'+(eraAdjusted?adjPts:(s.pts||''))+'</td>'+
        '<td>'+(eraAdjusted?adjReb:(s.reb||''))+'</td>'+
        '<td>'+(eraAdjusted?adjAst:(s.ast||''))+'</td>'+
        '<td>'+(eraAdjusted?adjStl:(s.stl||''))+'</td><td>'+(eraAdjusted?adjBlk:(s.blk||''))+'</td>'+
        '<td>'+((s.fg_pct||0)*100).toFixed(1)+'</td>'+
        '<td>'+((s.fg3_pct||0)*100).toFixed(1)+'</td>'+
        '<td>'+((s.ft_pct||0)*100).toFixed(1)+'</td></tr>';
    }).join('');
    return '<div class="card" style="margin-bottom:6px"><button class="cmp-season-toggle" onclick="_toggleSeasonAccordion(\'sa-'+idx+'\')"><span style="color:'+color+'">'+esc(p.name)+' ('+seasons.length+' seasons)</span><span style="font-size:18px">&#x25BE;</span></button>'+
      '<div id="sa-'+idx+'" class="cmp-season-body"><div style="overflow-x:auto"><table class="cmp-season-tbl"><thead><tr>'+thHtml+'</tr></thead><tbody>'+trHtml+'</tbody></table></div></div></div>';
  }
  _setHtml(el,makeTable(pA,colA,0)+makeTable(pB,colB,1));
}
```

- [ ] **Step 4: Implement export button**

```javascript
function _renderExportBtn(pA,pB){
  const el=document.getElementById('cmp-export');
  if(!el)return;
  _setHtml(el,'<div style="display:flex;gap:8px"><button onclick="_exportCmpCSV()" class="cmp-preset-btn">Export CSV</button><button onclick="_shareCmpURL()" class="cmp-preset-btn">Share Link</button></div>');
}

function _shareCmpURL(){
  const pA=document.getElementById('picker-a').value;
  const pB=document.getElementById('picker-b').value;
  if(!pA||!pB)return;
  const url=location.origin+location.pathname+'?compare='+encodeURIComponent(pA)+'&vs='+encodeURIComponent(pB);
  navigator.clipboard.writeText(url).then(()=>alert('Link copied!')).catch(()=>{});
}

function _exportCmpCSV(){
  const pA=DATA.players.find(p=>p.name===document.getElementById('picker-a').value);
  const pB=DATA.players.find(p=>p.name===document.getElementById('picker-b').value);
  if(!pA||!pB)return;
  const header='Player,PTS,REB,AST,STL,BLK,FG%,TS%,GP';
  const rowA=[pA.name,pA.pts,pA.reb,pA.ast,pA.stl||'',pA.blk||'',pA.fg,pA.ts||'',pA._seasons?pA._seasons.reduce((s,r)=>s+(r.gp||0),0):''].join(',');
  const rowB=[pB.name,pB.pts,pB.reb,pB.ast,pB.stl||'',pB.blk||'',pB.fg,pB.ts||'',pB._seasons?pB._seasons.reduce((s,r)=>s+(r.gp||0),0):''].join(',');
  const csv=header+'\n'+rowA+'\n'+rowB;
  const blob=new Blob([csv],{type:'text/csv'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob);
  a.download='comparison_'+pA.name.replace(/\s/g,'_')+'_vs_'+pB.name.replace(/\s/g,'_')+'.csv';
  a.click();
}
```

- [ ] **Step 5: Verify in browser**

Open dashboard, go to Players > Compare, select two players. Verify:
- Trading cards render with headshots, team gradients, big stats, badges
- Tug-of-war bars show for all 3 tabs
- Season accordion expands/collapses
- H2H placeholder shows overlap count
- Share/Export buttons work
- Era toggle updates all sections

- [ ] **Step 6: Commit**

```bash
git add dashboard/index.html
git commit -m "feat: complete comparison overhaul — orchestrator, accordion, export"
```

---

## Chunk 6: Radar Chart Update + Similar Players Enhancement

### Task 7: Update radar chart to use new 6-axis design

**Files:**
- Modify: `dashboard/index.html:5263-5290` (renderRadar)

- [ ] **Step 1: Update renderRadar for the new layout**

The existing radar uses Plotly scatterpolar with PER/BPM/TS which depends on LEGENDS_ADV. Update to use the 6 basic stats (PTS, REB, AST, STL, BLK, TS%) which are always available:

```javascript
function renderRadar(pA,pB,aA,aB,colA,colB){
  colA=colA||NEUTRAL_COLOR;colB=colB||NEUTRAL_COLOR_B;
  const cats=['PTS','REB','AST','STL','BLK','TS%'];
  // Normalize each stat to 0-100 scale across all players
  const allP=DATA.players;
  function pct(val,stat){
    const vals=allP.map(p=>stat==='ts'?(p.ts||0):(p[stat]||0));
    const mx=Math.max(...vals)||1;
    return Math.round((val/mx)*100);
  }
  const vA=[pct(eraAdjustCareer(pA,'pts'),'pts'),pct(eraAdjustCareer(pA,'reb'),'reb'),
            pct(eraAdjustCareer(pA,'ast'),'ast'),pct(pA.stl||0,'stl'),pct(pA.blk||0,'blk'),pct(pA.ts||0,'ts')];
  const vB=[pct(eraAdjustCareer(pB,'pts'),'pts'),pct(eraAdjustCareer(pB,'reb'),'reb'),
            pct(eraAdjustCareer(pB,'ast'),'ast'),pct(pB.stl||0,'stl'),pct(pB.blk||0,'blk'),pct(pB.ts||0,'ts')];
  _plotlyNewPlot('radar',[
    {type:'scatterpolar',r:vA.concat([vA[0]]),theta:cats.concat([cats[0]]),name:pA.name,
     fill:'toself',fillcolor:colA+'22',line:{color:colA,width:2},marker:{size:4}},
    {type:'scatterpolar',r:vB.concat([vB[0]]),theta:cats.concat([cats[0]]),name:pB.name,
     fill:'toself',fillcolor:colB+'22',line:{color:colB,width:2},marker:{size:4}},
  ],{polar:{radialaxis:{visible:true,range:[0,100],showticklabels:false,gridcolor:'rgba(255,255,255,.06)'},
            angularaxis:{gridcolor:'rgba(255,255,255,.06)',tickfont:{color:'var(--t2)',size:11}}},
     paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
     showlegend:true,legend:{font:{color:'#999',size:11}},margin:{t:30,b:30,l:50,r:50}},
  {responsive:true,displayModeBar:false});
}
```

- [ ] **Step 2: Update similar players to use 6 dimensions with z-score**

Replace `renderCmpSimilar` (lines 5201-5245) — update the distance function to use z-score normalization over 6 stats:

```javascript
function renderCmpSimilar(pA,pB,aA,aB,colA,colB){
  const el=document.getElementById('cmp-similar');
  if(!el||!DATA||!DATA.players||DATA.players.length<5)return;
  const stats=['pts','reb','ast','stl','blk','ts'];
  // Compute mean and stddev for each stat
  const means={},stds={};
  for(const s of stats){
    const vals=DATA.players.map(p=>p[s]||0);
    const mean=vals.reduce((a,b)=>a+b,0)/vals.length;
    const std=Math.sqrt(vals.reduce((a,v)=>a+Math.pow(v-mean,2),0)/vals.length)||1;
    means[s]=mean;stds[s]=std;
  }
  function getSimilar(target,excludeNames){
    return DATA.players
      .filter(p=>!excludeNames.includes(p.name))
      .map(p=>{
        let dist=0;
        for(const s of stats){
          const zA=((target[s]||0)-means[s])/stds[s];
          const zP=((p[s]||0)-means[s])/stds[s];
          dist+=Math.pow(zA-zP,2);
        }
        return {p,dist:Math.sqrt(dist)};
      })
      .sort((a,b)=>a.dist-b.dist)
      .slice(0,3)
      .map(x=>x.p);
  }
  const simA=getSimilar(pA,[pA.name,pB.name]);
  const simB=getSimilar(pB,[pA.name,pB.name]);
  // Render pills (reuse existing pill rendering logic from before)
  function pillsFor(players,labelName,col){
    if(!players.length)return '';
    const pills=players.map(q=>{
      const qc=getPlayerColors(q);
      const qCol=qc?qc[0]:'var(--t1)';
      const safeQ=esc(q.name);
      return '<button onclick="setCmpPlayer(&quot;'+safeQ+'&quot;)" style="display:inline-flex;align-items:center;gap:7px;padding:6px 11px;border-radius:10px;border:1px solid '+qCol+'44;background:'+qCol+'11;cursor:pointer;white-space:nowrap">'+
        headshot(q.nbaId,q.ini,24,q.team||q.primaryTeam,q.jersey)+
        '<span style="font-size:12px;font-weight:700;color:var(--t0)">'+esc(q.name)+'</span>'+
        '<span style="font-size:11px;color:var(--t2)">'+q.pts+' pts</span></button>';
    }).join('');
    return '<div style="margin-bottom:8px"><span style="font-size:11px;color:'+col+';font-weight:700">'+esc(labelName.split(' ').pop())+' comps: </span>'+pills+'</div>';
  }
  const inner=pillsFor(simA,pA.name,colA)+pillsFor(simB,pB.name,colB);
  if(!inner){_setHtml(el,'');return;}
  _setHtml(el,'<div class="card" style="padding:16px 20px"><div style="font-size:11px;font-weight:800;letter-spacing:.6px;text-transform:uppercase;color:var(--t2);margin-bottom:12px">Similar Players</div><div style="display:flex;flex-direction:column;gap:8px">'+inner+'</div></div>');
}
```

- [ ] **Step 3: Commit**

```bash
git add dashboard/index.html
git commit -m "feat: updated radar chart (6-axis) + z-score similar players"
```

---

## Chunk 7: Career Trend Update + URL Params + Cleanup

### Task 8: Update career trend chart for era adjustment

**Files:**
- Modify: `dashboard/index.html:5337-5358` (renderCareerTrend)

- [ ] **Step 1: Update career trend to support era adjustment**

```javascript
function renderCareerTrend(nA,nB,colA,colB){
  const chartEl=document.getElementById('cmp-career-chart');
  if(!chartEl)return;
  const pA=DATA.players.find(p=>p.name===nA);
  const pB=DATA.players.find(p=>p.name===nB);
  if(!pA||!pB){_setHtml(chartEl,'<div style="padding:24px;color:var(--t2);text-align:center;font-size:13px">Select two players</div>');return;}
  const seasA=pA._seasons||[];
  const seasB=pB._seasons||[];
  if(!seasA.length&&!seasB.length){
    _setHtml(chartEl,'<div style="padding:24px;color:var(--t2);text-align:center;font-size:12px">Limited historical data — career trend unavailable</div>');return;
  }
  function makeTrace(seasons,name,color){
    const x=seasons.map(s=>s.season||'');
    const y=seasons.map(s=>eraAdjusted?eraAdjustSeason(s.pts||0,'pts',s.season):(s.pts||0));
    return {x,y,name,mode:'lines+markers',line:{color,width:2.5},marker:{size:5}};
  }
  const traces=[];
  if(seasA.length)traces.push(makeTrace(seasA,pA.name,colA));
  if(seasB.length)traces.push(makeTrace(seasB,pB.name,colB));
  _plotlyNewPlot('cmp-career-chart',traces,{
    paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
    xaxis:{tickfont:{color:'#999',size:10},gridcolor:'rgba(255,255,255,.04)',tickangle:-45},
    yaxis:{title:eraAdjusted?'PPG (era-adj)':'PPG',titlefont:{color:'#999',size:11},tickfont:{color:'#999',size:10},gridcolor:'rgba(255,255,255,.06)'},
    showlegend:true,legend:{font:{color:'#999',size:11}},margin:{t:20,b:60,l:50,r:20}
  },{responsive:true,displayModeBar:false});
}
```

- [ ] **Step 2: Add URL parameter support for share links**

Add to the page initialization (near where other URL params are parsed):

Add this at the end of `_loadPlayerComparison()` (after the existing post-load logic around line 4104), just before the catch block:

```javascript
// Auto-open comparison if URL has ?compare=...&vs=...
const _cmpParams=new URLSearchParams(location.search);
const _cmpA=_cmpParams.get('compare'),_cmpB=_cmpParams.get('vs');
if(_cmpA&&_cmpB){
  switchTab('players','compare');
  initPickers();
  setTimeout(()=>{
    const elA=document.getElementById('picker-a');
    const elB=document.getElementById('picker-b');
    if(elA)elA.value=_cmpA;
    if(elB)elB.value=_cmpB;
    updateComparison();
  },100);
}
```

Also add to the page's initial load (inside the `DOMContentLoaded` handler or equivalent) a trigger to load player data if URL params are present:

```javascript
if(location.search.includes('compare='))_loadPlayerComparison();
```

- [ ] **Step 3: Remove old functions that are no longer needed**

Remove these old functions that are replaced by the new implementation:
- `renderBanner` (line 5254-5261) — replaced by `renderTradingCard`
- `renderBars` (line 5292-5315) — replaced by tug-of-war `_renderCmpStatBars`
- `renderShoot` (line 5317-5334) — removed (shooting zones required ADV data not available for most players)
- `renderEfficiencyCompare` (line 5361-5385) — folded into radar chart
- `renderCmpCopy` / `copyCmpText` (line 5166-5184) — replaced by `_exportCmpCSV` and `_shareCmpURL`
- `updatePreview` (line 5247-5252) — no longer used (trading cards replace previews)
- `LEGENDS_ADV` (line 2681-2688) — **DO NOT remove yet**; still referenced by `_loadPlayerComparison` (line ~4090) for enriching player data. Remove only after verifying no other code paths use it. If other code references LEGENDS_ADV, keep the constant but remove the functions that consumed it (renderShoot, renderEfficiencyCompare).
- Old `ERA_LEAGUE_AVG`, `ERA_BASELINE`, `getEraDecade`, `eraFactor` — replaced in Task 4

Keep: `getPlayerPrimaryTeam`, `getPlayerColors`, `TEAM_COLORS`, `setCmpPlayer`, `_debouncePicker`, `initPickers`, `setEraMode`

- [ ] **Step 4: Update service worker cache**

In `dashboard/sw.js`, add NBA CDN headshots to cache-first strategy:

```javascript
// In the fetch handler, add CDN headshots to cache-first pattern
// Match: cdn.nba.com/headshots/
```

- [ ] **Step 5: Full browser verification**

Test all scenarios from the spec testing plan:
1. Modern vs modern (LeBron vs Curry)
2. Legend vs modern (MJ vs LeBron)
3. Legend vs legend (Wilt vs Kareem)
4. Era toggle on/off
5. All 3 stat tabs
6. Season accordion expand/collapse
7. Similar players pills
8. Share link + CSV export
9. Mobile layout (resize browser)
10. Preset buttons

- [ ] **Step 6: Commit**

```bash
git add dashboard/index.html dashboard/sw.js
git commit -m "feat: career trend era support, URL params, cleanup old comparison code"
```

---

## Chunk 8: Final Integration + Run Full Test Suite

### Task 9: Run tests and rebuild

- [ ] **Step 1: Run full test suite**

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

Expected: all tests pass (baseline ~1407+)

- [ ] **Step 2: Rebuild all JSONs**

```bash
.venv/Scripts/python.exe scripts/build_player_comparison.py
```

Verify output includes legends in index, `best_season` is string type.

- [ ] **Step 3: Final commit**

```bash
git add dashboard/index.html dashboard/data/player_comparison.json dashboard/data/player_index.json scripts/build_player_comparison.py tests/test_build_player_comparison.py dashboard/sw.js
git commit -m "feat: player comparison overhaul complete — trading cards, era adjustment, tug-of-war bars"
```
