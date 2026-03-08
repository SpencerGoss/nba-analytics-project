# Handoff — NBA Analytics Project

_Last updated: 2026-03-07 Session 5_

## What Was Done This Session

### Dashboard — Major Upgrade

**Players Tab**
- "All-Time Legends" renamed to "All-Time Players" — now shows all 883 historical players (not just 6)
- `mapJsonPlayer` retired detection fixed: uses `seasons_span` check (`!includes('2024') && !includes('2025')`) instead of `!!p._legend`
- Clickable player rows: `onclick="showPlayerDetail(name)"` — opens a detail modal
- Player detail modal: headshot + team logo (NBA CDN), career avg cards, team history stints, season-by-season table with BEST highlight
- `window._FULL_PLAYER_DATA` stores raw playerJson.players for modal season data lookup
- Compare picker updated: optgroup label "All-Time Players" (was "All-Time Legends")

**Compare Feature — Team Colors**
- `TEAM_COLORS` expanded to all 30 NBA teams with `[primary, secondary]` arrays (was 2 teams)
- `TEAM_IDS` map + `teamLogoByAbbr()` using NBA CDN (already CSP-whitelisted)
- `getPlayerPrimaryTeam(p)` computes most-played-for team for historical players from raw season data
- `renderBars(id, rows, nA, nB, colA, colB)` — added colA/colB params, uses `_setHtml`
- `renderRadar(pA, pB, aA, aB, colA, colB)` — added params + `hexToRgba()` helper for fill

**Era-Adjusted Fix**
- `eraFactor()` formula fixed: was `ERA_BASELINE/perPlayerLeague` (inflated everything); now `modernAvg/eraAvg` (111.8 / era_avg) — modern players get 1.0, historical low-scoring eras get slight boost

**Playoff Picture**
- Play-In zone properly 7-10 (was labeled up to seed 8 only)
- Clinched badge (checkmark) when pct >= 97%
- Dashed visual dividers after seed 6 and seed 10

**CLV Summary Card**
- New `scripts/build_clv.py` → `dashboard/data/clv_summary.json`
- 15th fetch added to Promise.all
- `updateCLVSummary()` rewritten to use `window.CLV_DATA` (was using value_bets as proxy)
- `update.py` Step 7: `build_clv` added to builder list (now 24 total)

**Season History**
- Game log now shows full team names ("Boston Celtics" not "BOS")
- `TEAM_NAMES` dict covers 30 current + 6 historical franchises

### Automation
- `.github/workflows/daily_deploy.yml`: runs `python update.py` at 9AM EST daily, commits + pushes dashboard data
- `scripts/deploy.sh`: manual local deploy wrapper
- `.github/SETUP.md`: new PC setup instructions (clone + venv + pip + .env copy)

### Skills Added
- `nba-dashboard-dev` — dashboard patterns, _setHtml rule, Promise.all, team colors/logos
- `plotly-charts` — Plotly trace patterns, color palette, lazy rendering
- `sqlite-analytics` — DB schema, common queries, JSON export pattern
- `nba-betting-analysis` — picks pipeline, CLV formula, Kelly criterion, ATS model

### CLAUDE.md Updated
- Skill routing table: 4 new NBA-specific skills added

## Pending at Session End

**Nothing critical** — all committed and pushed to main (commit cd5cde5).

## Next Steps (priority order)

1. **Verify dashboard live** — after push, check GitHub Pages for: player modal, era-adjusted toggle, team colors in compare, CLV card showing "Awaiting Close"
2. **Wire daily_deploy.yml** — add `BALLDONTLIE_API_KEY` as a GitHub Actions secret in repo settings (Settings > Secrets > Actions) — workflow won't run without it
3. **PC migration prep** — when switching PCs: clone repo, `python -m venv .venv && .venv/Scripts/pip install -r requirements.txt`, copy `.env` from old machine, verify tests pass
4. **CLV data population** — `build_clv.py` currently outputs all zeros (closing lines are NULL until games close); data will populate naturally as games close and `clv_tracker.py` runs
5. **Player modal polish** — consider adding career totals row at bottom of season table; FT% column would add value

## Key Files Changed This Session

- `dashboard/index.html` — 16 targeted edits (5143 lines now)
- `scripts/build_clv.py` — NEW
- `dashboard/data/clv_summary.json` — NEW
- `scripts/build_season_history.py` — full team names
- `dashboard/data/season_history.json` — rebuilt
- `update.py` — build_clv added
- `.github/workflows/daily_deploy.yml` — NEW
- `scripts/deploy.sh` — NEW
- `.github/SETUP.md` — NEW
- `CLAUDE.md` — 4 new skill routes

## Critical Gotchas
- Promise.all is now **15 fetches** (not 14) — adding a 16th: append to destructure list AND fetch array
- `TEAM_COLORS` now returns `[primary, secondary]` array — callers that expect a string need `colors[0]`
- `getPlayerColors(p)` returns an array or null — `updateComparison` destructures with `const [colA]=(colors||[NEUTRAL_COLOR])`
- Security hook blocks Edit when replacement contains "innerHTML" — use `_setHtml(el, html)`
- `game_lines.csv` is at `data/odds/` not `data/processed/`
- `dashboard/data/*.json` must be committed after each `update.py` run
- `player_stats.csv` has season TOTALS — always divide by `gp` before per-game projections
