# Feature Landscape

**Domain:** NBA game outcome prediction and ATS betting analytics
**Researched:** 2026-03-01
**Confidence:** MEDIUM-HIGH (codebase analysis: HIGH; ATS betting literature: MEDIUM based on training data through August 2025)

---

## Current Feature Inventory (Already Built)

The following features already exist in `src/features/` and feed the model. This is the baseline — new work builds on top of this.

**Team rolling stats (5/10/20 game windows):**
- `pts_roll{5,10,20}`, `opp_pts_roll{5,10,20}`, `plus_minus_roll{5,10,20}`, `win_pct_roll{5,10,20}`
- `fg_pct_roll20`, `fg3_pct_roll20`, `ast_roll20`, `tov_roll20`
- `three_rate` (rolling 3PT attempt rate), `opponent_three_allowed_rate`, `three_style_mismatch`

**Schedule/fatigue:**
- `days_rest`, `rest_days`, `is_back_to_back`, `games_last_5_days`, `games_last_7_days`

**Strength of schedule:**
- `sos_roll10`, `sos_roll20` (rolling avg of opponent win% at game time)

**Home court:**
- `home_net_rating_season`, `away_net_rating_season`, `is_home`

**Injury proxy (BROKEN — all-null, never reaching model):**
- `missing_minutes`, `missing_usg_pct`, `rotation_availability`, `star_player_out`

**Psychological/motivation:**
- `revenge_game`, `blowout_loss_last_game`, `close_playoff_race`

**Era:**
- `era_label` (categorical: "3-Point Revolution 2015+", etc.)

**Trend/volatility:**
- Rolling linear regression slope on scoring, `plus_minus_volatility` (std over 10 games)
- `rebounding_edge` (own OREB vs opponent DREB)

**Matchup differentials (game_matchup_features.csv):**
- `diff_*` columns: home minus away for each rolling stat

---

## Table Stakes

Features that every serious NBA prediction model includes. Missing them causes systematic errors that compound over time.

| Feature | Why Expected | Complexity | Status |
|---------|--------------|------------|--------|
| **Offensive rating (ORtg) rolling** | Points scored per 100 possessions — normalizes for pace differences between teams; raw points is misleading when pace varies | Medium | MISSING — `off_rtg` is in `player_stats_advanced` but not in team game feature pipeline |
| **Defensive rating (DRtg) rolling** | Points allowed per 100 possessions — the strongest single-game defensive signal | Medium | MISSING — same gap as ORtg |
| **Net rating rolling** | ORtg minus DRtg — the most predictive single feature in academic NBA prediction literature; the model currently uses raw plus_minus as a proxy, which is correlated but not equivalent | Medium | PARTIAL — `plus_minus_roll{5,10,20}` is a proxy; true `net_rtg` (per-100 possessions) is not computed |
| **True shooting % (TS%) rolling** | Accounts for 3PT bonus and free throws; FG% misses half the picture in the 3-point era | Medium | MISSING — `ts_pct` is in player bio stats, not in team game feature pipeline |
| **Effective FG% (eFG%) rolling** | Adjusts for 3PT value without free throws; simplest shot quality metric | Low | MISSING |
| **Turnover rate rolling** | Turnovers per possession; raw turnover count is confounded by pace | Low | PARTIAL — `tov_roll20` is raw count, not per-possession rate |
| **Free throw rate rolling** | FTA/FGA; teams that get to the line consistently have sustainable offense | Low | MISSING |
| **Functional injury flag** | Whether a key rotation player is missing affects game outcome more than any other single feature; currently defined but silently null | High (debug existing) | BROKEN — must fix `injury_proxy.py` |
| **Official NBA injury report status** | Questionable/Probable/Out designations from official reports 1-2 hours before tip-off; far more accurate than rolling proxy | High | MISSING — no data source integrated |
| **Pace (possessions per game) rolling** | High-pace games are more random; knowing whether two fast teams are meeting allows the model to widen confidence intervals and signal total-score relevance for ATS | Medium | MISSING — `pace` column exists in `player_stats_advanced` but not in team game features |
| **Home/away performance splits** | Teams differ significantly in home vs road performance beyond simple home court dummy; some teams are dramatically stronger at home | Low | PARTIAL — `home_net_rating_season` / `away_net_rating_season` exist but may not flow through correctly to matchup features |
| **Vegas opening line (as feature)** | Market consensus encodes all public information; models that ignore Vegas systematically underfit to market efficiency | Medium | MISSING — odds exist in `data/odds/` but are not used as model input features |

---

## Differentiators

Features that create competitive advantage over basic models. Not expected by default, but meaningfully improve ATS prediction specifically.

| Feature | Value Proposition | Complexity | Status |
|---------|-------------------|------------|--------|
| **Closing line vs model delta** | The difference between the model's implied spread and the closing line is the most direct ATS signal; this is the output, not a feature, but tracking this history enables calibration | Low | MISSING — model outputs win probability; implied spread not computed |
| **Implied probability from moneyline** | Convert moneyline to no-vig probability; use as ATS feature and calibration baseline | Low | MISSING — moneyline available in `data/odds/game_lines.csv` |
| **ATS cover rate rolling (per team)** | How often has each team covered the spread in the last 10/20 games; some teams historically beat spreads due to coaching strategy or public perception bias | Medium | MISSING — requires historical spread data |
| **Public betting % (sharp/public split)** | Line movement driven by sharp money vs public money indicates market confidence; steep reverse-line movement is a strong contrarian signal | High | MISSING — requires paid data source (Action Network, Pinnacle) |
| **Referee crew foul rate** | NBA referees vary dramatically in foul calls per game; high-foul refs benefit teams that get to the line; affects total score, pace, and star player performance | Medium | MISSING — Proposal 10 in model_advisor_notes.md |
| **Travel distance and timezone change** | Cross-country back-to-backs (e.g., LAL @ BOS next night) are harder than same-timezone ones; standardized rest days miss this | Medium | MISSING — team city data is available via `get_teams.py`; distance/timezone lookup required |
| **Head-to-head ATS history** | Some team matchups have persistent ATS patterns; public tends to overvalue marquee home teams (e.g., Lakers, Celtics) in certain matchups | Medium | MISSING — requires historical spread data |
| **Season-segment / load management context** | Teams rest stars in late-regular-season meaningless games; Feb–March injury/rest patterns are different from Nov–Dec; model trained on season-wide averages misses this | Low | PARTIAL — era labels exist; month-of-season not included |
| **Player-level minutes projection** | Minutes are the primary driver of stat totals; if a player is on a minutes restriction, all raw rolling averages are inflated | Medium | MISSING — Proposal 9 in model_advisor_notes.md |
| **Four Factors differential** | Dean Oliver's Four Factors (eFG%, TOV%, ORB%, FT rate) are widely established as the most predictive team efficiency metrics; using them as a unit rather than individually often improves signal | Medium | MISSING — components are partially available from raw stats |
| **Line movement (open to close)** | The direction and magnitude of spread movement from open to close is a strong predictor; reverse-line movement especially signals sharp money | High | MISSING — requires multiple line snapshots per day |
| **Opponent pace-adjusted defensive efficiency** | When a fast-paced team faces a slow-paced team, who wins the tempo battle determines much of the outcome; this interaction is not captured by individual pace or ORtg features alone | Medium | MISSING |
| **Season debut after extended rest** | Players returning from 4+ game absence on injury report often underperform their rolling averages on first game back; systematically affects player prop and game projections | Medium | MISSING — partially captured by injury proxy but not first-game-back context |

---

## Anti-Features

Features to explicitly NOT build in the current milestone. Each risks wasted effort, data leakage, or model noise.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Play-by-play features (shot quality, LEBRON, etc.)** | Requires separate 11K-call shot chart pipeline (3-4 hours); adds enormous complexity for marginal gain over eFG% rolling averages | Use eFG% and TS% rolling from available league dash data |
| **Social media sentiment / injury news NLP** | Noisy, requires real-time scraping, frequently wrong; injury reports are the authoritative source | Use official NBA injury report status instead |
| **Player prop model features (per-game granularity beyond pts/reb/ast)** | Scope creep — player props are deferred until game outcome ATS model is proven; mixing targets muddies training | Keep player model to pts/reb/ast projections only for now |
| **Franchise-level historical ATS record (entire history)** | ATS tendencies shift with roster changes; 10-year ATS record for a team with different rosters is noise, not signal | Use rolling 20-game ATS cover rate (2-3 seasons max) |
| **Clutch stats as primary features** | Clutch data starts 2000-01 and tracking data starts 2004-05; including it truncates training data substantially; SHAP shows low importance for available clutch features | Use as supplementary feature for modern-era-only training only |
| **Deep player interaction features (lineup net ratings)** | Lineup data requires scraping Basketball Reference; very high complexity for marginal gain over injury proxy + rolling team stats | Fix injury proxy first; lineup data is a Phase 3 concern |
| **Win probability by quarter (in-game)** | Project scope is explicitly pregame only; in-game prediction is out of scope | Stay pregame only |
| **Elo ratings built from scratch** | The existing rolling net rating + win percentage is functionally an Elo proxy; a custom Elo system adds code complexity without proven gain over what's already there | Tune rolling window length instead of switching paradigms |
| **Weather / outdoor factors** | NBA is always indoors with controlled conditions | Not applicable |
| **Salary / contract year effects** | Player motivation through contract lens is noise at game level; injury proxy and usage rates capture availability better | Keep salary data out of scope |

---

## Feature Dependencies

```
Pace (possessions per game)
  → ORtg = (pts / possessions) * 100     [requires pace]
  → DRtg = (opp_pts / possessions) * 100 [requires pace]
  → Net rating = ORtg - DRtg
  → Pace-adjusted TOV rate = (tov / possessions)
  → Pace-adjusted FT rate  = (fta / fga)  [doesn't need pace, but grouped here]
  → Four Factors: eFG%, TOV%, ORB%, FT rate [all require pace-adjusted denominators]

Injury proxy fix
  → missing_minutes and missing_usg_pct correctly populated
  → star_player_out flag functional
  → Minutes projection (Proposal 9) — depends on injury proxy working

Official injury report
  → Questionable/Out status →supplements or replaces injury proxy
  → First-game-back flag (requires injury report history)

Vegas line as feature
  → Implied probability from moneyline
  → Opening spread
  → ATS cover rate (requires historical spreads — separate concern from daily odds)
  → Closing line value calculation (model spread vs closing spread)

ATS model target
  → Requires spread data in training labels (historical spreads not currently available)
  → Depends on calibrated win probability being loaded correctly (currently broken)
```

---

## MVP Recommendation

Prioritize in this order for maximum accuracy gain per unit of effort:

**Phase 1 — Fix broken foundations (highest leverage, lowest new complexity):**
1. Fix `injury_proxy.py` so `missing_minutes`, `missing_usg_pct`, `star_player_out` actually reach the model (bug, not a new feature)
2. Wire calibrated model into prediction pipeline (bug — calibrated artifact saved but never loaded)
3. Add ORtg/DRtg/net rating rolling features to team game pipeline (team_stats_advanced data already available in DB)
4. Add pace rolling average (same data source as above)
5. Add eFG% and TS% rolling to team pipeline

**Phase 2 — ATS enablers (new capability, medium complexity):**
6. Integrate Vegas moneyline as input feature (convert to implied probability)
7. Compute and store implied spread from model win probability (enables ATS comparison)
8. Add travel distance / timezone change to rest features
9. Add referee crew foul-rate feature (from Basketball Reference referee assignment data)

**Phase 3 — Differentiators (higher complexity, higher ceiling):**
10. Official NBA injury report integration (scrape or use a free-tier injury API)
11. Historical ATS cover rate per team (requires backfilling historical spreads)
12. Four Factors differential as composite feature
13. Minutes projection for player model

**Defer indefinitely:**
- Shot chart features (3-4 hour pipeline, separate concern)
- Lineup net ratings (high complexity, low ROI until Phases 1-2 prove out)
- Sharp/public split data (requires paid data source)

---

## Confidence Assessment

| Feature Category | Confidence | Source |
|-----------------|------------|--------|
| What's already built | HIGH | Direct codebase analysis |
| ORtg/DRtg/net rating as table stakes | HIGH | Established NBA analytics literature; Dean Oliver's Basketball on Paper; widely cited in prediction research |
| Injury proxy as highest-leverage fix | HIGH | CONCERNS.md and model_advisor_notes.md confirm this; feature defined but null |
| Vegas line as feature (market efficiency) | HIGH | Well-established in sports betting literature; efficient market hypothesis applied to sports |
| Pace as necessary for modern era | HIGH | model_advisor_notes.md cites explicitly; model degrades post-2014 specifically because pace changed |
| Referee foul rate | MEDIUM | Documented in model_advisor_notes.md (Proposal 10); real signal but requires data scraping |
| Travel distance effect | MEDIUM | Commonly cited in sports analytics; magnitude in NBA context is debated |
| ATS cover rate rolling | MEDIUM | Standard betting model feature; historical spread data availability is the constraint |
| Public/sharp split | LOW | Strong theoretical signal but data only available through paid sources |
| Four Factors as composite | MEDIUM | Dean Oliver's framework is high-confidence; implementation complexity is the risk |

---

## Sources

- Codebase analysis: `src/features/team_game_features.py`, `src/features/player_features.py`, `src/features/injury_proxy.py` (direct grep, 2026-03-01)
- `.planning/codebase/CONCERNS.md` — audit of known bugs and missing features (2026-03-01)
- `docs/model_advisor_notes.md` — model advisor proposals including Proposals 1-10 (2026-03-01)
- `.planning/PROJECT.md` — active requirements, constraints, key decisions (2026-03-01)
- Dean Oliver, "Basketball on Paper" (2004) — Four Factors framework; ORtg/DRtg definitions; foundational NBA analytics
- NBA Stats API advanced stats endpoints: `leaguedashteamstats` with `MeasureType=Advanced` provides ORtg, DRtg, TS%, eFG%, Pace — all available via existing `get_team_stats_advanced.py` fetcher; already stored in `database/nba.db` as `team_stats_advanced`
- Training data knowledge: Sports betting model design patterns (knowledge cutoff August 2025; LOW confidence for specific betting-system accuracy claims)
