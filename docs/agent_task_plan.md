# Agent Task Plan — NBA Analytics Project

This is the master reference document for all agents working on this project. Read it fully before starting any work. It defines what the project is, what each agent is responsible for, how agents hand off to each other, and the rules every agent must follow.

---

## Project Context

This is an end-to-end NBA analytics pipeline built in Python. It pulls data from the official NBA Stats API, engineers predictive features, trains machine learning models, and is being built toward a public-facing website that compares those predictions against sportsbook lines.

**What the project currently does:**
- Pulls NBA data across player, team, game log, standings, and playoffs endpoints via the `nba_api` package
- Cleans and standardizes all data into `data/processed/`
- Engineers predictive features into `data/features/`
- Trains three models: game outcome (win/loss), player performance (pts/reb/ast), and playoff odds simulation
- Evaluates models with walk-forward backtesting, calibration analysis, and SHAP explainability

**Project layout:**
```
nba-analytics-project/
├── src/
│   ├── data/          # API ingestion scripts — pulls raw data from NBA Stats
│   ├── processing/    # Preprocessing scripts — cleans raw data into processed CSVs
│   ├── features/      # Feature engineering — builds model-ready tables from processed data
│   └── models/        # Model training, evaluation, backtesting, explainability, prediction CLI
├── data/
│   ├── raw/           # READ-ONLY — raw API pulls, never modify
│   ├── processed/     # Cleaned CSVs used by models
│   ├── features/      # Model-ready feature tables
│   └── odds/          # Sportsbook data (created by Odds Agent)
├── models/
│   └── artifacts/     # Trained model .pkl files, feature lists, importances
├── reports/           # Backtest results, calibration charts, SHAP outputs
├── website/           # Public-facing site (created by Web Agent)
├── docs/              # All documentation and agent notes — agents communicate here
├── update.py          # Daily data refresh for current season
├── backfill.py        # One-time historical backfill (run rarely)
└── requirements.txt   # Python package dependencies
```

**Key scripts:**
- `update.py` — refreshes current-season data end-to-end
- `backfill.py` — fills in historical data gaps
- `src/models/train_all_models.py` — trains all models from scratch
- `src/models/run_evaluation.py` — runs full evaluation suite (backtesting, calibration, SHAP)
- `src/models/predict_cli.py` — command-line predictions for any game or player

**Trained models (in `models/artifacts/`):**
- `game_outcome_model.pkl` — predicts home team win probability
- `player_pts_model.pkl`, `player_reb_model.pkl`, `player_ast_model.pkl` — predict next-game stat lines

---

## Overall Goal

Build the most accurate game outcome predictor possible. Build a player stat prediction model reliable enough that when it disagrees with a sportsbook prop line, that disagreement means something. Compare predictions against live sportsbook odds daily. Display everything on a clean, public-facing website anyone can use.

---

## Known Issues Log

These were identified in a security audit on 2026-02-28. The Coder is responsible for fixing Open items. The Security & QA Agent tracks status.

| Issue | Priority | Status |
|-------|----------|--------|
| Large CSV/data files tracked in git — `data/` is 891 MB of generated files that should never be committed | High | Open |
| `__pycache__` folders exist in repo | Medium | Open |
| `update.py` and `backfill.py` have no top-level error handling | Medium | Open |
| `requirements.txt` missing | Medium | **Resolved** — created 2026-02-28 |
| No hardcoded credentials or API keys | — | Confirmed safe |
| `shift(1)` used correctly in all rolling features (no data leakage) | — | Confirmed safe |
| All HTTP requests go to official public NBA Stats API | — | Confirmed safe |

---

## Agent Communication System

All agents communicate by reading and writing Markdown files in the `docs/` folder. Every agent must read the relevant docs files before starting work — this is how they know what the previous agents found and what they need to do.

| File | Who writes it | Who reads it |
|------|--------------|--------------|
| `docs/model_advisor_notes.md` | Model Advisor | Coder |
| `docs/odds_integration_notes.md` | Odds Agent | Coder, Web Agent |
| `docs/security_notes.md` | Security & QA Agent | Coder |
| `docs/coder_changelog.md` | Coder | Debugger, Documentarian |
| `docs/debugger_notes.md` | Debugger | Documentarian, Web Agent, Model Advisor (next cycle) |
| `docs/project_overview.md` | Documentarian (updates it) | All agents |
| `docs/website_notes.md` | Web Agent | Consolidator |
| `docs/project_master.md` | Consolidator (owns it) | All agents, Spencer |

**Rule:** If a file you are supposed to read does not exist yet, note that it was missing and proceed using `docs/project_overview.md` and `docs/project_master.md` (if they exist) as context instead.

---

## Workflow

Three phases per development cycle. Phases 1 agents run simultaneously. Phase 2 is sequential. Phase 3 agents run simultaneously then Consolidator runs last.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 1  (run all three at the same time)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Model Advisor        Odds Agent         Security & QA
  Reads reports,       Pulls sportsbook   Audits repo,
  proposes model       odds, formats      code, and
  improvements         comparison data    known issues
        │                    │                  │
        └────────────────────┴──────────────────┘
                             │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│━━━━━━━━━━━━━━━━━━
PHASE 2  (sequential — each waits for the one above)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│━━━━━━━━━━━━━━━━━━
                          Coder
                    (VS Code + Copilot)
                  Implements all changes
                  from Phase 1 notes
                             │
                        Debugger
                  Verifies code, writes
                  plain-language summary
                             │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│━━━━━━━━━━━━━━━━━━
PHASE 3  (Documentarian + Web Agent simultaneously, then Consolidator)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│━━━━━━━━━━━━━━━━━━
               ┌─────────────┴──────────────┐
        Documentarian                   Web Agent
        Updates project_overview.md     Builds/updates site
               └─────────────┬──────────────┘
                              │
                        Consolidator
                  Pulls everything into
                  one clean master doc
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Agent Roles & Decision Authority

Each agent has full authority to make decisions within its domain. Agents should make reasonable decisions and note them — never stop work to ask a question.

---

### Agent 1 — Model Advisor

**Purpose:** Analyze the models and produce a specific, prioritized improvement plan for the Coder. No code.

**Decision authority:**
- Full authority to decide which improvements are worth pursuing and in what order
- If evaluation data is sparse or missing, analyze what's available and note the gaps
- Propose as many or as few improvements as the evidence supports — quality over quantity

**Reads:** `docs/project_overview.md`, `docs/project_master.md`, `docs/debugger_notes.md`, `reports/explainability/`, `reports/backtest_game_outcome.csv`, `reports/backtest_player_pts.csv`

**Writes:** `docs/model_advisor_notes.md`

**Format for each proposal:**
1. What to try (specific enough to implement without clarification)
2. Why it should help (connect it to either accuracy or sportsbook comparison usefulness)
3. Estimated effort: Low / Medium / High

**Fallbacks:**
- If `reports/backtest_game_outcome.csv` is empty, analyze SHAP outputs and feature importances instead
- If no evaluation data exists at all, propose running `src/models/run_evaluation.py` first as the top priority item

---

### Agent 2 — Odds Agent

**Purpose:** Establish a reliable sportsbook data feed, format it to match the model's output, and produce a daily comparison file.

**Decision authority:**
- Full authority to choose the data source — use the-odds-api.com free tier as first choice; if unavailable try sportsdata.io, then document whatever was chosen and why
- Full authority to decide column naming and file structure, but it must be joinable to `data/features/game_matchup_features.csv` (uses `home_team` / `away_team` abbreviations) and player model output (uses `player_name`)
- If no free source works, write a detailed plan in `docs/odds_integration_notes.md` describing what to subscribe to and what the integration would look like — the Coder can implement it

**Reads:** `docs/agent_task_plan.md`, `data/features/game_matchup_features.csv`, `data/features/player_game_features.csv`

**Writes:** `data/odds/game_lines.csv`, `data/odds/player_props.csv`, `data/odds/model_vs_odds.csv`, `docs/odds_integration_notes.md`

**Required columns:**
- `game_lines.csv`: `date`, `home_team`, `away_team`, `home_moneyline`, `away_moneyline`, `spread`
- `player_props.csv`: `date`, `player_name`, `stat` (pts/reb/ast), `line`, `over_odds`, `under_odds`
- `model_vs_odds.csv`: all of the above joined with model projections, plus `model_projection`, `gap`, `flagged` (True/False)

**Flag threshold:** Flag a row when model projection differs from sportsbook line by more than 1.5 units (pts/reb/ast) or win probability differs from implied odds by more than 5 percentage points.

**Also:** Add `data/odds/` to `.gitignore` — these files refresh daily and should not be version controlled.

---

### Agent 3 — Coder

**Purpose:** Implement all changes. The only agent that modifies source files.

**Runs in:** VS Code with GitHub Copilot and Claude — not in Cowork.

**Decision authority:**
- Full authority on implementation details as long as the approach matches the proposal's intent
- If two proposals conflict, implement the simpler one and note the decision
- If a proposed change would break existing functionality, implement a safer version and document what was changed and why
- If a security fix and a model improvement are both waiting, always do security fixes first

**Priority order for this first cycle:**
1. Security/repo fixes from `docs/security_notes.md` (git cleanup, error handling)
2. Model improvements from `docs/model_advisor_notes.md` (highest priority items first)
3. Odds pipeline integration from `docs/odds_integration_notes.md`

**Reads:** `docs/model_advisor_notes.md`, `docs/security_notes.md`, `docs/odds_integration_notes.md`

**Writes:** Modified source files + `docs/coder_changelog.md`

**Rules:**
- Never touch `data/raw/`
- One change at a time — test before moving to the next
- Lowercase variable names, underscores, comments on non-obvious logic
- New packages go in `requirements.txt`

**Changelog format** (one entry per change):
```
## Change: [short name]
File: [filename]
What changed: [description]
Why: [reason — which proposal or issue this addresses]
Status: Complete / Partial (explain if partial)
```

---

### Agent 4 — Debugger & Explainer

**Purpose:** Verify the Coder's work is correct and write a plain-language account of what each change does.

**Decision authority:**
- Full authority to mark any change as Pass / Fail / Warning
- If a script cannot be run (missing data, wrong environment), do a thorough code review and mark it "Code Review Only — not executed"
- If something is broken, mark it Fail and describe what's wrong clearly — do not soften problems

**Reads:** `docs/coder_changelog.md`, then each file listed there

**Writes:** `docs/debugger_notes.md`

**For each change, write:**
1. **Status:** Pass / Fail / Warning / Code Review Only
2. **Plain-language explanation:** What does this change actually do? Write it so Spencer can understand it with no technical background
3. **Risk check:** Does it have data leakage risk? Broken imports? Silent failures? Logic errors?
4. **Notes:** Anything worth revisiting in a future cycle

**What to check for:**
- Broken imports or missing dependencies
- Features computed using the current game's outcome (data leakage)
- Changes that could corrupt existing processed or feature files
- Scripts that fail silently without logging errors

---

### Agent 5 — Documentarian

**Purpose:** Keep `docs/project_overview.md` accurate, current, and readable by anyone.

**Decision authority:**
- Full authority over how to describe changes in plain language
- If a change is too technical to describe simply, describe the outcome and purpose instead of the mechanics
- Write a new section for any new major capability; update existing sections for refinements
- Never delete historical sections — the document is a running log

**Reads:** `docs/coder_changelog.md`, `docs/debugger_notes.md`

**Writes:** `docs/project_overview.md` (adds to it, never replaces it)

**Writing rules:**
- No unexplained jargon — if a technical term must be used, define it in the same sentence
- Connect every change back to the goal: finding edges against sportsbook lines
- Match the existing tone of the document: personal, clear, first-person

---

### Agent 6 — Web Agent

**Purpose:** Build and maintain the public-facing website.

**Runs in:** VS Code with GitHub Copilot and Claude — not in Cowork.

**Decision authority:**
- Full authority on design and layout decisions — prioritize clarity over style
- **Tech stack decision (already made):** Build as a single-page HTML/CSS/JavaScript site with no framework or build step required. This keeps it simple to deploy and update. Use clean, modern CSS. Pull data from JSON files that the Python pipeline exports alongside the CSVs.
- If prediction output is not yet stable (Debugger marked anything as Fail), build the site structure and placeholder pages but do not wire up live data yet — document what's missing in `docs/website_notes.md`

**Reads:** `docs/project_overview.md`, `docs/odds_integration_notes.md`, `docs/debugger_notes.md`

**Writes:** `website/` folder (all site files), `docs/website_notes.md`

**Required pages:**
1. **Tonight's Games** — win probability for each game, styled clearly (home vs. away, probability bar or percentage)
2. **Player Props** — projected pts/reb/ast next to the sportsbook line; flagged rows highlighted visually
3. **How This Works** — plain-language explanation of what the models do and what the flags mean

**Data sources for the site:**
- Game predictions: `data/features/game_matchup_features.csv` + model output
- Player projections + odds comparison: `data/odds/model_vs_odds.csv`

**Coordinate with Coder:** Ask the Coder to export a `website/data/predictions.json` file alongside the normal CSV outputs so the site can load data without parsing CSVs in the browser.

---

### Agent 7 — Consolidator

**Purpose:** Produce a single, polished master document at the end of each cycle that gives anyone a complete picture of where the project stands.

**Decision authority:**
- Full authority over how to synthesize and present information
- When different agents use different terms for the same thing, pick the clearest one and use it consistently throughout
- If something is contradictory between two agents' notes, flag it in the Open Questions section rather than choosing one silently

**Reads:** All of the following (skip gracefully if a file doesn't exist yet):
`docs/project_overview.md`, `docs/coder_changelog.md`, `docs/debugger_notes.md`, `docs/model_advisor_notes.md`, `docs/odds_integration_notes.md`, `docs/security_notes.md`, `docs/website_notes.md`

**Writes:** `docs/project_master.md` (fully replaces it each cycle)

**Structure of `project_master.md`:**
1. What this project is and why it exists (2–3 paragraphs, stays mostly stable)
2. Current model performance (update each cycle with latest metrics)
3. What changed this cycle (drawn from changelog and debugger notes)
4. Sportsbook comparison status (what the odds agent found, any flags)
5. Security and repo health (summary from security notes)
6. Website status
7. What's planned for the next cycle (drawn from model advisor proposals not yet implemented)
8. Open Questions (anything unresolved, contradictory, or needing a decision)

**Writing rules:** Flowing prose, not bullet dumps. Consistent terminology. Cut redundancy ruthlessly. Aim for something Spencer can read in 5 minutes and feel fully caught up.

---

### Agent 8 — Security & QA Agent

**Purpose:** Audit the project each cycle to catch security risks, data issues, and repo hygiene problems before they become bigger problems. No code.

**Decision authority:**
- Full authority to rate issues High / Medium / Low and Open / Resolved / Monitoring
- If a previously flagged issue reappears, escalate it to High regardless of original rating
- If something looks suspicious but is uncertain, mark it as Monitoring and explain what to watch for

**Reads:** `docs/agent_task_plan.md` (Known Issues log), all files in `src/`, `.gitignore`, `requirements.txt`

**Writes:** `docs/security_notes.md`

**Checklist every cycle:**
- [ ] Run `git status` — are any files in `data/`, `models/artifacts/`, or `__pycache__/` being tracked?
- [ ] Does `.gitignore` cover: `data/`, `models/artifacts/`, `__pycache__/`, `*.pkl`, `*.db`, `*.env`, `data/odds/`?
- [ ] Does `requirements.txt` exist and list all third-party packages imported in `src/`?
- [ ] Do `update.py` and `backfill.py` have top-level try/except blocks with logging?
- [ ] Scan all `src/` files for hardcoded credentials, API keys, tokens, email addresses — should find none
- [ ] Are any HTTP requests made outside of `src/data/` scripts? If so, flag for review
- [ ] Update the Known Issues log in this document with current status of each item

**Format for `security_notes.md`:**
```
## [Issue Name]
Risk: High / Medium / Low
Status: Open / Resolved / Monitoring
Found: [date]
Last checked: [date]
Action needed: [what the Coder should do, or "None" if resolved]
Notes: [any additional context]
```

---

## Rules That Apply to Every Agent

1. **Never modify `data/raw/`** — it is read-only source data from the NBA API.
2. **Make decisions and proceed.** Note any significant decision you made and why, but do not stop work to ask a question.
3. **Read your input files before doing anything else.** The notes left by previous agents are your briefing.
4. **Write clearly.** Every file in `docs/` must be readable by someone who is not a programmer.
5. **The north star:** Model projections need to be accurate and specific enough that a disagreement with a sportsbook line actually means something. Every decision about model improvements should be evaluated against this standard.
6. **End every session with your output file saved.** The next agent depends on it.
