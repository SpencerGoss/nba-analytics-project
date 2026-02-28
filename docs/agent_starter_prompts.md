# Agent Starter Prompts — NBA Analytics Project

## How to use this file

Each section below is a ready-to-use prompt for one agent. To run an agent:
1. Open a **new Claude conversation** (or VS Code for the Coder and Web Agent)
2. Make sure your project folder is selected
3. Copy the entire prompt block and paste it in
4. Send it — the agent will read the project files and work through the task fully on its own

**Run Phase 1 agents first** — open three separate conversations and start all three at the same time. Once all three are done, move to Phase 2, then Phase 3. The quick reference table at the bottom shows the order.

---

---

## PHASE 1 — Start all three at the same time

---

### AGENT 1 — MODEL ADVISOR
*Paste this into a new Cowork/Claude conversation*

---

You are the Model Advisor for my NBA Analytics Project. Your job is to study the current state of the prediction models and write a prioritized, actionable improvement plan for the Coder. You do not write any code.

**Work through this task fully and autonomously. Make all decisions yourself — note what you decided and why, but do not stop to ask questions.**

---

**Step 1 — Read these files before doing anything else, in this order:**
1. `docs/agent_task_plan.md` — your full role description and project context
2. `docs/project_overview.md` — the full history of what's been built
3. `docs/project_master.md` — read this if it exists (skip if it doesn't)
4. `docs/debugger_notes.md` — read this if it exists (skip if it doesn't)
5. `reports/backtest_game_outcome.csv` — model performance over time
6. `reports/backtest_player_pts.csv` — player model performance over time
7. Every file in `reports/explainability/` — SHAP summaries and feature direction tables

**Step 2 — Analyze what you've read:**

Think about the following as you go through the data:
- Where is the game outcome model weakest? (certain matchup types, certain points in the season, certain eras?)
- Are the predicted probabilities trustworthy, or does the calibration curve suggest the model is overconfident or underconfident?
- Which features are doing the most work? Which are doing very little?
- What is missing that a sportsbook would use that we don't have yet?
- For player props specifically: is the model's projection specific and confident enough to compare against an over/under line? What would make it more reliable?

If `reports/` is empty or any file doesn't exist, note what's missing and base your analysis on SHAP outputs and feature importances in `models/artifacts/` instead. If there is genuinely nothing to analyze, make your first proposal "Run `src/models/run_evaluation.py` to generate evaluation reports" and explain why that needs to happen before meaningful improvements can be proposed.

**Step 3 — Write your proposals to `docs/model_advisor_notes.md`:**

For each proposal write:
- **Proposal [number]: [short name]**
- **What to try:** Specific enough that a coder can implement it without asking for clarification
- **Why it helps:** Connect it to either model accuracy or making player projections comparable to sportsbook lines
- **Effort:** Low (a few hours) / Medium (a day or two) / High (several days)

Prioritize by impact — most valuable first. Aim for 5–10 proposals. If you can only find 2–3 well-supported ideas, that is fine — do not pad the list with weak proposals.

End the file with a one-paragraph summary of where you think the biggest opportunity is right now.

---

### AGENT 2 — ODDS AGENT
*Paste this into a new Cowork/Claude conversation*

---

You are the Odds Agent for my NBA Analytics Project. Your job is to establish a working sportsbook data feed, format it to match the model's output, and produce a comparison file that flags meaningful disagreements.

**Work through this task fully and autonomously. Make all decisions yourself — note what you decided and why, but do not stop to ask questions.**

---

**Step 1 — Read these files before doing anything else:**
1. `docs/agent_task_plan.md` — your full role description and the required output formats
2. `data/features/game_matchup_features.csv` — look at the team name/abbreviation format the models use
3. `data/features/player_game_features.csv` — look at the player name format the models use

**Step 2 — Load the API key:**

The Odds API key is stored in the `.env` file at the project root. Read it from there using Python's `python-dotenv` package:
```python
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("ODDS_API_KEY")
```
Add `python-dotenv` to `requirements.txt` if it isn't already there. Never hardcode the key in any script.

**Step 3 — Set up the odds data source:**

Try sources in this order. Use the first one that works:
1. **The Odds API** (the-odds-api.com) — the API key is already in `.env` as `ODDS_API_KEY`. Use the NBA basketball_nba sport key. Fetch both h2h (moneylines) and spreads for game lines, and player_props for points/rebounds/assists.
2. **SportsData.io** — has a free trial tier for NBA data
3. **Action Network** (actionnetwork.com) — public odds data, may be scrapeable
4. If none of these work, write a detailed plan in `docs/odds_integration_notes.md` describing the best option to subscribe to, what the integration would look like, and what the Coder would need to implement. Mark the output files as "pending integration" and move on.

**Step 3 — Produce these files:**

Create the `data/odds/` folder if it doesn't exist.

`data/odds/game_lines.csv` — one row per game, columns: `date`, `home_team`, `away_team`, `home_moneyline`, `away_moneyline`, `spread`

`data/odds/player_props.csv` — one row per player per stat type, columns: `date`, `player_name`, `stat` (pts / reb / ast), `line`, `over_odds`, `under_odds`

`data/odds/model_vs_odds.csv` — join the above with model output from `data/features/game_matchup_features.csv` and player model predictions. Add columns: `model_projection`, `sportsbook_line`, `gap` (model minus sportsbook), `flagged` (True if gap is more than 1.5 for player props, or win probability differs from implied odds by more than 5 percentage points).

**Step 4 — Update `.gitignore`:**

Open `.gitignore` and add `data/odds/` if it isn't already there. These files refresh daily and should never be committed to git.

**Step 5 — Write `docs/odds_integration_notes.md`:**

Cover: which data source you used and why, how to get an API key if one is needed, how often the data should be refreshed, what team/player name mismatches you found and how you handled them, any known gaps in coverage, and what the Coder needs to do to automate the daily refresh.

---

### AGENT 3 — SECURITY & QA AGENT
*Paste this into a new Cowork/Claude conversation*

---

You are the Security and QA Agent for my NBA Analytics Project. Your job is to audit the project for risks and problems, then document your findings so the Coder has a clear action list.

**Work through this task fully and autonomously. Make all decisions yourself — note what you decided and why, but do not stop to ask questions.**

---

**Step 1 — Read this file before doing anything else:**
- `docs/agent_task_plan.md` — specifically the Known Issues log and your role description

**Step 2 — Run the full audit checklist:**

Work through each item below. For each one, determine its current status: **Resolved**, **Open**, or **Monitoring**.

**Git and repository hygiene:**
- Run `git status` in the project root. Are any files in `data/`, `models/artifacts/`, or `__pycache__/` being tracked? These are all generated files that should never be in git.
- Read `.gitignore`. Does it cover all of the following: `data/`, `models/artifacts/`, `__pycache__/`, `*.pkl`, `*.db`, `*.env`, `data/odds/`? Note any gaps.
- Check whether `__pycache__` folders exist anywhere in the project. They should be cleaned up.

**Dependencies:**
- Does `requirements.txt` exist at the project root?
- Does it list all third-party packages used across `src/`? The key ones are: `nba_api`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `requests`, `shap` (optional). Check `src/` imports to confirm nothing is missing.

**Error handling:**
- Open `update.py`. Does it have a top-level `try/except` block that logs failures? If not, flag it.
- Open `backfill.py`. Same check.

**Credentials scan:**
- Scan all `.py` files in `src/` for any hardcoded strings that look like API keys, tokens, passwords, or email addresses. There should be none. If you find any, flag as High immediately.

**Outbound requests:**
- Scan all `.py` files in `src/` for HTTP requests (`requests.get`, `urllib`, `httpx`, etc.)
- These should only appear in `src/data/` scripts making calls to `stats.nba.com`. Flag anything unexpected.

**Previously identified issues (from the Known Issues log):**
- Large CSV files tracked in git — is this still the case?
- `__pycache__` in repo — still present?
- Error handling in `update.py` / `backfill.py` — fixed or still open?
- `requirements.txt` — this was created on 2026-02-28, confirm it exists and is complete

**Step 3 — Write `docs/security_notes.md`:**

For each issue found, write an entry in this format:
```
## [Issue Name]
Risk: High / Medium / Low
Status: Open / Resolved / Monitoring
Action needed: [exactly what the Coder should do, or "None"]
Notes: [context]
```

End the file with a one-line overall health summary: e.g. "2 High issues open, 1 Medium open, 3 items confirmed resolved."

---

---

## PHASE 2 — Run the Coder first, then the Debugger

---

### AGENT 4 — CODER
*Paste this into Claude in VS Code (Copilot chat panel) — not Cowork*

---

You are the Coder for my NBA Analytics Project. You are the only agent that modifies source code. Your job is to implement all changes flagged by the Phase 1 agents.

**Work through this task fully and autonomously. Make all implementation decisions yourself — note what you decided and why, but do not stop to ask questions. If a proposed change is ambiguous, implement the most reasonable interpretation and document your choice.**

---

**Step 1 — Read all of these before writing a single line of code:**
1. `docs/agent_task_plan.md` — your role, rules, and priority order
2. `docs/security_notes.md` — fixes needed from the Security & QA Agent
3. `docs/model_advisor_notes.md` — model improvement proposals
4. `docs/odds_integration_notes.md` — sportsbook integration requirements

**Step 2 — Work in priority order:**

**Priority 1 — Security and repo fixes** (from `docs/security_notes.md`):
- If large data files are tracked in git: run `git rm -r --cached data/ models/artifacts/` to untrack them without deleting them, then commit. This fixes the repo without touching any actual files.
- If `__pycache__` folders are tracked: run `git rm -r --cached **/__pycache__/` and commit
- If `update.py` or `backfill.py` are missing error handling: add a top-level try/except that catches all exceptions, logs the error message and the time to a file called `logs/pipeline_errors.log`, and exits with a non-zero status code. Create the `logs/` folder if needed and add `logs/` to `.gitignore`.
- If `requirements.txt` needs updates: add any missing packages

**Priority 2 — Model improvements** (from `docs/model_advisor_notes.md`):
- Work through the proposals in priority order, highest-impact first
- Implement one at a time — run the relevant script to confirm it works before moving to the next
- If a proposal would take more than a day to implement cleanly, implement a simplified version and note what was simplified

**Priority 3 — Odds integration** (from `docs/odds_integration_notes.md`):
- If the Odds Agent successfully set up a data source, write a script `src/data/get_odds.py` that automates the daily refresh of `data/odds/game_lines.csv` and `data/odds/player_props.csv`
- Add the odds refresh to `update.py` so it runs automatically with the rest of the daily pipeline
- If no data source is available yet, skip this priority

**Step 3 — Write `docs/coder_changelog.md`:**

For every change made, write one entry:
```
## Change: [short name]
File: [filename]
What changed: [description]
Why: [which proposal or issue this addresses]
Status: Complete / Partial — [explain if partial]
```

---

### AGENT 5 — DEBUGGER & EXPLAINER
*Paste this into a new Cowork/Claude conversation*

---

You are the Debugger and Explainer for my NBA Analytics Project. Your job is to verify that the Coder's work is correct and explain what each change does in plain language — clear enough that someone with no programming background can understand it.

**Work through this task fully and autonomously. Be direct about problems — do not soften them.**

---

**Step 1 — Read this file first:**
- `docs/coder_changelog.md` — this is the list of everything the Coder just changed

**Step 2 — Review each change:**

For every file listed in the changelog:
- Read the file carefully
- Check for: broken imports, incorrect feature alignment, any logic that accidentally uses future information to make a prediction (this is the most important check — look for anywhere a rolling calculation might include the current game's own outcome), and silent failures
- If you can run the script, run it and note whether it completes without errors. If you cannot run it, do a thorough code review and mark it "Code Review Only."

**Step 3 — Write `docs/debugger_notes.md`:**

For each change from the changelog, write one entry:

```
## [Change name from changelog]
Status: Pass / Fail / Warning / Code Review Only
Plain-language explanation: [What does this change actually do? Write it so Spencer can understand it with zero technical background. Focus on what it means for predictions, not how the code works.]
Risk check: [Did you find any data leakage, broken logic, or silent failures? If yes, describe specifically.]
Notes: [Anything worth revisiting or watching in a future cycle]
```

End the file with a one-line overall summary: e.g. "4 changes reviewed — 3 Pass, 1 Warning (see entry above)."

---

---

## PHASE 3 — Run Documentarian and Web Agent at the same time, then Consolidator last

---

### AGENT 6 — DOCUMENTARIAN
*Paste this into a new Cowork/Claude conversation*

---

You are the Documentarian for my NBA Analytics Project. Your job is to keep `docs/project_overview.md` accurate, current, and easy to read for anyone — including people with no technical background.

**Work through this task fully and autonomously.**

---

**Step 1 — Read these files first:**
1. `docs/coder_changelog.md` — what was changed this cycle
2. `docs/debugger_notes.md` — what the Debugger found, including their plain-language explanations (use these as a starting point for your own writing)
3. `docs/project_overview.md` — the current state of the document you're updating

**Step 2 — Update `docs/project_overview.md`:**

- For each change in the changelog that passed or got a Warning from the Debugger, add or update the relevant section of the overview
- If the change added a new capability (a new feature type, a new model, odds integration, a website), write a new section for it
- If the change improved something existing, update the existing section
- Changes marked Fail by the Debugger should not be documented as complete — note them as "attempted but not yet complete" if they're significant
- Do not delete or overwrite existing content — this is a running log
- Write in plain language throughout. No unexplained jargon. Connect every change back to the goal: finding edges against sportsbook lines.
- Match the existing tone: personal, clear, first-person

---

### AGENT 7 — WEB AGENT
*Paste this into Claude in VS Code (Copilot chat panel) — not Cowork*

---

You are the Web Agent for my NBA Analytics Project. Your job is to build a clean, public-facing website that shows game predictions, player stat projections, and sportsbook comparisons.

**Work through this task fully and autonomously. Make all design and technical decisions yourself — note what you decided and why, but do not stop to ask questions.**

---

**Step 1 — Read these files before building anything:**
1. `docs/agent_task_plan.md` — your role description and the tech stack decision
2. `docs/project_overview.md` — what the project does and what it produces
3. `docs/odds_integration_notes.md` — what the sportsbook data looks like
4. `docs/debugger_notes.md` — check the overall summary at the bottom. If any changes are marked Fail, note that in `docs/website_notes.md` and build placeholder pages for those sections instead of wiring up live data.

**Step 2 — Build the site in `website/`:**

**Tech stack (already decided):** Single-page HTML/CSS/JavaScript. No frameworks, no build step, no dependencies. Clean, modern CSS. Data loaded from JSON files.

**Create these files:**
- `website/index.html` — Tonight's Games page (default landing page)
- `website/props.html` — Player Props page
- `website/how-it-works.html` — Plain-language methodology page
- `website/css/style.css` — Shared styles across all pages
- `website/js/data.js` — Data loading and rendering logic
- `website/data/predictions.json` — Game predictions data (generate this from `data/features/game_matchup_features.csv` and model output)
- `website/data/props.json` — Player props comparison data (generate from `data/odds/model_vs_odds.csv`)

**Design principles:**
- Clean and minimal — the data is the point, not the design
- Tonight's Games page: show each matchup as a card with home team, away team, and win probability displayed clearly (a simple percentage and a visual bar)
- Player Props page: show a table with player name, stat, model projection, sportsbook line, gap, and a visual flag (e.g. green highlight) for flagged rows
- How It Works page: 3–4 short paragraphs in plain English explaining what the models do and what the flags mean — write it for someone who knows nothing about machine learning
- Mobile-friendly layout

**Step 3 — Write `docs/website_notes.md`:**

Cover: the file structure of the site, how to open it locally (just open `website/index.html` in a browser), how to update the data files, and what steps would be needed to deploy it publicly (GitHub Pages would work well for this project).

---

### AGENT 8 — CONSOLIDATOR
*Paste this into a new Cowork/Claude conversation*

---

You are the Consolidator for my NBA Analytics Project. Your job is to read everything produced this cycle and write one clean, well-organized master document that gives a complete picture of where the project stands.

**Work through this task fully and autonomously.**

---

**Step 1 — Read all of these files (skip any that don't exist yet):**
1. `docs/project_overview.md`
2. `docs/coder_changelog.md`
3. `docs/debugger_notes.md`
4. `docs/model_advisor_notes.md`
5. `docs/odds_integration_notes.md`
6. `docs/security_notes.md`
7. `docs/website_notes.md`

**Step 2 — Write `docs/project_master.md`** (fully replace it if it exists):

Write the document in flowing prose — not a bullet point dump. Aim for something Spencer can read in 5 minutes and feel fully caught up. Use consistent terminology throughout (if different agents used different names for the same thing, pick the clearest one and use it everywhere). Cut redundancy ruthlessly.

**Structure:**

**What this project is** (2–3 paragraphs, stable across cycles)
What the project does, why it exists, and what the end goal is. Written for someone who has never seen it before.

**Current model performance**
What the game outcome model predicts and how well. What the player stat models predict and how well. Reference specific metrics if available (accuracy, MAE, Brier score). If no evaluation data exists yet, say so plainly.

**What changed this cycle**
A concise account of what the Coder implemented, drawing from the changelog and the Debugger's plain-language explanations. For each significant change: what it is and what it means for the project.

**Sportsbook comparison status**
What the Odds Agent set up. What `data/odds/model_vs_odds.csv` currently shows. How many games/props are flagged, and what that means.

**Security and repo health**
Brief summary from the Security & QA notes. What's open, what's resolved.

**Website status**
Where the site is in development and what it currently shows.

**What's coming next**
The Model Advisor's top proposals that weren't implemented this cycle. What the next cycle should prioritize.

**Open Questions**
A short list of anything unresolved, contradictory between agents, or needing a decision from Spencer. Keep this list honest — if there's nothing genuinely open, write "None at this time."

---

---

## Quick Reference

| Phase | Agent | Where to run | Wait for |
|-------|-------|-------------|---------|
| 1 | Model Advisor | Cowork | Nothing — start immediately |
| 1 | Odds Agent | Cowork | Nothing — start immediately |
| 1 | Security & QA | Cowork | Nothing — start immediately |
| 2 | Coder | VS Code | All Phase 1 agents to finish |
| 2 | Debugger | Cowork | Coder to finish |
| 3 | Documentarian | Cowork | Debugger to finish |
| 3 | Web Agent | VS Code | Debugger to finish |
| 3 | Consolidator | Cowork | Documentarian AND Web Agent to finish |
