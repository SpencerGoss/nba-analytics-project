<!-- TL;DR: Maps tasks to files. Read this before reading anything else. -->
# AI Index — NBA Analytics Project

## How to Use
Find your task below. Read only the listed files. Skip everything else.

## Tasks

### Start a session
Read: `.planning/STATE.md` → `PROJECT_JOURNAL.md` → `CLAUDE.md`
Or: type `continue` to trigger session-kickoff automatically.

### Add/fix a data fetcher
Read: `ARCHITECTURE.md` → `src/data/<relevant>.py` → `.claude/rules/nba-domain.md`
Hard rule: never modify `data/raw/` files.

### Add/fix a feature
Read: `ARCHITECTURE.md` → `src/features/<relevant>.py` → `.claude/rules/nba-domain.md`
Hard rule: all rolling features must use `shift(1)` — no data leakage.

### Retrain a model
Read: `ARCHITECTURE.md` → `src/models/` → `.planning/STATE.md`
Hard rule: expanding-window validation only. After retraining → run calibration.py.

### Fix a bug
Read: `.planning/codebase/CONCERNS.md` → `CONTEXT.md` → failing file
Then: invoke `debug-session` skill.

### Review code
Use: `.claude/agents/code-reviewer.md` via Agent tool
Or: `/review` command

### Plan a new phase
Use: `gsd:plan-phase` skill → `.planning/STATE.md` for current position

### Commit changes
Use: `git-workflow` skill → `/wrap-up` command

### Security check before push
Use: `security-audit` skill

### Update journal / log progress
Use: `project-journal` skill → `/wrap-up` command

### Something is broken
Read: `.planning/codebase/CONCERNS.md` first
Then: invoke `debug-session` skill
