<!-- TL;DR: What each agent role does and doesn't do. -->
# Agents — NBA Analytics Project

## code-reviewer
**Does:** Reviews diffs for bugs, style violations, data leakage risks, missing tests
**Doesn't:** Rewrite code or make architecture calls
**Invoke:** Agent tool with `.claude/agents/code-reviewer.md` or `/review`

## spec-planner
**Does:** Turns feature/phase requests into SPEC.md with stages and acceptance criteria
**Doesn't:** Write any code
**Invoke:** `/plan` command

## refactor-agent
**Does:** Improves code structure without changing behavior; extracts utilities
**Doesn't:** Add features, touch raw data, modify tests to fit broken code
**Invoke:** Agent tool with `isolation: worktree`

## security-auditor
**Does:** Scans for secrets, hardcoded keys, SQL injection, data leakage in features
**Doesn't:** Fix code — reports findings only
**Invoke:** `security-audit` skill before any push

## Available Skills (auto-trigger — no prompting needed)
| Skill | Triggers when |
|-------|--------------|
| `session-kickoff` | "continue" / start of session |
| `project-journal` | end of session / logging progress |
| `git-workflow` | any commit, branch, or push |
| `debug-session` | broken build / failing tests (quick session) |
| `gsd:debug` | structured debugging with persistent state across resets |
| `nba-feature-dev` | adding features, scripts, or models to the pipeline |
| `nba-backfill` | backfilling or fetching historical data |
| `nba-model-evaluation` | evaluating or comparing model performance |
| `tdd-workflow` | implementing any feature or fix (write tests first) |
| `refactor-session` | cleaning up code structure without changing behavior |
| `env-config` | .env / API key setup |
| `context-file-maintainer` | CLAUDE.md too long or stale |
| `security-audit` | before push / any security concern |
| `handoff-bridge` | context full / switching to Claude.ai |
| `vscode-ai-project-scaffolder` | architecture change / new agent role |
| `api-integration` | adding a new data provider / API client |
| `webapp-testing` | testing the dashboard or any UI via Playwright |
| `gsd:plan-phase` | planning a new phase |
| `gsd:execute-phase` | executing a planned phase |
