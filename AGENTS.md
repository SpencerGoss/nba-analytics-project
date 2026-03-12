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
**Invoke:** Agent tool

## security-auditor
**Does:** Scans for secrets, hardcoded keys, SQL injection, data leakage in features
**Doesn't:** Fix code — reports findings only
**Invoke:** `security-audit` skill before any push

## Available Skills (auto-trigger — no prompting needed)
| Skill | Triggers when |
|-------|--------------|
| `session-kickoff` | start of session / "continue" |
| `session-wrap-up` | end of session / milestone close |
| `continue` | resume after context compaction |
| `project-journal` | logging progress / end of session |
| `handoff-bridge` | context full / switching to Claude.ai |
| `context-rescue` | context window filling up / Claude seems confused |
| `working-memory` | capture non-obvious insight / debug finding |
| `git-workflow` | any commit, branch, or push |
| `debug-session` | broken build / failing tests |
| `spec-driven-dev` | before building any feature or fix (spec first) |
| `nba-feature-dev` | adding features, scripts, or models to the pipeline |
| `nba-backfill` | backfilling or fetching historical data |
| `nba-model-evaluation` | evaluating or comparing model performance |
| `tdd-workflow` | TDD implementation step-by-step |
| `tdd-workflow` | implementing any feature or fix (TDD superpowers) |
| `superpowers:verification-before-completion` | before claiming work is complete |
| `spec-driven-dev` | writing a multi-step implementation plan |
| `superpowers:brainstorming` | brainstorming approach or architecture |
| `code-review-session` | requesting a code review |
| `superpowers:dispatching-parallel-agents` | independent tasks that can run in parallel |
| `refactor-session` | cleaning up code structure without changing behavior |
| `code-review-session` | reviewing code just written |
| `performance-tuning` | pipeline slow / memory or speed issues |
| `dependency-management` | adding or upgrading Python packages |
| `env-config` | .env / API key setup / new secret |
| `security-audit` | before push / any security concern |
| `webapp-testing` | testing the dashboard or any UI via Playwright |
| `frontend-design` | dashboard UI design / production polish |
| `python-development:python-testing-patterns` | comprehensive testing strategy |
| `python-development:python-configuration` | typed settings / env var configuration |
| `python-development:python-design-patterns` | KISS / SRP design patterns |
| `python-development:uv-package-manager` | managing packages with uv |
| `context-file-maintainer` | CLAUDE.md too long or stale |
| `vscode-ai-project-scaffolder` | architecture change / new agent role |

## Available Plugins

Plugins provide specialized capabilities agents can leverage automatically:

| Plugin | Purpose |
|--------|---------|
| `code-review` | 5 parallel review agents for PR reviews |
| `pr-review-toolkit` | 6 specialized agents for deep PR analysis |
| `machine-learning-ops` | ML pipeline work (training, evaluation, deployment) |
| `quantitative-trading` | Betting/risk analysis and strategy evaluation |
| `data-engineering` | ETL/pipeline design and optimization |
| `scientific-skills` | scikit-learn, statsmodels, plotly, TimesFM |
| `pyright-lsp` | Auto-activates on .py files for type checking |
| `security-guidance` | Passive security monitoring on all changes |
