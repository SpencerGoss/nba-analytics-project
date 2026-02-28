# Security & QA Notes
**Agent:** Security & QA Agent
**Date:** 2026-02-28
**Cycle:** 1 (first full audit)

---

## Large Data and Model Files Still Tracked in Git

Risk: **High**
Status: **Open**
Found: 2026-02-28
Last checked: 2026-02-28
Action needed: The Coder needs to untrack these files from git using `git rm --cached -r data/ models/artifacts/` followed by a new commit. The `.gitignore` now correctly excludes these folders, but adding a path to `.gitignore` does NOT retroactively remove files that were already committed — that step must be done manually. Until this is done, every `git status` will show these files as "modified," and any future `git push` could re-upload hundreds of megabytes of generated data.
Notes: `git ls-files data/` shows **80 tracked data files** across `data/raw/` and `data/processed/`. `git ls-files models/artifacts/` shows **12 tracked model artifact files** (importance CSVs). These are all generated files — they can always be recreated by running the pipeline — and they should never be stored in version control. This was flagged as High in the previous audit and remains unresolved.

---

## Error Handling Missing in update.py and backfill.py

Risk: **Medium**
Status: **Open**
Found: 2026-02-28
Last checked: 2026-02-28
Action needed: The Coder should wrap the body of each `main()` function in a `try/except Exception` block that catches any failure, logs it (including the error message and a timestamp), and exits with a non-zero status code. Without this, if the NBA API goes down mid-run or a file write fails, the script will crash silently — no error gets logged anywhere, and the pipeline just stops without any record of what went wrong.
Notes: Confirmed by reading both files. Neither `update.py` nor `backfill.py` has a top-level try/except block. The internal helper functions also do not catch exceptions. This was flagged in the previous audit and remains unresolved.

---

## .gitignore Missing *.pkl and *.db Patterns

Risk: **Medium**
Status: **Open**
Found: 2026-02-28
Last checked: 2026-02-28
Action needed: The Coder should add `*.pkl` and `*.db` to `.gitignore`. The current `.gitignore` covers the `models/artifacts/` folder by name, which protects the known model files. But if a `.pkl` file is ever created elsewhere in the project (during testing, experimentation, or by a future agent), it would be tracked automatically. Adding `*.pkl` as a blanket rule closes that gap. Same logic applies to `*.db` — no database files exist yet, but the rule should be in place before they do.
Notes: Current `.gitignore` covers: `data/raw/`, `data/processed/`, `data/features/`, `data/odds/`, `models/artifacts/`, `__pycache__/`, `*.py[cod]`, `.env`, `venv/`, `.venv/`, `logs/`, `*.log`. Missing from the audit checklist requirements: `*.pkl`, `*.db`. Also note: `.env` is listed but `*.env` is not — this is acceptable since `.env` is the standard file name, but worth knowing.

---

## __pycache__ Folders Present on Disk

Risk: **Low**
Status: **Monitoring**
Found: 2026-02-28
Last checked: 2026-02-28
Action needed: No action required in git — the `.gitignore` is correctly excluding these folders and they are **not** tracked in git (confirmed via `git ls-files`). However, five `__pycache__` folders exist locally on disk: at the project root and inside `src/data/`, `src/features/`, `src/models/`, and `src/processing/`. These are harmless but can be cleaned up with `find . -type d -name __pycache__ -exec rm -rf {} +` if desired.
Notes: This was listed as Open in the previous audit. Reclassified to Monitoring because `git ls-files | grep __pycache__` returned nothing — the files are not in git. The `.gitignore` rule `__pycache__/` is working correctly. The local folders on disk are normal Python behavior and pose no risk.

---

## requirements.txt Exists and Is Complete

Risk: —
Status: **Resolved**
Found: 2026-02-28 (created that date)
Last checked: 2026-02-28
Action needed: None
Notes: `requirements.txt` exists at the project root and covers all confirmed third-party packages used across `src/`: `nba_api`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `shap`, `requests`, and `python-dotenv`. The `argparse` import in `src/models/train_all_models.py` is part of Python's standard library and does not need to be listed. No missing packages found.

---

## No Hardcoded Credentials or API Keys

Risk: —
Status: **Resolved**
Found: 2026-02-28
Last checked: 2026-02-28
Action needed: None
Notes: Scanned all `.py` files in `src/` for API keys, tokens, passwords, email addresses, and long hardcoded strings. One match was flagged by the pattern scanner — a code comment in `src/data/get_player_positions.py` reading "First token before any hyphen." This is a description of string-parsing logic, not a credential. No real credentials of any kind were found. Confirmed safe.

---

## All HTTP Requests Contained Within src/data/

Risk: —
Status: **Resolved**
Found: 2026-02-28
Last checked: 2026-02-28
Action needed: None
Notes: Scanned all `.py` files in `src/` for direct HTTP calls (`requests.get`, `urllib`, `httpx`, etc.). Zero direct HTTP calls were found anywhere in `src/`. All network traffic to `stats.nba.com` is handled internally by the `nba_api` package, which is the expected and correct pattern. No unexpected outbound requests detected.

---

## python-dotenv Listed in requirements.txt But Not Yet Called

Risk: **Low**
Status: **Monitoring**
Found: 2026-02-28
Last checked: 2026-02-28
Action needed: No immediate action needed. When the Odds Agent integration is implemented and an API key is introduced (e.g., for the-odds-api.com), the Coder must add a `load_dotenv()` call at the top of whichever script uses that key — otherwise the `.env` file will exist but never be loaded, and the key will silently be unavailable.
Notes: `python-dotenv` is correctly listed as a dependency, which suggests it was added in anticipation of the Odds Agent's API key. But `grep` across all `src/` files, `update.py`, and `backfill.py` found zero calls to `load_dotenv()` or `os.getenv()`. This is not a problem yet — there are no API keys to load — but it will become one the moment an API key is added to `.env`. Flagging now so it doesn't get missed.

---

## Reports Folder Contains Generated Files Tracked in Git

Risk: **Low**
Status: **Monitoring**
Found: 2026-02-28
Last checked: 2026-02-28
Action needed: No urgent action needed, but the Coder and Spencer should decide whether `reports/` should be in git or excluded. If excluded, add `reports/` to `.gitignore` and untrack the existing files with `git rm --cached -r reports/`.
Notes: `git status` shows modified files in `reports/` including `reports/backtest_game_outcome.csv`, `reports/backtest_player_pts.csv`, and several SHAP explainability CSVs. These are generated outputs from the model evaluation pipeline. Unlike the raw data files (which are purely inputs), there is an argument for keeping reports in git as a historical performance record. However, they are regenerated every time `run_evaluation.py` runs, so they will generate git noise constantly. Decision left to Spencer and the Coder — this note is here to make sure the decision is made intentionally rather than overlooked.

---

## Overall Health Summary

**1 High issue open, 2 Medium issues open, 2 Low items monitoring, 3 items confirmed resolved.**

The biggest outstanding risk is the large data files still tracked in git — this inflates the repository, slows cloning, and could cause accidental data uploads. Error handling in the two main pipeline scripts is the next priority: without it, silent failures will be hard to diagnose. Everything else is either resolved or low-urgency.
