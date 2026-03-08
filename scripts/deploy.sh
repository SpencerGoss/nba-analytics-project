#!/bin/bash
# Manual deploy: run pipeline + push dashboard data to GitHub Pages
# Usage: bash scripts/deploy.sh
set -e

# Activate venv (Git Bash on Windows uses Scripts/, Linux/Mac uses bin/)
source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate

echo "Running pipeline..."
python update.py

echo "Staging dashboard data..."
git add dashboard/data/

if git diff --quiet --cached; then
  echo "No dashboard data changes to commit."
else
  git commit -m "data: daily dashboard update $(date -u +'%Y-%m-%d')"
  echo "Committed dashboard data."
fi

git push
echo "Deploy complete."
