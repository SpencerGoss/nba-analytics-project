Retrain and evaluate an NBA prediction model.

1. Invoke the `nba-model-evaluation` skill for NBA-specific evaluation criteria (Brier score is PRIMARY gate)
2. Use `machine-learning-ops` plugin agents for ML pipeline management and experiment tracking
3. Use `scientific-skills:scikit-learn` for model implementation
4. Use `scientific-skills:statsmodels` for calibration analysis
5. After training, run the model comparison workflow if a baseline exists
6. Document results in PROJECT_JOURNAL.md

Arguments: $ARGUMENTS (model name, feature changes, or "full retrain")
