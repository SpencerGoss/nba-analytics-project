# Models: one workflow per task

This folder is now organized around **one model workflow per product task**.

## Core task models

1. **Game outcome prediction**
   - File: `src/models/game_outcome_model.py`
   - Goal: classify `home_win` for each matchup.
   - Inputs: `data/features/game_matchup_features.csv`
   - Artifacts: `models/artifacts/game_outcome_*`

2. **Player performance prediction**
   - File: `src/models/player_performance_model.py`
   - Goal: regress next-game `pts`, `reb`, `ast`.
   - Inputs: `data/features/player_game_features.csv`
   - Artifacts: `models/artifacts/player_{pts|reb|ast}_*`

3. **Playoff odds simulation**
   - File: `src/models/playoff_odds_model.py`
   - Goal: simulate playoff/play-in/title probabilities.
   - Inputs: standings + team logs
   - Output: `data/features/playoff_odds.csv`

## Supporting analysis (not separate task models)

- `backtesting.py` → walk-forward robustness checks
- `calibration.py` → probability calibration and ECE/Brier analysis
- `model_explainability.py` → SHAP/permutation importance reports
- `run_evaluation.py` → runs the analysis suite

## Consolidated training entrypoint

Run all core tasks in order:

```bash
python src/models/train_all_models.py
```

This script is intentionally explicit so it is easy to see what each task is
training and in what order.
