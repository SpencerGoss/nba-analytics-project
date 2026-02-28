# Models: one workflow per task

This folder is organized around **one model workflow per product task**.

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

## Added feature engineering signals

Player feature engineering now incorporates:
- Team/opponent context from `team_game_features` (defensive form, SOS, rest, injuries)
- Optional season priors from scoring and clutch tables when available
- Form/efficiency features (e.g., `*_form_delta`, `*_per_min_roll10`, role opportunity index)

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

Rebuild features before training:

```bash
python src/models/train_all_models.py --rebuild-features
```

## Prediction CLI

After training models, make predictions with:

```bash
python src/models/predict_cli.py game --home BOS --away LAL
python src/models/predict_cli.py player --name "LeBron James"
```
