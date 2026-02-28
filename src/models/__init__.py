"""Model package for NBA analytics.

This package is organized around **three core prediction tasks**:

1. Game outcome classification
   - Module: ``src.models.game_outcome_model``
   - Entry function: ``train_game_outcome_model``
2. Player stat regression (PTS/REB/AST)
   - Module: ``src.models.player_performance_model``
   - Entry function: ``train_player_models``
3. Playoff odds simulation
   - Module: ``src.models.playoff_odds_model``
   - Entry function: ``simulate_playoff_odds``

Supporting analysis modules:
- ``backtesting.py``
- ``calibration.py``
- ``model_explainability.py``
- ``run_evaluation.py``
"""

from src.models.game_outcome_model import train_game_outcome_model, predict_game
from src.models.player_performance_model import train_player_models, predict_player_next_game
from src.models.playoff_odds_model import simulate_playoff_odds

TASK_MODELS = {
    "game_outcome": {
        "task": "Predict home-team win/loss for each NBA matchup",
        "module": "src.models.game_outcome_model",
        "trainer": "train_game_outcome_model",
    },
    "player_performance": {
        "task": "Predict next-game player points/rebounds/assists",
        "module": "src.models.player_performance_model",
        "trainer": "train_player_models",
    },
    "playoff_odds": {
        "task": "Simulate playoff and title probabilities",
        "module": "src.models.playoff_odds_model",
        "trainer": "simulate_playoff_odds",
    },
}

__all__ = [
    "TASK_MODELS",
    "train_game_outcome_model",
    "predict_game",
    "train_player_models",
    "predict_player_next_game",
    "simulate_playoff_odds",
]
