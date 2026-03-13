"""Ensure no model inference path uses fillna(0) on feature DataFrames."""
import pathlib

MODEL_FILES = [
    "src/models/margin_model.py",
    "src/models/game_outcome_model.py",
    "src/models/value_bet_detector.py",
    "src/models/model_explainability.py",
    "src/models/player_performance_model.py",
    "src/models/playoff_odds_model.py",
]

def test_no_fillna_zero_in_model_inference():
    """No model file should use .fillna(0) on feature DataFrames."""
    violations = []
    for fpath in MODEL_FILES:
        p = pathlib.Path(fpath)
        if not p.exists():
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.split("#")[0]  # Ignore comments
            if ".fillna(0)" in stripped:
                violations.append(f"{fpath}:{i}: {line.strip()}")
    assert not violations, (
        "Found fillna(0) in prediction paths (should use pipeline imputer):\n"
        + "\n".join(violations)
    )
