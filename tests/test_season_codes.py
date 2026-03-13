"""Ensure all season comparisons use integer, not string."""
import pathlib


MODEL_FILES = [
    "src/models/margin_model.py",
    "src/models/game_outcome_model.py",
]


def test_no_string_season_comparisons():
    """Season codes are 6-digit integers (e.g. 202425).

    String comparison works by coincidence for current codes but is
    semantically wrong. All season comparisons must use int.
    """
    violations = []
    for fpath in MODEL_FILES:
        p = pathlib.Path(fpath)
        if not p.exists():
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.split("#")[0]
            if ".astype(str)" in stripped and (">=" in stripped or "<=" in stripped):
                if "season" in stripped.lower():
                    violations.append(f"{fpath}:{i}: {line.strip()}")
    assert not violations, (
        "Found string-based season comparisons (should use int):\n"
        + "\n".join(violations)
    )
