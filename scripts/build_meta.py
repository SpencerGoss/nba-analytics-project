"""
scripts/build_meta.py -- write dashboard/data/meta.json

Writes pipeline metadata consumed by the dashboard header:
  - exported_at: current UTC timestamp (shown as "Updated <date>")
  - db_exists:   whether predictions_history.db is present
  - model_version: from models/artifacts/game_outcome_metadata.json
  - season:      current season label
  - sample_data: False (always real data in production)

Run: python scripts/build_meta.py
Output: dashboard/data/meta.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "database" / "predictions_history.db"
METADATA_PATH = PROJECT_ROOT / "models" / "artifacts" / "game_outcome_metadata.json"
OUT_JSON = PROJECT_ROOT / "dashboard" / "data" / "meta.json"

from src.config import get_current_season

_s = str(get_current_season())
CURRENT_SEASON = f"{_s[:4]}-{_s[4:]}"
DEFAULT_MODEL_VERSION = "gradient_boosting_v2.3"


def _load_model_version() -> str:
    """Read model_version from game_outcome_metadata.json if available."""
    try:
        if METADATA_PATH.exists():
            data = json.loads(METADATA_PATH.read_text())
            return data.get("model_version", DEFAULT_MODEL_VERSION)
    except Exception:
        pass
    return DEFAULT_MODEL_VERSION


def build_meta() -> dict:
    return {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "db_exists": DB_PATH.exists(),
        "model_version": _load_model_version(),
        "season": CURRENT_SEASON,
        "sample_data": False,
    }


def main() -> None:
    meta = build_meta()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(meta, indent=2))
    print(f"meta.json written -> {meta['exported_at']}")


if __name__ == "__main__":
    main()
