"""Daily JSON snapshot export from predictions_history.db.

Usage:
    from src.outputs.json_export import export_daily_snapshot
    path = export_daily_snapshot()   # exports today's predictions
    path = export_daily_snapshot("2026-03-01")  # exports specific date

Writes to data/outputs/predictions_YYYYMMDD.json.
FR-6.3, NFR-3.
"""
import sqlite3
import json
from datetime import date, datetime, timezone
from pathlib import Path

from src.outputs.prediction_store import STORE_PATH
import logging

log = logging.getLogger(__name__)

OUTPUT_DIR = "data/outputs"


def export_daily_snapshot(
    game_date: str | None = None,
    store_path: str = STORE_PATH,
    output_dir: str = OUTPUT_DIR,
) -> str:
    """Export all predictions for game_date to a JSON file.

    game_date: 'YYYY-MM-DD'. Defaults to today.
    Returns the path of the written file.
    """
    if game_date is None:
        game_date = date.today().isoformat()

    con = sqlite3.connect(store_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT * FROM game_predictions WHERE game_date = ? ORDER BY created_at",
        (game_date,),
    ).fetchall()
    con.close()

    records = [dict(r) for r in rows]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / f"predictions_{game_date.replace('-', '')}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "game_date": game_date,
                "count": len(records),
                "predictions": records,
            },
            f,
            indent=2,
        )

    log.info(f"  JSON snapshot -> {out_path}  ({len(records)} records)")
    return str(out_path)
