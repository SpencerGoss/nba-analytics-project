"""Sync NBA analytics CSV data and SQLite predictions into SQL Server 2019."""

import argparse
import os
import sqlite3
import sys
from pathlib import Path

try:
    import pyodbc
except ImportError:
    log.error("ERROR: pyodbc is not installed.")
    log.info("Install it with: pip install pyodbc")
    log.info("You also need ODBC Driver 17 for SQL Server installed.")
    sys.exit(1)

import pandas as pd
import logging

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_NAME = "nba_analytics"
ODBC_DRIVER = "ODBC Driver 17 for SQL Server"
SERVER = "localhost"

CSV_DIRS = {
    "processed": PROJECT_ROOT / "data" / "processed",
    "features": PROJECT_ROOT / "data" / "features",
    "odds": PROJECT_ROOT / "data" / "odds",
}

SQLITE_DB_PATH = PROJECT_ROOT / "database" / "predictions_history.db"
SQLITE_TABLES = ["game_predictions", "clv_tracking"]

DATE_COLUMNS = {"game_date", "date", "created_at", "logged_at", "updated_at"}

# SQL type mapping from pandas/numpy dtypes
DTYPE_MAP = {
    "int64": "BIGINT",
    "int32": "INT",
    "float64": "FLOAT",
    "float32": "REAL",
    "bool": "BIT",
    "object": "NVARCHAR(512)",
    "datetime64[ns]": "DATETIME2",
}


def _conn_string(database="master"):
    return (
        f"DRIVER={{{ODBC_DRIVER}}};"
        f"SERVER={SERVER};"
        f"DATABASE={database};"
        "Trusted_Connection=yes;"
    )


def _ensure_database(cursor):
    """Create nba_analytics database if it doesn't exist."""
    cursor.execute(
        "IF NOT EXISTS (SELECT 1 FROM sys.databases WHERE name = ?) "
        "BEGIN CREATE DATABASE [nba_analytics] END",
        (DB_NAME,),
    )
    cursor.commit()
    log.info(f"[OK] Database '{DB_NAME}' ensured.")


def _sql_type(dtype_str):
    return DTYPE_MAP.get(str(dtype_str), "NVARCHAR(512)")


def _table_name_from_path(csv_path, prefix):
    """Derive a table name like 'processed_player_stats' from file path."""
    stem = csv_path.stem  # e.g. player_stats
    return f"{prefix}_{stem}"


def _coerce_dates(df):
    """Convert columns that look like dates using format='mixed'."""
    for col in df.columns:
        if col.lower() in DATE_COLUMNS:
            try:
                df[col] = pd.to_datetime(df[col], format="mixed", errors="coerce")
            except Exception:
                pass
    return df


def _build_create_sql(table_name, df):
    """Build CREATE TABLE statement from DataFrame schema."""
    col_defs = []
    for col_name, dtype in zip(df.columns, df.dtypes):
        sql_type = _sql_type(dtype)
        safe_col = f"[{col_name}]"
        col_defs.append(f"  {safe_col} {sql_type} NULL")
    cols_sql = ",\n".join(col_defs)
    return (
        f"IF NOT EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.TABLES "
        f"WHERE TABLE_NAME = '{table_name}')\n"
        f"BEGIN\n"
        f"  CREATE TABLE [{table_name}] (\n{cols_sql}\n  )\n"
        f"END"
    )


def _clean_value(val):
    """Convert pandas NA/NaN/NaT to None; ensure native Python types for pyodbc."""
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    if hasattr(val, "item"):  # numpy scalar -> Python native
        return val.item()
    if pd.isna(val):
        return None
    return val


def _insert_dataframe(cursor, table_name, df):
    """Insert DataFrame rows using parameterized INSERT."""
    if df.empty:
        return 0
    cols = ", ".join(f"[{c}]" for c in df.columns)
    placeholders = ", ".join("?" for _ in df.columns)
    sql = f"INSERT INTO [{table_name}] ({cols}) VALUES ({placeholders})"

    rows = [
        tuple(_clean_value(v) for v in row)
        for row in df.itertuples(index=False, name=None)
    ]

    batch_size = 1000
    inserted = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        cursor.executemany(sql, batch)
        inserted += len(batch)
    return inserted


def _drop_table(cursor, table_name):
    cursor.execute(
        f"IF OBJECT_ID('[{table_name}]', 'U') IS NOT NULL "
        f"DROP TABLE [{table_name}]"
    )


def _truncate_table(cursor, table_name):
    cursor.execute(
        f"IF OBJECT_ID('[{table_name}]', 'U') IS NOT NULL "
        f"TRUNCATE TABLE [{table_name}]"
    )


def sync_csv_dir(cursor, dir_path, prefix, full_reload):
    """Load all CSVs from a directory into SQL Server tables."""
    if not dir_path.exists():
        log.warning(f"  [SKIP] Directory not found: {dir_path}")
        return
    csv_files = sorted(dir_path.glob("*.csv"))
    if not csv_files:
        log.warning(f"  [SKIP] No CSVs in {dir_path}")
        return

    for csv_path in csv_files:
        table_name = _table_name_from_path(csv_path, prefix)
        log.info(f"  Loading {csv_path.name} -> [{table_name}] ... ", end="", flush=True)
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception as exc:
            log.error(f"READ ERROR: {exc}")
            continue

        if df.empty:
            log.warning("empty file, skipped.")
            continue

        df = _coerce_dates(df)

        # Convert object columns with mixed types to strings to avoid
        # pyodbc type mismatches (e.g. column has both ints and strings)
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype(str).replace({"nan": None, "None": None, "": None})

        if full_reload:
            _drop_table(cursor, table_name)

        cursor.execute(_build_create_sql(table_name, df))
        _truncate_table(cursor, table_name)
        count = _insert_dataframe(cursor, table_name, df)
        cursor.commit()
        log.info(f"{count:,} rows.")


def sync_sqlite_tables(cursor, full_reload):
    """Sync SQLite tables into SQL Server."""
    if not SQLITE_DB_PATH.exists():
        log.warning(f"  [SKIP] SQLite DB not found: {SQLITE_DB_PATH}")
        return

    sqlite_conn = sqlite3.connect(str(SQLITE_DB_PATH))
    try:
        for src_table in SQLITE_TABLES:
            table_name = f"predictions_{src_table}"
            log.info(f"  Loading SQLite {src_table} -> [{table_name}] ... ", end="", flush=True)
            try:
                df = pd.read_sql_query(f"SELECT * FROM {src_table}", sqlite_conn)
            except Exception as exc:
                log.error(f"READ ERROR: {exc}")
                continue

            if df.empty:
                log.warning("empty table, skipped.")
                continue

            df = _coerce_dates(df)

            if full_reload:
                _drop_table(cursor, table_name)

            cursor.execute(_build_create_sql(table_name, df))
            _truncate_table(cursor, table_name)
            count = _insert_dataframe(cursor, table_name, df)
            cursor.commit()
            log.info(f"{count:,} rows.")
    finally:
        sqlite_conn.close()


def _create_views(cursor):
    """Create or replace useful analytical views."""
    views = {
        "vw_recent_predictions": """
            CREATE OR ALTER VIEW [vw_recent_predictions] AS
            SELECT
                game_date,
                home_team,
                away_team,
                home_win_prob,
                away_win_prob,
                model_name,
                actual_home_win,
                created_at
            FROM [predictions_game_predictions]
            WHERE game_date >= CONVERT(VARCHAR(10), DATEADD(DAY, -30, GETDATE()), 23)
        """,
        "vw_team_season_summary": """
            CREATE OR ALTER VIEW [vw_team_season_summary] AS
            SELECT
                [team_abbreviation],
                [season],
                COUNT(*) AS games_played,
                SUM(CASE WHEN [wl] = 'W' THEN 1 ELSE 0 END) AS wins,
                SUM(CASE WHEN [wl] = 'L' THEN 1 ELSE 0 END) AS losses,
                CAST(SUM(CASE WHEN [wl] = 'W' THEN 1.0 ELSE 0 END)
                    / NULLIF(COUNT(*), 0) AS DECIMAL(5, 3)) AS win_pct,
                AVG(CAST([pts] AS FLOAT)) AS avg_pts
            FROM [processed_team_game_logs]
            GROUP BY [team_abbreviation], [season]
        """,
        "vw_model_accuracy": """
            CREATE OR ALTER VIEW [vw_model_accuracy] AS
            SELECT
                model_name,
                CAST(game_date AS DATE) AS prediction_date,
                COUNT(*) AS total_predictions,
                SUM(CASE
                    WHEN actual_home_win IS NOT NULL
                         AND ((home_win_prob > 0.5 AND actual_home_win = 1)
                           OR (home_win_prob <= 0.5 AND actual_home_win = 0))
                    THEN 1 ELSE 0 END) AS correct,
                SUM(CASE WHEN actual_home_win IS NOT NULL THEN 1 ELSE 0 END)
                    AS graded
            FROM [predictions_game_predictions]
            GROUP BY model_name, CAST(game_date AS DATE)
        """,
        "vw_value_bet_history": """
            CREATE OR ALTER VIEW [vw_value_bet_history] AS
            SELECT
                m.[date] AS game_date,
                m.home_team,
                m.away_team,
                m.stat,
                m.model_projection,
                m.sportsbook_line,
                m.gap,
                m.flagged,
                m.player_name,
                c.opening_spread,
                c.closing_spread,
                c.clv
            FROM [odds_model_vs_odds] m
            LEFT JOIN [predictions_clv_tracking] c
                ON m.[date] = c.game_date
               AND m.home_team = c.home_team
        """,
    }

    for name, ddl in views.items():
        try:
            cursor.execute(ddl)
            cursor.commit()
            log.info(f"  [OK] View {name}")
        except pyodbc.ProgrammingError as exc:
            # View may reference a table that doesn't exist yet
            log.warning(f"  [WARN] View {name}: {exc.args[-1][:80]}")


def main():
    parser = argparse.ArgumentParser(
        description="Sync NBA analytics data to SQL Server 2019."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full reload: drop and recreate all tables (default: truncate+insert).",
    )
    args = parser.parse_args()

    mode = "FULL RELOAD" if args.full else "INCREMENTAL"
    log.info(f"=== NBA Analytics -> SQL Server sync ({mode}) ===\n")

    # Step 1: Connect to master and ensure database exists
    log.info("[1/5] Ensuring database exists ...")
    master_conn = pyodbc.connect(_conn_string("master"), autocommit=True)
    _ensure_database(master_conn.cursor())
    master_conn.close()

    # Step 2: Connect to nba_analytics
    conn = pyodbc.connect(_conn_string(DB_NAME), autocommit=False)
    cursor = conn.cursor()

    # Step 3: Sync CSV directories
    log.info("\n[2/5] Syncing CSV data ...")
    for prefix, dir_path in CSV_DIRS.items():
        log.info(f"\n  --- {prefix}/ ---")
        sync_csv_dir(cursor, dir_path, prefix, args.full)

    # Step 4: Sync SQLite tables
    log.info("\n[3/5] Syncing SQLite predictions ...")
    sync_sqlite_tables(cursor, args.full)

    # Step 5: Create views
    log.info("\n[4/5] Creating views ...")
    _create_views(cursor)

    # Done
    log.info("\n[5/5] Sync complete.")
    cursor.close()
    conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
