"""
Data integrity validation for every stage of the NBA Analytics pipeline.

Pipeline stages:
  1. Fetch      — raw CSVs per season per endpoint
  2. Preprocess — consolidated processed CSVs
  3. Features   — engineered feature CSVs
  4. Train      — model artifacts (.pkl, importances)
  5. Calibrate  — calibrated model artifacts
  6. Predict    — prediction output sanity

Each validator returns a list of ValidationResult objects (PASS / WARN / FAIL).
The validate_stage() dispatcher routes by name and supports --strict mode.

Usage:
    python -m src.validation.data_integrity            # all stages, warn mode
    python -m src.validation.data_integrity --strict    # all stages, raise on FAIL
    python -m src.validation.data_integrity --stage fetch --season 202425
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("nba.validation")

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(
    logging.Formatter("[%(levelname)s] %(message)s")
)
if not logger.handlers:
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class ValidationError(Exception):
    """Raised in --strict mode when a FAIL result is encountered."""


@dataclass(frozen=True)
class ValidationResult:
    """Immutable record of a single validation check."""

    stage: str
    check: str
    status: str          # "PASS", "WARN", "FAIL"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def log(self) -> None:
        """Emit the result to the validation logger."""
        level = {
            "PASS": logging.INFO,
            "WARN": logging.WARNING,
            "FAIL": logging.ERROR,
        }.get(self.status, logging.INFO)
        logger.log(level, "[%s] %s | %s — %s", self.status, self.stage, self.check, self.message)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project_root() -> str:
    """Return the project root directory (two levels up from this file)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _current_season() -> int:
    """Compute the current NBA season code (e.g. 202425).

    If month >= 10 (October), current season starts this calendar year.
    Otherwise, it started the previous calendar year.
    """
    now = datetime.now()
    start_year = now.year if now.month >= 10 else now.year - 1
    end_year = start_year + 1
    return int(f"{start_year}{end_year % 100:02d}")


def _safe_read_csv(path: str, **kwargs) -> Optional[pd.DataFrame]:
    """Read a CSV, returning None if the file is missing or unreadable."""
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Raw endpoint folders and their filename prefixes (matches preprocessing.py)
RAW_ENDPOINTS = {
    "player_stats":                  "player_stats_",
    "player_stats_advanced":         "player_stats_advanced_",
    "player_stats_clutch":           "player_stats_clutch_",
    "player_stats_scoring":          "player_stats_scoring_",
    "player_game_logs":              "player_game_logs_",
    "team_stats":                    "team_stats_",
    "team_stats_advanced":           "team_stats_advanced_",
    "team_game_logs":                "team_game_logs_",
    "standings":                     "standings_",
    "player_stats_playoffs":         "player_stats_playoffs_",
    "player_stats_advanced_playoffs": "player_stats_advanced_playoffs_",
    "player_game_logs_playoffs":     "player_game_logs_playoffs_",
    "team_stats_playoffs":           "team_stats_playoffs_",
    "team_stats_advanced_playoffs":  "team_stats_advanced_playoffs_",
    "team_game_logs_playoffs":       "team_game_logs_playoffs_",
    "player_hustle_stats":           "player_hustle_stats_",
    "team_hustle_stats":             "team_hustle_stats_",
    "player_bio_stats":              "player_bio_stats_",
}

# Endpoints that only exist for recent seasons
RECENT_ONLY_ENDPOINTS = {
    "player_stats_clutch":   200001,
    "player_stats_scoring":  200001,
    "player_hustle_stats":   201516,
    "team_hustle_stats":     201516,
    "player_bio_stats":      199697,
}

# Playoff endpoints only fetched after April
PLAYOFF_ENDPOINTS = {
    "player_stats_playoffs",
    "player_stats_advanced_playoffs",
    "player_game_logs_playoffs",
    "team_stats_playoffs",
    "team_stats_advanced_playoffs",
    "team_game_logs_playoffs",
}

# Processed files that contain game_id
GAME_ID_TABLES = [
    "player_game_logs",
    "team_game_logs",
    "player_game_logs_playoffs",
    "team_game_logs_playoffs",
]

# Processed files that contain game_date
GAME_DATE_TABLES = [
    "player_game_logs",
    "team_game_logs",
    "player_game_logs_playoffs",
    "team_game_logs_playoffs",
]

# All expected processed CSVs
EXPECTED_PROCESSED = [
    "players", "player_stats", "player_stats_advanced",
    "player_stats_clutch", "player_stats_scoring",
    "player_game_logs", "team_stats", "team_stats_advanced",
    "team_game_logs", "standings",
    "player_game_logs_playoffs", "team_game_logs_playoffs",
    "player_stats_playoffs", "player_stats_advanced_playoffs",
    "team_stats_playoffs", "team_stats_advanced_playoffs",
    "player_hustle_stats", "team_hustle_stats",
    "player_bio_stats", "teams",
]

# Expected feature columns in game_matchup_features.csv
EXPECTED_MATCHUP_FEATURE_COLS = [
    "game_id", "season", "game_date", "home_team", "away_team", "home_win",
    "home_days_rest", "home_cum_win_pct", "away_days_rest", "away_cum_win_pct",
    "home_win_pct_roll5", "home_win_pct_roll10", "home_win_pct_roll20",
    "away_win_pct_roll5", "away_win_pct_roll10", "away_win_pct_roll20",
    "home_pts_roll5", "home_pts_roll10", "home_pts_roll20",
    "diff_win_pct_roll10", "diff_cum_win_pct",
]

# Model artifacts expected after training
EXPECTED_MODEL_ARTIFACTS = [
    "game_outcome_model.pkl",
    "game_outcome_features.pkl",
    "game_outcome_importances.csv",
]

# Shooting percentage columns (should be in [0, 1])
SHOOTING_PCT_PATTERNS = ["fg_pct", "fg3_pct", "ft_pct"]

# Win percentage columns (should be in [0, 1])
WIN_PCT_PATTERNS = ["win_pct", "cum_win_pct"]

# Injury feature columns
INJURY_FEATURE_COLS = [
    "home_missing_minutes", "away_missing_minutes",
    "home_star_player_out", "away_star_player_out",
    "home_rotation_availability", "away_rotation_availability",
]


# ---------------------------------------------------------------------------
# Stage 1: validate_fetch
# ---------------------------------------------------------------------------

def validate_fetch(season: Optional[int] = None) -> List[ValidationResult]:
    """Validate raw CSVs after data fetch.

    Checks:
      - Every expected raw CSV exists for the target season
      - Each file has > 0 rows
      - No file suspiciously small (< 10 rows for a full season)

    Args:
        season: 6-digit season code (e.g. 202425). Defaults to current season.
    """
    root = _project_root()
    if season is None:
        season = _current_season()

    results: List[ValidationResult] = []
    now = datetime.now()

    for endpoint, prefix in RAW_ENDPOINTS.items():
        # Skip recent-only endpoints for old seasons
        min_season = RECENT_ONLY_ENDPOINTS.get(endpoint, 0)
        if season < min_season:
            continue

        # Skip playoff endpoints before April
        if endpoint in PLAYOFF_ENDPOINTS and now.month < 4:
            results.append(ValidationResult(
                stage="fetch",
                check=f"{endpoint}_exists",
                status="PASS",
                message=f"Skipped (playoff data not expected before April)",
                details={"endpoint": endpoint, "season": season},
            ))
            continue

        csv_path = os.path.join(root, "data", "raw", endpoint, f"{prefix}{season}.csv")
        if not os.path.exists(csv_path):
            results.append(ValidationResult(
                stage="fetch",
                check=f"{endpoint}_exists",
                status="FAIL",
                message=f"Missing raw CSV: {csv_path}",
                details={"endpoint": endpoint, "season": season, "path": csv_path},
            ))
            continue

        # Check row count
        try:
            df = pd.read_csv(csv_path)
            n_rows = len(df)
        except Exception as exc:
            results.append(ValidationResult(
                stage="fetch",
                check=f"{endpoint}_readable",
                status="FAIL",
                message=f"Cannot read CSV: {exc}",
                details={"endpoint": endpoint, "path": csv_path},
            ))
            continue

        if n_rows == 0:
            results.append(ValidationResult(
                stage="fetch",
                check=f"{endpoint}_nonempty",
                status="FAIL",
                message=f"CSV has 0 rows: {csv_path}",
                details={"endpoint": endpoint, "rows": 0},
            ))
        elif n_rows < 10:
            results.append(ValidationResult(
                stage="fetch",
                check=f"{endpoint}_row_count",
                status="WARN",
                message=f"Suspiciously few rows ({n_rows}) in {endpoint} for season {season}",
                details={"endpoint": endpoint, "rows": n_rows, "season": season},
            ))
        else:
            results.append(ValidationResult(
                stage="fetch",
                check=f"{endpoint}_ok",
                status="PASS",
                message=f"{endpoint}: {n_rows:,} rows",
                details={"endpoint": endpoint, "rows": n_rows},
            ))

    return results


# ---------------------------------------------------------------------------
# Stage 2: validate_preprocess
# ---------------------------------------------------------------------------

def validate_preprocess() -> List[ValidationResult]:
    """Validate processed CSVs after preprocessing.

    Checks:
      - All expected processed CSVs exist
      - game_id unique in game-level tables (per player or team)
      - No dates in the future
      - Season codes in expected range (194647 to current)
      - No column > 95% null
      - Row counts within a reasonable range
    """
    root = _project_root()
    results: List[ValidationResult] = []
    current = _current_season()

    for table in EXPECTED_PROCESSED:
        csv_path = os.path.join(root, "data", "processed", f"{table}.csv")

        if not os.path.exists(csv_path):
            results.append(ValidationResult(
                stage="preprocess",
                check=f"{table}_exists",
                status="WARN",
                message=f"Processed CSV missing: {table}.csv",
                details={"table": table, "path": csv_path},
            ))
            continue

        df = _safe_read_csv(csv_path)
        if df is None:
            results.append(ValidationResult(
                stage="preprocess",
                check=f"{table}_readable",
                status="FAIL",
                message=f"Cannot read {table}.csv",
            ))
            continue

        n_rows = len(df)

        # Row count sanity
        if n_rows == 0:
            results.append(ValidationResult(
                stage="preprocess",
                check=f"{table}_nonempty",
                status="FAIL",
                message=f"{table}.csv has 0 rows",
            ))
            continue
        else:
            results.append(ValidationResult(
                stage="preprocess",
                check=f"{table}_row_count",
                status="PASS",
                message=f"{table}: {n_rows:,} rows x {len(df.columns)} cols",
                details={"table": table, "rows": n_rows, "cols": len(df.columns)},
            ))

        # Game ID uniqueness (for team-game tables, game_id should be unique per team)
        if table in GAME_ID_TABLES and "game_id" in df.columns:
            if "player_id" in df.columns:
                # Player game logs: unique per (player_id, game_id)
                dup_mask = df.duplicated(subset=["player_id", "game_id"], keep=False)
            elif "team_id" in df.columns:
                # Team game logs: unique per (team_id, game_id)
                dup_mask = df.duplicated(subset=["team_id", "game_id"], keep=False)
            else:
                dup_mask = df.duplicated(subset=["game_id"], keep=False)

            n_dups = dup_mask.sum()
            if n_dups > 0:
                results.append(ValidationResult(
                    stage="preprocess",
                    check=f"{table}_game_id_unique",
                    status="WARN",
                    message=f"{table}: {n_dups:,} duplicate game_id rows",
                    details={"table": table, "duplicates": int(n_dups)},
                ))
            else:
                results.append(ValidationResult(
                    stage="preprocess",
                    check=f"{table}_game_id_unique",
                    status="PASS",
                    message=f"{table}: game_id unique (per entity)",
                ))

        # No future dates
        if table in GAME_DATE_TABLES and "game_date" in df.columns:
            try:
                dates = pd.to_datetime(df["game_date"], errors="coerce")
                future = dates > pd.Timestamp.now() + pd.Timedelta(days=1)
                n_future = future.sum()
                if n_future > 0:
                    results.append(ValidationResult(
                        stage="preprocess",
                        check=f"{table}_no_future_dates",
                        status="FAIL",
                        message=f"{table}: {n_future:,} rows have future game_date",
                        details={"table": table, "future_count": int(n_future)},
                    ))
                else:
                    results.append(ValidationResult(
                        stage="preprocess",
                        check=f"{table}_no_future_dates",
                        status="PASS",
                        message=f"{table}: no future dates",
                    ))
            except Exception:
                pass

        # Season range
        if "season" in df.columns:
            try:
                seasons = pd.to_numeric(df["season"], errors="coerce").dropna()
                if len(seasons) > 0:
                    s_min = int(seasons.min())
                    s_max = int(seasons.max())
                    out_of_range = (seasons < 194647) | (seasons > current)
                    n_bad = out_of_range.sum()
                    if n_bad > 0:
                        results.append(ValidationResult(
                            stage="preprocess",
                            check=f"{table}_season_range",
                            status="WARN",
                            message=f"{table}: {n_bad:,} rows with season outside 194647-{current} (range: {s_min}-{s_max})",
                            details={"table": table, "min": s_min, "max": s_max, "out_of_range": int(n_bad)},
                        ))
                    else:
                        results.append(ValidationResult(
                            stage="preprocess",
                            check=f"{table}_season_range",
                            status="PASS",
                            message=f"{table}: seasons {s_min}-{s_max}",
                        ))
            except Exception:
                pass

        # Column null rates
        null_rates = df.isnull().mean()
        high_null_cols = null_rates[null_rates > 0.95]
        if len(high_null_cols) > 0:
            col_list = high_null_cols.index.tolist()
            results.append(ValidationResult(
                stage="preprocess",
                check=f"{table}_null_columns",
                status="WARN",
                message=f"{table}: {len(col_list)} column(s) >95% null: {col_list[:5]}",
                details={"table": table, "columns": col_list},
            ))
        else:
            results.append(ValidationResult(
                stage="preprocess",
                check=f"{table}_null_columns",
                status="PASS",
                message=f"{table}: no columns >95% null",
            ))

    return results


# ---------------------------------------------------------------------------
# Stage 3: validate_features
# ---------------------------------------------------------------------------

def validate_features() -> List[ValidationResult]:
    """Validate engineered feature CSVs.

    Checks:
      - game_matchup_features.csv row count within 5% of team_game_logs.csv
      - All expected feature columns present
      - Value ranges: win% 0-1, no negative shooting%, days_rest >= 0
      - Injury features NOT all null
    """
    root = _project_root()
    results: List[ValidationResult] = []

    matchup_path = os.path.join(root, "data", "features", "game_matchup_features.csv")
    team_gl_path = os.path.join(root, "data", "processed", "team_game_logs.csv")
    player_feat_path = os.path.join(root, "data", "features", "player_game_features.csv")

    # --- game_matchup_features existence ---
    if not os.path.exists(matchup_path):
        results.append(ValidationResult(
            stage="features",
            check="matchup_exists",
            status="FAIL",
            message="game_matchup_features.csv missing",
        ))
        return results

    matchup_df = _safe_read_csv(matchup_path)
    if matchup_df is None:
        results.append(ValidationResult(
            stage="features",
            check="matchup_readable",
            status="FAIL",
            message="Cannot read game_matchup_features.csv",
        ))
        return results

    n_matchup = len(matchup_df)
    results.append(ValidationResult(
        stage="features",
        check="matchup_row_count",
        status="PASS",
        message=f"game_matchup_features: {n_matchup:,} rows x {len(matchup_df.columns)} cols",
        details={"rows": n_matchup, "cols": len(matchup_df.columns)},
    ))

    # --- Row count vs team_game_logs (each game has 2 team rows -> 1 matchup row) ---
    if os.path.exists(team_gl_path):
        team_gl_df = _safe_read_csv(team_gl_path)
        if team_gl_df is not None:
            expected_matchup_rows = len(team_gl_df) // 2
            if expected_matchup_rows > 0:
                ratio = n_matchup / expected_matchup_rows
                if abs(ratio - 1.0) > 0.05:
                    results.append(ValidationResult(
                        stage="features",
                        check="matchup_vs_gamelogs",
                        status="WARN",
                        message=(
                            f"Row count mismatch: matchup has {n_matchup:,} rows, "
                            f"expected ~{expected_matchup_rows:,} (team_game_logs / 2). "
                            f"Ratio: {ratio:.3f}"
                        ),
                        details={
                            "matchup_rows": n_matchup,
                            "expected": expected_matchup_rows,
                            "ratio": round(ratio, 4),
                        },
                    ))
                else:
                    results.append(ValidationResult(
                        stage="features",
                        check="matchup_vs_gamelogs",
                        status="PASS",
                        message=f"Matchup row count within 5% of expected ({ratio:.3f}x)",
                    ))

    # --- Expected columns ---
    missing_cols = [c for c in EXPECTED_MATCHUP_FEATURE_COLS if c not in matchup_df.columns]
    if missing_cols:
        results.append(ValidationResult(
            stage="features",
            check="matchup_expected_columns",
            status="FAIL",
            message=f"Missing expected columns: {missing_cols}",
            details={"missing": missing_cols},
        ))
    else:
        results.append(ValidationResult(
            stage="features",
            check="matchup_expected_columns",
            status="PASS",
            message=f"All {len(EXPECTED_MATCHUP_FEATURE_COLS)} expected columns present",
        ))

    # --- Value range checks ---
    # Win percentage columns in [0, 1] — skip diff_ columns (differentials can be negative)
    for pattern in WIN_PCT_PATTERNS:
        matching_cols = [
            c for c in matchup_df.columns
            if pattern in c and not c.startswith("diff_")
        ]
        for col in matching_cols:
            series = pd.to_numeric(matchup_df[col], errors="coerce").dropna()
            if len(series) == 0:
                continue
            out_of_range = (series < 0) | (series > 1)
            n_bad = out_of_range.sum()
            if n_bad > 0:
                results.append(ValidationResult(
                    stage="features",
                    check=f"matchup_{col}_range",
                    status="WARN",
                    message=f"{col}: {n_bad:,} values outside [0, 1]",
                    details={"column": col, "min": float(series.min()), "max": float(series.max())},
                ))

    # Shooting percentage columns: no negatives — skip diff_ columns (differentials can be negative)
    for pattern in SHOOTING_PCT_PATTERNS:
        matching_cols = [
            c for c in matchup_df.columns
            if pattern in c and not c.startswith("diff_")
        ]
        for col in matching_cols:
            series = pd.to_numeric(matchup_df[col], errors="coerce").dropna()
            if len(series) == 0:
                continue
            n_neg = (series < 0).sum()
            if n_neg > 0:
                results.append(ValidationResult(
                    stage="features",
                    check=f"matchup_{col}_no_negative",
                    status="WARN",
                    message=f"{col}: {n_neg:,} negative values",
                    details={"column": col, "min": float(series.min())},
                ))

    # days_rest >= 0
    for prefix in ["home_", "away_"]:
        col = f"{prefix}days_rest"
        if col in matchup_df.columns:
            series = pd.to_numeric(matchup_df[col], errors="coerce").dropna()
            n_neg = (series < 0).sum()
            if n_neg > 0:
                results.append(ValidationResult(
                    stage="features",
                    check=f"matchup_{col}_non_negative",
                    status="FAIL",
                    message=f"{col}: {n_neg:,} negative values (impossible)",
                    details={"column": col, "min": float(series.min())},
                ))
            else:
                results.append(ValidationResult(
                    stage="features",
                    check=f"matchup_{col}_non_negative",
                    status="PASS",
                    message=f"{col}: all values >= 0",
                ))

    # --- Injury features NOT all null ---
    injury_cols_present = [c for c in INJURY_FEATURE_COLS if c in matchup_df.columns]
    if not injury_cols_present:
        results.append(ValidationResult(
            stage="features",
            check="matchup_injury_columns_present",
            status="WARN",
            message="No injury feature columns found in matchup features",
            details={"expected": INJURY_FEATURE_COLS},
        ))
    else:
        all_null = all(matchup_df[c].isnull().all() for c in injury_cols_present)
        if all_null:
            results.append(ValidationResult(
                stage="features",
                check="matchup_injury_not_all_null",
                status="WARN",
                message=f"All injury feature columns are entirely null: {injury_cols_present}",
                details={"columns": injury_cols_present},
            ))
        else:
            # Report fill rates
            fill_rates = {c: f"{matchup_df[c].notna().mean():.1%}" for c in injury_cols_present}
            results.append(ValidationResult(
                stage="features",
                check="matchup_injury_not_all_null",
                status="PASS",
                message=f"Injury features have data: {fill_rates}",
                details={"fill_rates": fill_rates},
            ))

    # --- player_game_features existence ---
    if os.path.exists(player_feat_path):
        player_df = _safe_read_csv(player_feat_path, nrows=5)
        if player_df is not None:
            results.append(ValidationResult(
                stage="features",
                check="player_features_exists",
                status="PASS",
                message=f"player_game_features.csv exists ({len(player_df.columns)} cols)",
            ))
    else:
        results.append(ValidationResult(
            stage="features",
            check="player_features_exists",
            status="WARN",
            message="player_game_features.csv missing",
        ))

    return results


# ---------------------------------------------------------------------------
# Stage 4: validate_train
# ---------------------------------------------------------------------------

def validate_train() -> List[ValidationResult]:
    """Validate model artifacts after training.

    Checks:
      - Model .pkl files exist and are loadable via joblib
      - Feature importance CSV has correct column count
    """
    root = _project_root()
    results: List[ValidationResult] = []
    artifacts_dir = os.path.join(root, "models", "artifacts")

    if not os.path.isdir(artifacts_dir):
        results.append(ValidationResult(
            stage="train",
            check="artifacts_dir_exists",
            status="FAIL",
            message=f"Model artifacts directory missing: {artifacts_dir}",
        ))
        return results

    # Check expected model files
    for artifact in EXPECTED_MODEL_ARTIFACTS:
        artifact_path = os.path.join(artifacts_dir, artifact)
        if not os.path.exists(artifact_path):
            results.append(ValidationResult(
                stage="train",
                check=f"{artifact}_exists",
                status="FAIL",
                message=f"Missing artifact: {artifact}",
                details={"path": artifact_path},
            ))
            continue

        if artifact.endswith(".pkl"):
            # joblib is always available (sklearn dependency) and reads both
            # joblib.dump and pickle.dump serialized sklearn artifacts
            try:
                import joblib
                obj = joblib.load(artifact_path)
                results.append(ValidationResult(
                    stage="train",
                    check=f"{artifact}_loadable",
                    status="PASS",
                    message=f"{artifact}: loaded successfully ({type(obj).__name__})",
                    details={"type": type(obj).__name__},
                ))
            except Exception as exc:
                results.append(ValidationResult(
                    stage="train",
                    check=f"{artifact}_loadable",
                    status="FAIL",
                    message=f"{artifact}: failed to load — {exc}",
                ))

        elif artifact.endswith(".csv"):
            # Feature importances: check column structure
            imp_df = _safe_read_csv(artifact_path)
            if imp_df is not None:
                n_cols = len(imp_df.columns)
                n_rows = len(imp_df)
                if n_rows == 0:
                    results.append(ValidationResult(
                        stage="train",
                        check=f"{artifact}_nonempty",
                        status="FAIL",
                        message=f"{artifact}: 0 rows",
                    ))
                elif n_cols < 2:
                    results.append(ValidationResult(
                        stage="train",
                        check=f"{artifact}_columns",
                        status="WARN",
                        message=f"{artifact}: only {n_cols} column(s), expected at least 2 (feature + importance)",
                        details={"columns": imp_df.columns.tolist()},
                    ))
                else:
                    results.append(ValidationResult(
                        stage="train",
                        check=f"{artifact}_ok",
                        status="PASS",
                        message=f"{artifact}: {n_rows} features x {n_cols} cols",
                        details={"rows": n_rows, "cols": n_cols},
                    ))
            else:
                results.append(ValidationResult(
                    stage="train",
                    check=f"{artifact}_readable",
                    status="FAIL",
                    message=f"{artifact}: cannot read",
                ))

    # Check for additional model artifacts (player models, ATS, etc.)
    all_pkls = glob.glob(os.path.join(artifacts_dir, "*.pkl"))
    results.append(ValidationResult(
        stage="train",
        check="total_pkl_count",
        status="PASS",
        message=f"Total .pkl files in artifacts: {len(all_pkls)}",
        details={"files": [os.path.basename(f) for f in all_pkls]},
    ))

    return results


# ---------------------------------------------------------------------------
# Stage 5: validate_calibrate
# ---------------------------------------------------------------------------

def validate_calibrate() -> List[ValidationResult]:
    """Validate calibration artifacts.

    Checks:
      - Calibrated model file exists
      - Calibration report CSVs exist
    """
    root = _project_root()
    results: List[ValidationResult] = []

    cal_model = os.path.join(root, "models", "artifacts", "game_outcome_model_calibrated.pkl")
    if os.path.exists(cal_model):
        try:
            import joblib
            obj = joblib.load(cal_model)
            results.append(ValidationResult(
                stage="calibrate",
                check="calibrated_model_loadable",
                status="PASS",
                message=f"Calibrated model loaded: {type(obj).__name__}",
            ))
        except Exception as exc:
            results.append(ValidationResult(
                stage="calibrate",
                check="calibrated_model_loadable",
                status="FAIL",
                message=f"Calibrated model failed to load: {exc}",
            ))
    else:
        results.append(ValidationResult(
            stage="calibrate",
            check="calibrated_model_exists",
            status="WARN",
            message="No calibrated model found (game_outcome_model_calibrated.pkl)",
        ))

    # Check for calibration report directory
    cal_reports = os.path.join(root, "reports", "calibration")
    if os.path.isdir(cal_reports):
        report_files = os.listdir(cal_reports)
        results.append(ValidationResult(
            stage="calibrate",
            check="calibration_reports",
            status="PASS",
            message=f"Calibration reports directory: {len(report_files)} files",
            details={"files": report_files[:10]},
        ))
    else:
        results.append(ValidationResult(
            stage="calibrate",
            check="calibration_reports",
            status="WARN",
            message="No calibration reports directory found",
        ))

    return results


# ---------------------------------------------------------------------------
# Stage 6: validate_predict
# ---------------------------------------------------------------------------

def validate_predict(game_predictions: Optional[pd.DataFrame] = None) -> List[ValidationResult]:
    """Validate prediction output sanity.

    Checks:
      - All probabilities between 0.0 and 1.0
      - Home + away probabilities sum to ~1.0 (within 0.05 tolerance)

    Args:
        game_predictions: DataFrame with prediction columns. If None,
            returns a WARN indicating no predictions to validate.
    """
    results: List[ValidationResult] = []

    if game_predictions is None:
        results.append(ValidationResult(
            stage="predict",
            check="predictions_provided",
            status="WARN",
            message="No predictions DataFrame provided — skipping predict validation",
        ))
        return results

    df = game_predictions
    n_rows = len(df)

    if n_rows == 0:
        results.append(ValidationResult(
            stage="predict",
            check="predictions_nonempty",
            status="FAIL",
            message="Predictions DataFrame has 0 rows",
        ))
        return results

    results.append(ValidationResult(
        stage="predict",
        check="predictions_count",
        status="PASS",
        message=f"Predictions: {n_rows} rows",
    ))

    # Detect probability columns (common naming patterns)
    prob_cols = [c for c in df.columns if any(
        kw in c.lower() for kw in ["prob", "probability", "pred_prob", "confidence"]
    )]

    # Also check for home_win_prob / away_win_prob patterns
    for pattern in ["home_win_prob", "away_win_prob", "home_prob", "away_prob",
                     "win_probability", "predicted_probability"]:
        if pattern in df.columns and pattern not in prob_cols:
            prob_cols.append(pattern)

    if not prob_cols:
        results.append(ValidationResult(
            stage="predict",
            check="probability_columns_found",
            status="WARN",
            message=f"No probability columns detected. Columns: {df.columns.tolist()[:10]}",
        ))
        return results

    # Check all probabilities in [0, 1]
    for col in prob_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) == 0:
            continue
        below_zero = (series < 0).sum()
        above_one = (series > 1).sum()
        if below_zero > 0 or above_one > 0:
            results.append(ValidationResult(
                stage="predict",
                check=f"{col}_range",
                status="FAIL",
                message=f"{col}: {below_zero} values < 0, {above_one} values > 1 (range: {series.min():.4f} to {series.max():.4f})",
                details={"column": col, "min": float(series.min()), "max": float(series.max())},
            ))
        else:
            results.append(ValidationResult(
                stage="predict",
                check=f"{col}_range",
                status="PASS",
                message=f"{col}: all values in [0, 1]",
                details={"column": col, "min": float(series.min()), "max": float(series.max())},
            ))

    # Check home + away probabilities sum to ~1.0
    home_prob_col = next((c for c in prob_cols if "home" in c.lower()), None)
    away_prob_col = next((c for c in prob_cols if "away" in c.lower()), None)

    if home_prob_col and away_prob_col:
        home_p = pd.to_numeric(df[home_prob_col], errors="coerce")
        away_p = pd.to_numeric(df[away_prob_col], errors="coerce")
        prob_sum = home_p + away_p
        valid_sums = prob_sum.dropna()

        if len(valid_sums) > 0:
            bad_sums = ((valid_sums - 1.0).abs() > 0.05).sum()
            if bad_sums > 0:
                results.append(ValidationResult(
                    stage="predict",
                    check="prob_sum_to_one",
                    status="FAIL",
                    message=(
                        f"{home_prob_col} + {away_prob_col}: {bad_sums:,} rows "
                        f"where sum deviates from 1.0 by >0.05 "
                        f"(mean sum: {valid_sums.mean():.4f})"
                    ),
                    details={
                        "bad_count": int(bad_sums),
                        "mean_sum": float(valid_sums.mean()),
                        "min_sum": float(valid_sums.min()),
                        "max_sum": float(valid_sums.max()),
                    },
                ))
            else:
                results.append(ValidationResult(
                    stage="predict",
                    check="prob_sum_to_one",
                    status="PASS",
                    message=f"Home + Away probabilities sum to ~1.0 (mean: {valid_sums.mean():.4f})",
                ))

    return results


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

STAGE_VALIDATORS = {
    "fetch":      validate_fetch,
    "preprocess": validate_preprocess,
    "features":   validate_features,
    "train":      validate_train,
    "calibrate":  validate_calibrate,
    "predict":    validate_predict,
}


def validate_stage(
    stage_name: str,
    strict: bool = False,
    **kwargs,
) -> List[ValidationResult]:
    """Run validation for a single stage or all stages.

    Args:
        stage_name: One of "fetch", "preprocess", "features", "train",
                    "calibrate", "predict", or "all".
        strict:     If True, raise ValidationError on the first FAIL result.
        **kwargs:   Passed through to the stage validator (e.g. season=202425).

    Returns:
        List of ValidationResult objects.

    Raises:
        ValidationError: In strict mode, if any check has status "FAIL".
        ValueError: If stage_name is not recognized.
    """
    if stage_name == "all":
        stages = list(STAGE_VALIDATORS.keys())
    elif stage_name in STAGE_VALIDATORS:
        stages = [stage_name]
    else:
        raise ValueError(
            f"Unknown stage: '{stage_name}'. "
            f"Choose from: {', '.join(STAGE_VALIDATORS)} or 'all'."
        )

    all_results: List[ValidationResult] = []

    for stage in stages:
        validator = STAGE_VALIDATORS[stage]
        try:
            results = validator(**kwargs)
        except TypeError:
            # Stage validator doesn't accept these kwargs (e.g. predict without game_predictions)
            results = validator()

        for r in results:
            r.log()

        all_results.extend(results)

        if strict:
            fails = [r for r in results if r.status == "FAIL"]
            if fails:
                raise ValidationError(
                    f"Strict mode: {len(fails)} FAIL(s) in stage '{stage}'. "
                    f"First failure: {fails[0].check} — {fails[0].message}"
                )

    return all_results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(results: List[ValidationResult]) -> None:
    """Print a summary table of all validation results."""
    n_pass = sum(1 for r in results if r.status == "PASS")
    n_warn = sum(1 for r in results if r.status == "WARN")
    n_fail = sum(1 for r in results if r.status == "FAIL")
    total = len(results)

    print("\n" + "=" * 70)
    print(f"  VALIDATION SUMMARY: {n_pass} PASS | {n_warn} WARN | {n_fail} FAIL | {total} total")
    print("=" * 70)

    if n_fail > 0:
        print("\n  FAILURES:")
        for r in results:
            if r.status == "FAIL":
                print(f"    [{r.stage}] {r.check}: {r.message}")

    if n_warn > 0:
        print("\n  WARNINGS:")
        for r in results:
            if r.status == "WARN":
                print(f"    [{r.stage}] {r.check}: {r.message}")

    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for data integrity validation."""
    parser = argparse.ArgumentParser(
        description="NBA Analytics data integrity validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.validation.data_integrity                  # validate all stages\n"
            "  python -m src.validation.data_integrity --strict          # raise on FAIL\n"
            "  python -m src.validation.data_integrity --stage fetch --season 202425\n"
            "  python -m src.validation.data_integrity --stage preprocess\n"
        ),
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=list(STAGE_VALIDATORS.keys()) + ["all"],
        help="Pipeline stage to validate (default: all)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise ValidationError on FAIL (exit code 1)",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season code for fetch validation (e.g. 202425). Defaults to current season.",
    )

    args = parser.parse_args()

    kwargs = {}
    if args.season is not None:
        kwargs["season"] = args.season

    print(f"NBA Data Integrity Validator")
    print(f"Stage: {args.stage} | Strict: {args.strict}")
    if args.season:
        print(f"Season: {args.season}")
    print("-" * 70)

    try:
        results = validate_stage(args.stage, strict=args.strict, **kwargs)
        _print_summary(results)

        n_fail = sum(1 for r in results if r.status == "FAIL")
        sys.exit(1 if n_fail > 0 else 0)

    except ValidationError as exc:
        print(f"\nValidationError: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
