"""
Data integrity validation framework for NBA Analytics pipeline.

Validates data at each pipeline stage:
  Fetch -> Preprocess -> Features -> Train -> Calibrate -> Predict

Usage:
    from src.validation.data_integrity import validate_stage
    results = validate_stage("fetch", season=202425)
    results = validate_stage("preprocess")
    results = validate_stage("all")

    # Or run directly:
    python -m src.validation.data_integrity
    python -m src.validation.data_integrity --strict
"""

from src.validation.data_integrity import (
    validate_stage,
    validate_fetch,
    validate_preprocess,
    validate_features,
    validate_train,
    validate_predict,
    ValidationResult,
    ValidationError,
)

__all__ = [
    "validate_stage",
    "validate_fetch",
    "validate_preprocess",
    "validate_features",
    "validate_train",
    "validate_predict",
    "ValidationResult",
    "ValidationError",
]
