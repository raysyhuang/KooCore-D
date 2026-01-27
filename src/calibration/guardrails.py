# src/calibration/guardrails.py
"""
Model eligibility and data sufficiency guardrails.

Ensures calibration models are only trained when sufficient data exists
to produce meaningful predictions.
"""
from __future__ import annotations
from typing import List, Optional, Dict, Any


# Minimum thresholds for model eligibility
MIN_SAMPLES_GLOBAL = 500
MIN_POSITIVES_GLOBAL = 50
MIN_SAMPLES_REGIME = 200
MIN_POSITIVES_REGIME = 20

# Feature completeness threshold (minimum non-null fraction)
FEATURE_COMPLETENESS_THRESHOLD = 0.7


def is_model_eligible(
    n_rows: int,
    n_pos: int,
    is_regime_model: bool = True,
) -> bool:
    """
    Check if a calibration model has sufficient data to be trained.
    
    Args:
        n_rows: Total number of training samples
        n_pos: Number of positive outcomes (hits)
        is_regime_model: If True, use regime thresholds; else global thresholds
    
    Returns:
        True if model meets eligibility requirements
    """
    if is_regime_model:
        min_samples = MIN_SAMPLES_REGIME
        min_positives = MIN_POSITIVES_REGIME
    else:
        min_samples = MIN_SAMPLES_GLOBAL
        min_positives = MIN_POSITIVES_GLOBAL
    
    return (n_rows >= min_samples) and (n_pos >= min_positives)


def check_eligibility_details(
    n_rows: int,
    n_pos: int,
    is_regime_model: bool = True,
) -> Dict[str, Any]:
    """
    Check eligibility and return detailed status.
    
    Args:
        n_rows: Total samples
        n_pos: Positive outcomes
        is_regime_model: If True, use regime thresholds
    
    Returns:
        Dict with eligibility status and details
    """
    if is_regime_model:
        min_samples = MIN_SAMPLES_REGIME
        min_positives = MIN_POSITIVES_REGIME
    else:
        min_samples = MIN_SAMPLES_GLOBAL
        min_positives = MIN_POSITIVES_GLOBAL
    
    samples_ok = n_rows >= min_samples
    positives_ok = n_pos >= min_positives
    eligible = samples_ok and positives_ok
    
    return {
        "eligible": eligible,
        "n_rows": n_rows,
        "n_positives": n_pos,
        "min_samples_required": min_samples,
        "min_positives_required": min_positives,
        "samples_sufficient": samples_ok,
        "positives_sufficient": positives_ok,
        "skip_reason": None if eligible else (
            f"insufficient_samples ({n_rows} < {min_samples})" if not samples_ok
            else f"insufficient_positives ({n_pos} < {min_positives})"
        ),
    }


def filter_incomplete_rows(
    df,
    feature_cols: List[str],
    threshold: float = FEATURE_COMPLETENESS_THRESHOLD,
):
    """
    Drop rows with too many missing features.
    
    Args:
        df: DataFrame with features
        feature_cols: List of feature column names
        threshold: Minimum fraction of non-null features required (default 0.7)
    
    Returns:
        Filtered DataFrame
    """
    import pandas as pd
    
    if df.empty:
        return df
    
    # Only consider feature columns that exist in the dataframe
    cols_present = [c for c in feature_cols if c in df.columns]
    if not cols_present:
        return df
    
    # Compute non-null count per row for feature columns
    min_non_null = int(len(cols_present) * threshold)
    
    # Filter rows with at least min_non_null non-null values in feature columns
    non_null_counts = df[cols_present].notna().sum(axis=1)
    mask = non_null_counts >= min_non_null
    
    return df[mask].copy()


def validate_no_future_features(columns: List[str]) -> None:
    """
    Ensure no columns contain forbidden keywords suggesting look-ahead.
    
    Args:
        columns: List of column names
    
    Raises:
        ValueError: If a forbidden feature name is found
    """
    forbidden_keywords = ["future", "forward", "next_"]
    
    for col in columns:
        col_lower = col.lower()
        for keyword in forbidden_keywords:
            if keyword in col_lower:
                raise ValueError(
                    f"Forbidden look-ahead feature detected: '{col}' "
                    f"(contains '{keyword}'). Remove this feature to prevent data leakage."
                )
