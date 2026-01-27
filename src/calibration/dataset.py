# src/calibration/dataset.py
"""
Calibration dataset builder.

Joins decision-time snapshots with realized outcomes for model training.
"""
from __future__ import annotations
import pandas as pd


TARGET_COL = "hit"


def build_dataset(
    snapshots: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build calibration dataset by joining snapshots and outcomes.
    
    Args:
        snapshots: Decision-time features (from write_decision_snapshot)
        outcomes: Realized outcomes (from write_outcomes)
    
    Returns:
        Joined DataFrame with features and target (hit)
    
    Join key: (ticker, asof_date)
    """
    if snapshots is None or snapshots.empty:
        return pd.DataFrame()
    if outcomes is None or outcomes.empty:
        return pd.DataFrame()
    
    # Select relevant outcome columns
    outcome_cols = ["ticker", "asof_date", TARGET_COL]
    optional_cols = ["mfe_pct", "mae_pct", "exit_reason", "days_to_hit"]
    for col in optional_cols:
        if col in outcomes.columns:
            outcome_cols.append(col)
    
    # Inner join on (ticker, asof_date)
    df = snapshots.merge(
        outcomes[outcome_cols],
        on=["ticker", "asof_date"],
        how="inner",
    )
    
    return df


def split_by_date(
    df: pd.DataFrame,
    cutoff_date: str,
    date_col: str = "asof_date",
) -> tuple:
    """
    Split dataset by date for time-series aware train/test split.
    
    Args:
        df: Full dataset
        cutoff_date: Dates before this go to train, after to test
        date_col: Column name for date
    
    Returns:
        (train_df, test_df) tuple
    """
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    cutoff = pd.to_datetime(cutoff_date)
    df[date_col] = pd.to_datetime(df[date_col])
    
    train = df[df[date_col] < cutoff]
    test = df[df[date_col] >= cutoff]
    
    return train, test
