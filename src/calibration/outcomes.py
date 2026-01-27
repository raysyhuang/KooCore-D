# src/calibration/outcomes.py
"""
Realized outcome persistence for calibration training.

Stores actual trade outcomes after the horizon period.
"""
from __future__ import annotations
import pandas as pd
from typing import List
import os


def write_outcomes(
    trades_df: pd.DataFrame,
    path: str,
) -> None:
    """
    Write realized outcomes for calibration training.
    
    Args:
        trades_df: DataFrame from backtest with columns:
            ticker, asof_date, hit, mfe_pct, mae_pct, exit_reason
        path: Output path (parquet format)
    """
    if trades_df is None or trades_df.empty:
        return
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    
    trades_df.to_parquet(path, index=False)


def read_outcomes(paths: List[str]) -> pd.DataFrame:
    """
    Read and concatenate multiple outcome files.
    
    Args:
        paths: List of parquet file paths
    
    Returns:
        Combined DataFrame of all outcomes
    """
    dfs = []
    for p in paths:
        if os.path.exists(p):
            try:
                dfs.append(pd.read_parquet(p))
            except Exception:
                continue
    
    if not dfs:
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True)
