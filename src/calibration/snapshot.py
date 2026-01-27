# src/calibration/snapshot.py
"""
Decision-time snapshot persistence.

Writes features and scores at decision time for later calibration training.
Critical: snapshots must contain ONLY data known at decision time (no lookahead).
"""
from __future__ import annotations
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime
import os

from src.utils.time import utc_now_iso_z

class PointInTimeViolation(Exception):
    """Raised when point-in-time integrity is violated."""
    pass


def validate_pit_integrity(row: Dict[str, Any]) -> None:
    """
    G3.1: Validate point-in-time integrity for a snapshot row.
    
    Ensures feature_timestamp (if present) <= asof_date.
    
    Args:
        row: Snapshot row dict
    
    Raises:
        PointInTimeViolation: If lookahead is detected
    """
    asof_date = row.get("asof_date")
    feature_timestamp = row.get("feature_timestamp")
    
    if asof_date is None:
        return
    
    if feature_timestamp is not None:
        try:
            # Parse dates for comparison
            if isinstance(asof_date, str):
                asof_dt = pd.to_datetime(asof_date)
            else:
                asof_dt = pd.to_datetime(asof_date)
            
            if isinstance(feature_timestamp, str):
                feat_dt = pd.to_datetime(feature_timestamp)
            else:
                feat_dt = pd.to_datetime(feature_timestamp)
            
            if feat_dt > asof_dt:
                raise PointInTimeViolation(
                    f"Feature timestamp {feature_timestamp} is after asof_date {asof_date}. "
                    "This indicates potential look-ahead bias."
                )
        except (ValueError, TypeError):
            # If parsing fails, skip validation
            pass


def write_decision_snapshot(
    rows: List[Dict[str, Any]],
    path: str,
    validate_pit: bool = True,
) -> None:
    """
    Write decision-time data for calibration training.
    
    Args:
        rows: List of dicts, one per ticker, containing strictly decision-time data
        path: Output path (parquet format)
        validate_pit: Whether to validate point-in-time integrity (default True)
    
    Note:
        Each row should include:
        - run_date, asof_date, ticker
        - Scores: technical_score, breakout_score, momentum_adj
        - Features: rsi14, atr_pct, vol_ratio_3_20, dist_52w_high_pct
        - Gates: event_blocked
        - Liquidity: adv_20
        - Regime: regime (for PR7)
    
    Raises:
        PointInTimeViolation: If any row fails PIT validation
    """
    if not rows:
        return
    
    # G3.1: Validate point-in-time integrity for each row
    if validate_pit:
        for row in rows:
            validate_pit_integrity(row)
    
    # Add snapshot timestamp for audit trail
    snapshot_ts = utc_now_iso_z()
    for row in rows:
        if "snapshot_timestamp" not in row:
            row["snapshot_timestamp"] = snapshot_ts
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)


def read_decision_snapshots(paths: List[str]) -> pd.DataFrame:
    """
    Read and concatenate multiple snapshot files.
    
    Args:
        paths: List of parquet file paths
    
    Returns:
        Combined DataFrame of all snapshots
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
