#!/usr/bin/env python3
"""
Daily Movers Queue Manager

Manages a cooling period queue for daily movers before they enter the scanner.
"""

from __future__ import annotations
import os
import pandas as pd
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from ...utils.time import utc_now


def get_queue_file_path() -> Path:
    """Get path to mover queue file."""
    data_dir = Path("data") / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "mover_queue.csv"


def load_mover_queue() -> pd.DataFrame:
    """Load existing mover queue from disk."""
    queue_file = get_queue_file_path()
    
    if not queue_file.exists():
        # Return empty DataFrame with correct schema (strings for dates)
        return pd.DataFrame(columns=[
            "ticker", "mover_type", "first_seen_date_utc", "last_seen_date_utc",
            "cooling_days_required", "eligible_date_utc", "status", "notes"
        ])
    
    try:
        # Don't parse dates - keep as strings to avoid tz issues when updating
        df = pd.read_csv(queue_file)
        return df
    except Exception:
        return pd.DataFrame(columns=[
            "ticker", "mover_type", "first_seen_date_utc", "last_seen_date_utc",
            "cooling_days_required", "eligible_date_utc", "status", "notes"
        ])


def save_mover_queue(queue_df: pd.DataFrame) -> None:
    """Save mover queue to disk."""
    queue_file = get_queue_file_path()
    queue_df.to_csv(queue_file, index=False)


def _update_queue_statuses(
    queue_df: pd.DataFrame,
    asof_date_utc: datetime,
    config: dict,
) -> pd.DataFrame:
    """Update statuses: COOLING -> ELIGIBLE, expire old entries."""
    if queue_df.empty:
        return queue_df
    
    # Normalize asof_date to tz-naive for comparison
    if hasattr(asof_date_utc, 'tzinfo') and asof_date_utc.tzinfo is not None:
        asof_naive = asof_date_utc.replace(tzinfo=None)
    else:
        asof_naive = asof_date_utc
    
    # Convert to datetime if strings (tz-naive for comparison)
    # Use format='ISO8601' to handle both "T" separator and space separator
    for col in ["eligible_date_utc", "first_seen_date_utc"]:
        if col in queue_df.columns:
            queue_df[col] = pd.to_datetime(queue_df[col], format='ISO8601', utc=True).dt.tz_localize(None)
    
    # Update COOLING -> ELIGIBLE
    mask_cooling = queue_df["status"] == "COOLING"
    mask_eligible = queue_df["eligible_date_utc"] <= asof_naive
    queue_df.loc[mask_cooling & mask_eligible, "status"] = "ELIGIBLE"
    
    # Expire old entries
    max_age = timedelta(days=config.get("max_age_days", 5))
    cutoff_date = asof_naive - max_age
    mask_old = queue_df["first_seen_date_utc"] < cutoff_date
    queue_df.loc[mask_old, "status"] = "EXPIRED"
    
    return queue_df


def update_mover_queue(
    candidates_df: pd.DataFrame,
    asof_date_utc: datetime,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Update mover queue with new candidates.
    
    Args:
        candidates_df: DataFrame from mover_filters (must have filter_pass==True)
        asof_date_utc: Current date/time (UTC)
        config: Configuration dict
    
    Returns:
        Updated queue DataFrame
    """
    if config is None:
        config = {
            "cooling_days_required": 1,
            "max_age_days": 5,
        }
    
    # Load existing queue
    queue_df = load_mover_queue()
    
    # Filter to only passed candidates
    passed = candidates_df[candidates_df["filter_pass"] == True].copy()
    
    if passed.empty:
        # Still update statuses of existing entries
        return _update_queue_statuses(queue_df, asof_date_utc, config)
    
    # Update or add entries
    for _, row in passed.iterrows():
        ticker = row["ticker"]
        mover_type = row["mover_type"]
        
        # Check if already in queue
        existing = queue_df[queue_df["ticker"] == ticker]
        
        # Convert to ISO string for storage (avoids timezone compatibility issues)
        asof_str = asof_date_utc.isoformat() if hasattr(asof_date_utc, 'isoformat') else str(asof_date_utc)
        eligible_date = asof_date_utc + timedelta(days=config["cooling_days_required"])
        eligible_str = eligible_date.isoformat() if hasattr(eligible_date, 'isoformat') else str(eligible_date)
        
        if existing.empty:
            # New entry
            new_entry = {
                "ticker": ticker,
                "mover_type": mover_type,
                "first_seen_date_utc": asof_str,
                "last_seen_date_utc": asof_str,
                "cooling_days_required": config["cooling_days_required"],
                "eligible_date_utc": eligible_str,
                "status": "COOLING",
                "notes": ""
            }
            queue_df = pd.concat([queue_df, pd.DataFrame([new_entry])], ignore_index=True)
        else:
            # Update existing entry
            idx = existing.index[0]
            queue_df.loc[idx, "last_seen_date_utc"] = asof_str
            # Recompute eligible date if still cooling
            if queue_df.loc[idx, "status"] == "COOLING":
                queue_df.loc[idx, "eligible_date_utc"] = eligible_str
    
    # Update statuses (COOLING -> ELIGIBLE, expire old entries)
    queue_df = _update_queue_statuses(queue_df, asof_date_utc, config)
    
    # Save
    save_mover_queue(queue_df)
    
    return queue_df


def get_eligible_movers(
    queue_df: Optional[pd.DataFrame] = None,
    asof_date_utc: Optional[datetime] = None,
) -> list[str]:
    """
    Get list of tickers that are eligible (cooling period passed).
    
    Args:
        queue_df: Optional queue DataFrame (loads from disk if None)
        asof_date_utc: Optional current date (defaults to now)
    
    Returns:
        List of ticker symbols
    """
    if queue_df is None:
        queue_df = load_mover_queue()
    
    if queue_df.empty:
        return []
    
    if asof_date_utc is None:
        asof_date_utc = utc_now()
    
    # Update statuses before querying
    config = {"cooling_days_required": 1, "max_age_days": 5}
    queue_df = _update_queue_statuses(queue_df, asof_date_utc, config)
    
    # Get eligible tickers
    eligible = queue_df[queue_df["status"] == "ELIGIBLE"]["ticker"].tolist()
    
    return eligible

