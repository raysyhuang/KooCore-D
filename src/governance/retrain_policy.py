# src/governance/retrain_policy.py
"""
Retraining cadence rules and model versioning.

Determines when models should be retrained and manages version metadata.
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional
import subprocess
import os

from src.utils.time import utc_now, utc_now_iso_z


# Minimum new samples required to trigger retraining
MIN_NEW_SAMPLES_FOR_RETRAIN = 50

# Minimum days between retraining
MIN_DAYS_BETWEEN_RETRAIN = 7


def should_retrain(
    last_train_date: Optional[str],
    new_samples: int,
    min_samples: int = MIN_NEW_SAMPLES_FOR_RETRAIN,
    min_days: int = MIN_DAYS_BETWEEN_RETRAIN,
) -> Dict[str, Any]:
    """
    Determine if a model should be retrained.
    
    Args:
        last_train_date: ISO date of last training (e.g., "2026-01-15")
        new_samples: Number of new samples since last training
        min_samples: Minimum new samples required
        min_days: Minimum days since last training
    
    Returns:
        Dict with should_retrain flag and reason
    """
    if last_train_date is None:
        return {
            "should_retrain": True,
            "reason": "no_prior_training",
        }
    
    # Check sample threshold
    if new_samples < min_samples:
        return {
            "should_retrain": False,
            "reason": f"insufficient_new_samples ({new_samples} < {min_samples})",
        }
    
    # Check time threshold
    try:
        last = datetime.fromisoformat(last_train_date.replace("Z", "+00:00").split("T")[0])
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        days_since = (utc_now() - last).days
        
        if days_since < min_days:
            return {
                "should_retrain": False,
                "reason": f"too_recent ({days_since} days < {min_days} days)",
            }
    except (ValueError, AttributeError):
        # If date parsing fails, allow retraining
        pass
    
    return {
        "should_retrain": True,
        "reason": "cadence_met",
        "new_samples": new_samples,
    }


def get_git_commit_hash() -> Optional[str]:
    """
    Get current git commit hash for model versioning.
    
    Returns:
        Short git commit hash or None if not in a git repo
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def generate_model_version(
    regime: str,
    train_date: str,
    version_num: int = 1,
) -> str:
    """
    Generate a unique model version identifier.
    
    Args:
        regime: Model regime (global, bull, chop, stress)
        train_date: Training date (YYYY-MM-DD)
        version_num: Version number for same-day retraining
    
    Returns:
        Version string like "calibration_bull_2026-01-15_v1"
    """
    return f"calibration_{regime}_{train_date}_v{version_num}"


def build_version_metadata(
    regime: str,
    train_start: str,
    train_end: str,
    n_rows: int,
    n_positives: int,
    feature_cols: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Build comprehensive version metadata for a trained model.
    
    Args:
        regime: Model regime
        train_start: Training window start date
        train_end: Training window end date
        n_rows: Number of training samples
        n_positives: Number of positive outcomes
        feature_cols: List of feature columns used
    
    Returns:
        Version metadata dict
    """
    train_date = utc_now().strftime("%Y-%m-%d")
    git_commit = get_git_commit_hash()
    
    return {
        "model_version": generate_model_version(regime, train_date),
        "regime": regime,
        "trained_at": utc_now_iso_z(),
        "training_window": f"{train_start} → {train_end}",
        "train_start": train_start,
        "train_end": train_end,
        "rows": n_rows,
        "positives": n_positives,
        "positive_rate": round(n_positives / n_rows, 4) if n_rows > 0 else 0,
        "git_commit": git_commit,
        "feature_cols": feature_cols,
    }


def count_new_samples_since(
    snapshots_dir: str,
    last_train_date: str,
) -> int:
    """
    Count new snapshot samples since last training date.
    
    Args:
        snapshots_dir: Directory with snapshot parquet files
        last_train_date: Date of last training (YYYY-MM-DD)
    
    Returns:
        Number of new samples
    """
    import glob
    
    if not os.path.isdir(snapshots_dir):
        return 0
    
    # Find snapshot files
    files = glob.glob(os.path.join(snapshots_dir, "*.parquet"))
    
    count = 0
    try:
        import pandas as pd
        last_dt = datetime.fromisoformat(last_train_date)
        
        for f in files:
            # Check file modification time or parse date from filename
            try:
                df = pd.read_parquet(f)
                if "asof_date" in df.columns:
                    df["asof_date"] = pd.to_datetime(df["asof_date"])
                    count += len(df[df["asof_date"] > last_dt])
                else:
                    # Assume all rows are new if no date column
                    count += len(df)
            except Exception:
                continue
    except Exception:
        pass
    
    return count
