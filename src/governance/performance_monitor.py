# src/governance/performance_monitor.py
"""
Rolling performance monitoring and decay detection for calibration models.

Tracks live model performance and triggers fallback when decay is detected.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import json
import os


@dataclass
class DecayThresholds:
    """Thresholds for detecting model performance decay."""
    hit_rate_ratio: float = 0.5      # Trigger if hit_rate < training_hit_rate * ratio
    mae_multiplier: float = 1.5      # Trigger if avg_mae > training_mae * multiplier
    min_expectancy: float = 0.0      # Trigger if expectancy < this value


def rolling_metrics(trades, window: int) -> Dict[str, float]:
    """
    Compute rolling performance metrics over recent trades.
    
    Args:
        trades: DataFrame with columns: hit, mfe_pct, mae_pct
        window: Number of recent trades to consider
    
    Returns:
        Dict with hit_rate, avg_mfe, avg_mae, expectancy
    """
    import pandas as pd
    
    if trades is None or (hasattr(trades, 'empty') and trades.empty):
        return {}
    
    # Take last N trades
    w = trades.tail(window)
    if w.empty or len(w) == 0:
        return {}
    
    # Ensure required columns exist
    if "hit" not in w.columns:
        return {}
    
    hit_rate = float(w["hit"].mean()) if "hit" in w.columns else 0.0
    avg_mfe = float(w["mfe_pct"].mean()) if "mfe_pct" in w.columns else 0.0
    avg_mae = float(w["mae_pct"].mean()) if "mae_pct" in w.columns else 0.0
    
    # Expectancy: E[return] = P(hit)*MFE + P(miss)*(-MAE)
    # Note: MAE is typically negative (loss), so we need to handle sign
    if "hit" in w.columns and "mfe_pct" in w.columns and "mae_pct" in w.columns:
        hits = w["hit"].astype(float)
        # Expectancy = avg(hit * mfe + (1-hit) * (-mae))
        expectancy = float((hits * w["mfe_pct"] - (1 - hits) * abs(w["mae_pct"])).mean())
    else:
        expectancy = 0.0
    
    return {
        "hit_rate": round(hit_rate, 4),
        "avg_mfe": round(avg_mfe, 4),
        "avg_mae": round(avg_mae, 4),
        "expectancy": round(expectancy, 4),
        "n_trades": len(w),
    }


def check_decay(
    live_metrics: Dict[str, float],
    training_metrics: Dict[str, float],
    thresholds: Optional[DecayThresholds] = None,
) -> Dict[str, Any]:
    """
    Check if model performance has decayed compared to training.
    
    Args:
        live_metrics: Recent rolling performance metrics
        training_metrics: Performance metrics from training period
        thresholds: Decay detection thresholds
    
    Returns:
        Dict with decay_detected flag and trigger reasons
    """
    if thresholds is None:
        thresholds = DecayThresholds()
    
    triggers = []
    
    # Check hit rate decay
    live_hit = live_metrics.get("hit_rate")
    train_hit = training_metrics.get("positive_rate") or training_metrics.get("hit_rate")
    
    if live_hit is not None and train_hit is not None and train_hit > 0:
        if live_hit < train_hit * thresholds.hit_rate_ratio:
            triggers.append(f"hit_rate_decay: {live_hit:.2%} < {train_hit * thresholds.hit_rate_ratio:.2%}")
    
    # Check MAE expansion
    live_mae = live_metrics.get("avg_mae")
    train_mae = training_metrics.get("avg_mae") or training_metrics.get("mae")
    
    if live_mae is not None and train_mae is not None and train_mae != 0:
        # MAE is typically negative, compare absolute values
        if abs(live_mae) > abs(train_mae) * thresholds.mae_multiplier:
            triggers.append(f"mae_expansion: {abs(live_mae):.2%} > {abs(train_mae) * thresholds.mae_multiplier:.2%}")
    
    # Check negative expectancy
    live_exp = live_metrics.get("expectancy")
    if live_exp is not None and live_exp < thresholds.min_expectancy:
        triggers.append(f"negative_expectancy: {live_exp:.4f} < {thresholds.min_expectancy}")
    
    return {
        "decay_detected": len(triggers) > 0,
        "triggers": triggers,
        "live_metrics": live_metrics,
        "training_metrics": training_metrics,
    }


def load_training_metrics(meta_path: str) -> Dict[str, float]:
    """
    Load training metrics from model metadata file.
    
    Args:
        meta_path: Path to model metadata JSON
    
    Returns:
        Dict with training metrics (positive_rate, brier_score, etc.)
    """
    if not os.path.exists(meta_path):
        return {}
    
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        return {
            "positive_rate": meta.get("positive_rate"),
            "brier_score": meta.get("brier_score"),
            "auc_roc": meta.get("auc_roc"),
            "rows": meta.get("rows"),
        }
    except Exception:
        return {}


def load_recent_trades(outcomes_dir: str, n_recent: int = 50):
    """
    Load recent trade outcomes for decay monitoring.
    
    Args:
        outcomes_dir: Directory containing outcome parquet files
        n_recent: Number of recent trades to load
    
    Returns:
        DataFrame with recent trades
    """
    import pandas as pd
    import glob
    
    if not os.path.isdir(outcomes_dir):
        return pd.DataFrame()
    
    # Find outcome files
    files = sorted(glob.glob(os.path.join(outcomes_dir, "*.parquet")))
    if not files:
        return pd.DataFrame()
    
    # Load and concat, take most recent
    dfs = []
    for f in files[-10:]:  # Load last 10 files max
        try:
            dfs.append(pd.read_parquet(f))
        except Exception:
            continue
    
    if not dfs:
        return pd.DataFrame()
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Sort by date and take recent
    if "asof_date" in df.columns:
        df = df.sort_values("asof_date", ascending=False)
    
    return df.head(n_recent)
