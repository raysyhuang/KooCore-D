# src/backtest/metrics.py
"""
Backtest outcome metrics including MAE/MFE computation.

MAE (Maximum Adverse Excursion): Worst drawdown during trade
MFE (Maximum Favorable Excursion): Best unrealized gain during trade

These metrics help understand trade management quality:
- High MFE with low hit rate = poor exit timing
- High MAE with losses = poor stop placement
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np


@dataclass(frozen=True)
class PathMetrics:
    """
    Metrics describing the price path during a trade.
    
    Attributes:
        mfe: Maximum Favorable Excursion (%)
        mae: Maximum Adverse Excursion (%)
        days_to_hit: Trading days to reach target (None if not hit)
        hit: Whether target was achieved
        exit_reason: Why trade ended ("target_hit", "stop_hit", "timeout", "no_data")
    """
    mfe: Optional[float]      # max favorable excursion (%)
    mae: Optional[float]      # max adverse excursion (%)
    days_to_hit: Optional[int]
    hit: bool
    exit_reason: str


def compute_path_metrics(
    df: pd.DataFrame,
    entry_px: float,
    start_dt: pd.Timestamp,
    horizon_days: int,
    target_pct: float = 10.0,
    stop_pct: Optional[float] = None,
) -> PathMetrics:
    """
    Compute MAE/MFE using High/Low after entry.
    
    Uses intraday High/Low to measure excursions, which is more realistic
    than using only Close prices.
    
    Args:
        df: OHLCV DataFrame with DatetimeIndex
        entry_px: Entry price (after slippage/fees)
        start_dt: Entry timestamp
        horizon_days: Maximum holding period in trading days
        target_pct: Target return percentage (default 10%)
        stop_pct: Stop loss percentage (None = no stop)
    
    Returns:
        PathMetrics with MFE, MAE, days_to_hit, hit flag, and exit_reason
    """
    if df is None or df.empty or entry_px is None or entry_px <= 0:
        return PathMetrics(None, None, None, False, "no_data")

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        except Exception:
            return PathMetrics(None, None, None, False, "no_data")
    
    df = df.sort_index()
    
    # Slice forward window strictly AFTER entry time
    fwd = df[df.index > start_dt].head(horizon_days)
    if fwd.empty:
        return PathMetrics(None, None, None, False, "no_forward_window")

    # Compute excursions using High/Low
    highs = (fwd["High"] / entry_px - 1.0) * 100.0
    lows = (fwd["Low"] / entry_px - 1.0) * 100.0

    mfe = float(highs.max()) if not highs.empty else 0.0
    mae = float(lows.min()) if not lows.empty else 0.0

    # Check if target hit (using High)
    hit_mask = highs >= target_pct
    if hit_mask.any():
        hit_idx = highs[hit_mask].index[0]
        days = int((hit_idx - start_dt).days)
        return PathMetrics(mfe, mae, days, True, "target_hit")

    # Check if stop hit (using Low)
    if stop_pct is not None and stop_pct > 0:
        stop_mask = lows <= -abs(stop_pct)
        if stop_mask.any():
            stop_idx = lows[stop_mask].index[0]
            days = int((stop_idx - start_dt).days)
            return PathMetrics(mfe, mae, days, False, "stop_hit")

    # Timeout - neither target nor stop hit within horizon
    return PathMetrics(mfe, mae, None, False, "timeout")


def compute_expectancy(
    outcomes: list[dict],
    target_pct: float = 10.0,
) -> dict:
    """
    Compute aggregate metrics from a list of trade outcomes.
    
    Args:
        outcomes: List of dicts with 'hit', 'mfe_pct', 'mae_pct' keys
        target_pct: Target return for expectancy calculation
    
    Returns:
        Dict with hit_rate, avg_mfe, avg_mae, expectancy
    """
    if not outcomes:
        return {
            "hit_rate": None,
            "avg_mfe_pct": None,
            "avg_mae_pct": None,
            "expectancy_pct": None,
            "trade_count": 0,
        }
    
    hits = [o for o in outcomes if o.get("hit")]
    mfes = [o["mfe_pct"] for o in outcomes if o.get("mfe_pct") is not None]
    maes = [o["mae_pct"] for o in outcomes if o.get("mae_pct") is not None]
    
    hit_rate = len(hits) / len(outcomes) if outcomes else 0.0
    avg_mfe = np.mean(mfes) if mfes else None
    avg_mae = np.mean(maes) if maes else None
    
    # Simple expectancy: hit_rate * avg_win - (1 - hit_rate) * avg_loss
    if avg_mfe is not None and avg_mae is not None and hit_rate > 0:
        avg_win = min(avg_mfe, target_pct)  # Cap at target
        avg_loss = abs(avg_mae)
        expectancy = hit_rate * avg_win - (1 - hit_rate) * avg_loss
    else:
        expectancy = None
    
    return {
        "hit_rate": round(hit_rate, 4) if hit_rate is not None else None,
        "avg_mfe_pct": round(avg_mfe, 2) if avg_mfe is not None else None,
        "avg_mae_pct": round(avg_mae, 2) if avg_mae is not None else None,
        "expectancy_pct": round(expectancy, 2) if expectancy is not None else None,
        "trade_count": len(outcomes),
    }
