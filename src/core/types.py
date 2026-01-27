# src/core/types.py
"""
Core type definitions for the trading system.

FeatureSet is a frozen dataclass containing all computed features for a ticker,
designed to be passed to scoring functions without any DataFrame dependencies.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass(frozen=True)
class FeatureSet:
    """
    Immutable feature set for a single ticker at a point in time.
    
    This is the interface between feature computation (which uses DataFrames)
    and scoring (which uses only these values). This prevents lookahead leakage
    by ensuring scoring logic cannot access raw price data.
    """
    ticker: str
    asof_date: str

    # Price data
    last_close: Optional[float] = None
    last_volume: Optional[float] = None

    # Technical indicators
    rsi14: Optional[float] = None
    ma10: Optional[float] = None
    ma20: Optional[float] = None
    ma50: Optional[float] = None

    # Volume metrics
    vol_ratio_3_20: Optional[float] = None
    
    # Volatility metrics
    realized_vol_5d_ann_pct: Optional[float] = None
    atr14: Optional[float] = None
    atr_pct: Optional[float] = None

    # Price position
    dist_52w_high_pct: Optional[float] = None
    
    # Returns
    ret_5d: Optional[float] = None
    ret_10d: Optional[float] = None
    ret_20d: Optional[float] = None

    # Reserved for additional features (breakout, RS-line, etc.)
    extra: Dict[str, Any] = field(default_factory=dict)
