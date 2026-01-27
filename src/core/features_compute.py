# src/core/features_compute.py
"""
Feature computation from OHLCV DataFrames.

This module computes all technical features and returns a FeatureSet.
The FeatureSet can then be passed to scoring functions which do not
have access to the raw DataFrame (leak-proof scoring).
"""
from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

from .types import FeatureSet
from .technicals import rsi, sma, atr


def _safe_float(x) -> Optional[float]:
    """Safely convert to float, returning None for invalid values."""
    try:
        if x is None:
            return None
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except Exception:
        return None


def compute_features_weekly(df: pd.DataFrame, ticker: str, asof_date: str) -> FeatureSet:
    """
    Compute all features needed for weekly scoring.
    
    Args:
        df: OHLCV DataFrame with DatetimeIndex
        ticker: Ticker symbol
        asof_date: As-of date string (YYYY-MM-DD)
    
    Returns:
        FeatureSet with all computed features
    """
    # Return empty feature set if insufficient data
    if df is None or df.empty or len(df) < 20:
        return FeatureSet(
            ticker=ticker,
            asof_date=asof_date,
            extra={},
        )

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    # Basic price data
    last_close = _safe_float(close.iloc[-1])
    last_vol = _safe_float(vol.iloc[-1])

    # RSI
    rsi14_val = None
    if len(close) >= 15:
        rsi_series = rsi(close, 14)
        if len(rsi_series) > 0:
            rsi14_val = _safe_float(rsi_series.iloc[-1])

    # Moving averages
    ma10_val = _safe_float(sma(close, 10).iloc[-1]) if len(close) >= 10 else None
    ma20_val = _safe_float(sma(close, 20).iloc[-1]) if len(close) >= 20 else None
    ma50_val = _safe_float(sma(close, 50).iloc[-1]) if len(close) >= 50 else None

    # Volume ratio (3-day avg / 20-day avg)
    vol_ratio = None
    if len(vol) >= 20:
        v3 = _safe_float(vol.tail(3).mean())
        v20 = _safe_float(vol.tail(20).mean())
        if v3 is not None and v20 and v20 > 0:
            vol_ratio = _safe_float(v3 / v20)

    # Realized volatility (5-day annualized)
    realized_vol_5d = None
    if len(close) >= 6:
        rets = close.tail(6).pct_change().dropna()
        if len(rets) >= 3:
            std = _safe_float(rets.std())
            if std is not None:
                realized_vol_5d = _safe_float(std * np.sqrt(252) * 100.0)

    # ATR (atr function takes DataFrame, period)
    atr14_val = None
    atr_pct_val = None
    if len(df) >= 15:
        atr_series = atr(df, 14)
        if len(atr_series) > 0:
            atr14_val = _safe_float(atr_series.iloc[-1])
        if atr14_val is not None and last_close and last_close > 0:
            atr_pct_val = _safe_float((atr14_val / last_close) * 100.0)

    # Distance from 52-week high
    dist_52w = None
    if last_close is not None:
        h52 = _safe_float(high.tail(252).max()) if len(high) >= 252 else _safe_float(high.max())
        if h52 and h52 > 0:
            dist_52w = _safe_float((h52 - last_close) / h52 * 100.0)

    # Returns
    def _ret(n: int) -> Optional[float]:
        if len(close) >= (n + 1):
            a = _safe_float(close.iloc[-(n + 1)])
            if a and a > 0 and last_close is not None:
                return _safe_float((last_close / a - 1.0) * 100.0)
        return None

    return FeatureSet(
        ticker=ticker,
        asof_date=asof_date,
        last_close=last_close,
        last_volume=last_vol,
        rsi14=rsi14_val,
        ma10=ma10_val,
        ma20=ma20_val,
        ma50=ma50_val,
        vol_ratio_3_20=vol_ratio,
        realized_vol_5d_ann_pct=realized_vol_5d,
        atr14=atr14_val,
        atr_pct=atr_pct_val,
        dist_52w_high_pct=dist_52w,
        ret_5d=_ret(5),
        ret_10d=_ret(10),
        ret_20d=_ret(20),
        extra={},
    )
