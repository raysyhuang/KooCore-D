# src/core/breakout.py
"""
Breakout archetype scoring using shifted Donchian channels.

Key requirement: breakout level uses PREVIOUS-day Donchian high (shift(1))
to avoid same-bar lookahead bias.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from .technicals import atr as atr_func


@dataclass(frozen=True)
class BreakoutResult:
    """Result of breakout analysis."""
    breakout_score: float
    evidence: Dict[str, Any]


def compute_breakout_features(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute breakout-related features from OHLCV data.
    
    IMPORTANT: Donchian channels are shifted by 1 day to prevent
    same-bar lookahead (breakout level uses previous day's high).
    
    Args:
        df: OHLCV DataFrame with at least 21 rows
    
    Returns:
        Dict with breakout features
    """
    if df is None or df.empty or len(df) < 21:
        return {
            "donchian20_prev": np.nan,
            "donchian55_prev": np.nan,
            "atr14": np.nan,
            "vol_ratio_3_20": np.nan,
            "breakout_strength_atr": np.nan,
            "is_breakout_20": False,
        }
    
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    # ATR for normalization (atr_func takes DataFrame, period)
    atr_series = atr_func(df, 14)
    atr14_last = float(atr_series.iloc[-1]) if len(atr_series) else np.nan

    # Donchian channels - SHIFTED by 1 to use previous day's level
    don20 = high.rolling(20).max().shift(1)  # Previous day's 20-day high
    don55 = high.rolling(55).max().shift(1)  # Previous day's 55-day high

    don20_prev = float(don20.iloc[-1]) if len(don20) and not pd.isna(don20.iloc[-1]) else np.nan
    don55_prev = float(don55.iloc[-1]) if len(don55) and not pd.isna(don55.iloc[-1]) else np.nan

    last_close = float(close.iloc[-1])
    
    # Volume ratio
    vol_ratio = np.nan
    if len(vol) >= 20:
        v3 = float(vol.tail(3).mean())
        v20 = float(vol.tail(20).mean())
        if v20 > 0:
            vol_ratio = v3 / v20

    # Breakout strength in ATR units
    breakout_strength_atr = np.nan
    if atr14_last and not np.isnan(atr14_last) and atr14_last > 0 and not np.isnan(don20_prev):
        breakout_strength_atr = (last_close - don20_prev) / atr14_last

    # Is this a breakout?
    is_breakout_20 = (not np.isnan(don20_prev) and last_close > don20_prev)

    return {
        "donchian20_prev": don20_prev,
        "donchian55_prev": don55_prev,
        "atr14": atr14_last,
        "vol_ratio_3_20": vol_ratio,
        "breakout_strength_atr": breakout_strength_atr,
        "is_breakout_20": is_breakout_20,
    }


def score_breakout(df: pd.DataFrame) -> BreakoutResult:
    """
    Score a breakout setup.
    
    Scoring (max 10 points):
    - Trigger: Breaking 20-day Donchian high: +4.0
    - Confirmation: Volume ratio >= 1.5: +2.0
    - Strength: Breakout >= 0.5 ATR: +2.0
    - Strength: Breakout >= 1.0 ATR: +2.0 additional
    
    Args:
        df: OHLCV DataFrame
    
    Returns:
        BreakoutResult with score and evidence
    """
    f = compute_breakout_features(df)

    score = 0.0
    
    # Trigger: Breaking 20-day high (+4.0)
    if f["is_breakout_20"]:
        score += 4.0
    
    # Confirmation: Volume ratio >= 1.5 (+2.0)
    vr = f.get("vol_ratio_3_20")
    if vr is not None and not np.isnan(vr) and vr >= 1.5:
        score += 2.0
    
    # Strength in ATR units (+2.0 each tier)
    bs = f.get("breakout_strength_atr")
    if bs is not None and not np.isnan(bs):
        if bs >= 0.5:
            score += 2.0
        if bs >= 1.0:
            score += 2.0  # max +4 from strength

    score = float(min(10.0, score))
    
    # Round evidence values for cleaner output
    evidence = {}
    for k, v in f.items():
        if isinstance(v, float) and not np.isnan(v):
            evidence[k] = round(v, 4)
        else:
            evidence[k] = v
    
    return BreakoutResult(
        breakout_score=round(score, 2),
        evidence=evidence
    )
