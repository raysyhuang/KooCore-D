"""
Technical Scoring Functions

Pure functions that compute technical scores from OHLCV data.
Uses exact rubric for technical momentum scoring.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional
from .technicals import rsi, sma


def compute_technical_score_weekly(df: pd.DataFrame, ticker: str) -> dict:
    """
    Compute Technical Momentum Score (0-10) using OPTIMIZED rubric for Weekly Scanner.
    
    OPTIMIZED Jan 2026 based on backtest data:
    - RSI sweet spot refined: 55-65 gets +2.5, 50-70 gets +1.5, >70 gets -0.5
    - Volume tiering: ≥2.0x gets +2.5, ≥1.5x gets +2.0
    
    Points:
    +2.0 if within 5% of 52W high OR daily close breaks above resistance
    +2.0-2.5 if 3-day avg volume ≥ 1.5-2.0× 20-day avg volume
    +1.5-2.5 if RSI(14) in sweet spot [55, 65] or acceptable [50, 70]
    +2.0 if price > MA10, MA20, MA50 (all three)
    +2.0 if 5-day realized vol annualized ≥ 20%
    
    Args:
        df: DataFrame with OHLCV columns
        ticker: Ticker symbol (for logging/errors)
    
    Returns:
        dict with keys:
          - score: float (0-10)
          - cap_applied: Optional[float] (if data missing)
          - evidence: dict with all computed metrics
          - data_gaps: list[str] of missing data reasons
    """
    if df.empty or len(df) < 50:
        return {
            "score": 0.0,
            "cap_applied": 6.0,
            "evidence": {},
            "data_gaps": ["Insufficient price data"]
        }
    
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    
    last = float(close.iloc[-1])
    
    # Evidence dict
    evidence = {}
    data_gaps = []
    points = 0.0
    
    # 1. 52W high check (+2.0)
    try:
        high_52w = float(high.tail(252).max()) if len(high) >= 252 else float(high.max())
        dist_52w_pct = float((high_52w - last) / high_52w * 100) if high_52w else np.nan
        within_5pct = dist_52w_pct <= 5.0 if not np.isnan(dist_52w_pct) else False
        evidence["within_5pct_52w_high"] = within_5pct
        evidence["dist_to_52w_high_pct"] = round(dist_52w_pct, 2) if not np.isnan(dist_52w_pct) else None
        evidence["resistance_level"] = None  # Would need more complex logic
        if within_5pct:
            points += 2.0
    except Exception:
        data_gaps.append("52W high calculation failed")
        evidence["within_5pct_52w_high"] = None
    
    # 2. Volume ratio (OPTIMIZED: tiered scoring)
    # INSIGHT: Higher volume = stronger institutional confirmation
    try:
        if len(volume) >= 20:
            vol_3d = float(volume.tail(3).mean())
            vol_20d = float(volume.tail(20).mean())
            vol_ratio = vol_3d / vol_20d if vol_20d > 0 else np.nan
            evidence["volume_ratio_3d_to_20d"] = round(vol_ratio, 2) if not np.isnan(vol_ratio) else None
            if not np.isnan(vol_ratio):
                if vol_ratio >= 2.0:
                    # Institutional-grade volume confirmation
                    points += 2.5
                    evidence["volume_signal"] = "institutional_grade"
                elif vol_ratio >= 1.5:
                    # Standard volume confirmation
                    points += 2.0
                    evidence["volume_signal"] = "confirmed"
                elif vol_ratio < 1.0:
                    # Declining volume penalty
                    points -= 0.3
                    evidence["volume_signal"] = "declining"
                else:
                    evidence["volume_signal"] = "neutral"
        else:
            data_gaps.append("Insufficient data for volume ratio")
            evidence["volume_ratio_3d_to_20d"] = None
    except Exception:
        data_gaps.append("Volume ratio calculation failed")
        evidence["volume_ratio_3d_to_20d"] = None
    
    # 3. RSI(14) scoring (OPTIMIZED: tiered sweet spots)
    # INSIGHT: RSI 55-65 is the "sweet spot" for winners
    # INSIGHT: RSI > 70 (overbought) correlates with losers
    try:
        rsi14_series = rsi(close, 14)
        rsi14_val = float(rsi14_series.iloc[-1]) if len(rsi14_series) > 0 else np.nan
        evidence["rsi14"] = round(rsi14_val, 2) if not np.isnan(rsi14_val) else None
        if not np.isnan(rsi14_val):
            if 55 <= rsi14_val <= 65:
                # Sweet spot: strongest momentum zone
                points += 2.5
                evidence["rsi_zone"] = "sweet_spot"
            elif 50 <= rsi14_val <= 70:
                # Acceptable zone: still in momentum
                points += 1.5
                evidence["rsi_zone"] = "acceptable"
            elif rsi14_val > 70:
                # Overbought: penalty based on backtest losers
                points -= 0.5
                evidence["rsi_zone"] = "overbought"
            elif rsi14_val < 30:
                # Oversold: may reverse, neutral for momentum
                points += 0.0
                evidence["rsi_zone"] = "oversold"
            else:
                # Weak momentum (30-50)
                evidence["rsi_zone"] = "weak_momentum"
    except Exception:
        data_gaps.append("RSI calculation failed")
        evidence["rsi14"] = None
    
    # 4. Price above all MAs (+2.0)
    try:
        if len(close) >= 50:
            ma10 = float(sma(close, 10).iloc[-1])
            ma20 = float(sma(close, 20).iloc[-1])
            ma50 = float(sma(close, 50).iloc[-1])
            above_all = (last > ma10 and last > ma20 and last > ma50)
            evidence["above_ma10_ma20_ma50"] = above_all
            evidence["ma10"] = round(ma10, 2)
            evidence["ma20"] = round(ma20, 2)
            evidence["ma50"] = round(ma50, 2)
            if above_all:
                points += 2.0
        else:
            data_gaps.append("Insufficient data for MA calculation")
            evidence["above_ma10_ma20_ma50"] = None
    except Exception:
        data_gaps.append("MA calculation failed")
        evidence["above_ma10_ma20_ma50"] = None
    
    # 5. 5-day realized volatility ≥ 20% annualized (+2.0)
    try:
        if len(close) >= 6:
            returns_5d = close.tail(6).pct_change().dropna()
            std_5d = float(returns_5d.std())
            # Annualize: std * sqrt(252) where 252 is trading days per year
            vol_ann = std_5d * np.sqrt(252) * 100  # as percentage
            evidence["realized_vol_5d_ann_pct"] = round(vol_ann, 2) if not np.isnan(vol_ann) else None
            if not np.isnan(vol_ann) and vol_ann >= 20.0:
                points += 2.0
        else:
            data_gaps.append("Insufficient data for volatility calculation")
            evidence["realized_vol_5d_ann_pct"] = None
    except Exception:
        data_gaps.append("Volatility calculation failed")
        evidence["realized_vol_5d_ann_pct"] = None
    
    # Apply cap if data missing
    # Max possible: 2.0 (52W) + 2.5 (vol) + 2.5 (RSI) + 2.0 (MAs) + 2.0 (vol) = 11.0
    # Normalize to 10.0 scale
    score = min(10.0, max(0.0, points))
    cap_applied = None
    if data_gaps:
        cap_applied = 6.0
        score = min(score, cap_applied)
    
    return {
        "score": round(score, 2),
        "cap_applied": cap_applied,
        "evidence": evidence,
        "data_gaps": data_gaps
    }


def compute_score_30d_breakout(
    rvol_val: float,
    atr_pct_val: float,
    rsi14_val: float,
    dist_52w_high_pct: float,
    above_ma20: bool,
    above_ma50: bool,
) -> float:
    """
    Compute 30-Day Screener score for breakout setup.
    
    Formula: TapeScore + StructureScore + SetupBonus
    
    TapeScore = (RVOL * 2.0) + (ATR% * 1.4)
    StructureScore = rsi_structure + (dist_structure * 0.5) + ma_structure
    SetupBonus = 4.0 for breakouts
    
    Args:
        rvol_val: Relative volume
        atr_pct_val: ATR as percentage of price
        rsi14_val: RSI(14) value
        dist_52w_high_pct: Distance to 52W high in percentage
        above_ma20: Whether price is above MA20
        above_ma50: Whether price is above MA50
    
    Returns:
        Score (float)
    """
    # TapeScore
    tape_score = (rvol_val * 2.0) + (atr_pct_val * 1.4)
    
    # StructureScore
    rsi_structure = max(0.0, (70 - abs(rsi14_val - 62))) / 20.0  # prefer RSI ~ 58–66
    dist_structure = max(0.0, (100 - dist_52w_high_pct)) / 20.0
    ma_structure = (2.0 if above_ma20 and above_ma50 else (1.0 if above_ma20 else 0.0))
    structure_score = rsi_structure + (dist_structure * 0.5) + ma_structure
    
    # SetupBonus
    setup_bonus = 4.0
    
    return tape_score + structure_score + setup_bonus


def compute_score_30d_reversal(
    rvol_val: float,
    atr_pct_val: float,
    rsi14_val: float,
    dist_52w_high_pct: float,
) -> float:
    """
    Compute 30-Day Screener score for reversal setup.
    
    Formula: TapeScore + StructureScore + ReversalBonus
    
    TapeScore = (RVOL * 2.0) + (ATR% * 1.4)
    StructureScore = rsi_structure + (dist_structure * 0.5) + ma_structure
    ReversalBonus = 3.0 + reversal_structure
    
    Args:
        rvol_val: Relative volume
        atr_pct_val: ATR as percentage of price
        rsi14_val: RSI(14) value
        dist_52w_high_pct: Distance to 52W high in percentage
    
    Returns:
        Score (float)
    """
    # TapeScore
    tape_score = (rvol_val * 2.0) + (atr_pct_val * 1.4)
    
    # StructureScore
    rsi_structure = max(0.0, (45 - abs(rsi14_val - 28))) / 20.0  # prefer RSI ~ 20–35
    dist_structure = max(0.0, (100 - dist_52w_high_pct)) / 20.0
    # For reversals, MA structure is less important
    structure_score = rsi_structure + (dist_structure * 0.5)
    
    # ReversalBonus: more room to highs = better
    reversal_structure = min(6.0, dist_52w_high_pct / 8.0)
    setup_bonus = 3.0 + reversal_structure
    
    return tape_score + (structure_score * 0.7) + setup_bonus

