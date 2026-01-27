# src/regime/classifier.py
"""
Market regime classification.

Simple, stable classifier using SPY and VIX to determine:
- bull: SPY above MA50 and VIX < 20
- stress: VIX >= 25 or SPY below MA50 with downtrend
- chop: everything else

This enables regime-specific model selection and sizing adjustments.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Regime:
    """
    Market regime classification.
    
    Attributes:
        name: Regime name ("bull", "chop", "stress")
        confidence: Classification confidence (0-1)
        evidence: Supporting data for the classification
    """
    name: str  # {"bull", "chop", "stress"}
    confidence: float = 1.0
    evidence: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.evidence is None:
            object.__setattr__(self, 'evidence', {})


def classify_regime(
    spy_df: Optional[pd.DataFrame],
    vix_level: Optional[float],
) -> Regime:
    """
    Classify current market regime.
    
    Rules:
    - bull: SPY above MA50 AND VIX < 20
    - stress: VIX >= 25 OR (SPY below MA50 AND VIX >= 20)
    - chop: everything else
    
    Args:
        spy_df: SPY OHLCV DataFrame (needs at least 50 days)
        vix_level: Current VIX value
    
    Returns:
        Regime with name, confidence, and evidence
    """
    evidence: Dict[str, Any] = {}
    
    if spy_df is None or spy_df.empty:
        return Regime("chop", confidence=0.5, evidence={"reason": "no_spy_data"})

    spy_df = spy_df.sort_index()
    close = spy_df["Close"]
    
    # Compute MA50
    ma50 = close.rolling(50).mean()
    if close.empty or ma50.dropna().empty:
        logger.warning("Regime classifier: insufficient SPY history for MA50")
        return Regime("chop", confidence=0.5, evidence={"reason": "insufficient_history"})
    
    last_close = float(close.iloc[-1])
    last_ma50 = float(ma50.iloc[-1])
    above_ma = last_close > last_ma50
    
    evidence["spy_close"] = round(last_close, 2)
    evidence["spy_ma50"] = round(last_ma50, 2)
    evidence["above_ma50"] = above_ma
    evidence["vix"] = vix_level
    
    # Compute trend (20-day return)
    if len(close) >= 20:
        ret_20d = (last_close / float(close.iloc[-20]) - 1.0) * 100
        evidence["spy_ret_20d_pct"] = round(ret_20d, 2)
    else:
        ret_20d = 0.0
    
    vix = float(vix_level) if vix_level is not None else None

    # Classification logic
    if vix is not None and vix >= 25:
        return Regime("stress", confidence=0.9, evidence=evidence)
    
    if above_ma and (vix is None or vix < 20):
        confidence = 0.85 if (vix is not None and vix < 15) else 0.75
        return Regime("bull", confidence=confidence, evidence=evidence)
    
    if (not above_ma) and (vix is not None and vix >= 20):
        return Regime("stress", confidence=0.8, evidence=evidence)
    
    # Default to chop
    return Regime("chop", confidence=0.7, evidence=evidence)


def fetch_regime_data(asof_date: Optional[str] = None) -> tuple:
    """
    Fetch SPY and VIX data for regime classification.
    
    Args:
        asof_date: Optional as-of date (YYYY-MM-DD)
    
    Returns:
        (spy_df, vix_level) tuple
    """
    import yfinance as yf
    
    try:
        # Fetch SPY for 60 days of history
        spy = yf.Ticker("SPY")
        spy_df = spy.history(period="3mo")
        
        if asof_date:
            asof_dt = pd.to_datetime(asof_date)
            if getattr(spy_df.index, "tz", None) is not None and asof_dt.tzinfo is None:
                asof_dt = asof_dt.tz_localize(spy_df.index.tz)
            spy_df = spy_df[spy_df.index <= asof_dt]
        
        # Fetch VIX level
        vix = yf.Ticker("^VIX")
        vix_df = vix.history(period="5d")
        
        if asof_date:
            vix_asof = asof_dt
            if getattr(vix_df.index, "tz", None) is not None and vix_asof.tzinfo is None:
                vix_asof = vix_asof.tz_localize(vix_df.index.tz)
            vix_df = vix_df[vix_df.index <= vix_asof]
        
        vix_level = float(vix_df["Close"].iloc[-1]) if not vix_df.empty else None
        
        return spy_df, vix_level
    except Exception as e:
        logger.warning(f"Regime data fetch failed: {e}")
        return pd.DataFrame(), None


def get_current_regime(asof_date: Optional[str] = None) -> Regime:
    """
    Get current market regime with data fetch.
    
    Args:
        asof_date: Optional as-of date
    
    Returns:
        Classified Regime
    """
    spy_df, vix_level = fetch_regime_data(asof_date)
    return classify_regime(spy_df, vix_level)
