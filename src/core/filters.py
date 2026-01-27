"""
Filtering Functions

Hard filters for liquidity, price, and exclusion criteria.
"""

from __future__ import annotations
import pandas as pd
from typing import Optional


def apply_hard_filters(df: pd.DataFrame, params: dict) -> tuple[bool, list[str]]:
    """
    Apply hard liquidity and exclusion filters.
    
    Args:
        df: DataFrame with OHLCV columns
        params: Dict with filter parameters:
            - price_min: Minimum price
            - avg_dollar_volume_20d_min: Minimum 20-day average dollar volume
            - price_up_5d_max_pct: Maximum 5-day return (exclusion threshold)
    
    Returns:
        Tuple of (passed: bool, reasons_if_failed: list[str])
    """
    if df.empty or len(df) < 20:
        return False, ["Insufficient data"]
    
    close = df["Close"]
    volume = df["Volume"]
    last = float(close.iloc[-1])
    
    reasons = []
    
    # Price filter
    price_min = params.get("price_min", 2.0)
    if last < price_min:
        reasons.append(f"Price ${last:.2f} < ${price_min:.2f}")
    
    # Dollar volume filter
    if len(close) >= 20 and len(volume) >= 20:
        adv20 = float((close.tail(20) * volume.tail(20)).mean())
        adv_min = params.get("avg_dollar_volume_20d_min", 50_000_000)
        if adv20 < adv_min:
            reasons.append(f"$ADV20 ${adv20:,.0f} < ${adv_min:,.0f}")
    else:
        reasons.append("Insufficient data for $ADV20")
    
    # 5-day return exclusion
    price_up_5d_max = params.get("price_up_5d_max_pct", 15.0)
    if len(close) >= 6:
        ret5d = float((last / float(close.iloc[-6]) - 1) * 100)
        if ret5d > price_up_5d_max:
            reasons.append(f"5d return {ret5d:.1f}% > {price_up_5d_max:.1f}% (avoiding chased names)")
    else:
        reasons.append("Insufficient data for 5d return check")
    
    return len(reasons) == 0, reasons

