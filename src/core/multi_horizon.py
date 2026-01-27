"""
Multi-Horizon Signal Aggregator

Computes signals across 3 timeframes to reduce false positives.
Only recommend trades when at least 2/3 timeframes agree.

Timeframes:
- Short (5-day): Momentum, RVOL, RSI(5)
- Medium (20-day): Trend direction, price vs MA20, RSI(14)
- Long (60-day): Major trend, 52W position, MA50

This addresses the problem of single-timeframe blindness where
a stock may look bullish on one timeframe but bearish on another.
"""

from __future__ import annotations
import logging
from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from .technicals import rsi, sma, ema

logger = logging.getLogger(__name__)


class SignalDirection(Enum):
    """Direction of signal."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class HorizonSignal:
    """Signal for a single timeframe."""
    horizon: str  # "short", "medium", "long"
    direction: SignalDirection
    strength: float  # 0-1
    components: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "horizon": self.horizon,
            "direction": self.direction.value,
            "strength": self.strength,
            "components": self.components,
        }


@dataclass
class MultiHorizonResult:
    """Combined multi-horizon analysis."""
    ticker: str
    short_signal: HorizonSignal
    medium_signal: HorizonSignal
    long_signal: HorizonSignal
    
    # Aggregated results
    consensus: SignalDirection
    agreement_count: int  # 0, 1, 2, or 3
    confidence: float  # 0-1
    trade_recommendation: str  # "STRONG_BUY", "BUY", "HOLD", "AVOID"
    conflict_note: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "short_signal": self.short_signal.to_dict(),
            "medium_signal": self.medium_signal.to_dict(),
            "long_signal": self.long_signal.to_dict(),
            "consensus": self.consensus.value,
            "agreement_count": self.agreement_count,
            "confidence": self.confidence,
            "trade_recommendation": self.trade_recommendation,
            "conflict_note": self.conflict_note,
        }


def compute_short_horizon(df: pd.DataFrame, lookback: int = 5) -> HorizonSignal:
    """
    Short-term signal (5-day).
    
    Components:
    - 5-day momentum (price change)
    - RSI(5) position
    - Volume trend (last 5 days vs prior 5 days)
    
    Bullish: Strong recent momentum, RSI healthy (not oversold), rising volume
    Bearish: Negative momentum, weak RSI, declining volume
    """
    if len(df) < 20:
        return HorizonSignal(
            horizon="short",
            direction=SignalDirection.NEUTRAL,
            strength=0.0,
            components={"error": "Insufficient data (need 20+ rows)"}
        )
    
    close = df["Close"]
    volume = df["Volume"]
    
    # 5-day momentum
    try:
        momentum_5d = (float(close.iloc[-1]) / float(close.iloc[-lookback]) - 1) * 100
    except (IndexError, ZeroDivisionError):
        momentum_5d = 0
    
    # RSI(5)
    try:
        rsi_5_series = rsi(close, 5)
        rsi_5 = float(rsi_5_series.iloc[-1]) if len(rsi_5_series) > 0 and not np.isnan(rsi_5_series.iloc[-1]) else 50
    except Exception:
        rsi_5 = 50
    
    # Volume trend (last 5 days vs prior 5 days)
    try:
        vol_recent = float(volume.tail(5).mean())
        vol_prior = float(volume.iloc[-10:-5].mean()) if len(volume) >= 10 else vol_recent
        vol_trend = vol_recent / vol_prior if vol_prior > 0 else 1.0
    except Exception:
        vol_trend = 1.0
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SCORING
    # ═══════════════════════════════════════════════════════════════════════════
    bullish_points = 0.0
    bearish_points = 0.0
    
    # Momentum scoring
    if momentum_5d > 5:
        bullish_points += 2.5
    elif momentum_5d > 2:
        bullish_points += 1.5
    elif momentum_5d > 0:
        bullish_points += 0.5
    elif momentum_5d < -5:
        bearish_points += 2.5
    elif momentum_5d < -2:
        bearish_points += 1.5
    elif momentum_5d < 0:
        bearish_points += 0.5
    
    # RSI scoring
    if 50 <= rsi_5 <= 70:
        bullish_points += 1.0  # Healthy momentum zone
    elif rsi_5 > 70:
        bullish_points += 0.5  # Overbought but still bullish
        bearish_points += 0.25  # Slight caution
    elif 30 <= rsi_5 < 50:
        bearish_points += 0.5
    elif rsi_5 < 30:
        bearish_points += 0.5  # Oversold - potential bounce but currently weak
    
    # Volume confirmation
    if vol_trend > 1.3 and momentum_5d > 0:
        bullish_points += 1.0  # Volume confirming upward momentum
    elif vol_trend > 1.3 and momentum_5d < 0:
        bearish_points += 1.0  # Volume confirming downward momentum
    elif vol_trend < 0.7 and momentum_5d > 0:
        bullish_points += 0.25  # Low conviction rally
    
    # Determine direction
    total_points = bullish_points + bearish_points
    if total_points == 0:
        direction = SignalDirection.NEUTRAL
        strength = 0.3
    elif bullish_points > bearish_points + 0.5:
        direction = SignalDirection.BULLISH
        strength = min(1.0, bullish_points / 4.5)
    elif bearish_points > bullish_points + 0.5:
        direction = SignalDirection.BEARISH
        strength = min(1.0, bearish_points / 4.5)
    else:
        direction = SignalDirection.NEUTRAL
        strength = 0.3
    
    return HorizonSignal(
        horizon="short",
        direction=direction,
        strength=round(strength, 2),
        components={
            "momentum_5d_pct": round(momentum_5d, 2),
            "rsi_5": round(rsi_5, 1),
            "volume_trend": round(vol_trend, 2),
            "bullish_points": round(bullish_points, 2),
            "bearish_points": round(bearish_points, 2),
        }
    )


def compute_medium_horizon(df: pd.DataFrame, lookback: int = 20) -> HorizonSignal:
    """
    Medium-term signal (20-day).
    
    Components:
    - Price vs MA20 (trend position)
    - MA20 slope (trend direction)
    - RSI(14) position
    - 20-day momentum
    
    Bullish: Above rising MA20, healthy RSI, positive momentum
    Bearish: Below falling MA20, weak RSI, negative momentum
    """
    if len(df) < 50:
        return HorizonSignal(
            horizon="medium",
            direction=SignalDirection.NEUTRAL,
            strength=0.0,
            components={"error": "Insufficient data (need 50+ rows)"}
        )
    
    close = df["Close"]
    current = float(close.iloc[-1])
    
    # Price vs MA20
    try:
        ma20_series = sma(close, 20)
        ma20 = float(ma20_series.iloc[-1])
        price_vs_ma20 = (current / ma20 - 1) * 100 if ma20 > 0 else 0
    except Exception:
        ma20 = current
        price_vs_ma20 = 0
    
    # MA20 slope (5-day change in MA20)
    try:
        ma20_now = float(ma20_series.iloc[-1])
        ma20_5d_ago = float(ma20_series.iloc[-6]) if len(ma20_series) >= 6 else ma20_now
        ma20_slope = (ma20_now / ma20_5d_ago - 1) * 100 if ma20_5d_ago > 0 else 0
    except Exception:
        ma20_slope = 0
    
    # RSI(14)
    try:
        rsi_14_series = rsi(close, 14)
        rsi_14 = float(rsi_14_series.iloc[-1]) if len(rsi_14_series) > 0 and not np.isnan(rsi_14_series.iloc[-1]) else 50
    except Exception:
        rsi_14 = 50
    
    # 20-day momentum
    try:
        momentum_20d = (current / float(close.iloc[-lookback]) - 1) * 100
    except (IndexError, ZeroDivisionError):
        momentum_20d = 0
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SCORING
    # ═══════════════════════════════════════════════════════════════════════════
    bullish_points = 0.0
    bearish_points = 0.0
    
    # Price vs MA20
    if price_vs_ma20 > 3:
        bullish_points += 2.0
    elif price_vs_ma20 > 0:
        bullish_points += 1.0
    elif price_vs_ma20 < -3:
        bearish_points += 2.0
    elif price_vs_ma20 < 0:
        bearish_points += 1.0
    
    # MA20 slope
    if ma20_slope > 0.5:
        bullish_points += 1.0
    elif ma20_slope > 0:
        bullish_points += 0.5
    elif ma20_slope < -0.5:
        bearish_points += 1.0
    elif ma20_slope < 0:
        bearish_points += 0.5
    
    # RSI(14)
    if 50 <= rsi_14 <= 65:
        bullish_points += 1.0
    elif rsi_14 > 70:
        bullish_points += 0.5
        bearish_points += 0.5  # Overbought caution
    elif 35 <= rsi_14 < 50:
        bearish_points += 0.5
    elif rsi_14 < 30:
        bearish_points += 1.0  # Weak
    
    # 20-day momentum
    if momentum_20d > 10:
        bullish_points += 1.0
    elif momentum_20d > 5:
        bullish_points += 0.5
    elif momentum_20d < -10:
        bearish_points += 1.0
    elif momentum_20d < -5:
        bearish_points += 0.5
    
    # Determine direction
    total_points = bullish_points + bearish_points
    if total_points == 0:
        direction = SignalDirection.NEUTRAL
        strength = 0.3
    elif bullish_points > bearish_points + 0.5:
        direction = SignalDirection.BULLISH
        strength = min(1.0, bullish_points / 5.0)
    elif bearish_points > bullish_points + 0.5:
        direction = SignalDirection.BEARISH
        strength = min(1.0, bearish_points / 5.0)
    else:
        direction = SignalDirection.NEUTRAL
        strength = 0.3
    
    return HorizonSignal(
        horizon="medium",
        direction=direction,
        strength=round(strength, 2),
        components={
            "price_vs_ma20_pct": round(price_vs_ma20, 2),
            "ma20_slope_pct": round(ma20_slope, 2),
            "rsi_14": round(rsi_14, 1),
            "momentum_20d_pct": round(momentum_20d, 2),
            "bullish_points": round(bullish_points, 2),
            "bearish_points": round(bearish_points, 2),
        }
    )


def compute_long_horizon(df: pd.DataFrame, lookback: int = 60) -> HorizonSignal:
    """
    Long-term signal (60-day).
    
    Components:
    - Price vs MA50
    - Position in 52-week range
    - 60-day momentum
    - Distance to 52W high/low
    
    Bullish: Above MA50, near 52W highs, strong long-term momentum
    Bearish: Below MA50, near 52W lows, weak long-term momentum
    """
    if len(df) < 252:
        # Try with available data
        if len(df) < 60:
            return HorizonSignal(
                horizon="long",
                direction=SignalDirection.NEUTRAL,
                strength=0.0,
                components={"error": "Insufficient data for long horizon (need 60+ rows)"}
            )
    
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    current = float(close.iloc[-1])
    
    # Price vs MA50
    try:
        ma50_series = sma(close, 50)
        ma50 = float(ma50_series.iloc[-1])
        price_vs_ma50 = (current / ma50 - 1) * 100 if ma50 > 0 else 0
    except Exception:
        ma50 = current
        price_vs_ma50 = 0
    
    # 52W high/low (or available data)
    try:
        lookback_days = min(252, len(high))
        high_52w = float(high.tail(lookback_days).max())
        low_52w = float(low.tail(lookback_days).min())
        dist_to_high = (high_52w - current) / high_52w * 100 if high_52w > 0 else 0
        dist_to_low = (current - low_52w) / low_52w * 100 if low_52w > 0 else 0
        
        # Position in range (0 = at low, 100 = at high)
        range_val = high_52w - low_52w
        range_position = ((current - low_52w) / range_val * 100) if range_val > 0 else 50
    except Exception:
        dist_to_high = 0
        dist_to_low = 0
        range_position = 50
    
    # Long-term momentum
    try:
        actual_lookback = min(lookback, len(close) - 1)
        momentum_60d = (current / float(close.iloc[-actual_lookback]) - 1) * 100 if actual_lookback > 0 else 0
    except (IndexError, ZeroDivisionError):
        momentum_60d = 0
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SCORING
    # ═══════════════════════════════════════════════════════════════════════════
    bullish_points = 0.0
    bearish_points = 0.0
    
    # Position in 52W range
    if range_position > 85:
        bullish_points += 2.0  # Near highs = strong
    elif range_position > 70:
        bullish_points += 1.5
    elif range_position > 50:
        bullish_points += 0.5
    elif range_position < 15:
        bearish_points += 2.0  # Near lows = weak
    elif range_position < 30:
        bearish_points += 1.5
    elif range_position < 50:
        bearish_points += 0.5
    
    # Price vs MA50
    if price_vs_ma50 > 8:
        bullish_points += 1.5
    elif price_vs_ma50 > 3:
        bullish_points += 1.0
    elif price_vs_ma50 > 0:
        bullish_points += 0.5
    elif price_vs_ma50 < -8:
        bearish_points += 1.5
    elif price_vs_ma50 < -3:
        bearish_points += 1.0
    elif price_vs_ma50 < 0:
        bearish_points += 0.5
    
    # 60-day momentum
    if momentum_60d > 20:
        bullish_points += 1.5
    elif momentum_60d > 10:
        bullish_points += 1.0
    elif momentum_60d > 0:
        bullish_points += 0.5
    elif momentum_60d < -20:
        bearish_points += 1.5
    elif momentum_60d < -10:
        bearish_points += 1.0
    elif momentum_60d < 0:
        bearish_points += 0.5
    
    # Determine direction
    total_points = bullish_points + bearish_points
    if total_points == 0:
        direction = SignalDirection.NEUTRAL
        strength = 0.3
    elif bullish_points > bearish_points + 0.5:
        direction = SignalDirection.BULLISH
        strength = min(1.0, bullish_points / 5.0)
    elif bearish_points > bullish_points + 0.5:
        direction = SignalDirection.BEARISH
        strength = min(1.0, bearish_points / 5.0)
    else:
        direction = SignalDirection.NEUTRAL
        strength = 0.3
    
    return HorizonSignal(
        horizon="long",
        direction=direction,
        strength=round(strength, 2),
        components={
            "price_vs_ma50_pct": round(price_vs_ma50, 2),
            "dist_to_52w_high_pct": round(dist_to_high, 2),
            "dist_to_52w_low_pct": round(dist_to_low, 2),
            "range_position": round(range_position, 1),
            "momentum_60d_pct": round(momentum_60d, 2),
            "bullish_points": round(bullish_points, 2),
            "bearish_points": round(bearish_points, 2),
        }
    )


def compute_multi_horizon(df: pd.DataFrame, ticker: str) -> MultiHorizonResult:
    """
    Compute signals across all three horizons and aggregate.
    
    Args:
        df: DataFrame with OHLCV data
        ticker: Ticker symbol
    
    Returns:
        MultiHorizonResult with consensus, agreement count, and recommendation
    """
    short = compute_short_horizon(df)
    medium = compute_medium_horizon(df)
    long = compute_long_horizon(df)
    
    # Count agreement
    signals = [short.direction, medium.direction, long.direction]
    bullish_count = sum(1 for s in signals if s == SignalDirection.BULLISH)
    bearish_count = sum(1 for s in signals if s == SignalDirection.BEARISH)
    neutral_count = sum(1 for s in signals if s == SignalDirection.NEUTRAL)
    
    # Determine consensus
    if bullish_count >= 2:
        consensus = SignalDirection.BULLISH
        agreement_count = bullish_count
    elif bearish_count >= 2:
        consensus = SignalDirection.BEARISH
        agreement_count = bearish_count
    else:
        consensus = SignalDirection.NEUTRAL
        agreement_count = max(bullish_count, bearish_count, neutral_count)
    
    # Confidence based on agreement and average strength
    avg_strength = (short.strength + medium.strength + long.strength) / 3
    if agreement_count == 3:
        confidence = avg_strength * 1.0
    elif agreement_count == 2:
        confidence = avg_strength * 0.8
    else:
        confidence = avg_strength * 0.5
    
    # Trade recommendation
    if consensus == SignalDirection.BULLISH:
        if agreement_count == 3:
            trade_rec = "STRONG_BUY"
        elif agreement_count == 2 and avg_strength > 0.6:
            trade_rec = "BUY"
        elif agreement_count == 2:
            trade_rec = "LEAN_BUY"
        else:
            trade_rec = "HOLD"
    elif consensus == SignalDirection.BEARISH:
        if agreement_count >= 2:
            trade_rec = "AVOID"
        else:
            trade_rec = "CAUTION"
    else:
        trade_rec = "HOLD"
    
    # Conflict notes (when short and long disagree)
    conflict_note = None
    if short.direction != long.direction:
        if short.direction != SignalDirection.NEUTRAL and long.direction != SignalDirection.NEUTRAL:
            if short.direction == SignalDirection.BULLISH and long.direction == SignalDirection.BEARISH:
                conflict_note = "Short-term bullish but long-term bearish: potential bear market rally or dead cat bounce"
            elif short.direction == SignalDirection.BEARISH and long.direction == SignalDirection.BULLISH:
                conflict_note = "Short-term bearish but long-term bullish: potential pullback in uptrend - may be buying opportunity"
    
    return MultiHorizonResult(
        ticker=ticker,
        short_signal=short,
        medium_signal=medium,
        long_signal=long,
        consensus=consensus,
        agreement_count=agreement_count,
        confidence=round(confidence, 2),
        trade_recommendation=trade_rec,
        conflict_note=conflict_note,
    )


def format_multi_horizon_summary(result: MultiHorizonResult) -> str:
    """
    Format multi-horizon result for display.
    
    Args:
        result: MultiHorizonResult from compute_multi_horizon()
    
    Returns:
        Formatted string summary
    """
    short_dir = result.short_signal.direction.value.upper()
    medium_dir = result.medium_signal.direction.value.upper()
    long_dir = result.long_signal.direction.value.upper()
    
    lines = [
        f"Multi-Horizon Analysis: {result.ticker}",
        f"  Short (5d):  {short_dir:<8} (strength: {result.short_signal.strength:.2f})",
        f"  Medium (20d): {medium_dir:<8} (strength: {result.medium_signal.strength:.2f})",
        f"  Long (60d):  {long_dir:<8} (strength: {result.long_signal.strength:.2f})",
        f"  ────────────────────────────",
        f"  Consensus: {result.consensus.value.upper()} ({result.agreement_count}/3 agree)",
        f"  Confidence: {result.confidence:.2f}",
        f"  Recommendation: {result.trade_recommendation}",
    ]
    
    if result.conflict_note:
        lines.append(f"  Note: {result.conflict_note}")
    
    return "\n".join(lines)
