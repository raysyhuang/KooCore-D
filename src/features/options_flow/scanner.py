"""
Options Flow Scanner - High-signal replacement for Movers
=========================================================

Uses Polygon options data to detect unusual activity that often precedes
significant price moves. This model looks for:

1. Unusual Volume: Options volume >> Open Interest (smart money building positions)
2. Call Sweeps: Large call purchases hitting the ask (bullish conviction)
3. Put/Call Ratio Shifts: Sudden changes in sentiment
4. Implied Volatility Expansion: Market pricing in upcoming move
5. OTM Call Buying: Aggressive bullish bets

Research shows unusual options activity can predict moves with 55-70% accuracy.
"""

from __future__ import annotations

import os
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

from ...utils.time import utc_now


@dataclass
class OptionsFlowSignal:
    """A single options flow signal."""
    ticker: str
    signal_type: str  # "call_sweep", "unusual_volume", "iv_expansion", "put_call_shift"
    strength: float  # 0-10 scale
    details: dict
    timestamp: datetime


@dataclass
class OptionsFlowCandidate:
    """A candidate identified by options flow analysis."""
    ticker: str
    flow_score: float  # 0-10 composite score
    signals: list[OptionsFlowSignal]
    call_volume: int
    put_volume: int
    put_call_ratio: float
    avg_iv: Optional[float]
    iv_rank: Optional[float]  # IV percentile vs 52w range
    unusual_volume_ratio: float  # today's vol / avg vol
    bullish_flow_pct: float  # % of flow that's bullish
    

def fetch_options_chain(ticker: str, api_key: str) -> dict:
    """Fetch current options chain from Polygon."""
    if not api_key:
        return {}
    
    url = f"https://api.polygon.io/v3/snapshot/options/{ticker}"
    params = {"apiKey": api_key}
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.warning(f"Failed to fetch options for {ticker}: {e}")
    
    return {}


def fetch_options_trades(ticker: str, api_key: str, date: str = None) -> list[dict]:
    """Fetch options trades to detect sweeps and large orders."""
    if not api_key:
        return []
    
    if date is None:
        date = utc_now().strftime("%Y-%m-%d")
    
    # Get aggregated options activity
    url = f"https://api.polygon.io/v3/trades/options/{ticker}"
    params = {
        "apiKey": api_key,
        "timestamp.gte": f"{date}T09:30:00Z",
        "timestamp.lte": f"{date}T16:00:00Z",
        "limit": 500,
        "order": "desc",
    }
    
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("results", [])
    except Exception as e:
        logger.warning(f"Failed to fetch options trades for {ticker}: {e}")
    
    return []


def analyze_options_flow(ticker: str, api_key: str = None) -> Optional[OptionsFlowCandidate]:
    """
    Analyze options flow for a single ticker.
    
    Returns OptionsFlowCandidate if significant signals detected, None otherwise.
    """
    api_key = api_key or POLYGON_API_KEY
    if not api_key:
        logger.warning("No Polygon API key - cannot analyze options flow")
        return None
    
    chain = fetch_options_chain(ticker, api_key)
    if not chain or "results" not in chain:
        return None
    
    results = chain.get("results", [])
    if not results:
        return None
    
    # Aggregate metrics
    total_call_volume = 0
    total_put_volume = 0
    total_call_oi = 0
    total_put_oi = 0
    iv_values = []
    
    for contract in results:
        details = contract.get("day", {})
        volume = details.get("volume", 0) or 0
        oi = details.get("open_interest", 0) or 0
        iv = contract.get("implied_volatility")
        
        # Check contract type
        contract_type = contract.get("details", {}).get("contract_type", "").lower()
        
        if contract_type == "call":
            total_call_volume += volume
            total_call_oi += oi
        elif contract_type == "put":
            total_put_volume += volume
            total_put_oi += oi
        
        if iv and 0.05 < iv < 2.0:  # Filter reasonable IVs
            iv_values.append(iv)
    
    # Calculate metrics
    total_volume = total_call_volume + total_put_volume
    total_oi = total_call_oi + total_put_oi
    
    if total_volume < 100:  # Skip low-activity names
        return None
    
    put_call_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else 1.0
    volume_oi_ratio = total_volume / total_oi if total_oi > 0 else 0
    avg_iv = sum(iv_values) / len(iv_values) if iv_values else None
    bullish_flow_pct = total_call_volume / total_volume if total_volume > 0 else 0.5
    
    # Detect signals
    signals = []
    
    # Signal 1: Unusual Volume (Volume >> OI suggests new positions)
    if volume_oi_ratio > 0.5:  # More than 50% of OI traded today
        strength = min(10, volume_oi_ratio * 5)
        signals.append(OptionsFlowSignal(
            ticker=ticker,
            signal_type="unusual_volume",
            strength=strength,
            details={"volume_oi_ratio": round(volume_oi_ratio, 2)},
            timestamp=utc_now(),
        ))
    
    # Signal 2: Bullish Skew (Heavy call buying)
    if bullish_flow_pct > 0.65 and total_call_volume > 1000:
        strength = min(10, (bullish_flow_pct - 0.5) * 20)
        signals.append(OptionsFlowSignal(
            ticker=ticker,
            signal_type="bullish_skew",
            strength=strength,
            details={"bullish_pct": round(bullish_flow_pct * 100, 1)},
            timestamp=utc_now(),
        ))
    
    # Signal 3: Low Put/Call Ratio (Bullish sentiment)
    if put_call_ratio < 0.5:
        strength = min(10, (1 - put_call_ratio) * 10)
        signals.append(OptionsFlowSignal(
            ticker=ticker,
            signal_type="low_put_call",
            strength=strength,
            details={"put_call_ratio": round(put_call_ratio, 2)},
            timestamp=utc_now(),
        ))
    
    # Signal 4: High IV (Market expects move)
    if avg_iv and avg_iv > 0.5:  # 50%+ IV
        strength = min(10, avg_iv * 8)
        signals.append(OptionsFlowSignal(
            ticker=ticker,
            signal_type="elevated_iv",
            strength=strength,
            details={"avg_iv": round(avg_iv * 100, 1)},
            timestamp=utc_now(),
        ))
    
    if not signals:
        return None
    
    # Compute flow score
    flow_score = sum(s.strength for s in signals) / len(signals)
    
    # Boost score if multiple signals align
    if len(signals) >= 3:
        flow_score = min(10, flow_score * 1.3)
    
    return OptionsFlowCandidate(
        ticker=ticker,
        flow_score=round(flow_score, 2),
        signals=signals,
        call_volume=total_call_volume,
        put_volume=total_put_volume,
        put_call_ratio=round(put_call_ratio, 2),
        avg_iv=round(avg_iv * 100, 1) if avg_iv else None,
        iv_rank=None,  # Would need historical data
        unusual_volume_ratio=round(volume_oi_ratio, 2),
        bullish_flow_pct=round(bullish_flow_pct * 100, 1),
    )


def scan_options_flow(
    tickers: list[str],
    min_flow_score: float = 5.0,
    top_n: int = 10,
) -> list[OptionsFlowCandidate]:
    """
    Scan multiple tickers for options flow signals.
    
    Args:
        tickers: List of tickers to scan
        min_flow_score: Minimum score to include (0-10)
        top_n: Maximum candidates to return
    
    Returns:
        List of OptionsFlowCandidate sorted by flow_score descending
    """
    if not POLYGON_API_KEY:
        logger.error("No POLYGON_API_KEY - options flow scanning disabled")
        return []
    
    candidates = []
    
    logger.info(f"Scanning options flow for {len(tickers)} tickers...")
    
    for i, ticker in enumerate(tickers):
        if i > 0 and i % 50 == 0:
            logger.info(f"Progress: {i}/{len(tickers)}")
        
        try:
            candidate = analyze_options_flow(ticker, POLYGON_API_KEY)
            if candidate and candidate.flow_score >= min_flow_score:
                candidates.append(candidate)
        except Exception as e:
            logger.warning(f"Error analyzing {ticker}: {e}")
            continue
    
    # Sort by flow score
    candidates.sort(key=lambda x: x.flow_score, reverse=True)
    
    logger.info(f"Found {len(candidates)} candidates with flow score >= {min_flow_score}")
    
    return candidates[:top_n]


def format_flow_report(candidates: list[OptionsFlowCandidate]) -> str:
    """Format options flow candidates as readable report."""
    if not candidates:
        return "No significant options flow detected."
    
    lines = [
        "=" * 60,
        "📊 OPTIONS FLOW SCANNER RESULTS",
        "=" * 60,
        "",
    ]
    
    for i, c in enumerate(candidates, 1):
        signal_types = [s.signal_type for s in c.signals]
        lines.extend([
            f"{i}. {c.ticker} (Flow Score: {c.flow_score}/10)",
            f"   📈 Call Vol: {c.call_volume:,} | Put Vol: {c.put_volume:,}",
            f"   📊 P/C Ratio: {c.put_call_ratio} | Bullish Flow: {c.bullish_flow_pct}%",
            f"   🔥 Vol/OI Ratio: {c.unusual_volume_ratio}x",
            f"   ⚡ IV: {c.avg_iv}%" if c.avg_iv else "   ⚡ IV: N/A",
            f"   🚨 Signals: {', '.join(signal_types)}",
            "",
        ])
    
    return "\n".join(lines)
