"""
Sector Rotation Scanner
========================

Identifies sectors showing momentum and then finds the strongest individual
stocks within those sectors. Based on the principle that:

1. Sector momentum tends to persist (trend following)
2. Leading stocks in leading sectors have highest odds
3. Sector relative strength predicts future performance

Research shows sector selection explains 40-60% of stock returns.
"""

from __future__ import annotations

import os
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional
import requests

try:
    import pandas as pd
    import numpy as np
except ImportError:
    raise ImportError("pandas and numpy required for sector rotation")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

# Sector ETFs for tracking
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}

# Representative stocks by sector (for individual stock analysis)
SECTOR_STOCKS = {
    "Technology": ["AAPL", "MSFT", "NVDA", "AVGO", "CRM", "ADBE", "AMD", "INTC", "ORCL", "CSCO"],
    "Financials": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "USB"],
    "Healthcare": ["UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HAL"],
    "Industrials": ["CAT", "DE", "UNP", "UPS", "BA", "HON", "GE", "RTX", "LMT", "MMM"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "CMG", "MAR"],
    "Consumer Staples": ["PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "CL", "MDLZ", "KMB"],
    "Materials": ["LIN", "APD", "SHW", "FCX", "NEM", "ECL", "DOW", "DD", "NUE", "VMC"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "ED", "PEG"],
    "Real Estate": ["PLD", "AMT", "EQIX", "PSA", "SPG", "O", "WELL", "DLR", "AVB", "EQR"],
    "Communication Services": ["META", "GOOGL", "NFLX", "DIS", "CMCSA", "VZ", "T", "CHTR", "TMUS", "EA"],
}


@dataclass
class SectorMomentum:
    """Momentum data for a single sector."""
    sector: str
    etf: str
    return_1w: float
    return_2w: float
    return_1m: float
    rs_score: float  # Relative strength vs SPY
    momentum_score: float  # 0-10 composite
    trend: str  # "accelerating", "steady", "decelerating"


@dataclass
class SectorLeader:
    """A leading stock within a hot sector."""
    ticker: str
    sector: str
    stock_return_1w: float
    stock_return_1m: float
    vs_sector: float  # Outperformance vs sector ETF
    vs_spy: float  # Outperformance vs SPY
    relative_strength_rank: int  # Rank within sector (1 = best)
    composite_score: float  # 0-10


def fetch_price_history(ticker: str, days: int = 30) -> pd.Series:
    """Fetch price history for a ticker."""
    if not POLYGON_API_KEY:
        # Fallback to yfinance
        try:
            import yfinance as yf
            end = datetime.now()
            start = end - timedelta(days=days + 10)
            df = yf.download(ticker, start=start, end=end, progress=False)
            if not df.empty:
                return df["Close"]
        except Exception as e:
            logger.warning(f"yfinance fallback failed for {ticker}: {e}")
        return pd.Series(dtype=float)
    
    end = datetime.now()
    start = end - timedelta(days=days + 10)
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
    params = {"apiKey": POLYGON_API_KEY, "adjusted": "true", "sort": "asc"}
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            if results:
                dates = [datetime.fromtimestamp(r["t"] / 1000) for r in results]
                closes = [r["c"] for r in results]
                return pd.Series(closes, index=pd.DatetimeIndex(dates))
    except Exception as e:
        logger.warning(f"Failed to fetch {ticker}: {e}")
    
    return pd.Series(dtype=float)


def calculate_sector_momentum() -> list[SectorMomentum]:
    """Calculate momentum for all sectors."""
    spy_prices = fetch_price_history("SPY", days=30)
    if spy_prices.empty:
        logger.error("Failed to fetch SPY data")
        return []
    
    spy_1w = (spy_prices.iloc[-1] / spy_prices.iloc[-5] - 1) * 100 if len(spy_prices) >= 5 else 0
    spy_1m = (spy_prices.iloc[-1] / spy_prices.iloc[0] - 1) * 100 if len(spy_prices) >= 20 else 0
    
    sectors = []
    
    for etf, sector_name in SECTOR_ETFS.items():
        prices = fetch_price_history(etf, days=30)
        if prices.empty or len(prices) < 20:
            continue
        
        # Calculate returns
        ret_1w = (prices.iloc[-1] / prices.iloc[-5] - 1) * 100 if len(prices) >= 5 else 0
        ret_2w = (prices.iloc[-1] / prices.iloc[-10] - 1) * 100 if len(prices) >= 10 else 0
        ret_1m = (prices.iloc[-1] / prices.iloc[0] - 1) * 100 if len(prices) >= 20 else 0
        
        # Relative strength vs SPY
        rs_score = ret_1m - spy_1m
        
        # Determine trend
        if ret_1w > ret_2w / 2:
            trend = "accelerating"
        elif ret_1w < ret_2w / 4:
            trend = "decelerating"
        else:
            trend = "steady"
        
        # Momentum score (0-10)
        momentum_score = 5.0  # Base
        momentum_score += min(2.5, ret_1w / 2)  # Recent performance
        momentum_score += min(2.5, rs_score / 2)  # Relative strength
        if trend == "accelerating":
            momentum_score += 1
        elif trend == "decelerating":
            momentum_score -= 1
        momentum_score = max(0, min(10, momentum_score))
        
        sectors.append(SectorMomentum(
            sector=sector_name,
            etf=etf,
            return_1w=round(ret_1w, 2),
            return_2w=round(ret_2w, 2),
            return_1m=round(ret_1m, 2),
            rs_score=round(rs_score, 2),
            momentum_score=round(momentum_score, 2),
            trend=trend,
        ))
    
    # Sort by momentum score
    sectors.sort(key=lambda x: x.momentum_score, reverse=True)
    
    return sectors


def find_sector_leaders(
    top_n_sectors: int = 3,
    stocks_per_sector: int = 5,
) -> list[SectorLeader]:
    """
    Find leading stocks in the hottest sectors.
    
    Args:
        top_n_sectors: Number of top sectors to analyze
        stocks_per_sector: Leaders to return per sector
    
    Returns:
        List of SectorLeader candidates
    """
    # Get sector momentum
    sector_momentum = calculate_sector_momentum()
    
    if not sector_momentum:
        logger.error("No sector momentum data available")
        return []
    
    # Get SPY benchmark
    spy_prices = fetch_price_history("SPY", days=30)
    spy_1w = (spy_prices.iloc[-1] / spy_prices.iloc[-5] - 1) * 100 if len(spy_prices) >= 5 else 0
    spy_1m = (spy_prices.iloc[-1] / spy_prices.iloc[0] - 1) * 100 if len(spy_prices) >= 20 else 0
    
    # Focus on top sectors
    hot_sectors = [s for s in sector_momentum if s.momentum_score >= 6.0][:top_n_sectors]
    
    if not hot_sectors:
        logger.info("No sectors with momentum score >= 6.0, using top 3")
        hot_sectors = sector_momentum[:top_n_sectors]
    
    logger.info(f"Analyzing top sectors: {[s.sector for s in hot_sectors]}")
    
    all_leaders = []
    
    for sector_data in hot_sectors:
        sector_name = sector_data.sector
        sector_return = sector_data.return_1m
        
        stocks = SECTOR_STOCKS.get(sector_name, [])
        if not stocks:
            continue
        
        stock_scores = []
        
        for ticker in stocks:
            prices = fetch_price_history(ticker, days=30)
            if prices.empty or len(prices) < 20:
                continue
            
            ret_1w = (prices.iloc[-1] / prices.iloc[-5] - 1) * 100 if len(prices) >= 5 else 0
            ret_1m = (prices.iloc[-1] / prices.iloc[0] - 1) * 100 if len(prices) >= 20 else 0
            
            vs_sector = ret_1m - sector_return
            vs_spy = ret_1m - spy_1m
            
            # Composite score
            score = 5.0
            score += min(2.5, vs_sector / 3)  # Outperform sector
            score += min(2.5, vs_spy / 3)  # Outperform market
            score += min(1.0, ret_1w / 3)  # Recent momentum
            score = max(0, min(10, score))
            
            stock_scores.append({
                "ticker": ticker,
                "sector": sector_name,
                "ret_1w": round(ret_1w, 2),
                "ret_1m": round(ret_1m, 2),
                "vs_sector": round(vs_sector, 2),
                "vs_spy": round(vs_spy, 2),
                "score": round(score, 2),
            })
        
        # Rank within sector
        stock_scores.sort(key=lambda x: x["score"], reverse=True)
        
        for rank, s in enumerate(stock_scores[:stocks_per_sector], 1):
            all_leaders.append(SectorLeader(
                ticker=s["ticker"],
                sector=s["sector"],
                stock_return_1w=s["ret_1w"],
                stock_return_1m=s["ret_1m"],
                vs_sector=s["vs_sector"],
                vs_spy=s["vs_spy"],
                relative_strength_rank=rank,
                composite_score=s["score"],
            ))
    
    # Sort all leaders by composite score
    all_leaders.sort(key=lambda x: x.composite_score, reverse=True)
    
    return all_leaders


def format_sector_report(
    sectors: list[SectorMomentum],
    leaders: list[SectorLeader],
) -> str:
    """Format sector rotation analysis as readable report."""
    lines = [
        "=" * 60,
        "🏭 SECTOR ROTATION ANALYSIS",
        "=" * 60,
        "",
        "📊 SECTOR MOMENTUM RANKING",
        "-" * 40,
    ]
    
    for i, s in enumerate(sectors[:6], 1):
        trend_emoji = "🚀" if s.trend == "accelerating" else "📈" if s.trend == "steady" else "📉"
        lines.append(
            f"{i}. {s.sector} ({s.etf}): {s.momentum_score}/10 {trend_emoji}"
        )
        lines.append(
            f"   1W: {s.return_1w:+.1f}% | 1M: {s.return_1m:+.1f}% | RS: {s.rs_score:+.1f}"
        )
    
    lines.extend(["", "🏆 SECTOR LEADERS", "-" * 40])
    
    for i, l in enumerate(leaders[:10], 1):
        lines.append(
            f"{i}. {l.ticker} [{l.sector}] - Score: {l.composite_score}/10"
        )
        lines.append(
            f"   1W: {l.stock_return_1w:+.1f}% | vs Sector: {l.vs_sector:+.1f}% | vs SPY: {l.vs_spy:+.1f}%"
        )
    
    return "\n".join(lines)
