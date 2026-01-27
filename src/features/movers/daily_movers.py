#!/usr/bin/env python3
"""
Daily Movers Discovery Module

Fetches daily gainers/losers and applies strict filters.
This is a QUARANTINED idea funnel - candidates must pass all normal gates.

Supports both Polygon (primary) and Yahoo Finance (fallback) data sources.
Uses caching to avoid redundant downloads within the same session.
"""

from __future__ import annotations
import os
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, datetime, timedelta
from typing import Optional
from pathlib import Path

# Import helpers from core
from ...core.yf import get_ticker_df, download_daily_range_cached
from ...core.helpers import get_ny_date
from ...core.polygon import download_polygon_batch

logger = logging.getLogger(__name__)


def _compute_movers_from_dataframes(
    ticker_data: dict[str, pd.DataFrame],
    top_n: int,
    asof_date: date,
    source: str,
) -> dict:
    """
    Compute gainers/losers from a dict of ticker -> DataFrame.
    
    Args:
        ticker_data: Dict mapping ticker symbol to OHLCV DataFrame
        top_n: Number of top gainers/losers to return
        asof_date: Date for the computation
        source: Source identifier for metadata
    
    Returns:
        dict with keys: "gainers", "losers", "meta"
    """
    results = []
    
    for ticker, df in ticker_data.items():
        try:
            if df.empty or len(df) < 2:
                continue
            
            # Get last 2 trading days (sorted by date)
            df_sorted = df.sort_index()
            last_close = float(df_sorted["Close"].iloc[-1])
            prev_close = float(df_sorted["Close"].iloc[-2])
            
            # Compute % change
            pct_change = ((last_close - prev_close) / prev_close) * 100 if prev_close > 0 else 0.0
            
            # Get volume
            volume = float(df_sorted["Volume"].iloc[-1]) if not df_sorted["Volume"].empty else 0.0
            
            # Get daily range for close position calculation
            high = float(df_sorted["High"].iloc[-1])
            low = float(df_sorted["Low"].iloc[-1])
            
            results.append({
                "ticker": ticker,
                "last_close": last_close,
                "prev_close": prev_close,
                "pct_change_1d": round(pct_change, 2),
                "volume": volume,
                "high": high,
                "low": low,
                "asof_date_utc": datetime.utcnow().isoformat(),
                "source": source
            })
        except Exception:
            continue
    
    if not results:
        return {
            "gainers": pd.DataFrame(),
            "losers": pd.DataFrame(),
            "meta": {"count": 0, "asof_date": asof_date.isoformat(), "source": source}
        }
    
    df_all = pd.DataFrame(results)
    
    # Sort and select top/bottom
    df_gainers = df_all.nlargest(top_n, "pct_change_1d").copy()
    df_losers = df_all.nsmallest(top_n, "pct_change_1d").copy()
    
    meta = {
        "total_computed": len(results),
        "top_n": top_n,
        "gainers_count": len(df_gainers),
        "losers_count": len(df_losers),
        "asof_date": asof_date.isoformat(),
        "source": source
    }
    
    return {
        "gainers": df_gainers,
        "losers": df_losers,
        "meta": meta
    }


def compute_daily_movers_from_universe(
    tickers: list[str],
    top_n: int = 50,
    asof_date: Optional[date] = None,
    polygon_api_key: Optional[str] = None,
    use_polygon_primary: bool = True,
    polygon_max_workers: int = 8,
    quarantine_cfg: Optional[dict] = None,
    yf_retry_cfg: Optional[dict] = None,
    polygon_retry_cfg: Optional[dict] = None,
) -> dict:
    """
    Compute daily movers from universe tickers.
    
    Uses Polygon as primary data source (if API key provided and enabled),
    with Yahoo Finance as fallback for missing tickers.
    
    Args:
        tickers: List of ticker symbols
        top_n: Number of top gainers/losers to return
        asof_date: Date to compute movers for (defaults to today)
        polygon_api_key: Polygon.io API key (from env or config)
        use_polygon_primary: Whether to use Polygon as primary source
        polygon_max_workers: Max parallel workers for Polygon requests
    
    Returns:
        dict with keys: "gainers", "losers", "meta"
    """
    if asof_date is None:
        asof_date = get_ny_date()
    
    print(f"Computing daily movers for {len(tickers)} tickers (asof: {asof_date})...")
    
    # Need ~14 days of data to ensure we have at least 2 trading days
    lookback_days = 14
    end_str = asof_date.isoformat() if isinstance(asof_date, date) else str(asof_date)
    
    ticker_data: dict[str, pd.DataFrame] = {}
    source = "yfinance"
    
    # Try Polygon first if enabled and API key available
    if use_polygon_primary and polygon_api_key:
        print(f"[1/2] Fetching OHLCV from Polygon (primary)...")
        try:
            polygon_data = download_polygon_batch(
                tickers,
                lookback_days=lookback_days,
                asof_date=end_str,
                api_key=polygon_api_key,
                max_workers=polygon_max_workers,
                quarantine_cfg=quarantine_cfg,
                retry_cfg=polygon_retry_cfg,
            )
            # Filter out empty DataFrames
            ticker_data = {t: df for t, df in polygon_data.items() if not df.empty and len(df) >= 2}
            polygon_count = len(ticker_data)
            print(f"  Polygon populated {polygon_count}/{len(tickers)} tickers.")
            source = "polygon"
        except Exception as e:
            print(f"  Polygon fetch failed: {e}")
            ticker_data = {}
    
    # Fallback to Yahoo Finance for missing tickers (uses cache)
    missing_tickers = [t for t in tickers if t not in ticker_data]
    
    if missing_tickers:
        fallback_label = "fallback" if ticker_data else "primary"
        print(f"[{'2/2' if ticker_data else '1/1'}] Fetching OHLCV from yfinance for {len(missing_tickers)} tickers ({fallback_label})...")
        
        try:
            end_dt = pd.Timestamp(asof_date)
            start_dt = end_dt - pd.Timedelta(days=lookback_days)
            
            # Use cached download function
            yf_data_dict, report = download_daily_range_cached(
                tickers=missing_tickers,
                start=start_dt.to_pydatetime(),
                end=end_dt.to_pydatetime(),
                auto_adjust=False,
                threads=True,
                quarantine_cfg=quarantine_cfg,
                retry_cfg=yf_retry_cfg,
            )
            
            cache_hits = report.get("cache_hits", 0)
            if cache_hits > 0:
                logger.info(f"yfinance cache: {cache_hits} hits, {report.get('downloaded', 0)} downloaded")
            
            # Add valid DataFrames
            for ticker, df in yf_data_dict.items():
                if not df.empty and len(df) >= 2:
                    ticker_data[ticker] = df
            
            if not source == "polygon":
                source = "yfinance"
            else:
                source = "polygon+yfinance"
                
        except Exception as e:
            print(f"  yfinance fetch error: {e}")
            if not ticker_data:
                return {
                    "gainers": pd.DataFrame(),
                    "losers": pd.DataFrame(),
                    "meta": {"error": str(e), "asof_date": asof_date.isoformat()}
                }
    
    print(f"  Total tickers with valid data: {len(ticker_data)}")
    
    return _compute_movers_from_dataframes(ticker_data, top_n, asof_date, source)


def get_daily_movers(
    tickers: Optional[list[str]] = None,
    top_n: int = 50,
    asof_date: Optional[date] = None,
    polygon_api_key: Optional[str] = None,
    use_polygon_primary: bool = True,
    polygon_max_workers: int = 8,
) -> dict:
    """
    Main entry point for daily movers.
    
    If tickers is None, returns empty DataFrames (caller should provide universe).
    
    Args:
        tickers: List of ticker symbols to scan
        top_n: Number of top gainers/losers to return
        asof_date: Date to compute movers for
        polygon_api_key: Polygon.io API key
        use_polygon_primary: Whether to use Polygon as primary source
        polygon_max_workers: Max parallel workers for Polygon
    """
    if tickers is None or len(tickers) == 0:
        return {
            "gainers": pd.DataFrame(),
            "losers": pd.DataFrame(),
            "meta": {"error": "No tickers provided"}
        }
    
    return compute_daily_movers_from_universe(
        tickers,
        top_n,
        asof_date,
        polygon_api_key=polygon_api_key,
        use_polygon_primary=use_polygon_primary,
        polygon_max_workers=polygon_max_workers,
    )

