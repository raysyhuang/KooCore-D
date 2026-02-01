"""
Price data loading utilities for the dashboard.
Cached, dashboard-safe implementations.
"""

import os
import pandas as pd
import numpy as np
import requests
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

# API keys
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")


def _unique_tickers(tickers: List[str]) -> List[str]:
    """Deduplicate tickers while preserving order."""
    seen = set()
    out = []
    for t in tickers:
        t = str(t).strip().upper()
        if t and t not in seen:
            out.append(t)
            seen.add(t)
    return out


def _fetch_polygon_daily(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily OHLCV from Polygon for a single ticker."""
    if not POLYGON_API_KEY:
        return pd.DataFrame()
    try:
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
        resp = requests.get(
            url,
            params={"adjusted": "true", "sort": "asc", "limit": 5000, "apiKey": POLYGON_API_KEY},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json().get("results") or []
        if not results:
            return pd.DataFrame()
        df = pd.DataFrame(results)
        if df.empty or not {"o", "h", "l", "c", "v", "t"}.issubset(df.columns):
            return pd.DataFrame()
        df["Date"] = pd.to_datetime(df["t"], unit="ms")
        df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
        df = df.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def load_prices(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load close prices for multiple tickers.
    Uses Polygon if available, falls back to yfinance.
    
    Args:
        tickers: List of ticker symbols
        start_date: YYYY-MM-DD start date
        end_date: YYYY-MM-DD end date
    
    Returns:
        DataFrame with Date index and ticker columns containing Close prices
    """
    tickers = _unique_tickers(tickers)
    if not tickers:
        return pd.DataFrame()

    close_dfs = []
    failed = []

    # Try Polygon first if available
    if POLYGON_API_KEY:
        def worker(t):
            return t, _fetch_polygon_daily(t, start_date, end_date)

        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {ex.submit(worker, t): t for t in tickers}
            for fut in as_completed(futures):
                t, df = fut.result()
                if not df.empty and "Close" in df.columns:
                    close_dfs.append(df[["Close"]].rename(columns={"Close": t}))
                else:
                    failed.append(t)
    else:
        failed = tickers

    # Fallback to Yahoo Finance for failed tickers
    if failed:
        try:
            import yfinance as yf
            yf_data = yf.download(
                tickers=failed,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False,
                threads=True,
                group_by="column",
            )
            if not yf_data.empty:
                if isinstance(yf_data.columns, pd.MultiIndex):
                    yf_close = yf_data["Close"].copy()
                else:
                    yf_close = yf_data[["Close"]].copy()
                    yf_close.columns = [failed[0]]
                for t in failed:
                    if t in yf_close.columns:
                        close_dfs.append(yf_close[[t]])
        except Exception:
            pass

    if not close_dfs:
        return pd.DataFrame()

    close = pd.concat(close_dfs, axis=1).sort_index().ffill()
    return close


@st.cache_data(ttl=3600, show_spinner=False)
def load_benchmarks(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load benchmark prices (SPY, QQQ).
    
    Returns DataFrame with Date index and benchmark columns.
    """
    return load_prices(["SPY", "QQQ"], start_date, end_date)
