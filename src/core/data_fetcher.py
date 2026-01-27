"""
Unified Data Fetcher

Downloads historical price data using Polygon as primary source,
Yahoo Finance as secondary/fallback, and stores permanently in the price database.

No more re-downloading the same data!
"""

from __future__ import annotations

import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date, timedelta
from typing import Optional, List, Tuple
import pandas as pd
import requests

from src.core.price_db import get_price_db, PriceDatabase

logger = logging.getLogger(__name__)


def fetch_from_polygon(
    ticker: str,
    start_date: str,
    end_date: str,
    api_key: str,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Fetch daily OHLCV from Polygon.io API.
    
    Returns:
        Tuple of (DataFrame, error_message)
        DataFrame is empty if fetch failed.
    """
    if not api_key:
        return pd.DataFrame(), "No Polygon API key"
    
    try:
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
        resp = requests.get(
            url,
            params={"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": api_key},
            timeout=20,
        )
        resp.raise_for_status()
        
        results = resp.json().get("results") or []
        if not results:
            return pd.DataFrame(), f"No data returned for {ticker}"
        
        df = pd.DataFrame(results)
        if df.empty or not {"o", "h", "l", "c", "v", "t"}.issubset(df.columns):
            return pd.DataFrame(), f"Invalid data format for {ticker}"
        
        df["Date"] = pd.to_datetime(df["t"], unit="ms")
        df = df.rename(columns={
            "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"
        })
        df = df.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]].dropna()
        
        return df, None
        
    except requests.exceptions.Timeout:
        return pd.DataFrame(), f"Timeout fetching {ticker}"
    except requests.exceptions.RequestException as e:
        return pd.DataFrame(), f"Request error for {ticker}: {e}"
    except Exception as e:
        return pd.DataFrame(), f"Error fetching {ticker}: {e}"


def fetch_from_yfinance(
    ticker: str,
    start_date: str,
    end_date: str,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Fetch daily OHLCV from Yahoo Finance.
    
    Returns:
        Tuple of (DataFrame, error_message)
    """
    try:
        import yfinance as yf
        
        # yfinance end date is exclusive, add 1 day
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        end_str = end_dt.strftime("%Y-%m-%d")
        
        data = yf.download(
            ticker,
            start=start_date,
            end=end_str,
            progress=False,
            auto_adjust=False,
        )
        
        if data.empty:
            return pd.DataFrame(), f"No data from YF for {ticker}"
        
        # Handle MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        return data, None
        
    except Exception as e:
        return pd.DataFrame(), f"YF error for {ticker}: {e}"


def download_and_store(
    ticker: str,
    start_date: str,
    end_date: str,
    polygon_api_key: Optional[str] = None,
    use_yf_fallback: bool = True,
    verify_with_yf: bool = False,
    db: Optional[PriceDatabase] = None,
) -> Tuple[int, str]:
    """
    Download price data and store in permanent database.
    
    Uses Polygon as primary, Yahoo Finance as fallback/verification.
    
    Args:
        ticker: Stock symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        polygon_api_key: Polygon.io API key
        use_yf_fallback: Use Yahoo Finance if Polygon fails
        verify_with_yf: Cross-verify Polygon data with Yahoo Finance
        db: Price database instance
    
    Returns:
        Tuple of (records_stored, source_used)
    """
    if db is None:
        db = get_price_db()
    
    ticker = ticker.upper()
    source_used = "none"
    df = pd.DataFrame()
    error = None
    
    # Try Polygon first
    if polygon_api_key:
        df, error = fetch_from_polygon(ticker, start_date, end_date, polygon_api_key)
        if not df.empty:
            source_used = "polygon"
            logger.debug(f"{ticker}: Got {len(df)} records from Polygon")
    
    # Fallback to Yahoo Finance
    if df.empty and use_yf_fallback:
        df, error = fetch_from_yfinance(ticker, start_date, end_date)
        if not df.empty:
            source_used = "yfinance"
            logger.debug(f"{ticker}: Got {len(df)} records from Yahoo Finance")
    
    # Store in database
    if not df.empty:
        records = db.store_prices(ticker, df, source=source_used, verified=False)
        db.log_download(ticker, start_date, end_date, source_used, records, success=True)
        return records, source_used
    else:
        db.log_download(ticker, start_date, end_date, source_used, 0, success=False, error=error)
        return 0, "failed"


def bulk_download(
    tickers: List[str],
    start_date: str,
    end_date: str,
    polygon_api_key: Optional[str] = None,
    max_workers: int = 8,
    skip_existing: bool = True,
    min_coverage: float = 0.6,
    progress_callback=None,
) -> dict:
    """
    Download price data for multiple tickers.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        polygon_api_key: Polygon.io API key
        max_workers: Concurrent downloads
        skip_existing: Skip tickers that already have sufficient data
        min_coverage: Minimum data coverage to consider "existing"
        progress_callback: Optional callback(completed, total, ticker)
    
    Returns:
        Summary dict with counts
    """
    db = get_price_db()
    
    # Determine which tickers need downloading
    tickers_to_download = []
    already_have = 0
    
    if skip_existing:
        for ticker in tickers:
            if db.has_data(ticker, start_date, end_date, min_coverage):
                already_have += 1
            else:
                tickers_to_download.append(ticker)
        
        if already_have > 0:
            logger.info(f"Skipping {already_have} tickers with existing data")
    else:
        tickers_to_download = list(tickers)
    
    # Download missing data
    downloaded = 0
    failed = 0
    total_records = 0
    completed = 0
    
    def worker(ticker: str) -> Tuple[str, int, str]:
        records, source = download_and_store(
            ticker, start_date, end_date,
            polygon_api_key=polygon_api_key,
            use_yf_fallback=True,
            db=db,
        )
        return ticker, records, source
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, t): t for t in tickers_to_download}
        
        for future in as_completed(futures):
            ticker, records, source = future.result()
            completed += 1
            
            if records > 0:
                downloaded += 1
                total_records += records
            else:
                failed += 1
            
            if progress_callback:
                progress_callback(completed, len(tickers_to_download), ticker)
            
            # Log progress periodically
            if completed % 50 == 0 or completed == len(tickers_to_download):
                logger.info(f"Progress: {completed}/{len(tickers_to_download)} "
                           f"(downloaded: {downloaded}, failed: {failed})")
    
    return {
        "total_tickers": len(tickers),
        "already_had": already_have,
        "downloaded": downloaded,
        "failed": failed,
        "total_records": total_records,
    }


def ensure_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    polygon_api_key: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """
    Ensure we have data for tickers, downloading if needed.
    
    Returns dict mapping ticker -> DataFrame.
    This is the main function to use when you need price data.
    """
    db = get_price_db()
    result = {}
    
    # Check what we have
    tickers_need_download = []
    for ticker in tickers:
        df = db.get_prices(ticker, start_date, end_date)
        if df.empty or len(df) < 10:  # Need at least some data
            tickers_need_download.append(ticker)
        else:
            result[ticker] = df
    
    # Download missing
    if tickers_need_download:
        logger.info(f"Downloading data for {len(tickers_need_download)} tickers...")
        bulk_download(
            tickers_need_download,
            start_date,
            end_date,
            polygon_api_key=polygon_api_key,
            skip_existing=False,
        )
        
        # Fetch from database
        for ticker in tickers_need_download:
            df = db.get_prices(ticker, start_date, end_date)
            if not df.empty:
                result[ticker] = df
    
    return result


def get_prices_for_scanner(
    tickers: List[str],
    start_date: str,
    end_date: str,
    min_rows: int = 20,
    polygon_api_key: Optional[str] = None,
    auto_download: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Get price data for scanners from permanent database.
    
    This is the primary function scanners should use.
    Checks database first, downloads missing data if auto_download=True.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        min_rows: Minimum rows required for valid data
        polygon_api_key: Polygon API key for downloads
        auto_download: Whether to download missing data
    
    Returns:
        Dict mapping ticker -> DataFrame with OHLCV data
    """
    db = get_price_db()
    result = {}
    missing = []
    
    # Get from database
    for ticker in tickers:
        df = db.get_prices(ticker, start_date, end_date)
        if not df.empty and len(df) >= min_rows:
            result[ticker] = df
        else:
            missing.append(ticker)
    
    # Log what we found
    if result:
        logger.info(f"Price database: {len(result)} tickers loaded, {len(missing)} missing")
    
    # Download missing if enabled
    if missing and auto_download:
        if polygon_api_key is None:
            polygon_api_key = os.environ.get("POLYGON_API_KEY")
        
        logger.info(f"Auto-downloading {len(missing)} missing tickers...")
        bulk_download(
            missing,
            start_date,
            end_date,
            polygon_api_key=polygon_api_key,
            skip_existing=False,
            max_workers=8,
        )
        
        # Retry from database
        for ticker in missing:
            df = db.get_prices(ticker, start_date, end_date)
            if not df.empty and len(df) >= min_rows:
                result[ticker] = df
    
    return result
