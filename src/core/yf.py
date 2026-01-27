"""
Yahoo Finance Data Download Helpers

Wrappers for yfinance download with error handling and data validation.
Includes caching layer for reproducible backtests and reduced API calls.
"""

from __future__ import annotations
from typing import Optional
from datetime import datetime, timedelta, timezone
import time
import logging
import pandas as pd
import numpy as np
import yfinance as yf

from src.utils.time import utc_now
logger = logging.getLogger(__name__)

# Cache TTL settings
CACHE_TTL_LIVE = 3600           # 1 hour for live/recent data
CACHE_TTL_HISTORICAL = 86400 * 30  # 30 days for historical data (it doesn't change)


def _categorize_yf_error(err: Exception) -> str:
    msg = str(err).lower()
    if "timeout" in msg:
        return "timeout"
    if "ssl" in msg or "tls" in msg:
        return "ssl"
    if "rate" in msg or "429" in msg:
        return "rate_limit"
    if "delisted" in msg or "no price data" in msg or "pricesmissing" in msg:
        return "delisted_or_missing"
    return "download_error"


def _retry_params(cfg: Optional[dict]) -> tuple[int, float]:
    if not cfg:
        return 3, 1.0
    return int(cfg.get("max_retries", 3)), float(cfg.get("backoff_sec", 1.0))


def _maybe_quarantine(report: dict, quarantine_cfg: Optional[dict], source: str) -> None:
    if not quarantine_cfg or not quarantine_cfg.get("enabled", True):
        return
    try:
        from src.core.quarantine import record_bad_tickers
        record_bad_tickers(
            report.get("bad_tickers", []),
            report.get("reasons", {}),
            days=int(quarantine_cfg.get("days", 7)),
            source=source,
            path=quarantine_cfg.get("file", "data/bad_tickers.json"),
        )
    except Exception:
        return


def _get_price_cache():
    """Lazy import cache to avoid circular imports."""
    try:
        from src.core.cache import get_cache
        return get_cache()
    except Exception:
        return None


def _cache_key_for_ticker(ticker: str, start: str, end: str, interval: str = "1d") -> str:
    """Generate a consistent cache key for price data."""
    return f"prices:{ticker}:{start}:{end}:{interval}"


def _is_historical_range(end_date: datetime) -> bool:
    """Check if the date range is historical (ended more than 3 days ago)."""
    if end_date is None:
        return False
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)
    else:
        end_date = end_date.astimezone(timezone.utc)
    now = utc_now()
    return (now - end_date).days > 3


def get_ticker_df(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Robustly extract OHLCV for one ticker from yf.download output.
    
    Handles both:
      - dict-like access: data[t]
      - MultiIndex columns: ('Close','AAPL') or ('AAPL','Close') depending on yfinance
    
    Args:
        data: DataFrame from yf.download
        ticker: Ticker symbol to extract
    
    Returns:
        DataFrame with OHLCV columns, or empty DataFrame if not found
    """
    try:
        if isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex):
            # Try common layouts
            # Layout A: level0=PriceField, level1=Ticker
            if ticker in data.columns.get_level_values(-1):
                df = data.xs(ticker, axis=1, level=-1)
                return df.dropna(how="any")
            # Layout B: level0=Ticker, level1=PriceField
            if ticker in data.columns.get_level_values(0):
                df = data.xs(ticker, axis=1, level=0)
                return df.dropna(how="any")
        # Sometimes it's a dict-like object
        if ticker in data:
            return data[ticker].dropna(how="any")
    except Exception:
        pass
    return pd.DataFrame()


def download_daily(
    tickers: list[str],
    period: str = "1y",
    interval: str = "1d",
    auto_adjust: bool = False,
    threads: bool = True,
    progress: bool = False,
    quarantine_cfg: Optional[dict] = None,
    retry_cfg: Optional[dict] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Download daily OHLCV data for multiple tickers using yfinance.
    
    Args:
        tickers: List of ticker symbols
        period: Period string (e.g., "1y", "300d")
        interval: Interval string (must be "1d" for daily)
        auto_adjust: Whether to auto-adjust prices
        threads: Whether to use threads
        progress: Whether to show progress bar
    
    Returns:
        Tuple of (data DataFrame, report dict)
        report dict contains:
          - "bad_tickers": list of tickers with data issues
          - "reasons": dict mapping ticker -> reason string
    """
    if not tickers:
        return pd.DataFrame(), {"bad_tickers": [], "reasons": {}}
    
    data = pd.DataFrame()
    last_error = None
    max_retries, backoff = _retry_params(retry_cfg)
    for attempt in range(max_retries):
        try:
            data = yf.download(
                tickers=tickers,
                period=period,
                interval=interval,
                group_by="ticker",
                auto_adjust=auto_adjust,
                threads=threads,
                progress=progress,
            )
            last_error = None
            break
        except Exception as e:
            last_error = e
            # brief backoff then retry
            time.sleep(backoff * (attempt + 1))
    if last_error is not None:
        # If download fails completely, return empty with all tickers marked bad
        reason = _categorize_yf_error(last_error)
        report = {
            "bad_tickers": tickers,
            "reasons": {t: f"{reason}: {str(last_error)}" for t in tickers}
        }
        _maybe_quarantine(report, quarantine_cfg, source="yfinance")
        return pd.DataFrame(), report
    
    # Validate data for each ticker
    bad_tickers = []
    reasons = {}
    
    for ticker in tickers:
        df = get_ticker_df(data, ticker)
        
        if df.empty:
            bad_tickers.append(ticker)
            reasons[ticker] = "No data returned"
            continue
        
        # Check required columns
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            bad_tickers.append(ticker)
            reasons[ticker] = f"Missing columns: {missing_cols}"
            continue
        
        # Check for invalid price data
        if len(df) == 0:
            bad_tickers.append(ticker)
            reasons[ticker] = "Empty history"
            continue
        
        # Check for high < low (invalid bar)
        invalid_bars = (df["High"] < df["Low"]).sum()
        if invalid_bars > 0:
            reasons[ticker] = f"{invalid_bars} invalid bars (high < low)"
            # Don't mark as bad, but note it
        
        # Check for missing volume
        if df["Volume"].isna().all():
            bad_tickers.append(ticker)
            reasons[ticker] = "All volume data missing"
            continue
    
    report = {
        "bad_tickers": bad_tickers,
        "reasons": reasons,
    }
    _maybe_quarantine(report, quarantine_cfg, source="yfinance")
    return data, report


def download_daily_range(
    tickers: list[str],
    start: str | datetime,
    end: str | datetime,
    *,
    interval: str = "1d",
    auto_adjust: bool = False,
    threads: bool = True,
    progress: bool = False,
    use_cache: bool = True,
    quarantine_cfg: Optional[dict] = None,
    retry_cfg: Optional[dict] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Download daily OHLCV data for multiple tickers using yfinance with an explicit date range.
    
    Features caching layer for reproducible backtests:
    - Historical data (end > 3 days ago) is cached for 30 days
    - Recent data is cached for 1 hour
    - Cache is checked first, only missing tickers are downloaded

    Notes:
    - `end` is treated as inclusive by adding +1 day for yfinance's `end` behavior.
    - Intended for historical replay ("as-of date" runs).
    """
    if not tickers:
        return pd.DataFrame(), {"bad_tickers": [], "reasons": {}}

    try:
        start_dt = pd.to_datetime(start).to_pydatetime() if not isinstance(start, datetime) else start
        end_dt = pd.to_datetime(end).to_pydatetime() if not isinstance(end, datetime) else end
    except Exception:
        # fallback: let yfinance parse strings
        start_dt = pd.to_datetime(start).to_pydatetime()
        end_dt = pd.to_datetime(end).to_pydatetime()

    # Determine TTL based on whether this is historical data
    is_historical = _is_historical_range(end_dt)
    cache_ttl = CACHE_TTL_HISTORICAL if is_historical else CACHE_TTL_LIVE
    
    # Format dates for cache keys
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")
    
    # Try to get cached data
    cache = _get_price_cache() if use_cache else None
    cached_data = {}
    tickers_to_download = []
    
    if cache:
        for ticker in tickers:
            cached_df = cache.get_prices(ticker, start_str, end_str, interval)
            if cached_df is not None and not cached_df.empty:
                cached_data[ticker] = cached_df
            else:
                tickers_to_download.append(ticker)
        
        if cached_data:
            logger.debug(f"Cache hit: {len(cached_data)} tickers, downloading: {len(tickers_to_download)}")
    else:
        tickers_to_download = tickers
    
    # Download missing tickers
    downloaded_data = pd.DataFrame()
    if tickers_to_download:
        end_dt_download = end_dt + timedelta(days=1)  # yfinance end is exclusive
        
        last_error = None
        max_retries, backoff = _retry_params(retry_cfg)
        for attempt in range(max_retries):
            try:
                downloaded_data = yf.download(
                    tickers=tickers_to_download,
                    start=start_dt,
                    end=end_dt_download,
                    interval=interval,
                    group_by="ticker",
                    auto_adjust=auto_adjust,
                    threads=threads,
                    progress=progress,
                )
                last_error = None
                break
            except Exception as e:
                last_error = e
                time.sleep(backoff * (attempt + 1))
        
        if last_error is not None and not cached_data:
            # Complete failure and no cache
            reason = _categorize_yf_error(last_error)
            report = {
                "bad_tickers": tickers,
                "reasons": {t: f"{reason}: {str(last_error)}" for t in tickers},
            }
            _maybe_quarantine(report, quarantine_cfg, source="yfinance")
            return pd.DataFrame(), report

    # Validate and cache downloaded data
    bad_tickers = []
    reasons = {}
    
    # Process downloaded tickers
    for ticker in tickers_to_download:
        df = get_ticker_df(downloaded_data, ticker)
        if df.empty:
            bad_tickers.append(ticker)
            reasons[ticker] = "No data returned"
            continue
        
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            bad_tickers.append(ticker)
            reasons[ticker] = f"Missing columns: {missing_cols}"
            continue
        
        invalid_bars = (df["High"] < df["Low"]).sum()
        if invalid_bars > 0:
            reasons[ticker] = f"{invalid_bars} invalid bars (high < low)"
        
        if df["Volume"].isna().all():
            bad_tickers.append(ticker)
            reasons[ticker] = "All volume data missing"
            continue
        
        # Cache valid data
        if cache and not df.empty:
            cache.set_prices(ticker, start_str, end_str, df, interval)
    
    # Combine cached and downloaded data
    # For cached data, we already validated it when it was first cached
    # Return format: the raw yfinance DataFrame format for compatibility
    # Note: callers typically use get_ticker_df() to extract per-ticker data
    
    report = {"bad_tickers": bad_tickers, "reasons": reasons, "cache_hits": len(cached_data)}
    _maybe_quarantine(report, quarantine_cfg, source="yfinance")
    return downloaded_data, report


def download_daily_range_cached(
    tickers: list[str],
    start: str | datetime,
    end: str | datetime,
    *,
    interval: str = "1d",
    auto_adjust: bool = False,
    threads: bool = True,
    progress: bool = False,
    quarantine_cfg: Optional[dict] = None,
    retry_cfg: Optional[dict] = None,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """
    Download daily OHLCV data with full caching support.
    
    Returns a dict mapping ticker -> DataFrame instead of raw yfinance output.
    This is optimized for backtesting where you need:
    - Reproducible results (same data every time)
    - Efficient reuse across multiple runs
    - Per-ticker DataFrames ready to use
    
    Historical data (end > 3 days ago) is cached for 30 days.
    """
    if not tickers:
        return {}, {"bad_tickers": [], "reasons": {}}

    try:
        start_dt = pd.to_datetime(start).to_pydatetime() if not isinstance(start, datetime) else start
        end_dt = pd.to_datetime(end).to_pydatetime() if not isinstance(end, datetime) else end
    except Exception:
        start_dt = pd.to_datetime(start).to_pydatetime()
        end_dt = pd.to_datetime(end).to_pydatetime()

    # Determine TTL based on whether this is historical data
    is_historical = _is_historical_range(end_dt)
    cache_ttl = CACHE_TTL_HISTORICAL if is_historical else CACHE_TTL_LIVE
    
    # Format dates for cache keys
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")
    
    # Try to get cached data
    cache = _get_price_cache()
    result_data = {}
    tickers_to_download = []
    
    # Calculate expected trading days in range (roughly 252/365 = ~69% of calendar days are trading days)
    requested_days = (end_dt - start_dt).days
    min_expected_trading_days = max(10, int(requested_days * 0.6))  # At least 60% of calendar days
    
    if cache:
        for ticker in tickers:
            cached_df = cache.get_prices(ticker, start_str, end_str, interval)
            if cached_df is not None and not cached_df.empty:
                # Validate cache has sufficient coverage (at least 60% of expected trading days)
                if len(cached_df) >= min_expected_trading_days:
                    result_data[ticker] = cached_df
                else:
                    # Cache has data but insufficient coverage - need to re-download
                    tickers_to_download.append(ticker)
            else:
                tickers_to_download.append(ticker)
        
        if result_data:
            logger.info(f"Price cache: {len(result_data)} hits, {len(tickers_to_download)} to download")
    else:
        tickers_to_download = tickers
    
    # Download missing tickers
    bad_tickers = []
    reasons = {}
    
    if tickers_to_download:
        end_dt_download = end_dt + timedelta(days=1)  # yfinance end is exclusive
        
        downloaded_data = pd.DataFrame()
        last_error = None
        max_retries, backoff = _retry_params(retry_cfg)
        for attempt in range(max_retries):
            try:
                downloaded_data = yf.download(
                    tickers=tickers_to_download,
                    start=start_dt,
                    end=end_dt_download,
                    interval=interval,
                    group_by="ticker",
                    auto_adjust=auto_adjust,
                    threads=threads,
                    progress=progress,
                )
                last_error = None
                break
            except Exception as e:
                last_error = e
                time.sleep(backoff * (attempt + 1))
        
        if last_error is not None:
            reason = _categorize_yf_error(last_error)
            for t in tickers_to_download:
                if t not in result_data:
                    bad_tickers.append(t)
                    reasons[t] = f"{reason}: {str(last_error)}"
        else:
            # Process downloaded tickers
            for ticker in tickers_to_download:
                df = get_ticker_df(downloaded_data, ticker)
                if df.empty:
                    bad_tickers.append(ticker)
                    reasons[ticker] = "No data returned"
                    continue
                
                required_cols = ["Open", "High", "Low", "Close", "Volume"]
                missing_cols = [c for c in required_cols if c not in df.columns]
                if missing_cols:
                    bad_tickers.append(ticker)
                    reasons[ticker] = f"Missing columns: {missing_cols}"
                    continue
                
                if df["Volume"].isna().all():
                    bad_tickers.append(ticker)
                    reasons[ticker] = "All volume data missing"
                    continue
                
                # Valid data - add to result and cache
                result_data[ticker] = df
                
                if cache:
                    cache.set_prices(ticker, start_str, end_str, df, interval)

    report = {
        "bad_tickers": bad_tickers, 
        "reasons": reasons, 
        "cache_hits": len(result_data) - len(tickers_to_download) + len(bad_tickers),
        "downloaded": len(tickers_to_download) - len(bad_tickers),
    }
    _maybe_quarantine(report, quarantine_cfg, source="yfinance")
    return result_data, report


def prefetch_historical_prices(
    tickers: list[str],
    start: str | datetime,
    end: str | datetime,
    batch_size: int = 100,
) -> dict:
    """
    Pre-populate cache with historical prices for backtesting.
    
    Call this once before running backtests to ensure all data is cached.
    Subsequent backtest runs will be instant (no API calls).
    
    Args:
        tickers: List of tickers to prefetch
        start: Start date
        end: End date  
        batch_size: Number of tickers per batch download
    
    Returns:
        Summary dict with counts
    """
    logger.info(f"Prefetching historical prices for {len(tickers)} tickers from {start} to {end}")
    
    total_cached = 0
    total_downloaded = 0
    total_failed = 0
    
    # Process in batches
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        data, report = download_daily_range_cached(
            batch, 
            start, 
            end,
            progress=False,
        )
        
        total_cached += report.get("cache_hits", 0)
        total_downloaded += report.get("downloaded", 0)
        total_failed += len(report.get("bad_tickers", []))
        
        logger.info(f"  Batch {i//batch_size + 1}: {len(data)} ok, {len(report.get('bad_tickers', []))} failed")
    
    summary = {
        "total_tickers": len(tickers),
        "from_cache": total_cached,
        "downloaded": total_downloaded,
        "failed": total_failed,
    }
    
    logger.info(f"Prefetch complete: {summary}")
    return summary

