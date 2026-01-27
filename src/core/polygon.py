"""
Polygon (Massive) data helpers.

Centralized helpers to fetch OHLCV and splits from Polygon so that pipelines
can prefer Polygon as the primary source and fall back to Yahoo Finance when
needed.

Includes caching layer to avoid redundant API calls within the same session.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Cache TTL for live data (1 hour)
LIVE_CACHE_TTL = 3600


def _retry_params(cfg: Optional[dict]) -> tuple[int, float]:
    if not cfg:
        return 3, 0.75
    return int(cfg.get("max_retries", 3)), float(cfg.get("backoff_sec", 0.75))


def _categorize_polygon_error(exc: Exception) -> str:
    try:
        if hasattr(exc, "response") and exc.response is not None:
            status = exc.response.status_code
            if status == 429:
                return "rate_limit"
            if status == 403:
                return "forbidden"
            if status == 404:
                return "not_found"
            if status >= 500:
                return "server_error"
            return f"http_{status}"
    except Exception:
        pass
    return "request_error"


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


def intraday_rvol_polygon(ticker: str, params: dict, api_key: str) -> float:
    """Estimate intraday RVOL using Polygon 5m aggregates."""
    try:
        interval_min = int(params.get("polygon_intraday_interval", 5))
        lookback_days = int(params.get("polygon_intraday_lookback_days", 5))
        if interval_min <= 0 or lookback_days <= 0:
            return np.nan

        now_ny = pd.Timestamp.now(tz="America/New_York")
        if params.get("market_open_buffer_min") and _minutes_since_open(now_ny) < int(
            params.get("market_open_buffer_min", 20)
        ):
            return np.nan

        end_date = now_ny.normalize()
        start_date = end_date - pd.Timedelta(days=lookback_days)
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{interval_min}/minute/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        resp = requests.get(
            url,
            params={
                "adjusted": "true",
                "sort": "asc",
                "limit": 50000,
                "apiKey": api_key,
            },
            timeout=8,
        )
        resp.raise_for_status()
        results = resp.json().get("results") or []
        if not results:
            return np.nan
        df = pd.DataFrame(results)
        if df.empty or not {"t", "v"}.issubset(df.columns):
            return np.nan
        df["ts"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("America/New_York")
        df = df.dropna(subset=["ts", "v"])
        df["date"] = df["ts"].dt.date
        df["time"] = df["ts"].dt.time

        today = now_ny.date()
        today_df = df[df["date"] == today]
        hist_df = df[df["date"] < today]
        if today_df.empty or hist_df.empty:
            return np.nan

        t_cut = now_ny.time()
        v_today = float(today_df[today_df["time"] <= t_cut]["v"].sum())

        exp = []
        for _, g in hist_df.groupby("date"):
            g2 = g[g["time"] <= t_cut]
            if len(g2) >= 3:
                exp.append(float(g2["v"].sum()))
        if len(exp) < 2:
            return np.nan
        v_exp = float(np.median(exp))
        if v_exp <= 0:
            return np.nan
        return v_today / v_exp
    except Exception:
        return np.nan


def fetch_polygon_daily(
    ticker: str, 
    lookback_days: int, 
    asof_date: Optional[str], 
    api_key: Optional[str],
    use_cache: bool = True,
    retry_cfg: Optional[dict] = None,
    return_meta: bool = False,
) -> pd.DataFrame:
    """Fetch adjusted daily OHLCV from Polygon for a single ticker.
    
    Uses cache to avoid redundant API calls within the same session (1 hour TTL).
    """
    if not api_key:
        return pd.DataFrame()
    
    meta = {"reason": None}
    try:
        end_dt = pd.to_datetime(asof_date) if asof_date else pd.Timestamp.utcnow().normalize()
        start_dt = end_dt - pd.Timedelta(days=lookback_days + 5)
        start_str = start_dt.strftime('%Y-%m-%d')
        end_str = end_dt.strftime('%Y-%m-%d')
        
        # Check cache first
        cache = _get_price_cache() if use_cache else None
        if cache:
            cached_df = cache.get_prices(ticker, start_str, end_str, "1d")
            if cached_df is not None and not cached_df.empty:
                return (cached_df, meta) if return_meta else cached_df
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_str}/{end_str}"
        max_retries, backoff = _retry_params(retry_cfg)
        last_error = None
        for attempt in range(max_retries):
            try:
                resp = requests.get(
                    url,
                    params={"adjusted": "true", "sort": "asc", "apiKey": api_key},
                    timeout=10,
                )
                resp.raise_for_status()
                results = resp.json().get("results") or []
                if not results:
                    meta["reason"] = "no_data"
                    return (pd.DataFrame(), meta) if return_meta else pd.DataFrame()
                df = pd.DataFrame(results)
                if df.empty or not {"o", "h", "l", "c", "v", "t"}.issubset(df.columns):
                    meta["reason"] = "missing_columns"
                    return (pd.DataFrame(), meta) if return_meta else pd.DataFrame()
                df["Date"] = pd.to_datetime(df["t"], unit="ms")
                df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
                df = df.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]].dropna()
                if cache and not df.empty:
                    cache.set_prices(ticker, start_str, end_str, df, "1d", ttl_seconds=LIVE_CACHE_TTL)
                return (df, meta) if return_meta else df
            except Exception as e:
                last_error = e
                meta["reason"] = _categorize_polygon_error(e)
                if attempt < max_retries - 1:
                    import time
                    time.sleep(backoff * (attempt + 1))
                    continue
                break
        if last_error is not None:
            return (pd.DataFrame(), meta) if return_meta else pd.DataFrame()
        return (pd.DataFrame(), meta) if return_meta else pd.DataFrame()
    except Exception:
        meta["reason"] = "request_error"
        return (pd.DataFrame(), meta) if return_meta else pd.DataFrame()


def download_polygon_batch(
    tickers: list[str],
    lookback_days: int,
    asof_date: Optional[str],
    api_key: Optional[str],
    max_workers: int = 8,
    use_cache: bool = True,
    quarantine_cfg: Optional[dict] = None,
    retry_cfg: Optional[dict] = None,
) -> dict[str, pd.DataFrame]:
    """Download daily OHLCV for many tickers from Polygon.
    
    Uses cache to avoid redundant API calls. Cached data is reused within 1 hour.
    """
    if not api_key or not tickers:
        return {}
    
    results: dict[str, pd.DataFrame] = {}
    bad_tickers: list[str] = []
    reasons: dict[str, str] = {}
    cache_hits = 0
    api_calls = 0
    
    # Check cache for all tickers first
    cache = _get_price_cache() if use_cache else None
    tickers_to_fetch = []
    
    if cache:
        end_dt = pd.to_datetime(asof_date) if asof_date else pd.Timestamp.utcnow().normalize()
        start_dt = end_dt - pd.Timedelta(days=lookback_days + 5)
        start_str = start_dt.strftime('%Y-%m-%d')
        end_str = end_dt.strftime('%Y-%m-%d')
        
        for ticker in tickers:
            cached_df = cache.get_prices(ticker, start_str, end_str, "1d")
            if cached_df is not None and not cached_df.empty:
                results[ticker] = cached_df
                cache_hits += 1
            else:
                tickers_to_fetch.append(ticker)
    else:
        tickers_to_fetch = tickers
    
    # Log cache performance
    if cache_hits > 0:
        logger.info(f"Polygon cache: {cache_hits} hits, {len(tickers_to_fetch)} to fetch")
    
    # Fetch remaining tickers from API
    if tickers_to_fetch:
        def worker(t: str):
            df, meta = fetch_polygon_daily(
                t,
                lookback_days,
                asof_date,
                api_key,
                use_cache=use_cache,
                retry_cfg=retry_cfg,
                return_meta=True,
            )
            return t, df, meta

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(worker, t): t for t in tickers_to_fetch}
            for fut in as_completed(futures):
                t, df, meta = fut.result()
                results[t] = df
                if not df.empty:
                    api_calls += 1
                else:
                    bad_tickers.append(t)
                    reason = (meta or {}).get("reason") or "no_data"
                    reasons[t] = reason
    
    if bad_tickers:
        _maybe_quarantine({"bad_tickers": bad_tickers, "reasons": reasons}, quarantine_cfg, source="polygon")
    return results


def fetch_polygon_splits(ticker: str, api_key: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> list[dict]:
    """Fetch splits within a window from Polygon."""
    try:
        url = "https://api.polygon.io/v3/reference/splits"
        resp = requests.get(
            url,
            params={
                "ticker": ticker,
                "execution_date.gte": start_date.strftime("%Y-%m-%d"),
                "execution_date.lte": end_date.strftime("%Y-%m-%d"),
                "limit": 50,
                "order": "desc",
                "apiKey": api_key,
            },
            timeout=6,
        )
        resp.raise_for_status()
        return resp.json().get("results") or []
    except Exception:
        return []


def has_recent_split(ticker: str, api_key: str, lookback_days: int, block_days: int, asof_date: Optional[str]) -> bool:
    """Detect if a split occurred within block_days (drops ticker)."""
    try:
        end_dt = pd.to_datetime(asof_date) if asof_date else pd.Timestamp.now(tz="America/New_York")
        end_dt = end_dt.normalize()
        start_dt = end_dt - pd.Timedelta(days=lookback_days)
        splits = fetch_polygon_splits(ticker, api_key, start_dt, end_dt)
        if not splits:
            return False
        cutoff = end_dt - pd.Timedelta(days=block_days)
        for s in splits:
            exec_dt = s.get("execution_date")
            if not exec_dt:
                continue
            try:
                d = pd.to_datetime(exec_dt)
                if d >= cutoff:
                    return True
            except Exception:
                continue
        return False
    except Exception:
        return False


def _minutes_since_open(now_ny: pd.Timestamp) -> float:
    """Minutes since NY market open (09:30)."""
    market_open = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
    delta = now_ny - market_open
    return delta.total_seconds() / 60.0


def fetch_polygon_daily_range(
    ticker: str,
    start: str,
    end: str,
    api_key: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch adjusted daily OHLCV from Polygon for a date range.
    
    Uses cache to avoid redundant API calls.
    Historical data (end > 3 days ago) is cached for 30 days.
    """
    if not api_key:
        return pd.DataFrame()
    
    try:
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        start_str = start_dt.strftime('%Y-%m-%d')
        end_str = end_dt.strftime('%Y-%m-%d')
        
        # Determine TTL based on whether this is historical data
        now = pd.Timestamp.utcnow()
        is_historical = (now - end_dt).days > 3
        cache_ttl = 86400 * 30 if is_historical else LIVE_CACHE_TTL  # 30 days for historical
        
        # Check cache first
        cache = _get_price_cache() if use_cache else None
        if cache:
            cached_df = cache.get_prices(ticker, start_str, end_str, "1d")
            if cached_df is not None and not cached_df.empty:
                # Validate cache has sufficient coverage
                requested_days = (end_dt - start_dt).days
                min_expected = max(10, int(requested_days * 0.6))
                if len(cached_df) >= min_expected:
                    return cached_df
        
        # Fetch from Polygon API
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_str}/{end_str}"
        resp = requests.get(
            url,
            params={"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": api_key},
            timeout=15,
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
        
        # Store in cache
        if cache and not df.empty:
            cache.set_prices(ticker, start_str, end_str, df, "1d", ttl_seconds=cache_ttl)
        
        return df
    except Exception as e:
        logger.debug(f"Polygon fetch failed for {ticker}: {e}")
        return pd.DataFrame()


def prefetch_polygon_historical(
    tickers: list[str],
    start: str,
    end: str,
    api_key: str,
    batch_size: int = 100,
    max_workers: int = 8,
) -> dict:
    """
    Pre-populate cache with historical prices from Polygon for backtesting.
    
    Call this once before running backtests to ensure all data is cached.
    Subsequent backtest runs will be instant (no API calls).
    
    Args:
        tickers: List of tickers to prefetch
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        api_key: Polygon API key
        batch_size: Number of tickers per batch (for progress reporting)
        max_workers: Max concurrent API requests
    
    Returns:
        Summary dict with counts
    """
    logger.info(f"Prefetching historical prices (Polygon) for {len(tickers)} tickers from {start} to {end}")
    
    if not api_key:
        logger.error("No Polygon API key provided")
        return {
            "total_tickers": len(tickers),
            "from_cache": 0,
            "downloaded": 0,
            "failed": len(tickers),
        }
    
    total_cached = 0
    total_downloaded = 0
    total_failed = 0
    
    # Check cache for all tickers first
    cache = _get_price_cache()
    tickers_to_fetch = []
    
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    start_str = start_dt.strftime('%Y-%m-%d')
    end_str = end_dt.strftime('%Y-%m-%d')
    requested_days = (end_dt - start_dt).days
    min_expected = max(10, int(requested_days * 0.6))
    
    if cache:
        for ticker in tickers:
            cached_df = cache.get_prices(ticker, start_str, end_str, "1d")
            if cached_df is not None and not cached_df.empty and len(cached_df) >= min_expected:
                total_cached += 1
            else:
                tickers_to_fetch.append(ticker)
        
        if total_cached > 0:
            logger.info(f"Polygon cache: {total_cached} hits, {len(tickers_to_fetch)} to fetch")
    else:
        tickers_to_fetch = tickers
    
    # Fetch remaining tickers from API in batches
    for i in range(0, len(tickers_to_fetch), batch_size):
        batch = tickers_to_fetch[i:i + batch_size]
        batch_ok = 0
        batch_fail = 0
        
        def worker(t: str):
            df = fetch_polygon_daily_range(t, start, end, api_key, use_cache=True)
            return t, df
        
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(worker, t): t for t in batch}
            for fut in as_completed(futures):
                t, df = fut.result()
                if df is not None and not df.empty:
                    batch_ok += 1
                else:
                    batch_fail += 1
        
        total_downloaded += batch_ok
        total_failed += batch_fail
        
        logger.info(f"  Batch {i//batch_size + 1}: {batch_ok} ok, {batch_fail} failed")
    
    summary = {
        "total_tickers": len(tickers),
        "from_cache": total_cached,
        "downloaded": total_downloaded,
        "failed": total_failed,
    }
    
    logger.info(f"Polygon prefetch complete: {summary}")
    return summary
