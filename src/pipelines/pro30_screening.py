"""
Pro30 Screening Logic

Extracted from momentum_screener.py - handles attention pool, screening, and packet building.
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta
from typing import Optional
from ..core.technicals import atr, rsi, sma
from ..core.yf import get_ticker_df, download_daily, download_daily_range, download_daily_range_cached
from ..core.helpers import get_next_earnings_date
from ..core.analysis import analyze_headlines
from ..core.polygon import (
    intraday_rvol_polygon,
    fetch_polygon_daily,
    download_polygon_batch,
    has_recent_split,
)

# PR1: Data validation and quality ledger
from ..core.asof import validate_ohlcv, enforce_asof
from ..core.quality_ledger import QualityLedger, LedgerRow


# Leveraged ETFs (treated separately)
LEVERAGED_ETFS = set()  # Can be populated dynamically if needed


def _minutes_since_open(ts_local: pd.Timestamp) -> int:
    """Calculate minutes since market open (9:30 AM ET)."""
    open_time = ts_local.normalize() + pd.Timedelta(hours=9, minutes=30)
    return int(max(0, (ts_local - open_time).total_seconds() // 60))


def _intraday_prorated_rvol(ticker: str, params: dict) -> float:
    """
    Estimate intraday RVOL using prorated volume.
    
    Args:
        ticker: Ticker symbol
        params: Dict with intraday parameters
    
    Returns:
        RVOL estimate or NaN on failure
    """
    try:
        # Prefer Massive/Polygon intraday aggregates when enabled and live (no asof_date)
        if params.get("enable_polygon_intraday") and params.get("polygon_api_key") and not params.get("asof_date"):
            val = intraday_rvol_polygon(
                ticker=ticker,
                params=params,
                api_key=params.get("polygon_api_key"),
            )
            if np.isfinite(val):
                return float(val)

        interval = params.get("intraday_interval", "5m")
        days = int(params.get("intraday_lookback_days", 5))
        df = yf.download(ticker, period=f"{days}d", interval=interval, progress=False)
        if df is None or df.empty or "Volume" not in df.columns:
            return np.nan

        # Normalize timezone
        idx = pd.to_datetime(df.index, errors="coerce")
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert("America/New_York").tz_localize(None)
        df = df.copy()
        df.index = idx
        df = df.dropna()

        if len(df) < 50:
            return np.nan

        now_local = pd.Timestamp.now(tz="America/New_York").tz_localize(None)
        mins = _minutes_since_open(now_local)
        if mins < int(params.get("market_open_buffer_min", 20)):
            return np.nan

        # Split by date
        df["date"] = df.index.date
        today = now_local.date()
        today_df = df[df["date"] == today]
        hist_df = df[df["date"] != today]

        if today_df.empty or hist_df.empty:
            return np.nan

        # Cumulative volume so far today
        v_today = float(today_df["Volume"].sum())

        # Expected cumulative volume by same time-of-day
        t_cut = now_local.time()
        exp = []
        for d, g in hist_df.groupby("date"):
            g2 = g[g.index.time <= t_cut]
            if len(g2) >= 3:
                exp.append(float(g2["Volume"].sum()))
        if len(exp) < 2:
            return np.nan

        v_exp = float(np.median(exp))
        if v_exp <= 0:
            return np.nan

        return v_today / v_exp
    except Exception:
        return np.nan


def is_market_open() -> bool:
    """
    Rough check if US equity market is currently open (9:30 AM - 4:00 PM ET, weekdays).
    
    Returns:
        True if market is likely open, False otherwise
    """
    try:
        now = pd.Timestamp.now(tz="America/New_York")
        # Check if weekday (Monday=0, Sunday=6)
        if now.weekday() >= 5:  # Saturday or Sunday
            return False
        # Check time: 9:30 AM - 4:00 PM ET
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now <= market_close
    except Exception:
        # If timezone conversion fails, assume market might be open (conservative)
        return True


def build_attention_pool(tickers: list[str], params: dict) -> list[str]:
    """
    Build dynamic attention pool using objective signals (RVOL, ATR%, price moves).
    
    Args:
        tickers: List of ticker symbols to screen
        params: Dict with attention pool parameters
    
    Returns:
        List of tickers in attention pool
    """
    market_open = is_market_open()
    enable_intraday = params.get("enable_intraday_attention", False)
    allow_partial_day = params.get("allow_partial_day_attention", False)
    
    if market_open and not enable_intraday:
        if not allow_partial_day:
            print("[BLOCK] Market appears open and intraday attention is disabled. Skipping attention pool to avoid partial-day RVOL bias (use --intraday-attention or --allow-partial-day to override).")
            return []
        print("[WARNING] Market appears open; running EOD attention pool with partial-day data (allow_partial_day_attention=True). Results may be noisy.")
    elif market_open and enable_intraday:
        print("[INFO] Market is open, using intraday prorated RVOL for attention pool.")
    
    pool = []
    chunk_size = params.get("attention_chunk_size", 200)
    lookback_days = params.get("attention_lookback_days", 120)
    
    price_min = params.get("price_min", 5.0)
    avg_vol_min = params.get("avg_vol_min", 1_000_000)
    rvol_min = params.get("attention_rvol_min", 1.8)
    atr_pct_min = params.get("attention_atr_pct_min", 3.5)
    min_abs_day_move_pct = params.get("attention_min_abs_day_move_pct", None)

    print(f"Building attention pool from {len(tickers)} tickers (chunk size: {chunk_size})...")

    # Optional historical replay support
    asof_date = params.get("asof_date")  # "YYYY-MM-DD" or None
    use_polygon_primary = bool(params.get("attention_use_polygon", False) and params.get("polygon_api_key"))
    polygon_api_key = params.get("polygon_api_key")
    polygon_max_workers = params.get("polygon_max_workers", 8)

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i+chunk_size]
        try:
            # Use cached download for efficiency (shares data across pipeline steps)
            chunk_data_dict = {}
            polygon_data = {}
            if use_polygon_primary and polygon_api_key:
                try:
                    polygon_data = download_polygon_batch(
                        tickers=chunk,
                        lookback_days=lookback_days,
                        asof_date=asof_date,
                        api_key=polygon_api_key,
                        max_workers=polygon_max_workers,
                        quarantine_cfg=params.get("quarantine_cfg"),
                        retry_cfg=params.get("polygon_retry_cfg"),
                    )
                    for t, df in polygon_data.items():
                        if df is not None and not df.empty:
                            chunk_data_dict[t] = df
                    if i == 0:
                        populated = sum(1 for df in polygon_data.values() if df is not None and not df.empty)
                        print(f"  Attention pool polygon: {populated}/{len(chunk)} tickers populated.")
                except Exception as e:
                    if i == 0:
                        print(f"  [WARN] Attention pool polygon failed: {e}")
            if asof_date:
                end_dt = pd.to_datetime(asof_date)
                start_dt = end_dt - pd.Timedelta(days=int(lookback_days) + 20)
                missing = [t for t in chunk if t not in chunk_data_dict]
                if missing:
                    yfd, report = download_daily_range_cached(
                        tickers=missing,
                        start=start_dt.to_pydatetime(),
                        end=end_dt.to_pydatetime(),
                        auto_adjust=False,
                        threads=True,
                        quarantine_cfg=params.get("quarantine_cfg"),
                        retry_cfg=params.get("yfinance_retry_cfg"),
                    )
                    for t, df in yfd.items():
                        if df is not None and not df.empty:
                            chunk_data_dict[t] = df
                    cache_hits = report.get("cache_hits", 0)
                    if cache_hits > 0 and i == 0:  # Log only for first chunk
                        print(f"  Attention pool cache: {cache_hits} hits, {report.get('downloaded', 0)} to fetch per chunk")
            else:
                # For live runs without asof_date, use standard download
                missing = [t for t in chunk if t not in chunk_data_dict]
                if missing:
                    data, _ = download_daily(
                        tickers=missing,
                        period=f"{lookback_days}d",
                        interval="1d",
                        auto_adjust=False,
                        threads=True,
                        quarantine_cfg=params.get("quarantine_cfg"),
                        retry_cfg=params.get("yfinance_retry_cfg"),
                    )
                    for t in missing:
                        df = get_ticker_df(data, t)
                        if df is not None and not df.empty:
                            chunk_data_dict[t] = df

            for t in chunk:
                try:
                    # Get data from dict (cached) or raw DataFrame format
                    df = chunk_data_dict.get(t, pd.DataFrame())
                    if df.empty or len(df) < 40:
                        continue

                    close = df["Close"]
                    vol = df["Volume"]
                    
                    last = float(close.iloc[-1])
                    if last < price_min:
                        continue

                    avg20 = float(vol.tail(20).mean())
                    if avg20 < avg_vol_min:
                        continue

                    rvol = float(vol.iloc[-1] / avg20) if avg20 else np.nan
                    
                    # Intraday mode override if enabled
                    if enable_intraday:
                        rvol_i = _intraday_prorated_rvol(t, params)
                        if not np.isnan(rvol_i):
                            rvol = float(rvol_i)
                            if rvol < params.get("intraday_rvol_min", 2.0):
                                continue
                    
                    if np.isnan(rvol) or rvol < rvol_min:
                        continue

                    a = float(atr(df, 14).iloc[-1])
                    atr_pct_val = float(a / last * 100) if last else np.nan
                    if np.isnan(atr_pct_val) or atr_pct_val < atr_pct_min:
                        continue

                    if min_abs_day_move_pct is not None and len(df) >= 2:
                        prev = float(close.iloc[-2])
                        day_move = abs((last - prev) / prev * 100) if prev else 0.0
                        if day_move < min_abs_day_move_pct:
                            continue

                    pool.append(t)
                except Exception:
                    continue
        except Exception:
            continue

        # Progress indicator
        if (i + chunk_size) % (chunk_size * 5) == 0:
            print(f"  Processed {min(i + chunk_size, len(tickers))}/{len(tickers)} tickers, found {len(pool)} in pool so far...")

    return sorted(set(pool))


def screen_universe_30d(tickers: list[str], params: dict, regime: str | None = None) -> dict:
    """
    Screen universe for 30-day momentum candidates (breakout and reversal setups).

    Args:
        tickers: List of ticker symbols to screen
        params: Dict with screening parameters
        regime: Optional market regime ("bull", "chop", "stress"). When "stress",
                reversal setups get a higher bonus and breakout setups get a lower bonus
                per regime_overrides config.

    Returns:
        Dict with keys: breakout_df, reversal_df, combined_df, ledger
    """
    if not tickers:
        return {"breakout_df": pd.DataFrame(), "reversal_df": pd.DataFrame(), "combined_df": pd.DataFrame(), "ledger": QualityLedger()}
    
    # PR1: Initialize quality ledger
    ledger = QualityLedger()
    asof_date_str = params.get("asof_date")  # May be None for live runs

    print(f"  Loading price data for {len(tickers)} tickers...")
    asof_date = params.get("asof_date")
    polygon_api_key = params.get("polygon_api_key")
    use_polygon_primary = bool(params.get("enable_polygon_primary") and polygon_api_key)
    use_polygon_fallback = bool(params.get("enable_polygon_fallback") and polygon_api_key)
    lookback_days = int(params.get("lookback_days", 300))

    # Calculate date range
    if asof_date:
        end_date = pd.to_datetime(asof_date).strftime("%Y-%m-%d")
    else:
        end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
    start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=lookback_days + 20)).strftime("%Y-%m-%d")

    # ═══════════════════════════════════════════════════════════════════════════════
    # STEP 1: Check permanent price database FIRST (no API calls needed)
    # ═══════════════════════════════════════════════════════════════════════════════
    db_data: dict[str, pd.DataFrame] = {}
    tickers_need_download = []
    db = None
    
    try:
        from src.core.price_db import get_price_db
        db = get_price_db()
        
        for t in tickers:
            try:
                raw_df = db.get_prices(t, start_date, end_date)
                if raw_df is not None and not raw_df.empty:
                    # PR1: Validate and enforce as-of
                    clean_df, vstats = validate_ohlcv(raw_df, t)
                    clean_df, astats = enforce_asof(clean_df, asof_date_str, t, strict=False)
                    
                    if not clean_df.empty and len(clean_df) >= 120:
                        db_data[t] = clean_df
                        ledger.add(LedgerRow(
                            ticker=t,
                            stage="load_prices",
                            provider_used="price_db",
                            rows=len(clean_df),
                            first_date=str(clean_df.index.min().date()) if not clean_df.empty else None,
                            last_date=str(clean_df.index.max().date()) if not clean_df.empty else None,
                            missing_cols=";".join(vstats.get("missing_cols") or []) or None,
                            dropped_bad_rows=vstats.get("dropped_bad_rows"),
                            dropped_future_rows=astats.get("dropped_future_rows"),
                        ))
                    else:
                        tickers_need_download.append(t)
                else:
                    tickers_need_download.append(t)
            except Exception as e:
                ledger.add_exception(t, "load_prices", e, "price_db")
                tickers_need_download.append(t)
        
        if db_data:
            print(f"  Database: {len(db_data)} tickers loaded, {len(tickers_need_download)} need download")
    except Exception as e:
        print(f"  [WARN] Price database not available: {e}")
        tickers_need_download = tickers

    # ═══════════════════════════════════════════════════════════════════════════════
    # STEP 2: Download missing tickers (Polygon primary, yfinance fallback)
    # ═══════════════════════════════════════════════════════════════════════════════
    polygon_data: dict[str, pd.DataFrame] = {}
    if tickers_need_download and use_polygon_primary:
        print(f"  Fetching from Polygon for {len(tickers_need_download)} tickers...")
        polygon_data = download_polygon_batch(
            tickers=tickers_need_download,
            lookback_days=lookback_days,
            asof_date=asof_date,
            api_key=polygon_api_key,
            max_workers=params.get("polygon_max_workers", 8),
            quarantine_cfg=params.get("quarantine_cfg"),
            retry_cfg=params.get("polygon_retry_cfg"),
        )
        populated = sum(1 for df in polygon_data.values() if df is not None and not df.empty)
        print(f"  Polygon: {populated}/{len(tickers_need_download)} tickers populated.")
        
        # Store in permanent database for future use
        if db:
            try:
                for t, df in polygon_data.items():
                    if df is not None and not df.empty:
                        db.store_prices(t, df, source="polygon")
            except Exception:
                pass

    # Lazy yfinance backup (uses cache for efficiency)
    yf_data = None
    yf_data_dict = None  # Cached dict format
    def _ensure_yf_data():
        nonlocal yf_data, yf_data_dict
        if yf_data is not None or yf_data_dict is not None:
            return yf_data if yf_data is not None else yf_data_dict
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            if asof_date:
                end_dt = pd.to_datetime(asof_date)
                start_dt = end_dt - pd.Timedelta(days=int(params["lookback_days"]) + 20)
                # Use cached version for efficiency
                yf_data_dict, report = download_daily_range_cached(
                    tickers=tickers,
                    start=start_dt.to_pydatetime(),
                    end=end_dt.to_pydatetime(),
                    auto_adjust=False,
                    threads=True,
                    quarantine_cfg=params.get("quarantine_cfg"),
                    retry_cfg=params.get("yfinance_retry_cfg"),
                )
                cache_hits = report.get("cache_hits", 0)
                downloaded = report.get("downloaded", 0)
                if cache_hits > 0:
                    print(f"  yfinance backup: {cache_hits} from cache, {downloaded} downloaded")
                else:
                    print("  yfinance backup downloaded for remaining tickers.")
                return yf_data_dict
            else:
                yf_data_local, _ = download_daily(
                    tickers=tickers,
                    period=f"{params['lookback_days']}d",
                    interval="1d",
                    auto_adjust=False,
                    threads=True,
                )
                yf_data = yf_data_local
                print("  yfinance backup downloaded for remaining tickers.")
        return yf_data

    if not use_polygon_primary:
        _ensure_yf_data()

    print(f"  Download complete. Processing tickers...")

    breakout_rows, reversal_rows = [], []
    dropped: list[dict] = []
    failed_count = 0
    total_tickers = len(tickers)
    progress_interval = max(50, total_tickers // 20)

    for idx, t in enumerate(tickers):
        if (idx + 1) % progress_interval == 0 or (idx + 1) == total_tickers:
            pct = ((idx + 1) / total_tickers) * 100
            print(f"  Progress: {idx + 1}/{total_tickers} ({pct:.1f}%) | Found: {len(breakout_rows)} breakouts, {len(reversal_rows)} reversals", end="\r")
        try:
            # Check permanent database first
            df = db_data.get(t, pd.DataFrame())
            
            # Fallback to polygon if not in database
            if df.empty:
                df = polygon_data.get(t, pd.DataFrame()) if use_polygon_primary else pd.DataFrame()
            
            # Fallback to yfinance
            if df.empty:
                yf_source = _ensure_yf_data()
                # Handle both dict format (from cached) and DataFrame format (from raw yfinance)
                if isinstance(yf_source, dict):
                    df = yf_source.get(t, pd.DataFrame())
                else:
                    df = get_ticker_df(yf_source, t)
                # Store in database for future use
                if db and df is not None and not df.empty:
                    try:
                        db.store_prices(t, df, source="yfinance")
                    except Exception:
                        pass
                        
            if (df.empty or len(df) < 120) and use_polygon_fallback:
                alt_df = fetch_polygon_daily(
                    ticker=t,
                    lookback_days=int(params.get("lookback_days", 300)),
                    asof_date=asof_date,
                    api_key=polygon_api_key,
                )
                if not alt_df.empty:
                    df = alt_df
                    # Store in database for future use
                    if db:
                        try:
                            db.store_prices(t, df, source="polygon")
                        except Exception:
                            pass
                            
            if df.empty or len(df) < 120:
                dropped.append({"ticker": t, "stage": "data", "reason": "empty_or_short_history"})
                continue

            close = df["Close"]
            high = df["High"]
            vol = df["Volume"]

            last = float(close.iloc[-1])
            if last < params["price_min"]:
                dropped.append({"ticker": t, "stage": "filters", "reason": f"price {last:.2f} < {params['price_min']}"})
                continue

            # Corporate actions: block recent splits to avoid distorted returns
            if params.get("block_recent_splits", True) and polygon_api_key:
                if has_recent_split(
                    ticker=t,
                    api_key=polygon_api_key,
                    lookback_days=int(params.get("split_lookback_days", 120)),
                    block_days=int(params.get("split_block_days", 7)),
                    asof_date=asof_date,
                ):
                    dropped.append({"ticker": t, "stage": "data", "reason": "recent_split"})
                    continue

            avg20 = float(vol.tail(20).mean())
            if avg20 < params["avg_vol_min"]:
                dropped.append({"ticker": t, "stage": "filters", "reason": f"avg_vol_20d {avg20:.0f} < {params['avg_vol_min']}"})
                continue

            adv20 = float((close.tail(20) * vol.tail(20)).mean())
            if adv20 < params.get("avg_dollar_vol_min", 0):
                dropped.append({"ticker": t, "stage": "filters", "reason": f"adv20 {adv20:.0f} < {params.get('avg_dollar_vol_min', 0)}"})
                continue

            rvol_val = float(vol.iloc[-1] / avg20) if avg20 else np.nan
            if np.isnan(rvol_val) or rvol_val < params["rvol_min"]:
                dropped.append({"ticker": t, "stage": "filters", "reason": f"rvol {rvol_val:.2f} < {params['rvol_min']}"})
                continue

            a = float(atr(df, 14).iloc[-1])
            atr_pct_val = float(a / last * 100) if last else np.nan
            if np.isnan(atr_pct_val) or atr_pct_val < params["atr_pct_min"]:
                dropped.append({"ticker": t, "stage": "filters", "reason": f"atr_pct {atr_pct_val:.2f} < {params['atr_pct_min']}"})
                continue

            rsi14_val = float(rsi(close, 14).iloc[-1])

            # 52W high: use High, not Close
            high_52w = float(high.tail(252).max()) if len(high) >= 252 else float(high.max())
            dist_52w_high_pct = float((high_52w - last) / high_52w * 100) if high_52w else np.nan

            # MA structure
            ma20 = float(sma(close, 20).iloc[-1]) if len(close) >= 20 else np.nan
            ma50 = float(sma(close, 50).iloc[-1]) if len(close) >= 50 else np.nan
            
            above_ma20 = last >= ma20 if np.isfinite(ma20) else False
            above_ma50 = last >= ma50 if np.isfinite(ma50) else False
            
            # Returns
            ret20 = _compute_pct_change(close, periods=20)
            ret5 = _compute_pct_change(close, periods=5)
            if use_polygon_fallback and (np.isnan(ret5) or np.isnan(ret20)):
                alt_df = fetch_polygon_daily(
                    ticker=t,
                    lookback_days=int(params.get("lookback_days", 300)),
                    asof_date=asof_date,
                    api_key=polygon_api_key,
                )
                if not alt_df.empty:
                    close = alt_df["Close"]
                    high = alt_df["High"]
                    vol = alt_df["Volume"]
                    last = float(close.iloc[-1])
                    ret20 = _compute_pct_change(close, periods=20)
                    ret5 = _compute_pct_change(close, periods=5)

            # Setup flags
            is_breakout = (
                dist_52w_high_pct <= params["near_high_max_pct"]
                and rsi14_val >= params.get("breakout_rsi_min", 0)
                and above_ma20
                and above_ma50
                and (np.isnan(ret20) or ret20 > 0)
                and (np.isnan(ret5) or ret5 > 0)
            )
            is_reversal = (
                rsi14_val <= params.get("reversal_rsi_max", params["rsi_reversal_max"])
                and (dist_52w_high_pct >= params.get("reversal_dist_to_high_min_pct", 0))
            )

            if not (is_breakout or is_reversal):
                dropped.append({"ticker": t, "stage": "setup", "reason": "did_not_meet_breakout_or_reversal"})
                continue

            # Scoring: TapeScore + StructureScore + SetupBonus
            # Regime-aware bonuses: in stress/bear regime, boost reversals, penalize breakouts
            _is_bear = regime in ("stress", "bear")
            _breakout_bonus = (
                float(params.get("regime_bear_breakout_bonus", 2.0))
                if _is_bear else 4.0
            )
            _reversal_bonus = (
                float(params.get("regime_bear_reversal_bonus", 6.0))
                if _is_bear else 3.0
            )

            tape_score = (rvol_val * 2.0) + (atr_pct_val * 1.4)

            if is_breakout:
                rsi_structure = max(0.0, (70 - abs(rsi14_val - 62))) / 20.0
            else:
                rsi_structure = max(0.0, (45 - abs(rsi14_val - 28))) / 20.0
            dist_structure = max(0.0, (100 - dist_52w_high_pct)) / 20.0
            ma_structure = (2.0 if above_ma20 and above_ma50 else (1.0 if above_ma20 else 0.0))

            structure_score = rsi_structure + (dist_structure * 0.5) + ma_structure

            if is_breakout:
                setup_bonus = _breakout_bonus
                score = tape_score + structure_score + setup_bonus
                breakout_rows.append({
                    "Ticker": t,
                    "Last": round(last, 2),
                    "RVOL": round(rvol_val, 2),
                    "ATR%": round(atr_pct_val, 2),
                    "RSI14": round(rsi14_val, 1),
                    "Dist_to_52W_High%": round(dist_52w_high_pct, 2),
                    "$ADV20": round(adv20, 0),
                    "MA20": round(ma20, 2) if np.isfinite(ma20) else np.nan,
                    "MA50": round(ma50, 2) if np.isfinite(ma50) else np.nan,
                    "Above_MA20": int(above_ma20),
                    "Above_MA50": int(above_ma50),
                    "Ret20d%": round(ret20, 2) if np.isfinite(ret20) else np.nan,
                    "Ret5d%": round(ret5, 2) if np.isfinite(ret5) else np.nan,
                    "Setup": "Breakout",
                    "Score": round(score, 2),
                    "Is_Leveraged_ETF": t in LEVERAGED_ETFS
                })

            if is_reversal:
                reversal_structure = min(6.0, dist_52w_high_pct / 8.0)
                setup_bonus = _reversal_bonus
                score = tape_score + (structure_score * 0.7) + reversal_structure + setup_bonus
                reversal_rows.append({
                    "Ticker": t,
                    "Last": round(last, 2),
                    "RVOL": round(rvol_val, 2),
                    "ATR%": round(atr_pct_val, 2),
                    "RSI14": round(rsi14_val, 1),
                    "Dist_to_52W_High%": round(dist_52w_high_pct, 2),
                    "$ADV20": round(adv20, 0),
                    "MA20": round(ma20, 2) if np.isfinite(ma20) else np.nan,
                    "MA50": round(ma50, 2) if np.isfinite(ma50) else np.nan,
                    "Above_MA20": int(above_ma20),
                    "Above_MA50": int(above_ma50),
                    "Ret20d%": round(ret20, 2) if np.isfinite(ret20) else np.nan,
                    "Ret5d%": round(ret5, 2) if np.isfinite(ret5) else np.nan,
                    "Setup": "Reversal",
                    "Score": round(score, 2),
                    "Is_Leveraged_ETF": t in LEVERAGED_ETFS
                })

        except Exception:
            failed_count += 1
            dropped.append({"ticker": t, "stage": "exception", "reason": "processing_error"})
            continue

    print()  # New line after progress
    if failed_count > 0:
        print(f"  Skipped {failed_count} ticker(s) due to download/data errors")
    print(f"  Screening complete: {len(breakout_rows)} breakout candidates, {len(reversal_rows)} reversal candidates")

    # Create DataFrames
    if breakout_rows:
        breakout_df = pd.DataFrame(breakout_rows).sort_values("Score", ascending=False).reset_index(drop=True)
    else:
        breakout_df = pd.DataFrame(columns=["Ticker", "Last", "RVOL", "ATR%", "RSI14", "Dist_to_52W_High%", "$ADV20", "MA20", "MA50", "Above_MA20", "Above_MA50", "Ret20d%", "Ret5d%", "Setup", "Score", "Is_Leveraged_ETF"])
    
    if reversal_rows:
        reversal_df = pd.DataFrame(reversal_rows).sort_values("Score", ascending=False).reset_index(drop=True)
    else:
        reversal_df = pd.DataFrame(columns=["Ticker", "Last", "RVOL", "ATR%", "RSI14", "Dist_to_52W_High%", "$ADV20", "MA20", "MA50", "Above_MA20", "Above_MA50", "Ret20d%", "Ret5d%", "Setup", "Score", "Is_Leveraged_ETF"])

    breakout_df = breakout_df.head(params.get("top_n_breakout", 15)) if not breakout_df.empty else breakout_df
    reversal_df = reversal_df.head(params.get("top_n_reversal", 15)) if not reversal_df.empty else reversal_df

    combined_df = pd.concat([breakout_df, reversal_df], ignore_index=True)
    combined_df = combined_df.sort_values("Score", ascending=False).reset_index(drop=True)
    combined_df = combined_df.head(params.get("top_n_total", 25)) if not combined_df.empty else combined_df

    return {"breakout_df": breakout_df, "reversal_df": reversal_df, "combined_df": combined_df, "dropped": dropped, "ledger": ledger}


def _safe_float(x, default: float = np.nan) -> float:
    """Safely convert to float."""
    try:
        return float(x)
    except Exception:
        return default


def _compute_pct_change(series: pd.Series, periods: int) -> float:
    """Compute percent change with forward-fill and NaN safety."""
    if series is None:
        return np.nan
    s = pd.to_numeric(series, errors="coerce").ffill()
    if len(s) <= periods:
        return np.nan
    val = s.pct_change(periods=periods).iloc[-1] * 100
    return float(val) if np.isfinite(val) else np.nan




def compute_catalyst_completeness(
    ticker: str,
    earnings_date: str,
    news_df: pd.DataFrame,
    manual_headlines_df: Optional[pd.DataFrame] = None,
    lookback_days: int = 14,
) -> dict:
    """
    Compute catalyst completeness score (0-100).
    
    Returns:
        Dict with completeness_score, penalties, headline_count_recent, has_manual, earnings_known
    """
    penalties = []
    earnings_known = (earnings_date != "Unknown")

    # Recent headline count
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=lookback_days)
    count_recent = 0
    if news_df is not None and not news_df.empty:
        tmp = news_df[news_df["Ticker"].astype(str).str.strip().eq(ticker)].copy()
        if "published_utc" in tmp.columns:
            tmp["published_utc"] = pd.to_datetime(tmp["published_utc"], utc=True, errors="coerce")
            tmp = tmp[tmp["published_utc"].notna()]
            tmp = tmp[tmp["published_utc"] >= cutoff]
        if "title" in tmp.columns:
            tmp = tmp[tmp["title"].astype(str).str.strip().ne("")]
            tmp = tmp.drop_duplicates(subset=["title"], keep="first")
        count_recent = int(len(tmp))

    has_manual = False
    if manual_headlines_df is not None and not manual_headlines_df.empty:
        m = manual_headlines_df[manual_headlines_df["Ticker"].astype(str).str.strip().eq(ticker)]
        has_manual = bool(len(m) > 0)

    # Base completeness
    score = 70

    if not earnings_known:
        score -= 20
        penalties.append("Earnings date Unknown (penalize catalyst clarity).")

    if count_recent == 0 and not has_manual:
        score -= 25
        penalties.append("No recent headlines found (penalize narrative visibility).")
    elif count_recent < 3 and not has_manual:
        score -= 10
        penalties.append("Few recent headlines (weak narrative visibility).")
    elif count_recent >= 8:
        score += 5

    if has_manual:
        score += 10

    score = int(max(0, min(100, score)))

    return {
        "completeness_score": score,
        "penalties": penalties,
        "headline_count_recent": count_recent,
        "has_manual": has_manual,
        "earnings_known": earnings_known,
    }


def standard_trade_plan_guidance(setup: str, last: float, atr_pct: float) -> dict:
    """
    Returns standardized trade plan scaffolding.
    
    Args:
        setup: "Breakout" or "Reversal"
        last: Last price
        atr_pct: ATR as percentage of price
    
    Returns:
        Dict with entry_template, stop_template, tp_template, size_template, atr_dollar_est
    """
    atr_pct = _safe_float(atr_pct, np.nan)
    last = _safe_float(last, np.nan)

    atr_d = (last * atr_pct / 100.0) if (np.isfinite(last) and np.isfinite(atr_pct)) else np.nan

    if setup == "Breakout":
        entry = "Entry: Break & hold above prior day high OR key resistance on ≥1.5× volume; avoid chasing if >2× ATR extension."
        stop = "Stop: 1.2× ATR below breakout level (or below prior day low if tighter and logical)."
        tp = "TPs: TP1 = +1.0× ATR, TP2 = +2.0× ATR; trail stop after TP1."
    else:  # Reversal
        entry = "Entry: Reclaim MA20 (or prior swing level) AND RSI curl up; confirm with a green day + elevated volume."
        stop = "Stop: Below recent swing low OR 1.0–1.3× ATR below entry (whichever is tighter and logical)."
        tp = "TPs: TP1 = MA50 / prior supply zone, TP2 = gap-fill / next resistance; reduce into strength."

    size = "Position sizing: risk 1–2% of account per trade. Shares = (AccountRisk$) / (StopDistance$)."

    return {
        "atr_dollar_est": atr_d,
        "entry_template": entry,
        "stop_template": stop,
        "tp_template": tp,
        "size_template": size,
    }


def build_pro30_llm_packet(
    ticker: str,
    metrics_row: pd.Series,
    news_df: pd.DataFrame,
    max_headlines: int,
    regime_info: dict,
    manual_headlines_df: Optional[pd.DataFrame] = None,
    source_tags: Optional[list[str]] = None,
) -> str:
    """
    Build LLM packet for Pro30 screener.
    
    Returns:
        Formatted prompt string for LLM analysis
    """
    earnings_date = get_next_earnings_date(ticker)

    # Filter news for this ticker
    if not news_df.empty:
        n = news_df[(news_df["Ticker"] == ticker) & news_df["title"].astype(str).str.strip().ne("")].copy()
        if "published_utc" in n.columns:
            n["published_utc"] = pd.to_datetime(n["published_utc"], utc=True, errors="coerce")
            n = n.sort_values("published_utc", ascending=False)
        n = n.drop_duplicates(subset=["title"], keep="first").head(max_headlines)
    else:
        n = pd.DataFrame()

    headlines = []
    headline_titles = []

    # Manual headlines first
    if manual_headlines_df is not None and not manual_headlines_df.empty:
        manual = manual_headlines_df[manual_headlines_df["Ticker"].astype(str).str.strip().eq(ticker)]
        if not manual.empty:
            for _, r in manual.iterrows():
                date_str = r.get("Date", "N/A")
                source = r.get("Source", "Manual")
                headline = r.get("Headline", "")
                if str(headline).strip():
                    headline_titles.append(str(headline).strip())
                    headlines.append(f"- [{date_str}] {source}: {headline}")

    # yfinance headlines
    if not n.empty:
        for _, r in n.iterrows():
            ts = r.get("published_utc", pd.NaT)
            ts_str = ts.strftime("%Y-%m-%d %H:%M UTC") if pd.notna(ts) else "N/A"
            publisher = (r.get("publisher", "") or "").strip()
            title = (r.get("title", "") or "").strip()
            link = (r.get("link", "") or "").strip()
            if title:
                headline_titles.append(title)
                if link:
                    headlines.append(f"- [{ts_str}] {publisher}: {title}\n  {link}")
                else:
                    headlines.append(f"- [{ts_str}] {publisher}: {title}")

    if not headlines:
        headlines.append("- (No headlines returned. Treat catalysts as Unknown. Check: earnings guidance, SEC filings, major contracts, sector/BTC correlation, terminal/X headlines.)")
    
    # Analyze headlines
    flags = analyze_headlines(headline_titles)

    # Completeness score
    comp = compute_catalyst_completeness(
        ticker=ticker,
        earnings_date=earnings_date,
        news_df=news_df,
        manual_headlines_df=manual_headlines_df,
        lookback_days=14,
    )

    # Trade plan
    setup = str(metrics_row.get("Setup", "")).strip() or "Unknown"
    last = _safe_float(metrics_row.get("Last", np.nan))
    atr_pct_val = _safe_float(metrics_row.get("ATR%", np.nan))
    plan = standard_trade_plan_guidance(setup=setup, last=last, atr_pct=atr_pct_val)

    leveraged_note = "YES (leveraged ETF: separate risk bucket)" if bool(metrics_row.get("Is_Leveraged_ETF", False)) else "NO"

    # Missing checks
    missing_checks = []
    if earnings_date == "Unknown":
        missing_checks.append("Confirm next earnings date (company IR site / Nasdaq earnings calendar).")
    if comp["headline_count_recent"] == 0 and not comp["has_manual"]:
        missing_checks.append("Pull 14–30d headlines from a second source (SEC, PR, major outlets, terminal).")
    missing_checks.append("Check upcoming: FDA/clinical readouts, contracts, secondary offering/ATM, lockup expiry, guidance updates.")
    missing_checks.append("Check technical context: multi-year resistance, gap levels, supply zones, post-earnings drift behavior.")

    if np.isfinite(plan["atr_dollar_est"]):
        metrics_block = (
            f"Screener metrics:\n"
            f"- Last: {metrics_row.get('Last')}\n"
            f"- RVOL: {metrics_row.get('RVOL')}\n"
            f"- ATR%: {metrics_row.get('ATR%')} (≈ ${plan['atr_dollar_est']:.2f} ATR/day)\n"
        )
    else:
        metrics_block = (
            f"Screener metrics:\n"
            f"- Last: {metrics_row.get('Last')}\n"
            f"- RVOL: {metrics_row.get('RVOL')}\n"
            f"- ATR%: {metrics_row.get('ATR%')}\n"
        )
    
    metrics_block += (
        f"- RSI14: {metrics_row.get('RSI14')}\n"
        f"- Dist_to_52W_High%: {metrics_row.get('Dist_to_52W_High%')}\n"
    )
    
    # Add MA structure if available
    if "Above_MA20" in metrics_row and "Above_MA50" in metrics_row:
        ma20_val = metrics_row.get("MA20", np.nan)
        ma50_val = metrics_row.get("MA50", np.nan)
        above_ma20 = int(metrics_row.get("Above_MA20", 0))
        above_ma50 = int(metrics_row.get("Above_MA50", 0))
        ret20 = metrics_row.get("Ret20d%", np.nan)
        ma20_str = f"{ma20_val:.2f}" if np.isfinite(ma20_val) else "N/A"
        ma50_str = f"{ma50_val:.2f}" if np.isfinite(ma50_val) else "N/A"
        metrics_block += f"\n- MA20: {ma20_str}"
        metrics_block += f"\n- MA50: {ma50_str}"
        metrics_block += f"\n- Above_MA20: {above_ma20} (1=yes, 0=no)"
        metrics_block += f"\n- Above_MA50: {above_ma50} (1=yes, 0=no)"
        if np.isfinite(ret20):
            metrics_block += f"\n- Ret20d%: {ret20:.2f}%"
    
    if source_tags is None:
        source_tags = ["BASE_UNIVERSE"]
    source_tags_str = ", ".join(source_tags) if source_tags else "BASE_UNIVERSE"
    
    metrics_block += (
        f"\n- Setup: {setup}\n"
        f"- Score: {metrics_row.get('Score')}\n"
        f"- Leveraged ETF: {leveraged_note}\n"
        f"- Source tags: {source_tags_str}\n"
        f"- Dilution risk flag (headline scan): {flags['dilution_flag']}\n"
        f"- Catalyst tags (headline scan): {flags['catalyst_tags'] or 'None'}\n"
    )

    regime_block = (
        f"Market regime snapshot (best-effort):\n"
        f"- {regime_info.get('message','(not available)')}\n"
        f"- Regime Gate OK: {regime_info.get('ok', True)}\n"
    )

    completeness_block = (
        f"Catalyst completeness (penalize Unknowns):\n"
        f"- Completeness Score: {comp['completeness_score']}/100\n"
        f"- Earnings known: {comp['earnings_known']} (earnings={earnings_date})\n"
        f"- Recent headline count (14d): {comp['headline_count_recent']}\n"
        f"- Manual headlines: {comp['has_manual']}\n"
        + ("" if not comp["penalties"] else "- Penalties:\n  " + "\n  ".join([f"* {x}" for x in comp["penalties"]]) + "\n")
    )

    plan_block = (
        f"Standard trade plan template (by setup):\n"
        f"- {plan['entry_template']}\n"
        f"- {plan['stop_template']}\n"
        f"- {plan['tp_template']}\n"
        f"- {plan['size_template']}\n"
    )

    next_checks_block = "If data is missing, explicitly state Unknown and list checks:\n- " + "\n- ".join(missing_checks)

    prompt = f"""==============================
TICKER: {ticker}
==============================

Role: Momentum hedge fund analyst. You are running a probability audit (NOT a price predictor).
Objective: Decide if {ticker} has >60% probability of achieving +10% within 30 calendar days.

Hard constraints:
- You MUST penalize "Unknown" earnings + missing headlines (no assuming hidden catalysts).
- You MUST still score based on tape structure: RVOL, ATR%, RSI, distance to 52W high, setup type.
- You MUST output at most 2 BUY ratings across the entire batch (if you are given multiple tickers).
- If catalyst data is weak, default to WATCH/IGNORE even if technicals look good.
- If Dilution risk flag = 1 (headline scan), cap Verdict at WATCH unless there is a clear positive catalyst that outweighs it.
- For Breakouts, require structural confirmation: Above_MA20=1 and Above_MA50=1; otherwise downgrade Technical Alignment.

Inputs:
{regime_block}
Earnings (best-effort): {earnings_date}

{metrics_block}

{completeness_block}

Recent headlines:
{chr(10).join(headlines)}

{plan_block}

{next_checks_block}

Scoring rubric (0–100):
- Catalyst Immediacy (0–30)
- Narrative Velocity (0–25)
- Volatility Fit (0–20)
- Technical Alignment (0–25)

Output format (STRICT):
- Total Score: X/100
- Verdict: BUY / WATCH / IGNORE
- Setup: Breakout or Reversal
- 1-line Spark: what specifically could trigger +10%
- 1-line Trap: what invalidates the thesis
- Trade Plan: entry trigger, stop anchor, TP1/TP2, position size rule (risk 1–2%)
"""
    return prompt

