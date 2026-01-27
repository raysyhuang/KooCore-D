# src/core/asof.py
"""
Centralized as-of date enforcement and OHLCV validation.

This module provides:
- validate_ohlcv: Validates and cleans OHLCV data
- enforce_asof: Removes any rows beyond the as-of date (prevents lookahead)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import pandas as pd


REQUIRED_OHLCV = ("Open", "High", "Low", "Close", "Volume")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Handle common column name variations."""
    col_map = {}
    for c in df.columns:
        cl = str(c).lower()
        if cl == "open":
            col_map[c] = "Open"
        elif cl == "high":
            col_map[c] = "High"
        elif cl == "low":
            col_map[c] = "Low"
        elif cl == "close":
            col_map[c] = "Close"
        elif cl in ("volume", "vol"):
            col_map[c] = "Volume"
    if col_map:
        df = df.rename(columns=col_map)
    return df


def validate_ohlcv(df: pd.DataFrame, ticker: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validates and lightly cleans OHLCV data.
    
    Returns:
        (clean_df, stats) tuple where stats contains validation info.
    
    Does NOT enforce as-of; use enforce_asof for that.
    """
    stats: Dict[str, Any] = {
        "ticker": ticker,
        "missing_cols": [],
        "dropped_bad_rows": 0,
        "rows_in": int(len(df)) if df is not None else 0,
        "rows_out": 0,
    }

    if df is None or df.empty:
        stats["rows_out"] = 0
        return pd.DataFrame(), stats

    df = df.copy()
    df = _normalize_columns(df)

    missing = [c for c in REQUIRED_OHLCV if c not in df.columns]
    if missing:
        stats["missing_cols"] = missing
        return pd.DataFrame(), stats

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, errors="coerce")
        except Exception:
            return pd.DataFrame(), stats

    df = df[~df.index.isna()].sort_index()

    # Basic bad bar checks
    bad = (
        (df["High"] < df["Low"])
        | (df["Close"] <= 0)
        | (df["Open"] <= 0)
        | (df["High"] <= 0)
        | (df["Low"] <= 0)
        | (df["Volume"] < 0)
    )
    bad_count = int(bad.sum())
    if bad_count:
        df = df.loc[~bad]
        stats["dropped_bad_rows"] = bad_count

    df = df[list(REQUIRED_OHLCV)].dropna(how="any")
    stats["rows_out"] = int(len(df))
    return df, stats


def enforce_asof(
    df: pd.DataFrame,
    asof_date: Optional[str],
    ticker: str,
    strict: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Hard rule: remove any rows beyond asof_date (date-based).
    
    This prevents lookahead bias by ensuring no future data is used.
    
    Args:
        df: DataFrame with DatetimeIndex
        asof_date: Cut-off date string (YYYY-MM-DD)
        ticker: Ticker symbol for logging
        strict: If True, raise ValueError on violations
    
    Returns:
        (cut_df, stats) tuple
    """
    info: Dict[str, Any] = {"ticker": ticker, "asof_date": asof_date, "asof_violation": False}
    
    if df is None or df.empty or not asof_date:
        return df if df is not None else pd.DataFrame(), info

    cut = pd.to_datetime(asof_date).date()

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index, errors="coerce")
        except Exception:
            if strict:
                raise ValueError(f"{ticker}: cannot coerce index to datetime for asof enforcement")
            return pd.DataFrame(), info

    df = df[~df.index.isna()].sort_index()
    df_cut = df[df.index.date <= cut]

    if not df_cut.empty:
        last_dt = df_cut.index.max().date()
        if last_dt > cut:
            info["asof_violation"] = True
            if strict:
                raise ValueError(f"{ticker}: last bar {last_dt} beyond asof {cut}")

    # Detect if we dropped future rows
    if len(df_cut) < len(df):
        info["dropped_future_rows"] = int(len(df) - len(df_cut))

    return df_cut, info
