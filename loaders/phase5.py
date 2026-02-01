"""
Phase-5 data loader.
Auto-detects and loads Phase-5 parquet when available.
Falls back to proxy calculations when not.
"""

import os
import io
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import streamlit as st


def try_load_phase5_parquet_from_path(root: str) -> Optional[pd.DataFrame]:
    """
    Try to load Phase-5 merged parquet from local path.
    
    Args:
        root: Root directory path
    
    Returns:
        DataFrame if found, None otherwise
    """
    paths_to_try = [
        Path(root) / "outputs" / "phase5" / "merged" / "phase5_merged.parquet",
        Path(root) / "phase5" / "merged" / "phase5_merged.parquet",
        Path(root) / "phase5_merged.parquet",
    ]
    
    for p in paths_to_try:
        if p.exists():
            try:
                return pd.read_parquet(p)
            except Exception:
                continue
    
    return None


def try_load_phase5_from_artifact_files(files: Dict[str, bytes]) -> Optional[pd.DataFrame]:
    """
    Try to load Phase-5 merged parquet from artifact files dict.
    
    Args:
        files: Dict of {path: bytes} from artifact
    
    Returns:
        DataFrame if found, None otherwise
    """
    paths_to_try = [
        "outputs/phase5/merged/phase5_merged.parquet",
        "phase5/merged/phase5_merged.parquet",
        "outputs/phase5/phase5_merged.parquet",
        "phase5_merged.parquet",
    ]
    
    for path in paths_to_try:
        if path in files:
            try:
                return pd.read_parquet(io.BytesIO(files[path]))
            except Exception:
                continue
    
    return None


def has_phase5_data(files: Optional[Dict[str, bytes]] = None, root: Optional[str] = None) -> bool:
    """Check if Phase-5 data is available."""
    if files:
        df = try_load_phase5_from_artifact_files(files)
        return df is not None and not df.empty
    if root:
        df = try_load_phase5_parquet_from_path(root)
        return df is not None and not df.empty
    return False


def prepare_phase5_for_equity_curve(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare Phase-5 DataFrame for equity curve computation.
    Normalizes columns and filters to resolved outcomes.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    out = df.copy()
    
    # Normalize date column
    for col in ["scan_date", "asof", "pick_date", "date"]:
        if col in out.columns:
            out["scan_date"] = pd.to_datetime(out[col], errors="coerce")
            break
    
    # Normalize return column
    if "return_7d" in out.columns:
        out["return_7d"] = pd.to_numeric(out["return_7d"], errors="coerce")
    elif "pct_change_7d" in out.columns:
        out["return_7d"] = pd.to_numeric(out["pct_change_7d"], errors="coerce")
    
    # Normalize outcome column
    if "outcome_7d" in out.columns:
        out["outcome_7d"] = out["outcome_7d"].astype(str).str.lower()
    elif "hit_7d" in out.columns:
        import numpy as np
        out["outcome_7d"] = np.where(out["hit_7d"].astype(bool), "hit", "miss")
    
    # Filter to rows with valid returns
    if "return_7d" in out.columns:
        out = out.dropna(subset=["return_7d"])
    
    return out


@st.cache_data(ttl=300, show_spinner=False)
def compute_proxy_equity_curve(
    tracker_data: Dict[str, Dict[str, List[str]]],
    end_date: str,
    lookback_days: int = 7,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Compute proxy equity curve using yfinance data.
    Used when Phase-5 parquet is not available.
    
    Args:
        tracker_data: Dict of date -> {weekly: [...], pro30: [...], movers: [...]}
        end_date: End date for tracking
        lookback_days: Number of trading days to track each pick
    
    Returns:
        Tuple of (equity_df, stats_dict)
    """
    from loaders.prices import load_prices
    from datetime import datetime, timedelta
    
    date_returns = {}
    
    for date, sources in sorted(tracker_data.items()):
        # Get all tickers for this date
        tickers = []
        for src in ["weekly", "pro30", "movers"]:
            tickers.extend(sources.get(src, []))
        tickers = list(set([t.upper() for t in tickers if t]))
        
        if not tickers:
            continue
        
        # Calculate end date for this pick (baseline + lookback)
        try:
            pick_date = datetime.strptime(date, "%Y-%m-%d")
            pick_end = (pick_date + timedelta(days=lookback_days + 5)).strftime("%Y-%m-%d")
            
            # Don't go past the overall end date
            if pick_end > end_date:
                pick_end = end_date
            
            # Load prices
            close = load_prices(tickers, date, pick_end)
            
            if close.empty:
                continue
            
            # Compute returns for each ticker
            returns = []
            for ticker in close.columns:
                s = close[ticker].dropna()
                if len(s) < 2:
                    continue
                ret = (s.iloc[-1] / s.iloc[0] - 1) * 100
                returns.append(ret)
            
            if returns:
                date_returns[date] = sum(returns) / len(returns)
                
        except Exception:
            continue
    
    # Build equity curve
    if not date_returns:
        return pd.DataFrame(), {}
    
    rows = []
    equity = 100.0
    
    for date in sorted(date_returns.keys()):
        ret = date_returns[date]
        equity *= (1 + ret / 100.0)
        rows.append({
            "date": date,
            "avg_return": ret,
            "equity": equity,
        })
    
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    
    # Compute stats
    stats = {
        "total_return_pct": round((equity / 100 - 1) * 100, 2),
        "final_equity": round(equity, 2),
        "num_periods": len(df),
        "avg_daily_return": round(df["avg_return"].mean(), 2) if not df.empty else 0,
        "win_days": int((df["avg_return"] > 0).sum()) if not df.empty else 0,
        "lose_days": int((df["avg_return"] < 0).sum()) if not df.empty else 0,
        "data_source": "proxy (yfinance)",
    }
    
    return df, stats
