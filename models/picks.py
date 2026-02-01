"""
Pick events model - normalized table of all stock picks.
"""

import pandas as pd
from typing import Dict, List, Any


def build_pick_events_from_tracker_data(tracker_data: Dict[str, Dict[str, List[str]]]) -> pd.DataFrame:
    """
    Build pick events table from NOTEBOOK_TRACKER_DATA format.
    
    Args:
        tracker_data: Dict of date -> {weekly: [...], pro30: [...], movers: [...]}
    
    Returns:
        DataFrame with columns: date, ticker, source
    """
    rows = []
    for date, sources in tracker_data.items():
        for source_name, tickers in sources.items():
            for ticker in tickers:
                if ticker:  # Skip empty
                    rows.append({
                        "date": date,
                        "ticker": ticker.upper().strip(),
                        "source": source_name,
                    })
    
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def build_pick_events_from_hybrid_analyses(analyses: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Build pick events table from hybrid analysis JSONs.
    
    Args:
        analyses: List of hybrid analysis dicts from artifact
    
    Returns:
        DataFrame with columns: date, ticker, source, rank, score, confidence, regime
    """
    rows = []
    
    for h in analyses:
        date = h.get("date") or h.get("asof_trading_date")
        if not date:
            continue
        
        regime = h.get("regime", "unknown")
        
        # Weekly Top5
        for i, item in enumerate(h.get("weekly_top5", h.get("primary_top5", [])), 1):
            ticker = item.get("ticker") if isinstance(item, dict) else item
            if ticker:
                rows.append({
                    "date": date,
                    "ticker": str(ticker).upper().strip(),
                    "source": "weekly",
                    "rank": i,
                    "score": item.get("score") if isinstance(item, dict) else None,
                    "confidence": item.get("confidence") if isinstance(item, dict) else None,
                    "regime": regime,
                })
        
        # Hybrid Top3
        for i, item in enumerate(h.get("hybrid_top3", []), 1):
            ticker = item.get("ticker") if isinstance(item, dict) else item
            if ticker:
                rows.append({
                    "date": date,
                    "ticker": str(ticker).upper().strip(),
                    "source": "hybrid_top3",
                    "rank": i,
                    "score": item.get("hybrid_score") if isinstance(item, dict) else None,
                    "confidence": item.get("confidence") if isinstance(item, dict) else None,
                    "regime": regime,
                })
        
        # Pro30
        for ticker in h.get("pro30_tickers", []):
            if ticker:
                rows.append({
                    "date": date,
                    "ticker": str(ticker).upper().strip(),
                    "source": "pro30",
                    "rank": None,
                    "score": None,
                    "confidence": None,
                    "regime": regime,
                })
        
        # Movers
        for ticker in h.get("movers_tickers", []):
            if ticker:
                rows.append({
                    "date": date,
                    "ticker": str(ticker).upper().strip(),
                    "source": "movers",
                    "rank": None,
                    "score": None,
                    "confidence": None,
                    "regime": regime,
                })
        
        # Conviction picks
        for i, item in enumerate(h.get("conviction_picks", []), 1):
            ticker = item.get("ticker") if isinstance(item, dict) else item
            if ticker:
                rows.append({
                    "date": date,
                    "ticker": str(ticker).upper().strip(),
                    "source": "conviction",
                    "rank": i,
                    "score": item.get("score") if isinstance(item, dict) else None,
                    "confidence": item.get("confidence") if isinstance(item, dict) else None,
                    "regime": regime,
                })
    
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def get_ticker_pick_history(df_events: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Get all pick events for a specific ticker."""
    if df_events.empty:
        return pd.DataFrame()
    return df_events[df_events["ticker"] == ticker.upper()].copy()


def get_pick_dates_for_ticker(df_events: pd.DataFrame, ticker: str) -> List[str]:
    """Get list of dates when ticker was picked."""
    hist = get_ticker_pick_history(df_events, ticker)
    if hist.empty:
        return []
    return sorted(hist["date"].dt.strftime("%Y-%m-%d").unique().tolist())


def get_tickers_by_source(df_events: pd.DataFrame, date: str, source: str) -> List[str]:
    """Get tickers for a specific date and source."""
    if df_events.empty:
        return []
    mask = (df_events["date"].dt.strftime("%Y-%m-%d") == date) & (df_events["source"] == source)
    return df_events.loc[mask, "ticker"].unique().tolist()


def get_ticker_stats(df_events: pd.DataFrame, ticker: str) -> Dict[str, Any]:
    """Get summary stats for a ticker."""
    hist = get_ticker_pick_history(df_events, ticker)
    if hist.empty:
        return {}
    
    return {
        "ticker": ticker,
        "total_picks": len(hist),
        "unique_dates": hist["date"].nunique(),
        "sources": hist["source"].value_counts().to_dict(),
        "first_pick": hist["date"].min().strftime("%Y-%m-%d"),
        "last_pick": hist["date"].max().strftime("%Y-%m-%d"),
    }
