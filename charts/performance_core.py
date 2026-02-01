"""
Core performance computation functions.
Extracted from notebook tracker for reusability.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict


def compute_cumulative_returns(
    close_df: pd.DataFrame,
    baseline_date: str,
    end_date: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Compute cumulative returns from baseline date, aligned by trading days.
    
    Args:
        close_df: Date-indexed DataFrame, columns=tickers, values=Close prices
        baseline_date: YYYY-MM-DD (pick date)
        end_date: YYYY-MM-DD (latest available)
    
    Returns:
        df_cum: Cumulative returns (%) for each ticker
        td_counter: Trading day counter (1, 2, 3, ...)
        meta: Metadata with baseline dates and prices per ticker
    """
    start_dt = pd.to_datetime(baseline_date)
    end_dt = pd.to_datetime(end_date)

    w = close_df.loc[(close_df.index >= start_dt) & (close_df.index <= end_dt)].copy()
    w = w.dropna(how="all")
    
    if w.empty:
        return pd.DataFrame(), pd.Series(dtype=int), pd.DataFrame()

    # Trading-day counter (not calendar days)
    td_counter = pd.Series(
        range(1, len(w) + 1),
        index=w.index,
        name="TradingDay"
    )

    anchors = {}
    base_dates = {}

    for t in w.columns:
        s = w[t].dropna()
        if s.empty:
            continue
        anchors[t] = float(s.iloc[0])
        base_dates[t] = s.index[0]

    if not anchors:
        return pd.DataFrame(), td_counter, pd.DataFrame()

    anchor_series = pd.Series(anchors)

    # Cumulative returns as percentage
    df_cum = (w.divide(anchor_series, axis="columns") - 1.0) * 100.0
    df_cum = df_cum.dropna(how="all")

    meta = pd.DataFrame(
        {
            "Baseline_Date": [base_dates[t].date().isoformat() for t in anchors],
            "Baseline_Price": [round(anchors[t], 2) for t in anchors],
        },
        index=list(anchors.keys()),
    )

    return df_cum, td_counter, meta


def compute_final_performance(
    close_df: pd.DataFrame,
    baseline_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Compute final performance metrics for each ticker.
    
    Returns DataFrame with Symbol, Baseline, Current, Change, Percent_Change, etc.
    """
    if close_df.empty:
        return pd.DataFrame()

    start_dt = pd.to_datetime(baseline_date)
    end_dt = pd.to_datetime(end_date)

    w = close_df.loc[(close_df.index >= start_dt) & (close_df.index <= end_dt)].copy()
    w = w.dropna(how="all")

    if w.empty:
        return pd.DataFrame()

    rows = []
    for t in close_df.columns:
        if t not in w.columns:
            continue
        s = w[t].dropna()
        if s.empty or len(s) < 1:
            continue

        baseline = float(s.iloc[0])
        current = float(s.iloc[-1])
        if baseline == 0:
            continue

        change = current - baseline
        pct = (change / baseline) * 100.0

        rows.append({
            "Symbol": t,
            "Baseline": round(baseline, 2),
            "Current": round(current, 2),
            "Change": round(change, 2),
            "Percent_Change": pct,
            "Baseline_Date": s.index[0].strftime("%Y-%m-%d"),
            "End_Date": s.index[-1].strftime("%Y-%m-%d"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    return df.sort_values("Percent_Change", ascending=False).reset_index(drop=True)
