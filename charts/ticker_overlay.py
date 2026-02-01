"""
Ticker overlay charts - multiple pick dates for same ticker.
"Spaghetti overlay" view showing how the same stock performed from different entry points.
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import List, Optional


def plot_ticker_overlays(
    close: pd.Series,
    pick_dates: List[str],
    ticker: str,
    end_date: Optional[str] = None,
    highlight_best: bool = True,
    height: int = 520,
):
    """
    Plot cumulative returns for a single ticker from multiple pick dates.
    
    Args:
        close: Series of close prices for the ticker (Date-indexed)
        pick_dates: List of pick dates (YYYY-MM-DD strings)
        ticker: Ticker symbol (for title)
        end_date: Optional end date to limit the view
        highlight_best: Whether to highlight best/worst instances
        height: Chart height in pixels
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    if close.empty or not pick_dates:
        fig.update_layout(
            title=f"No data available for {ticker}",
            height=height,
        )
        return fig
    
    close = close.dropna().sort_index()
    
    if end_date:
        end_dt = pd.to_datetime(end_date)
        close = close[close.index <= end_dt]
    
    # Track final returns for highlighting
    final_returns = {}
    traces_data = []
    
    for d in pick_dates:
        start_dt = pd.to_datetime(d)
        w = close[close.index >= start_dt].copy()
        
        if w.empty or len(w) < 2:
            continue
        
        anchor = float(w.iloc[0])
        cum = (w / anchor - 1.0) * 100.0
        td = list(range(1, len(cum) + 1))
        
        final_ret = float(cum.iloc[-1])
        final_returns[d] = final_ret
        
        traces_data.append({
            "date": d,
            "td": td,
            "cum": cum.values,
            "actual_dates": [x.strftime("%Y-%m-%d") for x in cum.index],
            "final_return": final_ret,
            "anchor": anchor,
        })
    
    if not traces_data:
        fig.update_layout(
            title=f"No valid data for {ticker}",
            height=height,
        )
        return fig
    
    # Find best and worst
    best_date = max(final_returns, key=final_returns.get) if final_returns else None
    worst_date = min(final_returns, key=final_returns.get) if final_returns else None
    
    # Plot traces
    for trace in traces_data:
        is_best = highlight_best and trace["date"] == best_date
        is_worst = highlight_best and trace["date"] == worst_date
        
        line_style = {}
        if is_best:
            line_style = {"width": 3, "color": "#00CC44"}
        elif is_worst:
            line_style = {"width": 3, "color": "#FF4444"}
        else:
            line_style = {"width": 1.5}
        
        name = trace["date"]
        if is_best:
            name += f" (Best: {trace['final_return']:+.1f}%)"
        elif is_worst:
            name += f" (Worst: {trace['final_return']:+.1f}%)"
        
        fig.add_trace(go.Scatter(
            x=trace["td"],
            y=trace["cum"],
            mode="lines",
            name=name,
            line=line_style,
            hovertemplate=(
                f"<b>Pick: {trace['date']}</b><br>"
                f"Entry: ${trace['anchor']:.2f}<br>"
                "Trading Day: %{x}<br>"
                "Date: %{customdata}<br>"
                "Return: %{y:.2f}%<extra></extra>"
            ),
            customdata=trace["actual_dates"],
        ))
    
    fig.add_hline(y=0, line_dash="dot", line_color="black", opacity=0.4)
    
    # Add average line if multiple traces
    if len(traces_data) > 1:
        # Find common length (shortest trace)
        min_len = min(len(t["cum"]) for t in traces_data)
        avg_cum = np.mean([t["cum"][:min_len] for t in traces_data], axis=0)
        
        fig.add_trace(go.Scatter(
            x=list(range(1, min_len + 1)),
            y=avg_cum,
            mode="lines",
            name=f"Avg ({len(traces_data)} picks)",
            line={"width": 3, "dash": "dashdot", "color": "purple"},
            hovertemplate=(
                "<b>Average</b><br>"
                "Trading Day: %{x}<br>"
                "Return: %{y:.2f}%<extra></extra>"
            ),
        ))
    
    fig.update_layout(
        title=f"{ticker} — {len(traces_data)} Pick-Date Overlays",
        template="plotly_white",
        height=height,
        hovermode="x unified",
        xaxis_title="Trading Day (since pick date)",
        yaxis_title="Cumulative Return (%)",
        legend_title_text="Pick Dates",
    )
    
    return fig


def compute_ticker_pick_stats(
    close: pd.Series,
    pick_dates: List[str],
    ticker: str,
    lookback_days: int = 10,
) -> pd.DataFrame:
    """
    Compute stats for each pick instance of a ticker.
    
    Returns DataFrame with:
        pick_date, entry_price, current_price, return_pct, trading_days, max_gain, max_loss
    """
    if close.empty or not pick_dates:
        return pd.DataFrame()
    
    close = close.dropna().sort_index()
    rows = []
    
    for d in pick_dates:
        start_dt = pd.to_datetime(d)
        w = close[close.index >= start_dt].copy()
        
        if w.empty:
            continue
        
        anchor = float(w.iloc[0])
        current = float(w.iloc[-1])
        ret_pct = (current / anchor - 1.0) * 100.0
        
        cum = (w / anchor - 1.0) * 100.0
        max_gain = float(cum.max())
        max_loss = float(cum.min())
        
        rows.append({
            "pick_date": d,
            "entry_price": round(anchor, 2),
            "current_price": round(current, 2),
            "return_pct": round(ret_pct, 2),
            "trading_days": len(w),
            "max_gain_pct": round(max_gain, 2),
            "max_loss_pct": round(max_loss, 2),
        })
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("pick_date", ascending=False)
    return df
