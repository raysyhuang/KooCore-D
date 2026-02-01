"""
Equity curve visualization.
Works with proxy data (yfinance) now, auto-upgrades when Phase-5 parquet exists.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


def compute_equity_curve_from_returns(
    date_returns: Dict[str, float],
    initial_equity: float = 100.0,
) -> pd.DataFrame:
    """
    Compute equity curve from a dict of date -> average return (%).
    
    Args:
        date_returns: Dict of date -> average return percentage
        initial_equity: Starting equity value
    
    Returns:
        DataFrame with date, return_pct, equity columns
    """
    if not date_returns:
        return pd.DataFrame()
    
    rows = []
    equity = initial_equity
    
    for date in sorted(date_returns.keys()):
        ret_pct = date_returns[date]
        equity *= (1 + ret_pct / 100.0)
        rows.append({
            "date": date,
            "return_pct": ret_pct,
            "equity": equity,
        })
    
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def compute_equity_curve_from_phase5(
    df_phase5: pd.DataFrame,
    return_col: str = "return_7d",
    date_col: str = "scan_date",
    initial_equity: float = 100.0,
) -> pd.DataFrame:
    """
    Compute equity curve from Phase-5 merged parquet.
    
    Args:
        df_phase5: Phase-5 DataFrame with resolved outcomes
        return_col: Column name for returns
        date_col: Column name for scan dates
        initial_equity: Starting equity value
    
    Returns:
        DataFrame with date, avg_return, equity, count, win_rate columns
    """
    if df_phase5.empty or return_col not in df_phase5.columns:
        return pd.DataFrame()
    
    df = df_phase5.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df[return_col] = pd.to_numeric(df[return_col], errors="coerce")
    
    # Group by date and compute stats
    grouped = df.groupby(df[date_col].dt.date).agg({
        return_col: ["mean", "count"],
    })
    grouped.columns = ["avg_return", "count"]
    grouped = grouped.reset_index()
    grouped.columns = ["date", "avg_return", "count"]
    grouped = grouped.sort_values("date")
    
    # Compute win rate if outcome column exists
    if "outcome_7d" in df.columns:
        win_rates = df.groupby(df[date_col].dt.date).apply(
            lambda x: (x["outcome_7d"].isin(["hit", "1", "true", "yes"])).mean()
        )
        grouped["win_rate"] = grouped["date"].map(win_rates.to_dict())
    else:
        grouped["win_rate"] = np.nan
    
    # Compute equity curve
    equity = initial_equity
    equities = []
    for _, row in grouped.iterrows():
        equity *= (1 + row["avg_return"] / 100.0)
        equities.append(equity)
    grouped["equity"] = equities
    
    grouped["date"] = pd.to_datetime(grouped["date"])
    return grouped


def plot_equity_curve(
    df_equity: pd.DataFrame,
    title: str = "Equity Curve",
    show_returns: bool = True,
    height: int = 600,
):
    """
    Plot equity curve with optional daily returns bars.
    """
    if df_equity.empty:
        fig = go.Figure()
        fig.update_layout(title="No equity data available", height=height)
        return fig
    
    if show_returns and "avg_return" in df_equity.columns:
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=(title, "Daily Average Returns"),
            row_heights=[0.7, 0.3],
            vertical_spacing=0.1,
            shared_xaxes=True,
        )
        
        # Equity curve (top)
        fig.add_trace(
            go.Scatter(
                x=df_equity["date"],
                y=df_equity["equity"],
                mode="lines",
                name="Equity",
                line={"width": 2, "color": "#4A90D9"},
                fill="tozeroy",
                fillcolor="rgba(74, 144, 217, 0.1)",
                hovertemplate=(
                    "Date: %{x|%Y-%m-%d}<br>"
                    "Equity: %{y:.2f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        
        # Daily returns (bottom)
        colors = np.where(df_equity["avg_return"] >= 0, "#00CC44", "#FF4444")
        fig.add_trace(
            go.Bar(
                x=df_equity["date"],
                y=df_equity["avg_return"],
                name="Daily Return",
                marker_color=colors,
                hovertemplate=(
                    "Date: %{x|%Y-%m-%d}<br>"
                    "Return: %{y:+.2f}%<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )
        
        fig.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
        
        fig.update_yaxes(title_text="Equity", row=1, col=1)
        fig.update_yaxes(title_text="Return (%)", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        
    else:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_equity["date"],
            y=df_equity["equity"],
            mode="lines",
            name="Equity",
            line={"width": 2, "color": "#4A90D9"},
            fill="tozeroy",
            fillcolor="rgba(74, 144, 217, 0.1)",
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>"
                "Equity: %{y:.2f}<extra></extra>"
            ),
        ))
        
        fig.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5)
        fig.update_yaxes(title_text="Equity")
        fig.update_xaxes(title_text="Date")
        fig.update_layout(title=title)
    
    fig.update_layout(
        template="plotly_white",
        height=height,
        showlegend=False,
    )
    
    return fig


def compute_equity_stats(df_equity: pd.DataFrame) -> Dict:
    """Compute summary statistics for equity curve."""
    if df_equity.empty:
        return {}
    
    returns = df_equity.get("avg_return", pd.Series(dtype=float))
    equity = df_equity.get("equity", pd.Series(dtype=float))
    
    if equity.empty:
        return {}
    
    total_return = (equity.iloc[-1] / 100.0 - 1) * 100
    
    stats = {
        "total_return_pct": round(total_return, 2),
        "final_equity": round(equity.iloc[-1], 2),
        "num_periods": len(df_equity),
        "start_date": df_equity["date"].min().strftime("%Y-%m-%d"),
        "end_date": df_equity["date"].max().strftime("%Y-%m-%d"),
    }
    
    if not returns.empty:
        stats.update({
            "avg_daily_return": round(returns.mean(), 2),
            "best_day": round(returns.max(), 2),
            "worst_day": round(returns.min(), 2),
            "win_days": int((returns > 0).sum()),
            "lose_days": int((returns < 0).sum()),
            "win_rate": round((returns > 0).mean() * 100, 1) if len(returns) > 0 else 0,
        })
    
    if "win_rate" in df_equity.columns:
        valid_wr = df_equity["win_rate"].dropna()
        if not valid_wr.empty:
            stats["avg_pick_win_rate"] = round(valid_wr.mean() * 100, 1)
    
    return stats


def plot_equity_with_drawdown(
    df_equity: pd.DataFrame,
    title: str = "Equity Curve with Drawdown",
    height: int = 700,
):
    """
    Plot equity curve with drawdown visualization.
    """
    if df_equity.empty or "equity" not in df_equity.columns:
        fig = go.Figure()
        fig.update_layout(title="No data available", height=height)
        return fig
    
    # Compute drawdown
    equity = df_equity["equity"]
    rolling_max = equity.expanding().max()
    drawdown = (equity - rolling_max) / rolling_max * 100
    
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(title, "Drawdown"),
        row_heights=[0.7, 0.3],
        vertical_spacing=0.1,
        shared_xaxes=True,
    )
    
    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=df_equity["date"],
            y=equity,
            mode="lines",
            name="Equity",
            line={"width": 2, "color": "#4A90D9"},
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Equity: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    
    # Drawdown
    fig.add_trace(
        go.Scatter(
            x=df_equity["date"],
            y=drawdown,
            mode="lines",
            name="Drawdown",
            fill="tozeroy",
            line={"width": 1, "color": "#FF4444"},
            fillcolor="rgba(255, 68, 68, 0.3)",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Drawdown: %{y:.2f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )
    
    fig.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
    
    fig.update_yaxes(title_text="Equity", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    
    fig.update_layout(
        template="plotly_white",
        height=height,
        showlegend=False,
    )
    
    return fig
