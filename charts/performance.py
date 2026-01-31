"""
Performance Charts Module

Reusable chart functions for performance visualization.
All functions return Plotly figures - no .show() calls.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional


def create_returns_bar_chart(df_perf: pd.DataFrame, title: str = "Returns by Ticker") -> go.Figure:
    """
    Create a simple bar chart of returns.
    
    Args:
        df_perf: DataFrame with Symbol and Percent_Change columns
        title: Chart title
    
    Returns:
        Plotly Figure
    """
    if df_perf.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)
    
    colors = np.where(df_perf["Percent_Change"] >= 0, "#00CC44", "#FF4444")
    
    fig = go.Figure(data=[
        go.Bar(
            x=df_perf["Symbol"],
            y=df_perf["Percent_Change"],
            marker_color=colors,
            text=[f"{x:.1f}%" for x in df_perf["Percent_Change"]],
            textposition="outside",
        )
    ])
    
    avg_change = float(df_perf["Percent_Change"].mean())
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.4)
    fig.add_hline(
        y=avg_change,
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Avg: {avg_change:.2f}%",
        annotation_position="bottom right",
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Ticker",
        yaxis_title="Return (%)",
        template="plotly_white",
        height=400,
    )
    
    return fig


def create_group_comparison_chart(
    df_perf: pd.DataFrame,
    group_sets: dict,
    title: str = "Average Return by Source"
) -> go.Figure:
    """
    Create bar chart comparing average returns by source group.
    
    Args:
        df_perf: DataFrame with Symbol and Percent_Change columns
        group_sets: Dict with weekly, pro30, movers lists
        title: Chart title
    
    Returns:
        Plotly Figure
    """
    if df_perf.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)
    
    group_stats = []
    
    for group_name, tickers in [
        ("Weekly", group_sets.get("weekly", [])),
        ("Pro30", group_sets.get("pro30", [])),
        ("Movers", group_sets.get("movers", [])),
    ]:
        if not tickers:
            continue
        
        tickers_upper = [t.upper() for t in tickers]
        group_df = df_perf[df_perf["Symbol"].isin(tickers_upper)]
        
        if not group_df.empty:
            group_stats.append({
                "Group": group_name,
                "Avg_Return": group_df["Percent_Change"].mean(),
                "Count": len(group_df),
                "Winners": (group_df["Percent_Change"] > 0).sum(),
            })
    
    if not group_stats:
        return go.Figure().add_annotation(text="No group data available", showarrow=False)
    
    stats_df = pd.DataFrame(group_stats)
    
    colors = np.where(stats_df["Avg_Return"] >= 0, "#00CC44", "#FF4444")
    
    fig = go.Figure(data=[
        go.Bar(
            x=stats_df["Group"],
            y=stats_df["Avg_Return"],
            marker_color=colors,
            text=[f"{x:.1f}%<br>({w}/{c})" for x, w, c in 
                  zip(stats_df["Avg_Return"], stats_df["Winners"], stats_df["Count"])],
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Avg Return: %{y:.2f}%<br>"
                "<extra></extra>"
            ),
        )
    ])
    
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.4)
    
    fig.update_layout(
        title=title,
        xaxis_title="Source",
        yaxis_title="Average Return (%)",
        template="plotly_white",
        height=350,
    )
    
    return fig


def create_hit_rate_gauge(hit_rate: float, title: str = "Hit Rate") -> go.Figure:
    """
    Create a gauge chart showing hit rate.
    
    Args:
        hit_rate: Value between 0 and 1
        title: Chart title
    
    Returns:
        Plotly Figure
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=hit_rate * 100,
        number={"suffix": "%"},
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": title},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#00CC44" if hit_rate >= 0.5 else "#FF4444"},
            "steps": [
                {"range": [0, 40], "color": "#ffcccc"},
                {"range": [40, 60], "color": "#ffffcc"},
                {"range": [60, 100], "color": "#ccffcc"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 2},
                "thickness": 0.75,
                "value": 50,
            },
        },
    ))
    
    fig.update_layout(height=250)
    
    return fig


def create_returns_histogram(df_perf: pd.DataFrame, title: str = "Return Distribution") -> go.Figure:
    """
    Create histogram of returns.
    
    Args:
        df_perf: DataFrame with Percent_Change column
        title: Chart title
    
    Returns:
        Plotly Figure
    """
    if df_perf.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)
    
    fig = px.histogram(
        df_perf,
        x="Percent_Change",
        nbins=20,
        title=title,
        labels={"Percent_Change": "Return (%)"},
    )
    
    fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.5)
    
    mean_return = df_perf["Percent_Change"].mean()
    fig.add_vline(
        x=mean_return,
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Mean: {mean_return:.1f}%",
    )
    
    fig.update_layout(
        template="plotly_white",
        height=300,
    )
    
    return fig


def create_cumulative_comparison(
    returns_by_date: dict[str, float],
    benchmark_returns: Optional[dict[str, float]] = None,
    title: str = "Cumulative Performance"
) -> go.Figure:
    """
    Create cumulative returns comparison chart.
    
    Args:
        returns_by_date: Dict mapping date strings to portfolio returns
        benchmark_returns: Optional dict mapping date strings to benchmark returns
        title: Chart title
    
    Returns:
        Plotly Figure
    """
    if not returns_by_date:
        return go.Figure().add_annotation(text="No data available", showarrow=False)
    
    dates = sorted(returns_by_date.keys())
    
    # Calculate cumulative returns
    cum_returns = []
    cum = 1.0
    for d in dates:
        cum *= (1 + returns_by_date[d] / 100)
        cum_returns.append((cum - 1) * 100)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=cum_returns,
        mode="lines+markers",
        name="Portfolio",
        line=dict(color="purple", width=2),
    ))
    
    if benchmark_returns:
        bench_cum = []
        cum = 1.0
        for d in dates:
            if d in benchmark_returns:
                cum *= (1 + benchmark_returns[d] / 100)
            bench_cum.append((cum - 1) * 100)
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=bench_cum,
            mode="lines",
            name="Benchmark",
            line=dict(color="gray", width=2, dash="dot"),
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.4)
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        template="plotly_white",
        height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    
    return fig


def create_summary_metrics_cards(df_perf: pd.DataFrame) -> dict:
    """
    Calculate summary metrics from performance DataFrame.
    
    Args:
        df_perf: DataFrame with performance data
    
    Returns:
        Dict with metric values
    """
    if df_perf.empty:
        return {
            "total_picks": 0,
            "winners": 0,
            "losers": 0,
            "win_rate": 0,
            "avg_return": 0,
            "best_pick": "N/A",
            "best_return": 0,
            "worst_pick": "N/A",
            "worst_return": 0,
        }
    
    winners = (df_perf["Percent_Change"] > 0).sum()
    total = len(df_perf)
    
    best_idx = df_perf["Percent_Change"].idxmax()
    worst_idx = df_perf["Percent_Change"].idxmin()
    
    return {
        "total_picks": total,
        "winners": winners,
        "losers": total - winners,
        "win_rate": winners / total if total > 0 else 0,
        "avg_return": df_perf["Percent_Change"].mean(),
        "best_pick": df_perf.loc[best_idx, "Symbol"],
        "best_return": df_perf.loc[best_idx, "Percent_Change"],
        "worst_pick": df_perf.loc[worst_idx, "Symbol"],
        "worst_return": df_perf.loc[worst_idx, "Percent_Change"],
    }
