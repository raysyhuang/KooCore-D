"""
Learning Charts Module

Charts for Phase-5 learning diagnostics.
All functions return Plotly figures - no .show() calls.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional


def hit_rate_by_rank(df: pd.DataFrame, rank_col: str = "pro30_rank", outcome_col: str = "outcome_7d") -> go.Figure:
    """
    Create bar chart showing hit rate by rank.
    
    Args:
        df: Phase-5 merged DataFrame
        rank_col: Column name for rank
        outcome_col: Column name for outcome (1 = hit, 0 = miss)
    
    Returns:
        Plotly Figure
    """
    if df.empty or rank_col not in df.columns or outcome_col not in df.columns:
        return go.Figure().add_annotation(text="No data available", showarrow=False)
    
    # Group by rank and calculate hit rate
    grouped = df.groupby(rank_col)[outcome_col].agg(["mean", "count"]).reset_index()
    grouped.columns = [rank_col, "hit_rate", "count"]
    
    fig = px.bar(
        grouped,
        x=rank_col,
        y="hit_rate",
        title="Hit Rate by Rank",
        labels={rank_col: "Rank", "hit_rate": "Hit Rate"},
        text=[f"{x:.0%}<br>(n={n})" for x, n in zip(grouped["hit_rate"], grouped["count"])],
    )
    
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="50%")
    
    fig.update_traces(textposition="outside")
    fig.update_layout(
        template="plotly_white",
        height=400,
        yaxis_tickformat=".0%",
    )
    
    return fig


def hit_rate_by_regime(df: pd.DataFrame, regime_col: str = "regime", outcome_col: str = "outcome_7d") -> go.Figure:
    """
    Create bar chart showing hit rate by market regime.
    
    Args:
        df: Phase-5 merged DataFrame
        regime_col: Column name for regime
        outcome_col: Column name for outcome
    
    Returns:
        Plotly Figure
    """
    if df.empty or regime_col not in df.columns or outcome_col not in df.columns:
        return go.Figure().add_annotation(text="No data available", showarrow=False)
    
    grouped = df.groupby(regime_col)[outcome_col].agg(["mean", "count"]).reset_index()
    grouped.columns = [regime_col, "hit_rate", "count"]
    
    # Color by regime type
    color_map = {
        "bull": "#00CC44",
        "bear": "#FF4444",
        "neutral": "#888888",
        "volatile": "#FF8800",
    }
    colors = [color_map.get(r.lower(), "#888888") for r in grouped[regime_col]]
    
    fig = go.Figure(data=[
        go.Bar(
            x=grouped[regime_col],
            y=grouped["hit_rate"],
            marker_color=colors,
            text=[f"{x:.0%}<br>(n={n})" for x, n in zip(grouped["hit_rate"], grouped["count"])],
            textposition="outside",
        )
    ])
    
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="50%")
    
    fig.update_layout(
        title="Hit Rate by Market Regime",
        xaxis_title="Regime",
        yaxis_title="Hit Rate",
        template="plotly_white",
        height=400,
        yaxis_tickformat=".0%",
    )
    
    return fig


def hit_rate_by_source(df: pd.DataFrame, source_col: str = "source", outcome_col: str = "outcome_7d") -> go.Figure:
    """
    Create bar chart showing hit rate by signal source.
    
    Args:
        df: Phase-5 merged DataFrame
        source_col: Column name for source
        outcome_col: Column name for outcome
    
    Returns:
        Plotly Figure
    """
    if df.empty or source_col not in df.columns or outcome_col not in df.columns:
        return go.Figure().add_annotation(text="No data available", showarrow=False)
    
    grouped = df.groupby(source_col)[outcome_col].agg(["mean", "count"]).reset_index()
    grouped.columns = [source_col, "hit_rate", "count"]
    grouped = grouped.sort_values("hit_rate", ascending=False)
    
    colors = np.where(grouped["hit_rate"] >= 0.5, "#00CC44", "#FF4444")
    
    fig = go.Figure(data=[
        go.Bar(
            x=grouped[source_col],
            y=grouped["hit_rate"],
            marker_color=colors,
            text=[f"{x:.0%}<br>(n={n})" for x, n in zip(grouped["hit_rate"], grouped["count"])],
            textposition="outside",
        )
    ])
    
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="50%")
    
    fig.update_layout(
        title="Hit Rate by Signal Source",
        xaxis_title="Source",
        yaxis_title="Hit Rate",
        template="plotly_white",
        height=400,
        yaxis_tickformat=".0%",
    )
    
    return fig


def rank_decay_curve(df: pd.DataFrame, rank_col: str = "pro30_rank", outcome_col: str = "outcome_7d") -> go.Figure:
    """
    Create line chart showing hit rate decay as rank increases.
    
    Args:
        df: Phase-5 merged DataFrame
        rank_col: Column name for rank
        outcome_col: Column name for outcome
    
    Returns:
        Plotly Figure
    """
    if df.empty or rank_col not in df.columns or outcome_col not in df.columns:
        return go.Figure().add_annotation(text="No data available", showarrow=False)
    
    grouped = df.groupby(rank_col)[outcome_col].agg(["mean", "count"]).reset_index()
    grouped.columns = [rank_col, "hit_rate", "count"]
    grouped = grouped.sort_values(rank_col)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=grouped[rank_col],
        y=grouped["hit_rate"],
        mode="lines+markers",
        name="Hit Rate",
        line=dict(color="purple", width=2),
        marker=dict(size=8),
    ))
    
    # Add count as secondary trace
    fig.add_trace(go.Bar(
        x=grouped[rank_col],
        y=grouped["count"],
        name="Sample Count",
        marker_color="rgba(100, 100, 100, 0.3)",
        yaxis="y2",
    ))
    
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title="Rank Decay Curve (Hit Rate vs Rank)",
        xaxis_title="Rank",
        yaxis_title="Hit Rate",
        yaxis2=dict(
            title="Count",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        template="plotly_white",
        height=400,
        yaxis_tickformat=".0%",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )
    
    return fig


def overlap_effectiveness(df: pd.DataFrame, overlap_col: str = "overlap_count", outcome_col: str = "outcome_7d") -> go.Figure:
    """
    Create chart showing effectiveness of overlapping signals.
    
    Args:
        df: Phase-5 merged DataFrame
        overlap_col: Column name for overlap count
        outcome_col: Column name for outcome
    
    Returns:
        Plotly Figure
    """
    if df.empty or overlap_col not in df.columns or outcome_col not in df.columns:
        return go.Figure().add_annotation(text="No data available", showarrow=False)
    
    grouped = df.groupby(overlap_col)[outcome_col].agg(["mean", "count"]).reset_index()
    grouped.columns = [overlap_col, "hit_rate", "count"]
    
    fig = go.Figure(data=[
        go.Bar(
            x=[f"{int(x)} sources" for x in grouped[overlap_col]],
            y=grouped["hit_rate"],
            marker_color=["#888888", "#00AA00", "#006600"][:len(grouped)],
            text=[f"{x:.0%}<br>(n={n})" for x, n in zip(grouped["hit_rate"], grouped["count"])],
            textposition="outside",
        )
    ])
    
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="50%")
    
    fig.update_layout(
        title="Overlap Effectiveness (Multiple Signal Sources)",
        xaxis_title="Number of Overlapping Sources",
        yaxis_title="Hit Rate",
        template="plotly_white",
        height=400,
        yaxis_tickformat=".0%",
    )
    
    return fig


def hit_rate_over_time(
    df: pd.DataFrame,
    date_col: str = "pick_date",
    outcome_col: str = "outcome_7d",
    rolling_window: int = 10
) -> go.Figure:
    """
    Create time series chart of hit rate with rolling average.
    
    Args:
        df: Phase-5 merged DataFrame
        date_col: Column name for date
        outcome_col: Column name for outcome
        rolling_window: Window size for rolling average
    
    Returns:
        Plotly Figure
    """
    if df.empty or date_col not in df.columns or outcome_col not in df.columns:
        return go.Figure().add_annotation(text="No data available", showarrow=False)
    
    # Group by date
    daily = df.groupby(date_col)[outcome_col].agg(["mean", "count"]).reset_index()
    daily.columns = [date_col, "hit_rate", "count"]
    daily = daily.sort_values(date_col)
    
    # Calculate rolling average
    daily["rolling_hit_rate"] = daily["hit_rate"].rolling(window=rolling_window, min_periods=1).mean()
    
    fig = go.Figure()
    
    # Daily hit rate as scatter
    fig.add_trace(go.Scatter(
        x=daily[date_col],
        y=daily["hit_rate"],
        mode="markers",
        name="Daily",
        marker=dict(size=6, color="rgba(100, 100, 200, 0.5)"),
    ))
    
    # Rolling average as line
    fig.add_trace(go.Scatter(
        x=daily[date_col],
        y=daily["rolling_hit_rate"],
        mode="lines",
        name=f"{rolling_window}-day Rolling Avg",
        line=dict(color="purple", width=2),
    ))
    
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="50%")
    
    fig.update_layout(
        title="Hit Rate Over Time",
        xaxis_title="Date",
        yaxis_title="Hit Rate",
        template="plotly_white",
        height=400,
        yaxis_tickformat=".0%",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    
    return fig


def learning_summary_metrics(df: pd.DataFrame, outcome_col: str = "outcome_7d") -> dict:
    """
    Calculate summary metrics for learning dashboard.
    
    Args:
        df: Phase-5 merged DataFrame
        outcome_col: Column name for outcome
    
    Returns:
        Dict with metric values
    """
    if df.empty or outcome_col not in df.columns:
        return {
            "total_samples": 0,
            "overall_hit_rate": 0,
            "recent_hit_rate": 0,
            "best_source": "N/A",
            "best_source_hit_rate": 0,
        }
    
    total = len(df)
    overall_hr = df[outcome_col].mean()
    
    # Recent hit rate (last 20 samples)
    recent = df.tail(20)
    recent_hr = recent[outcome_col].mean() if not recent.empty else 0
    
    # Best source
    if "source" in df.columns:
        by_source = df.groupby("source")[outcome_col].mean()
        best_source = by_source.idxmax()
        best_source_hr = by_source.max()
    else:
        best_source = "N/A"
        best_source_hr = 0
    
    return {
        "total_samples": total,
        "overall_hit_rate": overall_hr,
        "recent_hit_rate": recent_hr,
        "best_source": best_source,
        "best_source_hit_rate": best_source_hr,
    }
