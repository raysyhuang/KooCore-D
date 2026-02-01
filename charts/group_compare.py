"""
Group comparison charts - Weekly vs Pro30 vs Movers performance.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

from charts.performance_core import compute_cumulative_returns


# Color scheme for groups
GROUP_COLORS = {
    "weekly": "#4A90D9",      # Blue
    "pro30": "#7CB342",       # Green
    "movers": "#FF7043",      # Orange
    "hybrid_top3": "#9C27B0", # Purple
    "conviction": "#F44336",  # Red
    "all": "#607D8B",         # Gray
}


def compute_group_average_returns(
    close_df: pd.DataFrame,
    baseline_date: str,
    end_date: str,
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute equal-weight average cumulative returns for a group of tickers.
    
    Returns:
        avg_cum: Average cumulative return series
        td_counter: Trading day counter
    """
    df_cum, td_counter, _ = compute_cumulative_returns(close_df, baseline_date, end_date)
    
    if df_cum.empty:
        return pd.Series(dtype=float), td_counter
    
    avg_cum = df_cum.mean(axis=1, skipna=True)
    return avg_cum, td_counter


def plot_group_comparison(
    group_data: Dict[str, pd.DataFrame],
    baseline_date: str,
    end_date: str,
    show_individual_tickers: bool = False,
    height: int = 600,
):
    """
    Plot group average cumulative returns comparison.
    
    Args:
        group_data: Dict of group_name -> close_df (tickers as columns)
        baseline_date: Common baseline date
        end_date: End date
        show_individual_tickers: Whether to show faint individual ticker lines
        height: Chart height
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    group_stats = []
    
    for group_name, close_df in group_data.items():
        if close_df.empty:
            continue
        
        avg_cum, td_counter = compute_group_average_returns(close_df, baseline_date, end_date)
        
        if avg_cum.empty:
            continue
        
        color = GROUP_COLORS.get(group_name, "#888888")
        final_return = float(avg_cum.iloc[-1]) if not avg_cum.empty else 0
        
        group_stats.append({
            "group": group_name,
            "tickers": len(close_df.columns),
            "final_return": final_return,
        })
        
        # Show individual tickers (faint)
        if show_individual_tickers:
            df_cum, _, _ = compute_cumulative_returns(close_df, baseline_date, end_date)
            for ticker in df_cum.columns:
                s = df_cum[ticker].dropna()
                if s.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=td_counter.loc[s.index],
                    y=s.values,
                    mode="lines",
                    name=ticker,
                    line={"width": 0.8, "color": color},
                    opacity=0.3,
                    legendgroup=group_name,
                    showlegend=False,
                    hoverinfo="skip",
                ))
        
        # Group average (bold)
        fig.add_trace(go.Scatter(
            x=td_counter.loc[avg_cum.index],
            y=avg_cum.values,
            mode="lines",
            name=f"{group_name.title()} ({len(close_df.columns)} tickers)",
            line={"width": 3, "color": color},
            legendgroup=group_name,
            hovertemplate=(
                f"<b>{group_name.title()}</b><br>"
                "Trading Day: %{x}<br>"
                "Avg Return: %{y:.2f}%<extra></extra>"
            ),
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.4)
    
    fig.update_layout(
        title=f"Group Performance: {baseline_date} → {end_date}",
        template="plotly_white",
        height=height,
        hovermode="x unified",
        xaxis_title="Trading Day",
        yaxis_title="Cumulative Return (%)",
        legend_title_text="Groups",
    )
    
    return fig, group_stats


def plot_group_bar_comparison(
    group_stats: List[Dict],
    baseline_date: str,
    end_date: str,
    height: int = 400,
):
    """
    Plot bar chart comparing final returns across groups.
    """
    if not group_stats:
        return go.Figure()
    
    df = pd.DataFrame(group_stats).sort_values("final_return", ascending=False)
    
    colors = [GROUP_COLORS.get(g, "#888888") for g in df["group"]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df["group"].str.title(),
        y=df["final_return"],
        marker_color=colors,
        text=[f"{r:+.1f}%" for r in df["final_return"]],
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Tickers: %{customdata}<br>"
            "Return: %{y:+.2f}%<extra></extra>"
        ),
        customdata=df["tickers"],
    ))
    
    avg_return = df["final_return"].mean()
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.4)
    fig.add_hline(
        y=avg_return,
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Avg: {avg_return:+.1f}%",
        annotation_position="bottom right",
    )
    
    fig.update_layout(
        title=f"Group Returns: {baseline_date} → {end_date}",
        template="plotly_white",
        height=height,
        xaxis_title="Signal Group",
        yaxis_title="Avg Return (%)",
    )
    
    return fig


def plot_combined_group_view(
    group_data: Dict[str, pd.DataFrame],
    baseline_date: str,
    end_date: str,
    show_individual_tickers: bool = False,
):
    """
    Create combined view with bar chart (top) and cumulative lines (bottom).
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            f"Final Returns by Group",
            f"Cumulative Returns Over Time",
        ),
        row_heights=[0.35, 0.65],
        vertical_spacing=0.12,
    )
    
    group_stats = []
    
    # Compute stats and plot cumulative lines
    for group_name, close_df in group_data.items():
        if close_df.empty:
            continue
        
        avg_cum, td_counter = compute_group_average_returns(close_df, baseline_date, end_date)
        
        if avg_cum.empty:
            continue
        
        color = GROUP_COLORS.get(group_name, "#888888")
        final_return = float(avg_cum.iloc[-1]) if not avg_cum.empty else 0
        
        group_stats.append({
            "group": group_name,
            "tickers": len(close_df.columns),
            "final_return": final_return,
            "color": color,
        })
        
        # Cumulative line (bottom)
        fig.add_trace(
            go.Scatter(
                x=td_counter.loc[avg_cum.index],
                y=avg_cum.values,
                mode="lines",
                name=f"{group_name.title()}",
                line={"width": 3, "color": color},
                hovertemplate=(
                    f"<b>{group_name.title()}</b><br>"
                    "TD: %{x}<br>"
                    "Return: %{y:.2f}%<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )
    
    # Bar chart (top)
    if group_stats:
        df = pd.DataFrame(group_stats).sort_values("final_return", ascending=False)
        
        fig.add_trace(
            go.Bar(
                x=df["group"].str.title(),
                y=df["final_return"],
                marker_color=df["color"].tolist(),
                text=[f"{r:+.1f}%" for r in df["final_return"]],
                textposition="outside",
                showlegend=False,
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Tickers: %{customdata}<br>"
                    "Return: %{y:+.2f}%<extra></extra>"
                ),
                customdata=df["tickers"],
            ),
            row=1,
            col=1,
        )
    
    # Reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.4, row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.4, row=2, col=1)
    
    fig.update_layout(
        height=800,
        template="plotly_white",
        legend_title_text="Groups",
    )
    
    fig.update_xaxes(title_text="Group", row=1, col=1)
    fig.update_xaxes(title_text="Trading Day", row=2, col=1)
    fig.update_yaxes(title_text="Return (%)", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Return (%)", row=2, col=1)
    
    return fig, group_stats
