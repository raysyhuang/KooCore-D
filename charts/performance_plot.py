"""
Performance visualization functions using Plotly.
Trading-day aligned cumulative return charts.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def plot_cumulative_returns(
    df_cum: pd.DataFrame,
    td_counter: pd.Series,
    title: str = "Cumulative Return Since Pick Date",
    show_portfolio_avg: bool = True,
    height: int = 500,
):
    """
    Plot cumulative returns aligned by trading day.
    
    Args:
        df_cum: Cumulative returns DataFrame (% values)
        td_counter: Trading day counter series
        title: Chart title
        show_portfolio_avg: Whether to show portfolio average line
        height: Chart height in pixels
    """
    if df_cum.empty:
        return go.Figure()

    fig = go.Figure()

    # Individual ticker lines
    for t in df_cum.columns:
        s = df_cum[t].dropna()
        if s.empty:
            continue
        
        fig.add_trace(
            go.Scatter(
                x=td_counter.loc[s.index],
                y=s.values,
                mode="lines",
                name=t,
                hovertemplate=(
                    f"<b>{t}</b><br>"
                    "Trading Day: %{x}<br>"
                    "Date: %{customdata}<br>"
                    "Return: %{y:.2f}%<extra></extra>"
                ),
                customdata=[d.strftime("%Y-%m-%d") for d in s.index],
            )
        )

    # Portfolio average
    if show_portfolio_avg and len(df_cum.columns) > 1:
        avg = df_cum.mean(axis=1, skipna=True)
        if not avg.empty:
            fig.add_trace(
                go.Scatter(
                    x=td_counter.loc[avg.index],
                    y=avg.values,
                    name="Portfolio Avg",
                    line=dict(width=4, dash="dashdot", color="purple"),
                    hovertemplate=(
                        "<b>Portfolio Avg</b><br>"
                        "Trading Day: %{x}<br>"
                        "Return: %{y:.2f}%<extra></extra>"
                    ),
                )
            )

    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.4)

    fig.update_layout(
        title=title,
        height=height,
        template="plotly_white",
        hovermode="x unified",
        yaxis_title="Cumulative Return (%)",
        xaxis_title="Trading Day",
        legend_title_text="Tickers",
    )

    return fig


def plot_performance_bar(
    df_perf: pd.DataFrame,
    title: str = "Total Return by Ticker",
    height: int = 400,
):
    """
    Plot bar chart of total returns.
    
    Args:
        df_perf: Performance DataFrame with Symbol and Percent_Change columns
        title: Chart title
        height: Chart height in pixels
    """
    if df_perf.empty:
        return go.Figure()

    colors = np.where(df_perf["Percent_Change"] >= 0, "#00CC44", "#FF4444")

    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_perf["Symbol"],
        y=df_perf["Percent_Change"],
        marker_color=colors,
        text=[f"{x:+.1f}%" for x in df_perf["Percent_Change"]],
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Entry: $%{customdata[0]:.2f}<br>"
            "Current: $%{customdata[1]:.2f}<br>"
            "Return: %{y:+.2f}%<extra></extra>"
        ),
        customdata=list(zip(df_perf["Baseline"], df_perf["Current"])),
    ))

    avg_change = float(df_perf["Percent_Change"].mean())
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.4)
    fig.add_hline(
        y=avg_change,
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Avg: {avg_change:+.2f}%",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title=title,
        height=height,
        template="plotly_white",
        xaxis_title="Ticker",
        yaxis_title="Return (%)",
    )
    fig.update_xaxes(tickangle=45)

    return fig


def plot_combined_performance(
    df_perf: pd.DataFrame,
    df_cum: pd.DataFrame,
    td_counter: pd.Series,
    baseline_date: str,
    end_date: str,
    show_portfolio_avg: bool = True,
):
    """
    Create combined bar + cumulative line chart.
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            f"Total Return: {baseline_date} → {end_date}",
            "Cumulative Return Over Time (Trading-Day Aligned)",
        ),
        row_heights=[0.4, 0.6],
        vertical_spacing=0.12,
    )

    # Bar chart (top)
    if not df_perf.empty:
        colors = np.where(df_perf["Percent_Change"] >= 0, "#00CC44", "#FF4444")
        
        fig.add_trace(
            go.Bar(
                x=df_perf["Symbol"],
                y=df_perf["Percent_Change"],
                marker_color=colors,
                text=[f"{x:+.1f}%" for x in df_perf["Percent_Change"]],
                textposition="outside",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Entry: $%{customdata[0]:.2f}<br>"
                    "Current: $%{customdata[1]:.2f}<br>"
                    "Return: %{y:+.2f}%<extra></extra>"
                ),
                customdata=list(zip(df_perf["Baseline"], df_perf["Current"])),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # Cumulative returns (bottom)
    if not df_cum.empty:
        for t in df_cum.columns:
            s = df_cum[t].dropna()
            if s.empty:
                continue
            
            fig.add_trace(
                go.Scatter(
                    x=td_counter.loc[s.index],
                    y=s.values,
                    mode="lines",
                    name=t,
                    hovertemplate=(
                        f"<b>{t}</b><br>"
                        "Trading Day: %{x}<br>"
                        "Return: %{y:.2f}%<extra></extra>"
                    ),
                ),
                row=2,
                col=1,
            )

        # Portfolio average
        if show_portfolio_avg and len(df_cum.columns) > 1:
            avg = df_cum.mean(axis=1, skipna=True)
            if not avg.empty:
                fig.add_trace(
                    go.Scatter(
                        x=td_counter.loc[avg.index],
                        y=avg.values,
                        name="Portfolio Avg",
                        line=dict(width=4, dash="dashdot", color="purple"),
                    ),
                    row=2,
                    col=1,
                )

    # Add reference lines
    if not df_perf.empty:
        avg_change = float(df_perf["Percent_Change"].mean())
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.4, row=1, col=1)
        fig.add_hline(y=avg_change, line_dash="dash", line_color="blue", row=1, col=1)
    
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.4, row=2, col=1)

    fig.update_layout(
        height=900,
        template="plotly_white",
        legend_title_text="Tickers",
    )
    
    fig.update_xaxes(title_text="Ticker", tickangle=45, row=1, col=1)
    fig.update_xaxes(title_text="Trading Day", row=2, col=1)
    fig.update_yaxes(title_text="Return (%)", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Return (%)", row=2, col=1)

    return fig
