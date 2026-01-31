"""
Overview View

Portfolio-level overview showing system health and aggregate performance.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import available_dates, load_components, get_all_tickers_for_date
from data.prices import build_performance_figure
from charts.performance import (
    create_group_comparison_chart,
    create_returns_histogram,
    create_summary_metrics_cards,
)


def render_overview():
    """Render the overview page."""
    st.header("Portfolio Overview")
    st.markdown("High-level view: Is the system working?")
    
    # Get available dates
    dates = available_dates()
    
    if not dates:
        st.warning("No output data found. Run the scanner first.")
        return
    
    # Date selection
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        baseline_date = st.selectbox(
            "Baseline Date (Entry)",
            dates,
            index=min(len(dates) - 1, 5),  # Default to ~5th most recent
            help="The date when picks were made (entry point)",
        )
    
    with col2:
        # Filter end dates to be on or after baseline
        valid_end_dates = [d for d in dates if d >= baseline_date]
        if not valid_end_dates:
            valid_end_dates = [baseline_date]
        
        end_date = st.selectbox(
            "End Date (Current)",
            valid_end_dates,
            index=0,  # Default to most recent
            help="The date to measure performance against",
        )
    
    with col3:
        st.markdown("###")
        refresh = st.button("Refresh Data", use_container_width=True)
    
    # Load components for baseline date
    comps = load_components(baseline_date)
    
    # Source filters
    st.markdown("---")
    st.markdown("**Filter by Source**")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        include_weekly = st.checkbox("Weekly Top 5", value=True, disabled=not comps["weekly"])
        weekly_count = len(comps["weekly"])
        if weekly_count:
            st.caption(f"{weekly_count} tickers")
    with col2:
        include_pro30 = st.checkbox("Pro30", value=True, disabled=not comps["pro30"])
        pro30_count = len(comps["pro30"])
        if pro30_count:
            st.caption(f"{pro30_count} tickers")
    with col3:
        include_movers = st.checkbox("Movers", value=True, disabled=not comps["movers"])
        movers_count = len(comps["movers"])
        if movers_count:
            st.caption(f"{movers_count} tickers")
    
    # Build ticker list
    tickers = []
    if include_weekly:
        tickers.extend(comps["weekly"])
    if include_pro30:
        tickers.extend(comps["pro30"])
    if include_movers:
        tickers.extend(comps["movers"])
    
    # Deduplicate while preserving order
    seen = set()
    unique_tickers = []
    for t in tickers:
        if t not in seen:
            unique_tickers.append(t)
            seen.add(t)
    tickers = unique_tickers
    
    if not tickers:
        st.warning("No tickers selected. Enable at least one source.")
        return
    
    st.markdown("---")
    
    # Performance section
    with st.spinner(f"Loading performance data for {len(tickers)} tickers..."):
        fig, df_perf = build_performance_figure(
            tickers,
            baseline_date=baseline_date,
            end_date=end_date,
            group_sets=comps,
        )
    
    if df_perf.empty:
        st.warning("No performance data available for the selected period.")
        return
    
    # Summary metrics
    metrics = create_summary_metrics_cards(df_perf)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Total Picks",
            metrics["total_picks"],
            help="Number of unique tickers tracked",
        )
    with col2:
        st.metric(
            "Win Rate",
            f"{metrics['win_rate']:.0%}",
            delta=f"{metrics['winners']}W / {metrics['losers']}L",
            delta_color="normal",
        )
    with col3:
        st.metric(
            "Avg Return",
            f"{metrics['avg_return']:.1f}%",
            delta_color="normal" if metrics["avg_return"] >= 0 else "inverse",
        )
    with col4:
        st.metric(
            "Best / Worst",
            f"{metrics['best_pick']} / {metrics['worst_pick']}",
            delta=f"+{metrics['best_return']:.1f}% / {metrics['worst_return']:.1f}%",
        )
    
    st.markdown("---")
    
    # Main performance chart
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Group comparison
    st.subheader("Performance by Source")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        group_fig = create_group_comparison_chart(df_perf, comps, title="Average Return by Source")
        st.plotly_chart(group_fig, use_container_width=True)
    
    with col2:
        hist_fig = create_returns_histogram(df_perf, title="Return Distribution")
        st.plotly_chart(hist_fig, use_container_width=True)
    
    # Performance table
    st.subheader("Detailed Performance Table")
    
    # Add source column to df
    df_display = df_perf.copy()
    
    def get_sources(symbol):
        sources = []
        if symbol in comps["weekly"]:
            sources.append("Weekly")
        if symbol in comps["pro30"]:
            sources.append("Pro30")
        if symbol in comps["movers"]:
            sources.append("Movers")
        return ", ".join(sources) if sources else "Unknown"
    
    df_display["Source"] = df_display["Symbol"].apply(get_sources)
    df_display["Return"] = df_display["Percent_Change"].apply(lambda x: f"{x:+.2f}%")
    
    # Reorder columns
    display_cols = ["Symbol", "Source", "Baseline", "Current", "Return", "Baseline_Date_Used"]
    df_display = df_display[display_cols]
    df_display.columns = ["Ticker", "Source", "Entry Price", "Current Price", "Return", "Entry Date"]
    
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
    )
    
    # Download button
    csv = df_perf.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"performance_{baseline_date}_to_{end_date}.csv",
        mime="text/csv",
    )
