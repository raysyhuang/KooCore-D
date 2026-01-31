"""
Daily Picks Explorer View

Explore individual picks and their performance.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import available_dates, load_components
from data.prices import build_performance_figure, get_ticker_performance


def render_picks():
    """Render the daily picks explorer page."""
    st.header("Daily Picks Explorer")
    st.markdown("Explore individual picks: What did we pick and how is it doing?")
    
    # Get available dates
    dates = available_dates()
    
    if not dates:
        st.warning("No output data found. Run the scanner first.")
        return
    
    # Date selection
    col1, col2 = st.columns([1, 1])
    
    with col1:
        scan_date = st.selectbox(
            "Scan Date",
            dates,
            index=0,
            help="The date when picks were made",
        )
    
    with col2:
        # End date defaults to today or most recent available
        today_str = datetime.now().strftime("%Y-%m-%d")
        end_date = st.selectbox(
            "End Date (for returns)",
            [today_str] + dates,
            index=0,
            help="The date to measure performance against",
        )
    
    # Load components
    comps = load_components(scan_date)
    
    # Source filter
    st.markdown("---")
    
    sources = st.multiselect(
        "Filter by Source",
        ["weekly", "pro30", "movers"],
        default=["weekly", "pro30"],
        format_func=lambda x: {"weekly": "Weekly Top 5", "pro30": "Pro30", "movers": "Movers"}.get(x, x),
    )
    
    # Build ticker list based on selection
    tickers = []
    ticker_source_map = {}  # Track which source each ticker came from
    
    for src in sources:
        for t in comps.get(src, []):
            if t not in ticker_source_map:
                tickers.append(t)
                ticker_source_map[t] = []
            ticker_source_map[t].append(src)
    
    if not tickers:
        st.warning("No tickers available for the selected date and sources.")
        return
    
    # Show summary
    st.markdown(f"**{len(tickers)} tickers** selected from {scan_date}")
    
    # Fetch performance data
    with st.spinner("Loading performance data..."):
        fig, df_perf = build_performance_figure(
            tickers,
            baseline_date=scan_date,
            end_date=end_date,
            group_sets=comps,
        )
    
    # Summary metrics row
    if not df_perf.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        winners = (df_perf["Percent_Change"] > 0).sum()
        losers = len(df_perf) - winners
        avg_return = df_perf["Percent_Change"].mean()
        
        with col1:
            st.metric("Total Picks", len(df_perf))
        with col2:
            st.metric("Winners", winners, delta=f"{winners/len(df_perf):.0%}")
        with col3:
            st.metric("Losers", losers)
        with col4:
            delta_color = "normal" if avg_return >= 0 else "inverse"
            st.metric("Avg Return", f"{avg_return:.2f}%")
    
    st.markdown("---")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Table View", "Chart View"])
    
    with tab1:
        if df_perf.empty:
            st.warning("No performance data available.")
        else:
            # Add source column
            df_display = df_perf.copy()
            df_display["Source"] = df_display["Symbol"].apply(
                lambda x: ", ".join(ticker_source_map.get(x, ["Unknown"]))
            )
            df_display["Status"] = df_display["Percent_Change"].apply(
                lambda x: "Winner" if x > 0 else "Loser"
            )
            
            # Format return
            df_display["Return"] = df_display["Percent_Change"].apply(lambda x: f"{x:+.2f}%")
            
            # Select and reorder columns
            display_cols = ["Symbol", "Source", "Status", "Baseline", "Current", "Return", "Baseline_Date_Used"]
            df_display = df_display[display_cols]
            df_display.columns = ["Ticker", "Source", "Status", "Entry", "Current", "Return", "Entry Date"]
            
            # Style the dataframe
            def highlight_status(row):
                if row["Status"] == "Winner":
                    return ["background-color: #ccffcc"] * len(row)
                else:
                    return ["background-color: #ffcccc"] * len(row)
            
            st.dataframe(
                df_display.style.apply(highlight_status, axis=1),
                use_container_width=True,
                hide_index=True,
            )
    
    with tab2:
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No chart data available.")
    
    # Individual ticker detail
    st.markdown("---")
    st.subheader("Individual Ticker Detail")
    
    selected_ticker = st.selectbox(
        "Select a ticker to view details",
        tickers,
        format_func=lambda x: f"{x} ({', '.join(ticker_source_map.get(x, ['Unknown']))})",
    )
    
    if selected_ticker:
        # Get detail from weekly_details if available
        weekly_details = {d["ticker"]: d for d in comps.get("weekly_details", []) if d.get("ticker")}
        hybrid_details = {d["ticker"]: d for d in comps.get("hybrid_top3", []) if d.get("ticker")}
        
        detail = weekly_details.get(selected_ticker) or hybrid_details.get(selected_ticker)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Pick Details**")
            
            # Show pick info
            if detail:
                if detail.get("name"):
                    st.markdown(f"**Name:** {detail['name']}")
                if detail.get("sector"):
                    st.markdown(f"**Sector:** {detail['sector']}")
                if detail.get("rank"):
                    st.markdown(f"**Rank:** #{detail['rank']}")
                if detail.get("composite_score"):
                    st.markdown(f"**Composite Score:** {detail['composite_score']:.2f}")
                if detail.get("confidence"):
                    st.markdown(f"**Confidence:** {detail['confidence']}")
                
                # Show scores if available
                scores = detail.get("scores", {})
                if scores:
                    st.markdown("**Component Scores:**")
                    for key, val in scores.items():
                        if val is not None:
                            st.markdown(f"- {key}: {val:.1f}")
            
            # Show sources
            st.markdown(f"**Sources:** {', '.join(ticker_source_map.get(selected_ticker, ['Unknown']))}")
        
        with col2:
            # Show individual ticker chart
            with st.spinner(f"Loading {selected_ticker} chart..."):
                ticker_fig, ticker_stats = get_ticker_performance(
                    selected_ticker,
                    scan_date,
                    end_date,
                )
            
            if ticker_fig:
                st.plotly_chart(ticker_fig, use_container_width=True)
            
            if ticker_stats:
                st.markdown(f"**Performance:** {ticker_stats.get('return_pct', 0):+.2f}%")
                st.markdown(f"Entry: ${ticker_stats.get('baseline', 0):.2f} → Current: ${ticker_stats.get('current', 0):.2f}")
        
        # Show catalyst if available
        if detail and detail.get("primary_catalyst"):
            st.markdown("---")
            st.markdown("**Primary Catalyst**")
            catalyst = detail["primary_catalyst"]
            st.markdown(f"**{catalyst.get('title', 'N/A')}**")
            if catalyst.get("why_it_matters"):
                st.markdown(f"_{catalyst['why_it_matters']}_")
            if catalyst.get("timing"):
                st.markdown(f"Timing: {catalyst['timing']}")
        
        # Show risk factors if available
        if detail and detail.get("risk_factors"):
            st.markdown("---")
            st.markdown("**Risk Factors**")
            for risk in detail["risk_factors"]:
                st.markdown(f"- {risk}")
