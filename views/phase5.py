"""
Phase-5 Learning Monitor View

Visualize learning diagnostics from Phase-5 data.
Read-only access to merged parquet and metrics.
"""

import streamlit as st
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import load_phase5_merged, load_phase5_metrics
from charts.learning import (
    hit_rate_by_rank,
    hit_rate_by_regime,
    hit_rate_by_source,
    rank_decay_curve,
    overlap_effectiveness,
    hit_rate_over_time,
    learning_summary_metrics,
)


def render_phase5():
    """Render the Phase-5 learning monitor page."""
    st.header("Phase-5 Learning Monitor")
    st.markdown("Is learning converging? Analyze hit rates and signal effectiveness.")
    
    # Load Phase-5 data
    df = load_phase5_merged()
    metrics = load_phase5_metrics()
    
    if df is None or df.empty:
        st.warning("No Phase-5 data available yet.")
        st.markdown("""
        Phase-5 data is generated after picks have had time to resolve (typically 7+ trading days).
        
        **Expected files:**
        - `outputs/phase5/merged/phase5_merged.parquet`
        - `outputs/phase5/metrics/*.json`
        
        Once your picks have outcomes recorded, the Phase-5 learning data will appear here.
        """)
        
        # Show placeholder charts
        st.markdown("---")
        st.subheader("What you'll see here")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Hit Rate Analysis**
            - By signal source (Weekly, Pro30, Movers)
            - By market regime (Bull, Bear, Neutral)
            - By rank position
            """)
        with col2:
            st.markdown("""
            **Learning Diagnostics**
            - Rank decay curves
            - Overlap effectiveness
            - Hit rate trends over time
            """)
        
        return
    
    # Detect available columns
    available_cols = df.columns.tolist()
    
    # Try to identify outcome column
    outcome_col = None
    for candidate in ["outcome_7d", "hit_7pct", "outcome", "hit"]:
        if candidate in available_cols:
            outcome_col = candidate
            break
    
    if outcome_col is None:
        st.warning("Could not identify outcome column in Phase-5 data.")
        st.markdown(f"Available columns: {', '.join(available_cols)}")
        return
    
    # Summary metrics
    summary = learning_summary_metrics(df, outcome_col)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Samples",
            summary["total_samples"],
            help="Total number of resolved picks",
        )
    
    with col2:
        hr = summary["overall_hit_rate"]
        st.metric(
            "Overall Hit Rate",
            f"{hr:.1%}",
            delta="Above 50%" if hr > 0.5 else "Below 50%",
            delta_color="normal" if hr > 0.5 else "inverse",
        )
    
    with col3:
        recent_hr = summary["recent_hit_rate"]
        st.metric(
            "Recent Hit Rate (last 20)",
            f"{recent_hr:.1%}",
            delta=f"{(recent_hr - hr):.1%} vs overall" if hr > 0 else None,
            delta_color="normal" if recent_hr >= hr else "inverse",
        )
    
    with col4:
        st.metric(
            "Best Source",
            summary["best_source"],
            delta=f"{summary['best_source_hit_rate']:.0%}",
        )
    
    st.markdown("---")
    
    # Main analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "By Source",
        "By Rank",
        "By Regime",
        "Over Time",
    ])
    
    with tab1:
        st.subheader("Hit Rate by Signal Source")
        
        if "source" in available_cols:
            fig = hit_rate_by_source(df, "source", outcome_col)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show source breakdown table
            source_stats = df.groupby("source")[outcome_col].agg(["mean", "count", "sum"]).reset_index()
            source_stats.columns = ["Source", "Hit Rate", "Total", "Hits"]
            source_stats["Hit Rate"] = source_stats["Hit Rate"].apply(lambda x: f"{x:.1%}")
            source_stats = source_stats.sort_values("Hits", ascending=False)
            
            st.dataframe(source_stats, use_container_width=True, hide_index=True)
        else:
            st.info("Source column not available in data.")
        
        # Overlap effectiveness
        if "overlap_count" in available_cols:
            st.markdown("---")
            st.subheader("Overlap Effectiveness")
            st.markdown("Do picks flagged by multiple sources perform better?")
            
            fig = overlap_effectiveness(df, "overlap_count", outcome_col)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Hit Rate by Rank")
        
        # Find rank column
        rank_col = None
        for candidate in ["pro30_rank", "rank", "weekly_rank"]:
            if candidate in available_cols:
                rank_col = candidate
                break
        
        if rank_col:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = hit_rate_by_rank(df, rank_col, outcome_col)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = rank_decay_curve(df, rank_col, outcome_col)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Interpretation:**
            - Ideally, hit rate should decay as rank increases (rank 1 > rank 2 > ...)
            - Steeper decay = stronger ranking signal
            - Flat curve = ranking may not be predictive
            """)
        else:
            st.info("Rank column not available in data.")
    
    with tab3:
        st.subheader("Hit Rate by Market Regime")
        
        regime_col = None
        for candidate in ["regime", "market_regime", "regime_label"]:
            if candidate in available_cols:
                regime_col = candidate
                break
        
        if regime_col:
            fig = hit_rate_by_regime(df, regime_col, outcome_col)
            st.plotly_chart(fig, use_container_width=True)
            
            # Regime breakdown
            regime_stats = df.groupby(regime_col)[outcome_col].agg(["mean", "count"]).reset_index()
            regime_stats.columns = ["Regime", "Hit Rate", "Count"]
            regime_stats["Hit Rate"] = regime_stats["Hit Rate"].apply(lambda x: f"{x:.1%}")
            
            st.dataframe(regime_stats, use_container_width=True, hide_index=True)
            
            st.markdown("""
            **Interpretation:**
            - System may perform better in certain market conditions
            - Large performance gaps suggest regime-dependent strategy effectiveness
            """)
        else:
            st.info("Regime column not available in data.")
    
    with tab4:
        st.subheader("Hit Rate Over Time")
        
        date_col = None
        for candidate in ["pick_date", "date", "entry_date", "scan_date"]:
            if candidate in available_cols:
                date_col = candidate
                break
        
        if date_col:
            # Rolling window selector
            window = st.slider("Rolling Window (days)", 5, 30, 10)
            
            fig = hit_rate_over_time(df, date_col, outcome_col, rolling_window=window)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Interpretation:**
            - Upward trend = learning is improving
            - Downward trend = model may need recalibration
            - High variance = more data needed
            """)
        else:
            st.info("Date column not available in data.")
    
    # Raw data explorer
    st.markdown("---")
    st.subheader("Data Explorer")
    
    with st.expander("View Raw Phase-5 Data"):
        # Column selector
        selected_cols = st.multiselect(
            "Select columns to display",
            available_cols,
            default=available_cols[:min(10, len(available_cols))],
        )
        
        if selected_cols:
            st.dataframe(
                df[selected_cols].tail(100),
                use_container_width=True,
                hide_index=True,
            )
        
        # Download
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Full Dataset",
            data=csv,
            file_name="phase5_merged.csv",
            mime="text/csv",
        )
    
    # Metrics files
    if metrics:
        with st.expander("View Phase-5 Metrics"):
            for i, m in enumerate(metrics[-5:]):  # Show last 5
                st.json(m)
