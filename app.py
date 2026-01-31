"""
KooCore-D Dashboard (Option A + History Snapshots)

Two data sources:
1. Live Mode: Pull latest artifact from GitHub Actions
2. Historical Mode: Load snapshots from data/phase5/*.parquet

Read-only analytics - no scans, no trades.
"""

import os
import io
import json
import zipfile
import glob
from datetime import datetime
from typing import Optional, Tuple, List, Dict

import pandas as pd
import numpy as np
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# =============================================================================
# Config
# =============================================================================
CORE_OWNER = os.getenv("CORE_OWNER", "raysyhuang")
CORE_REPO = os.getenv("CORE_REPO", "KooCore-D")
CORE_ARTIFACT_NAME = os.getenv("CORE_ARTIFACT_NAME", "koocore-outputs")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()

DATA_DIR = os.getenv("DATA_DIR", "data")
PHASE5_DIR = os.path.join(DATA_DIR, "phase5")
SCORECARD_DIR = os.path.join(DATA_DIR, "scorecards")


# =============================================================================
# GitHub API Helpers
# =============================================================================
def _gh_headers() -> Dict[str, str]:
    hdr = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if GITHUB_TOKEN:
        hdr["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return hdr


@st.cache_data(ttl=300, show_spinner=False)
def github_latest_artifact_zip(owner: str, repo: str, artifact_name: str) -> Tuple[bytes, Dict]:
    """Download latest artifact ZIP from GitHub Actions."""
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/artifacts?per_page=100"
    r = requests.get(url, headers=_gh_headers(), timeout=20)
    r.raise_for_status()
    
    arts = r.json().get("artifacts", [])
    candidates = [a for a in arts if a.get("name") == artifact_name and not a.get("expired", False)]
    
    if not candidates:
        raise RuntimeError(f"No artifact named '{artifact_name}' found (or all expired).")
    
    candidates.sort(key=lambda a: a.get("created_at", ""), reverse=True)
    art = candidates[0]
    
    dl_url = art["archive_download_url"]
    r2 = requests.get(dl_url, headers=_gh_headers(), timeout=120)
    r2.raise_for_status()
    
    meta = {
        "artifact_id": art.get("id"),
        "created_at": art.get("created_at"),
        "size_bytes": art.get("size_in_bytes", 0),
        "name": art.get("name"),
    }
    
    return r2.content, meta


def unzip_to_memory(zip_bytes: bytes) -> Dict[str, bytes]:
    """Extract ZIP to dict of {path: bytes}."""
    out = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        for info in z.infolist():
            if not info.is_dir():
                out[info.filename] = z.read(info.filename)
    return out


# =============================================================================
# Data Loaders
# =============================================================================
def load_phase5_from_artifact(files: Dict[str, bytes]) -> Optional[pd.DataFrame]:
    """Load phase5_merged.parquet from artifact files."""
    for path in [
        "outputs/phase5/merged/phase5_merged.parquet",
        "phase5/merged/phase5_merged.parquet",
        "outputs/phase5/phase5_merged.parquet",
    ]:
        if path in files:
            try:
                return pd.read_parquet(io.BytesIO(files[path]))
            except Exception:
                continue
    return None


def load_hybrid_analyses_from_artifact(files: Dict[str, bytes]) -> List[Dict]:
    """Load hybrid analysis JSONs from artifact."""
    analyses = []
    for path, content in files.items():
        if "hybrid_analysis" in path and path.endswith(".json"):
            try:
                data = json.loads(content.decode("utf-8"))
                data["_path"] = path
                analyses.append(data)
            except Exception:
                continue
    return sorted(analyses, key=lambda x: x.get("date", ""), reverse=True)


def load_picks_from_artifact(files: Dict[str, bytes]) -> pd.DataFrame:
    """Load all picks from hybrid analyses into a DataFrame."""
    analyses = load_hybrid_analyses_from_artifact(files)
    
    rows = []
    for h in analyses:
        date = h.get("date") or h.get("asof_trading_date")
        if not date:
            continue
        
        # Hybrid Top 3
        for pick in h.get("hybrid_top3", []):
            rows.append({
                "date": date,
                "ticker": pick.get("ticker", ""),
                "source": "hybrid_top3",
                "rank": pick.get("rank"),
                "score": pick.get("hybrid_score"),
                "confidence": pick.get("confidence"),
                "sources": ",".join(pick.get("sources", [])),
            })
        
        # Weekly/Primary Top 5
        for pick in h.get("primary_top5", []):
            rows.append({
                "date": date,
                "ticker": pick.get("ticker", ""),
                "source": "weekly",
                "rank": pick.get("rank"),
                "score": pick.get("composite_score"),
                "confidence": pick.get("confidence"),
                "sources": "Weekly",
            })
        
        # Pro30
        for ticker in h.get("pro30_tickers", []):
            rows.append({
                "date": date,
                "ticker": ticker,
                "source": "pro30",
                "rank": None,
                "score": None,
                "confidence": None,
                "sources": "Pro30",
            })
        
        # Movers
        for ticker in h.get("movers_tickers", []):
            rows.append({
                "date": date,
                "ticker": ticker,
                "source": "movers",
                "rank": None,
                "score": None,
                "confidence": None,
                "sources": "Movers",
            })
    
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def get_available_dates(files: Dict[str, bytes]) -> List[str]:
    """Extract available scan dates from artifact."""
    dates = set()
    for path in files.keys():
        parts = path.split("/")
        for part in parts:
            if len(part) == 10 and part[4] == "-" and part[7] == "-":
                try:
                    datetime.strptime(part, "%Y-%m-%d")
                    dates.add(part)
                except ValueError:
                    continue
    return sorted(dates, reverse=True)


def list_snapshot_files() -> List[str]:
    """List available historical snapshot files."""
    if not os.path.exists(PHASE5_DIR):
        return []
    return sorted(glob.glob(os.path.join(PHASE5_DIR, "phase5_merged_*.parquet")), reverse=True)


def parse_date_from_filename(path: str) -> Optional[str]:
    """Extract date from phase5_merged_YYYY-MM-DD.parquet."""
    base = os.path.basename(path)
    if "phase5_merged_" not in base:
        return None
    s = base.replace("phase5_merged_", "").replace(".parquet", "")
    try:
        datetime.strptime(s, "%Y-%m-%d")
        return s
    except ValueError:
        return None


# =============================================================================
# Charts
# =============================================================================
def chart_picks_over_time(analyses: List[Dict]):
    """Daily pick counts from hybrid analyses."""
    if not analyses:
        st.info("No hybrid analysis data.")
        return
    
    rows = []
    for h in analyses:
        date = h.get("date") or h.get("asof_trading_date")
        if not date:
            continue
        
        summary = h.get("summary", {})
        rows.append({
            "date": date,
            "Weekly": summary.get("weekly_top5_count") or summary.get("primary_top5_count") or len(h.get("primary_top5", [])),
            "Pro30": summary.get("pro30_candidates_count") or len(h.get("pro30_tickers", [])),
            "Movers": summary.get("movers_count") or len(h.get("movers_tickers", [])),
            "Hybrid Top3": summary.get("hybrid_top3_count") or len(h.get("hybrid_top3", [])),
        })
    
    if not rows:
        st.info("No summary data.")
        return
    
    df = pd.DataFrame(rows).sort_values("date")
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["date"], y=df["Weekly"], name="Weekly", marker_color="#4A90D9"))
    fig.add_trace(go.Bar(x=df["date"], y=df["Pro30"], name="Pro30", marker_color="#7CB342"))
    fig.add_trace(go.Bar(x=df["date"], y=df["Movers"], name="Movers", marker_color="#FF7043"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["Hybrid Top3"], name="Hybrid Top3", 
                             mode="lines+markers", line=dict(color="#9C27B0", width=3)))
    fig.update_layout(title="Daily Pick Counts by Source", barmode="group",
                      xaxis_title="Date", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)


def chart_picks_by_source(df: pd.DataFrame):
    """Pie chart of picks by source."""
    if df.empty:
        st.info("No picks data.")
        return
    
    counts = df["source"].value_counts().reset_index()
    counts.columns = ["source", "count"]
    
    fig = px.pie(counts, values="count", names="source", title="Picks by Source",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True)


def chart_unique_tickers_over_time(df: pd.DataFrame):
    """Unique tickers per day."""
    if df.empty or "date" not in df.columns:
        st.info("No data.")
        return
    
    daily = df.groupby("date")["ticker"].nunique().reset_index()
    daily.columns = ["date", "unique_tickers"]
    
    fig = px.line(daily, x="date", y="unique_tickers", markers=True,
                  title="Unique Tickers per Day")
    st.plotly_chart(fig, use_container_width=True)


def chart_top_tickers(df: pd.DataFrame, top_n: int = 15):
    """Most frequently picked tickers."""
    if df.empty:
        st.info("No data.")
        return
    
    counts = df["ticker"].value_counts().head(top_n).reset_index()
    counts.columns = ["ticker", "count"]
    
    fig = px.bar(counts, x="ticker", y="count", title=f"Top {top_n} Most Picked Tickers",
                 color="count", color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)


def chart_confidence_distribution(df: pd.DataFrame):
    """Distribution of confidence levels."""
    if df.empty or "confidence" not in df.columns:
        st.info("No confidence data.")
        return
    
    conf_df = df[df["confidence"].notna()].copy()
    if conf_df.empty:
        st.info("No confidence data.")
        return
    
    counts = conf_df["confidence"].value_counts().reset_index()
    counts.columns = ["confidence", "count"]
    
    color_map = {"HIGH": "#00CC44", "MEDIUM": "#FFB300", "LOW": "#FF4444"}
    fig = px.bar(counts, x="confidence", y="count", title="Confidence Distribution",
                 color="confidence", color_discrete_map=color_map)
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Main App
# =============================================================================
st.set_page_config(page_title="KooCore Dashboard", page_icon="📊", layout="wide")

st.title("📊 KooCore-D Dashboard")

# Mode selection
mode = st.radio(
    "Data Source",
    ["Live (GitHub Artifact)", "Historical Snapshots"],
    horizontal=True,
    help="Live pulls latest from GitHub Actions. Historical uses stored snapshots."
)

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    if mode == "Live (GitHub Artifact)":
        st.caption(f"Source: {CORE_OWNER}/{CORE_REPO}")
        st.caption(f"Artifact: {CORE_ARTIFACT_NAME}")
        
        if st.button("🔄 Refresh"):
            github_latest_artifact_zip.clear()
            st.rerun()
    
    else:
        snapshots = list_snapshot_files()
        if snapshots:
            snap_dates = [(parse_date_from_filename(p) or os.path.basename(p), p) for p in snapshots]
            snap_dates = [(d, p) for d, p in snap_dates if d]
            
            if snap_dates:
                selected_date = st.selectbox("Snapshot Date", [d for d, _ in snap_dates])
                selected_path = dict(snap_dates).get(selected_date)
            else:
                selected_path = None
        else:
            st.warning("No snapshots in data/phase5/")
            selected_path = None

# Load data
files = {}
analyses = []
picks_df = pd.DataFrame()
phase5_df = None
meta = {}

try:
    if mode == "Live (GitHub Artifact)":
        with st.spinner("Downloading latest artifact..."):
            zip_bytes, meta = github_latest_artifact_zip(CORE_OWNER, CORE_REPO, CORE_ARTIFACT_NAME)
            files = unzip_to_memory(zip_bytes)
        
        phase5_df = load_phase5_from_artifact(files)
        analyses = load_hybrid_analyses_from_artifact(files)
        picks_df = load_picks_from_artifact(files)
        
        # Info bar
        col1, col2, col3 = st.columns(3)
        col1.metric("Artifact", meta.get("name", "N/A"))
        col2.metric("Created", (meta.get("created_at") or "")[:16].replace("T", " "))
        col3.metric("Files", len(files))
    
    else:
        if selected_path:
            phase5_df = pd.read_parquet(selected_path)
            st.caption(f"Loaded snapshot: {selected_date}")
        else:
            st.warning("No snapshot selected or available.")
            st.info("Run the snapshot workflow to populate data/phase5/")
            st.stop()

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Get available dates
available_dates = get_available_dates(files) if files else []

st.divider()

# Summary metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Scan Dates", len(available_dates))
col2.metric("Hybrid Analyses", len(analyses))
col3.metric("Total Picks", len(picks_df) if not picks_df.empty else 0)
col4.metric("Unique Tickers", picks_df["ticker"].nunique() if not picks_df.empty else 0)

# Check if we have Phase-5 data
has_phase5 = phase5_df is not None and not phase5_df.empty

# Tabs
if has_phase5:
    tabs = st.tabs(["📈 Overview", "🧠 Phase-5 Learning", "🎯 Picks Explorer", "📁 Raw Files"])
else:
    tabs = st.tabs(["📈 Overview", "🎯 Picks Explorer", "📁 Raw Files"])

# Overview tab
with tabs[0]:
    st.subheader("System Overview")
    
    if analyses:
        chart_picks_over_time(analyses)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            chart_picks_by_source(picks_df)
        with col2:
            chart_unique_tickers_over_time(picks_df)
        
        st.divider()
        
        col3, col4 = st.columns(2)
        with col3:
            chart_top_tickers(picks_df)
        with col4:
            chart_confidence_distribution(picks_df)
    else:
        st.warning("No hybrid analyses found in artifact.")

# Phase-5 Learning tab (only if data exists)
if has_phase5:
    with tabs[1]:
        st.subheader("Phase-5 Learning")
        st.info("Phase-5 data loaded. Learning charts will appear here.")
        st.dataframe(phase5_df.head(50), use_container_width=True)

# Picks Explorer tab
picks_tab_idx = 2 if has_phase5 else 1
with tabs[picks_tab_idx]:
    st.subheader("Picks Explorer")
    
    if available_dates:
        selected_explore_date = st.selectbox("Select Date", available_dates)
        
        # Find analysis for this date
        analysis = None
        for h in analyses:
            if h.get("date") == selected_explore_date or h.get("asof_trading_date") == selected_explore_date:
                analysis = h
                break
        
        if analysis:
            st.markdown(f"### {selected_explore_date}")
            
            # Hybrid Top 3
            hybrid_top3 = analysis.get("hybrid_top3", [])
            if hybrid_top3:
                st.markdown("#### Hybrid Top 3")
                for pick in hybrid_top3:
                    ticker = pick.get("ticker", "?")
                    score = pick.get("hybrid_score", 0)
                    sources = pick.get("sources", [])
                    confidence = pick.get("confidence", "")
                    
                    col1, col2, col3, col4 = st.columns([2, 2, 3, 2])
                    col1.markdown(f"**{ticker}**")
                    col2.markdown(f"Score: {score:.2f}" if score else "")
                    col3.markdown(f"Sources: {', '.join(sources)}")
                    col4.markdown(f"{confidence}")
            
            st.divider()
            
            # All picks
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Weekly Top 5**")
                for p in analysis.get("primary_top5", [])[:5]:
                    if isinstance(p, dict):
                        st.write(f"#{p.get('rank', '?')} {p.get('ticker', '?')}")
            
            with col2:
                st.markdown("**Pro30**")
                for t in analysis.get("pro30_tickers", [])[:10]:
                    st.write(f"- {t}")
            
            with col3:
                st.markdown("**Movers**")
                for t in analysis.get("movers_tickers", [])[:10]:
                    st.write(f"- {t}")
        else:
            st.info(f"No analysis found for {selected_explore_date}")
    else:
        st.warning("No dates available.")

# Raw Files tab
raw_tab_idx = 3 if has_phase5 else 2
with tabs[raw_tab_idx]:
    st.subheader("Raw Files in Artifact")
    
    if files:
        paths = sorted(files.keys())
        
        # Filter
        filter_text = st.text_input("Filter paths", "")
        if filter_text:
            paths = [p for p in paths if filter_text.lower() in p.lower()]
        
        st.write(f"Showing {len(paths)} files")
        st.dataframe(pd.DataFrame({"path": paths}), use_container_width=True, height=400)
    else:
        st.warning("No files loaded.")

# Footer
st.divider()
st.caption("Read-only dashboard | No writes, no model logic | Data from GitHub artifact")

# Phase-5 status message
if not has_phase5:
    st.info("""
    **Phase-5 Learning data not available yet.**
    
    Phase-5 data appears after:
    1. Picks are made (scans run daily)
    2. 7+ trading days pass (outcomes resolve)
    3. Run `python main.py learn resolve` to resolve outcomes
    4. Run `python main.py learn merge` to create phase5_merged.parquet
    
    Once Phase-5 data exists, you'll see hit rate trends, rank decay curves, and regime analysis.
    """)
