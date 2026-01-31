"""
KooCore-D Dashboard (Option A + History Snapshots)

Features:
- Live Mode: Pull latest artifact from GitHub Actions
- Historical Mode: Load snapshots from data/phase5/*.parquet
- Equity Curve: Paper portfolio performance
- Model Version Selector: Phase-6 ready
- Regime Heatmap: Bull/Bear/Chop analysis

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


# =============================================================================
# GitHub API Helpers
# =============================================================================
def _gh_headers() -> Dict[str, str]:
    hdr = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if GITHUB_TOKEN:
        hdr["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return hdr


@st.cache_data(ttl=300, show_spinner=False)
def github_latest_artifact_zip(owner: str, repo: str, artifact_name: str) -> Tuple[bytes, Dict]:
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/artifacts?per_page=100"
    r = requests.get(url, headers=_gh_headers(), timeout=20)
    r.raise_for_status()
    
    arts = r.json().get("artifacts", [])
    candidates = [a for a in arts if a.get("name") == artifact_name and not a.get("expired", False)]
    
    if not candidates:
        raise RuntimeError(f"No artifact named '{artifact_name}' found.")
    
    candidates.sort(key=lambda a: a.get("created_at", ""), reverse=True)
    art = candidates[0]
    
    r2 = requests.get(art["archive_download_url"], headers=_gh_headers(), timeout=120)
    r2.raise_for_status()
    
    return r2.content, {"artifact_id": art.get("id"), "created_at": art.get("created_at"),
                        "size_bytes": art.get("size_in_bytes", 0), "name": art.get("name")}


def unzip_to_memory(zip_bytes: bytes) -> Dict[str, bytes]:
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
    for path in ["outputs/phase5/merged/phase5_merged.parquet", "phase5/merged/phase5_merged.parquet"]:
        if path in files:
            try:
                return pd.read_parquet(io.BytesIO(files[path]))
            except Exception:
                continue
    return None


def load_hybrid_analyses_from_artifact(files: Dict[str, bytes]) -> List[Dict]:
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
    analyses = load_hybrid_analyses_from_artifact(files)
    rows = []
    for h in analyses:
        date = h.get("date") or h.get("asof_trading_date")
        if not date:
            continue
        for pick in h.get("hybrid_top3", []):
            rows.append({"date": date, "ticker": pick.get("ticker", ""), "source": "hybrid_top3",
                        "rank": pick.get("rank"), "score": pick.get("hybrid_score"),
                        "confidence": pick.get("confidence"), "sources": ",".join(pick.get("sources", []))})
        for pick in h.get("primary_top5", []):
            rows.append({"date": date, "ticker": pick.get("ticker", ""), "source": "weekly",
                        "rank": pick.get("rank"), "score": pick.get("composite_score"),
                        "confidence": pick.get("confidence"), "sources": "Weekly"})
        for ticker in h.get("pro30_tickers", []):
            rows.append({"date": date, "ticker": ticker, "source": "pro30", "rank": None,
                        "score": None, "confidence": None, "sources": "Pro30"})
        for ticker in h.get("movers_tickers", []):
            rows.append({"date": date, "ticker": ticker, "source": "movers", "rank": None,
                        "score": None, "confidence": None, "sources": "Movers"})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def get_available_dates(files: Dict[str, bytes]) -> List[str]:
    dates = set()
    for path in files.keys():
        for part in path.split("/"):
            if len(part) == 10 and part[4] == "-" and part[7] == "-":
                try:
                    datetime.strptime(part, "%Y-%m-%d")
                    dates.add(part)
                except ValueError:
                    continue
    return sorted(dates, reverse=True)


def list_snapshot_files() -> List[str]:
    if not os.path.exists(PHASE5_DIR):
        return []
    return sorted(glob.glob(os.path.join(PHASE5_DIR, "phase5_merged_*.parquet")), reverse=True)


# =============================================================================
# Equity Curve Logic (Phase-6 Ready)
# =============================================================================
def build_equity_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Build paper portfolio equity curve from Phase-5 data."""
    if df.empty or "return_7d" not in df.columns or "scan_date" not in df.columns:
        return pd.DataFrame()
    
    trades = (
        df.dropna(subset=["return_7d"])
          .sort_values("scan_date")
          .groupby(["scan_date", "ticker"])
          .first()
          .reset_index()
    )
    
    if trades.empty:
        return pd.DataFrame()
    
    daily = (
        trades.groupby("scan_date")["return_7d"]
              .mean()
              .rename("daily_ret")
              .to_frame()
    )
    
    daily["equity"] = (1 + daily["daily_ret"] / 100).cumprod()
    daily["equity_norm"] = daily["equity"] / daily["equity"].iloc[0] * 100  # Start at 100
    daily["drawdown"] = daily["equity_norm"] / daily["equity_norm"].cummax() - 1
    
    return daily


def chart_equity_curve(equity_df: pd.DataFrame):
    """Plot equity curve with drawdown."""
    if equity_df.empty:
        st.info("No equity data yet. Waiting for Phase-5 resolved outcomes.")
        return
    
    fig = go.Figure()
    
    # Equity curve
    fig.add_trace(go.Scatter(
        x=equity_df.index,
        y=equity_df["equity_norm"],
        mode="lines",
        name="Portfolio",
        line=dict(color="#4A90D9", width=2),
        hovertemplate="Date: %{x}<br>Value: %{y:.1f}<extra></extra>"
    ))
    
    # Starting line
    fig.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title="📈 Paper Portfolio Equity Curve",
        xaxis_title="Date",
        yaxis_title="Portfolio Value (Started at 100)",
        template="plotly_white",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Drawdown chart
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=equity_df.index,
        y=equity_df["drawdown"] * 100,
        mode="lines",
        fill="tozeroy",
        name="Drawdown",
        line=dict(color="#FF4444"),
        fillcolor="rgba(255, 68, 68, 0.3)",
    ))
    fig2.update_layout(
        title="📉 Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template="plotly_white",
        height=250,
    )
    fig2.update_yaxes(tickformat=".1f")
    st.plotly_chart(fig2, use_container_width=True)


# =============================================================================
# Regime Heatmap (Phase-6 Ready)
# =============================================================================
def build_regime_heatmap(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Build regime × rank heatmap data."""
    if df.empty or "outcome_7d" not in df.columns or "regime" not in df.columns:
        return None
    
    # Create rank buckets
    rank_col = None
    for col in ["pro30_rank", "hybrid_rank", "rank"]:
        if col in df.columns:
            rank_col = col
            break
    
    if rank_col is None:
        return None
    
    df = df.copy()
    df[rank_col] = pd.to_numeric(df[rank_col], errors="coerce")
    
    def rank_bucket(r):
        if pd.isna(r):
            return "No Rank"
        r = int(r)
        if r <= 3:
            return "1-3"
        if r <= 5:
            return "4-5"
        if r <= 10:
            return "6-10"
        return "11+"
    
    df["rank_bucket"] = df[rank_col].apply(rank_bucket)
    
    # Determine hit
    df["is_hit"] = df["outcome_7d"].astype(str).str.lower().isin(["hit", "1", "true", "yes"])
    
    heat = (
        df.dropna(subset=["regime"])
          .groupby(["regime", "rank_bucket"])
          .agg(
              hit_rate=("is_hit", "mean"),
              avg_ret=("return_7d", "mean") if "return_7d" in df.columns else ("is_hit", "count"),
              n=("ticker", "count")
          )
          .reset_index()
    )
    
    return heat


def chart_regime_heatmap(heat_df: Optional[pd.DataFrame]):
    """Plot regime × rank heatmap."""
    if heat_df is None or heat_df.empty:
        st.info("No regime heatmap data yet. Waiting for Phase-5 resolved outcomes.")
        return
    
    # Pivot for heatmap
    pivot = heat_df.pivot(index="regime", columns="rank_bucket", values="hit_rate")
    
    # Reorder columns
    col_order = ["1-3", "4-5", "6-10", "11+", "No Rank"]
    pivot = pivot.reindex(columns=[c for c in col_order if c in pivot.columns])
    
    fig = px.imshow(
        pivot,
        text_auto=".0%",
        color_continuous_scale="RdYlGn",
        aspect="auto",
        title="🎯 Hit Rate: Regime × Rank"
    )
    fig.update_layout(
        xaxis_title="Rank Bucket",
        yaxis_title="Regime",
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Phase-5 Learning Charts
# =============================================================================
def chart_hit_rate_over_time(df: pd.DataFrame):
    if df.empty or "outcome_7d" not in df.columns or "scan_date" not in df.columns:
        st.info("No outcome data yet.")
        return
    
    d = df.dropna(subset=["scan_date"]).copy()
    d["is_hit"] = d["outcome_7d"].astype(str).str.lower().isin(["hit", "1", "true", "yes"])
    g = d.groupby(pd.Grouper(key="scan_date", freq="D"))["is_hit"].agg(["mean", "count"]).reset_index()
    g.columns = ["scan_date", "hit_rate", "count"]
    g = g.dropna()
    
    if g.empty:
        st.info("No resolved outcomes yet.")
        return
    
    fig = px.line(g, x="scan_date", y="hit_rate", markers=True, title="Hit Rate Over Time",
                  hover_data=["count"])
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
    st.plotly_chart(fig, use_container_width=True)


def chart_hit_rate_by_source(df: pd.DataFrame):
    if df.empty or "outcome_7d" not in df.columns:
        st.info("No outcome data yet.")
        return
    
    d = df.copy()
    d["is_hit"] = d["outcome_7d"].astype(str).str.lower().isin(["hit", "1", "true", "yes"])
    
    source_col = "source" if "source" in d.columns else "sources_str" if "sources_str" in d.columns else None
    if source_col is None:
        st.info("No source data.")
        return
    
    g = d.groupby(source_col)["is_hit"].agg(["mean", "count"]).reset_index()
    g.columns = [source_col, "hit_rate", "count"]
    g = g.sort_values("hit_rate", ascending=False)
    
    fig = px.bar(g, x=source_col, y="hit_rate", title="Hit Rate by Source", hover_data=["count"],
                 text_auto=".0%")
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
    st.plotly_chart(fig, use_container_width=True)


def chart_rank_decay(df: pd.DataFrame, rank_col: str, title: str):
    if df.empty or "outcome_7d" not in df.columns or rank_col not in df.columns:
        st.info(f"Missing data for {rank_col}.")
        return
    
    d = df.copy()
    d["is_hit"] = d["outcome_7d"].astype(str).str.lower().isin(["hit", "1", "true", "yes"])
    d[rank_col] = pd.to_numeric(d[rank_col], errors="coerce")
    d = d.dropna(subset=[rank_col])
    
    if d.empty:
        st.info(f"No {rank_col} data.")
        return
    
    def bucket(r):
        r = int(r)
        if r <= 3: return "1-3"
        if r <= 5: return "4-5"
        if r <= 10: return "6-10"
        return "11+"
    
    d["rank_bucket"] = d[rank_col].astype(int).map(bucket)
    order = ["1-3", "4-5", "6-10", "11+"]
    g = d.groupby("rank_bucket")["is_hit"].agg(["mean", "count"]).reindex(order).reset_index()
    g.columns = ["rank_bucket", "hit_rate", "count"]
    g = g.dropna()
    
    fig = px.bar(g, x="rank_bucket", y="hit_rate", title=title, hover_data=["count"], text_auto=".0%")
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Overview Charts
# =============================================================================
def chart_picks_over_time(analyses: List[Dict]):
    if not analyses:
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
        return
    
    df = pd.DataFrame(rows).sort_values("date")
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["date"], y=df["Weekly"], name="Weekly", marker_color="#4A90D9"))
    fig.add_trace(go.Bar(x=df["date"], y=df["Pro30"], name="Pro30", marker_color="#7CB342"))
    fig.add_trace(go.Bar(x=df["date"], y=df["Movers"], name="Movers", marker_color="#FF7043"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["Hybrid Top3"], name="Hybrid Top3",
                             mode="lines+markers", line=dict(color="#9C27B0", width=3)))
    fig.update_layout(title="Daily Pick Counts by Source", barmode="group")
    st.plotly_chart(fig, use_container_width=True)


def chart_top_tickers(df: pd.DataFrame, top_n: int = 15):
    if df.empty:
        return
    counts = df["ticker"].value_counts().head(top_n).reset_index()
    counts.columns = ["ticker", "count"]
    fig = px.bar(counts, x="ticker", y="count", title=f"Top {top_n} Most Picked Tickers",
                 color="count", color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Main App
# =============================================================================
st.set_page_config(page_title="KooCore Dashboard", page_icon="📊", layout="wide")

st.title("📊 KooCore-D Dashboard")

# Mode selection
mode = st.radio("Data Source", ["Live (GitHub Artifact)", "Historical Snapshots"], horizontal=True)

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    if mode == "Live (GitHub Artifact)":
        st.caption(f"Source: {CORE_OWNER}/{CORE_REPO}")
        if st.button("🔄 Refresh"):
            github_latest_artifact_zip.clear()
            st.rerun()
    else:
        snapshots = list_snapshot_files()
        if snapshots:
            selected_path = st.selectbox("Snapshot", snapshots,
                                         format_func=lambda x: os.path.basename(x))
        else:
            st.warning("No snapshots yet.")
            selected_path = None

# Load data
files = {}
analyses = []
picks_df = pd.DataFrame()
phase5_df = None
meta = {}

try:
    if mode == "Live (GitHub Artifact)":
        with st.spinner("Downloading artifact..."):
            zip_bytes, meta = github_latest_artifact_zip(CORE_OWNER, CORE_REPO, CORE_ARTIFACT_NAME)
            files = unzip_to_memory(zip_bytes)
        
        phase5_df = load_phase5_from_artifact(files)
        analyses = load_hybrid_analyses_from_artifact(files)
        picks_df = load_picks_from_artifact(files)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Artifact", meta.get("name", "N/A"))
        col2.metric("Created", (meta.get("created_at") or "")[:16].replace("T", " "))
        col3.metric("Files", len(files))
    else:
        if selected_path:
            phase5_df = pd.read_parquet(selected_path)
            st.caption(f"Loaded: {os.path.basename(selected_path)}")
        else:
            st.stop()
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# Normalize Phase-5 data
if phase5_df is not None and not phase5_df.empty:
    # Normalize columns
    for col in ["scan_date", "asof", "pick_date", "date"]:
        if col in phase5_df.columns:
            phase5_df["scan_date"] = pd.to_datetime(phase5_df[col], errors="coerce")
            break
    
    for col in ["ticker", "Ticker"]:
        if col in phase5_df.columns:
            phase5_df["ticker"] = phase5_df[col].astype(str).str.upper().str.strip()
            break
    
    if "regime" not in phase5_df.columns:
        phase5_df["regime"] = "unknown"

# Summary
available_dates = get_available_dates(files) if files else []
has_phase5 = phase5_df is not None and not phase5_df.empty

st.divider()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Scan Dates", len(available_dates))
col2.metric("Hybrid Analyses", len(analyses))
col3.metric("Total Picks", len(picks_df) if not picks_df.empty else 0)
col4.metric("Phase-5 Ready", "✅" if has_phase5 else "⏳")

# Model version selector (Phase-6 ready)
model_version = None
if has_phase5 and "model_version" in phase5_df.columns:
    versions = sorted(phase5_df["model_version"].dropna().unique().tolist())
    if versions:
        with st.sidebar:
            st.divider()
            model_version = st.selectbox("Model Version", versions, index=len(versions)-1)
            phase5_df = phase5_df[phase5_df["model_version"] == model_version]

# Tabs
if has_phase5:
    tabs = st.tabs(["📈 Overview", "💰 Equity Curve", "🧠 Learning", "🎯 Regime Analysis", "📁 Files"])
else:
    tabs = st.tabs(["📈 Overview", "🎯 Picks", "📁 Files"])

# Overview
with tabs[0]:
    st.subheader("System Overview")
    if analyses:
        chart_picks_over_time(analyses)
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if not picks_df.empty:
                chart_top_tickers(picks_df)
        with col2:
            if not picks_df.empty:
                counts = picks_df["source"].value_counts().reset_index()
                counts.columns = ["source", "count"]
                fig = px.pie(counts, values="count", names="source", title="Picks by Source")
                st.plotly_chart(fig, use_container_width=True)

if has_phase5:
    # Equity Curve Tab
    with tabs[1]:
        st.subheader("💰 Paper Portfolio Equity Curve")
        st.caption("Equal-weight portfolio, rebalanced daily based on resolved 7D returns")
        
        equity_df = build_equity_curve(phase5_df)
        chart_equity_curve(equity_df)
        
        if not equity_df.empty:
            col1, col2, col3 = st.columns(3)
            total_return = (equity_df["equity_norm"].iloc[-1] - 100)
            max_dd = equity_df["drawdown"].min() * 100
            sharpe = equity_df["daily_ret"].mean() / equity_df["daily_ret"].std() * np.sqrt(252) if equity_df["daily_ret"].std() > 0 else 0
            
            col1.metric("Total Return", f"{total_return:+.1f}%")
            col2.metric("Max Drawdown", f"{max_dd:.1f}%")
            col3.metric("Sharpe (ann.)", f"{sharpe:.2f}")
    
    # Learning Tab
    with tabs[2]:
        st.subheader("🧠 Phase-5 Learning")
        
        col1, col2 = st.columns(2)
        with col1:
            chart_hit_rate_over_time(phase5_df)
        with col2:
            chart_hit_rate_by_source(phase5_df)
        
        st.divider()
        st.subheader("Rank Decay Analysis")
        col3, col4 = st.columns(2)
        with col3:
            chart_rank_decay(phase5_df, "pro30_rank", "Pro30 Rank → Hit Rate")
        with col4:
            chart_rank_decay(phase5_df, "hybrid_rank", "Hybrid Rank → Hit Rate")
    
    # Regime Tab
    with tabs[3]:
        st.subheader("🎯 Regime Analysis")
        st.caption("Performance breakdown by market regime")
        
        heat_df = build_regime_heatmap(phase5_df)
        chart_regime_heatmap(heat_df)
        
        if heat_df is not None and not heat_df.empty:
            st.divider()
            st.dataframe(heat_df.sort_values(["regime", "rank_bucket"]), use_container_width=True)
    
    # Files Tab
    files_tab_idx = 4
else:
    # Picks Tab (when no Phase-5)
    with tabs[1]:
        st.subheader("🎯 Picks Explorer")
        if available_dates:
            selected = st.selectbox("Date", available_dates)
            for h in analyses:
                if h.get("date") == selected or h.get("asof_trading_date") == selected:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("**Hybrid Top 3**")
                        for p in h.get("hybrid_top3", []):
                            st.write(f"• {p.get('ticker')} ({p.get('confidence', '')})")
                    with col2:
                        st.markdown("**Weekly**")
                        for p in h.get("primary_top5", [])[:5]:
                            st.write(f"• #{p.get('rank')} {p.get('ticker')}")
                    with col3:
                        st.markdown("**Pro30**")
                        for t in h.get("pro30_tickers", [])[:10]:
                            st.write(f"• {t}")
                    break
    files_tab_idx = 2

# Files Tab
with tabs[files_tab_idx]:
    st.subheader("📁 Raw Files")
    if files:
        filter_text = st.text_input("Filter", "")
        paths = sorted([p for p in files.keys() if filter_text.lower() in p.lower()] if filter_text else files.keys())
        st.write(f"{len(paths)} files")
        st.dataframe(pd.DataFrame({"path": paths}), height=400)

# Phase-5 status
if not has_phase5:
    st.divider()
    st.info("""
    **Phase-5 Learning data not available yet.**
    
    Once available, you'll see:
    - 💰 Equity Curve (paper portfolio performance)
    - 🧠 Hit rate trends & rank decay
    - 🎯 Regime × Rank heatmap
    
    To generate Phase-5 data, run:
    ```bash
    python main.py learn resolve
    python main.py learn merge
    ```
    """)

st.divider()
st.caption("Read-only dashboard | No writes, no model logic | Phase-6 ready")
