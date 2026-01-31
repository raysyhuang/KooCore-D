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
    r2 = requests.get(dl_url, headers=_gh_headers(), timeout=60)
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


@st.cache_data(ttl=60)
def load_snapshot(path: str) -> pd.DataFrame:
    """Load a historical snapshot parquet."""
    return pd.read_parquet(path)


def load_all_snapshots() -> pd.DataFrame:
    """Load and concatenate all historical snapshots."""
    files = list_snapshot_files()
    if not files:
        return pd.DataFrame()
    
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            date = parse_date_from_filename(f)
            if date and "snapshot_date" not in df.columns:
                df["snapshot_date"] = date
            dfs.append(df)
        except Exception:
            continue
    
    if not dfs:
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True)


# =============================================================================
# Data Preparation
# =============================================================================
def prepare_phase5(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Phase-5 DataFrame columns."""
    out = df.copy()
    
    # Date normalization
    for col in ["scan_date", "asof", "pick_date", "date"]:
        if col in out.columns:
            out["scan_date"] = pd.to_datetime(out[col], errors="coerce")
            break
    else:
        out["scan_date"] = pd.NaT
    
    # Ticker
    for col in ["ticker", "Ticker", "symbol"]:
        if col in out.columns:
            out["ticker"] = out[col].astype(str).str.upper().str.strip()
            break
    else:
        out["ticker"] = ""
    
    # Outcome normalization
    if "outcome_7d" in out.columns:
        out["outcome_7d"] = out["outcome_7d"].astype(str).str.lower()
    elif "hit_7d" in out.columns:
        out["outcome_7d"] = np.where(out["hit_7d"].astype(bool), "hit", "miss")
    elif "hit_7pct" in out.columns:
        out["outcome_7d"] = np.where(out["hit_7pct"].astype(bool), "hit", "miss")
    
    # Return
    if "return_7d" in out.columns:
        out["return_7d"] = pd.to_numeric(out["return_7d"], errors="coerce")
    
    # Regime
    if "regime" not in out.columns:
        out["regime"] = "unknown"
    out["regime"] = out["regime"].astype(str).fillna("unknown")
    
    # Ranks
    for col in ["hybrid_rank", "pro30_rank", "swing_rank", "rank"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    
    # Sources
    if "hybrid_sources" in out.columns:
        out["sources_str"] = out["hybrid_sources"].apply(
            lambda x: ",".join(x) if isinstance(x, list) else str(x) if x else ""
        )
    elif "source" in out.columns:
        out["sources_str"] = out["source"].astype(str)
    else:
        out["sources_str"] = ""
    
    return out


# =============================================================================
# Charts
# =============================================================================
def chart_hit_rate_over_time(df: pd.DataFrame):
    """Hit rate trend over scan dates."""
    d = df.dropna(subset=["scan_date"]).copy()
    if "outcome_7d" not in d.columns:
        st.info("No outcome data yet.")
        return
    
    d["is_hit"] = d["outcome_7d"].isin(["hit", "1", "true", "yes"])
    g = d.groupby(pd.Grouper(key="scan_date", freq="D"))["is_hit"].agg(["mean", "count"]).reset_index()
    g.columns = ["scan_date", "hit_rate", "count"]
    g = g.dropna()
    
    if g.empty:
        st.info("No resolved outcomes to display.")
        return
    
    fig = px.line(g, x="scan_date", y="hit_rate", markers=True,
                  title="Hit Rate Over Time (7D)", hover_data=["count"])
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
    st.plotly_chart(fig, use_container_width=True)


def chart_avg_return_over_time(df: pd.DataFrame):
    """Average return trend over scan dates."""
    d = df.dropna(subset=["scan_date", "return_7d"]).copy()
    if d.empty:
        st.info("No return data yet.")
        return
    
    g = d.groupby(pd.Grouper(key="scan_date", freq="D"))["return_7d"].agg(["mean", "count"]).reset_index()
    g.columns = ["scan_date", "avg_return", "count"]
    g = g.dropna()
    
    fig = px.line(g, x="scan_date", y="avg_return", markers=True,
                  title="Avg 7D Return Over Time", hover_data=["count"])
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    st.plotly_chart(fig, use_container_width=True)


def chart_hit_rate_by_regime(df: pd.DataFrame):
    """Hit rate breakdown by market regime."""
    d = df.copy()
    if "outcome_7d" not in d.columns:
        st.info("No outcome data yet.")
        return
    
    d["is_hit"] = d["outcome_7d"].isin(["hit", "1", "true", "yes"])
    g = d.groupby("regime")["is_hit"].agg(["mean", "count"]).reset_index()
    g.columns = ["regime", "hit_rate", "count"]
    g = g.sort_values("hit_rate", ascending=False)
    
    colors = {"bull": "#00CC44", "bear": "#FF4444", "chop": "#FF8800", "neutral": "#888888"}
    
    fig = px.bar(g, x="regime", y="hit_rate", title="Hit Rate by Regime",
                 hover_data=["count"], color="regime",
                 color_discrete_map={r: colors.get(r.lower(), "#888") for r in g["regime"]})
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
    st.plotly_chart(fig, use_container_width=True)


def chart_signal_attribution(df: pd.DataFrame):
    """Hit rate by signal bucket (Weekly, Pro30, Overlap, etc.)."""
    d = df.copy()
    if "outcome_7d" not in d.columns:
        st.info("No outcome data yet.")
        return
    
    d["is_hit"] = d["outcome_7d"].isin(["hit", "1", "true", "yes"])
    
    buckets = []
    
    # Check various signal flags
    flag_cols = [
        ("Hybrid Top3", "in_hybrid_top3"),
        ("Conviction", "in_conviction_picks"),
        ("Pro30", "in_pro30"),
        ("Swing Top5", "in_swing_top5"),
        ("Weekly", "in_weekly"),
    ]
    
    for label, col in flag_cols:
        if col in d.columns:
            subset = d[d[col] == True]
            if len(subset) > 0:
                buckets.append({
                    "bucket": label,
                    "hit_rate": subset["is_hit"].mean(),
                    "avg_return": subset["return_7d"].mean() if "return_7d" in subset.columns else None,
                    "n": len(subset),
                })
    
    # Source-based attribution
    if "sources_str" in d.columns and not buckets:
        for src in ["Weekly", "Pro30", "Movers", "Swing"]:
            subset = d[d["sources_str"].str.contains(src, case=False, na=False)]
            if len(subset) > 0:
                buckets.append({
                    "bucket": src,
                    "hit_rate": subset["is_hit"].mean(),
                    "avg_return": subset["return_7d"].mean() if "return_7d" in subset.columns else None,
                    "n": len(subset),
                })
    
    if not buckets:
        st.info("No signal attribution data available.")
        return
    
    bdf = pd.DataFrame(buckets).sort_values("hit_rate", ascending=False)
    
    fig = px.bar(bdf, x="bucket", y="hit_rate", title="Hit Rate by Signal Source",
                 hover_data=["n", "avg_return"], text_auto=".0%")
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(bdf, use_container_width=True, hide_index=True)


def chart_rank_decay(df: pd.DataFrame, rank_col: str, title: str):
    """Rank decay curve - hit rate by rank bucket."""
    d = df.copy()
    if "outcome_7d" not in d.columns or rank_col not in d.columns:
        st.info(f"Missing {rank_col} or outcome_7d.")
        return
    
    d["is_hit"] = d["outcome_7d"].isin(["hit", "1", "true", "yes"])
    d[rank_col] = pd.to_numeric(d[rank_col], errors="coerce")
    d = d.dropna(subset=[rank_col])
    
    if d.empty:
        st.info(f"No {rank_col} data to display.")
        return
    
    # Create rank buckets
    bins = [0, 1, 3, 5, 10, 20, 50, 200]
    labels = ["1", "2-3", "4-5", "6-10", "11-20", "21-50", "51+"]
    d["rank_bin"] = pd.cut(d[rank_col], bins=bins, labels=labels, right=True)
    
    g = d.groupby("rank_bin", observed=True)["is_hit"].agg(["mean", "count"]).reset_index()
    g.columns = ["rank_bin", "hit_rate", "count"]
    
    fig = px.bar(g, x="rank_bin", y="hit_rate", title=title,
                 hover_data=["count"], text_auto=".0%")
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
    st.plotly_chart(fig, use_container_width=True)


def chart_return_distribution(df: pd.DataFrame):
    """Histogram of 7D returns."""
    d = df.dropna(subset=["return_7d"]).copy()
    if d.empty:
        st.info("No return data yet.")
        return
    
    fig = px.histogram(d, x="return_7d", nbins=40, title="Distribution of 7D Returns")
    fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.5)
    
    mean_ret = d["return_7d"].mean()
    fig.add_vline(x=mean_ret, line_dash="dash", line_color="blue",
                  annotation_text=f"Mean: {mean_ret:.1f}%")
    st.plotly_chart(fig, use_container_width=True)


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
            "weekly": summary.get("weekly_top5_count") or summary.get("primary_top5_count") or 0,
            "pro30": summary.get("pro30_candidates_count", 0),
            "movers": summary.get("movers_count", 0),
        })
    
    if not rows:
        st.info("No summary data.")
        return
    
    df = pd.DataFrame(rows).sort_values("date")
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["date"], y=df["weekly"], name="Weekly", marker_color="#4A90D9"))
    fig.add_trace(go.Bar(x=df["date"], y=df["pro30"], name="Pro30", marker_color="#7CB342"))
    fig.add_trace(go.Bar(x=df["date"], y=df["movers"], name="Movers", marker_color="#FF7043"))
    fig.update_layout(title="Daily Pick Counts", barmode="group")
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
            
            selected_date = st.selectbox(
                "Snapshot Date",
                [d for d, _ in snap_dates],
                help="Select a historical snapshot"
            )
            selected_path = dict(snap_dates).get(selected_date)
            
            load_all = st.checkbox("Load all snapshots", value=False,
                                   help="Combine all snapshots for trend analysis")
        else:
            st.warning("No snapshots in data/phase5/")
            selected_path = None
            load_all = False

# Load data
df = pd.DataFrame()
analyses = []
meta = {}

try:
    if mode == "Live (GitHub Artifact)":
        with st.spinner("Downloading latest artifact..."):
            zip_bytes, meta = github_latest_artifact_zip(CORE_OWNER, CORE_REPO, CORE_ARTIFACT_NAME)
            files = unzip_to_memory(zip_bytes)
        
        df = load_phase5_from_artifact(files)
        analyses = load_hybrid_analyses_from_artifact(files)
        
        # Info bar
        col1, col2, col3 = st.columns(3)
        col1.metric("Artifact", meta.get("name", "N/A"))
        col2.metric("Created", (meta.get("created_at") or "")[:16].replace("T", " "))
        col3.metric("Files", len(files))
    
    else:
        if load_all:
            df = load_all_snapshots()
            st.caption(f"Loaded {len(list_snapshot_files())} snapshots")
        elif selected_path:
            df = load_snapshot(selected_path)
            st.caption(f"Loaded snapshot: {selected_date}")
        else:
            st.warning("No snapshot selected.")
            st.stop()

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("""
    **Troubleshooting:**
    - For Live mode: Ensure GITHUB_TOKEN is set and artifact exists
    - For Historical mode: Run the snapshot workflow to populate data/phase5/
    """)
    st.stop()

# Prepare data
if df is not None and not df.empty:
    df = prepare_phase5(df)

# Summary metrics
if df is not None and not df.empty:
    st.divider()
    
    total_rows = len(df)
    unique_tickers = df["ticker"].nunique() if "ticker" in df.columns else 0
    unique_dates = df["scan_date"].nunique() if "scan_date" in df.columns else 0
    
    if "outcome_7d" in df.columns:
        resolved = df["outcome_7d"].isin(["hit", "miss", "1", "0", "true", "false"])
        resolved_count = resolved.sum()
        hits = df["outcome_7d"].isin(["hit", "1", "true"]).sum()
        hit_rate = hits / resolved_count if resolved_count > 0 else 0
    else:
        resolved_count = 0
        hit_rate = 0
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Rows", f"{total_rows:,}")
    c2.metric("Tickers", f"{unique_tickers:,}")
    c3.metric("Scan Dates", f"{unique_dates:,}")
    c4.metric("Resolved", f"{resolved_count:,}")
    c5.metric("Hit Rate", f"{hit_rate:.1%}" if resolved_count > 0 else "N/A")

# Tabs
tabs = st.tabs(["📈 Performance", "🎯 Attribution", "📉 Rank Decay", "🗂️ Raw Data"])

with tabs[0]:
    st.subheader("Performance Over Time")
    
    if df is None or df.empty:
        st.warning("No data loaded.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            chart_hit_rate_over_time(df)
        with col2:
            chart_avg_return_over_time(df)
        
        st.divider()
        
        col3, col4 = st.columns(2)
        with col3:
            chart_hit_rate_by_regime(df)
        with col4:
            chart_return_distribution(df)
        
        if analyses:
            st.divider()
            chart_picks_over_time(analyses)

with tabs[1]:
    st.subheader("Signal Attribution")
    
    if df is None or df.empty:
        st.warning("No data loaded.")
    else:
        chart_signal_attribution(df)

with tabs[2]:
    st.subheader("Rank Decay Analysis")
    st.caption("Higher ranks should have higher hit rates. Flat = weak signal.")
    
    if df is None or df.empty:
        st.warning("No data loaded.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            chart_rank_decay(df, "pro30_rank", "Pro30 Rank → Hit Rate")
        with col2:
            chart_rank_decay(df, "hybrid_rank", "Hybrid Rank → Hit Rate")

with tabs[3]:
    st.subheader("Raw Data")
    
    if df is None or df.empty:
        st.warning("No data loaded.")
    else:
        # Column filter
        all_cols = df.columns.tolist()
        selected_cols = st.multiselect("Columns", all_cols, default=all_cols[:15])
        
        if selected_cols:
            display_df = df[selected_cols].copy()
            
            # Sort
            if "scan_date" in display_df.columns:
                display_df = display_df.sort_values("scan_date", ascending=False)
            
            st.dataframe(display_df, use_container_width=True, height=500)
            
            # Download
            csv = display_df.to_csv(index=False)
            st.download_button("Download CSV", csv, "phase5_data.csv", "text/csv")
