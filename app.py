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
from plotly.subplots import make_subplots
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed


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
# Notebook Tracker Data (from Notebook_tracker_2026-01.ipynb)
# =============================================================================
NOTEBOOK_TRACKER_DATA = {
    "2026-01-01": {
        "weekly": [],
        "pro30": ["PATH"],
        "movers": ["CVX"],
    },
    "2026-01-02": {
        "weekly": ["DBRG", "RKLB", "SATS", "HII", "HWM"],
        "pro30": [],
        "movers": ["CVX"],
    },
    "2026-01-03": {
        "weekly": ["HAL", "AXSM", "DBRG", "DOCN", "EL"],
        "pro30": [],
        "movers": [],
    },
    "2026-01-05": {
        "weekly": ["LITE", "BK", "DAL", "INCY", "PNC"],
        "pro30": ["SNDK", "WDC", "CENX"],
        "movers": [],
    },
    "2026-01-06": {
        "weekly": ["PSX", "ABBV", "AEIS", "GH", "VSEC"],
        "pro30": ["MOD", "INTC"],
        "movers": [],
    },
    "2026-01-07": {
        "weekly": ["NOC", "F", "BBIO", "COF", "BFH"],
        "pro30": ["HOUS"],
        "movers": [],
    },
    "2026-01-08": {
        "weekly": ["BK", "DAL", "BBIO", "DG", "CBOE"],
        "pro30": ["LQDA", "CGON", "GLUE", "APLD", "HOUS", "KTOS", "COMP", "INTC"],
        "movers": [],
    },
    "2026-01-09": {
        "weekly": ["GD", "AKAM", "EXPE", "CADE", "NOC"],
        "pro30": ["LQDA", "SLNO"],
        "movers": [],
    },
    "2026-01-12": {
        "weekly": ["SATS", "OPCH", "BBIO", "FORM", "RKLB"],
        "pro30": ["TVTX", "TTMI", "ATEC", "HOUS", "FIG", "FORM", "STNG", "TGTX"],
        "movers": [],
    },
    "2026-01-13": {
        "weekly": ["AKAM", "MRNA", "INTC", "RVTY", "ON"],
        "pro30": ["APP"],
        "movers": [],
    },
    "2026-01-14": {
        "weekly": ["MS", "CADE", "DAL"],
        "pro30": ["OCUL", "COHR", "HOUS", "SEI", "ATEC", "BRZE", "DVN"],
        "movers": ["CRM", "FIG", "IRTC", "PATH", "STNG", "TVTX", "WGS"],
    },
    "2026-01-15": {
        "weekly": ["CLSK", "NUE", "MS"],
        "pro30": ["OCUL", "COHR", "HOUS", "SEI", "ATEC", "BRZE"],
        "movers": ["AEIS", "AFCG", "APP", "ATEC", "CRCL", "CRM", "DAVE", "FIG", "GKOS", "IRON", "IRTC", "LZ", "PATH", "PI", "PXED", "STNG", "TVTX", "UEC", "VIA", "WGS"],
    },
    "2026-01-16": {
        "weekly": [],
        "pro30": [],
        "movers": [],
    },
}


def get_notebook_tracker_dates() -> List[str]:
    """Get list of dates from notebook tracker data."""
    return sorted([d for d in NOTEBOOK_TRACKER_DATA.keys() if any(NOTEBOOK_TRACKER_DATA[d].values())], reverse=True)


def get_notebook_tracker_tickers(date: str, sources: List[str]) -> List[str]:
    """Get deduplicated tickers for a date and selected sources."""
    if date not in NOTEBOOK_TRACKER_DATA:
        return []
    data = NOTEBOOK_TRACKER_DATA[date]
    tickers = []
    for src in sources:
        tickers.extend(data.get(src, []))
    # Deduplicate while preserving order
    seen = set()
    result = []
    for t in tickers:
        if t not in seen:
            result.append(t)
            seen.add(t)
    return result


# =============================================================================
# Stock Tracker Functions
# =============================================================================
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")


def _unique_tickers(tickers: List[str]) -> List[str]:
    """Deduplicate tickers while preserving order."""
    seen = set()
    out = []
    for t in tickers:
        t = str(t).strip().upper()
        if t and t not in seen:
            out.append(t)
            seen.add(t)
    return out


def _fetch_polygon_daily(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily OHLCV from Polygon for a single ticker."""
    if not POLYGON_API_KEY:
        return pd.DataFrame()
    try:
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
        resp = requests.get(
            url,
            params={"adjusted": "true", "sort": "asc", "limit": 5000, "apiKey": POLYGON_API_KEY},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json().get("results") or []
        if not results:
            return pd.DataFrame()
        df = pd.DataFrame(results)
        if df.empty or not {"o", "h", "l", "c", "v", "t"}.issubset(df.columns):
            return pd.DataFrame()
        df["Date"] = pd.to_datetime(df["t"], unit="ms")
        df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
        df = df.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_prices(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch close prices for multiple tickers."""
    tickers = _unique_tickers(tickers)
    if not tickers:
        return pd.DataFrame()
    
    close_dfs = []
    failed = []
    
    # Try Polygon first if available
    if POLYGON_API_KEY:
        def worker(t):
            return t, _fetch_polygon_daily(t, start_date, end_date)
        
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {ex.submit(worker, t): t for t in tickers}
            for fut in as_completed(futures):
                t, df = fut.result()
                if not df.empty and "Close" in df.columns:
                    close_dfs.append(df[["Close"]].rename(columns={"Close": t}))
                else:
                    failed.append(t)
    else:
        failed = tickers
    
    # Fallback to Yahoo Finance for failed tickers
    if failed:
        try:
            yf_data = yf.download(
                tickers=failed,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False,
                threads=True,
                group_by="column",
            )
            if not yf_data.empty:
                if isinstance(yf_data.columns, pd.MultiIndex):
                    yf_close = yf_data["Close"].copy()
                else:
                    yf_close = yf_data[["Close"]].copy()
                    yf_close.columns = [failed[0]]
                for t in failed:
                    if t in yf_close.columns:
                        close_dfs.append(yf_close[[t]])
        except Exception:
            pass
    
    if not close_dfs:
        return pd.DataFrame()
    
    close = pd.concat(close_dfs, axis=1).sort_index().ffill()
    return close


def compute_performance(close: pd.DataFrame, baseline_date: str, end_date: str) -> pd.DataFrame:
    """Compute performance metrics for each ticker."""
    if close.empty:
        return pd.DataFrame()
    
    start_dt = pd.to_datetime(baseline_date)
    end_dt = pd.to_datetime(end_date)
    
    w = close.loc[(close.index >= start_dt) & (close.index <= end_dt)].copy()
    w = w.dropna(how="all")
    
    if w.empty:
        return pd.DataFrame()
    
    rows = []
    for t in close.columns:
        if t not in w.columns:
            continue
        s = w[t].dropna()
        if s.empty or len(s) < 1:
            continue
        
        baseline = float(s.iloc[0])
        current = float(s.iloc[-1])
        if baseline == 0:
            continue
        
        change = current - baseline
        pct = (change / baseline) * 100.0
        
        rows.append({
            "Symbol": t,
            "Baseline": round(baseline, 2),
            "Current": round(current, 2),
            "Change": round(change, 2),
            "Percent_Change": pct,
            "Baseline_Date": s.index[0].strftime("%Y-%m-%d"),
            "End_Date": s.index[-1].strftime("%Y-%m-%d"),
        })
    
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    
    return df.sort_values("Percent_Change", ascending=False).reset_index(drop=True)


def chart_stock_performance(df_perf: pd.DataFrame, close: pd.DataFrame, baseline_date: str, end_date: str):
    """Create performance charts: bar + cumulative returns."""
    if df_perf.empty:
        st.warning("No performance data to display.")
        return
    
    # Bar chart
    colors = np.where(df_perf["Percent_Change"] >= 0, "#00CC44", "#FF4444")
    
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=df_perf["Symbol"],
        y=df_perf["Percent_Change"],
        marker_color=colors,
        text=[f"{x:.1f}%" for x in df_perf["Percent_Change"]],
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Entry: $%{customdata[0]:.2f}<br>"
            "Current: $%{customdata[1]:.2f}<br>"
            "Return: %{y:.2f}%<extra></extra>"
        ),
        customdata=list(zip(df_perf["Baseline"], df_perf["Current"])),
    ))
    
    avg_change = float(df_perf["Percent_Change"].mean())
    fig_bar.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.4)
    fig_bar.add_hline(
        y=avg_change,
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Avg: {avg_change:.2f}%",
        annotation_position="bottom right",
    )
    
    fig_bar.update_layout(
        title=f"Total Return: {baseline_date} → {end_date}",
        xaxis_title="Ticker",
        yaxis_title="Return (%)",
        template="plotly_white",
        height=400,
    )
    fig_bar.update_xaxes(tickangle=45)
    
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Cumulative returns chart
    if not close.empty:
        start_dt = pd.to_datetime(baseline_date)
        end_dt = pd.to_datetime(end_date)
        w = close.loc[(close.index >= start_dt) & (close.index <= end_dt)].copy()
        w = w.dropna(how="all")
        
        if not w.empty:
            # Calculate cumulative returns
            anchors = {}
            for t in w.columns:
                s = w[t].dropna()
                if not s.empty:
                    anchors[t] = float(s.iloc[0])
            
            if anchors:
                anchor_series = pd.Series(anchors)
                cum_returns = (w.divide(anchor_series, axis="columns") - 1.0) * 100.0
                cum_returns = cum_returns.dropna(how="all")
                
                fig_cum = go.Figure()
                
                for t in cum_returns.columns:
                    s = cum_returns[t].dropna()
                    if s.empty:
                        continue
                    fig_cum.add_trace(go.Scatter(
                        x=s.index,
                        y=s.values,
                        mode="lines",
                        name=t,
                        hovertemplate=f"<b>{t}</b><br>Date: %{{x|%Y-%m-%d}}<br>Return: %{{y:.2f}}%<extra></extra>",
                    ))
                
                # Portfolio average
                avg_path = cum_returns.mean(axis=1, skipna=True)
                if not avg_path.empty:
                    fig_cum.add_trace(go.Scatter(
                        x=avg_path.index,
                        y=avg_path.values,
                        mode="lines",
                        name="Portfolio Avg",
                        line=dict(color="purple", width=3, dash="dashdot"),
                    ))
                
                fig_cum.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.4)
                
                fig_cum.update_layout(
                    title="Cumulative Returns Over Time",
                    xaxis_title="Date",
                    yaxis_title="Cumulative Return (%)",
                    template="plotly_white",
                    height=450,
                    legend_title_text="Tickers",
                )
                
                st.plotly_chart(fig_cum, use_container_width=True)


def extract_tickers_from_analyses(analyses: List[Dict], selected_date: str) -> Dict[str, List[str]]:
    """Extract tickers from hybrid analysis for a specific date."""
    for h in analyses:
        date = h.get("date") or h.get("asof_trading_date")
        if date == selected_date:
            return {
                "weekly": [t.get("ticker", t) if isinstance(t, dict) else t 
                          for t in h.get("primary_top5", h.get("weekly_top5", []))],
                "pro30": h.get("pro30_tickers", []),
                "movers": h.get("movers_tickers", []),
                "hybrid_top3": [t.get("ticker", t) if isinstance(t, dict) else t 
                               for t in h.get("hybrid_top3", [])],
            }
    return {"weekly": [], "pro30": [], "movers": [], "hybrid_top3": []}


def render_stock_tracker_tab(analyses: List[Dict]):
    """Render the Stock Tracker tab content."""
    st.subheader("Stock Tracker")
    st.caption("Track stock performance from pick dates to current prices.")
    
    # Get available dates from analyses
    available_dates = []
    for h in analyses:
        date = h.get("date") or h.get("asof_trading_date")
        if date:
            available_dates.append(date)
    available_dates = sorted(set(available_dates), reverse=True)
    
    # Input mode selection
    input_mode = st.radio(
        "Ticker Source",
        ["From Scan Date", "Custom Tickers"],
        horizontal=True,
    )
    
    col1, col2 = st.columns(2)
    
    if input_mode == "From Scan Date":
        with col1:
            if available_dates:
                selected_scan_date = st.selectbox(
                    "Select Scan Date (Baseline)",
                    available_dates,
                    help="Picks from this date will be tracked"
                )
            else:
                st.warning("No scan dates available.")
                return
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                help="Track performance until this date"
            ).strftime("%Y-%m-%d")
        
        # Extract tickers for selected date
        date_tickers = extract_tickers_from_analyses(analyses, selected_scan_date)
        
        # Source selection
        st.markdown("**Select Sources:**")
        src_cols = st.columns(4)
        with src_cols[0]:
            use_weekly = st.checkbox("Weekly Top5", value=True, disabled=not date_tickers["weekly"])
            st.caption(f"{len(date_tickers['weekly'])} tickers")
        with src_cols[1]:
            use_pro30 = st.checkbox("Pro30", value=False, disabled=not date_tickers["pro30"])
            st.caption(f"{len(date_tickers['pro30'])} tickers")
        with src_cols[2]:
            use_movers = st.checkbox("Movers", value=False, disabled=not date_tickers["movers"])
            st.caption(f"{len(date_tickers['movers'])} tickers")
        with src_cols[3]:
            use_hybrid = st.checkbox("Hybrid Top3", value=True, disabled=not date_tickers["hybrid_top3"])
            st.caption(f"{len(date_tickers['hybrid_top3'])} tickers")
        
        # Build ticker list
        tickers = []
        if use_weekly:
            tickers.extend(date_tickers["weekly"])
        if use_pro30:
            tickers.extend(date_tickers["pro30"])
        if use_movers:
            tickers.extend(date_tickers["movers"])
        if use_hybrid:
            tickers.extend(date_tickers["hybrid_top3"])
        
        tickers = _unique_tickers(tickers)
        baseline_date = selected_scan_date
        
    else:  # Custom Tickers
        with col1:
            baseline_date = st.date_input(
                "Baseline Date",
                value=datetime.now() - pd.Timedelta(days=30),
                help="Entry date for tracking"
            ).strftime("%Y-%m-%d")
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                help="Track performance until this date"
            ).strftime("%Y-%m-%d")
        
        tickers_input = st.text_area(
            "Enter Tickers (comma, space, or newline separated)",
            placeholder="AAPL, MSFT, GOOGL, NVDA, TSLA",
            height=100,
        )
        
        # Parse tickers
        if tickers_input:
            tickers_input = tickers_input.replace("\n", ",").replace(";", ",").replace(" ", ",")
            tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
            tickers = _unique_tickers(tickers)
        else:
            tickers = []
    
    if not tickers:
        st.info("Select sources or enter tickers to track performance.")
        return
    
    st.markdown(f"**Tracking {len(tickers)} tickers:** {', '.join(tickers[:15])}{'...' if len(tickers) > 15 else ''}")
    
    # Fetch and display performance
    if st.button("Track Performance", type="primary"):
        with st.spinner(f"Fetching prices for {len(tickers)} tickers..."):
            close = fetch_stock_prices(tickers, baseline_date, end_date)
            df_perf = compute_performance(close, baseline_date, end_date)
        
        if df_perf.empty:
            st.warning("No price data available for the selected tickers and date range.")
            return
        
        # Summary metrics
        st.divider()
        winners = (df_perf["Percent_Change"] > 0).sum()
        losers = len(df_perf) - winners
        avg_return = df_perf["Percent_Change"].mean()
        
        m_cols = st.columns(5)
        m_cols[0].metric("Tickers", len(df_perf))
        m_cols[1].metric("Winners", winners, delta=f"{winners/len(df_perf):.0%}")
        m_cols[2].metric("Losers", losers)
        m_cols[3].metric("Avg Return", f"{avg_return:+.2f}%")
        m_cols[4].metric("Best", f"{df_perf.iloc[0]['Symbol']} ({df_perf.iloc[0]['Percent_Change']:+.1f}%)")
        
        st.divider()
        
        # Charts
        chart_stock_performance(df_perf, close, baseline_date, end_date)
        
        # Performance table
        st.subheader("Performance Details")
        
        df_display = df_perf.copy()
        df_display["Return"] = df_display["Percent_Change"].apply(lambda x: f"{x:+.2f}%")
        df_display["Status"] = df_display["Percent_Change"].apply(lambda x: "Winner" if x > 0 else "Loser")
        
        st.dataframe(
            df_display[["Symbol", "Status", "Baseline", "Current", "Change", "Return", "Baseline_Date", "End_Date"]],
            use_container_width=True,
            hide_index=True,
        )
        
        # Download
        csv = df_perf.to_csv(index=False)
        st.download_button("Download CSV", csv, f"stock_tracker_{baseline_date}_{end_date}.csv", "text/csv")


def render_notebook_tracker_tab():
    """Render the Notebook Tracker tab - data from Notebook_tracker_2026-01.ipynb."""
    st.subheader("Notebook Tracker (January 2026)")
    st.caption("Track daily stock picks from the January 2026 notebook tracker.")
    
    available_dates = get_notebook_tracker_dates()
    
    if not available_dates:
        st.warning("No tracker data available.")
        return
    
    # View mode selection
    view_mode = st.radio(
        "View Mode",
        ["Single Date", "All Dates Summary"],
        horizontal=True,
    )
    
    if view_mode == "Single Date":
        col1, col2 = st.columns(2)
        
        with col1:
            selected_date = st.selectbox(
                "Select Tracking Date (Baseline)",
                available_dates,
                help="Pick date from the notebook tracker",
                key="nb_select_date"
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                help="Track performance until this date",
                key="nb_end_date_single"
            ).strftime("%Y-%m-%d")
        
        # Get tickers for selected date
        date_data = NOTEBOOK_TRACKER_DATA.get(selected_date, {})
        
        # Source selection
        st.markdown("**Select Sources:**")
        src_cols = st.columns(3)
        with src_cols[0]:
            use_weekly = st.checkbox(
                "Weekly Top5", 
                value=bool(date_data.get("weekly")), 
                disabled=not date_data.get("weekly"),
                key="nb_weekly_single"
            )
            st.caption(f"{len(date_data.get('weekly', []))} tickers")
        with src_cols[1]:
            use_pro30 = st.checkbox(
                "Pro30", 
                value=bool(date_data.get("pro30")), 
                disabled=not date_data.get("pro30"),
                key="nb_pro30_single"
            )
            st.caption(f"{len(date_data.get('pro30', []))} tickers")
        with src_cols[2]:
            use_movers = st.checkbox(
                "Movers", 
                value=bool(date_data.get("movers")), 
                disabled=not date_data.get("movers"),
                key="nb_movers_single"
            )
            st.caption(f"{len(date_data.get('movers', []))} tickers")
        
        # Build source list
        sources = []
        if use_weekly:
            sources.append("weekly")
        if use_pro30:
            sources.append("pro30")
        if use_movers:
            sources.append("movers")
        
        tickers = get_notebook_tracker_tickers(selected_date, sources)
        
        if not tickers:
            st.info("No tickers selected. Enable at least one source.")
            return
        
        # Display ticker info
        st.markdown(f"**Tracking {len(tickers)} tickers from {selected_date}:**")
        
        ticker_info_cols = st.columns(3)
        if use_weekly and date_data.get("weekly"):
            with ticker_info_cols[0]:
                st.markdown(f"**Weekly:** {', '.join(date_data['weekly'])}")
        if use_pro30 and date_data.get("pro30"):
            with ticker_info_cols[1]:
                st.markdown(f"**Pro30:** {', '.join(date_data['pro30'])}")
        if use_movers and date_data.get("movers"):
            with ticker_info_cols[2]:
                st.markdown(f"**Movers:** {', '.join(date_data['movers'][:10])}{'...' if len(date_data['movers']) > 10 else ''}")
        
        # Track performance
        if st.button("Track Performance", type="primary", key="notebook_track_single"):
            with st.spinner(f"Fetching prices for {len(tickers)} tickers..."):
                close = fetch_stock_prices(tickers, selected_date, end_date)
                df_perf = compute_performance(close, selected_date, end_date)
            
            if df_perf.empty:
                st.warning("No price data available for the selected tickers and date range.")
                return
            
            # Summary metrics
            st.divider()
            winners = (df_perf["Percent_Change"] > 0).sum()
            losers = len(df_perf) - winners
            avg_return = df_perf["Percent_Change"].mean()
            
            m_cols = st.columns(5)
            m_cols[0].metric("Tickers", len(df_perf))
            m_cols[1].metric("Winners", winners, delta=f"{winners/len(df_perf):.0%}")
            m_cols[2].metric("Losers", losers)
            m_cols[3].metric("Avg Return", f"{avg_return:+.2f}%")
            if len(df_perf) > 0:
                m_cols[4].metric("Best", f"{df_perf.iloc[0]['Symbol']} ({df_perf.iloc[0]['Percent_Change']:+.1f}%)")
            
            st.divider()
            
            # Charts
            chart_stock_performance(df_perf, close, selected_date, end_date)
            
            # Performance table
            st.subheader("Performance Details")
            
            df_display = df_perf.copy()
            df_display["Return"] = df_display["Percent_Change"].apply(lambda x: f"{x:+.2f}%")
            df_display["Status"] = df_display["Percent_Change"].apply(lambda x: "Winner" if x > 0 else "Loser")
            
            st.dataframe(
                df_display[["Symbol", "Status", "Baseline", "Current", "Change", "Return", "Baseline_Date", "End_Date"]],
                use_container_width=True,
                hide_index=True,
            )
            
            # Download
            csv = df_perf.to_csv(index=False)
            st.download_button(
                "Download CSV", 
                csv, 
                f"notebook_tracker_{selected_date}_{end_date}.csv", 
                "text/csv",
                key="notebook_download_single"
            )
    
    else:  # All Dates Summary
        end_date = st.date_input(
            "End Date (Performance measured to this date)",
            value=datetime.now(),
            help="Track performance until this date",
            key="nb_end_date_all"
        ).strftime("%Y-%m-%d")
        
        # Source selection for all dates
        st.markdown("**Include Sources:**")
        src_cols = st.columns(3)
        with src_cols[0]:
            use_weekly = st.checkbox("Weekly Top5", value=True, key="nb_weekly_all")
        with src_cols[1]:
            use_pro30 = st.checkbox("Pro30", value=True, key="nb_pro30_all")
        with src_cols[2]:
            use_movers = st.checkbox("Movers", value=False, key="nb_movers_all")
        
        sources = []
        if use_weekly:
            sources.append("weekly")
        if use_pro30:
            sources.append("pro30")
        if use_movers:
            sources.append("movers")
        
        if not sources:
            st.info("Select at least one source.")
            return
        
        # Show summary of all dates
        st.markdown("### Available Dates")
        
        summary_rows = []
        for date in available_dates:
            tickers = get_notebook_tracker_tickers(date, sources)
            if tickers:
                data = NOTEBOOK_TRACKER_DATA[date]
                summary_rows.append({
                    "Date": date,
                    "Weekly": len(data.get("weekly", [])),
                    "Pro30": len(data.get("pro30", [])),
                    "Movers": len(data.get("movers", [])),
                    "Total": len(tickers),
                    "Tickers": ", ".join(tickers[:8]) + ("..." if len(tickers) > 8 else ""),
                })
        
        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
        
        if st.button("Calculate All Performance", type="primary", key="notebook_track_all"):
            all_perf = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, date in enumerate(available_dates):
                tickers = get_notebook_tracker_tickers(date, sources)
                if not tickers:
                    continue
                
                status_text.text(f"Processing {date}...")
                progress_bar.progress((i + 1) / len(available_dates))
                
                try:
                    close = fetch_stock_prices(tickers, date, end_date)
                    df_perf = compute_performance(close, date, end_date)
                    
                    if not df_perf.empty:
                        avg_return = df_perf["Percent_Change"].mean()
                        winners = (df_perf["Percent_Change"] > 0).sum()
                        best = df_perf.iloc[0]
                        worst = df_perf.iloc[-1]
                        
                        all_perf.append({
                            "Baseline Date": date,
                            "Tickers": len(df_perf),
                            "Winners": winners,
                            "Win Rate": f"{winners/len(df_perf):.0%}" if len(df_perf) > 0 else "N/A",
                            "Avg Return": avg_return,
                            "Best": f"{best['Symbol']} ({best['Percent_Change']:+.1f}%)",
                            "Worst": f"{worst['Symbol']} ({worst['Percent_Change']:+.1f}%)",
                        })
                except Exception as e:
                    st.warning(f"Error processing {date}: {e}")
                    continue
            
            progress_bar.empty()
            status_text.empty()
            
            if all_perf:
                st.divider()
                st.subheader("Performance Summary by Date")
                
                perf_df = pd.DataFrame(all_perf)
                
                # Overall metrics
                total_dates = len(perf_df)
                overall_avg = perf_df["Avg Return"].mean()
                best_date = perf_df.loc[perf_df["Avg Return"].idxmax()]
                worst_date = perf_df.loc[perf_df["Avg Return"].idxmin()]
                
                m_cols = st.columns(4)
                m_cols[0].metric("Dates Tracked", total_dates)
                m_cols[1].metric("Overall Avg Return", f"{overall_avg:+.2f}%")
                m_cols[2].metric("Best Date", f"{best_date['Baseline Date']} ({best_date['Avg Return']:+.1f}%)")
                m_cols[3].metric("Worst Date", f"{worst_date['Baseline Date']} ({worst_date['Avg Return']:+.1f}%)")
                
                st.divider()
                
                # Performance by date chart
                fig = px.bar(
                    perf_df, 
                    x="Baseline Date", 
                    y="Avg Return",
                    title="Average Return by Baseline Date",
                    color="Avg Return",
                    color_continuous_scale=["#FF4444", "#FFFF00", "#00CC44"],
                    color_continuous_midpoint=0,
                )
                fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
                fig.add_hline(y=overall_avg, line_dash="dash", line_color="blue", 
                             annotation_text=f"Avg: {overall_avg:.1f}%")
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed table
                st.subheader("Detailed Performance Table")
                display_df = perf_df.copy()
                display_df["Avg Return"] = display_df["Avg Return"].apply(lambda x: f"{x:+.2f}%")
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Download
                csv = perf_df.to_csv(index=False)
                st.download_button(
                    "Download Summary CSV", 
                    csv, 
                    f"notebook_tracker_summary_{end_date}.csv", 
                    "text/csv",
                    key="notebook_download_all"
                )
            else:
                st.warning("No performance data could be calculated.")


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
tabs = st.tabs(["📈 Performance", "🎯 Attribution", "📉 Rank Decay", "📊 Stock Tracker", "📓 Notebook Tracker", "🗂️ Raw Data"])

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
    render_stock_tracker_tab(analyses)

with tabs[4]:
    render_notebook_tracker_tab()

with tabs[5]:
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
