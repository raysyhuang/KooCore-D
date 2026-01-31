"""
KooCore-D Dashboard (Option A: GitHub Artifact Pull)

Read-only observability dashboard that pulls artifacts from GitHub Actions.
Never runs scans - only reads versioned artifacts.
"""

from __future__ import annotations

import os
import io
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st


# =============================================================================
# Config
# =============================================================================
@dataclass(frozen=True)
class DashboardConfig:
    source_repo: str  # e.g. "raysyhuang/KooCore-D"
    artifact_name: str  # e.g. "koocore-outputs"
    github_token: str  # fine-grained token with Actions read
    branch: Optional[str] = None


def _cfg_from_env() -> DashboardConfig:
    repo = os.getenv("GITHUB_REPO", "").strip()
    name = os.getenv("ARTIFACT_NAME", "koocore-outputs").strip()
    tok = os.getenv("GITHUB_TOKEN", "").strip()
    br = os.getenv("GITHUB_BRANCH", "").strip() or None

    if not repo:
        raise RuntimeError("Missing env var GITHUB_REPO (e.g. raysyhuang/KooCore-D)")
    if not tok:
        raise RuntimeError("Missing env var GITHUB_TOKEN (GitHub token with Actions read)")

    return DashboardConfig(source_repo=repo, artifact_name=name, github_token=tok, branch=br)


# =============================================================================
# GitHub API helpers
# =============================================================================
_GH_API = "https://api.github.com"


def _gh_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "koocore-dashboard",
    }


@st.cache_data(ttl=120, show_spinner=False)
def fetch_latest_artifact_zip_bytes(
    repo: str,
    artifact_name: str,
    token: str,
    branch: Optional[str] = None,
) -> Tuple[bytes, Dict]:
    """
    Returns (zip_bytes, metadata).
    Finds the most recent successful artifact matching artifact_name.
    """
    url = f"{_GH_API}/repos/{repo}/actions/artifacts?per_page=100"
    r = requests.get(url, headers=_gh_headers(token), timeout=30)
    r.raise_for_status()
    data = r.json()

    arts = data.get("artifacts", []) or []
    if not arts:
        raise RuntimeError("No artifacts found in the source repo.")

    filtered = [
        a for a in arts
        if a.get("name") == artifact_name and not a.get("expired", False)
    ]

    if not filtered:
        raise RuntimeError(f"No artifacts found with name='{artifact_name}'. Check ARTIFACT_NAME.")

    filtered.sort(key=lambda a: a.get("created_at") or "", reverse=True)
    chosen = filtered[0]

    dl_url = chosen.get("archive_download_url")
    if not dl_url:
        raise RuntimeError("Artifact missing archive_download_url.")

    zr = requests.get(dl_url, headers=_gh_headers(token), timeout=60)
    zr.raise_for_status()

    meta = {
        "artifact_id": chosen.get("id"),
        "created_at": chosen.get("created_at"),
        "size_in_bytes": chosen.get("size_in_bytes"),
        "workflow_run_id": chosen.get("workflow_run", {}).get("id") if isinstance(chosen.get("workflow_run"), dict) else None,
        "name": chosen.get("name"),
    }
    return zr.content, meta


def unzip_to_memory(zip_bytes: bytes) -> Dict[str, bytes]:
    """Returns dict: {path_in_zip: file_bytes}"""
    out: Dict[str, bytes] = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            out[info.filename] = z.read(info.filename)
    return out


# =============================================================================
# Data loading
# =============================================================================
def _try_read_parquet(files: Dict[str, bytes], path: str) -> Optional[pd.DataFrame]:
    b = files.get(path)
    if not b:
        return None
    try:
        return pd.read_parquet(io.BytesIO(b))
    except Exception:
        return None


def _try_read_json(files: Dict[str, bytes], path: str) -> Optional[dict]:
    b = files.get(path)
    if not b:
        return None
    try:
        return json.loads(b.decode("utf-8"))
    except Exception:
        return None


def _try_read_csv(files: Dict[str, bytes], path: str) -> Optional[pd.DataFrame]:
    b = files.get(path)
    if not b:
        return None
    try:
        return pd.read_csv(io.BytesIO(b))
    except Exception:
        return None


def _read_jsonl(files: Dict[str, bytes], path: str) -> Optional[pd.DataFrame]:
    b = files.get(path)
    if not b:
        return None
    rows = []
    for line in b.decode("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    if not rows:
        return None
    return pd.DataFrame(rows)


def load_phase5_merged(files: Dict[str, bytes]) -> Optional[pd.DataFrame]:
    """Load Phase-5 merged parquet."""
    # Try multiple possible paths
    for path in [
        "outputs/phase5/merged/phase5_merged.parquet",
        "phase5/merged/phase5_merged.parquet",
        "outputs/phase5/phase5_merged.parquet",
    ]:
        df = _try_read_parquet(files, path)
        if df is not None and not df.empty:
            return df

    # Fallback: try JSONL rows
    candidates = [p for p in files.keys() if "phase5" in p and p.endswith(".jsonl")]
    if candidates:
        candidates.sort(reverse=True)
        df = _read_jsonl(files, candidates[0])
        if df is not None:
            return df

    return None


def load_scorecards(files: Dict[str, bytes]) -> List[dict]:
    """Load Phase-5 scorecard JSONs."""
    paths = [p for p in files.keys() if "phase5" in p and "scorecard" in p and p.endswith(".json")]
    out = []
    for p in sorted(paths):
        j = _try_read_json(files, p)
        if isinstance(j, dict):
            j["_path"] = p
            out.append(j)
    return out


def load_hybrid_analyses(files: Dict[str, bytes]) -> List[dict]:
    """Load hybrid analysis JSONs from outputs."""
    paths = [p for p in files.keys() if "hybrid_analysis" in p and p.endswith(".json")]
    out = []
    for p in sorted(paths, reverse=True):
        j = _try_read_json(files, p)
        if isinstance(j, dict):
            j["_path"] = p
            out.append(j)
    return out


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


# =============================================================================
# Data preparation
# =============================================================================
def prepare_phase5(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Phase-5 fields."""
    out = df.copy()

    # Date normalization
    for col in ["scan_date", "asof", "pick_date", "date"]:
        if col in out.columns:
            out["scan_date"] = pd.to_datetime(out[col], errors="coerce")
            break
    else:
        out["scan_date"] = pd.NaT

    # Ticker normalization
    for col in ["ticker", "Ticker", "symbol"]:
        if col in out.columns:
            out["ticker"] = out[col].astype(str).str.upper().str.strip()
            break
    else:
        out["ticker"] = ""

    # Outcome normalization
    if "return_7d" in out.columns:
        out["return_7d"] = pd.to_numeric(out["return_7d"], errors="coerce")

    if "outcome_7d" in out.columns:
        out["outcome_7d"] = out["outcome_7d"].astype(str)
    elif "hit_7d" in out.columns:
        out["outcome_7d"] = np.where(out["hit_7d"].astype(bool), "hit", "miss")
    elif "hit_7pct" in out.columns:
        out["outcome_7d"] = np.where(out["hit_7pct"].astype(bool), "hit", "miss")

    # Source normalization
    if "hybrid_sources" in out.columns:
        out["hybrid_sources_str"] = out["hybrid_sources"].apply(
            lambda x: ",".join(x) if isinstance(x, list) else str(x) if x else ""
        )
    elif "source" in out.columns:
        out["hybrid_sources_str"] = out["source"].astype(str)
    else:
        out["hybrid_sources_str"] = ""

    # Regime
    if "regime" not in out.columns:
        out["regime"] = "unknown"
    out["regime"] = out["regime"].astype(str).fillna("unknown")

    # Ranks
    for col in ["hybrid_rank", "pro30_rank", "rank"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Score
    if "hybrid_score" in out.columns:
        out["hybrid_score"] = pd.to_numeric(out["hybrid_score"], errors="coerce")

    return out


# =============================================================================
# Visualization
# =============================================================================
def chart_hit_rate_over_time(df: pd.DataFrame):
    d = df.dropna(subset=["scan_date"]).copy()
    if "outcome_7d" not in d.columns:
        st.info("No outcome data yet. Outcomes appear after picks resolve (7+ days).")
        return

    d["is_hit"] = d["outcome_7d"].astype(str).str.lower().isin(["hit", "1", "true", "yes"])
    g = d.groupby(pd.Grouper(key="scan_date", freq="D"))["is_hit"].mean().reset_index()
    g = g.dropna()

    if g.empty:
        st.info("No outcome data to display.")
        return

    fig = px.line(g, x="scan_date", y="is_hit", markers=True, title="Hit Rate Over Time (7d)")
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
    st.plotly_chart(fig, use_container_width=True)


def chart_hit_rate_by_regime(df: pd.DataFrame):
    d = df.copy()
    if "outcome_7d" not in d.columns:
        st.info("No outcome data yet.")
        return

    d["is_hit"] = d["outcome_7d"].astype(str).str.lower().isin(["hit", "1", "true", "yes"])
    g = d.groupby("regime")["is_hit"].agg(["mean", "count"]).reset_index()
    g = g.sort_values("mean", ascending=False)

    if g.empty:
        st.info("No regime data to display.")
        return

    colors = {"bull": "#00CC44", "bear": "#FF4444", "neutral": "#888888", "unknown": "#CCCCCC"}
    g["color"] = g["regime"].str.lower().map(lambda x: colors.get(x, "#888888"))

    fig = px.bar(g, x="regime", y="mean", hover_data=["count"], title="Hit Rate by Regime",
                 color="regime", color_discrete_map={r: colors.get(r.lower(), "#888") for r in g["regime"]})
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
    st.plotly_chart(fig, use_container_width=True)


def chart_hit_rate_by_source(df: pd.DataFrame):
    d = df.copy()
    if "outcome_7d" not in d.columns or "hybrid_sources_str" not in d.columns:
        st.info("No outcome or source data yet.")
        return

    d["is_hit"] = d["outcome_7d"].astype(str).str.lower().isin(["hit", "1", "true", "yes"])

    # Expand sources
    rows = []
    for _, row in d.iterrows():
        sources = str(row["hybrid_sources_str"]).split(",")
        for src in sources:
            src = src.strip()
            if src:
                rows.append({"source": src, "is_hit": row["is_hit"]})

    if not rows:
        st.info("No source data to display.")
        return

    src_df = pd.DataFrame(rows)
    g = src_df.groupby("source")["is_hit"].agg(["mean", "count"]).reset_index()
    g = g.sort_values("mean", ascending=False)

    fig = px.bar(g, x="source", y="mean", hover_data=["count"], title="Hit Rate by Source")
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
    st.plotly_chart(fig, use_container_width=True)


def chart_rank_decay(df: pd.DataFrame, rank_col: str, title: str):
    d = df.copy()
    if "outcome_7d" not in d.columns:
        st.info("No outcome data yet.")
        return
    if rank_col not in d.columns:
        st.info(f"Missing {rank_col} column.")
        return

    d["is_hit"] = d["outcome_7d"].astype(str).str.lower().isin(["hit", "1", "true", "yes"])
    d[rank_col] = pd.to_numeric(d[rank_col], errors="coerce")
    d = d.dropna(subset=[rank_col])

    if d.empty:
        st.info(f"No {rank_col} data to display.")
        return

    def bucket(r):
        r = int(r)
        if r <= 3:
            return "1-3"
        if r <= 5:
            return "4-5"
        if r <= 10:
            return "6-10"
        if r <= 20:
            return "11-20"
        return "21+"

    d["rank_bucket"] = d[rank_col].astype(int).map(bucket)
    order = ["1-3", "4-5", "6-10", "11-20", "21+"]
    g = d.groupby("rank_bucket")["is_hit"].agg(["mean", "count"]).reindex(order).reset_index()
    g = g.dropna()

    fig = px.bar(g, x="rank_bucket", y="mean", hover_data=["count"], title=title)
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
    st.plotly_chart(fig, use_container_width=True)


def chart_returns_distribution(df: pd.DataFrame):
    d = df.copy()
    if "return_7d" not in d.columns:
        st.info("No return_7d data yet.")
        return

    d = d.dropna(subset=["return_7d"])
    if d.empty:
        st.info("No return data to display.")
        return

    fig = px.histogram(d, x="return_7d", nbins=40, title="Distribution of 7-Day Returns")
    fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.5)

    mean_ret = d["return_7d"].mean()
    fig.add_vline(x=mean_ret, line_dash="dash", line_color="blue",
                  annotation_text=f"Mean: {mean_ret:.1f}%")

    st.plotly_chart(fig, use_container_width=True)


def chart_picks_summary(hybrid_analyses: List[dict]):
    """Show picks over time from hybrid analyses."""
    if not hybrid_analyses:
        st.info("No hybrid analysis data found.")
        return

    rows = []
    for h in hybrid_analyses:
        date = h.get("date") or h.get("asof_trading_date")
        if not date:
            continue

        summary = h.get("summary", {})
        rows.append({
            "date": date,
            "weekly": summary.get("weekly_top5_count") or summary.get("primary_top5_count") or 0,
            "pro30": summary.get("pro30_candidates_count", 0),
            "movers": summary.get("movers_count", 0),
            "hybrid_top3": summary.get("hybrid_top3_count", 0),
        })

    if not rows:
        st.info("No summary data to display.")
        return

    df = pd.DataFrame(rows).sort_values("date")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["date"], y=df["weekly"], name="Weekly", marker_color="#4A90D9"))
    fig.add_trace(go.Bar(x=df["date"], y=df["pro30"], name="Pro30", marker_color="#7CB342"))
    fig.add_trace(go.Bar(x=df["date"], y=df["movers"], name="Movers", marker_color="#FF7043"))

    fig.update_layout(
        title="Daily Pick Counts by Source",
        barmode="group",
        xaxis_title="Date",
        yaxis_title="Count",
    )
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Main App
# =============================================================================
st.set_page_config(page_title="KooCore Dashboard", page_icon="📊", layout="wide")

st.title("📊 KooCore-D Dashboard")
st.caption("Read-only observability | Option A: GitHub Artifact Pull")

# Sidebar config
with st.sidebar:
    st.header("Data Source")
    st.caption("Pulls latest artifact from GitHub Actions")

    try:
        cfg = _cfg_from_env()
        st.success(f"Connected to {cfg.source_repo}")
    except Exception as e:
        st.error(str(e))
        st.info("""
        **Required Environment Variables:**
        - `GITHUB_REPO`: e.g. `raysyhuang/KooCore-D`
        - `GITHUB_TOKEN`: GitHub token with Actions read
        - `ARTIFACT_NAME`: e.g. `koocore-outputs` (optional)
        """)
        st.stop()

    st.divider()

    if st.button("🔄 Refresh Data"):
        fetch_latest_artifact_zip_bytes.clear()
        st.rerun()

    st.markdown(f"""
    **Config**
    - Repo: `{cfg.source_repo}`
    - Artifact: `{cfg.artifact_name}`
    """)

# Fetch artifact
try:
    with st.spinner("Downloading latest artifact..."):
        zip_bytes, meta = fetch_latest_artifact_zip_bytes(
            repo=cfg.source_repo,
            artifact_name=cfg.artifact_name,
            token=cfg.github_token,
            branch=cfg.branch,
        )
    files = unzip_to_memory(zip_bytes)
except Exception as e:
    st.error(f"Failed to fetch artifact: {e}")
    st.info("""
    **Possible causes:**
    1. No artifacts uploaded yet - run `python main.py all` in KooCore-D with GitHub Actions
    2. Artifact name mismatch - check `ARTIFACT_NAME` matches workflow
    3. Token permissions - ensure token has `actions:read` scope
    """)
    st.stop()

# Artifact info
col1, col2, col3, col4 = st.columns(4)
col1.metric("Artifact", meta.get("name", "N/A"))
col2.metric("Created", (meta.get("created_at") or "")[:16].replace("T", " "))
col3.metric("Size", f"{(meta.get('size_in_bytes', 0) / 1024 / 1024):.1f} MB")
col4.metric("Files", len(files))

st.divider()

# Load data
df_phase5 = load_phase5_merged(files)
scorecards = load_scorecards(files)
hybrid_analyses = load_hybrid_analyses(files)
available_dates = get_available_dates(files)

# Tabs
tabs = st.tabs(["📈 Overview", "🧠 Phase-5 Learning", "🎯 Picks", "📋 Scorecards", "📁 Raw Files"])

with tabs[0]:
    st.subheader("Overview")

    if df_phase5 is not None and not df_phase5.empty:
        dfx = prepare_phase5(df_phase5)

        # Summary metrics
        total_picks = len(dfx)
        if "outcome_7d" in dfx.columns:
            resolved = dfx["outcome_7d"].astype(str).str.lower().isin(["hit", "miss", "1", "0", "true", "false"])
            resolved_count = resolved.sum()
            hits = dfx["outcome_7d"].astype(str).str.lower().isin(["hit", "1", "true"]).sum()
            hit_rate = hits / resolved_count if resolved_count > 0 else 0
        else:
            resolved_count = 0
            hit_rate = 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Picks", total_picks)
        c2.metric("Resolved", resolved_count)
        c3.metric("Hit Rate", f"{hit_rate:.1%}" if resolved_count > 0 else "N/A")
        c4.metric("Dates", len(available_dates))

        st.divider()
        chart_picks_summary(hybrid_analyses)

    else:
        st.warning("No Phase-5 data found in artifact.")
        st.info("Ensure your workflow uploads `outputs/phase5/merged/phase5_merged.parquet`")

        if hybrid_analyses:
            st.subheader("Available Hybrid Analyses")
            chart_picks_summary(hybrid_analyses)

with tabs[1]:
    st.subheader("Phase-5 Learning")

    if df_phase5 is None or df_phase5.empty:
        st.warning("No Phase-5 data available yet.")
        st.info("""
        Phase-5 data appears after:
        1. Picks are made (scan runs)
        2. 7+ trading days pass
        3. `python main.py learn resolve` runs
        4. `python main.py learn merge` runs
        """)
    else:
        dfx = prepare_phase5(df_phase5)

        c1, c2 = st.columns(2)
        with c1:
            chart_hit_rate_over_time(dfx)
        with c2:
            chart_hit_rate_by_regime(dfx)

        st.divider()

        c3, c4 = st.columns(2)
        with c3:
            chart_hit_rate_by_source(dfx)
        with c4:
            chart_returns_distribution(dfx)

        st.divider()
        st.subheader("Rank Decay Analysis")

        c5, c6 = st.columns(2)
        with c5:
            chart_rank_decay(dfx, "pro30_rank", "Pro30 Rank → Hit Rate")
        with c6:
            chart_rank_decay(dfx, "hybrid_rank", "Hybrid Rank → Hit Rate")

with tabs[2]:
    st.subheader("Picks Explorer")

    if not available_dates:
        st.warning("No scan dates found in artifact.")
    else:
        selected_date = st.selectbox("Select Date", available_dates)

        # Find hybrid analysis for this date
        analysis = None
        for h in hybrid_analyses:
            if h.get("date") == selected_date or h.get("asof_trading_date") == selected_date:
                analysis = h
                break

        if analysis:
            st.markdown(f"**Date:** {selected_date}")

            # Show hybrid top 3
            hybrid_top3 = analysis.get("hybrid_top3", [])
            if hybrid_top3:
                st.markdown("### Hybrid Top 3")
                for pick in hybrid_top3:
                    ticker = pick.get("ticker", "?")
                    score = pick.get("hybrid_score", 0)
                    sources = pick.get("sources", [])
                    confidence = pick.get("confidence", "")

                    st.markdown(f"**{ticker}** - Score: {score:.2f} | Sources: {', '.join(sources)} | {confidence}")

            # Show all picks
            st.markdown("### All Picks")

            weekly = analysis.get("primary_top5", [])
            pro30 = analysis.get("pro30_tickers", [])
            movers = analysis.get("movers_tickers", [])

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Weekly**")
                for p in weekly[:5]:
                    if isinstance(p, dict):
                        st.write(f"- {p.get('ticker', '?')} (#{p.get('rank', '?')})")
                    else:
                        st.write(f"- {p}")

            with c2:
                st.markdown("**Pro30**")
                for t in pro30[:10]:
                    st.write(f"- {t}")

            with c3:
                st.markdown("**Movers**")
                for t in movers[:10]:
                    st.write(f"- {t}")
        else:
            st.info(f"No hybrid analysis found for {selected_date}")

with tabs[3]:
    st.subheader("Phase-5 Scorecards")

    if not scorecards:
        st.info("No scorecards found. Run `python main.py learn analyze` to generate them.")
    else:
        for sc in scorecards[-5:][::-1]:
            path = sc.get("_path", "scorecard")
            with st.expander(path.split("/")[-1]):
                st.json(sc)

with tabs[4]:
    st.subheader("Raw Files in Artifact")

    paths = sorted(files.keys())
    st.write(f"Total files: {len(paths)}")

    # Filter
    filter_text = st.text_input("Filter paths", "")
    if filter_text:
        paths = [p for p in paths if filter_text.lower() in p.lower()]

    st.dataframe(pd.DataFrame({"path": paths}), use_container_width=True, height=400)
