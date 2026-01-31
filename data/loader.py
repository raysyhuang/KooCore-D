"""
Data Loader Module

Read-only access to outputs/ directory.
No writes, no side effects.
"""

from pathlib import Path
import json
import pandas as pd
from typing import Optional
import os

# Determine outputs path (relative to dashboard or absolute)
def _get_outputs_path() -> Path:
    """Get path to outputs directory, handling both local and Heroku."""
    # Check environment variable first (for Heroku)
    env_path = os.getenv("KOOCORE_OUTPUTS_PATH")
    if env_path:
        return Path(env_path)
    
    # Local development: outputs/ is sibling to dashboard/
    dashboard_dir = Path(__file__).parent.parent
    project_root = dashboard_dir.parent
    return project_root / "outputs"


OUTPUTS = _get_outputs_path()


def available_dates() -> list[str]:
    """
    Return sorted list of available scan dates.
    Scans outputs/ for date-formatted directories (YYYY-MM-DD).
    """
    if not OUTPUTS.exists():
        return []
    
    dates = []
    for p in OUTPUTS.iterdir():
        if p.is_dir() and len(p.name) == 10 and p.name[4] == "-" and p.name[7] == "-":
            # Basic validation: looks like YYYY-MM-DD
            try:
                # Verify it's a valid date format
                parts = p.name.split("-")
                if len(parts) == 3 and all(part.isdigit() for part in parts):
                    dates.append(p.name)
            except (ValueError, IndexError):
                continue
    
    return sorted(dates, reverse=True)  # Most recent first


def load_components(date_str: str) -> dict:
    """
    Load components (weekly, pro30, movers) for a specific date.
    
    Returns dict with keys:
    - weekly: list of ticker strings
    - pro30: list of ticker strings
    - movers: list of ticker strings
    - weekly_details: list of dicts with full pick info
    - hybrid_top3: list of dicts with hybrid picks
    """
    run_dir = OUTPUTS / date_str
    
    weekly = []
    weekly_details = []
    pro30 = []
    movers = []
    hybrid_top3 = []
    
    if not run_dir.exists():
        return {
            "weekly": weekly,
            "pro30": pro30,
            "movers": movers,
            "weekly_details": weekly_details,
            "hybrid_top3": hybrid_top3,
        }
    
    # Weekly Top 5 (preferred source)
    p_top5 = run_dir / f"weekly_scanner_top5_{date_str}.json"
    if p_top5.exists():
        try:
            obj = json.loads(p_top5.read_text())
            for x in obj.get("top5", []):
                if isinstance(x, dict) and x.get("ticker"):
                    weekly.append(x["ticker"].strip().upper())
                    weekly_details.append(x)
        except (json.JSONDecodeError, KeyError):
            pass
    
    # Hybrid analysis (fallback for weekly + movers)
    p_hybrid = run_dir / f"hybrid_analysis_{date_str}.json"
    if p_hybrid.exists():
        try:
            h = json.loads(p_hybrid.read_text())
            
            # Weekly from hybrid if not found above
            if not weekly:
                for x in h.get("primary_top5", []):
                    if isinstance(x, dict) and x.get("ticker"):
                        weekly.append(x["ticker"].strip().upper())
                        weekly_details.append(x)
            
            # Movers
            movers = [t.strip().upper() for t in h.get("movers_tickers", []) if t]
            
            # Pro30
            pro30_raw = h.get("pro30_tickers", [])
            pro30 = [t.strip().upper() for t in pro30_raw if t]
            
            # Hybrid Top 3
            hybrid_top3 = h.get("hybrid_top3", [])
            
        except (json.JSONDecodeError, KeyError):
            pass
    
    # Pro30 from CSV (more complete list)
    p_pro30 = run_dir / f"30d_momentum_candidates_{date_str}.csv"
    if p_pro30.exists():
        try:
            df = pd.read_csv(p_pro30)
            if "Ticker" in df.columns:
                pro30 = [t.strip().upper() for t in df["Ticker"].dropna().tolist() if t]
        except Exception:
            pass
    
    # Deduplicate while preserving order
    def _dedup(lst):
        seen = set()
        out = []
        for x in lst:
            if x and x not in seen:
                out.append(x)
                seen.add(x)
        return out
    
    return {
        "weekly": _dedup(weekly),
        "pro30": _dedup(pro30),
        "movers": _dedup(movers),
        "weekly_details": weekly_details,
        "hybrid_top3": hybrid_top3,
    }


def load_run_metadata(date_str: str) -> Optional[dict]:
    """Load run metadata for a date."""
    p = OUTPUTS / date_str / "run_metadata.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, KeyError):
            return None
    return None


def load_phase5_merged() -> Optional[pd.DataFrame]:
    """
    Load Phase-5 merged parquet file.
    Returns None if not available.
    """
    # Check multiple possible locations
    possible_paths = [
        OUTPUTS / "phase5" / "merged" / "phase5_merged.parquet",
        OUTPUTS / "phase5" / "phase5_merged.parquet",
        OUTPUTS.parent / "data" / "phase5" / "phase5_merged.parquet",
    ]
    
    for p in possible_paths:
        if p.exists():
            try:
                return pd.read_parquet(p)
            except Exception:
                continue
    
    return None


def load_phase5_metrics() -> list[dict]:
    """
    Load Phase-5 metric JSON files.
    Returns list of metric dicts.
    """
    metrics_dir = OUTPUTS / "phase5" / "metrics"
    if not metrics_dir.exists():
        return []
    
    metrics = []
    for p in sorted(metrics_dir.glob("*.json")):
        try:
            metrics.append(json.loads(p.read_text()))
        except (json.JSONDecodeError, KeyError):
            continue
    
    return metrics


def load_performance_history() -> Optional[pd.DataFrame]:
    """Load historical performance data if available."""
    perf_dir = OUTPUTS / "performance"
    if not perf_dir.exists():
        return None
    
    # Try to load consolidated performance file
    for name in ["performance_history.csv", "portfolio_performance.csv", "returns_history.csv"]:
        p = perf_dir / name
        if p.exists():
            try:
                return pd.read_csv(p, parse_dates=["date"] if "date" in pd.read_csv(p, nrows=1).columns else None)
            except Exception:
                continue
    
    return None


def get_all_tickers_for_date(date_str: str) -> list[str]:
    """Get all unique tickers for a date (weekly + pro30 + movers)."""
    comp = load_components(date_str)
    
    seen = set()
    out = []
    for source in ["weekly", "pro30", "movers"]:
        for t in comp.get(source, []):
            if t and t not in seen:
                out.append(t)
                seen.add(t)
    
    return out
