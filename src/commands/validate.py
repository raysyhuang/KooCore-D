"""
Model Validation Command - Multi-period backtesting and improvement suggestions.

Usage:
    python main.py validate                    # Full validation with all periods
    python main.py validate --quick            # Quick 14-day check
    python main.py validate --start 2025-12-01 # Custom date range
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.utils.time import utc_now

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
OUTPUTS_ROOT = Path("outputs")


def _dedup_keep_order(items) -> list[str]:
    out, seen = [], set()
    for x in items:
        t = str(x).strip().upper()
        if t and t not in seen:
            out.append(t)
            seen.add(t)
    return out


def _safe_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@dataclass
class DatePicks:
    date_str: str
    swing_top5: list[str]
    weekly_top5: list[str]
    pro30: list[str]
    movers: list[str]
    combined: list[str]
    metadata: dict


def iter_output_dates(root: Path = OUTPUTS_ROOT) -> list[str]:
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir() and _DATE_RE.match(p.name)])


def load_picks_for_date(date_str: str, root: Path = OUTPUTS_ROOT) -> DatePicks:
    run_dir = root / date_str
    metadata = {}
    
    # Swing Top 5 (preferred)
    swing = []
    swing_top5_json = run_dir / f"swing_top5_{date_str}.json"
    if swing_top5_json.exists():
        obj = _safe_json(swing_top5_json)
        for x in obj.get("top5", []):
            if isinstance(x, dict) and x.get("ticker"):
                ticker = str(x["ticker"]).strip().upper()
                swing.append(ticker)
                metadata[ticker] = {
                    "source": "swing_top5",
                    "rank": len(swing),
                    "composite_score": x.get("composite_score"),
                    "technical_score": x.get("technical_score"),
                    "catalyst_score": x.get("catalyst_score"),
                }
    swing = _dedup_keep_order(swing)

    # Weekly Top 5 (secondary)
    weekly = []
    top5_json = run_dir / f"weekly_scanner_top5_{date_str}.json"
    if top5_json.exists():
        obj = _safe_json(top5_json)
        for x in obj.get("top5", []):
            if isinstance(x, dict) and x.get("ticker"):
                ticker = str(x["ticker"]).strip().upper()
                weekly.append(ticker)
                metadata.setdefault(ticker, {
                    "source": "weekly_top5",
                    "rank": len(weekly),
                    "composite_score": x.get("composite_score"),
                    "technical_score": x.get("technical_score"),
                    "catalyst_score": x.get("catalyst_score"),
                })
    else:
        hybrid = _safe_json(run_dir / f"hybrid_analysis_{date_str}.json")
        for x in hybrid.get("weekly_top5", []):
            if isinstance(x, dict) and x.get("ticker"):
                ticker = str(x["ticker"]).strip().upper()
                weekly.append(ticker)
                metadata.setdefault(ticker, {"source": "weekly_top5", "rank": len(weekly)})
    
    weekly = _dedup_keep_order(weekly)
    
    # Pro30
    pro30 = []
    for pattern in ["30d_momentum_candidates", "30d_breakout_candidates", "30d_reversal_candidates"]:
        csv_path = run_dir / f"{pattern}_{date_str}.csv"
        df = _safe_csv(csv_path)
        if not df.empty and "Ticker" in df.columns:
            for _, row in df.iterrows():
                ticker = str(row["Ticker"]).strip().upper()
                if ticker and ticker not in metadata:
                    pro30.append(ticker)
                    metadata[ticker] = {
                        "source": pattern.replace("_candidates", ""),
                        "score": row.get("Score") or row.get("Composite_Score"),
                    }
    pro30 = _dedup_keep_order(pro30)
    
    # Movers
    movers = []
    hybrid = _safe_json(run_dir / f"hybrid_analysis_{date_str}.json")
    for t in hybrid.get("movers_tickers", []):
        ticker = str(t).strip().upper()
        if ticker:
            movers.append(ticker)
            if ticker not in metadata:
                metadata[ticker] = {"source": "movers"}
    movers = _dedup_keep_order(movers)
    
    combined = _dedup_keep_order(swing + weekly + pro30 + movers)
    
    return DatePicks(
        date_str=date_str,
        swing_top5=swing,
        weekly_top5=weekly,
        pro30=pro30,
        movers=movers,
        combined=combined,
        metadata=metadata,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PRICE DATA (Polygon preferred, Yahoo Finance fallback)
# ═══════════════════════════════════════════════════════════════════════════════

import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
USE_POLYGON = bool(POLYGON_API_KEY)


@dataclass
class PriceData:
    close: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame


def _fetch_polygon_daily(ticker: str, start_date: str, end_date: str, api_key: str) -> pd.DataFrame:
    """Fetch daily OHLCV from Polygon for a single ticker."""
    try:
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
        resp = requests.get(
            url,
            params={"adjusted": "true", "sort": "asc", "limit": 5000, "apiKey": api_key},
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


def _download_polygon_batch(tickers: list[str], start_date: str, end_date: str, api_key: str, max_workers: int = 8) -> dict[str, pd.DataFrame]:
    """Download daily OHLCV for many tickers from Polygon."""
    results: dict[str, pd.DataFrame] = {}
    
    def worker(t: str):
        return t, _fetch_polygon_daily(t, start_date, end_date, api_key)
    
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(worker, t): t for t in tickers}
        for fut in as_completed(futures):
            t, df = fut.result()
            results[t] = df
    return results


def _download_yfinance(tickers: list[str], start_date: str, end_date: str) -> PriceData:
    """Download from Yahoo Finance (fallback)."""
    import yfinance as yf
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
    
    data = yf.download(
        tickers=tickers,
        start=start_dt,
        end=end_dt,
        progress=False,
        auto_adjust=False,
        threads=True,
        group_by="column",
    )
    
    if data is None or data.empty:
        return PriceData(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    
    if isinstance(data.columns, pd.MultiIndex):
        close = data.get("Close", pd.DataFrame()).copy()
        high = data.get("High", pd.DataFrame()).copy()
        low = data.get("Low", pd.DataFrame()).copy()
    else:
        close = data[["Close"]].copy()
        close.columns = [tickers[0]]
        high = data[["High"]].copy()
        high.columns = [tickers[0]]
        low = data[["Low"]].copy()
        low.columns = [tickers[0]]
    
    return PriceData(close.sort_index(), high.sort_index(), low.sort_index())


def download_prices(tickers: list[str], start_date: str, end_date: str) -> PriceData:
    """
    Download OHLC data. Uses Polygon if API key available, falls back to Yahoo Finance.
    """
    tickers = _dedup_keep_order(tickers)
    if not tickers:
        return PriceData(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    
    logger.info(f"Downloading prices for {len(tickers)} tickers...")
    
    # Try Polygon first if available
    if USE_POLYGON:
        logger.info("Using Polygon.io as primary data source...")
        polygon_data = _download_polygon_batch(tickers, start_date, end_date, POLYGON_API_KEY)
        
        close_dfs, high_dfs, low_dfs = [], [], []
        success_count = 0
        failed_tickers = []
        
        for ticker in tickers:
            df = polygon_data.get(ticker, pd.DataFrame())
            if not df.empty:
                close_dfs.append(df[["Close"]].rename(columns={"Close": ticker}))
                high_dfs.append(df[["High"]].rename(columns={"High": ticker}))
                low_dfs.append(df[["Low"]].rename(columns={"Low": ticker}))
                success_count += 1
            else:
                failed_tickers.append(ticker)
        
        logger.info(f"Polygon: {success_count}/{len(tickers)} tickers succeeded")
        
        # Fall back to Yahoo Finance for failed tickers
        if failed_tickers:
            logger.info(f"Falling back to Yahoo Finance for {len(failed_tickers)} tickers...")
            yf_data = _download_yfinance(failed_tickers, start_date, end_date)
            if not yf_data.close.empty:
                for ticker in failed_tickers:
                    if ticker in yf_data.close.columns:
                        close_dfs.append(yf_data.close[[ticker]])
                        high_dfs.append(yf_data.high[[ticker]])
                        low_dfs.append(yf_data.low[[ticker]])
        
        if close_dfs:
            close = pd.concat(close_dfs, axis=1).sort_index()
            high = pd.concat(high_dfs, axis=1).sort_index()
            low = pd.concat(low_dfs, axis=1).sort_index()
            return PriceData(close, high, low)
        
        logger.warning("Polygon failed completely, falling back to Yahoo Finance...")
    
    # Yahoo Finance fallback
    return _download_yfinance(tickers, start_date, end_date)


def compute_forward_returns(
    prices: PriceData,
    ticker: str,
    entry_date: str,
    holding_periods: list[int],
) -> dict[int, dict]:
    results = {}
    ticker = ticker.upper()
    
    if prices.close.empty or ticker not in prices.close.columns:
        return {p: {} for p in holding_periods}
    
    close_s = prices.close[ticker]
    high_s = prices.high[ticker] if ticker in prices.high.columns else close_s
    low_s = prices.low[ticker] if ticker in prices.low.columns else close_s
    
    entry_ts = pd.Timestamp(entry_date)
    valid_mask = (close_s.index >= entry_ts) & close_s.notna()
    
    if not valid_mask.any():
        return {p: {} for p in holding_periods}
    
    entry_idx = valid_mask.idxmax()
    entry_pos = close_s.index.get_loc(entry_idx)
    entry_price = float(close_s.iloc[entry_pos])
    
    for period in holding_periods:
        start_pos = entry_pos + 1
        end_pos = start_pos + period
        
        if start_pos >= len(close_s):
            results[period] = {"entry_price": entry_price, "insufficient_data": True}
            continue
        
        close_window = close_s.iloc[start_pos:end_pos].dropna()
        high_window = high_s.iloc[start_pos:end_pos].dropna()
        low_window = low_s.iloc[start_pos:end_pos].dropna()
        
        if close_window.empty:
            results[period] = {"entry_price": entry_price, "insufficient_data": True}
            continue
        
        exit_price = float(close_window.iloc[-1])
        max_price = float(high_window.max()) if len(high_window) > 0 else None
        min_price = float(low_window.min()) if len(low_window) > 0 else None
        
        return_pct = ((exit_price / entry_price) - 1) * 100
        max_return_pct = ((max_price / entry_price) - 1) * 100 if max_price else None
        max_drawdown_pct = ((min_price / entry_price) - 1) * 100 if min_price else None
        
        results[period] = {
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "max_price": round(max_price, 2) if max_price else None,
            "min_price": round(min_price, 2) if min_price else None,
            "return_pct": round(return_pct, 2),
            "max_return_pct": round(max_return_pct, 2) if max_return_pct else None,
            "max_drawdown_pct": round(max_drawdown_pct, 2) if max_drawdown_pct else None,
            "trading_days": len(close_window),
        }
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_backtest(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    holding_periods: list[int] = [5, 7, 10, 14],
    hit_thresholds: list[float] = [5.0, 7.0, 10.0, 15.0],  # Added +7%
) -> pd.DataFrame:
    all_dates = iter_output_dates()
    
    if start_date:
        all_dates = [d for d in all_dates if d >= start_date]
    if end_date:
        all_dates = [d for d in all_dates if d <= end_date]
    
    if not all_dates:
        logger.warning("No scan dates found in range")
        return pd.DataFrame()
    
    logger.info(f"Found {len(all_dates)} scan dates: {all_dates[0]} → {all_dates[-1]}")
    
    all_picks = [load_picks_for_date(d) for d in all_dates]
    all_tickers = _dedup_keep_order([t for p in all_picks for t in p.combined])
    logger.info(f"Total unique tickers: {len(all_tickers)}")
    
    max_period = max(holding_periods)
    price_end = (datetime.strptime(all_dates[-1], "%Y-%m-%d") + timedelta(days=max_period * 2)).strftime("%Y-%m-%d")
    today = utc_now().strftime("%Y-%m-%d")
    price_end = min(price_end, today)
    
    prices = download_prices(all_tickers, all_dates[0], price_end)
    
    if prices.close.empty:
        logger.error("Failed to download price data")
        return pd.DataFrame()
    
    rows = []
    for picks in all_picks:
        date_str = picks.date_str
        swing_set = set(picks.swing_top5)
        weekly_set = set(picks.weekly_top5)
        pro30_set = set(picks.pro30)
        movers_set = set(picks.movers)
        
        for ticker in picks.combined:
            forward_returns = compute_forward_returns(prices, ticker, date_str, holding_periods)
            meta = picks.metadata.get(ticker, {})
            
            for period, ret_data in forward_returns.items():
                if not ret_data or ret_data.get("insufficient_data"):
                    continue
                
                row = {
                    "scan_date": date_str,
                    "ticker": ticker,
                    "period": period,
                    "in_swing_top5": ticker in swing_set,
                    "in_weekly_top5": ticker in weekly_set,
                    "in_pro30": ticker in pro30_set,
                    "in_movers": ticker in movers_set,
                    "weekly_rank": meta.get("rank") if ticker in weekly_set else None,
                    "source": meta.get("source", "unknown"),
                    "composite_score": meta.get("composite_score"),
                    **ret_data,
                }
                
                for thresh in hit_thresholds:
                    max_ret = ret_data.get("max_return_pct")
                    row[f"hit_{int(thresh)}pct"] = bool(max_ret is not None and max_ret >= thresh)
                
                rows.append(row)
    
    df = pd.DataFrame(rows)
    logger.info(f"Backtest complete: {len(df)} observations")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_scorecard(df: pd.DataFrame, primary_period: int = 7, primary_threshold: int = 7) -> dict:
    """Generate model scorecard. Default primary KPI is +7% (adjusted from +10%)."""
    if df.empty:
        return {"status": "No data"}
    
    primary_df = df[df["period"] == primary_period].copy()
    
    # Use the specified threshold (default +7%)
    hit_col = f"hit_{primary_threshold}pct"
    
    scorecard = {
        "data_summary": {
            "total_observations": len(primary_df),
            "scan_dates": primary_df["scan_date"].nunique(),
            "unique_tickers": primary_df["ticker"].nunique(),
            "date_range": f"{primary_df['scan_date'].min()} → {primary_df['scan_date'].max()}",
        },
        "primary_kpi": {
            "metric": f"Hit +{primary_threshold}% within T+{primary_period} days",
            "hit_rate": primary_df[hit_col].mean() if hit_col in primary_df.columns else None,
            "avg_return": primary_df["return_pct"].mean(),
            "avg_max_return": primary_df["max_return_pct"].mean(),
            "win_rate": (primary_df["return_pct"] > 0).mean(),
        },
        "strategy_ranking": [],
        "model_health": "Unknown",
        "recommendations": [],
    }
    
    for name, mask in [("swing_top5", primary_df["in_swing_top5"]),
                       ("weekly_top5", primary_df["in_weekly_top5"]), 
                       ("pro30", primary_df["in_pro30"]),
                       ("movers", primary_df["in_movers"])]:
        sub = primary_df[mask]
        if len(sub) > 0:
            scorecard["strategy_ranking"].append({
                "strategy": name,
                "n": len(sub),
                "hit_rate": sub[hit_col].mean() if hit_col in sub.columns else None,
                "avg_return": sub["return_pct"].mean(),
            })
    
    scorecard["strategy_ranking"] = sorted(
        scorecard["strategy_ranking"], 
        key=lambda x: x.get("hit_rate") or 0, 
        reverse=True
    )
    
    hit_rate = scorecard["primary_kpi"]["hit_rate"] or 0
    win_rate = scorecard["primary_kpi"]["win_rate"] or 0
    
    # Adjusted thresholds for +7% target (more achievable)
    if hit_rate >= 0.40 and win_rate >= 0.55:
        scorecard["model_health"] = "🟢 Excellent"
    elif hit_rate >= 0.30 and win_rate >= 0.50:
        scorecard["model_health"] = "🟡 Good"
    elif hit_rate >= 0.20:
        scorecard["model_health"] = "🟠 Needs Attention"
    else:
        scorecard["model_health"] = "🔴 Poor"
    
    return scorecard


def generate_improvements(df: pd.DataFrame, primary_period: int = 7, primary_threshold: int = 7) -> list[dict]:
    """Generate improvement suggestions. Now uses +7% as primary threshold."""
    suggestions = []
    period_df = df[df["period"] == primary_period].copy()
    
    if period_df.empty:
        return suggestions
    
    hit_col = f"hit_{primary_threshold}pct"
    weekly_hit = period_df[period_df["in_weekly_top5"]][hit_col].mean() if hit_col in period_df.columns else 0
    pro30_hit = period_df[period_df["in_pro30"]][hit_col].mean() if hit_col in period_df.columns else 0
    
    if pro30_hit > weekly_hit * 1.3 and pro30_hit > 0.25:
        suggestions.append({
            "category": "Strategy",
            "priority": "High",
            "suggestion": f"Pro30 outperforms Weekly ({pro30_hit*100:.1f}% vs {weekly_hit*100:.1f}%). Increase Pro30 weight.",
            "config_change": "Increase pro30 weight in hybrid analysis",
        })
    
    score_df = period_df[period_df["composite_score"].notna()]
    if not score_df.empty:
        high_score = score_df[score_df["composite_score"] >= 5.5][hit_col].mean() if hit_col in score_df.columns else 0
        low_score = score_df[score_df["composite_score"] < 5.0][hit_col].mean() if hit_col in score_df.columns else 0
        
        if high_score > low_score * 1.5 and high_score > 0.25:
            suggestions.append({
                "category": "Quality Filters",
                "priority": "High",
                "suggestion": f"High scores (≥5.5) outperform ({high_score*100:.1f}% vs {low_score*100:.1f}%). Raise minimum threshold.",
                "config_change": "quality_filters_weekly.min_composite_score: 5.5",
            })
    
    period_hits = {}
    for p in df["period"].unique():
        p_df = df[df["period"] == p]
        if hit_col in p_df.columns:
            period_hits[p] = p_df[hit_col].mean()
    
    if period_hits:
        best_period = max(period_hits, key=period_hits.get)
        if best_period != primary_period and period_hits[best_period] > period_hits.get(primary_period, 0) * 1.2:
            suggestions.append({
                "category": "Holding Period",
                "priority": "Medium",
                "suggestion": f"T+{best_period}d has higher hit rate ({period_hits[best_period]*100:.1f}% vs {period_hits.get(primary_period, 0)*100:.1f}%).",
                "config_change": f"forward_trading_days: {best_period}",
            })
    
    return suggestions


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN COMMAND
# ═══════════════════════════════════════════════════════════════════════════════

def run_validation(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    holding_periods: list[int] = [5, 7, 10, 14],
    hit_thresholds: list[float] = [5.0, 7.0, 10.0, 15.0],  # Added +7% as key threshold
    output_dir: str = "outputs/performance",
    quick_mode: bool = False,
) -> int:
    """
    Run multi-period model validation.
    """
    logger.info("=" * 60)
    logger.info("MODEL VALIDATION")
    logger.info("=" * 60)
    
    logger.info(f"Holding periods: {holding_periods}")
    logger.info(f"Hit thresholds: {hit_thresholds}")
    
    # Run backtest
    df = run_full_backtest(
        start_date=start_date,
        end_date=end_date,
        holding_periods=holding_periods,
        hit_thresholds=hit_thresholds,
    )
    
    if df.empty:
        logger.error("No data to validate")
        return 1
    
    # Generate scorecard with +7% as primary threshold
    scorecard = generate_scorecard(df, primary_period=7, primary_threshold=7)
    
    # Print results
    print("\n" + "═" * 60)
    print("📊 MODEL QUALITY SCORECARD")
    print("═" * 60)
    
    summary = scorecard.get("data_summary", {})
    print(f"\n📈 Data Summary:")
    print(f"   • Observations: {summary.get('total_observations', 0):,}")
    print(f"   • Scan Dates: {summary.get('scan_dates', 0)}")
    print(f"   • Unique Tickers: {summary.get('unique_tickers', 0)}")
    print(f"   • Date Range: {summary.get('date_range', 'N/A')}")
    
    kpi = scorecard.get("primary_kpi", {})
    print(f"\n🎯 Primary KPI ({kpi.get('metric', 'N/A')}):")
    print(f"   • Hit Rate: {(kpi.get('hit_rate') or 0) * 100:.1f}%")
    print(f"   • Win Rate: {(kpi.get('win_rate') or 0) * 100:.1f}%")
    print(f"   • Avg Return: {kpi.get('avg_return', 0):.1f}%")
    print(f"   • Avg Max Return: {kpi.get('avg_max_return', 0):.1f}%")
    
    print(f"\n🏆 Strategy Ranking:")
    for i, s in enumerate(scorecard.get("strategy_ranking", []), 1):
        hr = (s.get('hit_rate') or 0) * 100
        ar = s.get('avg_return', 0)
        print(f"   {i}. {s['strategy']}: Hit={hr:.1f}% | Return={ar:.1f}% | N={s['n']}")
    
    print(f"\n💊 Model Health: {scorecard.get('model_health', 'Unknown')}")
    
    # Hit rate matrix (now includes +7%)
    print("\n📊 Hit Rate Matrix (Period × Threshold):")
    print("-" * 60)
    print(f"{'Period':<10} {'N':<6} {'+5%':<8} {'+7%':<8} {'+10%':<8} {'+15%':<8} {'Avg Ret':<8}")
    print("-" * 60)
    
    for period in sorted(df["period"].unique()):
        period_df = df[df["period"] == period]
        n = len(period_df)
        h5 = period_df["hit_5pct"].mean() * 100 if "hit_5pct" in period_df.columns else 0
        h7 = period_df["hit_7pct"].mean() * 100 if "hit_7pct" in period_df.columns else 0
        h10 = period_df["hit_10pct"].mean() * 100 if "hit_10pct" in period_df.columns else 0
        h15 = period_df["hit_15pct"].mean() * 100 if "hit_15pct" in period_df.columns else 0
        avg_ret = period_df["return_pct"].mean()
        print(f"T+{period}d{'':<5} {n:<6} {h5:<8.1f} {h7:<8.1f} {h10:<8.1f} {h15:<8.1f} {avg_ret:<+8.1f}")
    
    # Improvements
    suggestions = generate_improvements(df)
    if suggestions:
        print("\n💡 MODEL IMPROVEMENT SUGGESTIONS")
        print("=" * 60)
        for i, s in enumerate(suggestions, 1):
            priority_icon = "🔴" if s["priority"] == "High" else "🟡"
            print(f"\n{i}. [{priority_icon} {s['priority']}] {s['category']}")
            print(f"   {s['suggestion']}")
            print(f"   → Config: {s['config_change']}")
    
    # Save results
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    timestamp = utc_now().strftime("%Y-%m-%d")
    
    df.to_csv(out_path / f"validation_detail_{timestamp}.csv", index=False)
    with open(out_path / f"validation_scorecard_{timestamp}.json", "w") as f:
        json.dump(scorecard, f, indent=2, default=str)
    
    # Append validation results to model history
    _append_validation_to_history(scorecard, suggestions, timestamp)
    
    logger.info(f"\n✅ Results saved to {output_dir}")
    print("\n" + "═" * 60)
    
    return 0


def _append_validation_to_history(scorecard: dict, suggestions: list, timestamp: str) -> None:
    """Append validation summary to model_history.md."""
    history_path = Path("outputs/model_history.md")
    
    kpi = scorecard.get("primary_kpi", {})
    hit_rate = (kpi.get("hit_rate") or 0) * 100
    win_rate = (kpi.get("win_rate") or 0) * 100
    avg_return = kpi.get("avg_return", 0)
    health = scorecard.get("model_health", "Unknown")
    
    lines = [
        f"\n### 📊 Validation Check ({timestamp})",
        f"",
        f"**KPI:** {kpi.get('metric', 'N/A')}",
        f"- Hit Rate: {hit_rate:.1f}%",
        f"- Win Rate: {win_rate:.1f}%", 
        f"- Avg Return: {avg_return:.1f}%",
        f"- Model Health: {health}",
        f"",
    ]
    
    # Strategy ranking
    ranking = scorecard.get("strategy_ranking", [])
    if ranking:
        lines.append("**Strategy Performance:**")
        for s in ranking:
            hr = (s.get('hit_rate') or 0) * 100
            lines.append(f"- {s['strategy']}: {hr:.1f}% hit rate (n={s['n']})")
        lines.append("")
    
    # Suggestions
    if suggestions:
        lines.append("**Improvement Suggestions:**")
        for s in suggestions:
            lines.append(f"- [{s['priority']}] {s['suggestion']}")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    
    try:
        with open(history_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except Exception:
        pass  # Don't fail validation if history append fails
