"""
Price Data Module

Dashboard-safe wrapper around price fetching.
Returns figures instead of showing them - no .show() calls.
"""

from __future__ import annotations

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Polygon configuration
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
USE_POLYGON = bool(POLYGON_API_KEY)


def _to_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _unique_tickers(tickers: list[str]) -> list[str]:
    seen = set()
    out = []
    for t in tickers:
        t = str(t).strip().upper()
        if t and t not in seen:
            out.append(t)
            seen.add(t)
    return out


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


class StockTracker:
    """
    Dashboard-safe StockTracker.
    
    Key difference from notebook version:
    - plot_* methods return fig instead of calling .show()
    - No side effects, pure data transformation
    """
    
    def __init__(
        self,
        tickers: list[str],
        baseline_date: str,
        verbose: bool = False,
        auto_extend_if_short: bool = True,
        min_trading_days: int = 2,
        extra_backfill_days: int = 10,
    ):
        self.tickers = _unique_tickers(tickers)
        self.baseline_date = baseline_date
        self.verbose = verbose

        self.auto_extend_if_short = auto_extend_if_short
        self.min_trading_days = int(min_trading_days)
        self.extra_backfill_days = int(extra_backfill_days)

        self.close: pd.DataFrame = pd.DataFrame()
        self.baseline_prices: dict[str, float] = {}
        self.target_prices: dict[str, float] = {}

        self.window_start_ts: pd.Timestamp | None = None
        self.window_end_ts: pd.Timestamp | None = None
        self.per_ticker_baseline_ts: dict[str, pd.Timestamp] = {}

    def _log(self, msg: str):
        if self.verbose:
            logger.info(msg)

    def _download_close_yfinance(self, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        """Yahoo Finance download helper (fallback)."""
        try:
            data = yf.download(
                tickers=self.tickers,
                start=start_dt,
                end=end_dt,
                progress=False,
                auto_adjust=False,
                threads=True,
                group_by="column",
            )

            if data.empty:
                return pd.DataFrame()

            if isinstance(data.columns, pd.MultiIndex):
                close = data["Close"].copy()
            else:
                close = data[["Close"]].copy()
                close.columns = [self.tickers[0]]

            close = close.sort_index()
            return close
        except Exception as e:
            logger.warning(f"Yahoo Finance download failed: {e}")
            return pd.DataFrame()

    def _download_close(self, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        """Download Close prices. Uses Polygon if available, falls back to Yahoo Finance."""
        start_str = start_dt.strftime("%Y-%m-%d")
        end_str = end_dt.strftime("%Y-%m-%d")
        
        if USE_POLYGON:
            self._log(f"Using Polygon.io for {len(self.tickers)} tickers...")
            polygon_data = _download_polygon_batch(self.tickers, start_str, end_str, POLYGON_API_KEY)
            
            close_dfs = []
            success_count = 0
            failed_tickers = []
            
            for ticker in self.tickers:
                df = polygon_data.get(ticker, pd.DataFrame())
                if not df.empty and "Close" in df.columns:
                    close_dfs.append(df[["Close"]].rename(columns={"Close": ticker}))
                    success_count += 1
                else:
                    failed_tickers.append(ticker)
            
            self._log(f"Polygon: {success_count}/{len(self.tickers)} tickers succeeded")
            
            # Fall back to Yahoo Finance for failed tickers
            if failed_tickers and len(failed_tickers) < len(self.tickers):
                try:
                    yf_data = yf.download(
                        tickers=failed_tickers,
                        start=start_dt,
                        end=end_dt,
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
                            yf_close.columns = [failed_tickers[0]]
                        for ticker in failed_tickers:
                            if ticker in yf_close.columns:
                                close_dfs.append(yf_close[[ticker]])
                except Exception:
                    pass
            
            if close_dfs:
                close = pd.concat(close_dfs, axis=1).sort_index()
                return close
            
            self._log("Polygon failed completely, falling back to Yahoo Finance...")
        
        return self._download_close_yfinance(start_dt, end_dt)

    def download_close_once(self, end_date: str) -> pd.DataFrame:
        """One download for daily closes: baseline_date → end_date (+1 day)."""
        if not self.tickers:
            return pd.DataFrame()

        start_dt = _to_dt(self.baseline_date)
        end_dt = _to_dt(end_date) + timedelta(days=1)

        self._log(f"Downloading daily closes: {self.baseline_date} → {end_date} | tickers={len(self.tickers)}")

        close = self._download_close(start_dt, end_dt)

        if close.empty:
            self._log("No data returned.")
            self.close = pd.DataFrame()
            return self.close

        valid_rows = close.dropna(how="all")
        if self.auto_extend_if_short and len(valid_rows) < self.min_trading_days:
            start_dt2 = start_dt - timedelta(days=self.extra_backfill_days)
            close2 = self._download_close(start_dt2, end_dt)
            if not close2.empty:
                close = close2

        close = close.sort_index().ffill()
        self.close = close
        return close

    def compute_baseline_and_target(self, end_date: str) -> tuple[dict[str, float], dict[str, float]]:
        """Per-ticker baseline/target within the requested window."""
        if self.close.empty:
            return {}, {}

        start_dt = _to_dt(self.baseline_date)
        end_dt = _to_dt(end_date)

        w = self.close.loc[(self.close.index >= start_dt) & (self.close.index <= end_dt)].copy()
        w = w.dropna(how="all")
        if w.empty:
            return {}, {}

        self.window_start_ts = w.index[0]
        self.window_end_ts = w.index[-1]

        baseline_prices: dict[str, float] = {}
        target_prices: dict[str, float] = {}
        per_base_ts: dict[str, pd.Timestamp] = {}

        for t in self.tickers:
            if t not in w.columns:
                continue
            s = w[t].dropna()
            if s.empty:
                continue
            per_base_ts[t] = s.index[0]
            baseline_prices[t] = float(s.iloc[0])
            target_prices[t] = float(s.iloc[-1])

        self.baseline_prices = {k: round(v, 2) for k, v in baseline_prices.items()}
        self.target_prices = {k: round(v, 2) for k, v in target_prices.items()}
        self.per_ticker_baseline_ts = per_base_ts

        return self.baseline_prices, self.target_prices

    def make_returns_table(self) -> pd.DataFrame:
        """Performance table based on per-ticker baseline/target."""
        rows = []
        for t in self.tickers:
            bp = self.baseline_prices.get(t)
            tp = self.target_prices.get(t)
            if bp is None or tp is None or bp == 0:
                continue
            change = tp - bp
            pct = (change / bp) * 100.0
            base_ts = self.per_ticker_baseline_ts.get(t)
            rows.append({
                "Symbol": t,
                "Baseline": bp,
                "Current": tp,
                "Change": round(change, 2),
                "Percent_Change": pct,
                "Baseline_Date_Used": base_ts.date().isoformat() if isinstance(base_ts, pd.Timestamp) else None,
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df = df.sort_values("Percent_Change", ascending=False).reset_index(drop=True)
        return df

    def make_cumulative_returns_v2(self, end_date: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Bulletproof v2 cumulative returns."""
        if self.close.empty:
            return pd.DataFrame(), pd.Series(dtype=int), pd.DataFrame()

        start_dt = _to_dt(self.baseline_date)
        end_dt = _to_dt(end_date)

        w = self.close.loc[(self.close.index >= start_dt) & (self.close.index <= end_dt)].copy()
        w = w.dropna(how="all")
        if w.empty:
            return pd.DataFrame(), pd.Series(dtype=int), pd.DataFrame()

        td_counter = pd.Series(range(1, len(w.index) + 1), index=w.index, name="TradingDay")

        anchors = {}
        base_dates = {}
        for t in self.tickers:
            if t not in w.columns:
                continue
            s = w[t].dropna()
            if s.empty:
                continue
            anchors[t] = float(s.iloc[0])
            base_dates[t] = s.index[0]

        if not anchors:
            return pd.DataFrame(), td_counter, pd.DataFrame()

        anchor_series = pd.Series(anchors)
        df_cum = (w.divide(anchor_series, axis="columns") - 1.0) * 100.0
        df_cum = df_cum.dropna(how="all")

        meta = pd.DataFrame({
            "Ticker": list(anchors.keys()),
            "Baseline_Date_Used": [base_dates[t].date().isoformat() for t in anchors.keys()],
            "Baseline_Price_Used": [round(anchors[t], 2) for t in anchors.keys()],
        }).set_index("Ticker")

        return df_cum, td_counter, meta

    def get_bench_data(self, end_date: str) -> tuple[pd.DataFrame, pd.Series]:
        """Download and normalize benchmark close prices (S&P 500, Nasdaq 100)."""
        bench_symbols = ["^GSPC", "^NDX"]
        bench_label_map = {"^GSPC": "S&P 500", "^NDX": "Nasdaq 100"}
        bench_norm = pd.DataFrame()
        bench_returns = pd.Series(dtype=float)

        try:
            bench_raw = yf.download(bench_symbols, start=self.baseline_date, end=end_date, progress=False)
            if not bench_raw.empty:
                if isinstance(bench_raw.columns, pd.MultiIndex):
                    level0 = bench_raw.columns.get_level_values(0)
                    if "Adj Close" in level0:
                        bench_raw = bench_raw["Adj Close"]
                    elif "Close" in level0:
                        bench_raw = bench_raw["Close"]
                    else:
                        bench_raw = bench_raw.xs(level0[0], level=0, axis=1)
                else:
                    if "Adj Close" in bench_raw.columns:
                        bench_raw = bench_raw[["Adj Close"]]
                    elif "Close" in bench_raw.columns:
                        bench_raw = bench_raw[["Close"]]
                bench_raw = bench_raw.ffill().bfill()
                if not bench_raw.empty:
                    bench_norm = (bench_raw.divide(bench_raw.iloc[0]) - 1.0) * 100.0
                    bench_norm = bench_norm.rename(columns=bench_label_map)
                    bench_returns = ((bench_raw.iloc[-1] / bench_raw.iloc[0] - 1.0) * 100.0).rename(index=bench_label_map)
        except Exception as e:
            logger.warning(f"Benchmark download failed: {e}")

        return bench_norm, bench_returns

    def plot_bar_and_line_v2(
        self,
        df_perf: pd.DataFrame,
        end_date: str,
        bench_norm: Optional[pd.DataFrame] = None,
        bench_returns: Optional[pd.Series] = None,
    ) -> Optional[go.Figure]:
        """
        Create performance figure with bar chart and cumulative returns.
        
        Returns Figure object (does NOT call .show()).
        """
        if df_perf.empty:
            return None

        df_cum, td_counter, meta = self.make_cumulative_returns_v2(end_date)

        if bench_norm is None or bench_returns is None:
            bench_norm, bench_returns = self.get_bench_data(end_date)

        if not df_cum.empty and isinstance(bench_norm, pd.DataFrame) and not bench_norm.empty:
            bench_norm = bench_norm.reindex(df_cum.index).ffill()

        bench_dash_cycle = ["dot", "dash"]
        bench_colors = ["#666", "#999"]

        g0 = self.window_start_ts.date().isoformat() if self.window_start_ts is not None else self.baseline_date
        g1 = self.window_end_ts.date().isoformat() if self.window_end_ts is not None else end_date

        title_top = f"Total Return (per-ticker baseline): {g0} → {g1}"
        title_bottom = f"Cumulative Return (Start=0% per ticker) | X = Trading Day"

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=False,
            vertical_spacing=0.12,
            subplot_titles=(title_top, title_bottom),
            row_heights=[0.45, 0.55],
        )

        # Row 1: Bar chart
        colors = np.where(df_perf["Percent_Change"] >= 0, "#00CC44", "#FF4444")
        fig.add_trace(
            go.Bar(
                x=df_perf["Symbol"],
                y=df_perf["Percent_Change"],
                marker_color=colors,
                text=[f"{x:.1f}%" for x in df_perf["Percent_Change"]],
                textposition="outside",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Baseline: $%{customdata[0]:.2f}<br>"
                    "Current: $%{customdata[1]:.2f}<br>"
                    "Return: %{y:.2f}%<br>"
                    "Baseline Date: %{customdata[2]}<extra></extra>"
                ),
                customdata=list(zip(df_perf["Baseline"], df_perf["Current"], df_perf["Baseline_Date_Used"])),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        avg_change = float(df_perf["Percent_Change"].mean())
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.4, row=1, col=1)
        fig.add_hline(
            y=avg_change,
            line_dash="dash",
            line_color="blue",
            annotation_text=f"Avg: {avg_change:.2f}%",
            annotation_position="bottom right",
            row=1,
            col=1,
        )

        # Row 2: Cumulative return
        if not df_cum.empty:
            date_str = pd.Series(df_cum.index.strftime("%Y-%m-%d"), index=df_cum.index)
            mode = "lines" if len(df_cum.index) >= 3 else "lines+markers"

            for t in self.tickers:
                if t not in df_cum.columns:
                    continue
                s = df_cum[t].dropna()
                if s.empty:
                    continue

                x_td = td_counter.loc[s.index].values
                y = s.values
                hover_dates = date_str.loc[s.index].values

                if t in meta.index:
                    bdate = meta.loc[t, "Baseline_Date_Used"]
                    bpx = meta.loc[t, "Baseline_Price_Used"]
                else:
                    bdate, bpx = "N/A", np.nan

                fig.add_trace(
                    go.Scatter(
                        x=x_td,
                        y=y,
                        mode=mode,
                        name=t,
                        customdata=np.column_stack([hover_dates]),
                        hovertemplate=(
                            f"<b>{t}</b><br>"
                            "Trading Day: %{x}<br>"
                            "Date: %{customdata[0]}<br>"
                            "Return: %{y:.2f}%<br>"
                            f"Baseline: {bdate} @ {bpx}<extra></extra>"
                        ),
                    ),
                    row=2,
                    col=1,
                )

            avg_path = df_cum.mean(axis=1, skipna=True)
            if not avg_path.empty:
                fig.add_trace(
                    go.Scatter(
                        x=td_counter.loc[avg_path.index].values,
                        y=avg_path.values,
                        mode="lines",
                        name="Portfolio avg",
                        line=dict(color="purple", width=3, dash="dashdot"),
                        hovertemplate="Trading Day: %{x}<br>Avg return: %{y:.2f}%<extra></extra>",
                    ),
                    row=2,
                    col=1,
                )

            if not bench_norm.empty:
                for i, col in enumerate(bench_norm.columns):
                    s = bench_norm[col].dropna()
                    if s.empty:
                        continue
                    x_td = td_counter.loc[s.index].values
                    hover_dates = date_str.loc[s.index].values
                    display_name = f"{col}" if "benchmark" not in str(col).lower() else str(col)
                    fig.add_trace(
                        go.Scatter(
                            x=x_td,
                            y=s.values,
                            mode="lines",
                            name=display_name,
                            line=dict(
                                color=bench_colors[i % len(bench_colors)],
                                dash=bench_dash_cycle[i % len(bench_dash_cycle)],
                                width=2,
                            ),
                            customdata=np.column_stack([hover_dates]),
                            hovertemplate=(
                                f"<b>{display_name}</b><br>"
                                "Trading Day: %{x}<br>"
                                "Date: %{customdata[0]}<br>"
                                "Return: %{y:.2f}%<extra></extra>"
                            ),
                        ),
                        row=2,
                        col=1,
                    )

            fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.4, row=2, col=1)

            max_td = int(td_counter.max()) if not td_counter.empty else 1
            if max_td <= 15:
                fig.update_xaxes(dtick=1, row=2, col=1)
            elif max_td <= 60:
                fig.update_xaxes(dtick=5, row=2, col=1)
            else:
                fig.update_xaxes(dtick=10, row=2, col=1)

        fig.update_layout(
            height=800,
            template="plotly_white",
            hovermode="closest",
            title={"text": f"Stock Performance | End: {end_date}", "x": 0.5},
            legend_title_text="Tickers",
        )

        fig.update_xaxes(tickangle=45, row=1, col=1)
        fig.update_yaxes(title_text="Total Return (%)", row=1, col=1)
        fig.update_xaxes(title_text="Trading Day", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative Return (%)", row=2, col=1)

        return fig


def build_performance_figure(
    tickers: list[str],
    baseline_date: str,
    end_date: str,
    group_sets: Optional[dict] = None,
) -> tuple[Optional[go.Figure], pd.DataFrame]:
    """
    High-level function to build performance figure.
    
    Returns:
        (fig, df_perf) - figure and performance DataFrame
    """
    if not tickers:
        return None, pd.DataFrame()
    
    tracker = StockTracker(
        tickers,
        baseline_date,
        verbose=False,
        auto_extend_if_short=True,
        min_trading_days=2,
        extra_backfill_days=10,
    )

    tracker.download_close_once(end_date)
    tracker.compute_baseline_and_target(end_date)

    df_perf = tracker.make_returns_table()
    if df_perf.empty:
        return None, df_perf

    bench_norm, bench_returns = tracker.get_bench_data(end_date)

    fig = tracker.plot_bar_and_line_v2(
        df_perf,
        end_date,
        bench_norm=bench_norm,
        bench_returns=bench_returns,
    )

    return fig, df_perf


def get_ticker_performance(
    ticker: str,
    baseline_date: str,
    end_date: str,
) -> tuple[Optional[go.Figure], dict]:
    """
    Get performance figure for a single ticker.
    
    Returns:
        (fig, stats_dict)
    """
    fig, df = build_performance_figure([ticker], baseline_date, end_date)
    
    stats = {}
    if not df.empty:
        row = df.iloc[0]
        stats = {
            "ticker": ticker,
            "baseline": row["Baseline"],
            "current": row["Current"],
            "return_pct": row["Percent_Change"],
            "baseline_date": row["Baseline_Date_Used"],
        }
    
    return fig, stats
