#!/usr/bin/env python3
"""
Decision-grade portfolio simulator for backtest detail CSVs.

Simulates a proper portfolio with:
- Equal-weight sizing, hard position cap
- T+1 entry, actual exit dates
- Mark-to-market daily using real close prices
- Overlapping positions, capital recycling
- Idle cash earns zero

Uses the same CN data download path as the backtest harness.

Usage:
    python scripts/portfolio_sim.py outputs/backtest/backtest_detail_mr_new_default_3y.csv
    python scripts/portfolio_sim.py --compare \
        outputs/backtest/backtest_detail_mr_new_default_3y.csv \
        outputs/backtest/backtest_detail_mr_b1_wider_payoff_3y.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
load_dotenv(project_root / ".env")

from src.core.config import load_config
from src.core.data import get_data_functions

logger = logging.getLogger(__name__)


def _load_price_data(
    tickers: list[str], start: date, end: date, config_path: str = "config/default.yaml"
) -> dict[str, pd.DataFrame]:
    """Download OHLCV for all tickers needed by the simulation."""
    config = load_config(config_path)
    _, download_range_fn, provider_config, _ = get_data_functions(config)

    start_str = (start - timedelta(days=10)).strftime("%Y-%m-%d")
    end_str = (end + timedelta(days=30)).strftime("%Y-%m-%d")

    logger.info("Downloading price data for %d tickers (%s to %s)...", len(tickers), start_str, end_str)
    t0 = time.time()
    data_map, report = download_range_fn(
        tickers=tickers, start=start_str, end=end_str,
        provider_config=provider_config,
    )
    logger.info("Download: %d OK, %d failed (%.1f min)",
                len(data_map), len(report.get("bad_tickers", [])), (time.time() - t0) / 60)
    return data_map


def _build_calendar_from_data(data_map: dict[str, pd.DataFrame]) -> list[date]:
    """Extract actual trading dates from downloaded data."""
    all_dates: set[date] = set()
    for df in data_map.values():
        if df.empty:
            continue
        idx = df.index
        if hasattr(idx, 'date'):
            all_dates.update(idx.date)
        else:
            all_dates.update(pd.to_datetime(idx).date)
    return sorted(all_dates)


def _get_close(df: pd.DataFrame, target_date: date) -> float | None:
    """Get closing price for a specific date from a ticker DataFrame."""
    if df.empty:
        return None
    close_col = "Close" if "Close" in df.columns else "close"
    idx = df.index
    if hasattr(idx, 'date'):
        mask = idx.date == target_date
    else:
        mask = pd.to_datetime(idx).date == target_date
    matches = df.loc[mask]
    if matches.empty:
        return None
    return float(matches[close_col].iloc[-1])


def simulate_portfolio(
    detail_csv: str,
    data_map: dict[str, pd.DataFrame],
    calendar: list[date],
    initial_capital: float = 100_000.0,
    max_positions: int = 5,
) -> dict:
    """
    Mark-to-market portfolio simulation.

    Positions open on T+1 at entry_price, close on signal_date + exit_day
    trading days at exit_price. Daily equity = cash + sum of marked positions.
    """
    df = pd.read_csv(detail_csv)
    df = df[df['pnl_pct'].notna() & df['exit_day'].notna()].copy()
    if df.empty:
        return {"error": "no valid trades"}

    df['signal_date'] = pd.to_datetime(df['date']).dt.date
    df['exit_day'] = df['exit_day'].astype(int)
    df['score'] = df['score'].astype(float)

    cal_idx = {d: i for i, d in enumerate(calendar)}

    # Map each trade to actual calendar open/close dates
    trades = []
    for _, row in df.iterrows():
        sig_date = row['signal_date']
        if sig_date not in cal_idx:
            continue
        sig_pos = cal_idx[sig_date]

        # T+1: open next trading day
        open_pos = sig_pos + 1
        if open_pos >= len(calendar):
            continue

        # Close on exit_day trading days after signal
        close_pos = sig_pos + int(row['exit_day'])
        if close_pos >= len(calendar):
            close_pos = len(calendar) - 1

        trades.append({
            'ticker': row['ticker'],
            'signal_date': sig_date,
            'open_date': calendar[open_pos],
            'close_date': calendar[close_pos],
            'entry_price': float(row['entry_price']),
            'exit_price': float(row['exit_price']),
            'score': float(row['score']),
            'engine': row.get('engine', ''),
            'regime': row.get('regime', ''),
        })

    if not trades:
        return {"error": "no trades mapped to calendar"}

    trade_df = pd.DataFrame(trades)
    sim_start = trade_df['open_date'].min()
    sim_end = trade_df['close_date'].max()
    sim_days = [d for d in calendar if sim_start <= d <= sim_end]

    # === Day-by-day simulation ===
    cash = initial_capital
    open_positions: list[dict] = []  # {ticker, shares, entry_price, close_date, score, last_value}
    equity_curve = []
    skipped_trades = 0
    peak_equity = initial_capital
    max_dd = 0.0
    max_dd_date = None

    for day in sim_days:
        # 1. Close positions that expire today
        still_open = []
        for pos in open_positions:
            if pos['close_date'] <= day:
                # Sell at actual close on exit day (or last known close)
                sell_price = _get_close(data_map.get(pos['ticker'], pd.DataFrame()), day)
                if sell_price is None:
                    sell_price = pos['last_value'] / pos['shares'] if pos['shares'] > 0 else pos['entry_price']
                cash += sell_price * pos['shares']
            else:
                still_open.append(pos)
        open_positions = still_open

        # 2. Open new positions (trades with open_date == today)
        new_today = trade_df[trade_df['open_date'] == day].sort_values('score', ascending=False)

        for _, t in new_today.iterrows():
            if len(open_positions) >= max_positions:
                skipped_trades += 1
                continue

            # Allocate equal weight of current total equity
            total_equity = cash + sum(p['last_value'] for p in open_positions)
            alloc = total_equity / max_positions
            alloc = min(alloc, cash)  # can't spend more than available cash

            if alloc <= 0 or t['entry_price'] <= 0:
                skipped_trades += 1
                continue

            shares = alloc / t['entry_price']
            cost = shares * t['entry_price']
            cash -= cost

            open_positions.append({
                'ticker': t['ticker'],
                'shares': shares,
                'entry_price': t['entry_price'],
                'close_date': t['close_date'],
                'score': t['score'],
                'last_value': cost,
            })

        # 3. Mark-to-market all open positions
        for pos in open_positions:
            close_price = _get_close(data_map.get(pos['ticker'], pd.DataFrame()), day)
            if close_price is not None:
                pos['last_value'] = close_price * pos['shares']
            # else keep last_value unchanged (no trade day for this ticker)

        # 4. Compute total equity
        total_equity = cash + sum(p['last_value'] for p in open_positions)

        equity_curve.append({
            'date': day.isoformat(),
            'equity': round(total_equity, 2),
            'cash': round(cash, 2),
            'positions': len(open_positions),
        })

        # Track drawdown
        if total_equity > peak_equity:
            peak_equity = total_equity
        if peak_equity > 0:
            dd = (peak_equity - total_equity) / peak_equity
            if dd > max_dd:
                max_dd = dd
                max_dd_date = day

    # Compute final stats
    final_equity = equity_curve[-1]['equity'] if equity_curve else initial_capital
    total_return = (final_equity / initial_capital - 1) * 100

    # Annualized return
    years = len(sim_days) / 252.0
    if years > 0 and final_equity > 0:
        annual_return = ((final_equity / initial_capital) ** (1 / years) - 1) * 100
    else:
        annual_return = 0.0

    # Sharpe from daily equity changes
    equities = pd.Series([e['equity'] for e in equity_curve])
    daily_returns = equities.pct_change().dropna()
    if daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Average utilization
    positions_series = [e['positions'] for e in equity_curve]
    active_days = [p for p in positions_series if p > 0]

    result = {
        "input_file": detail_csv,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "initial_capital": initial_capital,
        "max_positions": max_positions,
        "selection_policy": "highest_score_first",
        "mark_to_market_method": "daily_close",
        "calendar_source": "actual_trading_dates",
        "total_trades_in_file": len(df),
        "total_trades_simulated": len(trades) - skipped_trades,
        "skipped_trades_capacity": skipped_trades,
        "sim_days": len(sim_days),
        "final_equity": round(final_equity, 2),
        "total_return_pct": round(total_return, 2),
        "equity_multiple": round(final_equity / initial_capital, 2),
        "annualized_return_pct": round(annual_return, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "max_drawdown_date": max_dd_date.isoformat() if max_dd_date else None,
        "sharpe_ratio": round(sharpe, 2),
        "avg_positions_when_active": round(np.mean(active_days), 1) if active_days else 0,
        "max_concurrent_positions": max(positions_series) if positions_series else 0,
        "utilization_pct": round(len(active_days) / max(len(sim_days), 1) * 100, 1),
    }

    # Save daily equity CSV
    eq_path = Path(detail_csv).with_name(
        Path(detail_csv).stem.replace("detail", "equity") + ".csv"
    )
    pd.DataFrame(equity_curve).to_csv(eq_path, index=False)
    result["equity_csv"] = str(eq_path)

    return result


def main():
    parser = argparse.ArgumentParser(description="Decision-grade portfolio simulator")
    parser.add_argument("files", nargs="+", help="Backtest detail CSV file(s)")
    parser.add_argument("--capital", type=float, default=100_000)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

    # Collect all unique tickers and date range across all input files
    all_tickers: set[str] = set()
    min_date = date.max
    max_date = date.min
    for f in args.files:
        df = pd.read_csv(f)
        df = df[df['pnl_pct'].notna()].copy()
        all_tickers.update(df['ticker'].unique())
        dates = pd.to_datetime(df['date']).dt.date
        min_date = min(min_date, dates.min())
        max_date = max(max_date, dates.max())

    # Download price data once for all sims
    data_map = _load_price_data(sorted(all_tickers), min_date, max_date, args.config)
    calendar = _build_calendar_from_data(data_map)
    logger.info("Trading calendar: %d days (%s to %s)", len(calendar), calendar[0], calendar[-1])

    results = []
    for f in args.files:
        logger.info("Simulating: %s", f)
        result = simulate_portfolio(f, data_map, calendar, args.capital, args.max_positions)
        results.append(result)

        print(f"\n=== {Path(f).stem} ===")
        print(f"  Trades: {result['total_trades_simulated']} (skipped {result['skipped_trades_capacity']} over cap)")
        print(f"  Equity: {result['equity_multiple']}x ({result['total_return_pct']:+.1f}%)")
        print(f"  Max DD: {result['max_drawdown_pct']:.1f}% (on {result['max_drawdown_date']})")
        print(f"  Sharpe: {result['sharpe_ratio']:.2f}")
        print(f"  Ann. Return: {result['annualized_return_pct']:+.1f}%")
        print(f"  Avg positions: {result['avg_positions_when_active']}")
        print(f"  Utilization: {result['utilization_pct']:.0f}%")

        # Save result
        out_path = Path(f).with_name(Path(f).stem.replace("detail", "portfolio") + ".json")
        with open(out_path, 'w') as fout:
            json.dump(result, fout, indent=2, default=str)
        print(f"  Saved: {out_path}")

    if args.compare and len(results) >= 2:
        a, b = results[0], results[1]
        print(f"\n{'='*60}")
        print(f"COMPARISON: {Path(args.files[0]).stem} vs {Path(args.files[1]).stem}")
        print(f"{'='*60}")
        metrics = [
            ('total_trades_simulated', 'd'),
            ('skipped_trades_capacity', 'd'),
            ('equity_multiple', '.2f'),
            ('max_drawdown_pct', '.1f'),
            ('sharpe_ratio', '.2f'),
            ('annualized_return_pct', '.1f'),
            ('avg_positions_when_active', '.1f'),
            ('utilization_pct', '.0f'),
        ]
        print(f"  {'Metric':<28} {'Default':>10} {'B1':>10} {'Delta':>10}")
        print(f"  {'-'*58}")
        for key, fmt in metrics:
            va, vb = a.get(key, 0), b.get(key, 0)
            delta = vb - va
            print(f"  {key:<28} {va:>10{fmt}} {vb:>10{fmt}} {delta:>+10{fmt}}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
