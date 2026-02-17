"""Backtest runner — orchestrates execution, metrics, and portfolio constraints.

Replays historical picks from the outcome database against price data,
applying realistic entry pricing, path metrics, and portfolio constraints.

Usage:
    python -m src.backtest.runner --start 2025-01-01 --end 2026-02-17 --max-positions 5
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.execution import ExecutionModel, entry_price
from src.backtest.metrics import compute_path_metrics, compute_expectancy
from src.backtest.portfolio import enforce_overlap_limit, compute_portfolio_metrics
from src.core.outcome_db import OutcomeDatabase
from src.core.price_db import PriceDatabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

HOLDING_PERIOD_DAYS = 10
TARGET_PCT = 10.0
STOP_PCT = 7.0
SLIPPAGE_BPS = 20.0


@dataclass
class BacktestConfig:
    start_date: str
    end_date: str
    max_positions: int = 5
    holding_period_days: int = HOLDING_PERIOD_DAYS
    target_pct: float = TARGET_PCT
    stop_pct: float = STOP_PCT
    slippage_bps: float = SLIPPAGE_BPS
    position_size_pct: float = 20.0
    sources: list[str] | None = None


def run_backtest(config: BacktestConfig) -> dict:
    """Execute a full backtest over the configured date range.

    Flow:
    1. Load picks from outcome DB
    2. For each pick, compute entry price via ExecutionModel
    3. Compute PathMetrics (MFE, MAE, days_to_hit) from OHLCV
    4. Apply portfolio overlap constraints
    5. Aggregate via compute_expectancy + portfolio metrics
    6. Output standardized JSON
    """
    start_time = time.monotonic()
    logger.info("=" * 60)
    logger.info(
        "KooCore-D Backtest: %s → %s (max_pos=%d, hold=%dd)",
        config.start_date, config.end_date,
        config.max_positions, config.holding_period_days,
    )
    logger.info("=" * 60)

    outcome_db = OutcomeDatabase()
    price_db = PriceDatabase()

    exec_model = ExecutionModel(
        entry="next_open",
        slippage_bps=config.slippage_bps / 2,
        fee_bps=config.slippage_bps / 2,
    )

    # 1. Load picks from outcome DB
    df = outcome_db.get_training_data(
        min_date=config.start_date,
        max_date=config.end_date,
        sources=config.sources,
    )

    if df.empty:
        logger.warning("No outcome data found for date range")
        return _empty_report(config)

    logger.info("Loaded %d outcomes from DB", len(df))

    # 2-3. Process each pick
    trades = []
    for _, row in df.iterrows():
        ticker = row["ticker"]
        pick_date = str(row["pick_date"])[:10]

        # Fetch OHLCV for this ticker around the pick date
        start = pd.to_datetime(pick_date) - timedelta(days=5)
        end = pd.to_datetime(pick_date) + timedelta(days=config.holding_period_days + 15)
        ohlcv = price_db.get_prices(ticker, str(start.date()), str(end.date()))

        if ohlcv.empty:
            continue

        # Build DatetimeIndex for execution model
        ohlcv_indexed = ohlcv.copy()
        if "date" in ohlcv_indexed.columns:
            ohlcv_indexed.index = pd.to_datetime(ohlcv_indexed["date"])
        # Ensure column names match what execution model expects
        col_map = {}
        for col in ohlcv_indexed.columns:
            col_map[col] = col.capitalize() if col in ("open", "high", "low", "close", "volume") else col
        ohlcv_indexed.rename(columns=col_map, inplace=True)

        # Compute realistic entry price
        ep = entry_price(ohlcv_indexed, pick_date, exec_model)
        if ep is None:
            continue

        entry_dt = pd.to_datetime(pick_date)

        # Compute path metrics
        pm = compute_path_metrics(
            ohlcv_indexed,
            entry_px=ep,
            start_dt=entry_dt,
            horizon_days=config.holding_period_days,
            target_pct=config.target_pct,
            stop_pct=config.stop_pct,
        )

        # Determine exit date and price
        if pm.days_to_hit is not None:
            exit_dt = entry_dt + timedelta(days=pm.days_to_hit)
        else:
            exit_dt = entry_dt + timedelta(days=config.holding_period_days)

        # Compute actual PnL based on exit reason
        if pm.exit_reason == "target_hit":
            pnl_pct = min(pm.mfe or 0, config.target_pct)
        elif pm.exit_reason == "stop_hit":
            pnl_pct = -(config.stop_pct)
        else:
            # Timeout — use close at horizon
            pnl_pct = pm.mfe * 0.5 if pm.mfe else 0.0  # Approximation

        trade = {
            "ticker": ticker,
            "signal_date": pick_date,
            "asof_date": pick_date,
            "entry_date": str((entry_dt + timedelta(days=1)).date()),
            "entry_price": round(ep, 2),
            "exit_date": str(exit_dt.date()),
            "exit_reason": pm.exit_reason,
            "pnl_pct": round(pnl_pct, 2),
            "mfe_pct": round(pm.mfe, 2) if pm.mfe is not None else None,
            "mae_pct": round(pm.mae, 2) if pm.mae is not None else None,
            "days_to_hit": pm.days_to_hit,
            "hit": pm.hit,
            "hold_days": pm.days_to_hit or config.holding_period_days,
            "source": row.get("source", "unknown"),
        }
        trades.append(trade)

    logger.info("Processed %d trades with valid price data", len(trades))

    if not trades:
        return _empty_report(config)

    # 4. Apply portfolio overlap constraints
    trades = enforce_overlap_limit(trades, max_concurrent=config.max_positions)
    executed = [t for t in trades if not t.get("portfolio_blocked")]

    # 5. Compute expectancy
    expectancy_input = [
        {"hit": t["hit"], "mfe_pct": t["mfe_pct"], "mae_pct": t["mae_pct"]}
        for t in executed
    ]
    expectancy = compute_expectancy(expectancy_input, target_pct=config.target_pct)

    # Portfolio metrics
    portfolio = compute_portfolio_metrics(
        trades,
        position_size_pct=config.position_size_pct,
    )

    # 6. Compute summary statistics
    pnls = [t["pnl_pct"] for t in executed if t["pnl_pct"] is not None]
    summary = _compute_summary(pnls, executed, config)

    elapsed = time.monotonic() - start_time

    # Build equity curve
    equity_curve = _build_equity_curve(executed)

    # Format standardized output
    report = {
        "engine": "koocore-d",
        "run_date": str(date.today()),
        "date_range": {"start": config.start_date, "end": config.end_date},
        "config": {
            "holding_period_days": config.holding_period_days,
            "max_positions": config.max_positions,
            "slippage_bps": config.slippage_bps,
            "target_pct": config.target_pct,
            "stop_pct": config.stop_pct,
        },
        "summary": summary,
        "expectancy": expectancy,
        "portfolio": portfolio,
        "trades": [
            {
                "ticker": t["ticker"],
                "signal_date": t["signal_date"],
                "entry_date": t["entry_date"],
                "entry_price": t["entry_price"],
                "exit_date": t["exit_date"],
                "exit_reason": t["exit_reason"],
                "pnl_pct": t["pnl_pct"],
                "mfe_pct": t["mfe_pct"],
                "mae_pct": t["mae_pct"],
                "hold_days": t["hold_days"],
            }
            for t in executed
        ],
        "equity_curve": equity_curve,
        "elapsed_s": round(elapsed, 2),
    }

    # Save report
    output_dir = Path("backtest_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"koocore_backtest_{config.start_date}_{config.end_date}.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))

    logger.info(
        "Backtest complete in %.1fs: %d trades, win_rate=%.1f%%, report=%s",
        elapsed, len(executed), summary.get("win_rate", 0) * 100, report_path,
    )

    return report


def _compute_summary(pnls: list[float], trades: list[dict], config: BacktestConfig) -> dict:
    """Compute standardized summary metrics."""
    if not pnls:
        return {
            "total_trades": 0, "win_rate": 0, "avg_return_pct": 0,
            "sharpe": 0, "sortino": 0, "max_drawdown_pct": 0,
            "profit_factor": 0, "expectancy_pct": 0, "calmar": 0,
            "avg_hold_days": 0,
        }

    arr = np.array(pnls)
    wins = arr[arr > 0]
    losses = arr[arr <= 0]

    win_rate = len(wins) / len(arr) if len(arr) > 0 else 0
    avg_return = float(np.mean(arr))

    # Sharpe (annualized, ~50 trades/year)
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 1.0
    sharpe = avg_return / std * np.sqrt(50) if std > 0 else 0

    # Sortino
    downside = arr[arr < 0]
    ds_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 1.0
    sortino = avg_return / ds_std * np.sqrt(50) if ds_std > 0 else 0

    # Max drawdown
    cumulative = np.cumsum(arr)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0

    # Profit factor
    total_gains = float(np.sum(wins)) if len(wins) > 0 else 0
    total_losses_abs = abs(float(np.sum(losses))) if len(losses) > 0 else 0
    pf = total_gains / total_losses_abs if total_losses_abs > 0 else (float("inf") if total_gains > 0 else 0)

    # Expectancy
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0
    expectancy = avg_win * win_rate + avg_loss * (1 - win_rate)

    # Calmar
    calmar = float(np.sum(arr)) / max_dd if max_dd > 0 else 0

    # Avg hold days
    hold_days = [t.get("hold_days", config.holding_period_days) for t in trades]
    avg_hold = np.mean(hold_days) if hold_days else 0

    return {
        "total_trades": len(arr),
        "win_rate": round(float(win_rate), 4),
        "avg_return_pct": round(avg_return, 2),
        "sharpe": round(float(sharpe), 2),
        "sortino": round(float(sortino), 2),
        "max_drawdown_pct": round(-max_dd, 2),
        "profit_factor": round(float(pf), 2) if pf != float("inf") else 999.0,
        "expectancy_pct": round(float(expectancy), 2),
        "calmar": round(float(calmar), 2),
        "avg_hold_days": round(float(avg_hold), 1),
    }


def _build_equity_curve(trades: list[dict]) -> list[dict]:
    """Build cumulative PnL equity curve from executed trades."""
    sorted_trades = sorted(trades, key=lambda t: t.get("exit_date", t.get("entry_date", "")))
    cumulative = 0.0
    curve = []
    for t in sorted_trades:
        pnl = t.get("pnl_pct", 0) or 0
        cumulative += pnl
        curve.append({
            "date": t.get("exit_date", t.get("entry_date")),
            "cumulative_pnl_pct": round(cumulative, 2),
        })
    return curve


def _empty_report(config: BacktestConfig) -> dict:
    return {
        "engine": "koocore-d",
        "run_date": str(date.today()),
        "date_range": {"start": config.start_date, "end": config.end_date},
        "config": asdict(config) if hasattr(config, "__dataclass_fields__") else {},
        "summary": {"total_trades": 0},
        "trades": [],
        "equity_curve": [],
    }


def main():
    parser = argparse.ArgumentParser(description="KooCore-D Backtest Runner")
    parser.add_argument("--start", type=str, default="2025-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=str(date.today()), help="End date (YYYY-MM-DD)")
    parser.add_argument("--max-positions", type=int, default=5, help="Max concurrent positions")
    parser.add_argument("--holding-period", type=int, default=HOLDING_PERIOD_DAYS, help="Holding period in days")
    parser.add_argument("--target-pct", type=float, default=TARGET_PCT, help="Target return %%")
    parser.add_argument("--stop-pct", type=float, default=STOP_PCT, help="Stop loss %%")
    parser.add_argument("--slippage-bps", type=float, default=SLIPPAGE_BPS, help="Slippage in basis points")
    parser.add_argument("--sources", nargs="*", help="Filter by source (weekly_top5, pro30, movers)")
    args = parser.parse_args()

    config = BacktestConfig(
        start_date=args.start,
        end_date=args.end,
        max_positions=args.max_positions,
        holding_period_days=args.holding_period,
        target_pct=args.target_pct,
        stop_pct=args.stop_pct,
        slippage_bps=args.slippage_bps,
        sources=args.sources,
    )

    report = run_backtest(config)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
