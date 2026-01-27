# src/backtest/portfolio.py
"""
Portfolio-level constraints for backtests.

Handles:
- Maximum concurrent positions
- Overlap blocking
- Capital allocation (future)
"""
from __future__ import annotations
from typing import Dict, List, Optional
import pandas as pd


def enforce_overlap_limit(
    trades: List[Dict],
    max_concurrent: int,
    horizon_days: int = 7,
) -> List[Dict]:
    """
    Simple overlap gate to limit concurrent positions.
    
    Sorts trades by entry date and allows up to max_concurrent open at once.
    Blocked trades are marked but not removed.
    
    Args:
        trades: List of trade dicts with 'asof_date', optionally 'days_to_hit'
        max_concurrent: Maximum concurrent positions allowed
        horizon_days: Default holding period if days_to_hit is None
    
    Returns:
        List of trades with 'portfolio_blocked' flag added to blocked ones
    """
    if max_concurrent is None or max_concurrent <= 0:
        return trades
    
    if not trades:
        return trades

    # Sort by entry date
    sorted_trades = sorted(trades, key=lambda x: x.get("asof_date", ""))
    
    active: List[Dict] = []
    result: List[Dict] = []

    for t in sorted_trades:
        entry_str = t.get("asof_date")
        if not entry_str:
            result.append(t)
            continue
            
        entry_dt = pd.to_datetime(entry_str)
        
        # Determine exit date
        days_held = t.get("days_to_hit")
        if days_held is None or days_held <= 0:
            days_held = horizon_days
        exit_dt = entry_dt + pd.Timedelta(days=days_held)

        # Remove expired positions
        active = [a for a in active if a["exit_dt"] > entry_dt]

        if len(active) < max_concurrent:
            # Position can be taken
            t_copy = dict(t)
            t_copy["portfolio_blocked"] = False
            active.append({"exit_dt": exit_dt, **t_copy})
            result.append(t_copy)
        else:
            # Position blocked due to overlap
            t_copy = dict(t)
            t_copy["portfolio_blocked"] = True
            result.append(t_copy)

    return result


def compute_portfolio_metrics(
    trades: List[Dict],
    initial_capital: float = 100000.0,
    position_size_pct: float = 20.0,
) -> Dict:
    """
    Compute portfolio-level metrics from trade list.
    
    Args:
        trades: List of completed trade dicts
        initial_capital: Starting capital
        position_size_pct: Position size as % of capital
    
    Returns:
        Dict with portfolio metrics
    """
    if not trades:
        return {
            "total_trades": 0,
            "blocked_trades": 0,
            "executed_trades": 0,
            "total_return_pct": 0.0,
        }
    
    blocked = sum(1 for t in trades if t.get("portfolio_blocked"))
    executed = len(trades) - blocked
    
    # Simple P&L calculation (assumes fixed position sizing)
    position_value = initial_capital * (position_size_pct / 100.0)
    total_pnl = 0.0
    
    for t in trades:
        if t.get("portfolio_blocked"):
            continue
        mfe = t.get("mfe_pct", 0) or 0
        mae = t.get("mae_pct", 0) or 0
        hit = t.get("hit", False)
        
        # Simplified: assume we capture target if hit, else use close at horizon
        if hit:
            trade_return = min(mfe, 10.0)  # Cap at target
        else:
            # Use average of MFE and final (approximation)
            trade_return = mae * 0.5  # Simplified loss estimate
        
        total_pnl += position_value * (trade_return / 100.0)
    
    total_return_pct = (total_pnl / initial_capital) * 100.0
    
    return {
        "total_trades": len(trades),
        "blocked_trades": blocked,
        "executed_trades": executed,
        "total_return_pct": round(total_return_pct, 2),
        "total_pnl": round(total_pnl, 2),
    }
