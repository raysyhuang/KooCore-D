# src/backtest/execution.py
"""
Execution model for backtests.

Provides realistic entry price computation with:
- Entry models: next_open (default), same_close (explicit opt-in)
- Slippage and fees (basis points)
- Guardrails to prevent look-ahead in entry prices
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass(frozen=True)
class ExecutionModel:
    """
    Execution parameters for backtest trades.
    
    Attributes:
        entry: Entry timing model ("next_open" or "same_close")
        slippage_bps: Slippage in basis points (default 5)
        fee_bps: Fees in basis points (default 2)
    """
    entry: str = "next_open"   # {"next_open", "same_close"}
    slippage_bps: float = 5.0  # basis points
    fee_bps: float = 2.0       # basis points


def entry_price(
    df: pd.DataFrame,
    asof_date: str,
    model: ExecutionModel,
) -> Optional[float]:
    """
    Compute entry price with no look-ahead.
    
    - same_close: close on asof_date (allowed only if explicitly set)
    - next_open: next trading day's Open after asof_date (default)
    
    Args:
        df: OHLCV DataFrame with DatetimeIndex
        asof_date: Decision date (YYYY-MM-DD)
        model: ExecutionModel with entry type and costs
    
    Returns:
        Entry price with slippage+fees applied, or None if unavailable
    """
    if df is None or df.empty:
        return None

    # Ensure sorted datetime index
    df = df.sort_index()
    asof = pd.to_datetime(asof_date)

    if model.entry == "same_close":
        # Use close on the decision date (explicit opt-in only)
        # Note: This has look-ahead risk if decision was made during the day
        try:
            # Handle both exact datetime and date-only matching
            if asof in df.index:
                px = float(df.loc[asof, "Close"])
            else:
                # Try matching by date
                mask = df.index.date == asof.date()
                if mask.any():
                    px = float(df.loc[mask, "Close"].iloc[-1])
                else:
                    return None
        except (KeyError, IndexError):
            return None
    else:
        # next_open (default, no look-ahead)
        # Entry on next trading day's open after asof_date
        after = df.index[df.index > asof]
        if len(after) == 0:
            return None
        next_dt = after[0]
        try:
            px = float(df.loc[next_dt, "Open"])
        except (KeyError, IndexError):
            return None

    # Apply slippage + fees (entry cost increases price for buys)
    cost_bps = model.slippage_bps + model.fee_bps
    px *= (1.0 + cost_bps / 10000.0)
    
    return px


def exit_price(
    df: pd.DataFrame,
    exit_date: str,
    model: ExecutionModel,
    use_close: bool = True,
) -> Optional[float]:
    """
    Compute exit price with transaction costs.
    
    Args:
        df: OHLCV DataFrame with DatetimeIndex
        exit_date: Exit date (YYYY-MM-DD)
        model: ExecutionModel with costs
        use_close: If True, use Close; else use Open
    
    Returns:
        Exit price with slippage+fees applied (reduces proceeds)
    """
    if df is None or df.empty:
        return None

    df = df.sort_index()
    exit_dt = pd.to_datetime(exit_date)
    
    col = "Close" if use_close else "Open"
    
    try:
        if exit_dt in df.index:
            px = float(df.loc[exit_dt, col])
        else:
            # Try matching by date
            mask = df.index.date == exit_dt.date()
            if mask.any():
                px = float(df.loc[mask, col].iloc[-1])
            else:
                return None
    except (KeyError, IndexError):
        return None
    
    # Apply slippage + fees (exit cost reduces proceeds)
    cost_bps = model.slippage_bps + model.fee_bps
    px *= (1.0 - cost_bps / 10000.0)
    
    return px
