# src/core/event_gate.py
"""
Event gate for blocking picks near binary events (earnings, FDA, etc.).

This prevents inclusion in Top5 for stocks with imminent catalyst risk,
while still allowing them to appear in watchlist/candidate pools.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime, date

import pandas as pd


@dataclass(frozen=True)
class EventGateResult:
    """Result of event gate check."""
    blocked: bool
    reason: Optional[str]
    evidence: Dict[str, Any]


def earnings_proximity_gate(
    asof_date: str,
    earnings_date: Optional[str],
    block_days: int = 3,
) -> EventGateResult:
    """
    Check if earnings are within N days, blocking Top5 inclusion if so.
    
    The idea is to avoid binary event risk while still surfacing
    these names in the candidate/watchlist pool.
    
    Args:
        asof_date: Current as-of date (YYYY-MM-DD)
        earnings_date: Known earnings date (YYYY-MM-DD) or None
        block_days: Number of days before earnings to block (default 3)
    
    Returns:
        EventGateResult with blocked flag and evidence
    """
    e: Dict[str, Any] = {
        "earnings_date": earnings_date,
        "block_days": block_days
    }
    
    # No blocking if dates unknown
    if not asof_date or not earnings_date:
        return EventGateResult(
            blocked=False,
            reason=None,
            evidence=e
        )

    try:
        a = pd.to_datetime(asof_date).date()
        ed = pd.to_datetime(earnings_date).date()
        
        # Calculate days until earnings (simple calendar days)
        # For more accuracy, upgrade to trading-day calendar
        delta = (ed - a).days
        e["days_to_earnings"] = delta
        
        # Block if earnings within block_days (and not in the past)
        if 0 <= delta <= block_days:
            return EventGateResult(
                blocked=True,
                reason="earnings_soon",
                evidence=e
            )
        
        return EventGateResult(
            blocked=False,
            reason=None,
            evidence=e
        )
        
    except Exception as ex:
        return EventGateResult(
            blocked=False,
            reason=None,
            evidence={"earnings_date_parse_failed": True, "error": str(ex)[:100]}
        )


def catalyst_gate(
    asof_date: str,
    catalyst_date: Optional[str],
    catalyst_type: Optional[str] = None,
    block_days: int = 2,
) -> EventGateResult:
    """
    Generic catalyst gate for FDA dates, conferences, etc.
    
    Args:
        asof_date: Current as-of date
        catalyst_date: Known catalyst date or None
        catalyst_type: Type of catalyst (e.g., "FDA", "conference")
        block_days: Days before catalyst to block
    
    Returns:
        EventGateResult
    """
    e: Dict[str, Any] = {
        "catalyst_date": catalyst_date,
        "catalyst_type": catalyst_type,
        "block_days": block_days
    }
    
    if not asof_date or not catalyst_date:
        return EventGateResult(blocked=False, reason=None, evidence=e)
    
    try:
        a = pd.to_datetime(asof_date).date()
        cd = pd.to_datetime(catalyst_date).date()
        delta = (cd - a).days
        e["days_to_catalyst"] = delta
        
        if 0 <= delta <= block_days:
            return EventGateResult(
                blocked=True,
                reason=f"catalyst_soon_{catalyst_type or 'unknown'}",
                evidence=e
            )
        
        return EventGateResult(blocked=False, reason=None, evidence=e)
        
    except Exception:
        return EventGateResult(
            blocked=False,
            reason=None,
            evidence={"catalyst_date_parse_failed": True}
        )
