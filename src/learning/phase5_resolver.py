"""
Phase-5 Outcome Resolver

Resolves outcomes for Phase-5 learning rows after the holding period.

Resolution rule:
- A row becomes "resolvable" when scan_date + holding_days has passed (trading days)
- Default horizon: 7 trading days

Outcome fields computed:
- return_7d: actual return over holding period
- max_drawdown_7d: worst drawdown during holding
- max_gain_7d: best gain during holding
- days_to_target: how many days to hit +7% (if hit)
- outcome_7d: hit | miss | neutral
- exit_reason: target_hit | stop | expiry
"""

from __future__ import annotations

import logging
from datetime import datetime, date, timedelta
from typing import Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class Phase5Resolver:
    """
    Resolves outcomes for Phase-5 learning rows.
    """
    
    # Default thresholds
    DEFAULT_TARGET_PCT = 7.0    # +7% = hit
    DEFAULT_STOP_PCT = -10.0   # -10% = stop
    DEFAULT_HOLDING_DAYS = 7   # Trading days
    
    def __init__(
        self,
        target_pct: float = DEFAULT_TARGET_PCT,
        stop_pct: float = DEFAULT_STOP_PCT,
        holding_days: int = DEFAULT_HOLDING_DAYS,
    ):
        self.target_pct = target_pct
        self.stop_pct = stop_pct
        self.holding_days = holding_days
    
    def resolve_outcomes(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        dry_run: bool = False,
    ) -> dict:
        """
        Resolve outcomes for Phase-5 rows in date range.
        
        Returns dict with counts: resolved, skipped, errors, not_ready
        """
        from src.learning.phase5_store import get_phase5_store
        
        store = get_phase5_store()
        result = {
            "resolved": 0,
            "skipped": 0,
            "errors": 0,
            "not_ready": 0,
            "dry_run": dry_run,
        }
        
        # Load all rows
        all_rows = store.load_rows()
        if not all_rows:
            logger.info("No Phase-5 rows to resolve")
            return result
        
        # Load existing outcomes (to avoid re-resolving)
        existing_outcomes = store.load_outcomes()
        existing_keys = set()
        for outcome in existing_outcomes:
            key = (outcome["scan_date"], outcome["ticker"], outcome["primary_strategy"])
            existing_keys.add(key)
        
        # Filter by date range
        if start_date:
            all_rows = [r for r in all_rows if r.identity.scan_date >= start_date]
        if end_date:
            all_rows = [r for r in all_rows if r.identity.scan_date <= end_date]
        
        # Group by scan_date for batch resolution
        rows_by_date = {}
        for row in all_rows:
            key = row.get_key()
            if key in existing_keys:
                result["skipped"] += 1
                continue
            
            scan_date = row.identity.scan_date
            if scan_date not in rows_by_date:
                rows_by_date[scan_date] = []
            rows_by_date[scan_date].append(row)
        
        logger.info(f"Found {len(all_rows) - result['skipped']} rows to potentially resolve across {len(rows_by_date)} dates")
        
        # Resolve each date
        today = date.today()
        
        for scan_date, rows in sorted(rows_by_date.items()):
            # Check if enough time has passed
            scan_dt = datetime.strptime(scan_date, "%Y-%m-%d").date()
            # Add buffer for trading days (roughly 1.5x calendar days)
            min_calendar_days = int(self.holding_days * 1.5)
            
            if (today - scan_dt).days < min_calendar_days:
                result["not_ready"] += len(rows)
                logger.debug(f"Scan date {scan_date} not yet resolvable (need {min_calendar_days} days)")
                continue
            
            # Resolve each row
            outcomes_to_write = []
            
            for row in rows:
                try:
                    outcome = self._resolve_single(row, scan_dt)
                    if outcome:
                        outcomes_to_write.append(outcome)
                        result["resolved"] += 1
                    else:
                        result["errors"] += 1
                except Exception as e:
                    logger.warning(f"Error resolving {row.identity.ticker}: {e}")
                    result["errors"] += 1
            
            # Write outcomes for this date
            if outcomes_to_write and not dry_run:
                write_result = store.write_outcomes(outcomes_to_write, scan_date)
                logger.info(f"Resolved {write_result['written']} outcomes for {scan_date}")
        
        return result
    
    def _resolve_single(self, row, scan_dt: date) -> Optional[dict]:
        """Resolve outcome for a single row."""
        from src.core.price_db import PriceDatabase
        
        ticker = row.identity.ticker
        
        # Get prices for holding period
        # Start from day after scan (entry next day at open)
        entry_date = scan_dt + timedelta(days=1)
        # Add extra days to ensure we have enough trading days
        end_date = entry_date + timedelta(days=self.holding_days * 2)
        
        db = PriceDatabase()
        prices = db.get_prices(ticker, entry_date, end_date)
        
        if prices.empty or len(prices) < 2:
            logger.debug(f"Insufficient price data for {ticker}")
            return None
        
        # Take first holding_days trading days
        prices = prices.head(self.holding_days + 1)
        
        if len(prices) < 2:
            return None
        
        # Entry price = open on first trading day after scan
        entry_price = float(prices.iloc[0]["Open"])
        
        if entry_price <= 0:
            logger.warning(f"Invalid entry price for {ticker}: {entry_price}")
            return None
        
        # Calculate returns for each day
        daily_returns = []
        for i, (dt, row_data) in enumerate(prices.iterrows()):
            if i == 0:
                continue  # Skip entry day for return calc
            
            # Use close for return
            close = float(row_data["Close"])
            high = float(row_data["High"])
            low = float(row_data["Low"])
            
            ret = (close - entry_price) / entry_price * 100
            max_intraday = (high - entry_price) / entry_price * 100
            min_intraday = (low - entry_price) / entry_price * 100
            
            daily_returns.append({
                "day": i,
                "date": dt,
                "close_return": ret,
                "max_return": max_intraday,
                "min_return": min_intraday,
            })
        
        if not daily_returns:
            return None
        
        # Calculate metrics
        final_return = daily_returns[-1]["close_return"]
        max_gain = max(d["max_return"] for d in daily_returns)
        max_drawdown = min(d["min_return"] for d in daily_returns)
        
        # Determine outcome
        days_to_target = None
        exit_reason = "expiry"
        outcome = "neutral"
        
        # Check if target was hit
        for d in daily_returns:
            if d["max_return"] >= self.target_pct:
                days_to_target = d["day"]
                exit_reason = "target_hit"
                outcome = "hit"
                break
            if d["min_return"] <= self.stop_pct:
                exit_reason = "stop"
                outcome = "miss"
                break
        
        # If no target/stop, determine by final return
        if exit_reason == "expiry":
            if final_return >= self.target_pct * 0.5:  # At least half target
                outcome = "hit"
            elif final_return <= self.stop_pct * 0.5:  # Lost half stop level
                outcome = "miss"
            else:
                outcome = "neutral"
        
        # Build outcome dict
        exit_price = float(prices.iloc[-1]["Close"])
        resolved_date = prices.index[-1].strftime("%Y-%m-%d")
        
        return {
            "scan_date": row.identity.scan_date,
            "ticker": row.identity.ticker,
            "primary_strategy": row.identity.primary_strategy,
            "outcome_7d": outcome,
            "return_7d": round(final_return, 4),
            "max_gain_7d": round(max_gain, 4),
            "max_drawdown_7d": round(max_drawdown, 4),
            "days_to_target": days_to_target,
            "exit_reason": exit_reason,
            "entry_price": round(entry_price, 4),
            "exit_price": round(exit_price, 4),
            "resolved_date": resolved_date,
        }
    
    def get_resolvable_dates(self) -> List[str]:
        """Get list of scan dates that are ready for resolution."""
        from src.learning.phase5_store import get_phase5_store
        
        store = get_phase5_store()
        all_rows = store.load_rows()
        
        # Get existing outcome dates
        existing_outcomes = store.load_outcomes()
        resolved_keys = set()
        for outcome in existing_outcomes:
            key = (outcome["scan_date"], outcome["ticker"], outcome["primary_strategy"])
            resolved_keys.add(key)
        
        today = date.today()
        min_calendar_days = int(self.holding_days * 1.5)
        
        resolvable_dates = set()
        
        for row in all_rows:
            key = row.get_key()
            if key in resolved_keys:
                continue
            
            scan_dt = datetime.strptime(row.identity.scan_date, "%Y-%m-%d").date()
            if (today - scan_dt).days >= min_calendar_days:
                resolvable_dates.add(row.identity.scan_date)
        
        return sorted(resolvable_dates)


# =============================================================================
# Singleton Access
# =============================================================================

_resolver_instance: Optional[Phase5Resolver] = None


def get_phase5_resolver(**kwargs) -> Phase5Resolver:
    """Get or create Phase5Resolver singleton."""
    global _resolver_instance
    if _resolver_instance is None:
        _resolver_instance = Phase5Resolver(**kwargs)
    return _resolver_instance
