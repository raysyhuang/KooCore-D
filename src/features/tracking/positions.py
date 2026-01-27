"""
Position Tracking Module

Track actual trades and compare to system predictions.
Provides performance analytics and trade journaling.
"""

from __future__ import annotations
import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass, field, asdict
import pandas as pd

logger = logging.getLogger(__name__)

from ...utils.time import utc_now, utc_now_timestamp


@dataclass
class Position:
    """
    Represents a single trading position.
    
    OPTIMIZED Jan 2026: Added trailing stop support
    """
    
    # Identification
    ticker: str
    position_id: str = ""
    
    # Entry
    entry_date: str = ""  # YYYY-MM-DD
    entry_price: float = 0.0
    shares: int = 0
    entry_reason: str = ""
    
    # Source tracking
    source: Literal["weekly", "pro30", "movers", "manual"] = "manual"
    predicted_rank: Optional[int] = None
    predicted_score: Optional[float] = None
    
    # Exit (filled when closed)
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    
    # Status
    status: Literal["open", "closed", "cancelled"] = "open"
    
    # Calculated fields
    pnl_dollars: Optional[float] = None
    pnl_percent: Optional[float] = None
    holding_days: Optional[int] = None
    hit_10pct: Optional[bool] = None
    
    # Trailing stop fields (NEW - Jan 2026)
    high_water_mark: Optional[float] = None  # Highest price since entry
    high_water_date: Optional[str] = None    # Date of high water mark
    trailing_stop_active: bool = False        # Whether trailing stop is triggered
    trailing_stop_price: Optional[float] = None  # Current trailing stop level
    max_unrealized_pnl_pct: Optional[float] = None  # Peak unrealized gain
    
    # Notes
    notes: str = ""
    tags: list[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.position_id:
            self.position_id = f"{self.ticker}_{self.entry_date}_{utc_now_timestamp():.0f}"
        # Initialize high water mark to entry price
        if self.high_water_mark is None and self.entry_price > 0:
            self.high_water_mark = self.entry_price
            self.high_water_date = self.entry_date
    
    def update_trailing_stop(
        self,
        current_price: float,
        current_date: str,
        trigger_pct: float = 5.0,
        trail_distance_pct: float = 3.0,
    ) -> dict:
        """
        Update trailing stop based on current price.
        
        OPTIMIZED Jan 2026: Trailing stop logic based on backtest data
        - Avg max return = 8.8% but avg exit = 3.2%
        - Trailing stop helps capture more upside
        
        Args:
            current_price: Current stock price
            current_date: Current date (YYYY-MM-DD)
            trigger_pct: Gain % to activate trailing stop (default 5%)
            trail_distance_pct: Distance to trail from high (default 3%)
        
        Returns:
            Dict with trailing stop status
        """
        if self.entry_price <= 0:
            return {"status": "error", "reason": "Invalid entry price"}
        
        # Update high water mark if new high
        if current_price > (self.high_water_mark or 0):
            self.high_water_mark = current_price
            self.high_water_date = current_date
            
            # Update max unrealized PnL
            gain_pct = ((current_price / self.entry_price) - 1) * 100
            self.max_unrealized_pnl_pct = gain_pct
            
            # Check if trailing stop should be activated
            if gain_pct >= trigger_pct and not self.trailing_stop_active:
                self.trailing_stop_active = True
        
        # Calculate current trailing stop price if active
        result = {
            "current_price": current_price,
            "high_water_mark": self.high_water_mark,
            "gain_from_entry_pct": ((current_price / self.entry_price) - 1) * 100,
            "trailing_stop_active": self.trailing_stop_active,
        }
        
        if self.trailing_stop_active and self.high_water_mark:
            self.trailing_stop_price = self.high_water_mark * (1 - trail_distance_pct / 100)
            result["trailing_stop_price"] = self.trailing_stop_price
            result["should_exit"] = current_price <= self.trailing_stop_price
            result["distance_to_stop_pct"] = ((current_price / self.trailing_stop_price) - 1) * 100
        else:
            result["trailing_stop_price"] = None
            result["should_exit"] = False
        
        return result
    
    def close(
        self,
        exit_date: str,
        exit_price: float,
        exit_reason: str = "",
    ) -> None:
        """Close the position and calculate PnL."""
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.exit_reason = exit_reason
        self.status = "closed"
        
        # Calculate PnL
        if self.entry_price > 0 and self.shares > 0:
            self.pnl_dollars = (exit_price - self.entry_price) * self.shares
            self.pnl_percent = ((exit_price / self.entry_price) - 1) * 100
        
        # Calculate holding days
        if self.entry_date and self.exit_date:
            entry_dt = datetime.strptime(self.entry_date, "%Y-%m-%d")
            exit_dt = datetime.strptime(self.exit_date, "%Y-%m-%d")
            self.holding_days = (exit_dt - entry_dt).days
        
        # Check if hit 10% target
        if self.pnl_percent is not None:
            self.hit_10pct = self.pnl_percent >= 10.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class TradeLog:
    """Log entry for trade journal."""
    
    timestamp: str
    action: Literal["entry", "exit", "update", "note"]
    position_id: str
    ticker: str
    details: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return asdict(self)


class PositionTracker:
    """
    Tracks trading positions and calculates performance metrics.
    
    Features:
    - Track entries and exits
    - Compare predictions to actual outcomes
    - Calculate hit rates and PnL
    - Export to CSV for analysis
    """
    
    def __init__(self, data_path: str = "data/positions.json"):
        self.data_path = Path(data_path)
        self.positions: dict[str, Position] = {}
        self.trade_log: list[TradeLog] = []
        self._load()
    
    def _load(self) -> None:
        """Load positions from file."""
        if not self.data_path.exists():
            return
        
        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for pos_data in data.get("positions", []):
                pos = Position(**pos_data)
                self.positions[pos.position_id] = pos
            
            for log_data in data.get("trade_log", []):
                self.trade_log.append(TradeLog(**log_data))
                
        except Exception as e:
            logger.error(f"Failed to load positions: {e}")
    
    def save(self) -> None:
        """Save positions to file."""
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "positions": [pos.to_dict() for pos in self.positions.values()],
            "trade_log": [log.to_dict() for log in self.trade_log],
            "updated_at": utc_now().isoformat().replace("+00:00", "")
        }
        
        with open(self.data_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
    
    def log_entry(
        self,
        ticker: str,
        entry_date: str,
        entry_price: float,
        shares: int,
        source: str = "manual",
        predicted_rank: Optional[int] = None,
        predicted_score: Optional[float] = None,
        entry_reason: str = "",
        notes: str = "",
        tags: Optional[list[str]] = None,
    ) -> Position:
        """
        Log a new position entry.
        
        Args:
            ticker: Stock ticker symbol
            entry_date: Entry date (YYYY-MM-DD)
            entry_price: Entry price per share
            shares: Number of shares
            source: Signal source (weekly, pro30, movers, manual)
            predicted_rank: Rank from scanner (if applicable)
            predicted_score: Score from scanner (if applicable)
            entry_reason: Why the trade was taken
            notes: Additional notes
            tags: Tags for categorization
        
        Returns:
            Created Position object
        """
        position = Position(
            ticker=ticker.upper(),
            entry_date=entry_date,
            entry_price=entry_price,
            shares=shares,
            source=source,
            predicted_rank=predicted_rank,
            predicted_score=predicted_score,
            entry_reason=entry_reason,
            notes=notes,
            tags=tags or [],
        )
        
        self.positions[position.position_id] = position
        
        # Add to trade log
        self.trade_log.append(TradeLog(
            timestamp=utc_now().isoformat().replace("+00:00", ""),
            action="entry",
            position_id=position.position_id,
            ticker=ticker.upper(),
            details={
                "entry_price": entry_price,
                "shares": shares,
                "source": source,
            }
        ))
        
        self.save()
        logger.info(f"Logged entry: {ticker} @ ${entry_price:.2f} x {shares} shares")
        
        return position
    
    def log_exit(
        self,
        position_id: str,
        exit_date: str,
        exit_price: float,
        exit_reason: str = "",
    ) -> Optional[Position]:
        """
        Log a position exit.
        
        Args:
            position_id: Position ID to close
            exit_date: Exit date (YYYY-MM-DD)
            exit_price: Exit price per share
            exit_reason: Why the trade was exited
        
        Returns:
            Updated Position object or None if not found
        """
        position = self.positions.get(position_id)
        if not position:
            logger.warning(f"Position not found: {position_id}")
            return None
        
        position.close(exit_date, exit_price, exit_reason)
        
        # Add to trade log
        self.trade_log.append(TradeLog(
            timestamp=utc_now().isoformat().replace("+00:00", ""),
            action="exit",
            position_id=position_id,
            ticker=position.ticker,
            details={
                "exit_price": exit_price,
                "pnl_dollars": position.pnl_dollars,
                "pnl_percent": position.pnl_percent,
                "exit_reason": exit_reason,
            }
        ))
        
        self.save()
        
        pnl_str = f"${position.pnl_dollars:+.2f}" if position.pnl_dollars else "N/A"
        pct_str = f"{position.pnl_percent:+.2f}%" if position.pnl_percent else "N/A"
        logger.info(f"Logged exit: {position.ticker} @ ${exit_price:.2f} | PnL: {pnl_str} ({pct_str})")
        
        return position
    
    def get_open_positions(self) -> list[Position]:
        """Get all open positions."""
        return [p for p in self.positions.values() if p.status == "open"]
    
    def get_closed_positions(self) -> list[Position]:
        """Get all closed positions."""
        return [p for p in self.positions.values() if p.status == "closed"]
    
    def get_positions_by_source(self, source: str) -> list[Position]:
        """Get positions by signal source."""
        return [p for p in self.positions.values() if p.source == source]
    
    def get_positions_by_ticker(self, ticker: str) -> list[Position]:
        """Get all positions for a ticker."""
        ticker = ticker.upper()
        return [p for p in self.positions.values() if p.ticker == ticker]
    
    def update_trailing_stops(
        self,
        prices: dict[str, float],
        current_date: str,
        trigger_pct: float = 5.0,
        trail_distance_pct: float = 3.0,
    ) -> list[dict]:
        """
        Update trailing stops for all open positions.
        
        OPTIMIZED Jan 2026: Batch trailing stop update
        
        Args:
            prices: Dict of ticker -> current price
            current_date: Current date (YYYY-MM-DD)
            trigger_pct: Gain % to activate trailing stop
            trail_distance_pct: Distance to trail from high
        
        Returns:
            List of dicts with positions that should exit
        """
        exit_signals = []
        
        for position in self.get_open_positions():
            if position.ticker not in prices:
                continue
            
            current_price = prices[position.ticker]
            result = position.update_trailing_stop(
                current_price=current_price,
                current_date=current_date,
                trigger_pct=trigger_pct,
                trail_distance_pct=trail_distance_pct,
            )
            
            if result.get("should_exit"):
                exit_signals.append({
                    "position_id": position.position_id,
                    "ticker": position.ticker,
                    "entry_price": position.entry_price,
                    "current_price": current_price,
                    "trailing_stop_price": result.get("trailing_stop_price"),
                    "high_water_mark": position.high_water_mark,
                    "max_gain_pct": position.max_unrealized_pnl_pct,
                    "current_gain_pct": result.get("gain_from_entry_pct"),
                    "reason": "trailing_stop_hit",
                })
        
        # Save updated positions
        self.save()
        
        return exit_signals
    
    def get_stop_levels(self) -> list[dict]:
        """
        Get current stop levels for all open positions.
        
        Returns source-specific stop levels based on config.
        """
        # Source-specific stops based on backtest data
        source_stops = {
            "pro30": -7.0,   # Wider stop (higher vol, higher hit rate)
            "weekly": -6.0,  # Standard stop
            "movers": -5.0,  # Tighter stop (worst performance)
            "manual": -6.0,  # Default
        }
        
        stop_levels = []
        for position in self.get_open_positions():
            stop_pct = source_stops.get(position.source, -6.0)
            stop_price = position.entry_price * (1 + stop_pct / 100)
            
            stop_levels.append({
                "position_id": position.position_id,
                "ticker": position.ticker,
                "source": position.source,
                "entry_price": position.entry_price,
                "stop_pct": stop_pct,
                "stop_price": stop_price,
                "high_water_mark": position.high_water_mark,
                "trailing_stop_active": position.trailing_stop_active,
                "trailing_stop_price": position.trailing_stop_price,
            })
        
        return stop_levels
    
    def compute_performance_metrics(self) -> dict:
        """
        Compute overall performance metrics.
        
        Returns:
            Dict with performance statistics
        """
        closed = self.get_closed_positions()
        
        if not closed:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "hit_10pct_rate": 0.0,
                "total_pnl_dollars": 0.0,
                "avg_pnl_percent": 0.0,
                "avg_holding_days": 0.0,
            }
        
        winning = [p for p in closed if (p.pnl_percent or 0) > 0]
        losing = [p for p in closed if (p.pnl_percent or 0) <= 0]
        hit_10 = [p for p in closed if p.hit_10pct]
        
        total_pnl = sum(p.pnl_dollars or 0 for p in closed)
        avg_pnl = sum(p.pnl_percent or 0 for p in closed) / len(closed)
        avg_days = sum(p.holding_days or 0 for p in closed) / len(closed)
        
        return {
            "total_trades": len(closed),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": len(winning) / len(closed) * 100,
            "hit_10pct_rate": len(hit_10) / len(closed) * 100,
            "total_pnl_dollars": total_pnl,
            "avg_pnl_percent": avg_pnl,
            "avg_holding_days": avg_days,
            "best_trade_pct": max((p.pnl_percent or 0) for p in closed),
            "worst_trade_pct": min((p.pnl_percent or 0) for p in closed),
        }
    
    def compute_performance_by_source(self) -> pd.DataFrame:
        """Compute performance metrics grouped by signal source."""
        closed = self.get_closed_positions()
        
        if not closed:
            return pd.DataFrame()
        
        rows = []
        for source in ["weekly", "pro30", "movers", "manual"]:
            positions = [p for p in closed if p.source == source]
            if not positions:
                continue
            
            winning = [p for p in positions if (p.pnl_percent or 0) > 0]
            hit_10 = [p for p in positions if p.hit_10pct]
            
            rows.append({
                "source": source,
                "total_trades": len(positions),
                "win_rate": len(winning) / len(positions) * 100,
                "hit_10pct_rate": len(hit_10) / len(positions) * 100,
                "avg_pnl_pct": sum(p.pnl_percent or 0 for p in positions) / len(positions),
                "total_pnl": sum(p.pnl_dollars or 0 for p in positions),
            })
        
        return pd.DataFrame(rows)
    
    def compare_predicted_vs_actual(self) -> pd.DataFrame:
        """
        Compare predicted rankings/scores to actual outcomes.
        
        Returns:
            DataFrame with prediction accuracy analysis
        """
        closed = self.get_closed_positions()
        positions_with_predictions = [
            p for p in closed 
            if p.predicted_rank is not None or p.predicted_score is not None
        ]
        
        if not positions_with_predictions:
            return pd.DataFrame()
        
        rows = []
        for p in positions_with_predictions:
            rows.append({
                "ticker": p.ticker,
                "entry_date": p.entry_date,
                "source": p.source,
                "predicted_rank": p.predicted_rank,
                "predicted_score": p.predicted_score,
                "actual_pnl_pct": p.pnl_percent,
                "hit_10pct": p.hit_10pct,
                "holding_days": p.holding_days,
            })
        
        df = pd.DataFrame(rows)
        
        return df
    
    def export_to_csv(self, output_path: str) -> str:
        """Export all positions to CSV."""
        positions_data = [pos.to_dict() for pos in self.positions.values()]
        df = pd.DataFrame(positions_data)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(df)} positions to {output_path}")
        
        return str(output_path)
    
    def get_summary(self) -> str:
        """Get text summary of portfolio."""
        metrics = self.compute_performance_metrics()
        open_positions = self.get_open_positions()
        
        lines = [
            "=" * 50,
            "POSITION TRACKER SUMMARY",
            "=" * 50,
            f"\nOpen Positions: {len(open_positions)}",
        ]
        
        for pos in open_positions:
            lines.append(f"  • {pos.ticker}: {pos.shares} shares @ ${pos.entry_price:.2f}")
        
        lines.extend([
            f"\nClosed Trades: {metrics['total_trades']}",
            f"Win Rate: {metrics['win_rate']:.1f}%",
            f"Hit 10% Rate: {metrics['hit_10pct_rate']:.1f}%",
            f"Total P&L: ${metrics['total_pnl_dollars']:,.2f}",
            f"Avg P&L: {metrics['avg_pnl_percent']:.2f}%",
            f"Avg Holding: {metrics['avg_holding_days']:.1f} days",
            "=" * 50,
        ])
        
        return "\n".join(lines)


# Convenience functions

def load_tracker(data_path: str = "data/positions.json") -> PositionTracker:
    """Load or create position tracker."""
    return PositionTracker(data_path)


def save_tracker(tracker: PositionTracker) -> None:
    """Save position tracker."""
    tracker.save()
