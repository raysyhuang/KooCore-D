"""
Position Tracker Module

Tracks open positions from picks and monitors for drawdown alerts.
Based on backtest analysis: Winners avg -2.2% drawdown, losers avg -5.4%.
"""

from __future__ import annotations
import math
import os
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import Optional
from dataclasses import dataclass, asdict
import pandas as pd

logger = logging.getLogger(__name__)

from ...utils.time import utc_now_iso_z


@dataclass
class Position:
    """Represents an open position from a pick with all features for learning."""
    ticker: str
    entry_date: str  # YYYY-MM-DD
    entry_price: float
    source: str  # "weekly_top5", "pro30", "movers"
    rank: Optional[int] = None  # For weekly picks
    composite_score: Optional[float] = None
    holding_days_target: int = 7
    stop_loss_pct: float = -7.0
    profit_target_pct: float = 7.0
    
    # Technical features at pick time (for learning)
    technical_score: Optional[float] = None
    rsi14: Optional[float] = None
    volume_ratio_3d_to_20d: Optional[float] = None
    dist_to_52w_high_pct: Optional[float] = None
    realized_vol_5d_ann_pct: Optional[float] = None
    above_ma10: Optional[bool] = None
    above_ma20: Optional[bool] = None
    above_ma50: Optional[bool] = None
    
    # Additional context for learning
    sector: Optional[str] = None
    market_cap_usd: Optional[float] = None
    
    # Overlap tracking (how many scanners flagged this ticker)
    overlap_count: int = 1
    overlap_sources: Optional[list] = None  # List of sources that flagged this ticker
    
    # Conviction score from adaptive scorer (if available)
    conviction_score: Optional[float] = None
    confidence: Optional[str] = None  # "HIGH", "MEDIUM", "LOW"
    
    # Current state (updated by monitor)
    current_price: Optional[float] = None
    current_return_pct: Optional[float] = None
    max_price: Optional[float] = None
    max_return_pct: Optional[float] = None
    min_price: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    last_updated: Optional[str] = None
    status: str = "open"  # "open", "stopped", "target_hit", "expired", "closed"
    days_held: int = 0
    days_to_peak: Optional[int] = None  # Days from entry to max price


class PositionTracker:
    """
    Tracks open positions and monitors for drawdown alerts.
    
    Usage:
        tracker = PositionTracker()
        tracker.add_positions_from_scan(scan_date, weekly_picks, pro30_picks)
        alerts = tracker.monitor_positions()  # Check current prices
        tracker.save()
    """
    
    POSITIONS_FILE = "data/positions/open_positions.json"
    HISTORY_FILE = "data/positions/position_history.json"
    
    def __init__(self, positions_file: Optional[str] = None):
        self.positions_file = Path(positions_file or self.POSITIONS_FILE)
        self.positions_file.parent.mkdir(parents=True, exist_ok=True)
        self.positions: dict[str, Position] = {}  # ticker -> Position
        self._load()
    
    def _load(self):
        """Load positions from file."""
        if self.positions_file.exists():
            try:
                with open(self.positions_file, "r") as f:
                    data = json.load(f)
                for ticker, pos_dict in data.get("positions", {}).items():
                    self.positions[ticker] = Position(**pos_dict)
                logger.info(f"Loaded {len(self.positions)} open positions")
            except Exception as e:
                logger.warning(f"Could not load positions: {e}")
    
    def save(self):
        """Save positions to file."""
        try:
            data = {
            "last_updated": utc_now_iso_z(),
                "positions": {ticker: asdict(pos) for ticker, pos in self.positions.items()}
            }
            with open(self.positions_file, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved {len(self.positions)} positions")
        except Exception as e:
            logger.error(f"Failed to save positions: {e}")
    
    def add_position(self, position: Position) -> bool:
        """
        Add a new position to track.
        
        Returns True if added, False if position already exists.
        """
        if position.ticker in self.positions:
            existing = self.positions[position.ticker]
            if existing.status == "open":
                logger.debug(f"Position {position.ticker} already open, skipping")
                return False
        
        self.positions[position.ticker] = position
        logger.info(f"Added position: {position.ticker} @ ${position.entry_price:.2f} ({position.source})")
        return True
    
    def add_positions_from_scan(
        self,
        scan_date: str,
        weekly_picks: list[dict],
        pro30_picks: list[str],
        movers_picks: list[str] = None,
        config: dict = None,
        features_map: dict = None,
    ) -> int:
        """
        Add positions from a scan run.
        
        Args:
            scan_date: YYYY-MM-DD
            weekly_picks: List of weekly top 5 dicts with ticker, composite_score, rank, current_price
            pro30_picks: List of Pro30 ticker strings
            movers_picks: List of movers ticker strings
            config: Config dict with risk_management settings
            features_map: Optional dict mapping ticker -> features dict for learning
                          Each features dict can contain: technical_score, rsi14,
                          volume_ratio_3d_to_20d, dist_to_52w_high_pct, realized_vol_5d_ann_pct,
                          above_ma10, above_ma20, above_ma50, sector, market_cap_usd
        
        Returns:
            Number of new positions added
        """
        config = config or {}
        features_map = features_map or {}
        risk_cfg = config.get("risk_management", {})
        stop_loss = risk_cfg.get("suggested_stop_loss_pct", -7.0)
        profit_target = risk_cfg.get("suggested_profit_target_pct", 7.0)
        holding_days = risk_cfg.get("default_holding_days", 7)
        
        added = 0
        
        # Track all tickers for overlap detection
        weekly_tickers = {p.get("ticker") for p in weekly_picks if p.get("ticker")}
        pro30_tickers = set(pro30_picks or [])
        movers_tickers = set(movers_picks or [])
        
        def compute_overlap(ticker: str, primary_source: str) -> tuple[int, list]:
            """Compute overlap count and sources for a ticker."""
            sources = [primary_source]
            if primary_source != "weekly_top5" and ticker in weekly_tickers:
                sources.append("weekly_top5")
            if primary_source != "pro30" and ticker in pro30_tickers:
                sources.append("pro30")
            if primary_source != "movers" and ticker in movers_tickers:
                sources.append("movers")
            return len(sources), sources
        
        def get_features(ticker: str) -> dict:
            """Get features for a ticker from features_map."""
            return features_map.get(ticker, {})
        
        # Add weekly/hybrid picks
        # Hybrid Top 3 picks have source_type="hybrid_top3" and sources list
        for pick in weekly_picks:
            ticker = pick.get("ticker")
            if not ticker:
                continue
            
            # Check if this is a hybrid_top3 pick (best across all models)
            is_hybrid = pick.get("source_type") == "hybrid_top3"
            
            if is_hybrid:
                # Hybrid picks: use sources from weighted ranking
                hybrid_sources = pick.get("sources", [])
                overlap_count = len(hybrid_sources)
                overlap_sources = hybrid_sources
                source = "hybrid_top3"  # Tag as hybrid pick
            else:
                # Regular weekly picks
                overlap_count, overlap_sources = compute_overlap(ticker, "weekly_top5")
                source = "weekly_top5"
            
            features = get_features(ticker)

            # Validate entry price to prevent ghost positions (Phase 6 fix)
            raw_price = pick.get("current_price")
            if raw_price is None or (isinstance(raw_price, float) and math.isnan(raw_price)) or raw_price <= 0:
                logger.warning(f"  Skipping {ticker}: invalid entry_price={raw_price}")
                continue

            pos = Position(
                ticker=ticker,
                entry_date=scan_date,
                entry_price=raw_price,
                source=source,
                rank=pick.get("rank"),
                composite_score=pick.get("composite_score") or pick.get("hybrid_score", 0),
                holding_days_target=holding_days,
                stop_loss_pct=stop_loss,
                profit_target_pct=profit_target,
                # Technical features
                technical_score=features.get("technical_score") or pick.get("technical_score"),
                rsi14=features.get("rsi14"),
                volume_ratio_3d_to_20d=features.get("volume_ratio_3d_to_20d"),
                dist_to_52w_high_pct=features.get("dist_to_52w_high_pct"),
                realized_vol_5d_ann_pct=features.get("realized_vol_5d_ann_pct"),
                above_ma10=features.get("above_ma10"),
                above_ma20=features.get("above_ma20"),
                above_ma50=features.get("above_ma50"),
                # Context
                sector=features.get("sector") or pick.get("sector"),
                market_cap_usd=features.get("market_cap_usd"),
                # Overlap
                overlap_count=overlap_count,
                overlap_sources=overlap_sources,
                # Conviction (if provided)
                conviction_score=pick.get("conviction_score"),
                confidence=pick.get("confidence"),
            )
            if self.add_position(pos):
                added += 1
        
        # Add Pro30 picks (need to fetch prices)
        for ticker in pro30_picks or []:
            overlap_count, overlap_sources = compute_overlap(ticker, "pro30")
            features = get_features(ticker)
            
            pos = Position(
                ticker=ticker,
                entry_date=scan_date,
                entry_price=0.0,  # Will be updated by monitor
                source="pro30",
                holding_days_target=holding_days,
                stop_loss_pct=stop_loss,
                profit_target_pct=profit_target,
                # Technical features
                technical_score=features.get("technical_score"),
                rsi14=features.get("rsi14"),
                volume_ratio_3d_to_20d=features.get("volume_ratio_3d_to_20d"),
                dist_to_52w_high_pct=features.get("dist_to_52w_high_pct"),
                realized_vol_5d_ann_pct=features.get("realized_vol_5d_ann_pct"),
                above_ma10=features.get("above_ma10"),
                above_ma20=features.get("above_ma20"),
                above_ma50=features.get("above_ma50"),
                # Context
                sector=features.get("sector"),
                market_cap_usd=features.get("market_cap_usd"),
                # Overlap
                overlap_count=overlap_count,
                overlap_sources=overlap_sources,
            )
            if self.add_position(pos):
                added += 1
        
        # Add movers picks
        for ticker in movers_picks or []:
            overlap_count, overlap_sources = compute_overlap(ticker, "movers")
            features = get_features(ticker)
            
            pos = Position(
                ticker=ticker,
                entry_date=scan_date,
                entry_price=0.0,  # Will be updated by monitor
                source="movers",
                holding_days_target=holding_days,
                stop_loss_pct=stop_loss,
                profit_target_pct=profit_target,
                # Technical features
                technical_score=features.get("technical_score"),
                rsi14=features.get("rsi14"),
                volume_ratio_3d_to_20d=features.get("volume_ratio_3d_to_20d"),
                dist_to_52w_high_pct=features.get("dist_to_52w_high_pct"),
                realized_vol_5d_ann_pct=features.get("realized_vol_5d_ann_pct"),
                above_ma10=features.get("above_ma10"),
                above_ma20=features.get("above_ma20"),
                above_ma50=features.get("above_ma50"),
                # Context
                sector=features.get("sector"),
                market_cap_usd=features.get("market_cap_usd"),
                # Overlap
                overlap_count=overlap_count,
                overlap_sources=overlap_sources,
            )
            if self.add_position(pos):
                added += 1
        
        # Store picks in outcome database for learning
        self._store_picks_for_learning(scan_date)
        
        logger.info(f"Added {added} new positions from {scan_date} scan")
        return added
    
    def _store_picks_for_learning(self, scan_date: str):
        """Store picks in outcome database for model learning."""
        try:
            from src.core.outcome_db import get_outcome_db, PickRecord
            
            db = get_outcome_db()
            for ticker, pos in self.positions.items():
                if pos.entry_date == scan_date and pos.status == "open":
                    pick = PickRecord(
                        ticker=pos.ticker,
                        pick_date=pos.entry_date,
                        source=pos.source,
                        entry_price=pos.entry_price,
                        rank=pos.rank,
                        technical_score=pos.technical_score,
                        rsi14=pos.rsi14,
                        volume_ratio_3d_to_20d=pos.volume_ratio_3d_to_20d,
                        dist_to_52w_high_pct=pos.dist_to_52w_high_pct,
                        realized_vol_5d_ann_pct=pos.realized_vol_5d_ann_pct,
                        above_ma10=pos.above_ma10,
                        above_ma20=pos.above_ma20,
                        above_ma50=pos.above_ma50,
                        sector=pos.sector,
                        market_cap_usd=pos.market_cap_usd,
                        composite_score=pos.composite_score,
                        overlap_count=pos.overlap_count,
                        overlap_sources=pos.overlap_sources or [],
                    )
                    db.store_pick(pick)
        except ImportError:
            logger.debug("Outcome database not available, skipping pick storage")
        except Exception as e:
            logger.warning(f"Failed to store picks for learning: {e}")
    
    def monitor_positions(self) -> list[dict]:
        """
        Monitor all open positions for drawdown/profit alerts.
        
        Returns list of alert dicts: {ticker, alert_type, message, position}
        """
        alerts = []
        today = date.today()
        
        # Fetch current prices for all open positions
        open_tickers = [t for t, p in self.positions.items() if p.status == "open"]
        if not open_tickers:
            return alerts
        
        # Get prices via yfinance
        try:
            import yfinance as yf
            price_data = yf.download(
                tickers=open_tickers,
                period="5d",
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False,
            )
        except Exception as e:
            logger.error(f"Failed to fetch prices for monitoring: {e}")
            return alerts

        if price_data is None or price_data.empty:
            logger.warning("Price monitoring skipped: empty price data response")
            return alerts
        
        now = utc_now_iso_z()
        
        for ticker, pos in self.positions.items():
            if pos.status != "open":
                continue
            
            # Get current price
            try:
                if len(open_tickers) == 1:
                    current = float(price_data["Close"].iloc[-1])
                else:
                    current = float(price_data[ticker]["Close"].iloc[-1])
            except Exception:
                logger.warning(f"Could not get price for {ticker}")
                continue
            
            # Update entry price if not set (for pro30/movers that didn't have it)
            if pos.entry_price <= 0:
                # Use first available close as entry
                try:
                    if len(open_tickers) == 1:
                        pos.entry_price = float(price_data["Close"].iloc[0])
                    else:
                        pos.entry_price = float(price_data[ticker]["Close"].iloc[0])
                except (KeyError, IndexError, TypeError):
                    pos.entry_price = current
            
            # Calculate return (with safety check for entry_price)
            pos.current_price = current
            if pos.entry_price and pos.entry_price > 0:
                pos.current_return_pct = ((current - pos.entry_price) / pos.entry_price) * 100
            else:
                pos.current_return_pct = 0.0
            pos.last_updated = now
            
            # Track max/min and days to peak
            if pos.max_price is None or current > pos.max_price:
                pos.max_price = current
                pos.max_return_pct = pos.current_return_pct
                # Update days_to_peak when we set new max
                try:
                    entry_dt = datetime.strptime(pos.entry_date, "%Y-%m-%d").date()
                    pos.days_to_peak = (today - entry_dt).days
                except (ValueError, TypeError):
                    pos.days_to_peak = 0
            if pos.min_price is None or current < pos.min_price:
                pos.min_price = current
                if pos.entry_price and pos.entry_price > 0:
                    pos.max_drawdown_pct = ((pos.min_price - pos.entry_price) / pos.entry_price) * 100
            
            # Calculate days held
            try:
                entry_dt = datetime.strptime(pos.entry_date, "%Y-%m-%d").date()
                pos.days_held = (today - entry_dt).days
            except (ValueError, TypeError):
                pos.days_held = 0
            
            # Check alerts
            alert = None
            prev_status = pos.status
            
            # Stop loss hit
            if pos.current_return_pct <= pos.stop_loss_pct:
                alert = {
                    "ticker": ticker,
                    "alert_type": "stop_loss",
                    "severity": "high",
                    "message": f"🛑 STOP LOSS: {ticker} down {pos.current_return_pct:.1f}% (limit: {pos.stop_loss_pct}%)",
                    "position": asdict(pos),
                }
                pos.status = "stopped"
            
            # Profit target hit
            elif pos.current_return_pct >= pos.profit_target_pct:
                alert = {
                    "ticker": ticker,
                    "alert_type": "profit_target",
                    "severity": "info",
                    "message": f"🎯 TARGET HIT: {ticker} up {pos.current_return_pct:.1f}% (target: {pos.profit_target_pct}%)",
                    "position": asdict(pos),
                }
                pos.status = "target_hit"
            
            # Holding period expired
            elif pos.days_held >= pos.holding_days_target:
                alert = {
                    "ticker": ticker,
                    "alert_type": "expired",
                    "severity": "medium",
                    "message": f"⏰ HOLDING EXPIRED: {ticker} at {pos.current_return_pct:.1f}% after {pos.days_held} days",
                    "position": asdict(pos),
                }
                pos.status = "expired"
            
            # Warning: approaching stop loss
            elif pos.current_return_pct <= -5.0 and pos.current_return_pct > pos.stop_loss_pct:
                alert = {
                    "ticker": ticker,
                    "alert_type": "drawdown_warning",
                    "severity": "warning",
                    "message": f"⚠️ DRAWDOWN WARNING: {ticker} down {pos.current_return_pct:.1f}% (approaching {pos.stop_loss_pct}% stop)",
                    "position": asdict(pos),
                }
            
            # Record outcome if position closed
            if prev_status == "open" and pos.status != "open":
                self._record_outcome(pos)
            
            if alert:
                alerts.append(alert)
        
        self.save()
        return alerts
    
    def get_summary(self) -> dict:
        """Get summary of current positions."""
        open_positions = [p for p in self.positions.values() if p.status == "open"]
        closed_positions = [p for p in self.positions.values() if p.status != "open"]
        
        summary = {
            "total_open": len(open_positions),
            "total_closed": len(closed_positions),
            "by_source": {},
            "by_status": {},
        }
        
        for pos in self.positions.values():
            # By source
            src = pos.source
            if src not in summary["by_source"]:
                summary["by_source"][src] = {"open": 0, "closed": 0}
            if pos.status == "open":
                summary["by_source"][src]["open"] += 1
            else:
                summary["by_source"][src]["closed"] += 1
            
            # By status
            if pos.status not in summary["by_status"]:
                summary["by_status"][pos.status] = 0
            summary["by_status"][pos.status] += 1
        
        return summary
    
    def close_position(self, ticker: str, reason: str = "manual") -> bool:
        """Manually close a position."""
        if ticker in self.positions:
            self.positions[ticker].status = f"closed_{reason}"
            # Record outcome for learning
            self._record_outcome(self.positions[ticker])
            self.save()
            return True
        return False
    
    def _record_outcome(self, pos: Position):
        """
        Record outcome for a closed position to the outcome database.
        
        Note: Retries are compute-only - outcomes are never recorded twice.
        """
        try:
            from src.core.outcome_db import get_outcome_db, OutcomeRecord
            from src.core.retry_guard import is_retry_attempt, log_retry_suppression
            
            # Retries re-run computation but MUST NOT emit side effects
            # Guard against double-counting outcomes on retry attempts
            if is_retry_attempt():
                log_retry_suppression("outcome persistence", ticker=pos.ticker)
                return
            
            # Validate entry_price before recording
            if pos.entry_price is None or pos.entry_price <= 0:
                logger.warning(f"Skipping outcome for {pos.ticker}: invalid entry_price ({pos.entry_price})")
                return
            
            # Calculate hit thresholds
            max_ret = pos.max_return_pct or 0
            hit_5 = max_ret >= 5.0
            hit_7 = max_ret >= 7.0
            hit_10 = max_ret >= 10.0
            hit_15 = max_ret >= 15.0
            
            # Calculate final return
            final_return = None
            if pos.current_price and pos.entry_price and pos.entry_price > 0:
                final_return = ((pos.current_price - pos.entry_price) / pos.entry_price) * 100
            
            # Extract exit reason from status
            exit_reason = pos.status
            if pos.status.startswith("closed_"):
                exit_reason = pos.status[7:]  # Remove "closed_" prefix
            
            outcome = OutcomeRecord(
                ticker=pos.ticker,
                pick_date=pos.entry_date,
                source=pos.source,
                entry_price=pos.entry_price,
                rank=pos.rank,
                technical_score=pos.technical_score,
                rsi14=pos.rsi14,
                volume_ratio_3d_to_20d=pos.volume_ratio_3d_to_20d,
                dist_to_52w_high_pct=pos.dist_to_52w_high_pct,
                realized_vol_5d_ann_pct=pos.realized_vol_5d_ann_pct,
                above_ma10=pos.above_ma10,
                above_ma20=pos.above_ma20,
                above_ma50=pos.above_ma50,
                sector=pos.sector,
                market_cap_usd=pos.market_cap_usd,
                composite_score=pos.composite_score,
                overlap_count=pos.overlap_count,
                overlap_sources=pos.overlap_sources or [],
                exit_date=date.today().strftime("%Y-%m-%d"),
                exit_price=pos.current_price,
                exit_reason=exit_reason,
                max_price=pos.max_price,
                max_return_pct=pos.max_return_pct,
                min_price=pos.min_price,
                max_drawdown_pct=pos.max_drawdown_pct,
                final_return_pct=final_return,
                days_held=pos.days_held,
                days_to_peak=pos.days_to_peak,
                hit_5pct=hit_5,
                hit_7pct=hit_7,
                hit_10pct=hit_10,
                hit_15pct=hit_15,
            )
            
            db = get_outcome_db()
            if db.record_outcome(outcome):
                logger.info(f"Recorded outcome for {pos.ticker}: hit_7pct={hit_7}, max_return={max_ret:.1f}%")
            else:
                logger.warning(f"Failed to record outcome for {pos.ticker} (db returned False)")
            
        except ImportError:
            logger.debug("Outcome database not available, skipping outcome recording")
        except Exception as e:
            logger.warning(f"Failed to record outcome for {pos.ticker}: {e}")
    
    def clear_closed(self) -> int:
        """Remove closed positions (archive to history file)."""
        closed = {t: p for t, p in self.positions.items() if p.status != "open"}
        
        if closed:
            # Archive to history
            history_file = Path(self.HISTORY_FILE)
            history_file.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                if history_file.exists():
                    with open(history_file, "r") as f:
                        history = json.load(f)
                else:
                    history = {"archived_positions": []}
                
                for pos in closed.values():
                    history["archived_positions"].append({
                        "archived_at": utc_now_iso_z(),
                        **asdict(pos)
                    })
                
                with open(history_file, "w") as f:
                    json.dump(history, f, indent=2)
            except Exception as e:
                logger.warning(f"Could not archive to history: {e}")
            
            # Remove from active positions
            for t in closed:
                del self.positions[t]
            
            self.save()
        
        return len(closed)


def send_position_alerts(alerts: list[dict], config: dict = None):
    """Send position alerts via configured channels."""
    if not alerts:
        return
    
    try:
        from src.core.alerts import AlertConfig, AlertManager
        
        # Build config
        alert_config = AlertConfig()
        if config:
            alerts_cfg = config.get("alerts", {})
            alert_config.enabled = alerts_cfg.get("enabled", False)
            alert_config.channels = alerts_cfg.get("channels", [])
        
        if not alert_config.enabled:
            logger.info("Position alerts disabled, skipping")
            return
        
        manager = AlertManager(alert_config)
        
        # Group by severity
        high = [a for a in alerts if a.get("severity") == "high"]
        warning = [a for a in alerts if a.get("severity") == "warning"]
        info = [a for a in alerts if a.get("severity") == "info"]
        
        # Send high priority alerts individually
        for alert in high:
            manager.send_alert(
                title=f"Position Alert: {alert['ticker']}",
                message=alert["message"],
                data=alert.get("position"),
                priority="high"
            )
        
        # Send warning alerts as batch
        if warning:
            msg = "Position Warnings:\n" + "\n".join(a["message"] for a in warning)
            manager.send_alert(
                title="Position Drawdown Warnings",
                message=msg,
                priority="normal"
            )
        
        # Send info alerts as batch
        if info:
            msg = "Position Updates:\n" + "\n".join(a["message"] for a in info)
            manager.send_alert(
                title="Position Profit Targets",
                message=msg,
                priority="low"
            )
        
        logger.info(f"Sent {len(alerts)} position alerts")
        
    except Exception as e:
        logger.error(f"Failed to send position alerts: {e}")
