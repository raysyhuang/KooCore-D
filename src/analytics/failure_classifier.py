"""
Failure Classification System

Analyzes closed positions to understand WHY they failed.
This enables pattern detection and automatic adjustment.

Failure reasons are categorized into:
- Execution failures (gave back gains, immediate reversal, slow bleed)
- Catalyst failures (missed, priced in, negative)
- Market failures (regime shift, sector rotation)
- Setup failures (crowded trade, false breakout, dead cat bounce)
"""

from __future__ import annotations
import logging
from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import pandas as pd

logger = logging.getLogger(__name__)


class FailureReason(Enum):
    """Enumeration of failure categories."""
    
    # Execution failures
    GAVE_BACK_GAINS = "gave_back_gains"        # Was up >5%, ended negative
    IMMEDIATE_REVERSAL = "immediate_reversal"  # Peaked Day 1, then fell
    SLOW_BLEED = "slow_bleed"                  # Never got traction, slow decline
    
    # Catalyst failures
    CATALYST_MISSED = "catalyst_missed"        # Expected catalyst didn't happen
    CATALYST_PRICED_IN = "catalyst_priced_in"  # Catalyst happened, no move
    CATALYST_NEGATIVE = "catalyst_negative"    # Catalyst was bad news
    
    # Market failures
    REGIME_SHIFT = "regime_shift"              # VIX spiked, market sold off
    SECTOR_ROTATION = "sector_rotation"        # Sector fell while market rose
    CORRELATION_BREAK = "correlation_break"    # Stock decoupled from expected behavior
    
    # Setup failures
    CROWDED_TRADE = "crowded_trade"            # Everyone had same idea
    FALSE_BREAKOUT = "false_breakout"          # Breakout reversed
    DEAD_CAT_BOUNCE = "dead_cat_bounce"        # Reversal didn't hold
    
    # Success (not a failure)
    SUCCESS = "success"
    
    # Unknown
    UNKNOWN = "unknown"


@dataclass
class FailureAnalysis:
    """
    Complete failure analysis for a position.
    """
    reason: FailureReason
    confidence: float  # 0-1, how confident we are in this classification
    notes: str
    contributing_factors: List[str]
    
    # Metrics that led to this classification
    metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "reason": self.reason.value,
            "confidence": self.confidence,
            "notes": self.notes,
            "contributing_factors": self.contributing_factors,
            "metrics": self.metrics,
        }


def classify_failure(
    entry_price: float,
    exit_price: float,
    max_price: float,
    min_price: float,
    days_held: int,
    days_to_peak: Optional[int] = None,
    vix_at_entry: Optional[float] = None,
    vix_at_exit: Optional[float] = None,
    spy_return: Optional[float] = None,
    sector_return: Optional[float] = None,
    setup_type: Optional[str] = None,  # "breakout", "reversal", "momentum"
    hit_target: bool = False,
) -> FailureAnalysis:
    """
    Classify why a position failed (or succeeded).
    
    Args:
        entry_price: Price at entry
        exit_price: Price at exit
        max_price: Maximum price during hold period
        min_price: Minimum price during hold period
        days_held: Number of days position was held
        days_to_peak: Days from entry to max price (optional)
        vix_at_entry: VIX level at entry (optional)
        vix_at_exit: VIX level at exit (optional)
        spy_return: SPY return during hold period (optional)
        sector_return: Sector ETF return during hold (optional)
        setup_type: Type of setup ("breakout", "reversal", "momentum")
        hit_target: Whether position hit the target (7% or 10%)
    
    Returns:
        FailureAnalysis with reason, confidence, and notes
    """
    # Calculate key metrics
    final_return = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
    max_return = (max_price - entry_price) / entry_price if entry_price > 0 else 0
    max_drawdown = (min_price - entry_price) / entry_price if entry_price > 0 else 0
    gave_back = max_return - final_return if max_return > 0 else 0
    
    metrics = {
        "final_return_pct": round(final_return * 100, 2),
        "max_return_pct": round(max_return * 100, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "gave_back_pct": round(gave_back * 100, 2),
        "days_held": days_held,
        "days_to_peak": days_to_peak,
    }
    
    factors = []
    
    # SUCCESS: Hit target
    if hit_target or final_return >= 0.07:
        return FailureAnalysis(
            reason=FailureReason.SUCCESS,
            confidence=1.0,
            notes=f"Position succeeded with {final_return*100:.1f}% return",
            contributing_factors=["Hit target"],
            metrics=metrics,
        )
    
    # Not a loss (small gain or breakeven)
    if final_return >= -0.01:
        return FailureAnalysis(
            reason=FailureReason.SUCCESS,
            confidence=0.7,
            notes=f"Position closed near breakeven ({final_return*100:.1f}%)",
            contributing_factors=["Did not lose money"],
            metrics=metrics,
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FAILURE CLASSIFICATION RULES (ordered by specificity)
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Rule 1: GAVE_BACK_GAINS - Was up significantly, ended negative
    if max_return >= 0.05 and final_return < 0:
        factors.append(f"Was up {max_return*100:.1f}%, ended {final_return*100:.1f}%")
        if days_to_peak is not None:
            factors.append(f"Peaked on day {days_to_peak}")
        factors.append(f"Gave back {gave_back*100:.1f}%")
        
        return FailureAnalysis(
            reason=FailureReason.GAVE_BACK_GAINS,
            confidence=0.9,
            notes=f"Position was profitable but gave back {gave_back*100:.1f}% to end negative",
            contributing_factors=factors,
            metrics=metrics,
        )
    
    # Rule 2: IMMEDIATE_REVERSAL - Peaked on day 1, then fell
    if days_to_peak is not None and days_to_peak <= 1 and final_return < -0.03:
        factors.append("Peaked immediately after entry")
        factors.append(f"Then fell {abs(final_return)*100:.1f}%")
        
        return FailureAnalysis(
            reason=FailureReason.IMMEDIATE_REVERSAL,
            confidence=0.85,
            notes="Entry was at or near the top - position reversed immediately",
            contributing_factors=factors,
            metrics=metrics,
        )
    
    # Rule 3: REGIME_SHIFT - VIX spiked and market sold off
    if vix_at_entry is not None and vix_at_exit is not None:
        vix_change = vix_at_exit - vix_at_entry
        if vix_change > 5 and (spy_return is not None and spy_return < -0.02):
            factors.append(f"VIX jumped {vix_change:.1f} points")
            factors.append(f"SPY fell {spy_return*100:.1f}%")
            metrics["vix_change"] = round(vix_change, 1)
            metrics["spy_return_pct"] = round(spy_return * 100, 2) if spy_return else None
            
            return FailureAnalysis(
                reason=FailureReason.REGIME_SHIFT,
                confidence=0.8,
                notes="Market risk-off event occurred during hold period",
                contributing_factors=factors,
                metrics=metrics,
            )
    
    # Rule 4: SECTOR_ROTATION - Sector fell while market was flat/up
    if sector_return is not None and spy_return is not None:
        if sector_return < -0.03 and spy_return > -0.01:
            factors.append(f"Sector fell {sector_return*100:.1f}%")
            factors.append(f"While market was {spy_return*100:+.1f}%")
            metrics["sector_return_pct"] = round(sector_return * 100, 2)
            
            return FailureAnalysis(
                reason=FailureReason.SECTOR_ROTATION,
                confidence=0.75,
                notes="Sector-specific weakness while broader market held",
                contributing_factors=factors,
                metrics=metrics,
            )
    
    # Rule 5: FALSE_BREAKOUT - For breakout setups that reversed
    if setup_type == "breakout" and max_return > 0.03 and final_return < -0.03:
        factors.append("Setup was breakout")
        factors.append(f"Initial move: +{max_return*100:.1f}%")
        factors.append(f"Then reversed to {final_return*100:.1f}%")
        
        return FailureAnalysis(
            reason=FailureReason.FALSE_BREAKOUT,
            confidence=0.75,
            notes="Breakout failed and reversed - classic bull trap",
            contributing_factors=factors,
            metrics=metrics,
        )
    
    # Rule 6: DEAD_CAT_BOUNCE - For reversal setups that didn't hold
    if setup_type == "reversal" and max_return > 0.03 and final_return < -0.05:
        factors.append("Setup was reversal/mean reversion")
        factors.append(f"Bounce: +{max_return*100:.1f}%")
        factors.append(f"Then failed to {final_return*100:.1f}%")
        
        return FailureAnalysis(
            reason=FailureReason.DEAD_CAT_BOUNCE,
            confidence=0.75,
            notes="Reversal bounce was temporary - trend resumed",
            contributing_factors=factors,
            metrics=metrics,
        )
    
    # Rule 7: SLOW_BLEED - Never got traction, gradual decline
    if max_return < 0.02 and final_return < -0.05:
        factors.append(f"Max gain was only {max_return*100:.1f}%")
        factors.append("Never showed strength")
        factors.append(f"Gradual decline to {final_return*100:.1f}%")
        
        return FailureAnalysis(
            reason=FailureReason.SLOW_BLEED,
            confidence=0.7,
            notes="Position never gained traction - slow steady decline",
            contributing_factors=factors,
            metrics=metrics,
        )
    
    # Rule 8: Check for market correlation
    if spy_return is not None and final_return < -0.03:
        # Position fell more than market
        if spy_return < -0.02:
            factors.append(f"Market fell {spy_return*100:.1f}%")
            factors.append(f"Position fell {final_return*100:.1f}%")
            
            return FailureAnalysis(
                reason=FailureReason.REGIME_SHIFT,
                confidence=0.6,
                notes="Position fell with overall market weakness",
                contributing_factors=factors,
                metrics=metrics,
            )
        else:
            factors.append(f"Market was {spy_return*100:+.1f}%")
            factors.append(f"But position fell {final_return*100:.1f}%")
            
            return FailureAnalysis(
                reason=FailureReason.CORRELATION_BREAK,
                confidence=0.5,
                notes="Position underperformed while market held - stock-specific issue",
                contributing_factors=factors,
                metrics=metrics,
            )
    
    # Default: UNKNOWN
    factors.append(f"Final return: {final_return*100:.1f}%")
    factors.append(f"Max return: {max_return*100:.1f}%")
    factors.append("Could not determine specific failure mode")
    
    return FailureAnalysis(
        reason=FailureReason.UNKNOWN,
        confidence=0.3,
        notes="Insufficient data to determine specific failure reason",
        contributing_factors=factors,
        metrics=metrics,
    )


def get_failure_pattern_stats(outcomes_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Aggregate failure patterns to identify systemic issues.
    
    Args:
        outcomes_df: DataFrame with outcome records including 'failure_reason' column
    
    Returns:
        Dict with failure statistics and patterns
    """
    if outcomes_df is None or outcomes_df.empty:
        return {}
    
    stats = {
        "total_outcomes": len(outcomes_df),
        "reason_counts": {},
        "most_common_failure": None,
        "by_setup_type": {},
        "gave_back_rate": 0.0,
        "regime_shift_rate": 0.0,
        "immediate_reversal_rate": 0.0,
    }
    
    # Check if failure_reason column exists
    if "failure_reason" not in outcomes_df.columns:
        logger.debug("No failure_reason column in outcomes_df")
        return stats
    
    # Filter to actual failures (not success)
    failures_df = outcomes_df[
        (outcomes_df["failure_reason"].notna()) & 
        (outcomes_df["failure_reason"] != FailureReason.SUCCESS.value) &
        (outcomes_df["failure_reason"] != "success")
    ]
    
    if failures_df.empty:
        return stats
    
    # Count by reason
    reason_counts = failures_df["failure_reason"].value_counts().to_dict()
    stats["reason_counts"] = reason_counts
    
    # Most common failure
    if reason_counts:
        stats["most_common_failure"] = max(reason_counts.items(), key=lambda x: x[1])[0]
    
    # Calculate rates
    total = len(outcomes_df)
    stats["gave_back_rate"] = reason_counts.get(FailureReason.GAVE_BACK_GAINS.value, 0) / total if total > 0 else 0
    stats["regime_shift_rate"] = reason_counts.get(FailureReason.REGIME_SHIFT.value, 0) / total if total > 0 else 0
    stats["immediate_reversal_rate"] = reason_counts.get(FailureReason.IMMEDIATE_REVERSAL.value, 0) / total if total > 0 else 0
    
    # Failure by setup type
    if "setup_type" in failures_df.columns:
        by_setup = failures_df.groupby(["setup_type", "failure_reason"]).size().unstack(fill_value=0)
        stats["by_setup_type"] = by_setup.to_dict() if not by_setup.empty else {}
    
    # Failure by source
    if "source" in failures_df.columns:
        by_source = failures_df.groupby(["source", "failure_reason"]).size().unstack(fill_value=0)
        stats["by_source"] = by_source.to_dict() if not by_source.empty else {}
    
    return stats


def get_actionable_insights(failure_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate actionable insights from failure statistics.
    
    Args:
        failure_stats: Output from get_failure_pattern_stats()
    
    Returns:
        List of insight dicts with type, message, priority, action
    """
    insights = []
    
    if not failure_stats or failure_stats.get("total_outcomes", 0) == 0:
        return insights
    
    # Check GAVE_BACK_GAINS rate
    if failure_stats.get("gave_back_rate", 0) > 0.20:
        insights.append({
            "type": "EXIT_STRATEGY",
            "message": f"{failure_stats['gave_back_rate']*100:.0f}% of losses are 'gave back gains'. Consider implementing trailing stops or earlier profit-taking.",
            "priority": "HIGH",
            "action": "Add trailing stop logic when position is up >5%",
            "data": {"rate": failure_stats["gave_back_rate"]},
        })
    
    # Check REGIME_SHIFT rate
    if failure_stats.get("regime_shift_rate", 0) > 0.15:
        insights.append({
            "type": "REGIME_DETECTION",
            "message": f"{failure_stats['regime_shift_rate']*100:.0f}% of losses from regime shifts. Consider more sensitive regime gate.",
            "priority": "HIGH",
            "action": "Tighten VIX threshold or add SPY momentum filter",
            "data": {"rate": failure_stats["regime_shift_rate"]},
        })
    
    # Check IMMEDIATE_REVERSAL rate
    if failure_stats.get("immediate_reversal_rate", 0) > 0.15:
        insights.append({
            "type": "ENTRY_TIMING",
            "message": f"{failure_stats['immediate_reversal_rate']*100:.0f}% of losses from immediate reversals. Entry timing may be chasing.",
            "priority": "MEDIUM",
            "action": "Consider waiting for pullback or using limit orders below current price",
            "data": {"rate": failure_stats["immediate_reversal_rate"]},
        })
    
    # Check for setup-specific issues
    by_setup = failure_stats.get("by_setup_type", {})
    if by_setup:
        for setup_type, reasons in by_setup.items():
            if isinstance(reasons, dict):
                total_failures = sum(reasons.values())
                if total_failures > 5:
                    # Find dominant failure for this setup
                    dominant = max(reasons.items(), key=lambda x: x[1])
                    if dominant[1] / total_failures > 0.4:
                        insights.append({
                            "type": "SETUP_SPECIFIC",
                            "message": f"{setup_type} setups have {dominant[1]/total_failures*100:.0f}% failure rate from '{dominant[0]}'",
                            "priority": "MEDIUM",
                            "action": f"Review {setup_type} entry criteria or consider de-weighting",
                            "data": {"setup_type": setup_type, "reason": dominant[0], "count": dominant[1]},
                        })
    
    return insights
