# src/portfolio/sizing.py
"""
Position sizing using Kelly criterion with conservative adjustments.

Provides:
- Kelly fraction calculation for binary outcomes
- Configurable shrinkage (fractional Kelly)
- Hard caps on position sizes
- Regime-aware sizing adjustments
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict
import math


@dataclass(frozen=True)
class SizingConfig:
    """
    Configuration for position sizing.
    
    Attributes:
        method: Sizing method ("kelly", "vol_target", "equal")
        kelly_shrink: Fraction of full Kelly to use (0.25 = quarter Kelly)
        kelly_cap: Maximum position from Kelly formula
        min_weight: Minimum position weight (0 = allow zero)
        max_weight: Hard cap per position (pre-liquidity)
        portfolio_gross: Target gross exposure (1.0 = 100%)
        vol_target_annual: Target annual volatility for vol_target method
    """
    method: str = "kelly"          # {"kelly", "vol_target", "equal"}
    kelly_shrink: float = 0.25     # use 1/4 Kelly by default (conservative)
    kelly_cap: float = 0.05        # max 5% notional per name from Kelly
    min_weight: float = 0.0
    max_weight: float = 0.10       # hard cap per name (pre-liquidity)
    portfolio_gross: float = 1.0   # 1.0 = 100% gross
    vol_target_annual: float = 20.0  # used by vol_target sizing


# Regime-specific Kelly shrinkage multipliers
REGIME_SHRINK_MULTIPLIER: Dict[str, float] = {
    "bull": 1.2,    # More aggressive in bull markets
    "chop": 1.0,    # Default in choppy markets
    "stress": 0.4,  # Very conservative in stress
}


def kelly_fraction(p: float, b: float) -> float:
    """
    Kelly criterion for binary outcome with payoff b.
    
    Win: returns +b (proportion of stake)
    Loss: returns -1 (lose entire stake)
    
    f* = (p*(b+1) - 1) / b
    
    Where:
    - p = probability of winning
    - b = odds (win amount / loss amount)
    - f* = optimal fraction of capital to bet
    
    Args:
        p: Probability of winning (0-1)
        b: Payoff ratio (win/loss)
    
    Returns:
        Optimal Kelly fraction (can be negative if edge is negative)
    """
    if b <= 0:
        return 0.0
    return (p * (b + 1.0) - 1.0) / b


def size_from_prob(
    prob_hit: Optional[float],
    payoff_b: Optional[float],
    cfg: SizingConfig,
    regime: Optional[str] = None,
) -> float:
    """
    Compute position size from probability and payoff.
    
    Args:
        prob_hit: Probability of target hit (0-1)
        payoff_b: Payoff ratio (win/loss)
        cfg: SizingConfig with caps and shrinkage
        regime: Optional regime for adjusted shrinkage
    
    Returns:
        Position weight (0 to max_weight)
    """
    if prob_hit is None or payoff_b is None:
        return 0.0

    p = max(0.0, min(1.0, float(prob_hit)))
    b = max(0.01, float(payoff_b))

    f = kelly_fraction(p, b)
    f = max(0.0, f)  # no shorting in this version
    
    # Apply shrinkage
    shrink = cfg.kelly_shrink
    if regime and regime in REGIME_SHRINK_MULTIPLIER:
        shrink *= REGIME_SHRINK_MULTIPLIER[regime]
    f *= shrink
    
    # Apply caps
    f = min(f, cfg.kelly_cap)
    f = max(cfg.min_weight, min(cfg.max_weight, f))
    
    return f


def size_equal_weight(n_positions: int, cfg: SizingConfig) -> float:
    """
    Equal weight sizing.
    
    Args:
        n_positions: Number of positions
        cfg: SizingConfig
    
    Returns:
        Weight per position
    """
    if n_positions <= 0:
        return 0.0
    w = cfg.portfolio_gross / n_positions
    return min(w, cfg.max_weight)


def size_vol_target(
    realized_vol_pct: Optional[float],
    cfg: SizingConfig,
) -> float:
    """
    Vol-target sizing: weight inversely proportional to realized vol.
    
    Args:
        realized_vol_pct: Realized annualized volatility (%)
        cfg: SizingConfig with vol_target_annual
    
    Returns:
        Weight based on vol targeting
    """
    if realized_vol_pct is None or realized_vol_pct <= 0:
        return 0.0
    
    # Target weight = (target_vol / realized_vol)
    w = cfg.vol_target_annual / realized_vol_pct
    w = max(cfg.min_weight, min(cfg.max_weight, w))
    
    return w
