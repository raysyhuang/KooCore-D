# src/portfolio/liquidity.py
"""
Liquidity-aware position caps.

Prevents oversized positions in illiquid names by capping
based on Average Daily Volume (ADV).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LiquidityConfig:
    """
    Liquidity constraints for position sizing.
    
    Attributes:
        max_adv_pct: Maximum position as % of ADV (default 1%)
        min_adv_usd: Minimum ADV required to trade (default $2M)
    """
    max_adv_pct: float = 1.0      # max 1% of ADV per name
    min_adv_usd: float = 2_000_000.0  # minimum liquidity threshold


def apply_liquidity_cap(
    weight: float,
    adv_usd: Optional[float],
    portfolio_usd: float,
    cfg: LiquidityConfig,
) -> float:
    """
    Cap position weight based on liquidity constraints.
    
    Args:
        weight: Raw position weight (from sizing model)
        adv_usd: Average daily volume in USD
        portfolio_usd: Total portfolio value
        cfg: LiquidityConfig with caps
    
    Returns:
        Capped weight (may be reduced or zeroed)
    """
    if weight <= 0:
        return 0.0
    
    if adv_usd is None or adv_usd <= 0 or portfolio_usd <= 0:
        return weight  # No liquidity data, use original weight
    
    # Check minimum liquidity threshold
    if adv_usd < cfg.min_adv_usd:
        return 0.0  # Too illiquid to trade
    
    # Compute maximum notional as % of ADV
    max_notional = (cfg.max_adv_pct / 100.0) * adv_usd
    max_weight = max_notional / portfolio_usd
    
    return min(weight, max_weight)


def estimate_market_impact(
    notional_usd: float,
    adv_usd: Optional[float],
    spread_bps: float = 10.0,
) -> Optional[float]:
    """
    Estimate market impact cost in basis points.
    
    Simple square-root model: impact ~ sqrt(participation_rate) * spread
    
    Args:
        notional_usd: Position size in USD
        adv_usd: Average daily volume in USD
        spread_bps: Average bid-ask spread in basis points
    
    Returns:
        Estimated impact cost in basis points
    """
    if adv_usd is None or adv_usd <= 0 or notional_usd <= 0:
        return None
    
    participation = notional_usd / adv_usd
    impact = (participation ** 0.5) * spread_bps
    
    return impact
