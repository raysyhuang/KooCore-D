# src/portfolio/construct.py
"""
Portfolio construction using greedy allocation.

Builds a trade plan from ranked candidates by:
1. Filtering by event gate and probability threshold
2. Sorting by expected value (or technical score fallback)
3. Computing raw weights from Kelly (or alternative method)
4. Applying liquidity caps
5. Normalizing to target gross exposure
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
import os

from .sizing import SizingConfig, size_from_prob, size_equal_weight
from .liquidity import LiquidityConfig, apply_liquidity_cap


@dataclass
class PortfolioConfig:
    """
    Complete portfolio configuration.
    
    Attributes:
        portfolio_usd: Total portfolio value
        max_positions: Maximum number of positions
        min_prob: Minimum probability threshold to include
        sizing: SizingConfig for position sizing
        liquidity: LiquidityConfig for liquidity caps
    """
    portfolio_usd: float = 100_000.0
    max_positions: int = 5
    min_prob: float = 0.0
    sizing: SizingConfig = field(default_factory=SizingConfig)
    liquidity: LiquidityConfig = field(default_factory=LiquidityConfig)


def payoff_proxy_b(target_pct: float, stop_pct: Optional[float]) -> float:
    """
    Convert target/stop to payoff ratio b for Kelly formula.
    
    Win: +target_pct
    Loss: -stop_pct (or -target_pct if stop_pct is None)
    b = win / loss
    
    Args:
        target_pct: Target return percentage (e.g., 10.0 for 10%)
        stop_pct: Stop loss percentage (e.g., 5.0 for 5%)
    
    Returns:
        Payoff ratio b
    """
    loss = abs(stop_pct) if stop_pct is not None else target_pct
    loss = max(0.1, loss)  # Prevent division issues
    return float(target_pct / loss)


def build_trade_plan(
    candidates: List[Dict[str, Any]],
    cfg: PortfolioConfig,
    target_pct: float = 10.0,
    stop_pct: Optional[float] = 5.0,
    regime: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Build trade plan from ranked candidates.
    
    Process:
    1. Filter out event-gated and low-probability candidates
    2. Sort by expected_value (or technical_score fallback)
    3. Take top max_positions
    4. Compute raw weights from sizing method
    5. Apply liquidity caps
    6. Normalize to portfolio gross target
    
    Args:
        candidates: List of candidate dicts with scores and features
        cfg: PortfolioConfig
        target_pct: Target return for Kelly calculation
        stop_pct: Stop loss for Kelly calculation
        regime: Current market regime for sizing adjustments
    
    Returns:
        List of trade plan entries with weights and notionals
    """
    # Filter candidates
    rows = []
    for c in candidates:
        # Skip event-gated (earnings proximity blocked)
        event_gate = c.get("event_gate", {})
        if isinstance(event_gate, dict) and event_gate.get("blocked"):
            continue
        if c.get("event_gate_blocked"):
            continue
        
        # Skip if below probability threshold (when probability is available)
        p = c.get("prob_hit_10")
        if p is not None and p < cfg.min_prob:
            continue
        
        rows.append(c)

    # Sort by expected value (if available) or technical score
    def sort_key(r):
        ev = r.get("expected_value")
        if ev is not None:
            return ev
        return r.get("technical_score", 0)
    
    rows.sort(key=sort_key, reverse=True)
    rows = rows[:cfg.max_positions]

    if not rows:
        return []

    # Compute payoff ratio for Kelly
    b = payoff_proxy_b(target_pct, stop_pct)

    # Build plan with weights
    plan = []
    for r in rows:
        w = 0.0
        
        if cfg.sizing.method == "equal":
            w = size_equal_weight(len(rows), cfg.sizing)
        elif cfg.sizing.method == "kelly":
            w = size_from_prob(
                prob_hit=r.get("prob_hit_10"),
                payoff_b=b,
                cfg=cfg.sizing,
                regime=regime,
            )
            # Fallback to equal weight if no probability
            if w == 0 and r.get("prob_hit_10") is None:
                w = size_equal_weight(len(rows), cfg.sizing) * 0.5  # Half weight when no prob
        else:
            # Proportional to expected value (fallback method)
            ev = r.get("expected_value")
            if ev is not None and ev > 0:
                w = min(float(ev) / 10.0, cfg.sizing.max_weight)
            else:
                w = size_equal_weight(len(rows), cfg.sizing) * 0.5

        # Apply liquidity cap
        adv = r.get("adv_20") or r.get("avg_dollar_volume_20d")
        w = apply_liquidity_cap(w, adv, cfg.portfolio_usd, cfg.liquidity)

        plan.append({
            "ticker": r["ticker"],
            "weight": w,
            "prob_hit_10": r.get("prob_hit_10"),
            "expected_value": r.get("expected_value"),
            "technical_score": r.get("technical_score"),
            "breakout_score": r.get("breakout_score"),
            "adv_20": adv,
        })

    # Normalize to portfolio gross if over
    gross = sum(p["weight"] for p in plan)
    if gross > cfg.sizing.portfolio_gross and gross > 0:
        scale = cfg.sizing.portfolio_gross / gross
        for p in plan:
            p["weight"] *= scale

    # Add final calculations
    for p in plan:
        p["weight"] = round(p["weight"], 4)
        p["notional_usd"] = round(p["weight"] * cfg.portfolio_usd, 2)

    return plan


def write_trade_plan(
    trade_plan: List[Dict[str, Any]],
    out_dir: str,
    date_str: str,
) -> str:
    """Write trade plan to CSV."""
    import pandas as pd
    
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"trade_plan_{date_str}.csv")
    
    if trade_plan:
        pd.DataFrame(trade_plan).to_csv(path, index=False)
    else:
        # Write empty file with headers
        pd.DataFrame(columns=[
            "ticker", "weight", "prob_hit_10", "expected_value",
            "technical_score", "breakout_score", "adv_20", "notional_usd"
        ]).to_csv(path, index=False)
    
    return path


def write_portfolio_summary(
    trade_plan: List[Dict[str, Any]],
    cfg: PortfolioConfig,
    out_dir: str,
    date_str: str,
    regime: Optional[str] = None,
) -> str:
    """Write portfolio summary to JSON."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"portfolio_summary_{date_str}.json")
    
    summary = {
        "portfolio_usd": cfg.portfolio_usd,
        "max_positions": cfg.max_positions,
        "sizing_method": cfg.sizing.method,
        "kelly_shrink": cfg.sizing.kelly_shrink,
        "regime": regime,
        "gross_weight": round(sum(p["weight"] for p in trade_plan), 4),
        "positions": len(trade_plan),
        "tickers": [p["ticker"] for p in trade_plan],
        "total_notional_usd": sum(p["notional_usd"] for p in trade_plan),
    }
    
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    
    return path
