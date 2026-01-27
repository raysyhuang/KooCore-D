# tests/test_portfolio_construct.py
"""
Tests for portfolio construction and trade plan building.
"""
import pytest

from src.portfolio.construct import (
    PortfolioConfig,
    build_trade_plan,
    payoff_proxy_b,
)
from src.portfolio.sizing import SizingConfig
from src.portfolio.liquidity import LiquidityConfig


def test_payoff_proxy_b():
    """Test payoff ratio calculation."""
    # 10% target, 5% stop -> b = 2.0
    b = payoff_proxy_b(target_pct=10.0, stop_pct=5.0)
    assert b == 2.0
    
    # 10% target, 10% stop -> b = 1.0
    b = payoff_proxy_b(target_pct=10.0, stop_pct=10.0)
    assert b == 1.0
    
    # 10% target, None stop -> b = 1.0 (default to target)
    b = payoff_proxy_b(target_pct=10.0, stop_pct=None)
    assert b == 1.0


def test_build_trade_plan_basic():
    """Test basic trade plan construction."""
    candidates = [
        {"ticker": "AAPL", "technical_score": 8.0, "adv_20": 5_000_000_000},
        {"ticker": "GOOGL", "technical_score": 7.0, "adv_20": 3_000_000_000},
        {"ticker": "MSFT", "technical_score": 9.0, "adv_20": 4_000_000_000},
    ]
    
    cfg = PortfolioConfig(
        portfolio_usd=100_000,
        max_positions=3,
        sizing=SizingConfig(method="equal"),
    )
    
    plan = build_trade_plan(candidates, cfg)
    
    assert len(plan) == 3
    # MSFT should be first (highest score)
    assert plan[0]["ticker"] == "MSFT"


def test_build_trade_plan_max_positions():
    """Test that max_positions limits the plan."""
    candidates = [
        {"ticker": f"T{i}", "technical_score": float(i), "adv_20": 10_000_000_000}
        for i in range(10)
    ]
    
    cfg = PortfolioConfig(
        portfolio_usd=100_000,
        max_positions=3,
    )
    
    plan = build_trade_plan(candidates, cfg)
    
    assert len(plan) <= 3


def test_build_trade_plan_event_gate_blocked():
    """Test that event-gated candidates are excluded."""
    candidates = [
        {"ticker": "AAPL", "technical_score": 8.0, "adv_20": 5_000_000_000},
        {"ticker": "GOOGL", "technical_score": 9.0, "adv_20": 3_000_000_000, "event_gate_blocked": True},
    ]
    
    cfg = PortfolioConfig(max_positions=5)
    
    plan = build_trade_plan(candidates, cfg)
    
    assert len(plan) == 1
    assert plan[0]["ticker"] == "AAPL"


def test_build_trade_plan_event_gate_dict():
    """Test event gate as dict format."""
    candidates = [
        {"ticker": "AAPL", "technical_score": 8.0, "adv_20": 5_000_000_000},
        {"ticker": "GOOGL", "technical_score": 9.0, "adv_20": 3_000_000_000, 
         "event_gate": {"blocked": True, "reason": "earnings_soon"}},
    ]
    
    cfg = PortfolioConfig(max_positions=5)
    
    plan = build_trade_plan(candidates, cfg)
    
    assert len(plan) == 1
    assert plan[0]["ticker"] == "AAPL"


def test_build_trade_plan_with_probability():
    """Test trade plan with probability-based sizing."""
    candidates = [
        {"ticker": "AAPL", "technical_score": 8.0, "prob_hit_10": 0.6, "adv_20": 5_000_000_000},
        {"ticker": "GOOGL", "technical_score": 7.0, "prob_hit_10": 0.7, "adv_20": 3_000_000_000},
    ]
    
    cfg = PortfolioConfig(
        portfolio_usd=100_000,
        max_positions=5,
        sizing=SizingConfig(method="kelly", kelly_shrink=0.25),
    )
    
    plan = build_trade_plan(candidates, cfg, target_pct=10.0, stop_pct=5.0)
    
    assert len(plan) == 2
    # Weights should be positive for positive probabilities
    for p in plan:
        assert p["weight"] >= 0


def test_build_trade_plan_empty():
    """Test trade plan with no candidates."""
    plan = build_trade_plan([], PortfolioConfig())
    assert plan == []


def test_build_trade_plan_normalization():
    """Test that weights are normalized to portfolio_gross."""
    candidates = [
        {"ticker": f"T{i}", "technical_score": 9.0, "prob_hit_10": 0.9, "adv_20": 100_000_000_000}
        for i in range(5)
    ]
    
    cfg = PortfolioConfig(
        portfolio_usd=100_000,
        max_positions=5,
        sizing=SizingConfig(method="kelly", kelly_shrink=1.0, kelly_cap=0.5, max_weight=0.5),
    )
    
    plan = build_trade_plan(candidates, cfg)
    
    gross = sum(p["weight"] for p in plan)
    assert gross <= 1.0  # Should not exceed portfolio_gross


def test_build_trade_plan_notional_calculated():
    """Test that notional_usd is correctly calculated."""
    candidates = [
        {"ticker": "AAPL", "technical_score": 8.0, "adv_20": 5_000_000_000},
    ]
    
    cfg = PortfolioConfig(
        portfolio_usd=100_000,
        max_positions=1,
        sizing=SizingConfig(method="equal"),
    )
    
    plan = build_trade_plan(candidates, cfg)
    
    assert len(plan) == 1
    assert plan[0]["notional_usd"] == plan[0]["weight"] * 100_000
