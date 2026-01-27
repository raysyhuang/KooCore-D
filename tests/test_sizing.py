# tests/test_sizing.py
"""
Tests for position sizing using Kelly criterion.
"""
import pytest

from src.portfolio.sizing import (
    kelly_fraction,
    size_from_prob,
    size_equal_weight,
    SizingConfig,
    REGIME_SHRINK_MULTIPLIER,
)


def test_kelly_fraction_basic():
    """Test basic Kelly fraction calculation."""
    # With 60% probability and 1:1 odds, Kelly says bet 20%
    f = kelly_fraction(p=0.6, b=1.0)
    assert f > 0
    assert round(f, 2) == 0.20  # (0.6*2 - 1) / 1 = 0.2


def test_kelly_fraction_no_edge():
    """Test Kelly fraction with no edge (50/50)."""
    f = kelly_fraction(p=0.5, b=1.0)
    assert f == 0  # No edge, no bet


def test_kelly_fraction_negative_edge():
    """Test Kelly fraction with negative edge."""
    f = kelly_fraction(p=0.4, b=1.0)
    assert f < 0  # Negative edge


def test_kelly_fraction_zero_payoff():
    """Test Kelly fraction with zero payoff."""
    f = kelly_fraction(p=0.6, b=0.0)
    assert f == 0


def test_size_cap():
    """Test that size is capped at kelly_cap."""
    cfg = SizingConfig(kelly_shrink=1.0, kelly_cap=0.02, max_weight=0.10)
    w = size_from_prob(prob_hit=0.9, payoff_b=2.0, cfg=cfg)
    assert w <= 0.02


def test_size_max_weight_cap():
    """Test that size is capped at max_weight."""
    cfg = SizingConfig(kelly_shrink=1.0, kelly_cap=1.0, max_weight=0.05)
    w = size_from_prob(prob_hit=0.9, payoff_b=2.0, cfg=cfg)
    assert w <= 0.05


def test_size_shrinkage():
    """Test Kelly shrinkage is applied."""
    cfg_full = SizingConfig(kelly_shrink=1.0, kelly_cap=1.0, max_weight=1.0)
    cfg_quarter = SizingConfig(kelly_shrink=0.25, kelly_cap=1.0, max_weight=1.0)
    
    w_full = size_from_prob(0.7, 1.5, cfg_full)
    w_quarter = size_from_prob(0.7, 1.5, cfg_quarter)
    
    assert w_quarter < w_full
    assert abs(w_quarter - w_full * 0.25) < 0.01


def test_size_none_inputs():
    """Test that None inputs return zero size."""
    cfg = SizingConfig()
    
    assert size_from_prob(None, 1.0, cfg) == 0.0
    assert size_from_prob(0.6, None, cfg) == 0.0


def test_size_equal_weight():
    """Test equal weight sizing."""
    cfg = SizingConfig(portfolio_gross=1.0, max_weight=0.5)
    
    w = size_equal_weight(n_positions=5, cfg=cfg)
    assert w == 0.2  # 1.0 / 5
    
    w = size_equal_weight(n_positions=1, cfg=cfg)
    assert w == 0.5  # Capped at max_weight


def test_size_equal_weight_zero_positions():
    """Test equal weight with zero positions."""
    cfg = SizingConfig()
    assert size_equal_weight(0, cfg) == 0.0


def test_regime_multiplier_exists():
    """Test regime shrink multipliers are defined."""
    assert "bull" in REGIME_SHRINK_MULTIPLIER
    assert "chop" in REGIME_SHRINK_MULTIPLIER
    assert "stress" in REGIME_SHRINK_MULTIPLIER


def test_regime_stress_reduces_size():
    """Test that stress regime reduces position size."""
    cfg = SizingConfig(kelly_shrink=0.25, kelly_cap=1.0, max_weight=1.0)
    
    w_chop = size_from_prob(0.7, 2.0, cfg, regime="chop")
    w_stress = size_from_prob(0.7, 2.0, cfg, regime="stress")
    
    assert w_stress < w_chop
