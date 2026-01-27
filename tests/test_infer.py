# tests/test_infer.py
"""
Tests for calibration inference.
"""
import pytest

from src.calibration.infer import infer_probability, infer_probabilities_batch, compute_expected_value


def test_infer_no_model():
    """Test that missing model returns None."""
    p = infer_probability("missing_model.pkl", {})
    
    assert p is None


def test_infer_empty_row():
    """Test inference with empty features."""
    p = infer_probability("missing_model.pkl", {})
    
    assert p is None


def test_infer_probabilities_batch_no_model():
    """Test batch inference with missing model."""
    rows = [{"a": 1}, {"a": 2}]
    probs = infer_probabilities_batch("missing_model.pkl", rows)
    
    assert len(probs) == 2
    assert all(p is None for p in probs)


def test_compute_expected_value():
    """Test expected value computation."""
    ev = compute_expected_value(
        prob_hit=0.5,
        atr_pct=5.0,
        target_pct=10.0
    )
    
    # EV = 0.5 * (10/5) = 0.5 * 2 = 1.0
    assert ev is not None
    assert ev > 0


def test_compute_expected_value_none_prob():
    """Test EV with None probability."""
    ev = compute_expected_value(
        prob_hit=None,
        atr_pct=5.0,
        target_pct=10.0
    )
    
    assert ev is None


def test_compute_expected_value_high_atr():
    """Test EV with high ATR (easier target, lower payoff)."""
    ev_low_atr = compute_expected_value(prob_hit=0.5, atr_pct=2.0, target_pct=10.0)
    ev_high_atr = compute_expected_value(prob_hit=0.5, atr_pct=10.0, target_pct=10.0)
    
    # Higher ATR means lower payoff proxy
    assert ev_low_atr > ev_high_atr
