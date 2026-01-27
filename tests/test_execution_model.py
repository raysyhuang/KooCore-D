# tests/test_execution_model.py
"""
Tests for execution model and entry price computation.
"""
import pandas as pd
import pytest

from src.backtest.execution import ExecutionModel, entry_price, exit_price


def test_next_open_entry():
    """Test next_open entry uses next day's open."""
    idx = pd.to_datetime(["2025-01-01", "2025-01-02"])
    df = pd.DataFrame({
        "Open": [10, 11],
        "High": [10, 12],
        "Low": [9, 10],
        "Close": [10, 11],
        "Volume": [100, 100]
    }, index=idx)

    model = ExecutionModel(entry="next_open", slippage_bps=0, fee_bps=0)
    px = entry_price(df, "2025-01-01", model)
    
    assert px == 11.0  # Next day's open


def test_same_close_entry():
    """Test same_close entry uses decision day's close."""
    idx = pd.to_datetime(["2025-01-01"])
    df = pd.DataFrame({
        "Open": [10],
        "High": [10],
        "Low": [9],
        "Close": [10],
        "Volume": [100]
    }, index=idx)

    model = ExecutionModel(entry="same_close", slippage_bps=0, fee_bps=0)
    px = entry_price(df, "2025-01-01", model)
    
    assert px == 10.0


def test_slippage_and_fees_applied():
    """Test that slippage and fees increase entry price."""
    idx = pd.to_datetime(["2025-01-01", "2025-01-02"])
    df = pd.DataFrame({
        "Open": [100, 100],
        "High": [100, 100],
        "Low": [100, 100],
        "Close": [100, 100],
        "Volume": [100, 100]
    }, index=idx)

    # 10 bps slippage + 5 bps fees = 15 bps = 0.15%
    model = ExecutionModel(entry="next_open", slippage_bps=10, fee_bps=5)
    px = entry_price(df, "2025-01-01", model)
    
    # Entry price should be 100 * (1 + 15/10000) = 100.15
    assert round(px, 2) == 100.15


def test_entry_no_data_returns_none():
    """Test that entry_price returns None when no data available."""
    df = pd.DataFrame()
    model = ExecutionModel()
    
    assert entry_price(df, "2025-01-01", model) is None


def test_entry_no_future_data_returns_none():
    """Test that next_open returns None if no future data exists."""
    idx = pd.to_datetime(["2025-01-01"])
    df = pd.DataFrame({
        "Open": [10],
        "High": [10],
        "Low": [9],
        "Close": [10],
        "Volume": [100]
    }, index=idx)

    model = ExecutionModel(entry="next_open", slippage_bps=0, fee_bps=0)
    px = entry_price(df, "2025-01-01", model)
    
    assert px is None  # No day after 2025-01-01


def test_default_execution_model():
    """Test default ExecutionModel values."""
    model = ExecutionModel()
    
    assert model.entry == "next_open"
    assert model.slippage_bps == 5.0
    assert model.fee_bps == 2.0


def test_exit_price_reduces_proceeds():
    """Test that exit price reduces proceeds by slippage/fees."""
    idx = pd.to_datetime(["2025-01-01"])
    df = pd.DataFrame({
        "Open": [100],
        "High": [100],
        "Low": [100],
        "Close": [100],
        "Volume": [100]
    }, index=idx)

    model = ExecutionModel(slippage_bps=10, fee_bps=5)
    px = exit_price(df, "2025-01-01", model, use_close=True)
    
    # Exit price should be 100 * (1 - 15/10000) = 99.85
    assert round(px, 2) == 99.85
