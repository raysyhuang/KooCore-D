# tests/test_asof_and_leakage.py
"""
Tests for as-of date enforcement and lookahead prevention.
"""
import pandas as pd
import numpy as np
import pytest

from src.core.asof import enforce_asof, validate_ohlcv


def test_enforce_asof_removes_future_rows():
    """Test that enforce_asof correctly removes rows after the as-of date."""
    idx = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-10"])
    df = pd.DataFrame(
        {
            "Open": [1, 1, 1],
            "High": [1, 1, 1],
            "Low": [1, 1, 1],
            "Close": [1, 1, 1],
            "Volume": [10, 10, 10]
        },
        index=idx,
    )
    cut, info = enforce_asof(df, "2025-01-02", "TEST", strict=True)
    
    assert len(cut) == 2
    assert cut.index.max().date().isoformat() == "2025-01-02"
    assert info.get("dropped_future_rows") == 1


def test_enforce_asof_handles_empty_df():
    """Test that enforce_asof handles empty DataFrames gracefully."""
    df = pd.DataFrame()
    cut, info = enforce_asof(df, "2025-01-02", "TEST", strict=False)
    
    assert cut.empty
    assert info["ticker"] == "TEST"


def test_enforce_asof_no_date_passthrough():
    """Test that enforce_asof passes through data when no as-of date is provided."""
    idx = pd.to_datetime(["2025-01-01", "2025-01-02"])
    df = pd.DataFrame(
        {
            "Open": [1, 1],
            "High": [1, 1],
            "Low": [1, 1],
            "Close": [1, 1],
            "Volume": [10, 10]
        },
        index=idx,
    )
    cut, info = enforce_asof(df, None, "TEST", strict=False)
    
    assert len(cut) == 2  # All rows kept


def test_validate_ohlcv_drops_bad_bars():
    """Test that validate_ohlcv removes bars with invalid OHLCV data."""
    idx = pd.to_datetime(["2025-01-01", "2025-01-02"])
    df = pd.DataFrame(
        {
            "Open": [1, 1],
            "High": [1, 0.5],  # High < Low is invalid
            "Low": [1, 1],
            "Close": [1, 1],
            "Volume": [10, 10]
        },
        index=idx,
    )
    clean, stats = validate_ohlcv(df, "TEST")
    
    assert len(clean) == 1
    assert stats["dropped_bad_rows"] == 1


def test_validate_ohlcv_detects_missing_columns():
    """Test that validate_ohlcv detects missing required columns."""
    idx = pd.to_datetime(["2025-01-01"])
    df = pd.DataFrame(
        {
            "Open": [1],
            "High": [1],
            # Missing: Low, Close, Volume
        },
        index=idx,
    )
    clean, stats = validate_ohlcv(df, "TEST")
    
    assert clean.empty
    assert "Low" in stats["missing_cols"]
    assert "Close" in stats["missing_cols"]
    assert "Volume" in stats["missing_cols"]


def test_validate_ohlcv_normalizes_column_names():
    """Test that validate_ohlcv handles various column name cases."""
    idx = pd.to_datetime(["2025-01-01"])
    df = pd.DataFrame(
        {
            "open": [10],  # lowercase
            "HIGH": [12],  # uppercase
            "Low": [9],    # mixed
            "close": [11],
            "vol": [1000],  # alternative name
        },
        index=idx,
    )
    clean, stats = validate_ohlcv(df, "TEST")
    
    assert not clean.empty
    assert "Open" in clean.columns
    assert "High" in clean.columns
    assert "Volume" in clean.columns
    assert stats["rows_out"] == 1


def test_validate_ohlcv_drops_negative_volume():
    """Test that validate_ohlcv removes bars with negative volume."""
    idx = pd.to_datetime(["2025-01-01", "2025-01-02"])
    df = pd.DataFrame(
        {
            "Open": [1, 1],
            "High": [2, 2],
            "Low": [0.5, 0.5],
            "Close": [1.5, 1.5],
            "Volume": [100, -10],  # Negative volume is invalid
        },
        index=idx,
    )
    clean, stats = validate_ohlcv(df, "TEST")
    
    assert len(clean) == 1
    assert stats["dropped_bad_rows"] == 1


def test_validate_ohlcv_drops_zero_price():
    """Test that validate_ohlcv removes bars with zero/negative prices."""
    idx = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])
    df = pd.DataFrame(
        {
            "Open": [1, 0, 1],  # Zero open on day 2
            "High": [2, 2, 2],
            "Low": [0.5, 0.5, 0.5],
            "Close": [1.5, 1.5, 1.5],
            "Volume": [100, 100, 100],
        },
        index=idx,
    )
    clean, stats = validate_ohlcv(df, "TEST")
    
    assert len(clean) == 2
    assert stats["dropped_bad_rows"] == 1
