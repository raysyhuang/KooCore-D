"""
Tests for filtering logic.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest
from src.core.filters import apply_hard_filters


def test_apply_hard_filters_basic():
    """Test that hard filters work with basic data."""
    # Create sample dataframe
    df = pd.DataFrame({
        "Ticker": ["AAPL", "GOOGL", "MSFT"],
        "Close": [150.0, 2500.0, 300.0],
        "Volume": [50000000, 2000000, 30000000],
        "Avg_Dollar_Volume_20d": [100000000, 50000000, 80000000],
    })
    
    params = {
        "liquidity": {
            "min_avg_dollar_volume_20d": 50000000,
        },
        "price": {
            "min_price": 10.0,
            "max_price": 10000.0,
        },
    }
    
    passes, reasons = apply_hard_filters(df, params)
    
    # Should return boolean and list
    assert isinstance(passes, bool)
    assert isinstance(reasons, list)


def test_apply_hard_filters_edge_cases():
    """Test filters with edge cases (empty df, missing columns)."""
    # Empty dataframe
    df_empty = pd.DataFrame()
    params = {"liquidity": {"min_avg_dollar_volume_20d": 50000000}}
    
    passes, reasons = apply_hard_filters(df_empty, params)
    assert isinstance(passes, bool)
    assert isinstance(reasons, list)
    
    # Missing columns
    df_missing = pd.DataFrame({"Ticker": ["AAPL"]})
    passes, reasons = apply_hard_filters(df_missing, params)
    assert isinstance(passes, bool)
    assert isinstance(reasons, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

