"""
Pytest configuration and fixtures.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV DataFrame for testing."""
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    days = 300
    
    dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq='D')
    close = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, days))
    
    return pd.DataFrame({
        'Open': close * (1 + np.random.normal(0, 0.005, days)),
        'High': close * (1 + np.abs(np.random.normal(0, 0.01, days))),
        'Low': close * (1 - np.abs(np.random.normal(0, 0.01, days))),
        'Close': close,
        'Volume': (1_000_000 * (1 + np.random.uniform(-0.3, 0.5, days))).astype(int),
    }, index=dates)


@pytest.fixture
def tmp_cache_path(tmp_path):
    """Create temporary cache database path."""
    return str(tmp_path / "test_cache.db")


@pytest.fixture
def sample_position_data():
    """Create sample position data for testing."""
    return {
        "ticker": "AAPL",
        "entry_date": "2024-01-15",
        "entry_price": 150.00,
        "shares": 100,
        "source": "weekly",
        "predicted_rank": 1,
        "predicted_score": 7.5,
    }


@pytest.fixture
def sample_hybrid_analysis():
    """Create sample hybrid analysis JSON structure."""
    return {
        "date": "2024-01-15",
        "summary": {
            "weekly_top5_count": 5,
            "pro30_candidates_count": 10,
            "movers_count": 3
        },
        "overlaps": {
            "all_three": ["AAPL"],
            "weekly_pro30": ["MSFT", "GOOGL"],
            "weekly_movers": [],
            "pro30_movers": ["AMZN"]
        },
        "weekly_top5": [
            {
                "rank": 1,
                "ticker": "AAPL",
                "composite_score": 7.5,
                "confidence": "HIGH"
            },
            {
                "rank": 2,
                "ticker": "MSFT",
                "composite_score": 7.2,
                "confidence": "MEDIUM"
            }
        ],
        "pro30_tickers": ["AAPL", "MSFT", "GOOGL", "AMZN"],
        "movers_tickers": ["AAPL", "AMZN", "NVDA"]
    }
