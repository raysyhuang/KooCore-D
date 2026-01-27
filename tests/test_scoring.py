"""
Tests for scoring module.

Tests technical score calculation and scoring rubrics.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# Import from src
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.scoring import (
    compute_technical_score_weekly,
    compute_score_30d_breakout,
    compute_score_30d_reversal,
)


def create_mock_ohlcv(
    days: int = 300,
    start_price: float = 100.0,
    trend: float = 0.001,
    volatility: float = 0.02,
    volume_base: int = 1_000_000,
) -> pd.DataFrame:
    """Create mock OHLCV data for testing."""
    np.random.seed(42)  # For reproducibility
    
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Generate price series with trend and noise
    returns = np.random.normal(trend, volatility, days)
    close = start_price * np.cumprod(1 + returns)
    
    # Generate OHLC
    high = close * (1 + np.abs(np.random.normal(0, 0.01, days)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, days)))
    open_price = close * (1 + np.random.normal(0, 0.005, days))
    
    # Generate volume with some variation
    volume = volume_base * (1 + np.random.uniform(-0.3, 0.5, days))
    
    df = pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume.astype(int),
    }, index=dates)
    
    return df


class TestComputeTechnicalScoreWeekly:
    """Tests for compute_technical_score_weekly function."""
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame returns zero score."""
        df = pd.DataFrame()
        result = compute_technical_score_weekly(df, "TEST")
        
        assert result["score"] == 0.0
        assert result["cap_applied"] == 6.0
        assert "Insufficient price data" in result["data_gaps"]
    
    def test_insufficient_data(self):
        """Test with insufficient data (< 50 rows)."""
        df = create_mock_ohlcv(days=30)
        result = compute_technical_score_weekly(df, "TEST")
        
        assert result["score"] == 0.0
        assert "Insufficient price data" in result["data_gaps"]
    
    def test_valid_data_returns_score(self):
        """Test with valid data returns a score."""
        df = create_mock_ohlcv(days=300)
        result = compute_technical_score_weekly(df, "TEST")
        
        assert 0 <= result["score"] <= 10
        assert "evidence" in result
        assert isinstance(result["evidence"], dict)
    
    def test_score_components(self):
        """Test that score components are computed."""
        df = create_mock_ohlcv(days=300)
        result = compute_technical_score_weekly(df, "TEST")
        
        evidence = result["evidence"]
        
        # Check all expected evidence fields exist
        assert "within_5pct_52w_high" in evidence
        assert "volume_ratio_3d_to_20d" in evidence
        assert "rsi14" in evidence
        assert "above_ma10_ma20_ma50" in evidence
        assert "realized_vol_5d_ann_pct" in evidence
    
    def test_near_52w_high_bonus(self):
        """Test that being near 52W high increases score."""
        # Create data that ends near 52W high
        df = create_mock_ohlcv(days=300, trend=0.002)
        result = compute_technical_score_weekly(df, "TEST")
        
        # Should have positive score if near high
        if result["evidence"].get("within_5pct_52w_high"):
            assert result["score"] >= 2.0
    
    def test_rsi_in_range_bonus(self):
        """Test RSI scoring logic."""
        df = create_mock_ohlcv(days=300)
        result = compute_technical_score_weekly(df, "TEST")
        
        rsi_val = result["evidence"].get("rsi14")
        
        if rsi_val is not None and 50 <= rsi_val <= 70:
            # Should have gotten RSI bonus
            assert result["score"] >= 2.0


class TestComputeScore30dBreakout:
    """Tests for compute_score_30d_breakout function."""
    
    def test_basic_calculation(self):
        """Test basic breakout score calculation."""
        score = compute_score_30d_breakout(
            rvol_val=2.5,
            atr_pct_val=5.0,
            rsi14_val=60.0,
            dist_52w_high_pct=5.0,
            above_ma20=True,
            above_ma50=True,
        )
        
        assert score > 0
        assert isinstance(score, float)
    
    def test_high_rvol_increases_score(self):
        """Test that higher RVOL increases score."""
        base_score = compute_score_30d_breakout(
            rvol_val=2.0,
            atr_pct_val=5.0,
            rsi14_val=60.0,
            dist_52w_high_pct=5.0,
            above_ma20=True,
            above_ma50=True,
        )
        
        high_rvol_score = compute_score_30d_breakout(
            rvol_val=4.0,  # Higher RVOL
            atr_pct_val=5.0,
            rsi14_val=60.0,
            dist_52w_high_pct=5.0,
            above_ma20=True,
            above_ma50=True,
        )
        
        assert high_rvol_score > base_score
    
    def test_ma_structure_bonus(self):
        """Test that MA structure affects score."""
        both_ma_score = compute_score_30d_breakout(
            rvol_val=2.5,
            atr_pct_val=5.0,
            rsi14_val=60.0,
            dist_52w_high_pct=5.0,
            above_ma20=True,
            above_ma50=True,
        )
        
        no_ma_score = compute_score_30d_breakout(
            rvol_val=2.5,
            atr_pct_val=5.0,
            rsi14_val=60.0,
            dist_52w_high_pct=5.0,
            above_ma20=False,
            above_ma50=False,
        )
        
        assert both_ma_score > no_ma_score


class TestComputeScore30dReversal:
    """Tests for compute_score_30d_reversal function."""
    
    def test_basic_calculation(self):
        """Test basic reversal score calculation."""
        score = compute_score_30d_reversal(
            rvol_val=2.5,
            atr_pct_val=5.0,
            rsi14_val=28.0,
            dist_52w_high_pct=25.0,
        )
        
        assert score > 0
        assert isinstance(score, float)
    
    def test_low_rsi_preferred(self):
        """Test that lower RSI (oversold) is preferred for reversals."""
        low_rsi_score = compute_score_30d_reversal(
            rvol_val=2.5,
            atr_pct_val=5.0,
            rsi14_val=25.0,  # More oversold
            dist_52w_high_pct=25.0,
        )
        
        high_rsi_score = compute_score_30d_reversal(
            rvol_val=2.5,
            atr_pct_val=5.0,
            rsi14_val=40.0,  # Less oversold
            dist_52w_high_pct=25.0,
        )
        
        # Lower RSI should score at least as well for reversals
        assert low_rsi_score >= high_rsi_score - 1  # Allow small margin
    
    def test_far_from_high_bonus(self):
        """Test that distance from high affects reversal score."""
        far_score = compute_score_30d_reversal(
            rvol_val=2.5,
            atr_pct_val=5.0,
            rsi14_val=28.0,
            dist_52w_high_pct=40.0,  # Far from high
        )
        
        near_score = compute_score_30d_reversal(
            rvol_val=2.5,
            atr_pct_val=5.0,
            rsi14_val=28.0,
            dist_52w_high_pct=10.0,  # Near high
        )
        
        # For reversals, being far from high means more upside potential
        assert far_score > near_score


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
