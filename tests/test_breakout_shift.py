# tests/test_breakout_shift.py
"""
Test that breakout detection uses shifted (previous day) Donchian levels
to avoid same-bar lookahead bias.
"""
import pandas as pd
import numpy as np
import pytest

from src.core.breakout import compute_breakout_features, score_breakout


def test_breakout_uses_prev_donchian_level():
    """
    Test that Donchian channel uses previous day's high, not current day's.
    
    If we have 21 days where day 21 makes a new high, the shifted donchian
    should use the prior 20 days' maximum (not include day 21's high).
    """
    dates = pd.date_range("2025-01-01", periods=21, freq="D")
    
    # Days 1-20: highs from 1 to 20
    # Day 21: spike to 100
    high = list(range(1, 21)) + [100]
    close = list(range(1, 21)) + [100]
    low = [h - 0.5 for h in high]
    vol = [100] * 21
    
    df = pd.DataFrame({
        "Open": close,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": vol
    }, index=dates)

    f = compute_breakout_features(df)
    
    # Previous-day donchian20 should be max(high of first 20 days) = 20, not 100
    assert round(f["donchian20_prev"], 6) == 20.0
    
    # Day 21 close (100) is above donchian20_prev (20), so it's a breakout
    assert f["is_breakout_20"] is True


def test_breakout_not_triggered_when_below_prev_high():
    """Test that breakout is not triggered when close is below previous Donchian high."""
    dates = pd.date_range("2025-01-01", periods=21, freq="D")
    
    # All days have same high of 10, close is below
    high = [10] * 21
    close = [9] * 21  # Below the Donchian high
    low = [8] * 21
    vol = [100] * 21
    
    df = pd.DataFrame({
        "Open": close,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": vol
    }, index=dates)

    f = compute_breakout_features(df)
    
    assert f["donchian20_prev"] == 10.0
    assert f["is_breakout_20"] is False  # 9 is not > 10


def test_breakout_score_components():
    """Test that breakout scoring rewards trigger, volume, and strength."""
    dates = pd.date_range("2025-01-01", periods=25, freq="D")
    
    # Create a clear breakout setup
    high = [10] * 20 + [11, 12, 13, 14, 15]  # Breaking out
    close = [9] * 20 + [11, 12, 13, 14, 15]  # Closing above prev high
    low = [8] * 25
    vol = [100] * 20 + [200, 250, 300, 300, 300]  # Volume spike
    
    df = pd.DataFrame({
        "Open": close,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": vol
    }, index=dates)

    result = score_breakout(df)
    
    # Should have a positive breakout score
    assert result.breakout_score > 0
    
    # Should detect breakout
    assert result.evidence["is_breakout_20"] is True
    
    # Volume ratio should be elevated (recent vol > historical)
    assert result.evidence["vol_ratio_3_20"] > 1.0


def test_breakout_handles_insufficient_data():
    """Test that breakout handles DataFrames with insufficient history."""
    dates = pd.date_range("2025-01-01", periods=10, freq="D")  # Only 10 days
    
    df = pd.DataFrame({
        "Open": [10] * 10,
        "High": [11] * 10,
        "Low": [9] * 10,
        "Close": [10] * 10,
        "Volume": [100] * 10
    }, index=dates)

    f = compute_breakout_features(df)
    
    # Should return NaN for features requiring more history
    assert np.isnan(f["donchian20_prev"])
    assert f["is_breakout_20"] is False


def test_donchian55_also_shifted():
    """Test that 55-day Donchian is also properly shifted."""
    # Need 56 days so that day 56 spike is NOT included in the shifted 55-day window
    dates = pd.date_range("2025-01-01", periods=56, freq="D")
    
    # Days 1-55: highs all at 10
    # Day 56: spike to 1000 (should NOT be in the shifted donchian)
    high = [10] * 55 + [1000]
    close = [10] * 55 + [1000]
    low = [9] * 56
    vol = [100] * 56
    
    df = pd.DataFrame({
        "Open": close,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": vol
    }, index=dates)

    f = compute_breakout_features(df)
    
    # donchian55_prev should be 10 (the max of days 1-55), not 1000 (day 56's spike)
    # Because shift(1) means we look at the 55-day max as of day 55, not day 56
    assert round(f["donchian55_prev"], 6) == 10.0
