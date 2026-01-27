# tests/test_mae_mfe.py
"""
Tests for MAE/MFE (Maximum Adverse/Favorable Excursion) computation.
"""
import pandas as pd
import pytest

from src.backtest.metrics import compute_path_metrics, compute_expectancy


def test_mfe_mae_basic():
    """Test basic MFE/MAE computation."""
    idx = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-04"])
    df = pd.DataFrame({
        "Open": [10, 10, 10],
        "High": [11, 12, 11],  # Max high = 12 = +20%
        "Low": [9, 8, 9],       # Min low = 8 = -20%
        "Close": [10, 11, 10],
        "Volume": [100, 100, 100]
    }, index=idx)

    pm = compute_path_metrics(
        df=df,
        entry_px=10,
        start_dt=pd.to_datetime("2025-01-01"),
        horizon_days=3
    )
    
    assert round(pm.mfe, 2) == 20.0   # (12-10)/10 * 100
    assert round(pm.mae, 2) == -20.0  # (8-10)/10 * 100


def test_target_hit_detection():
    """Test that target hit is detected correctly."""
    idx = pd.to_datetime(["2025-01-02", "2025-01-03"])
    df = pd.DataFrame({
        "Open": [10, 10],
        "High": [10, 12],  # Day 2: +20% from entry
        "Low": [10, 10],
        "Close": [10, 11],
        "Volume": [100, 100]
    }, index=idx)

    pm = compute_path_metrics(
        df=df,
        entry_px=10,
        start_dt=pd.to_datetime("2025-01-01"),
        horizon_days=5,
        target_pct=10.0
    )
    
    assert pm.hit is True
    assert pm.exit_reason == "target_hit"
    assert pm.days_to_hit is not None


def test_stop_hit_detection():
    """Test that stop hit is detected correctly."""
    idx = pd.to_datetime(["2025-01-02", "2025-01-03"])
    df = pd.DataFrame({
        "Open": [10, 9],
        "High": [10, 9.5],
        "Low": [9.5, 8.5],  # Day 2: -15% from entry
        "Close": [9.5, 9],
        "Volume": [100, 100]
    }, index=idx)

    pm = compute_path_metrics(
        df=df,
        entry_px=10,
        start_dt=pd.to_datetime("2025-01-01"),
        horizon_days=5,
        target_pct=10.0,
        stop_pct=10.0  # 10% stop
    )
    
    assert pm.hit is False
    assert pm.exit_reason == "stop_hit"


def test_timeout_when_no_target_or_stop():
    """Test timeout when neither target nor stop is hit."""
    idx = pd.to_datetime(["2025-01-02", "2025-01-03"])
    df = pd.DataFrame({
        "Open": [10, 10],
        "High": [10.5, 10.3],  # Small moves, no target hit
        "Low": [9.8, 9.9],     # No stop hit either
        "Close": [10, 10],
        "Volume": [100, 100]
    }, index=idx)

    pm = compute_path_metrics(
        df=df,
        entry_px=10,
        start_dt=pd.to_datetime("2025-01-01"),
        horizon_days=2,
        target_pct=10.0,
        stop_pct=10.0
    )
    
    assert pm.hit is False
    assert pm.exit_reason == "timeout"
    assert pm.days_to_hit is None


def test_no_data_returns_empty_metrics():
    """Test that missing data returns appropriate metrics."""
    pm = compute_path_metrics(
        df=pd.DataFrame(),
        entry_px=10,
        start_dt=pd.to_datetime("2025-01-01"),
        horizon_days=5
    )
    
    assert pm.hit is False
    assert pm.exit_reason == "no_data"
    assert pm.mfe is None
    assert pm.mae is None


def test_invalid_entry_price_returns_empty():
    """Test that invalid entry price returns empty metrics."""
    idx = pd.to_datetime(["2025-01-02"])
    df = pd.DataFrame({
        "Open": [10],
        "High": [11],
        "Low": [9],
        "Close": [10],
        "Volume": [100]
    }, index=idx)

    pm = compute_path_metrics(
        df=df,
        entry_px=0,  # Invalid
        start_dt=pd.to_datetime("2025-01-01"),
        horizon_days=5
    )
    
    assert pm.exit_reason == "no_data"


def test_compute_expectancy():
    """Test expectancy computation from outcomes."""
    outcomes = [
        {"hit": True, "mfe_pct": 15.0, "mae_pct": -3.0},
        {"hit": True, "mfe_pct": 12.0, "mae_pct": -5.0},
        {"hit": False, "mfe_pct": 5.0, "mae_pct": -8.0},
        {"hit": False, "mfe_pct": 3.0, "mae_pct": -12.0},
    ]
    
    result = compute_expectancy(outcomes, target_pct=10.0)
    
    assert result["hit_rate"] == 0.5  # 2/4
    assert result["trade_count"] == 4
    assert result["avg_mfe_pct"] is not None
    assert result["avg_mae_pct"] is not None


def test_compute_expectancy_empty():
    """Test expectancy with empty outcomes."""
    result = compute_expectancy([], target_pct=10.0)
    
    assert result["trade_count"] == 0
    assert result["hit_rate"] is None
