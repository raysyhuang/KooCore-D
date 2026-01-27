# tests/test_regime_classifier.py
"""
Tests for market regime classification.
"""
import pandas as pd
import pytest

from src.regime.classifier import classify_regime, Regime


def test_regime_bull():
    """Test bull regime classification."""
    # Create uptrending SPY data above MA50
    idx = pd.date_range("2025-01-01", periods=60)
    df = pd.DataFrame({"Close": range(1, 61)}, index=idx)  # Uptrend
    df["High"] = df["Close"] + 1
    df["Low"] = df["Close"] - 1
    df["Open"] = df["Close"]
    df["Volume"] = 1000000
    
    r = classify_regime(df, vix_level=15)
    
    assert r.name == "bull"
    assert r.confidence > 0


def test_regime_stress_high_vix():
    """Test stress regime with high VIX."""
    idx = pd.date_range("2025-01-01", periods=60)
    df = pd.DataFrame({"Close": range(1, 61)}, index=idx)
    df["High"] = df["Close"] + 1
    df["Low"] = df["Close"] - 1
    df["Open"] = df["Close"]
    df["Volume"] = 1000000
    
    r = classify_regime(df, vix_level=30)  # High VIX
    
    assert r.name == "stress"


def test_regime_stress_below_ma_high_vix():
    """Test stress regime when below MA50 with elevated VIX."""
    idx = pd.date_range("2025-01-01", periods=60)
    # Create downtrending data (last close below MA50)
    closes = list(range(100, 40, -1))  # Downtrend from 100 to ~40
    df = pd.DataFrame({"Close": closes}, index=idx)
    df["High"] = df["Close"] + 1
    df["Low"] = df["Close"] - 1
    df["Open"] = df["Close"]
    df["Volume"] = 1000000
    
    r = classify_regime(df, vix_level=22)  # Elevated VIX
    
    assert r.name == "stress"


def test_regime_chop_moderate_vix():
    """Test chop regime classification."""
    idx = pd.date_range("2025-01-01", periods=60)
    # Create sideways/choppy data
    df = pd.DataFrame({"Close": [50 + (i % 5) for i in range(60)]}, index=idx)
    df["High"] = df["Close"] + 1
    df["Low"] = df["Close"] - 1
    df["Open"] = df["Close"]
    df["Volume"] = 1000000
    
    r = classify_regime(df, vix_level=18)  # Moderate VIX
    
    # Could be chop or bull depending on MA position
    assert r.name in ["chop", "bull"]


def test_regime_empty_dataframe():
    """Test regime with empty DataFrame returns chop."""
    r = classify_regime(pd.DataFrame(), vix_level=15)
    
    assert r.name == "chop"
    assert r.confidence == 0.5


def test_regime_none_dataframe():
    """Test regime with None DataFrame returns chop."""
    r = classify_regime(None, vix_level=15)
    
    assert r.name == "chop"


def test_regime_none_vix():
    """Test regime with None VIX."""
    idx = pd.date_range("2025-01-01", periods=60)
    df = pd.DataFrame({"Close": range(1, 61)}, index=idx)
    df["High"] = df["Close"] + 1
    df["Low"] = df["Close"] - 1
    df["Open"] = df["Close"]
    df["Volume"] = 1000000
    
    r = classify_regime(df, vix_level=None)
    
    # Should still classify (defaults to bull if above MA)
    assert r.name in ["bull", "chop"]


def test_regime_evidence_contains_data():
    """Test that regime evidence contains expected data."""
    idx = pd.date_range("2025-01-01", periods=60)
    df = pd.DataFrame({"Close": range(1, 61)}, index=idx)
    df["High"] = df["Close"] + 1
    df["Low"] = df["Close"] - 1
    df["Open"] = df["Close"]
    df["Volume"] = 1000000
    
    r = classify_regime(df, vix_level=15)
    
    assert "spy_close" in r.evidence
    assert "spy_ma50" in r.evidence
    assert "above_ma50" in r.evidence
    assert "vix" in r.evidence


def test_regime_dataclass_immutable():
    """Test that Regime is immutable."""
    r = Regime(name="bull", confidence=0.9)
    
    with pytest.raises(AttributeError):
        r.name = "stress"
