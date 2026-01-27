# tests/test_calibration_dataset.py
"""
Tests for calibration dataset building.
"""
import pandas as pd
import pytest

from src.calibration.dataset import build_dataset, split_by_date


def test_dataset_join():
    """Test that snapshots and outcomes are joined correctly."""
    snapshots = pd.DataFrame({
        "ticker": ["A", "B"],
        "asof_date": ["2025-01-01", "2025-01-01"],
        "technical_score": [8, 7],
    })
    outcomes = pd.DataFrame({
        "ticker": ["A", "B"],
        "asof_date": ["2025-01-01", "2025-01-01"],
        "hit": [1, 0],
    })
    
    df = build_dataset(snapshots, outcomes)
    
    assert len(df) == 2
    assert "hit" in df.columns
    assert "technical_score" in df.columns


def test_dataset_join_partial_match():
    """Test that only matching rows are included."""
    snapshots = pd.DataFrame({
        "ticker": ["A", "B", "C"],
        "asof_date": ["2025-01-01", "2025-01-01", "2025-01-01"],
        "technical_score": [8, 7, 6],
    })
    outcomes = pd.DataFrame({
        "ticker": ["A", "B"],  # No outcome for C
        "asof_date": ["2025-01-01", "2025-01-01"],
        "hit": [1, 0],
    })
    
    df = build_dataset(snapshots, outcomes)
    
    assert len(df) == 2  # Only A and B match


def test_dataset_join_empty_snapshots():
    """Test handling of empty snapshots."""
    snapshots = pd.DataFrame()
    outcomes = pd.DataFrame({
        "ticker": ["A"],
        "asof_date": ["2025-01-01"],
        "hit": [1],
    })
    
    df = build_dataset(snapshots, outcomes)
    
    assert df.empty


def test_dataset_join_empty_outcomes():
    """Test handling of empty outcomes."""
    snapshots = pd.DataFrame({
        "ticker": ["A"],
        "asof_date": ["2025-01-01"],
        "technical_score": [8],
    })
    outcomes = pd.DataFrame()
    
    df = build_dataset(snapshots, outcomes)
    
    assert df.empty


def test_split_by_date():
    """Test time-series split by date."""
    df = pd.DataFrame({
        "ticker": ["A", "B", "C", "D"],
        "asof_date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
        "hit": [1, 0, 1, 0],
    })
    
    train, test = split_by_date(df, cutoff_date="2025-01-03")
    
    assert len(train) == 2  # Jan 1-2
    assert len(test) == 2   # Jan 3-4


def test_split_by_date_empty():
    """Test split with empty DataFrame."""
    df = pd.DataFrame()
    
    train, test = split_by_date(df, cutoff_date="2025-01-01")
    
    assert train.empty
    assert test.empty
