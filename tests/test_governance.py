# tests/test_governance.py
"""
Tests for model governance guardrails.
"""
import pandas as pd
import pytest
import tempfile
import os
import json

from src.calibration.guardrails import (
    is_model_eligible,
    check_eligibility_details,
    filter_incomplete_rows,
    validate_no_future_features,
    MIN_SAMPLES_REGIME,
    MIN_POSITIVES_REGIME,
    MIN_SAMPLES_GLOBAL,
    MIN_POSITIVES_GLOBAL,
)
from src.governance.performance_monitor import (
    rolling_metrics,
    check_decay,
    DecayThresholds,
)
from src.governance.retrain_policy import (
    should_retrain,
    generate_model_version,
    build_version_metadata,
)
from src.governance.artifacts import (
    GovernanceRecord,
    write_governance,
    GovernanceContext,
)


# G1: Data Sufficiency Tests

def test_model_eligibility():
    """Test basic model eligibility check."""
    # Eligible regime model
    assert is_model_eligible(300, 30, is_regime_model=True)
    
    # Not eligible - too few samples
    assert not is_model_eligible(100, 10, is_regime_model=True)
    
    # Not eligible - too few positives
    assert not is_model_eligible(300, 10, is_regime_model=True)


def test_model_eligibility_global():
    """Test global model eligibility (stricter thresholds)."""
    # Global model needs more samples
    assert is_model_eligible(600, 60, is_regime_model=False)
    assert not is_model_eligible(400, 40, is_regime_model=False)


def test_eligibility_details():
    """Test detailed eligibility check."""
    result = check_eligibility_details(300, 30, is_regime_model=True)
    
    assert result["eligible"] is True
    assert result["samples_sufficient"] is True
    assert result["positives_sufficient"] is True
    assert result["skip_reason"] is None


def test_eligibility_details_failure():
    """Test eligibility failure details."""
    result = check_eligibility_details(50, 10, is_regime_model=True)
    
    assert result["eligible"] is False
    assert "insufficient_samples" in result["skip_reason"]


def test_filter_incomplete_rows():
    """Test filtering rows with missing features."""
    df = pd.DataFrame({
        "feature1": [1, 2, None, 4, None],
        "feature2": [1, None, None, 4, None],
        "feature3": [1, 2, 3, 4, None],
    })
    
    # With threshold 0.7 (need 2 of 3 non-null)
    result = filter_incomplete_rows(df, ["feature1", "feature2", "feature3"], threshold=0.67)
    
    # Row 0: 3/3, Row 1: 2/3, Row 2: 1/3, Row 3: 3/3, Row 4: 0/3
    assert len(result) == 3  # Rows 0, 1, 3


def test_filter_incomplete_rows_empty():
    """Test filtering empty dataframe."""
    df = pd.DataFrame()
    result = filter_incomplete_rows(df, ["a", "b"])
    assert result.empty


# G3: Integrity Tests

def test_validate_no_future_features():
    """Test detection of forbidden look-ahead columns."""
    # Should pass
    validate_no_future_features(["technical_score", "rsi14", "close_price"])
    
    # Should fail
    with pytest.raises(ValueError, match="Forbidden"):
        validate_no_future_features(["technical_score", "future_return"])
    
    with pytest.raises(ValueError, match="Forbidden"):
        validate_no_future_features(["forward_price", "rsi14"])
    
    with pytest.raises(ValueError, match="Forbidden"):
        validate_no_future_features(["next_day_open", "close"])


# G2: Performance Monitoring Tests

def test_rolling_metrics_basic():
    """Test rolling metrics computation."""
    trades = pd.DataFrame({
        "hit": [True, False, True, True, False],
        "mfe_pct": [15.0, 8.0, 12.0, 10.0, 5.0],
        "mae_pct": [-3.0, -10.0, -2.0, -4.0, -15.0],
    })
    
    metrics = rolling_metrics(trades, window=5)
    
    assert "hit_rate" in metrics
    assert metrics["hit_rate"] == 0.6  # 3/5
    assert "avg_mfe" in metrics
    assert "avg_mae" in metrics
    assert "expectancy" in metrics


def test_rolling_metrics_empty():
    """Test rolling metrics with empty data."""
    metrics = rolling_metrics(pd.DataFrame(), window=10)
    assert metrics == {}


def test_check_decay_no_decay():
    """Test decay check when model is healthy."""
    live_metrics = {"hit_rate": 0.55, "avg_mae": -5.0, "expectancy": 0.5}
    training_metrics = {"positive_rate": 0.50, "avg_mae": -4.0}
    
    result = check_decay(live_metrics, training_metrics)
    
    assert result["decay_detected"] is False
    assert result["triggers"] == []


def test_check_decay_hit_rate_decay():
    """Test decay detection when hit rate drops."""
    live_metrics = {"hit_rate": 0.20, "avg_mae": -5.0, "expectancy": -0.5}
    training_metrics = {"positive_rate": 0.60, "avg_mae": -4.0}
    
    result = check_decay(live_metrics, training_metrics)
    
    assert result["decay_detected"] is True
    assert len(result["triggers"]) > 0


def test_check_decay_negative_expectancy():
    """Test decay detection with negative expectancy."""
    live_metrics = {"hit_rate": 0.50, "avg_mae": -5.0, "expectancy": -0.1}
    training_metrics = {"positive_rate": 0.50, "avg_mae": -4.0}
    
    result = check_decay(live_metrics, training_metrics)
    
    assert result["decay_detected"] is True
    assert any("negative_expectancy" in t for t in result["triggers"])


# G4: Retraining Policy Tests

def test_should_retrain_no_prior():
    """Test retraining when no prior training exists."""
    result = should_retrain(last_train_date=None, new_samples=100)
    
    assert result["should_retrain"] is True
    assert result["reason"] == "no_prior_training"


def test_should_retrain_insufficient_samples():
    """Test retraining blocked due to insufficient new samples."""
    result = should_retrain(last_train_date="2026-01-01", new_samples=30)
    
    assert result["should_retrain"] is False
    assert "insufficient_new_samples" in result["reason"]


def test_should_retrain_too_recent():
    """Test retraining blocked due to too recent training."""
    from datetime import datetime
    today = datetime.utcnow().strftime("%Y-%m-%d")
    
    result = should_retrain(last_train_date=today, new_samples=100)
    
    assert result["should_retrain"] is False
    assert "too_recent" in result["reason"]


def test_generate_model_version():
    """Test model version generation."""
    version = generate_model_version("bull", "2026-01-15", version_num=1)
    
    assert "calibration_bull" in version
    assert "2026-01-15" in version
    assert "v1" in version


def test_build_version_metadata():
    """Test version metadata building."""
    meta = build_version_metadata(
        regime="bull",
        train_start="2025-10-01",
        train_end="2026-01-10",
        n_rows=1000,
        n_positives=250,
    )
    
    assert meta["regime"] == "bull"
    assert meta["rows"] == 1000
    assert meta["positives"] == 250
    assert meta["positive_rate"] == 0.25
    assert "training_window" in meta


# G6: Governance Artifacts Tests

def test_governance_record_creation():
    """Test GovernanceRecord dataclass."""
    record = GovernanceRecord(
        calibration_used=True,
        model_version="calibration_bull_2026-01-15_v1",
        regime="bull",
        eligibility_passed=True,
        decay_detected=False,
    )
    
    assert record.calibration_used is True
    assert record.regime == "bull"
    assert record.run_timestamp is not None


def test_governance_record_to_dict():
    """Test GovernanceRecord serialization."""
    record = GovernanceRecord(
        calibration_used=True,
        regime="bull",
    )
    
    d = record.to_dict()
    
    assert isinstance(d, dict)
    assert d["calibration_used"] is True
    assert d["regime"] == "bull"


def test_write_governance():
    """Test writing governance JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "governance.json")
        
        payload = {
            "calibration_used": True,
            "regime": "bull",
            "decay_detected": False,
        }
        
        write_governance(path, payload)
        
        assert os.path.exists(path)
        
        with open(path) as f:
            data = json.load(f)
        
        assert data["calibration_used"] is True
        assert data["regime"] == "bull"
        assert "run_timestamp" in data


def test_governance_context():
    """Test GovernanceContext helper."""
    with GovernanceContext() as gov:
        gov.set_regime("bull")
        gov.set_calibration_used(True, "calibration_bull_v1")
        gov.add_flag("test_flag")
    
    record = gov.record
    
    assert record.regime == "bull"
    assert record.calibration_used is True
    assert record.model_version == "calibration_bull_v1"
    assert "test_flag" in record.governance_flags
