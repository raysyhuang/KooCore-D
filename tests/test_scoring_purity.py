# tests/test_scoring_purity.py
"""
Test that scoring module does not import pandas (leak-proof scoring).

This ensures that the scoring logic cannot access raw DataFrames,
preventing any possibility of lookahead bias in the scoring code.
"""
import inspect
import pytest

import src.core.score_weekly as sw


def test_score_weekly_no_pandas_import():
    """
    Verify that score_weekly.py does not import pandas.
    
    The scoring module should only work with FeatureSet values,
    not raw DataFrames. This architectural constraint prevents
    lookahead leakage in the scoring logic.
    """
    src = inspect.getsource(sw)
    lines = src.split('\n')
    
    # Check for actual import statements (not mentions in docstrings/comments)
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Skip comments
        if stripped.startswith('#'):
            continue
        # Skip lines that are clearly inside docstrings (start with quotes or just text)
        if stripped.startswith('"""') or stripped.startswith("'''"):
            continue
        # Check for actual import statements
        if stripped.startswith('import pandas') or stripped.startswith('from pandas'):
            pytest.fail(f"Line {i+1} imports pandas: {stripped}")
        # Check for aliased pandas usage
        if stripped.startswith('import ') and ' as pd' in stripped:
            pytest.fail(f"Line {i+1} imports pandas as pd: {stripped}")
    
    # Also check that pd. is not used (in case of aliased import elsewhere)
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Skip comments and docstrings
        if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
            continue
        if 'pd.DataFrame' in line or 'pd.Series' in line:
            pytest.fail(f"Line {i+1} uses pandas DataFrame/Series: {stripped}")


def test_score_weekly_only_uses_featureset():
    """
    Verify that score_weekly function signature only accepts FeatureSet.
    """
    sig = inspect.signature(sw.score_weekly)
    params = list(sig.parameters.keys())
    
    # Should only have 'features' parameter
    assert params == ['features'], f"Unexpected parameters: {params}"
    
    # Check the type annotation
    features_param = sig.parameters['features']
    annotation = features_param.annotation
    
    # Should be annotated as FeatureSet
    assert 'FeatureSet' in str(annotation), f"Parameter should be FeatureSet, got: {annotation}"


def test_score_result_is_immutable():
    """Verify that ScoreResult is a frozen dataclass (immutable)."""
    from src.core.score_weekly import ScoreResult
    
    result = ScoreResult(
        score=7.5,
        cap_applied=None,
        evidence={"test": 1},
        data_gaps=[]
    )
    
    # Should not be able to modify frozen dataclass
    with pytest.raises(AttributeError):
        result.score = 8.0


def test_featureset_is_immutable():
    """Verify that FeatureSet is a frozen dataclass (immutable)."""
    from src.core.types import FeatureSet
    
    fs = FeatureSet(
        ticker="TEST",
        asof_date="2025-01-01",
        last_close=100.0,
    )
    
    # Should not be able to modify frozen dataclass
    with pytest.raises(AttributeError):
        fs.last_close = 200.0


def test_score_weekly_handles_missing_data():
    """Test that scoring gracefully handles missing feature data."""
    from src.core.types import FeatureSet
    
    # Create a minimal FeatureSet with mostly None values
    fs = FeatureSet(
        ticker="TEST",
        asof_date="2025-01-01",
        last_close=None,  # Missing price data
    )
    
    result = sw.score_weekly(fs)
    
    # Should return a valid result with low score
    assert result.score == 0.0
    assert "Insufficient price data" in result.data_gaps


def test_score_weekly_returns_expected_fields():
    """Test that score_weekly returns all expected fields in evidence."""
    from src.core.types import FeatureSet
    
    fs = FeatureSet(
        ticker="TEST",
        asof_date="2025-01-01",
        last_close=100.0,
        last_volume=1000000.0,
        rsi14=60.0,
        ma10=95.0,
        ma20=90.0,
        ma50=85.0,
        vol_ratio_3_20=1.8,
        realized_vol_5d_ann_pct=25.0,
        dist_52w_high_pct=3.0,
    )
    
    result = sw.score_weekly(fs)
    
    # Check expected evidence fields
    assert "within_5pct_52w_high" in result.evidence
    assert "volume_ratio_3d_to_20d" in result.evidence
    assert "rsi14" in result.evidence
    assert "above_ma10_ma20_ma50" in result.evidence
    assert "realized_vol_5d_ann_pct" in result.evidence
    
    # Score should be reasonable for good setup
    assert result.score > 0
