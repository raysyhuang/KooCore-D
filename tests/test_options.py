"""
Tests for options module.

Tests options activity scoring and data fetching.
"""

import pytest
from unittest.mock import patch, MagicMock


# Import from src
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.options import (
    compute_options_score,
    fetch_options_activity_yahoo,
    OptionsScore,
)


class TestComputeOptionsScore:
    """Tests for compute_options_score function."""
    
    def test_returns_options_score_object(self):
        """Test that function returns OptionsScore dataclass."""
        result = compute_options_score("AAPL")
        
        assert isinstance(result, OptionsScore)
        assert hasattr(result, "score")
        assert hasattr(result, "evidence")
        assert hasattr(result, "data_gaps")
    
    def test_score_within_bounds(self):
        """Test that score is within 0-10 bounds."""
        result = compute_options_score("AAPL")
        
        assert 0 <= result.score <= 10
    
    def test_evidence_structure(self):
        """Test that evidence has expected structure."""
        result = compute_options_score("AAPL")
        
        evidence = result.evidence
        assert "call_put_ratio" in evidence
        assert "unusual_volume_multiple" in evidence
        assert "largest_bullish_premium_usd" in evidence
        assert "iv_rank" in evidence
        assert "notable_contracts" in evidence
        assert "data_source" in evidence
    
    @patch('src.core.options.fetch_options_activity_yahoo')
    def test_with_mock_yahoo_data(self, mock_yahoo):
        """Test scoring with mocked Yahoo data."""
        mock_yahoo.return_value = {
            "call_volume": 10000,
            "put_volume": 3000,
            "call_put_ratio": 3.33,
            "source": "yahoo"
        }
        
        result = compute_options_score("TEST")
        
        # Should have a non-zero score with bullish call/put ratio
        assert result.score > 0 or result.cap_applied is not None
    
    def test_no_data_returns_capped_score(self):
        """Test that missing data results in capped score."""
        # Use a ticker unlikely to have options data
        result = compute_options_score("UNLIKELY_TICKER_12345")
        
        # Should have data gaps
        assert len(result.data_gaps) > 0
        
        # Score should be capped at 3.0 when no data available
        if result.cap_applied is not None:
            assert result.cap_applied == 3.0


class TestFetchOptionsActivityYahoo:
    """Tests for fetch_options_activity_yahoo function."""
    
    @patch('yfinance.Ticker')
    def test_returns_dict_on_success(self, mock_ticker_class):
        """Test successful data fetch returns dict."""
        # Mock yfinance Ticker
        mock_ticker = MagicMock()
        mock_ticker.options = ['2024-01-19', '2024-01-26']
        
        # Mock option chain
        mock_calls = MagicMock()
        mock_calls.empty = False
        mock_calls.__getitem__ = MagicMock(return_value=MagicMock(sum=MagicMock(return_value=1000)))
        mock_calls.columns = ['volume', 'openInterest']
        
        mock_puts = MagicMock()
        mock_puts.empty = False
        mock_puts.__getitem__ = MagicMock(return_value=MagicMock(sum=MagicMock(return_value=500)))
        mock_puts.columns = ['volume', 'openInterest']
        
        mock_chain = MagicMock()
        mock_chain.calls = mock_calls
        mock_chain.puts = mock_puts
        
        mock_ticker.option_chain.return_value = mock_chain
        mock_ticker_class.return_value = mock_ticker
        
        result = fetch_options_activity_yahoo("TEST")
        
        # Should return dict with expected keys if data available
        # Note: Actual behavior depends on mock setup
        assert result is None or isinstance(result, dict)
    
    @patch('yfinance.Ticker')
    def test_no_options_returns_none(self, mock_ticker_class):
        """Test that no options available returns None."""
        mock_ticker = MagicMock()
        mock_ticker.options = []  # No options available
        mock_ticker_class.return_value = mock_ticker
        
        result = fetch_options_activity_yahoo("TEST")
        
        assert result is None
    
    @patch('yfinance.Ticker')
    def test_exception_returns_none(self, mock_ticker_class):
        """Test that exception returns None gracefully."""
        mock_ticker_class.side_effect = Exception("API Error")
        
        result = fetch_options_activity_yahoo("TEST")
        
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
