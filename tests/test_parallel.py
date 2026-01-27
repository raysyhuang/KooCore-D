"""
Tests for parallel processing module.

Tests parallel execution and batch processing.
"""

import pytest
import time
from unittest.mock import MagicMock


# Import from src
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.parallel import (
    parallel_map,
    parallel_batch_map,
    ParallelResult,
    ParallelScreener,
    create_ticker_processor,
)


class TestParallelMap:
    """Tests for parallel_map function."""
    
    def test_empty_items(self):
        """Test with empty items list."""
        result = parallel_map(
            func=lambda x: x,
            items=[],
            show_progress=False
        )
        
        assert isinstance(result, ParallelResult)
        assert result.success_count == 0
        assert result.failed_count == 0
    
    def test_successful_processing(self):
        """Test successful parallel processing."""
        items = ["a", "b", "c", "d", "e"]
        
        result = parallel_map(
            func=lambda x: x.upper(),
            items=items,
            show_progress=False
        )
        
        assert result.success_count == 5
        assert result.failed_count == 0
        
        # Check results
        results_dict = dict(result.success)
        assert results_dict["a"] == "A"
        assert results_dict["e"] == "E"
    
    def test_handles_failures(self):
        """Test that failures are captured."""
        def failing_func(x):
            if x == "fail":
                raise ValueError("Intentional failure")
            return x.upper()
        
        items = ["a", "fail", "c"]
        
        result = parallel_map(
            func=failing_func,
            items=items,
            show_progress=False
        )
        
        assert result.success_count == 2
        assert result.failed_count == 1
        
        # Check failed item
        failed_items = [item for item, _ in result.failed]
        assert "fail" in failed_items
    
    def test_success_rate(self):
        """Test success rate calculation."""
        def half_fail(x):
            if int(x) % 2 == 0:
                raise ValueError("Even number fails")
            return x
        
        items = ["1", "2", "3", "4"]
        
        result = parallel_map(
            func=half_fail,
            items=items,
            show_progress=False
        )
        
        assert result.success_rate == 0.5
    
    def test_duration_tracked(self):
        """Test that duration is tracked."""
        def slow_func(x):
            time.sleep(0.01)
            return x
        
        items = ["a", "b", "c"]
        
        result = parallel_map(
            func=slow_func,
            items=items,
            max_workers=2,
            show_progress=False
        )
        
        assert result.duration_seconds > 0


class TestParallelBatchMap:
    """Tests for parallel_batch_map function."""
    
    def test_empty_items(self):
        """Test with empty items list."""
        result = parallel_batch_map(
            func=lambda batch: {x: x for x in batch},
            items=[],
            show_progress=False
        )
        
        assert result.success_count == 0
    
    def test_batch_processing(self):
        """Test batch processing works correctly."""
        items = ["a", "b", "c", "d", "e", "f"]
        
        result = parallel_batch_map(
            func=lambda batch: {x: x.upper() for x in batch},
            items=items,
            batch_size=2,
            show_progress=False
        )
        
        assert result.success_count == 6
        
        results_dict = dict(result.success)
        assert results_dict["a"] == "A"
        assert results_dict["f"] == "F"
    
    def test_handles_batch_failures(self):
        """Test that batch failures are handled."""
        def failing_batch_func(batch):
            if "fail" in batch:
                raise ValueError("Batch contains fail")
            return {x: x.upper() for x in batch}
        
        items = ["a", "b", "fail", "d"]
        
        result = parallel_batch_map(
            func=failing_batch_func,
            items=items,
            batch_size=2,
            show_progress=False
        )
        
        # First batch ["a", "b"] should succeed
        # Second batch ["fail", "d"] should fail
        assert result.success_count == 2
        assert result.failed_count == 2


class TestParallelScreener:
    """Tests for ParallelScreener class."""
    
    @pytest.fixture
    def screener(self):
        """Create a screener instance."""
        return ParallelScreener(
            max_workers=4,
            batch_size=10,
            retry_failed=False,
        )
    
    def test_screen_tickers(self, screener):
        """Test screening tickers."""
        tickers = ["AAPL", "GOOGL", "MSFT"]
        
        def screen_func(ticker):
            return {"ticker": ticker, "score": len(ticker)}
        
        passed, dropped = screener.screen_tickers(
            tickers=tickers,
            screen_func=screen_func,
        )
        
        assert len(passed) == 3
        assert len(dropped) == 0
    
    def test_screen_tickers_with_filter(self, screener):
        """Test screening with filter function."""
        tickers = ["AAPL", "GOOGL", "MSFT"]
        
        def screen_func(ticker):
            return {"ticker": ticker, "score": len(ticker)}
        
        def filter_func(result):
            return result["score"] >= 5  # Only tickers with 5+ chars
        
        passed, dropped = screener.screen_tickers(
            tickers=tickers,
            screen_func=screen_func,
            filter_func=filter_func,
        )
        
        assert len(passed) == 1  # GOOGL
        assert len(dropped) == 2  # AAPL, MSFT
    
    def test_screen_tickers_handles_none(self, screener):
        """Test that None results are handled."""
        tickers = ["AAPL", "BAD", "MSFT"]
        
        def screen_func(ticker):
            if ticker == "BAD":
                return None
            return {"ticker": ticker}
        
        passed, dropped = screener.screen_tickers(
            tickers=tickers,
            screen_func=screen_func,
        )
        
        assert len(passed) == 2
        assert len(dropped) == 1


class TestCreateTickerProcessor:
    """Tests for create_ticker_processor factory."""
    
    def test_creates_callable(self):
        """Test that factory creates a callable."""
        data_dict = {"AAPL": MagicMock()}
        params = {"threshold": 5}
        
        def process_func(ticker, df, params):
            return {"ticker": ticker}
        
        processor = create_ticker_processor(data_dict, params, process_func)
        
        assert callable(processor)
    
    def test_processor_uses_data_dict(self):
        """Test that processor uses data from dict."""
        mock_df = MagicMock()
        mock_df.empty = False
        
        data_dict = {"AAPL": mock_df, "GOOGL": mock_df}
        params = {}
        
        def process_func(ticker, df, params):
            return {"ticker": ticker, "has_data": df is not None}
        
        processor = create_ticker_processor(data_dict, params, process_func)
        
        result = processor("AAPL")
        assert result is not None
        assert result["ticker"] == "AAPL"
    
    def test_processor_returns_none_for_missing(self):
        """Test that processor returns None for missing tickers."""
        data_dict = {"AAPL": MagicMock()}
        params = {}
        
        def process_func(ticker, df, params):
            return {"ticker": ticker}
        
        processor = create_ticker_processor(data_dict, params, process_func)
        
        result = processor("MISSING")
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
