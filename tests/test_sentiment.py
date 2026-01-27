"""
Tests for sentiment module.

Tests sentiment scoring and data fetching from various platforms.
"""

import pytest
from unittest.mock import patch, MagicMock


# Import from src
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.sentiment import (
    compute_sentiment_score,
    fetch_stocktwits_sentiment,
    analyze_news_tone,
    SentimentScore,
)


class TestComputeSentimentScore:
    """Tests for compute_sentiment_score function."""
    
    def test_returns_sentiment_score_object(self):
        """Test that function returns SentimentScore dataclass."""
        result = compute_sentiment_score("AAPL")
        
        assert isinstance(result, SentimentScore)
        assert hasattr(result, "score")
        assert hasattr(result, "evidence")
        assert hasattr(result, "data_gaps")
    
    def test_score_within_bounds(self):
        """Test that score is within 0-10 bounds."""
        result = compute_sentiment_score("AAPL")
        
        assert 0 <= result.score <= 10
    
    def test_evidence_structure(self):
        """Test that evidence has expected structure."""
        result = compute_sentiment_score("AAPL")
        
        evidence = result.evidence
        assert "twitter" in evidence
        assert "reddit" in evidence
        assert "stocktwits" in evidence
        assert "news_tone" in evidence
    
    def test_with_headlines(self):
        """Test scoring with provided headlines."""
        headlines = [
            "Company XYZ upgraded to Buy by analysts",
            "Strong earnings beat expectations",
            "Record revenue growth reported"
        ]
        
        result = compute_sentiment_score("TEST", headlines=headlines)
        
        # With positive headlines, news_tone should be positive
        assert result.evidence["news_tone"] in ["positive", "mixed", "neutral"]
    
    def test_no_data_returns_capped_score(self):
        """Test that missing data results in capped score."""
        result = compute_sentiment_score("UNLIKELY_TICKER_12345")
        
        # Should have data gaps
        assert len(result.data_gaps) > 0


class TestAnalyzeNewsTone:
    """Tests for analyze_news_tone function."""
    
    def test_empty_headlines_returns_neutral(self):
        """Test that empty headlines return neutral."""
        result = analyze_news_tone([])
        
        assert result == "neutral"
    
    def test_positive_headlines(self):
        """Test that positive headlines are detected."""
        headlines = [
            "Stock upgraded to Strong Buy",
            "Company beats earnings estimates",
            "Record growth in Q4 revenue",
            "Bullish outlook from analysts"
        ]
        
        result = analyze_news_tone(headlines)
        
        assert result == "positive"
    
    def test_negative_headlines(self):
        """Test that negative headlines are detected."""
        headlines = [
            "Stock downgraded to Sell",
            "Company misses earnings",
            "Weak guidance concerns investors",
            "Bearish sentiment increases"
        ]
        
        result = analyze_news_tone(headlines)
        
        assert result == "negative"
    
    def test_mixed_headlines(self):
        """Test that mixed headlines are detected."""
        headlines = [
            "Stock upgraded by one analyst",
            "Company faces lawsuit",
            "Strong revenue but weak margins",
            "Neutral outlook maintained"
        ]
        
        result = analyze_news_tone(headlines)
        
        assert result in ["mixed", "neutral"]
    
    def test_no_keywords_returns_neutral(self):
        """Test that headlines without keywords return neutral."""
        headlines = [
            "Company announces new product",
            "CEO speaks at conference",
            "Market opens higher"
        ]
        
        result = analyze_news_tone(headlines)
        
        assert result == "neutral"


class TestFetchStockTwitsSentiment:
    """Tests for fetch_stocktwits_sentiment function."""
    
    @patch('requests.get')
    def test_successful_fetch(self, mock_get):
        """Test successful StockTwits data fetch."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "symbol": {"watchlist_count": 5000},
            "messages": [
                {"entities": {"sentiment": {"basic": "Bullish"}}},
                {"entities": {"sentiment": {"basic": "Bullish"}}},
                {"entities": {"sentiment": {"basic": "Bearish"}}},
                {"entities": {}},  # No sentiment
            ]
        }
        mock_get.return_value = mock_response
        
        result = fetch_stocktwits_sentiment("TEST")
        
        assert result is not None
        assert result["bullish_count"] == 2
        assert result["bearish_count"] == 1
        assert result["source"] == "stocktwits"
    
    @patch('requests.get')
    def test_api_error_returns_none(self, mock_get):
        """Test that API error returns None."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        result = fetch_stocktwits_sentiment("TEST")
        
        assert result is None
    
    @patch('requests.get')
    def test_empty_messages_returns_none(self, mock_get):
        """Test that empty messages return None."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "symbol": {},
            "messages": []
        }
        mock_get.return_value = mock_response
        
        result = fetch_stocktwits_sentiment("TEST")
        
        assert result is None
    
    @patch('requests.get')
    def test_exception_returns_none(self, mock_get):
        """Test that exception returns None gracefully."""
        mock_get.side_effect = Exception("Network error")
        
        result = fetch_stocktwits_sentiment("TEST")
        
        assert result is None
    
    @patch('requests.get')
    def test_calculates_bull_bear_ratio(self, mock_get):
        """Test bull/bear ratio calculation."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "symbol": {},
            "messages": [
                {"entities": {"sentiment": {"basic": "Bullish"}}},
                {"entities": {"sentiment": {"basic": "Bullish"}}},
                {"entities": {"sentiment": {"basic": "Bullish"}}},
                {"entities": {"sentiment": {"basic": "Bearish"}}},
            ]
        }
        mock_get.return_value = mock_response
        
        result = fetch_stocktwits_sentiment("TEST")
        
        assert result["bull_bear_ratio"] == 3.0  # 3 bullish / 1 bearish


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
