"""
Sentiment Analysis Module

Fetches social sentiment data to enhance the 4-factor scoring model.
Supports multiple platforms: Polygon News, StockTwits, Reddit.

Data source priority:
1. Polygon.io News (if POLYGON_API_KEY set) - News with sentiment scores
2. StockTwits (free) - Social sentiment
3. Reddit (free) - Social mentions
"""

from __future__ import annotations
import os
import re
import logging
from datetime import datetime, timedelta
from typing import Optional, TypedDict
from dataclasses import dataclass, field
import requests

logger = logging.getLogger(__name__)


def _safe_float(value, default: float = 0.0) -> float:
    """Safely convert value to float, handling None and invalid types."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


class SentimentEvidence(TypedDict):
    """Sentiment evidence structure for scoring."""
    twitter: dict
    reddit: dict
    stocktwits: dict
    news_tone: str
    data_source: str


@dataclass
class SentimentScore:
    """Sentiment momentum score result."""
    score: float  # 0-10 scale
    evidence: SentimentEvidence
    data_gaps: list[str] = field(default_factory=list)
    cap_applied: Optional[float] = None


def fetch_stocktwits_sentiment(ticker: str) -> Optional[dict]:
    """
    Fetch sentiment data from StockTwits API.
    
    StockTwits provides:
    - Message stream with sentiment labels
    - Bull/bear counts
    - Watchlist counts
    
    Free API, no authentication required.
    """
    try:
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None
        
        data = response.json()
        
        symbol_data = data.get("symbol", {})
        messages = data.get("messages", [])
        
        if not messages:
            return None
        
        # Count sentiment from recent messages
        bullish_count = 0
        bearish_count = 0
        
        for msg in messages[:50]:  # Last 50 messages
            sentiment = msg.get("entities", {}).get("sentiment", {})
            if sentiment:
                if sentiment.get("basic") == "Bullish":
                    bullish_count += 1
                elif sentiment.get("basic") == "Bearish":
                    bearish_count += 1
        
        total_with_sentiment = bullish_count + bearish_count
        bull_bear_ratio = (
            bullish_count / bearish_count 
            if bearish_count > 0 
            else float('inf') if bullish_count > 0 else None
        )
        
        return {
            "message_count": len(messages),
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "bull_bear_ratio": bull_bear_ratio,
            "bullish_pct": (bullish_count / total_with_sentiment * 100) if total_with_sentiment > 0 else None,
            "watchlist_count": symbol_data.get("watchlist_count"),
            "source": "stocktwits"
        }
        
    except Exception as e:
        logger.debug(f"StockTwits fetch failed for {ticker}: {e}")
        return None


def fetch_reddit_sentiment(
    ticker: str,
    subreddits: list[str] = ["wallstreetbets", "stocks", "investing", "options"],
    lookback_hours: int = 24,
) -> Optional[dict]:
    """
    Fetch sentiment from Reddit using public JSON endpoints.
    
    Note: For production, consider using PRAW with OAuth for better rate limits.
    """
    try:
        # Search across subreddits for ticker mentions
        total_mentions = 0
        total_upvotes = 0
        total_comments = 0
        posts = []
        
        for subreddit in subreddits:
            try:
                # Use pushshift or Reddit JSON endpoint
                url = f"https://www.reddit.com/r/{subreddit}/search.json"
                params = {
                    "q": ticker,
                    "restrict_sr": "on",
                    "sort": "new",
                    "limit": 25,
                    "t": "day"
                }
                headers = {
                    "User-Agent": "MomentumScanner/1.0"
                }
                
                response = requests.get(url, params=params, headers=headers, timeout=10)
                if response.status_code != 200:
                    continue
                
                data = response.json()
                children = data.get("data", {}).get("children", [])
                
                for child in children:
                    post_data = child.get("data", {})
                    title = post_data.get("title", "").upper()
                    selftext = post_data.get("selftext", "").upper()
                    
                    # Check if ticker is actually mentioned (not just partial match)
                    ticker_pattern = rf'\b{ticker}\b'
                    if re.search(ticker_pattern, title) or re.search(ticker_pattern, selftext):
                        total_mentions += 1
                        total_upvotes += post_data.get("ups", 0)
                        total_comments += post_data.get("num_comments", 0)
                        posts.append({
                            "subreddit": subreddit,
                            "title": post_data.get("title"),
                            "upvotes": post_data.get("ups"),
                            "comments": post_data.get("num_comments"),
                            "url": f"https://reddit.com{post_data.get('permalink', '')}"
                        })
                        
            except Exception as e:
                logger.debug(f"Reddit fetch failed for {subreddit}: {e}")
                continue
        
        if total_mentions == 0:
            return None
        
        avg_upvotes = total_upvotes / total_mentions if total_mentions > 0 else 0
        
        return {
            "mention_count": total_mentions,
            "total_upvotes": total_upvotes,
            "total_comments": total_comments,
            "avg_upvotes": avg_upvotes,
            "subreddits_searched": len(subreddits),
            "top_posts": posts[:5],
            "source": "reddit"
        }
        
    except Exception as e:
        logger.debug(f"Reddit fetch failed for {ticker}: {e}")
        return None


def fetch_news_sentiment_polygon(ticker: str, api_key: Optional[str] = None, limit: int = 50) -> Optional[dict]:
    """
    Fetch news sentiment from Polygon.io News API.
    
    Requires POLYGON_API_KEY environment variable.
    Polygon news includes sentiment analysis on paid plans.
    """
    api_key = api_key or os.environ.get("POLYGON_API_KEY")
    if not api_key:
        return None
    
    try:
        # Polygon News API v2
        url = "https://api.polygon.io/v2/reference/news"
        params = {
            "ticker": ticker,
            "limit": limit,
            "order": "desc",
            "sort": "published_utc",
            "apiKey": api_key
        }
        
        response = requests.get(url, params=params, timeout=15)
        if response.status_code != 200:
            logger.debug(f"Polygon news API returned {response.status_code}")
            return None
        
        data = response.json()
        articles = data.get("results", [])
        
        if not articles:
            return None
        
        # Analyze sentiment from articles
        # Polygon includes sentiment in "insights" field on certain plans
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        headlines = []
        
        for article in articles:
            title = article.get("title", "")
            headlines.append(title)
            
            # Check for insights/sentiment data (paid feature)
            insights = article.get("insights", [])
            for insight in insights:
                if insight.get("ticker", "").upper() == ticker.upper():
                    sentiment = insight.get("sentiment", "neutral").lower()
                    if sentiment == "positive" or sentiment == "bullish":
                        bullish_count += 1
                    elif sentiment == "negative" or sentiment == "bearish":
                        bearish_count += 1
                    else:
                        neutral_count += 1
                    break
            else:
                # No sentiment in insights, use headline analysis
                tone = analyze_news_tone([title])
                if tone == "positive":
                    bullish_count += 1
                elif tone == "negative":
                    bearish_count += 1
                else:
                    neutral_count += 1
        
        total_analyzed = bullish_count + bearish_count + neutral_count
        
        # Calculate sentiment score (-1 to 1 scale)
        if total_analyzed > 0:
            avg_sentiment = (bullish_count - bearish_count) / total_analyzed
        else:
            avg_sentiment = 0
        
        return {
            "article_count": len(articles),
            "analyzed_count": total_analyzed,
            "avg_sentiment_score": avg_sentiment,
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "neutral_count": neutral_count,
            "bullish_pct": (bullish_count / total_analyzed * 100) if total_analyzed > 0 else None,
            "headlines": headlines[:10],
            "source": "polygon"
        }
        
    except Exception as e:
        logger.debug(f"Polygon news sentiment failed for {ticker}: {e}")
        return None


def fetch_news_sentiment_alphavantage(ticker: str, api_key: Optional[str] = None) -> Optional[dict]:
    """
    Fetch news sentiment from Alpha Vantage News Sentiment API.
    
    Requires ALPHAVANTAGE_API_KEY environment variable.
    Free tier: 25 requests/day.
    
    NOTE: Consider using Polygon instead if you have POLYGON_API_KEY.
    """
    api_key = api_key or os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return None
    
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "apikey": api_key,
            "limit": 50
        }
        
        response = requests.get(url, params=params, timeout=15)
        if response.status_code != 200:
            return None
        
        data = response.json()
        
        if "feed" not in data:
            return None
        
        articles = data.get("feed", [])
        
        if not articles:
            return None
        
        # Analyze sentiment from articles
        total_sentiment = 0
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        
        for article in articles:
            # Find ticker-specific sentiment
            ticker_sentiments = article.get("ticker_sentiment", [])
            for ts in ticker_sentiments:
                if ts.get("ticker", "").upper() == ticker.upper():
                    sentiment_score = _safe_float(ts.get("ticker_sentiment_score"), 0.0)
                    total_sentiment += sentiment_score
                    
                    label = ts.get("ticker_sentiment_label", "Neutral")
                    if "Bullish" in label:
                        bullish_count += 1
                    elif "Bearish" in label:
                        bearish_count += 1
                    else:
                        neutral_count += 1
                    break
        
        total_analyzed = bullish_count + bearish_count + neutral_count
        avg_sentiment = total_sentiment / total_analyzed if total_analyzed > 0 else 0
        
        return {
            "article_count": len(articles),
            "analyzed_count": total_analyzed,
            "avg_sentiment_score": avg_sentiment,  # -1 to 1 scale
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "neutral_count": neutral_count,
            "bullish_pct": (bullish_count / total_analyzed * 100) if total_analyzed > 0 else None,
            "source": "alphavantage"
        }
        
    except Exception as e:
        logger.debug(f"Alpha Vantage news sentiment failed for {ticker}: {e}")
        return None


def analyze_news_tone(headlines: list[str]) -> str:
    """
    Simple rule-based news tone analysis.
    
    Returns: "positive", "negative", "mixed", or "neutral"
    """
    if not headlines:
        return "neutral"
    
    positive_keywords = [
        "upgrade", "raised", "beat", "strong", "growth", "surge", "rally",
        "breakthrough", "approval", "wins", "awarded", "exceeds", "outperform",
        "bullish", "buy", "positive", "record", "soar", "jump", "gain"
    ]
    
    negative_keywords = [
        "downgrade", "cut", "miss", "weak", "decline", "fall", "drop",
        "concern", "warning", "risk", "lawsuit", "investigation", "loss",
        "bearish", "sell", "negative", "plunge", "crash", "tumble"
    ]
    
    positive_count = 0
    negative_count = 0
    
    for headline in headlines:
        headline_lower = headline.lower()
        
        for kw in positive_keywords:
            if kw in headline_lower:
                positive_count += 1
                break
        
        for kw in negative_keywords:
            if kw in headline_lower:
                negative_count += 1
                break
    
    total = positive_count + negative_count
    
    if total == 0:
        return "neutral"
    
    positive_ratio = positive_count / total
    
    if positive_ratio >= 0.7:
        return "positive"
    elif positive_ratio <= 0.3:
        return "negative"
    else:
        return "mixed"


def compute_sentiment_score(
    ticker: str,
    headlines: Optional[list[str]] = None,
    api_key: Optional[str] = None,
) -> SentimentScore:
    """
    Compute sentiment momentum score (0-10) for a ticker.
    
    Scoring rubric:
    - Twitter/X (40% of sentiment): mention velocity, quality accounts, engagement
    - Other platforms (60%):
      - Reddit (20%): mention velocity, upvote ratio
      - StockTwits (20%): bull/bear ratio
      - News/Analysts (20%): sentiment tone
    
    If social data is unavailable, caps score at 4.0.
    """
    data_gaps = []
    evidence: SentimentEvidence = {
        "twitter": {
            "mention_velocity_vs_7d": None,
            "quality_accounts_est": None,
            "bullish_pct_est": None
        },
        "reddit": {
            "mention_velocity": None,
            "upvote_ratio_est": None
        },
        "stocktwits": {
            "bull_bear_ratio_est": None
        },
        "news_tone": "neutral",
        "data_source": "none"
    }
    
    score = 0.0
    cap_applied = None
    sources_found = 0
    
    # StockTwits sentiment
    stocktwits_data = fetch_stocktwits_sentiment(ticker)
    if stocktwits_data:
        sources_found += 1
        evidence["data_source"] = "stocktwits"
        evidence["stocktwits"]["bull_bear_ratio_est"] = stocktwits_data.get("bull_bear_ratio")
        
        # Score based on bull/bear ratio
        bb_ratio = stocktwits_data.get("bull_bear_ratio")
        if bb_ratio is not None:
            if bb_ratio >= 3.0:
                score += 2.0
            elif bb_ratio >= 2.0:
                score += 1.5
            elif bb_ratio >= 1.5:
                score += 1.0
            elif bb_ratio >= 1.0:
                score += 0.5
        
        # Bonus for high message volume
        msg_count = stocktwits_data.get("message_count", 0)
        if msg_count >= 100:
            score += 0.5
    
    # Reddit sentiment
    reddit_data = fetch_reddit_sentiment(ticker)
    if reddit_data:
        sources_found += 1
        if evidence["data_source"] == "none":
            evidence["data_source"] = "reddit"
        else:
            evidence["data_source"] += "+reddit"
        
        evidence["reddit"]["mention_velocity"] = reddit_data.get("mention_count")
        evidence["reddit"]["upvote_ratio_est"] = reddit_data.get("avg_upvotes")
        
        # Score based on mentions and engagement
        mentions = reddit_data.get("mention_count", 0)
        avg_upvotes = reddit_data.get("avg_upvotes", 0)
        
        if mentions >= 10:
            score += 1.5
        elif mentions >= 5:
            score += 1.0
        elif mentions >= 2:
            score += 0.5
        
        if avg_upvotes >= 100:
            score += 0.5
    
    # News sentiment (Polygon first, then Alpha Vantage fallback)
    news_data = fetch_news_sentiment_polygon(ticker, api_key)
    if not news_data:
        news_data = fetch_news_sentiment_alphavantage(ticker)
    
    if news_data:
        sources_found += 1
        
        avg_sentiment = news_data.get("avg_sentiment_score", 0)
        bullish_pct = news_data.get("bullish_pct", 50)
        
        # Score based on news sentiment
        if avg_sentiment >= 0.3:
            score += 1.5
            evidence["news_tone"] = "positive"
        elif avg_sentiment >= 0.1:
            score += 1.0
            evidence["news_tone"] = "positive"
        elif avg_sentiment <= -0.3:
            evidence["news_tone"] = "negative"
        elif avg_sentiment <= -0.1:
            evidence["news_tone"] = "negative"
        else:
            evidence["news_tone"] = "neutral"
            score += 0.5
    elif headlines:
        # Fallback to simple headline analysis
        tone = analyze_news_tone(headlines)
        evidence["news_tone"] = tone
        
        if tone == "positive":
            score += 1.0
        elif tone == "mixed":
            score += 0.5
    
    # Twitter data (placeholder - requires premium API)
    twitter_api_key = os.environ.get("TWITTER_BEARER_TOKEN")
    if twitter_api_key:
        # TODO: Implement Twitter API integration
        # This requires Twitter API v2 with elevated access
        data_gaps.append("Twitter API integration pending")
    else:
        data_gaps.append("Twitter data unavailable (no API key)")
    
    # Apply cap if insufficient data
    if sources_found == 0:
        data_gaps.append("Social sentiment data unavailable; score capped at 4.0")
        cap_applied = 4.0
    elif sources_found == 1:
        data_gaps.append("Limited sentiment sources; score capped at 6.0")
        cap_applied = 6.0
    
    if cap_applied is not None:
        score = min(score, cap_applied)
    
    # Ensure score is within bounds
    score = max(0.0, min(10.0, score))
    
    return SentimentScore(
        score=round(score, 2),
        evidence=evidence,
        data_gaps=data_gaps,
        cap_applied=cap_applied
    )


def get_social_mention_velocity(ticker: str, platform: str = "all") -> Optional[dict]:
    """
    Get mention velocity (mentions per hour) compared to 7-day average.
    
    This is a placeholder for services like:
    - Quiver Quantitative
    - Alternative.me
    - Sentdex
    """
    # Placeholder implementation
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# NEW INTEGRATIONS: FMP, Adanos, Enhanced Polygon News
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_fmp_earnings_calendar(
    ticker: str,
    api_key: Optional[str] = None,
    days_ahead: int = 30
) -> Optional[dict]:
    """
    Fetch earnings calendar from Financial Modeling Prep (FMP).
    
    Requires FMP_API_KEY environment variable (free tier available).
    Sign up at: https://site.financialmodelingprep.com
    
    Returns:
        dict with earnings_date, eps_estimate, revenue_estimate, etc.
    """
    api_key = api_key or os.environ.get("FMP_API_KEY")
    if not api_key:
        return None
    
    try:
        # FMP Stable Earnings Calendar endpoint (updated URL format)
        from_date = datetime.now().strftime("%Y-%m-%d")
        to_date = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        
        url = f"https://financialmodelingprep.com/stable/earnings-calendar?from={from_date}&to={to_date}&apikey={api_key}"
        
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            logger.debug(f"FMP earnings API returned {response.status_code}")
            return None
        
        data = response.json()
        
        if not data:
            return None
        
        # Find this ticker's earnings
        for entry in data:
            if entry.get("symbol", "").upper() == ticker.upper():
                return {
                    "ticker": ticker,
                    "earnings_date": entry.get("date"),
                    "eps_estimate": entry.get("epsEstimated"),
                    "revenue_estimate": entry.get("revenueEstimated"),
                    "time": entry.get("time", "unknown"),  # "bmo" (before market open) or "amc" (after market close)
                    "fiscal_quarter": entry.get("fiscalDateEnding"),
                    "source": "fmp"
                }
        
        return None
        
    except Exception as e:
        logger.debug(f"FMP earnings fetch failed for {ticker}: {e}")
        return None


def fetch_fmp_news(
    ticker: str,
    api_key: Optional[str] = None,
    limit: int = 20
) -> Optional[dict]:
    """
    Fetch news headlines from Financial Modeling Prep (FMP).
    
    Requires FMP_API_KEY environment variable.
    Note: News endpoint may not be available on all FMP plans.
    
    Returns:
        dict with headlines, sentiment analysis, and source info
    """
    api_key = api_key or os.environ.get("FMP_API_KEY")
    if not api_key:
        return None
    
    try:
        # Try FMP Stable endpoint first (new format)
        url = f"https://financialmodelingprep.com/stable/stock-news?symbol={ticker}&limit={limit}&apikey={api_key}"
        
        response = requests.get(url, timeout=15)
        
        # If stable endpoint fails, try legacy v3 endpoint
        if response.status_code != 200:
            url = f"https://financialmodelingprep.com/api/v3/stock_news?tickers={ticker}&limit={limit}&apikey={api_key}"
            response = requests.get(url, timeout=15)
        
        if response.status_code != 200:
            logger.debug(f"FMP news API returned {response.status_code}")
            return None
        
        articles = response.json()
        
        if not articles:
            return None
        
        headlines = []
        for article in articles:
            headlines.append({
                "title": article.get("title", ""),
                "publisher": article.get("site", article.get("publisher", "Unknown")),
                "url": article.get("url", ""),
                "published_at": article.get("publishedDate", article.get("published_at", "")),
                "image": article.get("image", ""),
            })
        
        # Analyze sentiment from headlines
        headline_titles = [h["title"] for h in headlines if h["title"]]
        tone = analyze_news_tone(headline_titles)
        
        return {
            "article_count": len(articles),
            "headlines": headlines,
            "news_tone": tone,
            "source": "fmp"
        }
        
    except Exception as e:
        logger.debug(f"FMP news fetch failed for {ticker}: {e}")
        return None


def fetch_adanos_sentiment(
    ticker: str,
    api_key: Optional[str] = None
) -> Optional[dict]:
    """
    Fetch Reddit sentiment from Adanos API.
    
    Requires ADANOS_API_KEY environment variable.
    Sign up at: https://adanos.org (free tier: 250 calls/month)
    
    API Base: https://api.adanos.org/reddit/stocks
    Auth: X-API-Key header
    
    Returns:
        dict with reddit sentiment, buzz_score, trend, etc.
    """
    api_key = api_key or os.environ.get("ADANOS_API_KEY")
    if not api_key:
        return None
    
    try:
        # Adanos Reddit Stocks API - Get trending to find ticker
        # Base URL: https://api.adanos.org/reddit/stocks
        url = "https://api.adanos.org/reddit/stocks/v1/trending"
        headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
        params = {
            "days": 1,
            "limit": 100  # Get top 100 to find our ticker
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=15)
        if response.status_code != 200:
            logger.debug(f"Adanos API returned {response.status_code}: {response.text[:100]}")
            return None
        
        data = response.json()
        
        if not data:
            return None
        
        # Find our ticker in the trending list
        ticker_upper = ticker.upper()
        ticker_data = None
        for item in data:
            if item.get("ticker", "").upper() == ticker_upper:
                ticker_data = item
                break
        
        if not ticker_data:
            # Ticker not in trending list - not necessarily an error
            logger.debug(f"Adanos: {ticker} not in trending list")
            return None
        
        # Map trend string to velocity
        trend = ticker_data.get("trend", "stable")
        velocity = 1.0
        if trend == "rising":
            velocity = 1.5
        elif trend == "falling":
            velocity = 0.5
        
        return {
            "ticker": ticker,
            "twitter": {
                "sentiment_score": None,  # Adanos Reddit API doesn't include Twitter
                "mention_count": None,
                "mention_velocity_vs_7d": None,
                "bullish_pct": None,
                "engagement": None,
            },
            "reddit": {
                "sentiment_score": ticker_data.get("sentiment_score"),  # -1 to 1
                "mention_count": ticker_data.get("mentions"),
                "mention_velocity_vs_7d": velocity,
                "bullish_pct": ticker_data.get("bullish_pct"),
                "bearish_pct": ticker_data.get("bearish_pct"),
                "subreddit_count": ticker_data.get("subreddit_count"),
                "total_upvotes": ticker_data.get("total_upvotes"),
            },
            "overall_sentiment": ticker_data.get("sentiment_score"),
            "buzz_score": ticker_data.get("buzz_score"),  # 0 to 100
            "trending": trend == "rising",
            "trend": trend,
            "company_name": ticker_data.get("company_name"),
            "source": "adanos"
        }
        
    except Exception as e:
        logger.debug(f"Adanos sentiment fetch failed for {ticker}: {e}")
        return None


def fetch_adanos_trending(
    api_key: Optional[str] = None,
    limit: int = 20,
    days: int = 1
) -> Optional[list]:
    """
    Fetch trending stocks from Adanos Reddit Sentiment API.
    
    Returns list of trending tickers with buzz scores.
    """
    api_key = api_key or os.environ.get("ADANOS_API_KEY")
    if not api_key:
        return None
    
    try:
        url = "https://api.adanos.org/reddit/stocks/v1/trending"
        headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
        params = {
            "days": days,
            "limit": limit
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=15)
        if response.status_code != 200:
            logger.debug(f"Adanos trending API returned {response.status_code}")
            return None
        
        return response.json()
        
    except Exception as e:
        logger.debug(f"Adanos trending fetch failed: {e}")
        return None


def fetch_polygon_news_headlines(
    ticker: str,
    api_key: Optional[str] = None,
    limit: int = 20,
    days_back: int = 7
) -> Optional[dict]:
    """
    Fetch actual news headlines from Polygon.io News API.
    Enhanced version that returns full headline data for LLM analysis.
    
    Requires POLYGON_API_KEY environment variable.
    
    Returns:
        dict with headlines, publishers, dates, and sentiment if available
    """
    api_key = api_key or os.environ.get("POLYGON_API_KEY")
    if not api_key:
        return None
    
    try:
        # Calculate date range
        published_after = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        # Polygon News API v2
        url = "https://api.polygon.io/v2/reference/news"
        params = {
            "ticker": ticker,
            "limit": limit,
            "order": "desc",
            "sort": "published_utc",
            "published_utc.gte": published_after,
            "apiKey": api_key
        }
        
        response = requests.get(url, params=params, timeout=15)
        if response.status_code != 200:
            logger.debug(f"Polygon news API returned {response.status_code}")
            return None
        
        data = response.json()
        articles = data.get("results", [])
        
        if not articles:
            return None
        
        headlines = []
        bullish_count = 0
        bearish_count = 0
        
        for article in articles:
            title = article.get("title", "")
            publisher = article.get("publisher", {})
            
            headline_entry = {
                "title": title,
                "publisher": publisher.get("name", "Unknown") if isinstance(publisher, dict) else str(publisher),
                "url": article.get("article_url", ""),
                "published_at": article.get("published_utc", "")[:10] if article.get("published_utc") else "",
                "description": article.get("description", "")[:200] if article.get("description") else "",
            }
            
            # Check for Polygon's sentiment insights (paid feature)
            insights = article.get("insights", [])
            for insight in insights:
                if insight.get("ticker", "").upper() == ticker.upper():
                    sentiment = insight.get("sentiment", "").lower()
                    headline_entry["sentiment"] = sentiment
                    if sentiment in ["positive", "bullish"]:
                        bullish_count += 1
                    elif sentiment in ["negative", "bearish"]:
                        bearish_count += 1
                    break
            
            headlines.append(headline_entry)
        
        # Analyze tone from headlines
        headline_titles = [h["title"] for h in headlines if h["title"]]
        tone = analyze_news_tone(headline_titles)
        
        total_with_sentiment = bullish_count + bearish_count
        
        return {
            "article_count": len(articles),
            "headlines": headlines,
            "news_tone": tone,
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "bullish_pct": (bullish_count / total_with_sentiment * 100) if total_with_sentiment > 0 else None,
            "source": "polygon"
        }
        
    except Exception as e:
        logger.debug(f"Polygon news headlines fetch failed for {ticker}: {e}")
        return None


def compute_enhanced_sentiment_score(
    ticker: str,
    headlines: Optional[list[str]] = None,
    polygon_api_key: Optional[str] = None,
    fmp_api_key: Optional[str] = None,
    adanos_api_key: Optional[str] = None,
) -> SentimentScore:
    """
    Enhanced sentiment scoring using multiple sources:
    - Adanos (Twitter/X + Reddit) - 40% weight
    - Polygon News - 20% weight
    - StockTwits - 20% weight  
    - Reddit direct - 20% weight
    
    This is the upgraded version of compute_sentiment_score with more sources.
    """
    data_gaps = []
    evidence: SentimentEvidence = {
        "twitter": {
            "mention_velocity_vs_7d": None,
            "quality_accounts_est": None,
            "bullish_pct_est": None
        },
        "reddit": {
            "mention_velocity": None,
            "upvote_ratio_est": None
        },
        "stocktwits": {
            "bull_bear_ratio_est": None
        },
        "news_tone": "neutral",
        "data_source": "none"
    }
    
    score = 0.0
    cap_applied = None
    sources_found = 0
    sources_list = []
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # 1. ADANOS - Twitter/X + Reddit (Best source for social sentiment)
    # ═══════════════════════════════════════════════════════════════════════════════
    adanos_data = fetch_adanos_sentiment(ticker, adanos_api_key)
    if adanos_data:
        sources_found += 1
        sources_list.append("adanos")
        
        # Twitter data from Adanos
        twitter = adanos_data.get("twitter", {})
        if twitter:
            evidence["twitter"]["mention_velocity_vs_7d"] = twitter.get("mention_velocity_vs_7d")
            evidence["twitter"]["bullish_pct_est"] = twitter.get("bullish_pct")
            
            # Score based on Twitter sentiment
            twitter_sentiment = twitter.get("sentiment_score", 0)  # -1 to 1
            if twitter_sentiment >= 0.3:
                score += 2.5  # Strong bullish
            elif twitter_sentiment >= 0.1:
                score += 1.5  # Mild bullish
            elif twitter_sentiment <= -0.3:
                pass  # Bearish - no points
            elif twitter_sentiment <= -0.1:
                pass  # Mild bearish
            else:
                score += 0.5  # Neutral
            
            # Velocity bonus
            velocity = twitter.get("mention_velocity_vs_7d", 1)
            if velocity and velocity >= 2.0:
                score += 0.5  # Trending
        
        # Reddit data from Adanos
        reddit = adanos_data.get("reddit", {})
        if reddit:
            evidence["reddit"]["mention_velocity"] = reddit.get("mention_count")
            
            reddit_sentiment = reddit.get("sentiment_score", 0)
            if reddit_sentiment >= 0.2:
                score += 1.0
        
        # Buzz score bonus
        buzz = adanos_data.get("buzz_score", 0)
        if buzz >= 70:
            score += 0.5
    else:
        data_gaps.append("Adanos data unavailable (no API key or rate limited)")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # 2. STOCKTWITS (Free backup for social sentiment)
    # ═══════════════════════════════════════════════════════════════════════════════
    stocktwits_data = fetch_stocktwits_sentiment(ticker)
    if stocktwits_data:
        sources_found += 1
        sources_list.append("stocktwits")
        evidence["stocktwits"]["bull_bear_ratio_est"] = stocktwits_data.get("bull_bear_ratio")
        
        bb_ratio = stocktwits_data.get("bull_bear_ratio")
        if bb_ratio is not None:
            if bb_ratio >= 3.0:
                score += 1.5
            elif bb_ratio >= 2.0:
                score += 1.0
            elif bb_ratio >= 1.5:
                score += 0.5
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # 3. REDDIT (Free direct access)
    # ═══════════════════════════════════════════════════════════════════════════════
    if not adanos_data:  # Only if Adanos didn't provide Reddit data
        reddit_data = fetch_reddit_sentiment(ticker)
        if reddit_data:
            sources_found += 1
            sources_list.append("reddit")
            evidence["reddit"]["mention_velocity"] = reddit_data.get("mention_count")
            evidence["reddit"]["upvote_ratio_est"] = reddit_data.get("avg_upvotes")
            
            mentions = reddit_data.get("mention_count", 0)
            if mentions >= 10:
                score += 1.0
            elif mentions >= 5:
                score += 0.5
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # 4. NEWS SENTIMENT (Polygon or FMP)
    # ═══════════════════════════════════════════════════════════════════════════════
    news_data = fetch_polygon_news_headlines(ticker, polygon_api_key)
    if not news_data:
        news_data = fetch_fmp_news(ticker, fmp_api_key)
    
    if news_data:
        sources_found += 1
        sources_list.append(news_data.get("source", "news"))
        
        tone = news_data.get("news_tone", "neutral")
        evidence["news_tone"] = tone
        
        if tone == "positive":
            score += 1.5
        elif tone == "mixed":
            score += 0.5
        
        # Bullish percentage bonus if available
        bullish_pct = news_data.get("bullish_pct")
        if bullish_pct and bullish_pct >= 70:
            score += 0.5
    elif headlines:
        # Fallback to provided headlines
        tone = analyze_news_tone(headlines)
        evidence["news_tone"] = tone
        if tone == "positive":
            score += 1.0
    
    # Set data source string
    evidence["data_source"] = "+".join(sources_list) if sources_list else "none"
    
    # Apply caps based on data availability
    if sources_found == 0:
        data_gaps.append("No sentiment sources available; score capped at 3.0")
        cap_applied = 3.0
    elif sources_found == 1:
        data_gaps.append("Limited sentiment sources; score capped at 5.0")
        cap_applied = 5.0
    elif sources_found == 2:
        data_gaps.append("Moderate sentiment coverage; score capped at 7.0")
        cap_applied = 7.0
    # 3+ sources = no cap
    
    if cap_applied is not None:
        score = min(score, cap_applied)
    
    # Ensure score is within bounds
    score = max(0.0, min(10.0, score))
    
    return SentimentScore(
        score=round(score, 2),
        evidence=evidence,
        data_gaps=data_gaps,
        cap_applied=cap_applied
    )


def fetch_all_catalysts(
    ticker: str,
    polygon_api_key: Optional[str] = None,
    fmp_api_key: Optional[str] = None,
) -> dict:
    """
    Fetch all catalyst data from multiple sources:
    - Earnings calendar (FMP)
    - News headlines (Polygon, FMP)
    
    Returns combined catalyst data for LLM packets.
    """
    result = {
        "earnings": None,
        "headlines": [],
        "sources_used": [],
    }
    
    # Fetch earnings from FMP
    earnings = fetch_fmp_earnings_calendar(ticker, fmp_api_key)
    if earnings:
        result["earnings"] = earnings
        result["sources_used"].append("fmp_earnings")
    
    # Fetch headlines from Polygon (primary)
    polygon_news = fetch_polygon_news_headlines(ticker, polygon_api_key)
    if polygon_news:
        result["headlines"].extend(polygon_news.get("headlines", []))
        result["news_tone"] = polygon_news.get("news_tone")
        result["sources_used"].append("polygon_news")
    
    # Fetch additional headlines from FMP (supplementary)
    fmp_news = fetch_fmp_news(ticker, fmp_api_key)
    if fmp_news:
        # Add FMP headlines that aren't duplicates
        existing_titles = {h.get("title", "").lower() for h in result["headlines"]}
        for h in fmp_news.get("headlines", []):
            if h.get("title", "").lower() not in existing_titles:
                result["headlines"].append(h)
        result["sources_used"].append("fmp_news")
    
    # Deduplicate and limit
    result["headlines"] = result["headlines"][:20]
    
    return result
