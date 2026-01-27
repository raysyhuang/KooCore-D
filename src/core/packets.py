"""
Packet Building Functions

Build LLM packets for ranking and analysis.
"""

from __future__ import annotations
import os
import pandas as pd
from typing import Optional
from src.core.analysis import analyze_headlines
from src.core.options import compute_options_score
from src.core.sentiment import (
    compute_sentiment_score,
    compute_enhanced_sentiment_score,
    fetch_all_catalysts,
    fetch_fmp_earnings_calendar,
)


def build_weekly_scanner_packet(
    ticker: str,
    row: pd.Series,
    news_df: pd.DataFrame,
    earnings_date: str,
    manual_headlines_df: Optional[pd.DataFrame] = None,
    source_tags: Optional[list[str]] = None,
    polygon_api_key: Optional[str] = None,
    fmp_api_key: Optional[str] = None,
    adanos_api_key: Optional[str] = None,
    fetch_options: bool = True,
    fetch_sentiment: bool = True,
    use_enhanced_sentiment: bool = True,
) -> dict:
    """
    Build a packet for LLM Weekly Scanner ranking.
    
    Args:
        ticker: Ticker symbol
        row: Series with candidate data (from screening)
        news_df: DataFrame with news headlines
        earnings_date: Next earnings date string
        manual_headlines_df: Optional DataFrame with manual headlines
        source_tags: Optional list of source tags (e.g., ["BASE_UNIVERSE", "DAILY_MOVER"])
    
    Returns:
        Dict with all data needed for LLM scoring
    """
    # ═══════════════════════════════════════════════════════════════════════════════
    # ENHANCED NEWS/CATALYST FETCHING (Polygon + FMP + Yahoo)
    # ═══════════════════════════════════════════════════════════════════════════════
    
    # First, try to fetch from enhanced sources (Polygon News, FMP)
    headlines = []
    headline_titles = []
    enhanced_earnings = None
    catalyst_sources = []
    
    # Fetch from Polygon/FMP if API keys available
    api_key_polygon = polygon_api_key or os.environ.get("POLYGON_API_KEY")
    api_key_fmp = fmp_api_key or os.environ.get("FMP_API_KEY")
    
    if api_key_polygon or api_key_fmp:
        try:
            catalyst_data = fetch_all_catalysts(ticker, api_key_polygon, api_key_fmp)
            
            # Use enhanced headlines
            for h in catalyst_data.get("headlines", []):
                title = h.get("title", "").strip()
                if title and title not in headline_titles:
                    headline_titles.append(title)
                    headlines.append(h)
            
            # Use FMP earnings if available
            if catalyst_data.get("earnings"):
                enhanced_earnings = catalyst_data["earnings"]
            
            catalyst_sources = catalyst_data.get("sources_used", [])
        except Exception as e:
            pass  # Fall back to traditional sources
    
    # Filter news from traditional source (Yahoo via news_df)
    ticker_news = news_df[news_df["Ticker"] == ticker].copy() if not news_df.empty else pd.DataFrame()
    if not ticker_news.empty and "published_utc" in ticker_news.columns:
        ticker_news["published_utc"] = pd.to_datetime(ticker_news["published_utc"], utc=True, errors="coerce")
        ticker_news = ticker_news.sort_values("published_utc", ascending=False)
        ticker_news = ticker_news.head(15)  # Top 15 headlines
    
    # Add manual headlines first (highest priority)
    if manual_headlines_df is not None and not manual_headlines_df.empty:
        manual = manual_headlines_df[manual_headlines_df["Ticker"].astype(str).str.strip().eq(ticker)]
        for _, m in manual.iterrows():
            date_str = str(m.get("Date", "Unknown")).strip()
            source = str(m.get("Source", "Manual")).strip()
            headline = str(m.get("Headline", "")).strip()
            if headline and headline not in headline_titles:
                headline_titles.append(headline)
                headlines.insert(0, {  # Insert at beginning (highest priority)
                    "title": headline,
                    "publisher": source,
                    "url": "",
                    "published_at": date_str
                })
    
    # Add yfinance headlines (if not already from Polygon/FMP)
    for _, n in ticker_news.iterrows():
        title = str(n.get("title", "")).strip()
        if title and title not in headline_titles:
            publisher = str(n.get("publisher", "")).strip()
            link = str(n.get("link", "")).strip()
            pub_time = n.get("published_utc", pd.NaT)
            pub_str = pub_time.strftime("%Y-%m-%d") if pd.notna(pub_time) else "Unknown"
            headline_titles.append(title)
            headlines.append({
                "title": title,
                "publisher": publisher,
                "url": link,
                "published_at": pub_str
            })
    
    # Use enhanced earnings date if available
    final_earnings_date = earnings_date
    if enhanced_earnings and enhanced_earnings.get("earnings_date"):
        final_earnings_date = enhanced_earnings.get("earnings_date")
        # Add earnings time info (before/after market)
        earnings_time = enhanced_earnings.get("time", "")
    
    # Analyze headlines for flags
    flags = analyze_headlines(headline_titles)
    
    # Fetch options data if enabled
    options_score_result = None
    options_evidence = {}
    options_data_available = False
    if fetch_options:
        try:
            api_key = polygon_api_key or os.environ.get("POLYGON_API_KEY")
            options_score_result = compute_options_score(ticker, api_key=api_key)
            if options_score_result and options_score_result.evidence.get("data_source") != "none":
                options_data_available = True
                options_evidence = {
                    "call_put_ratio": options_score_result.evidence.get("call_put_ratio"),
                    "unusual_volume_multiple": options_score_result.evidence.get("unusual_volume_multiple"),
                    "largest_bullish_premium_usd": options_score_result.evidence.get("largest_bullish_premium_usd"),
                    "iv_rank": options_score_result.evidence.get("iv_rank"),
                    "implied_volatility": options_score_result.evidence.get("implied_volatility"),
                    "notable_contracts": options_score_result.evidence.get("notable_contracts", []),
                    "data_source": options_score_result.evidence.get("data_source", "none"),
                }
        except Exception as e:
            # Silently fail - options data is optional
            pass
    
    # Fetch sentiment data if enabled
    # Uses enhanced sentiment (Adanos for Twitter/Reddit + Polygon + StockTwits)
    sentiment_score_result = None
    sentiment_evidence = {}
    sentiment_data_available = False
    if fetch_sentiment:
        try:
            api_key_poly = polygon_api_key or os.environ.get("POLYGON_API_KEY")
            api_key_fmp_sent = fmp_api_key or os.environ.get("FMP_API_KEY")
            api_key_adanos = adanos_api_key or os.environ.get("ADANOS_API_KEY")
            
            # Extract headline titles for sentiment analysis
            headline_titles_list = [h.get("title", "") for h in headlines if h.get("title")]
            
            # Use enhanced sentiment if enabled (includes Adanos for Twitter)
            if use_enhanced_sentiment:
                sentiment_score_result = compute_enhanced_sentiment_score(
                    ticker, 
                    headlines=headline_titles_list if headline_titles_list else None,
                    polygon_api_key=api_key_poly,
                    fmp_api_key=api_key_fmp_sent,
                    adanos_api_key=api_key_adanos,
                )
            else:
                # Fallback to basic sentiment
                sentiment_score_result = compute_sentiment_score(
                    ticker, 
                    headlines=headline_titles_list if headline_titles_list else None,
                    api_key=api_key_poly
                )
            
            if sentiment_score_result and sentiment_score_result.evidence.get("data_source") != "none":
                sentiment_data_available = True
                sentiment_evidence = {
                    "twitter": sentiment_score_result.evidence.get("twitter", {}),
                    "reddit": sentiment_score_result.evidence.get("reddit", {}),
                    "stocktwits": sentiment_score_result.evidence.get("stocktwits", {}),
                    "news_tone": sentiment_score_result.evidence.get("news_tone", "unknown"),
                    "data_source": sentiment_score_result.evidence.get("data_source", "none"),
                }
        except Exception as e:
            # Silently fail - sentiment data is optional
            pass
    
    packet = {
        "ticker": ticker,
        "name": row.get("name", ticker),
        "exchange": row.get("exchange", "Unknown"),
        "sector": row.get("sector", "Unknown"),
        "current_price": row.get("current_price"),
        "market_cap_usd": row.get("market_cap_usd"),
        "avg_dollar_volume_20d": row.get("avg_dollar_volume_20d"),
        "asof_price_utc": row.get("asof_price_utc"),
        
        # Technical (LOCKED - from Python)
        "technical_score": row.get("technical_score"),
        "technical_evidence": row.get("technical_evidence", {}),
        "technical_data_gaps": row.get("technical_data_gaps", []),
        
        # Catalyst data (enhanced with FMP/Polygon)
        "earnings_date": final_earnings_date,
        "earnings_info": enhanced_earnings,  # Full earnings info from FMP if available
        "headlines": headlines[:20],  # Limit to 20 headlines
        "dilution_flag": flags.get("dilution_flag", 0),
        "catalyst_tags": flags.get("catalyst_tags", ""),
        "catalyst_sources": catalyst_sources,  # Track which sources were used
        
        # Options data (now populated if available)
        "options_data_available": options_data_available,
        "options_score": options_score_result.score if options_score_result else None,
        "options_evidence": options_evidence,
        "options_data_gaps": options_score_result.data_gaps if options_score_result else [],
        "options_cap_applied": options_score_result.cap_applied if options_score_result else None,
        
        # Sentiment data (enhanced with Adanos for Twitter)
        "sentiment_data_available": sentiment_data_available,
        "sentiment_score": sentiment_score_result.score if sentiment_score_result else None,
        "sentiment_evidence": sentiment_evidence,
        "sentiment_data_gaps": sentiment_score_result.data_gaps if sentiment_score_result else [],
        "sentiment_cap_applied": sentiment_score_result.cap_applied if sentiment_score_result else None,
        
        # Source tags
        "source_tags": source_tags if source_tags else ["BASE_UNIVERSE"],
    }
    
    return packet

