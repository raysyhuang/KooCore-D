# Data Sources Integration Guide

This guide explains how to integrate options data, social sentiment, and catalyst/headline sources into the trading tracker.

---

## Quick Setup (3 Steps)

Add these to your `.env` file:

```bash
# Required (you already have this)
POLYGON_API_KEY=your_polygon_key

# Recommended additions (free tiers available)
FMP_API_KEY=your_fmp_key          # Earnings calendar + news (free tier)
ADANOS_API_KEY=your_adanos_key    # Twitter sentiment (free: 250 calls/month)
```

Then run `python main.py all` - the system will automatically use these sources!

---

## 1. Options Data - Polygon.io ✅ INTEGRATED

### Status: Fully Working
- ✅ Fetches options chain from Polygon API
- ✅ Computes call/put ratio, unusual volume, IV rank
- ✅ Includes in LLM packets for scoring

### Setup
Your `.env` should have:
```bash
POLYGON_API_KEY=your_polygon_api_key_here
```

### What It Provides
- Call/put volume ratio (bullish indicator)
- Unusual options volume detection
- Notable contracts (large trades, short-dated OTM calls)
- Implied volatility metrics

---

## 2. Social Sentiment Data Sources

### Free Sources (Built-in) ✅

| Source | Status | Data Provided |
|--------|--------|---------------|
| **StockTwits** | ✅ Working | Bull/bear ratio, message count |
| **Reddit** | ✅ Working | Mentions on WSB/stocks/investing, upvotes |
| **Polygon News** | ✅ Working | News sentiment, headlines |

### Adanos - Twitter/X Sentiment ✅ NEW

**Status**: ✅ Integrated (requires API key)

**Sign Up**: https://adanos.org (free tier: ~250 calls/month)

**Add to `.env`**:
```bash
ADANOS_API_KEY=your_adanos_key_here
```

**What It Provides**:
- Twitter/X stock sentiment (-1 to +1 scale)
- Mention velocity vs 7-day average
- Buzz score (0-100)
- Reddit sentiment from 50+ subreddits
- Trending detection

### Enhanced Sentiment Scoring

When Adanos is configured, the system uses multi-source sentiment:

| Source | Weight | Data |
|--------|--------|------|
| Adanos (Twitter) | 40% | Twitter sentiment, velocity, engagement |
| StockTwits | 20% | Bull/bear ratio |
| Reddit | 20% | Mentions, upvotes |
| News | 20% | Headline tone |

---

## 3. Catalyst/Headline Sources

### Financial Modeling Prep (FMP) ✅ NEW

**Status**: ✅ Integrated (requires API key)

**Sign Up**: https://site.financialmodelingprep.com (free tier available)

**Add to `.env`**:
```bash
FMP_API_KEY=your_fmp_key_here
```

**What It Provides**:
- **Earnings Calendar**: Confirmed dates, EPS estimates, revenue estimates
- **News Headlines**: Real-time news feed
- **Timing Info**: Before market open (BMO) / After market close (AMC)

### Polygon News (Enhanced) ✅

Already integrated with your `POLYGON_API_KEY`. Provides:
- Up to 20 headlines per ticker
- Sentiment analysis on headlines
- Publisher info and dates

### Headlines Priority Order

1. Manual headlines (from `manual_headlines.csv`)
2. Polygon News headlines
3. FMP News headlines
4. Yahoo Finance headlines (fallback)

---

## 4. Environment Variables Summary

### Your `.env` File

```bash
# ═══════════════════════════════════════════════════════════════════
# REQUIRED - You already have this
# ═══════════════════════════════════════════════════════════════════
POLYGON_API_KEY=your_polygon_key

# ═══════════════════════════════════════════════════════════════════
# RECOMMENDED - Free tiers available
# ═══════════════════════════════════════════════════════════════════

# Financial Modeling Prep - Earnings calendar + news
# Sign up: https://site.financialmodelingprep.com
FMP_API_KEY=your_fmp_key

# Adanos - Twitter/X sentiment (free: 250 calls/month)
# Sign up: https://adanos.org
ADANOS_API_KEY=your_adanos_key

# ═══════════════════════════════════════════════════════════════════
# OPTIONAL - For Telegram alerts (you already have this set up)
# ═══════════════════════════════════════════════════════════════════
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

---

## 5. Config Options

In `config/default.yaml`:

```yaml
features:
  # Fetch options data from Polygon API
  fetch_options: true
  
  # Fetch sentiment data
  fetch_sentiment: true
  
  # Use enhanced sentiment (Adanos + FMP)
  # Set to false to use basic sentiment only
  use_enhanced_sentiment: true
```

---

## 6. Testing Your Setup

### Test Individual Sources

```python
# Test Adanos (Twitter sentiment)
from src.core.sentiment import fetch_adanos_sentiment
result = fetch_adanos_sentiment("AAPL")
print(result)

# Test FMP (Earnings calendar)
from src.core.sentiment import fetch_fmp_earnings_calendar
result = fetch_fmp_earnings_calendar("AAPL")
print(result)

# Test FMP (News)
from src.core.sentiment import fetch_fmp_news
result = fetch_fmp_news("AAPL")
print(result)

# Test Polygon News (Enhanced)
from src.core.sentiment import fetch_polygon_news_headlines
result = fetch_polygon_news_headlines("AAPL")
print(result)

# Test Enhanced Sentiment (All sources)
from src.core.sentiment import compute_enhanced_sentiment_score
result = compute_enhanced_sentiment_score("AAPL")
print(f"Score: {result.score}, Sources: {result.evidence['data_source']}")
```

### Run Full Scan

```bash
python main.py all
```

Check the output for:
```
Fetching options & sentiment data (Polygon, FMP, Adanos) (this may take a moment)...
```

---

## 7. Cost Considerations

| Source | Free Tier | Paid Plans |
|--------|-----------|------------|
| **Polygon.io** | You already pay | Options subscription |
| **StockTwits** | ✅ Free | N/A |
| **Reddit** | ✅ Free | N/A |
| **Yahoo Finance** | ✅ Free | N/A |
| **FMP** | ✅ 250 calls/day | ~$14/month |
| **Adanos** | ✅ 250 calls/month | Plans available |

**Recommendation**: Start with free tiers. With 30 candidates per scan, you'd use ~30 Adanos calls per run, giving you ~8 runs/month on the free tier.

---

## 8. What Data Appears in Packets

After integration, each LLM packet includes:

```json
{
  "ticker": "AAPL",
  
  // Earnings (from FMP)
  "earnings_date": "2026-02-05",
  "earnings_info": {
    "earnings_date": "2026-02-05",
    "eps_estimate": 2.35,
    "revenue_estimate": 120000000000,
    "time": "amc",
    "source": "fmp"
  },
  
  // Headlines (Polygon + FMP + Yahoo)
  "headlines": [
    {"title": "Apple Announces New Product Line", "publisher": "Reuters", ...},
    ...
  ],
  "catalyst_sources": ["polygon_news", "fmp_news"],
  
  // Sentiment (Adanos + StockTwits + Reddit + News)
  "sentiment_score": 7.5,
  "sentiment_evidence": {
    "twitter": {"mention_velocity_vs_7d": 2.3, "bullish_pct_est": 68},
    "reddit": {"mention_velocity": 15, "upvote_ratio_est": 85},
    "stocktwits": {"bull_bear_ratio_est": 2.5},
    "news_tone": "positive",
    "data_source": "adanos+stocktwits+polygon"
  },
  
  // Options (Polygon)
  "options_score": 6.5,
  "options_evidence": {
    "call_put_ratio": 1.8,
    "unusual_volume_multiple": 2.3,
    ...
  }
}
```

---

## 9. Troubleshooting

### No Adanos Data
- Check `ADANOS_API_KEY` is set in `.env`
- Verify at https://adanos.org that your key is active
- Check if you've exceeded the free tier (250 calls/month)

### No FMP Earnings
- Check `FMP_API_KEY` is set in `.env`
- Verify at FMP dashboard that your key is active
- Some tickers may not have upcoming earnings

### Sentiment Score Capped
- Score capped at 3.0 = no sources available
- Score capped at 5.0 = only 1 source
- Score capped at 7.0 = only 2 sources
- No cap = 3+ sources active
