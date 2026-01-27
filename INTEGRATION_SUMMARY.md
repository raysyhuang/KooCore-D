# Options & Sentiment Data Integration - Summary

## What Was Done

I've updated your codebase to integrate **Polygon options data** and **social sentiment data** into the weekly scanner packets. Previously, these were marked as unavailable, but now they will be fetched and included when you have the proper API keys.

## Changes Made

### 1. Updated `src/core/packets.py`
- Added imports for `compute_options_score` and `compute_sentiment_score`
- Modified `build_weekly_scanner_packet()` to:
  - Accept `polygon_api_key`, `fetch_options`, and `fetch_sentiment` parameters
  - Fetch options data when enabled
  - Fetch sentiment data when enabled
  - Include options and sentiment scores/evidence in packets

### 2. Updated `src/pipelines/weekly.py`
- Modified packet building to:
  - Get `POLYGON_API_KEY` from environment
  - Check config for `features.fetch_options` and `features.fetch_sentiment` flags
  - Pass API key and flags to packet builder

### 3. Updated `src/core/llm.py`
- Updated LLM prompt to reflect that options and sentiment data are now available
- Changed instructions from "data is NOT available" to "data is available when `options_data_available`/`sentiment_data_available` is true"

### 4. Updated `config/default.yaml`
- Added new `features` section with:
  - `fetch_options: true` (enable/disable options fetching)
  - `fetch_sentiment: true` (enable/disable sentiment fetching)

### 5. Created `DATA_SOURCES_GUIDE.md`
- Comprehensive guide explaining:
  - How Polygon options API works (you already have this)
  - Social sentiment data sources (free and paid options)
  - Catalyst/headline sources (beyond Yahoo Finance)
  - How to set up additional APIs if needed

## How to Use

### Step 1: Verify Your Polygon API Key
Make sure your `.env` file has:
```bash
POLYGON_API_KEY=your_actual_polygon_api_key
```

### Step 2: Run the Scanner
You can run either command - **both will use the new options and sentiment data**:

```bash
# Option A: Weekly scanner only
python main.py weekly

# Option B: Complete scan (all systems + hybrid analysis) - RECOMMENDED
python main.py all
```

**Note**: `python main.py all` calls the same `run_weekly()` function, so all changes apply automatically to both commands.

The system will now:
1. ✅ Fetch options data from Polygon for each candidate
2. ✅ Fetch sentiment data from StockTwits, Reddit, and Polygon News
3. ✅ Include this data in the LLM packets
4. ✅ LLM will use actual scores instead of capping at 3.0/4.0

### Step 3: Check the Results
Look at the generated packets JSON file:
```bash
outputs/2026-01-XX/weekly_scanner_packets_2026-01-XX.json
```

You should now see:
- `options_data_available: true` (if data was fetched)
- `options_score: X.X` (actual score 0-10)
- `options_evidence: {...}` (call/put ratios, IV, etc.)
- `sentiment_data_available: true` (if data was fetched)
- `sentiment_score: X.X` (actual score 0-10)
- `sentiment_evidence: {...}` (StockTwits, Reddit, news tone)

### Step 4: (Optional) Disable Features
If you want to disable options or sentiment fetching (e.g., to save API calls), edit `config/default.yaml`:
```yaml
features:
  fetch_options: false  # Disable options fetching
  fetch_sentiment: false  # Disable sentiment fetching
```

## What Data Sources Are Used

### Options Data
- **Primary**: Polygon.io Options Snapshot API (`/v3/snapshot/options/{ticker}`)
- **Fallback**: Yahoo Finance (if Polygon fails)
- **Requires**: Your existing `POLYGON_API_KEY` with options subscription

### Sentiment Data
- **StockTwits**: Free API, no key required
- **Reddit**: Free API, no key required  
- **Polygon News**: Uses your `POLYGON_API_KEY`
- **All are already implemented** in `src/core/sentiment.py`

## Expected Improvements

With options and sentiment data now available:

1. **Better Scoring**: LLM can use actual options flow and sentiment signals instead of capped scores
2. **Higher Confidence**: When all 4 factors have data, confidence levels should improve from "SPECULATIVE" to "MEDIUM" or "HIGH"
3. **Better Rankings**: More data = better differentiation between candidates

## Troubleshooting

### Options Data Not Appearing
- Check that `POLYGON_API_KEY` is set in `.env`
- Verify your Polygon subscription includes options data
- Check logs for API errors
- Try: `python -c "from src.core.options import compute_options_score; print(compute_options_score('AAPL'))"`

### Sentiment Data Not Appearing
- StockTwits and Reddit are free but may have rate limits
- Polygon News requires `POLYGON_API_KEY`
- Check logs for API errors
- Try: `python -c "from src.core.sentiment import compute_sentiment_score; print(compute_sentiment_score('AAPL'))"`

### Performance Concerns
- Options/sentiment fetching adds API calls (may slow down packet building)
- Consider disabling for testing: set `fetch_options: false` or `fetch_sentiment: false` in config
- The system gracefully handles failures (won't crash if APIs are down)

## Next Steps (Optional Enhancements)

If you want even better data coverage, see `DATA_SOURCES_GUIDE.md` for:
- Additional news/catalyst sources (Financial Modeling Prep, EODHD)
- Premium sentiment APIs (Adanos, StockGeist)
- Earnings calendar APIs

## Files Modified

- ✅ `src/core/packets.py` - Added options/sentiment fetching
- ✅ `src/pipelines/weekly.py` - Integrated fetching into pipeline
- ✅ `src/core/llm.py` - Updated prompt to use new data
- ✅ `config/default.yaml` - Added feature flags
- ✅ `DATA_SOURCES_GUIDE.md` - Comprehensive guide (NEW)
- ✅ `INTEGRATION_SUMMARY.md` - This file (NEW)
