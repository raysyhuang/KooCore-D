# 30-Day Momentum Screener PRO

A production-grade momentum trading screener that identifies stocks with high probability of achieving +10% gains within 30 days. Features dynamic multi-universe scanning, regime filtering, catalyst completeness scoring, dilution risk detection, and LLM-powered dual-horizon analysis.

## 🚀 Key Features

- **Multi-Universe Scanning**: SP500 + NASDAQ100 + Russell 2000 (with intelligent caching and fallbacks)
- **Dual-Horizon Analysis**: 7-10 day and 30-day probability scoring via LLM
- **Regime Gate**: Market condition filtering (SPY vs MA20, VIX threshold)
- **Two-Layer Filtering**: Attention pool (Layer 1) + Quality filter (Layer 2)
- **Catalyst Completeness Scoring**: Penalizes unknown earnings and missing headlines
- **Dilution Risk Detection**: Automatic keyword scanning for offerings/ATM/secondary risks
- **Structure Filters**: MA20/MA50 alignment and return requirements for breakouts
- **News Integration**: yfinance headlines with deduplication and manual override support
- **Intraday RVOL Mode**: Optional real-time prorated volume analysis during market hours
- **Split Outputs**: Separate breakout and reversal candidate files

## 📋 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the screener
python momentum_screener.py
```

The script generates multiple output files (see [Output Files](#output-files) section below).

## 📁 Output Files

### `30d_momentum_candidates.csv`
Combined ranked list of top momentum candidates (breakout + reversal).

### `30d_breakout_candidates.csv`
Ranked breakout setups only (near 52W highs with strong structure).

### `30d_reversal_candidates.csv`
Ranked reversal setups only (oversold mean reversion candidates).

**Key Metrics:**
- **Ticker**: Stock symbol
- **Last**: Current price
- **RVOL**: Relative volume (current volume / 20-day average)
- **ATR%**: Average True Range as percentage of price
- **RSI14**: 14-period RSI indicator
- **Dist_to_52W_High%**: Distance from 52-week high (uses High, not Close)
- **$ADV20**: Average dollar volume (20-day, liquidity gate)
- **MA20, MA50**: Moving averages
- **Above_MA20, Above_MA50**: Binary flags (1=above, 0=below)
- **Ret20d%**: 20-day return percentage
- **Setup**: "Breakout" or "Reversal"
- **Score**: Normalized composite momentum score
- **Is_Leveraged_ETF**: Boolean flag
- **Dilution_Risk_Flag**: 0/1 from headline scan
- **Catalyst_Keyword_Flags**: Comma-separated tags (m&a, contract, guidance, etc.)

### `news_dump.csv`
Recent news headlines with deduplication (keeps freshest by ticker+title):
- **published_utc**: UTC timestamp
- **published_local**: ET timezone string
- **title, publisher, link**: Headline details

### `llm_packets.txt`
Formatted analysis packets, one per ticker, containing:
- Market regime snapshot
- Technical metrics with MA structure
- Catalyst completeness score and penalties
- Dilution risk flags
- Earnings information (if available)
- Recent headlines (deduplicated, freshest first)
- Standard trade plan templates by setup type

### `gpt_analysis_output - [DATE].txt`
Dual-horizon analysis output from LLM containing:
- Summary table with scores and verdicts (BUY/WATCH/IGNORE) for both horizons
- Detailed scoring breakdown per ticker
- Top rankings for 7-10d and 30d horizons
- Trade plans with entry triggers, stops, and targets

### `universe_cache.csv`
Cached universe ticker list (auto-generated, refreshed every 7 days by default).

## 🔄 Workflow

### Step 1: Run Screener

```bash
python momentum_screener.py
```

**Market Hours Warning**: If run during market hours without intraday mode enabled, you'll see a warning about partial-day volume. For best accuracy, run after market close or enable `enable_intraday_attention=True` in PARAMS.

### Step 2: Review Candidates

Open the CSV files to see ranked candidates:
- `30d_breakout_candidates.csv` - Near-high momentum plays
- `30d_reversal_candidates.csv` - Oversold mean reversion plays
- `30d_momentum_candidates.csv` - Combined top N

### Step 3: Add Manual Headlines (Optional)

Since yfinance news feed is often incomplete, manually add headlines for important tickers:

1. Create/edit `manual_headlines.csv` (if it doesn't exist)
2. Add headlines in format: `Ticker,Date,Source,Headline`
3. Example:

   ```csv
   WDC,2025-12-20,Yahoo Finance,Western Digital announces new partnership
   AVGO,2025-12-19,Bloomberg,Broadcom reports strong quarterly results
   ```

The script automatically loads and includes these in the LLM packets on the next run.

### Step 4: LLM Analysis

#### Option A: Batch Analysis (Recommended)

1. Open `batch_prompt_gpt.txt` and `llm_packets.txt`
2. Copy the batch prompt
3. Paste into ChatGPT/GPT-4
4. Copy all ticker blocks from `llm_packets.txt`
5. Paste below the prompt
6. Get ranked dual-horizon analysis with top BUY recommendations

The batch prompt will:
- Score each ticker for both 7-10 day and 30-day horizons
- Apply strict BUY gates (max 1 BUY for 7-10d, max 2 BUY for 30d)
- Produce summary table with entry/stops per horizon
- Rank top candidates with rationale

#### Option B: Individual Analysis

Paste individual ticker blocks from `llm_packets.txt` into ChatGPT/Gemini one at a time for focused analysis.

## ⚙️ Configuration

Edit the `PARAMS` dictionary in `momentum_screener.py` to customize the screener:

### Universe Selection

- `universe_mode`: "SP500", "SP500+NASDAQ100", or "SP500+NASDAQ100+R2000" (default: "SP500+NASDAQ100+R2000")
- `universe_cache_file`: Cache file path (default: "universe_cache.csv")
- `universe_cache_max_age_days`: Cache refresh interval (default: 7)

### Attention Pool (Layer 1)

- `attention_rvol_min`: Minimum relative volume for attention pool (default: 1.8)
- `attention_atr_pct_min`: Minimum ATR% for attention pool (default: 3.5)
- `attention_min_abs_day_move_pct`: Minimum daily move % filter (default: 3.0, set None to disable)
- `attention_lookback_days`: Lookback period (default: 120)
- `attention_chunk_size`: Download chunk size (default: 200)

### Quality Filters (Layer 2)

- `price_min`: Minimum stock price (default: 7.0)
- `avg_vol_min`: Minimum average daily volume in shares (default: 1,000,000)
- `avg_dollar_vol_min`: **NEW** Minimum average dollar volume (default: 20,000,000) - liquidity gate
- `rvol_min`: Minimum relative volume threshold (default: 2.0)
- `atr_pct_min`: Minimum ATR% threshold (default: 4.0)
- `near_high_max_pct`: Max distance from 52W high for breakout (default: 8.0%)
- `rsi_reversal_max`: Max RSI for reversal candidates (default: 35.0)

### Setup-Specific Filters

- `breakout_rsi_min`: Minimum RSI for breakouts (default: 55.0) - avoids weak near-high names
- `reversal_dist_to_high_min_pct`: Minimum distance from 52W high for reversals (default: 15.0) - avoids tiny pullbacks
- `reversal_rsi_max`: Max RSI for reversals (default: 32.0) - stricter than general threshold

### Output Controls

- `top_n_breakout`: Top N breakout candidates (default: 15)
- `top_n_reversal`: Top N reversal candidates (default: 15)
- `top_n_total`: Top N combined candidates (default: 25)

### Regime Gate

- `enable_regime_gate`: Enable/disable regime filtering (default: True)
- `spy_symbol`: SPY symbol (default: "SPY")
- `vix_symbol`: VIX symbol (default: "^VIX")
- `spy_ma_days`: Moving average period for SPY (default: 20)
- `vix_max`: Maximum VIX for risk-on regime (default: 25.0)
- `regime_action`: Action when regime gate fails - "WARN" (continue) or "BLOCK" (stop) (default: "WARN")

### Intraday Mode (Optional)

- `enable_intraday_attention`: Enable prorated intraday RVOL (default: False)
- `intraday_interval`: Bar interval for intraday data (default: "5m")
- `intraday_lookback_days`: Historical days for intraday comparison (default: 5)
- `market_open_buffer_min`: Ignore first N minutes after open (default: 20)
- `intraday_rvol_min`: Minimum intraday RVOL threshold (default: 2.0)

**Note**: Intraday mode requires market to be open. For best accuracy, run after market close with `enable_intraday_attention=False`.

### News & Packets

- `news_max_items`: Maximum news items per ticker (default: 25)
- `packet_headlines`: Headlines per LLM packet (default: 12)
- `throttle_sec`: Delay between API calls (default: 0.15)

### Operational

- `lookback_days`: Historical data lookback period (default: 365)

## 🏗️ System Architecture

The screener uses a sophisticated multi-layer filtering approach:

### 1. Regime Gate (Pre-Filter)
- Checks if SPY > MA20 (trend confirmation)
- Checks if VIX <= threshold (risk-on environment)
- Can warn or block candidates if regime is unfavorable

### 2. Dynamic Universe (Multi-Source)
- Fetches SP500, NASDAQ100, and Russell 2000 tickers
- Uses intelligent caching (7-day default refresh)
- Robust retry logic with timeouts
- Graceful fallback if Russell 2000 fetch fails

### 3. Attention Pool (Layer 1)
- Scans entire universe for high-RVOL, high-ATR names
- Optional intraday prorated RVOL mode (market hours)
- Market-open detection with warnings
- Creates initial candidate pool

### 4. Quality Filter (Layer 2)
- **Liquidity Gate**: $ADV20 filter ($20M/day default)
- **Structure Requirements**: 
  - Breakouts require: Above MA20 AND Above MA50 AND Ret20d > 0
  - Reversals require: RSI <= threshold AND sufficient distance from highs
- **52W High Fix**: Uses High (not Close) for accurate distance calculation
- **Setup Separation**: Breakout and reversal scored separately with normalized scoring

### 5. Catalyst Analysis
- **Completeness Scoring**: 0-100 score penalizing unknown earnings and missing headlines
- **Dilution Risk Scanning**: Automatic keyword detection (offering, ATM, secondary, etc.)
- **Catalyst Tagging**: M&A, contracts, guidance, FDA/clinical, crypto tags

### 6. News Integration
- Fetches recent headlines from yfinance
- Deduplication by (Ticker, title) keeping freshest
- Local timezone formatting (ET)
- Manual headline override support

### 7. LLM Packet Generation
- Formats all data into structured prompts
- Includes regime info, metrics, MA structure, completeness scores
- Standard trade plan templates by setup type
- Ready for dual-horizon probability scoring

## 📊 Scoring System

The screener uses a **normalized scoring system** for comparability:

### Components

1. **TapeScore** = f(RVOL, ATR%)
   - RVOL weighted 2.0x
   - ATR% weighted 1.4x

2. **StructureScore** = f(RSI, distance to high, MA alignment)
   - RSI structure bonus (prefers 58-66 range)
   - Distance to high bonus
   - MA alignment bonus (Above MA20/50)

3. **SetupBonus** = Fixed bonus by setup type
   - Breakout: +4.0
   - Reversal: +3.0 + distance bonus

**Final Score** = TapeScore + StructureScore + SetupBonus

This ensures breakout and reversal scores are comparable when ranking combined candidates.

## 🎯 Key Improvements

### Production-Ready Features

1. **Market Hours Detection**: Warns if running during market hours without intraday mode
2. **Universe Reliability**: Retry logic, timeouts, and fallbacks for all data sources
3. **Dilution Risk Detection**: Automatic scanning prevents false breakouts from offerings
4. **Structure Filters**: MA20/MA50 requirements prevent "falling knife" breakouts
5. **Normalized Scoring**: Comparable scores across setup types

### Data Quality

1. **52W High Fix**: Uses High (not Close) for accurate breakout detection
2. **News Deduplication**: Keeps freshest headlines, removes duplicates
3. **Catalyst Completeness**: Explicit scoring penalizes missing data
4. **Liquidity Gate**: $ADV20 filter ensures tradeable names

### Analysis Enhancements

1. **Split Outputs**: Separate breakout and reversal files for focused analysis
2. **Dual-Horizon LLM**: 7-10 day and 30-day probability scoring
3. **Standard Trade Plans**: Setup-specific entry/stop/TP templates
4. **Dilution Flags**: Explicit risk warnings in metrics

## 📝 Usage Tips

### Best Practices

1. **Run After Market Close**: For best accuracy, run with `enable_intraday_attention=False` after 4 PM ET
2. **Intraday Mode**: If running during market hours, set `enable_intraday_attention=True` for prorated RVOL
3. **Manual Headlines**: Add important catalysts to `manual_headlines.csv` for better completeness scores
4. **Universe Caching**: First run fetches universe (may take 1-2 min), subsequent runs use cache
5. **Russell 2000**: If fetch fails, screener continues with SP500+NASDAQ100 (graceful degradation)

### Tuning for Different Timeframes

- **7-10 Day Movers**: Increase `atr_pct_min` (5.0-6.5) and `rvol_min` (2.2-3.0)
- **30 Day Movers**: Current defaults work well
- **Wider Net**: Lower `attention_rvol_min` (1.6) and `rvol_min` (1.5)
- **Tighter Filter**: Raise `avg_dollar_vol_min` ($30M+) and `atr_pct_min` (5.0+)

## ⚠️ Disclaimer

This is a candidate generator tool for educational/research purposes only. Not financial advice. Always conduct your own due diligence before making trading decisions. The screener identifies potential setups but does not guarantee outcomes.

## 📦 Requirements

See `requirements.txt` for dependencies. Key packages:

- `yfinance>=0.2.0`: Stock data, news, and earnings
- `pandas>=1.5.0`: Data manipulation
- `numpy>=1.24.0`: Numerical calculations
- `lxml>=4.9.0`, `html5lib>=1.1`: HTML parsing for universe lists
- `requests>=2.28.0`: HTTP requests with retry logic

## 🔧 Troubleshooting

### Russell 2000 Fetch Fails
- **Symptom**: Warning message about Russell 2000 fetch failure
- **Impact**: Continues with SP500+NASDAQ100 (still ~600 tickers)
- **Solution**: Check internet connection, or set `universe_mode="SP500+NASDAQ100"` to skip Russell

### No Candidates Found
- **Symptom**: Empty CSV files
- **Solutions**: 
  - Lower `rvol_min` (try 1.5) or `atr_pct_min` (try 3.5)
  - Lower `attention_rvol_min` (try 1.6)
  - Check if regime gate is blocking (set `regime_action="WARN"`)

### Market Hours Warning
- **Symptom**: Warning about partial-day volume
- **Solutions**:
  - Run after market close (recommended)
  - Set `enable_intraday_attention=True` for intraday mode
  - Accept warning and continue (less accurate)

### News Feed Empty
- **Symptom**: No headlines in packets
- **Impact**: Lower catalyst completeness scores
- **Solution**: Add manual headlines to `manual_headlines.csv`

## 📚 File Structure

```
short-term trading tracker pro/
├── momentum_screener.py          # Main screener script
├── batch_prompt_gpt.txt          # Dual-horizon LLM prompt
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── manual_headlines.csv          # Manual headline overrides (optional)
├── universe_cache.csv            # Cached universe (auto-generated)
├── 30d_momentum_candidates.csv  # Combined output
├── 30d_breakout_candidates.csv   # Breakout setups
├── 30d_reversal_candidates.csv   # Reversal setups
├── news_dump.csv                 # News headlines
└── llm_packets.txt               # LLM analysis packets
```

## 🚀 Future Enhancements

Potential improvements (not yet implemented):
- Real-time price alerts
- Performance tracking integration
- Additional universe sources (international, crypto)
- Advanced technical pattern detection
- Multi-timeframe confirmation

---

**Version**: 2.0 (Production-Ready)  
**Last Updated**: December 2025
