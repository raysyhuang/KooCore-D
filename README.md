# Momentum Trading System

**A comprehensive momentum trading system with multi-factor scoring, automated backtesting, performance tracking, intelligent caching, and real-time alerts.**

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables (create .env file)
POLYGON_API_KEY=your_polygon_key
OPENAI_API_KEY=your_openai_key
TELEGRAM_BOT_TOKEN=your_telegram_token  # Optional
TELEGRAM_CHAT_ID=your_chat_id           # Optional

# ⭐ Just run this - everything is integrated!
python main.py all
```

**`python main.py all` now automatically runs:**
1. Daily Movers Discovery (with momentum/reversal analysis)
2. Weekly Scanner (with rank-based filtering)
3. 30-Day Screener (Pro30) - weighted 2x in hybrid analysis
4. LLM Ranking & Weighted Hybrid Analysis
5. **Position Tracking** ← Auto-tracks new picks!
6. **Drawdown Monitoring** ← Alerts at -5% warning, -7% stop
7. **Quick Model Validation** ← Auto-included!
8. **Confluence Analysis** ← Auto-included!

---

## 📋 Available Commands

| Command | Description |
|---------|-------------|
| `python main.py all` | **⭐ RECOMMENDED** - Everything in one command |
| `python main.py weekly` | Weekly scanner only |
| `python main.py pro30` | 30-day screener only (57.9% hit rate!) |
| `python main.py movers` | Daily movers with momentum analysis |
| `python main.py confluence` | Multi-factor signal alignment |
| `python main.py options` | Options flow / smart money scanner |
| `python main.py sector` | Sector rotation & leaders |
| `python main.py validate` | Full multi-period model validation |
| `python main.py performance` | Backtest historical picks |
| `python main.py track` | Position tracking & journaling |
| `python main.py track monitor` | **NEW** - Monitor positions for drawdown alerts |
| `python main.py cache` | **NEW** - Cache management (stats, clear, prefetch) |
| `python main.py api` | Start REST API dashboard |
| `python main.py replay` | Replay historical dates |

---

## 🎯 Core Features

### 1. Multi-Factor Scoring Model

The system uses a 4-factor model to identify high-probability momentum plays:

| Factor | Weight | Data Source |
|--------|--------|-------------|
| Technical Momentum | 35% | yfinance / Polygon |
| Upcoming Catalyst | 35% | News APIs + LLM |
| Options Activity | 15% | Polygon (paid) |
| Sentiment Momentum | 15% | Polygon News + StockTwits |

### 2. Dual Screening Pipelines

- **Weekly Scanner**: High-velocity 7-day momentum bursts, catalyst-driven (Top 5)
- **30-Day Screener**: Conservative positions with dual-horizon analysis (Top 15-25)
- **Daily Movers**: Quarantined idea funnel for unusual price action

### 3. Weighted Hybrid Analysis

Cross-references all screeners with **weighted scoring** based on backtested performance:

| Source | Weight | Rationale |
|--------|--------|-----------|
| Pro30 | 2.0x | Highest hit rate (57.9%) |
| Weekly Top 5 | 1.0x | Good catalyst identification |
| Weekly Rank 1 | +0.5 bonus | 38% hit rate vs 27.6% average |
| Movers | 0.5x | Lower historical hit rate |

**Overlap Bonuses:**
- ⭐ **ALL THREE** overlap: +3.0 bonus (highest conviction)
- 🔥 **Weekly + Pro30** overlap: +1.5 bonus

### 4. Risk Management & Position Tracking

Automatic position monitoring with configurable alerts:

| Threshold | Action |
|-----------|--------|
| -5% drawdown | ⚠️ Warning alert |
| -7% drawdown | 🛑 Stop-loss alert |
| +7% gain | ✅ Profit target alert |
| 7 days held | ⏰ Holding period expiry |

```yaml
# config/default.yaml
risk_management:
  warning_drawdown_pct: -5.0
  suggested_stop_loss_pct: -7.0
  suggested_profit_target_pct: 7.0
  default_holding_days: 7
```

---

## 💾 Cache System (NEW)

The system includes intelligent price data caching to reduce API calls and ensure consistent backtesting:

```bash
# View cache statistics
python main.py cache stats

# Clear expired entries
python main.py cache cleanup

# Clear all cache data
python main.py cache clear --confirm

# Prefetch historical data for faster scans
python main.py cache prefetch --start 2025-09-01 --end 2026-01-15
```

**Cache Features:**
- **SQLite backend** - Fast, file-based, no external dependencies
- **Smart TTL** - 30 days for historical data, 1 hour for live data
- **Coverage validation** - Requires 60% date coverage before treating as cache hit
- **Shared across pipelines** - Polygon and yfinance data in unified cache

---

## 📊 Model Validation & Backtesting

### Quick Validation (after each scan)

```bash
python main.py validate --quick
```

Output:
```
📊 MODEL QUALITY SCORECARD
═══════════════════════════════════════════════════

🎯 Primary KPI (Hit +7% within T+7 days):
   • Hit Rate: 43.9%
   • Win Rate: 68.2%
   • Avg Return: 3.9%

🏆 Strategy Ranking:
   1. pro30: Hit=57.9% | Return=3.7%  ← BEST!
   2. weekly_top5: Hit=40.0% | Return=4.0%

💊 Model Health: 🟢 Excellent
```

> **Note**: Target adjusted from +10% to +7% based on backtesting. Pro30 significantly outperforms Weekly - prioritize Pro30 picks!

### Full Multi-Period Validation

```bash
# Test all holding periods (5, 7, 10, 14 days)
python main.py validate

# Custom date range
python main.py validate --start 2025-12-01 --end 2026-01-10

# Custom periods and thresholds
python main.py validate --periods 5,7,14 --thresholds 5,10,20
```

### Performance Backtest

```bash
# Backtest all historical picks
python main.py performance

# Custom parameters
python main.py performance --forward-days 10 --threshold 15
```

---

## 🎯 Advanced Scanners (NEW!)

### 1. Confluence Scanner - Highest Conviction Picks

The confluence scanner finds stocks where **multiple independent signals align**, dramatically improving hit rates:

```bash
# Find stocks with 2+ aligned signals
python main.py confluence --min-signals 2

# Include options flow and sector analysis
python main.py confluence --min-signals 3
```

| Signals Aligned | Expected Hit Rate |
|-----------------|-------------------|
| 1 signal | ~40-50% |
| 2 signals | ~55-65% |
| 3+ signals | ~65-75% |

### 2. Options Flow Scanner - Smart Money Tracking

Detects unusual options activity that often precedes significant moves:

```bash
# Scan specific tickers
python main.py options --tickers AAPL,MSFT,NVDA

# Auto-scan from today's picks
python main.py options
```

**Signals detected:**
- Unusual Volume (Volume >> Open Interest)
- Call Sweeps (aggressive bullish buying)
- Put/Call Ratio shifts
- IV expansion (market pricing in a move)

### 3. Sector Rotation Scanner

Identifies hot sectors and their leading stocks:

```bash
python main.py sector
```

**Output:**
```
🏭 SECTOR ROTATION ANALYSIS
═══════════════════════════════════

📊 SECTOR MOMENTUM RANKING
1. Technology (XLK): 8.2/10 🚀
   1W: +3.5% | 1M: +8.2% | RS: +2.1

🏆 SECTOR LEADERS
1. NVDA [Technology] - Score: 9.1/10
   1W: +5.2% | vs Sector: +1.7%
```

---

## 📈 Position Tracking

Track your trades and measure actual performance:

```bash
# Log a new position
python main.py track entry AAPL --price 150.00 --shares 100 --source weekly

# Close a position
python main.py track exit POS_001 --price 165.00

# View positions
python main.py track list --status open

# Performance summary
python main.py track summary

# Export to CSV
python main.py track export --output my_trades.csv

# NEW: Monitor open positions for drawdown/profit alerts
python main.py track monitor --alert
```

**Auto-Tracking**: When you run `python main.py all`, positions are automatically tracked from scan results. The system monitors all open positions and generates alerts when thresholds are hit.

---

## 🔔 Alert System

Get notified when high-conviction signals are detected.

### Supported Channels

| Channel | Setup Required |
|---------|----------------|
| **Telegram** | `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` in `.env` |
| **Desktop** | None (uses native OS notifications) |
| **File Log** | None (writes to `outputs/alerts.log`) |
| **Email** | SMTP settings in config |
| **Slack** | Webhook URL in config |
| **Discord** | Webhook URL in config |

### Enhanced Telegram Alerts (NEW)

Telegram notifications now include:
- **Ticker lists** for non-zero categories (Weekly Top 5, Pro30, Movers)
- **Model health metrics** (Hit Rate, Win Rate)
- **Per-source performance** breakdown
- **Position alerts** for drawdown warnings and stop-loss triggers

Example alert:
```
📊 Scan Complete: 2026-01-15

• Weekly Top 5: 5
  AKAM, INTC, CAH, APH, CHRW

• Pro30 Candidates: 3
  NVDA, AMD, TSLA

📈 Model Health:
  Hit Rate (+7%): 36.2%
  Win Rate: 65.2%
```

### Configuration (`config/default.yaml`)

```yaml
alerts:
  enabled: true
  channels: ["telegram", "desktop", "file"]
  
  telegram:
    bot_token: ${TELEGRAM_BOT_TOKEN}
    chat_id: ${TELEGRAM_CHAT_ID}
```

---

## 🔌 Data Sources

### Polygon.io (Recommended)

Set `POLYGON_API_KEY` in `.env` for better data quality:

| Feature | Free Tier | Paid Tier |
|---------|-----------|-----------|
| Daily OHLCV | ✅ | ✅ |
| Options Snapshots | ❌ | ✅ (IV, Volume, OI) |
| News Sentiment | ✅ (limited) | ✅ (full) |
| Intraday Data | ❌ | ✅ |

### Fallback: Yahoo Finance

If Polygon is unavailable, the system automatically falls back to Yahoo Finance.

---

## 📓 Jupyter Notebooks

### `Notebook_tracker_2026-01.ipynb`

**Purpose**: Track performance of scanner picks over time

Features:
- Visual bar charts of returns
- Cumulative return line charts
- Benchmark comparison (S&P 500, Nasdaq 100)
- Strategy-level breakdown (Weekly vs Pro30 vs Movers)

### `Notebook_model_validation.ipynb`

**Purpose**: Comprehensive model validation and improvement analysis

Features:
- Multi-period hit rate matrix (T+5, T+7, T+10, T+14)
- Strategy comparison charts
- Factor attribution analysis
- Automated improvement suggestions
- Model health scorecard

---

## 📁 Output Structure

```
outputs/
├── YYYY-MM-DD/                          # Per-run outputs
│   ├── hybrid_analysis_{date}.json      # Full picks + weighted scores
│   ├── hybrid_report_{date}.html        # Visual report with charts
│   ├── weekly_scanner_candidates_{date}.csv  # Raw candidates
│   ├── weekly_scanner_packets_{date}.json    # LLM input packets
│   ├── top5_{date}.csv                  # Final Weekly Top 5
│   ├── dropped_weekly_{date}.csv        # Filtered tickers + reasons
│   ├── dropped_pro30_{date}.csv         # Pro30 filtered tickers
│   ├── run_card_{date}.json             # Run summary
│   ├── summary_{date}.md                # Human-readable summary
│   └── report_{date}.html               # HTML report
├── performance/                         # Validation outputs
│   ├── perf_detail.csv                  # Per-pick outcomes
│   ├── perf_by_component.csv            # Strategy comparison
│   ├── validation_scorecard_{date}.json
│   └── recommendations.md               # Config suggestions
├── logs/                                # Run logs
│   └── all_run_{date}.log               # Detailed execution log
├── model_history.md                     # Running log (auto-appended)
└── alerts.log                           # Alert history

data/
├── price_cache.db                       # SQLite price cache
├── positions/
│   └── open_positions.json              # Tracked positions
└── processed/
    └── mover_queue.csv                  # Movers cooling queue
```

---

## 🔄 Recommended Daily Workflow

```bash
# 1. Run daily scan (after market close)
python main.py all

# 2. Quick performance check
python main.py validate --quick

# 3. Review model_history.md for trends
# 4. Weekly: Full validation
python main.py validate
```

---

## 🛠️ Configuration

All settings in `config/default.yaml`:

```yaml
# Universe
universe:
  mode: "SP500+NASDAQ100+R2000"

# Liquidity filters
liquidity:
  min_avg_dollar_volume_20d: 50000000

# Quality filters (calibrated from backtesting)
quality_filters_weekly:
  min_technical_score: 6.0
  min_composite_score: 5.25
  top_ranks_only: 3        # Only Ranks 1-3 (excludes underperforming 4,5)
  prefer_rank_1: true

# Risk management
risk_management:
  warning_drawdown_pct: -5.0
  suggested_stop_loss_pct: -7.0
  suggested_profit_target_pct: 7.0
  default_holding_days: 7

# Hybrid weighting (based on backtest performance)
hybrid_weighting:
  pro30_weight: 2.0          # Pro30 has highest hit rate
  weekly_weight: 1.0
  weekly_rank1_bonus: 0.5    # Rank 1 has 38% hit rate
  movers_weight: 0.5
  all_three_overlap_bonus: 3.0
  weekly_pro30_overlap_bonus: 1.5

# Movers with momentum analysis
movers:
  enabled: true
  momentum_analysis:
    enabled: true
    min_momentum_score: 3
    gainer_rsi_overbought: 70
    loser_rsi_oversold: 30

# Alerts
alerts:
  enabled: true
  channels: ["telegram", "desktop", "file"]

# Cache (reduces API calls)
cache:
  enabled: true
  backend: "sqlite"
  price_ttl_seconds: 3600
  historical_ttl_seconds: 2592000  # 30 days
```

---

## 📊 Using Cursor for Model Improvement

After running scans for a few days/weeks, ask Cursor:

> "Look at `outputs/model_history.md` and `outputs/performance/perf_detail.csv`. Which picks consistently failed? What patterns do you see?"

> "Analyze the `dropped_pro30` files - are we filtering out stocks that would have been winners?"

> "The validation shows Pro30 outperforms Weekly. What config changes should I make?"

---

## 🧪 API Server (Development)

```bash
# Start the REST API
python main.py api --port 8000 --reload

# Endpoints:
# GET  /api/latest          - Latest scan results
# GET  /api/scans           - List all scans
# GET  /api/scan/{date}     - Specific date results
# GET  /api/performance     - Performance metrics
# GET  /                    - Web dashboard
```

---

## 📈 System Effectiveness

**For 7-day, 10% moves:**
- **Methodology**: Solid multi-factor approach
- **Realistic Expectation**: 25-35% hit rate (based on backtests)
- **Key Differentiator**: Catalyst identification via LLM

**Recommendations:**
- Use as a **filter**, not the sole signal
- Focus on **overlaps** (Weekly + Pro30) for highest conviction
- Track performance with `validate --quick` after each run
- Always use stop-losses and proper risk management

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| SSL/TLS Errors | yfinance retries automatically |
| No Polygon data | Check API key; system falls back to Yahoo |
| Empty candidates | Lower filters in config or expand universe |
| Telegram 404 | Verify bot token format: `bot<TOKEN>` |
| Slow downloads | Run `python main.py cache prefetch` first |
| 0 candidates after cache clear | Cache needs 60%+ date coverage; prefetch historical data |
| `zsh: terminated` | Out of memory; reduce batch size or universe |
| Stale cache data | Run `python main.py cache clear --confirm` |
| Date mismatch errors | System auto-uses last trading day; verify with logs |

---

## 📦 Dependencies

Core:
- `pandas`, `numpy` - Data processing
- `yfinance` - Market data (fallback)
- `requests` - API calls
- `pyyaml` - Configuration
- `plotly` - Visualizations (notebooks)

Optional:
- `python-dotenv` - Environment variables
- `pydantic` - Typed configuration
- `fastapi`, `uvicorn` - REST API
- `redis` - Distributed caching

Install all:
```bash
pip install -r requirements.txt
```

---

## 📝 Changelog

### v3.2 (January 2026)

**New Features:**
- **Intelligent Price Cache** - SQLite-based caching with coverage validation (60% minimum)
- **Cache CLI** - `python main.py cache stats|clear|prefetch|cleanup`
- **Position Tracking** - Auto-tracks picks, monitors drawdowns, generates alerts
- **Position Monitor** - `python main.py track monitor --alert`
- **Weighted Hybrid Analysis** - Pro30 weighted 2x based on backtested performance
- **Rank-Based Filtering** - Only Ranks 1-3 included (Ranks 4-5 underperform)
- **Enhanced Movers** - Momentum continuation & reversal analysis (RSI, MAs, volume)
- **Rich Telegram Alerts** - Ticker lists, model health, per-source metrics

**Bug Fixes:**
- Fixed date handling - all pipelines now consistently use last trading day
- Fixed mover queue timezone issues (ISO8601 parsing)
- Fixed cache coverage validation (prevents stale data from being used)
- Fixed `polygon_api_key` scoping in weekly scanner
- Replaced bare `except:` clauses with specific exception handling

**Performance:**
- Shared cache across Polygon and yfinance reduces API calls
- Prefetch command for faster subsequent scans
- Historical data cached for 30 days

---

## 📄 License

MIT License - See LICENSE file for details.
