# KooCore-D Performance Dashboard

Read-only observability dashboard for KooCore-D stock picking system.

## Design Principles

- **Read-only**: Never writes to `outputs/`
- **No model logic**: Pure visualization, no learning side-effects
- **Derived from outputs/**: All data comes from frozen pipeline outputs
- **Independently deployable**: Can run on Heroku without affecting the main system

## Architecture

```
KooCore-D (cron / GH Actions)
        ↓
   outputs/
        ↓
Dashboard app (Streamlit)
        ↓
   Heroku (read-only)
```

## Pages

### 1. Overview
Portfolio-level view answering "Is the system working?"
- Cumulative returns vs benchmarks (S&P 500, Nasdaq 100)
- Average return by source (Weekly, Pro30, Movers)
- Win rate and summary metrics

### 2. Daily Picks Explorer
Individual pick tracking answering "What did we pick and how is it doing?"
- Date-based pick browsing
- Per-ticker performance charts
- Pick details (scores, catalyst, risk factors)

### 3. Phase-5 Learning Monitor
Learning diagnostics answering "Is learning converging?"
- Hit rate by source, rank, and regime
- Rank decay curves
- Overlap effectiveness analysis
- Temporal trends

## Local Development

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

The dashboard expects `outputs/` to be in the parent directory.

## Heroku Deployment

1. Create Heroku app:
```bash
heroku create koocore-dashboard
```

2. Set config vars (optional):
```bash
heroku config:set POLYGON_API_KEY=your_key_here
```

3. Deploy:
```bash
git subtree push --prefix dashboard heroku main
```

Or deploy from the dashboard folder:
```bash
cd dashboard
git init
git add .
git commit -m "Initial dashboard"
heroku git:remote -a koocore-dashboard
git push heroku main
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `POLYGON_API_KEY` | No | For faster price data (falls back to Yahoo Finance) |
| `KOOCORE_OUTPUTS_PATH` | No | Custom path to outputs directory |

## Safety Rules

These rules ensure the dashboard never interferes with the learning system:

- ❌ No writes to `outputs/`
- ❌ No learning logic imported
- ❌ No retry logic
- ✅ Pure visualization only

## Project Structure

```
dashboard/
├── app.py                  # Streamlit entry point
├── views/
│   ├── overview.py         # Portfolio overview page
│   ├── picks.py            # Daily picks explorer
│   └── phase5.py           # Phase-5 learning monitor
├── data/
│   ├── loader.py           # Read from outputs/
│   └── prices.py           # StockTracker wrapper
├── charts/
│   ├── performance.py      # Performance charts
│   └── learning.py         # Learning diagnostics charts
├── requirements.txt
├── Procfile
└── README.md
```
