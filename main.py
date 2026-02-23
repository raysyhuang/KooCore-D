#!/usr/bin/env python3
"""
Momentum Trading System - Unified Entry Point

Simple, consolidated system with progress indicators.
All functionality in one place for easy copying to AI chatbots.

Invariant R1: Retries re-run computation but MUST NOT emit side effects.

Usage:
    python main.py weekly      # Weekly scanner
    python main.py pro30       # 30-day screener
    python main.py movers      # Daily movers only
    python main.py all         # Run everything + hybrid analysis
"""

from __future__ import annotations
import sys
import argparse
import logging
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, skip .env loading
    pass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.commands import (
        cmd_weekly, cmd_pro30, cmd_llm, cmd_movers, cmd_all, 
        cmd_performance, cmd_replay, cmd_learn, cmd_learn_status, cmd_learn_export,
    )
    from src.commands.learn import cmd_learn_memory
    from src.core.logging_utils import setup_logging
except ImportError as e:
    print(f"Error: {e}")
    print("Make sure you're running from the project root directory.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Momentum Trading System - Unified CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Global flags
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--log-file", help="Write logs to file")
    
    # ALL - Run everything
    p_all = subparsers.add_parser("all", help="Run all screeners + LLM + hybrid analysis (RECOMMENDED)")
    p_all.add_argument("--date", help="Date (YYYY-MM-DD), defaults to today")
    p_all.add_argument("--config", default="config/default.yaml")
    p_all.add_argument("--no-movers", action="store_true")
    p_all.add_argument("--legacy-pro30", action="store_true", help="Use legacy Pro30 thresholds (20M ADV, min_score off, movers off)")
    p_all.add_argument("--allow-partial-day", action="store_true", help="Allow attention pool on partial-day EOD data (otherwise blocks during market hours)")
    p_all.add_argument("--intraday-attention", action="store_true", help="Force intraday attention pool when market open")
    p_all.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    p_all.add_argument("--model", default="gpt-5.2", help="Model name (default: gpt-5.2, falls back to gpt-4o if not available)")
    p_all.add_argument("--api-key")
    p_all.add_argument("--open", action="store_true", help="Open HTML report in browser after completion")
    p_all.add_argument("--no-debate", action="store_true", help="Disable bull/bear debate analysis")
    p_all.add_argument("--debate-rounds", type=int, default=1, choices=[1, 2, 3], help="Debate rounds per candidate (default: 1)")
    
    # Weekly
    p_weekly = subparsers.add_parser("weekly", help="Run Weekly Scanner only")
    p_weekly.add_argument("--config", default="config/default.yaml")
    p_weekly.add_argument("--no-movers", action="store_true")
    p_weekly.add_argument("--date", help="As-of date (YYYY-MM-DD) for historical replay")
    
    # Pro30
    p_pro30 = subparsers.add_parser("pro30", help="Run 30-Day Screener only")
    p_pro30.add_argument("--config", default="config/default.yaml")
    p_pro30.add_argument("--no-movers", action="store_true")
    p_pro30.add_argument("--date", help="As-of date (YYYY-MM-DD) for historical replay")
    p_pro30.add_argument("--legacy", action="store_true", help="Use legacy Pro30 thresholds (20M ADV, min_score off, movers off)")
    p_pro30.add_argument("--allow-partial-day", action="store_true", help="Allow attention pool on partial-day EOD data (otherwise blocks during market hours)")
    p_pro30.add_argument("--intraday-attention", action="store_true", help="Force intraday attention pool when market open")
    
    # LLM
    p_llm = subparsers.add_parser("llm", help="Run LLM ranking on weekly packets only")
    p_llm.add_argument("--date", help="Date (YYYY-MM-DD)")
    p_llm.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    p_llm.add_argument("--model", default="gpt-5.2", help="Model name (default: gpt-5.2)")
    p_llm.add_argument("--api-key")
    
    # Movers
    p_movers = subparsers.add_parser("movers", help="Daily movers discovery only")
    p_movers.add_argument("--config", default="config/default.yaml")

    # Performance backtest
    p_perf = subparsers.add_parser("performance", help="Backtest picks from outputs/ (Hit +10%% within 7 trading days)")
    p_perf.add_argument("--outputs-root", default="outputs", help="Root outputs directory (default: outputs)")
    p_perf.add_argument("--start", help="Start date YYYY-MM-DD (inclusive)")
    p_perf.add_argument("--end", help="End date YYYY-MM-DD (inclusive)")
    p_perf.add_argument("--out-dir", default="outputs/performance", help="Where to write performance artifacts")
    p_perf.add_argument("--forward-days", type=int, default=7, help="Forward trading days window (default: 7)")
    p_perf.add_argument("--threshold", type=float, default=10.0, help="Hit threshold percent (default: 10.0)")
    p_perf.add_argument("--use-close-only", action="store_true", help="Use Close instead of High for max-forward-price")
    p_perf.add_argument("--include-entry-day", action="store_true", help="Include entry day in the forward window (default excludes)")
    p_perf.add_argument("--auto-adjust", action="store_true", help="yfinance auto_adjust prices")
    p_perf.add_argument("--no-threads", action="store_true", help="Disable threaded yfinance download")

    # Historical replay
    p_replay = subparsers.add_parser("replay", help="Replay past dates to regenerate outputs/YYYY-MM-DD/ (optionally with LLM)")
    p_replay.add_argument("--start", required=True, help="Start date YYYY-MM-DD (inclusive)")
    p_replay.add_argument("--end", required=True, help="End date YYYY-MM-DD (inclusive)")
    p_replay.add_argument("--config", default="config/default.yaml")
    p_replay.add_argument("--no-movers", action="store_true")
    p_replay.add_argument("--llm", action="store_true", help="Also generate weekly LLM Top5 for each day (requires API key)")
    p_replay.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    p_replay.add_argument("--model", default="gpt-5.2")
    p_replay.add_argument("--api-key")
    p_replay.add_argument("--max-days", type=int, default=0, help="Limit number of replayed days (0 = no limit)")
    
    # Model Validation (NEW)
    p_validate = subparsers.add_parser("validate", help="Run multi-period model validation and generate improvement suggestions")
    p_validate.add_argument("--start", help="Start date YYYY-MM-DD (inclusive)")
    p_validate.add_argument("--end", help="End date YYYY-MM-DD (inclusive)")
    p_validate.add_argument("--periods", default="5,7,10,14", help="Comma-separated holding periods to test (default: 5,7,10,14)")
    p_validate.add_argument("--thresholds", default="5,7,10,15", help="Comma-separated hit thresholds (default: 5,7,10,15)")
    p_validate.add_argument("--quick", action="store_true", help="Quick check (last 14 days, periods 5,7 only)")
    p_validate.add_argument("--out-dir", default="outputs/performance", help="Output directory")
    
    # Model Learning (Self-Improving System)
    p_learn = subparsers.add_parser("learn", help="Retrain model weights from outcome data (self-improving system)")
    learn_sub = p_learn.add_subparsers(dest="learn_command")
    
    learn_train = learn_sub.add_parser("train", help="Analyze outcomes and update model weights")
    learn_train.add_argument("--force", action="store_true", help="Train even with limited data")
    learn_train.add_argument("--no-report", action="store_true", help="Skip detailed insights report")
    
    learn_status = learn_sub.add_parser("status", help="Show model status without retraining")
    
    learn_export = learn_sub.add_parser("export", help="Export outcome data to CSV")
    learn_export.add_argument("--output", default="outputs/outcomes_export.csv", help="Output file path")
    
    learn_memory = learn_sub.add_parser("memory", help="Sync outcomes to ChromaDB memory for similarity learning")
    
    # Phase-5 Learning Commands
    learn_resolve = learn_sub.add_parser("resolve", help="Resolve outcomes for Phase-5 learning rows")
    learn_resolve.add_argument("--start", help="Start date (YYYY-MM-DD)")
    learn_resolve.add_argument("--end", help="End date (YYYY-MM-DD)")
    learn_resolve.add_argument("--dry-run", action="store_true", help="Show what would be resolved without writing")
    
    learn_merge = learn_sub.add_parser("merge", help="Merge Phase-5 rows and outcomes into training dataset")
    
    learn_analyze = learn_sub.add_parser("analyze", help="Analyze Phase-5 data and generate scorecard")
    learn_analyze.add_argument("--no-save", action="store_true", help="Don't save scorecard to file")
    
    learn_stats = learn_sub.add_parser("stats", help="Display Phase-5 storage statistics")

    # Phase-6 Learning Commands
    learn_recommend = learn_sub.add_parser("recommend", help="Generate Phase-6 weight-update recommendations")
    learn_recommend.add_argument("--config", help="Path to config YAML (default: config/phase6.yaml)")

    learn_apply = learn_sub.add_parser("apply", help="Apply Phase-6 recommendations to model weights")
    learn_apply.add_argument("--confirm", action="store_true", help="Required safety gate to apply changes")
    learn_apply.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    learn_apply.add_argument("--config", help="Path to config YAML (default: config/phase6.yaml)")
    
    # Calibration Training (PR5 + PR7)
    p_calibrate = subparsers.add_parser("train-calibration", help="Train probability calibration model from historical data")
    p_calibrate.add_argument("--snapshots", nargs="+", required=True, help="Glob patterns for snapshot parquet files")
    p_calibrate.add_argument("--outcomes", nargs="+", required=True, help="Glob patterns for outcome parquet files")
    p_calibrate.add_argument("--model-path", help="Output path for trained model (joblib) - for single model mode")
    p_calibrate.add_argument("--meta-path", help="Output path for model metadata (JSON) - for single model mode")
    p_calibrate.add_argument("--by-regime", action="store_true", help="Train separate models per regime (bull/chop/stress)")
    p_calibrate.add_argument("--out-dir", default="models", help="Output directory for per-regime models (default: models)")
    p_calibrate.add_argument("--min-samples", type=int, default=200, help="Minimum samples required per regime (default: 200)")
    
    # API Server (NEW)
    p_api = subparsers.add_parser("api", help="Start the REST API dashboard server")
    p_api.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    p_api.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    p_api.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    # Confluence Scanner (NEW - Multi-factor alignment)
    p_confluence = subparsers.add_parser("confluence", help="Find stocks with multiple aligned signals (highest conviction)")
    p_confluence.add_argument("--date", help="Date to scan (default: today or latest)")
    p_confluence.add_argument("--min-signals", type=int, default=2, help="Minimum aligned signals (default: 2)")
    p_confluence.add_argument("--no-options", action="store_true", help="Skip options flow analysis")
    p_confluence.add_argument("--no-sector", action="store_true", help="Skip sector rotation analysis")
    
    # Options Flow Scanner (NEW - Smart money tracking)
    p_options = subparsers.add_parser("options", help="Scan for unusual options activity (smart money)")
    p_options.add_argument("--tickers", help="Comma-separated tickers to scan (default: from today's picks)")
    p_options.add_argument("--min-score", type=float, default=5.0, help="Minimum flow score (default: 5.0)")
    p_options.add_argument("--top-n", type=int, default=15, help="Top N candidates to show (default: 15)")
    
    # Sector Rotation Scanner (NEW)
    p_sector = subparsers.add_parser("sector", help="Analyze sector momentum and find sector leaders")
    p_sector.add_argument("--top-sectors", type=int, default=3, help="Number of top sectors to analyze (default: 3)")
    p_sector.add_argument("--stocks-per-sector", type=int, default=5, help="Leaders per sector (default: 5)")
    
    # Position Tracker (NEW)
    p_track = subparsers.add_parser("track", help="Position tracking commands")
    track_sub = p_track.add_subparsers(dest="track_command")
    
    track_entry = track_sub.add_parser("entry", help="Log a new position entry")
    track_entry.add_argument("ticker", help="Ticker symbol")
    track_entry.add_argument("--price", type=float, required=True, help="Entry price")
    track_entry.add_argument("--shares", type=int, required=True, help="Number of shares")
    track_entry.add_argument("--date", help="Entry date (format: YYYY-MM-DD, default: today)")
    track_entry.add_argument("--source", default="manual", choices=["weekly", "pro30", "movers", "manual"])
    track_entry.add_argument("--rank", type=int, help="Predicted rank from scanner")
    track_entry.add_argument("--score", type=float, help="Predicted score from scanner")
    track_entry.add_argument("--reason", default="", help="Entry reason/notes")
    
    track_exit = track_sub.add_parser("exit", help="Log a position exit")
    track_exit.add_argument("position_id", help="Position ID to close")
    track_exit.add_argument("--price", type=float, required=True, help="Exit price")
    track_exit.add_argument("--date", help="Exit date (format: YYYY-MM-DD, default: today)")
    track_exit.add_argument("--reason", default="", help="Exit reason")
    
    track_list = track_sub.add_parser("list", help="List positions")
    track_list.add_argument("--status", choices=["open", "closed", "all"], default="all")
    track_list.add_argument("--ticker", help="Filter by ticker")
    
    track_summary = track_sub.add_parser("summary", help="Show performance summary")
    track_export = track_sub.add_parser("export", help="Export positions to CSV")
    track_export.add_argument("--output", default="data/positions_export.csv", help="Output file path")
    
    track_monitor = track_sub.add_parser("monitor", help="Monitor open positions for drawdown/profit alerts")
    track_monitor.add_argument("--config", default="config/default.yaml", help="Config file")
    track_monitor.add_argument("--alert", action="store_true", help="Send alerts via configured channels")
    
    # Cache Management
    p_cache = subparsers.add_parser("cache", help="Manage price data cache")
    cache_sub = p_cache.add_subparsers(dest="cache_command")
    
    cache_stats = cache_sub.add_parser("stats", help="Show cache statistics")
    
    cache_clear = cache_sub.add_parser("clear", help="Clear all cached data")
    cache_clear.add_argument("--confirm", action="store_true", help="Confirm clearing cache")
    
    cache_prefetch = cache_sub.add_parser("prefetch", help="Pre-populate cache with historical prices for backtesting")
    cache_prefetch.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    cache_prefetch.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    cache_prefetch.add_argument("--universe", help="Universe to prefetch (default: from config)")
    cache_prefetch.add_argument("--config", default="config/default.yaml", help="Config file (default: config/default.yaml)")
    cache_prefetch.add_argument("--batch-size", type=int, default=100, help="Tickers per batch (default: 100)")
    cache_prefetch.add_argument("--provider", choices=["polygon", "yfinance", "auto"], help="Data provider (default: auto, uses config setting)")
    
    cache_cleanup = cache_sub.add_parser("cleanup", help="Remove expired entries from cache")
    
    # Price Database Management (NEW - Permanent Storage)
    p_db = subparsers.add_parser("db", help="Manage permanent price database (no expiration)")
    db_sub = p_db.add_subparsers(dest="db_command")
    
    db_stats = db_sub.add_parser("stats", help="Show price database statistics")
    
    db_download = db_sub.add_parser("download", help="Download historical prices to permanent storage")
    db_download.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    db_download.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    db_download.add_argument("--universe", help="Universe to download (default: from config)")
    db_download.add_argument("--config", default="config/default.yaml", help="Config file")
    db_download.add_argument("--workers", type=int, default=8, help="Concurrent downloads (default: 8)")
    db_download.add_argument("--force", action="store_true", help="Re-download even if data exists")
    
    db_check = db_sub.add_parser("check", help="Check data completeness for tickers")
    db_check.add_argument("--ticker", help="Check specific ticker")
    db_check.add_argument("--start", help="Start date YYYY-MM-DD")
    db_check.add_argument("--end", help="End date YYYY-MM-DD")
    
    db_export = db_sub.add_parser("export", help="Export price data to CSV")
    db_export.add_argument("--ticker", required=True, help="Ticker to export")
    db_export.add_argument("--start", help="Start date")
    db_export.add_argument("--end", help="End date")
    db_export.add_argument("--output", help="Output file path")
    
    args = parser.parse_args()
    
    # Initialize logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    log_file = Path(args.log_file) if args.log_file else None
    setup_logging(level=log_level, log_file=log_file)
    
    if not args.command:
        parser.print_help()
        return 1
    
    # API command handler
    def cmd_api(args):
        try:
            from src.api import run_server
            run_server(host=args.host, port=args.port, reload=args.reload)
            return 0
        except ImportError as e:
            logging.getLogger(__name__).error(f"API server requires FastAPI: {e}")
            logging.getLogger(__name__).info("Install with: pip install fastapi uvicorn")
            return 1
    
    # Track command handler
    def cmd_track(args):
        from datetime import datetime
        try:
            from src.features.tracking import load_tracker
        except ImportError:
            logging.getLogger(__name__).error("Position tracking module not available")
            return 1
        
        tracker = load_tracker()
        
        if args.track_command == "entry":
            entry_date = args.date or datetime.now().strftime("%Y-%m-%d")
            pos = tracker.log_entry(
                ticker=args.ticker,
                entry_date=entry_date,
                entry_price=args.price,
                shares=args.shares,
                source=args.source,
                predicted_rank=args.rank,
                predicted_score=args.score,
                entry_reason=args.reason,
            )
            print(f"✓ Logged entry: {pos.position_id}")
            return 0
        
        elif args.track_command == "exit":
            exit_date = args.date or datetime.now().strftime("%Y-%m-%d")
            pos = tracker.log_exit(
                position_id=args.position_id,
                exit_date=exit_date,
                exit_price=args.price,
                exit_reason=args.reason,
            )
            if pos:
                pnl_str = f"${pos.pnl_dollars:+.2f}" if pos.pnl_dollars else "N/A"
                pct_str = f"{pos.pnl_percent:+.2f}%" if pos.pnl_percent else "N/A"
                print(f"✓ Closed: {pos.ticker} | PnL: {pnl_str} ({pct_str})")
            return 0
        
        elif args.track_command == "list":
            if args.status == "open":
                positions = tracker.get_open_positions()
            elif args.status == "closed":
                positions = tracker.get_closed_positions()
            else:
                positions = list(tracker.positions.values())
            
            if args.ticker:
                positions = [p for p in positions if p.ticker == args.ticker.upper()]
            
            print(f"\n{'='*60}")
            print(f"POSITIONS ({len(positions)})")
            print('='*60)
            for pos in positions:
                status_emoji = "🟢" if pos.status == "open" else "⚪"
                pnl = f"{pos.pnl_percent:+.2f}%" if pos.pnl_percent else ""
                print(f"{status_emoji} {pos.ticker}: {pos.shares}@${pos.entry_price:.2f} [{pos.source}] {pnl}")
            return 0
        
        elif args.track_command == "summary":
            print(tracker.get_summary())
            return 0
        
        elif args.track_command == "export":
            path = tracker.export_to_csv(args.output)
            print(f"✓ Exported to {path}")
            return 0
        
        elif args.track_command == "monitor":
            # Use new position tracker with drawdown monitoring
            try:
                from src.features.positions.tracker import PositionTracker, send_position_alerts
                import yaml
                
                # Load config
                config = {}
                if args.config and Path(args.config).exists():
                    with open(args.config, 'r') as f:
                        config = yaml.safe_load(f)
                
                tracker = PositionTracker()
                print(f"\n{'='*60}")
                print("POSITION MONITOR")
                print('='*60)
                
                summary = tracker.get_summary()
                print(f"\nOpen positions: {summary['total_open']}")
                print(f"Closed positions: {summary['total_closed']}")
                
                if summary['total_open'] > 0:
                    print("\nMonitoring for drawdown/profit alerts...")
                    alerts = tracker.monitor_positions()
                    
                    if alerts:
                        print(f"\n⚠️  {len(alerts)} ALERTS:")
                        for alert in alerts:
                            print(f"  {alert['message']}")
                        
                        if args.alert:
                            send_position_alerts(alerts, config)
                            print("\n✓ Alerts sent via configured channels")
                    else:
                        print("✓ No alerts - all positions within limits")
                else:
                    print("No open positions to monitor")
                
                return 0
            except ImportError as e:
                logging.getLogger(__name__).error(f"Position tracker not available: {e}")
                return 1
        
        else:
            print("Use: track entry|exit|list|summary|export|monitor")
            return 1
    
    # Validate command handler
    def cmd_validate(args):
        from src.commands.validate import run_validation
        periods = [int(p.strip()) for p in args.periods.split(",")]
        thresholds = [float(t.strip()) for t in args.thresholds.split(",")]
        
        if args.quick:
            from datetime import datetime, timedelta
            cutoff = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")
            args.start = cutoff
            periods = [5, 7]
        
        return run_validation(
            start_date=args.start,
            end_date=args.end,
            holding_periods=periods,
            hit_thresholds=thresholds,
            output_dir=args.out_dir,
            quick_mode=args.quick,
        )
    
    # Confluence Scanner command handler
    def cmd_confluence(args):
        from src.pipelines.confluence import run_confluence_scan, format_confluence_report, save_confluence_results
        
        candidates = run_confluence_scan(
            date=args.date,
            min_signals=args.min_signals,
            include_options=not args.no_options,
            include_sector=not args.no_sector,
        )
        
        print(format_confluence_report(candidates))
        
        if candidates:
            path = save_confluence_results(candidates, date=args.date)
            print(f"\n✓ Results saved to {path}")
        
        return 0
    
    # Options Flow Scanner command handler
    def cmd_options(args):
        from src.features.options_flow.scanner import scan_options_flow, format_flow_report
        
        # Get tickers
        if args.tickers:
            tickers = [t.strip().upper() for t in args.tickers.split(",")]
        else:
            # Load from today's picks
            from datetime import datetime
            from pathlib import Path
            import pandas as pd
            
            date = datetime.now().strftime("%Y-%m-%d")
            tickers = set()
            
            for pattern in ["30d_momentum_candidates_*.csv", "weekly_scanner_candidates_*.csv"]:
                for f in Path("outputs").glob(f"**/{pattern}"):
                    try:
                        df = pd.read_csv(f)
                        col = "ticker" if "ticker" in df.columns else "symbol"
                        if col in df.columns:
                            tickers.update(df[col].dropna().tolist())
                    except:
                        pass
            
            tickers = list(tickers)[:50]  # Limit for API efficiency
            
            if not tickers:
                print("No tickers found. Specify --tickers manually.")
                return 1
        
        print(f"Scanning options flow for {len(tickers)} tickers...")
        candidates = scan_options_flow(tickers, min_flow_score=args.min_score, top_n=args.top_n)
        print(format_flow_report(candidates))
        return 0
    
    # Sector Rotation Scanner command handler
    def cmd_sector(args):
        from src.features.sector.rotation import calculate_sector_momentum, find_sector_leaders, format_sector_report
        
        print("Analyzing sector momentum...")
        sectors = calculate_sector_momentum()
        
        print("Finding sector leaders...")
        leaders = find_sector_leaders(
            top_n_sectors=args.top_sectors,
            stocks_per_sector=args.stocks_per_sector,
        )
        
        print(format_sector_report(sectors, leaders))
        return 0
    
    # Cache Management command handler
    def cmd_cache(args):
        from src.core.cache import get_cache
        
        cache = get_cache()
        
        if args.cache_command == "stats":
            stats = cache.get_stats()
            print("\n" + "=" * 50)
            print("PRICE CACHE STATISTICS")
            print("=" * 50)
            print(f"  Total entries:   {stats.get('total_entries', 'N/A')}")
            print(f"  Active entries:  {stats.get('active_entries', 'N/A')}")
            print(f"  Expired entries: {stats.get('expired_entries', 'N/A')}")
            print(f"  Database size:   {stats.get('db_size_mb', 0):.2f} MB")
            print(f"  Database path:   data/price_cache.db")
            print("=" * 50)
            return 0
        
        elif args.cache_command == "clear":
            if not args.confirm:
                print("Add --confirm to clear the cache")
                return 1
            cache._cache.clear()
            print("✓ Cache cleared")
            return 0
        
        elif args.cache_command == "cleanup":
            removed = cache.cleanup()
            print(f"✓ Removed {removed} expired entries")
            return 0
        
        elif args.cache_command == "prefetch":
            import os
            import yaml
            from src.core.universe import build_universe
            
            # Load config
            config = {}
            if args.config and Path(args.config).exists():
                with open(args.config, 'r') as f:
                    config = yaml.safe_load(f) or {}
            
            universe_cfg = config.get("universe", {})
            runtime_cfg = config.get("runtime", {})
            
            # Build universe with config settings (including manual tickers)
            universe_mode = args.universe or universe_cfg.get("mode", "SP500+NASDAQ100+R2000")
            manual_include_file = universe_cfg.get("manual_include_file", "tickers/manual_include_tickers.txt")
            r2000_include_file = universe_cfg.get("r2000_include_file", "tickers/r2000.txt")
            manual_include_mode = universe_cfg.get("manual_include_mode", "ALWAYS")
            cache_file = universe_cfg.get("cache_file", "universe_cache.csv")
            cache_max_age_days = universe_cfg.get("cache_max_age_days", 7)
            
            print(f"Building universe: {universe_mode}")
            print(f"  Manual include file: {manual_include_file}")
            print(f"  R2000 include file:  {r2000_include_file}")
            
            universe = build_universe(
                mode=universe_mode,
                cache_file=cache_file,
                cache_max_age_days=cache_max_age_days,
                manual_include_file=manual_include_file,
                r2000_include_file=r2000_include_file,
                manual_include_mode=manual_include_mode,
            )
            print(f"Found {len(universe)} tickers")
            
            # Determine data provider
            polygon_primary = runtime_cfg.get("polygon_primary", True)
            polygon_api_key = os.environ.get("POLYGON_API_KEY", "")
            
            if args.provider:
                use_polygon = args.provider == "polygon" or (args.provider == "auto" and polygon_primary and polygon_api_key)
            else:
                use_polygon = polygon_primary and polygon_api_key
            
            if use_polygon and not polygon_api_key:
                print("\n⚠️  POLYGON_API_KEY not set. Falling back to Yahoo Finance.")
                use_polygon = False
            
            provider_name = "Polygon.io" if use_polygon else "Yahoo Finance"
            print(f"\nPrefetching historical prices from {args.start} to {args.end}...")
            print(f"Data provider: {provider_name}")
            
            if use_polygon:
                from src.core.polygon import prefetch_polygon_historical
                summary = prefetch_polygon_historical(
                    tickers=universe,
                    start=args.start,
                    end=args.end,
                    api_key=polygon_api_key,
                    batch_size=args.batch_size,
                    max_workers=runtime_cfg.get("polygon_max_workers", 8),
                )
            else:
                from src.core.yf import prefetch_historical_prices
                summary = prefetch_historical_prices(
                    tickers=universe,
                    start=args.start,
                    end=args.end,
                    batch_size=args.batch_size,
                )
            
            print("\n" + "=" * 50)
            print("PREFETCH COMPLETE")
            print("=" * 50)
            print(f"  Data provider: {provider_name}")
            print(f"  Total tickers: {summary['total_tickers']}")
            print(f"  From cache:    {summary['from_cache']}")
            print(f"  Downloaded:    {summary['downloaded']}")
            print(f"  Failed:        {summary['failed']}")
            print("=" * 50)
            print("\n✓ Future backtests will use cached data (no API calls)")
            return 0
        
        else:
            print("Use: cache stats|clear|cleanup|prefetch")
            return 1
    
    # Price Database command handler (NEW)
    def cmd_db(args):
        import os
        import yaml
        from src.core.price_db import get_price_db
        
        db = get_price_db()
        
        if args.db_command == "stats":
            stats = db.get_stats()
            print("\n" + "=" * 50)
            print("PRICE DATABASE STATISTICS")
            print("=" * 50)
            print(f"  Total records:   {stats.get('total_records', 0):,}")
            print(f"  Total tickers:   {stats.get('total_tickers', 0)}")
            print(f"  Date range:      {stats.get('first_date', 'N/A')} to {stats.get('last_date', 'N/A')}")
            print(f"  Database size:   {stats.get('db_size_mb', 0):.2f} MB")
            print(f"  Database path:   {stats.get('db_path', 'N/A')}")
            print("=" * 50)
            print("\n✓ Historical data is stored PERMANENTLY (no expiration)")
            return 0
        
        elif args.db_command == "download":
            from src.core.universe import build_universe
            from src.core.data_fetcher import bulk_download
            
            # Load config
            config = {}
            if args.config and Path(args.config).exists():
                with open(args.config, 'r') as f:
                    config = yaml.safe_load(f) or {}
            
            universe_cfg = config.get("universe", {})
            
            # Build universe
            universe_mode = args.universe or universe_cfg.get("mode", "SP500+NASDAQ100+R2000")
            manual_include_file = universe_cfg.get("manual_include_file", "tickers/manual_include_tickers.txt")
            r2000_include_file = universe_cfg.get("r2000_include_file", "tickers/r2000.txt")
            manual_include_mode = universe_cfg.get("manual_include_mode", "ALWAYS")
            
            print(f"Building universe: {universe_mode}")
            universe = build_universe(
                mode=universe_mode,
                manual_include_file=manual_include_file,
                r2000_include_file=r2000_include_file,
                manual_include_mode=manual_include_mode,
            )
            print(f"Found {len(universe)} tickers")
            
            # Get Polygon API key
            polygon_api_key = os.environ.get("POLYGON_API_KEY", "")
            if not polygon_api_key:
                print("\n⚠️  POLYGON_API_KEY not set. Using Yahoo Finance only.")
            
            print(f"\nDownloading historical prices from {args.start} to {args.end}...")
            print(f"Data source: {'Polygon.io (primary) + Yahoo Finance (fallback)' if polygon_api_key else 'Yahoo Finance'}")
            print(f"Workers: {args.workers}")
            print(f"Skip existing: {not args.force}")
            print()
            
            def progress(completed, total, ticker):
                pct = (completed / total) * 100 if total > 0 else 0
                print(f"  Progress: {completed}/{total} ({pct:.1f}%) - {ticker}    ", end="\r")
            
            summary = bulk_download(
                tickers=universe,
                start_date=args.start,
                end_date=args.end,
                polygon_api_key=polygon_api_key,
                max_workers=args.workers,
                skip_existing=not args.force,
                progress_callback=progress,
            )
            
            print("\n\n" + "=" * 50)
            print("DOWNLOAD COMPLETE")
            print("=" * 50)
            print(f"  Total tickers:   {summary['total_tickers']}")
            print(f"  Already had:     {summary['already_had']}")
            print(f"  Downloaded:      {summary['downloaded']}")
            print(f"  Failed:          {summary['failed']}")
            print(f"  Total records:   {summary['total_records']:,}")
            print("=" * 50)
            print("\n✓ Data stored permanently - will never expire!")
            return 0
        
        elif args.db_command == "check":
            if args.ticker:
                first, last = db.get_ticker_date_range(args.ticker.upper())
                if first:
                    df = db.get_prices(args.ticker.upper(), first, last)
                    print(f"\n{args.ticker.upper()}:")
                    print(f"  Date range: {first} to {last}")
                    print(f"  Records: {len(df)}")
                else:
                    print(f"\nNo data for {args.ticker.upper()}")
            else:
                tickers = db.get_tickers_with_data()
                print(f"\nTickers with data: {len(tickers)}")
                if len(tickers) <= 20:
                    for t in tickers:
                        first, last = db.get_ticker_date_range(t)
                        print(f"  {t}: {first} to {last}")
            return 0
        
        elif args.db_command == "export":
            ticker = args.ticker.upper()
            start = args.start or "2020-01-01"
            end = args.end or datetime.now().strftime("%Y-%m-%d")
            
            df = db.get_prices(ticker, start, end)
            if df.empty:
                print(f"No data for {ticker}")
                return 1
            
            output_path = args.output or f"data/{ticker}_prices.csv"
            df.to_csv(output_path)
            print(f"✓ Exported {len(df)} records to {output_path}")
            return 0
        
        else:
            print("Use: db stats|download|check|export")
            return 1
    
    # Learn command handler (Self-Improving System)
    def cmd_learn_handler(args):
        from src.commands.learn import (
            cmd_learn_resolve, cmd_learn_merge, cmd_learn_analyze, cmd_learn_stats,
            cmd_learn_recommend, cmd_learn_apply,
        )
        
        if args.learn_command == "train" or args.learn_command is None:
            # Set report flag
            args.report = not getattr(args, "no_report", False)
            return cmd_learn(args)
        elif args.learn_command == "status":
            return cmd_learn_status(args)
        elif args.learn_command == "export":
            return cmd_learn_export(args)
        elif args.learn_command == "memory":
            return cmd_learn_memory(args)
        # Phase-5 Learning Commands
        elif args.learn_command == "resolve":
            return cmd_learn_resolve(args)
        elif args.learn_command == "merge":
            return cmd_learn_merge(args)
        elif args.learn_command == "analyze":
            args.save = not getattr(args, "no_save", False)
            return cmd_learn_analyze(args)
        elif args.learn_command == "stats":
            return cmd_learn_stats(args)
        # Phase-6 Learning Commands
        elif args.learn_command == "recommend":
            return cmd_learn_recommend(args)
        elif args.learn_command == "apply":
            return cmd_learn_apply(args)
        else:
            print("Use: learn train|status|export|memory|resolve|merge|analyze|stats|recommend|apply")
            return 1
    
    # Train calibration command handler (PR5)
    def cmd_train_calibration(args):
        from src.commands.train_calibration import cmd_train_calibration as _cmd
        return _cmd(args)
    
    commands = {
        "all": cmd_all,
        "weekly": cmd_weekly,
        "pro30": cmd_pro30,
        "llm": cmd_llm,
        "movers": cmd_movers,
        "performance": cmd_performance,
        "replay": cmd_replay,
        "api": cmd_api,
        "track": cmd_track,
        "validate": cmd_validate,
        "confluence": cmd_confluence,
        "options": cmd_options,
        "sector": cmd_sector,
        "cache": cmd_cache,
        "db": cmd_db,
        "learn": cmd_learn_handler,
        "train-calibration": cmd_train_calibration,
    }
    
    handler = commands.get(args.command)
    if handler:
        try:
            return handler(args)
        except KeyboardInterrupt:
            logging.getLogger(__name__).info("\nInterrupted by user")
            return 130
        except Exception as e:
            logging.getLogger(__name__).error(f"Unexpected error: {e}", exc_info=True)
            return 1
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

